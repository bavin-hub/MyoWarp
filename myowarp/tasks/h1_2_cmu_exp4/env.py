"""One real GPU model, virtual H1 observations and two independent rewards."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from myowarp.tasks.h1_2_tracking import H1TrackingEnv
from myowarp.tasks.h1_2_tracking.config import config_from_dict
from myowarp.tasks.h1_2_tracking.env import TRACKED_BODY_NAMES
from myowarp.tasks.h1_with_osl_velocity.math import quat_error_magnitude
from myowarp.tasks.h1_with_osl_velocity.sim import StandaloneMujocoWarpSim
from myowarp.tasks.musculoskeletal_prosthesis.geometry import joint_map
from myowarp.tasks.musculoskeletal_prosthesis.prosthesis_observations import (
    hip_actor_history,
)
from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import SEATorqueController

from .model import build_model
from .reference import fingerprint, model_fingerprint
from .state import H1CMUState


class Exp4Env:
    def __init__(self, cfg, evaluation=False):
        cfg.validate(evaluation=evaluation)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_envs = cfg.num_envs
        self.evaluation = evaluation
        self.step_dt = 0.01
        self.num_actions = 2
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        saved = torch.load(cfg.human_checkpoint, map_location="cpu", weights_only=True)
        if saved.get("infos", {}).get("pipeline") != "h1_2_tracking":
            raise ValueError("Expected an intact H1 tracking actor AND critic")
        hc = config_from_dict(saved["infos"]["env_config"])
        hc.num_envs = cfg.num_envs
        hc.device = cfg.device
        hc.motion_path = cfg.motion_path
        hc.enable_observation_noise = False
        hc.domain_randomization.enabled = False
        hc.domain_randomization.enable_pushes = False
        hc.motion.sampling_mode = "start"
        for key in vars(hc.motion):
            if key.startswith(("pose_", "velocity_")) or key == "joint_position_range":
                setattr(hc.motion, key, (0.0, 0.0))
        shadow_class, sim_class, state_class = (
            H1TrackingEnv,
            StandaloneMujocoWarpSim,
            H1CMUState,
        )
        if self.device.type == "cpu":
            from .native import NativeH1Shadow, NativeSim, NativeState

            shadow_class, sim_class, state_class = (
                NativeH1Shadow,
                NativeSim,
                NativeState,
            )
        self.shadow = shadow_class(hc)
        self.model = build_model()
        self.sim = sim_class(
            self.model,
            num_envs=cfg.num_envs,
            device=cfg.device,
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
        )
        if self.device.type == "cuda":
            self.sim.stream = wp.get_stream(cfg.device)
        self.sim.cpu_model = self.model
        self.data = SimpleNamespace(
            **{
                key: self.sim.tensor(key)
                for key in (
                    "qpos",
                    "qvel",
                    "ctrl",
                    "act",
                    "xpos",
                    "xmat",
                    "cvel",
                    "subtree_com",
                    "geom_xpos",
                    "actuator_force",
                    "actuator_velocity",
                )
            }
        )
        with np.load(cfg.reference_path) as ref:
            if (
                str(ref["source_sha256"]) != fingerprint(cfg.motion_path)
                or str(ref["model_sha256"]) != model_fingerprint()
            ):
                raise ValueError("Adapted reference is stale: run prepare")
            self.reference = {
                key: torch.as_tensor(ref[key].copy(), device=self.device)
                for key in (
                    "qpos",
                    "qvel",
                    "foot_pos",
                    "foot_rot",
                    "foot_vel",
                    "contact",
                    "body_pos",
                    "body_rot",
                    "body_vel",
                    "body_ang_vel",
                )
            }
            self.reference["body_names"] = ref["body_names"].tolist()
            if float(ref["fps"].item()) != 100:
                raise ValueError("Adapted reference must be 100 Hz")
        self.nframes = len(self.reference["qpos"])
        self.max_steps = round(cfg.episode_seconds / 0.01)
        if self.max_steps > self.nframes - 1:
            raise ValueError("Episode exceeds reference duration")
        self.state = state_class(self.model, self.device, cfg, self.reference)
        self.controller = SEATorqueController(
            sim=self.sim, num_envs=cfg.num_envs, device=cfg.device
        )
        self.sq, self.pq, self.sv, self.pv = [
            torch.as_tensor(v, device=self.device)
            for v in joint_map(self.shadow.model, self.model, allow_missing=True)
        ]
        self.mapping = []
        for h, c, sign in [
            ("right_knee_joint", "knee", 1),
            ("right_ankle_pitch_joint", "ankle", -1),
        ]:
            hj, cj = self.shadow.model.joint(h), self.model.joint(c)
            self.mapping.append(
                (
                    int(hj.qposadr[0]),
                    int(hj.dofadr[0]),
                    int(cj.qposadr[0]),
                    int(cj.dofadr[0]),
                    sign,
                )
            )
        self.roll = self.shadow.model.joint("right_ankle_roll_joint")
        ignored = (
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        )
        self.intact_actions = torch.tensor(
            [
                i
                for i, n in enumerate(self.shadow.action_joint_names)
                if n not in ignored
            ],
            device=self.device,
        )
        self.intact_actuators = torch.tensor(
            [
                self.model.actuator(self.shadow.action_joint_names[i]).id
                for i in self.intact_actions.tolist()
            ],
            device=self.device,
        )
        self.tracking_mask = torch.tensor(
            [
                i
                for i, n in enumerate(TRACKED_BODY_NAMES)
                if n not in ("right_knee_link", "right_ankle_roll_link")
            ],
            device=self.device,
        )
        self.history = torch.zeros(cfg.num_envs, 5, 29, device=self.device)
        self.previous_action = torch.zeros(cfg.num_envs, 2, device=self.device)
        self.previous_human_action = torch.zeros(cfg.num_envs, 27, device=self.device)
        self.human_action = torch.zeros_like(self.previous_human_action)
        self.frame = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = torch.zeros_like(self.frame)
        self.episode_returns = {
            name: torch.zeros(cfg.num_envs, device=self.device)
            for name in ("h1", "prosthesis")
        }
        self.scale = torch.tensor(cfg.torque_scale, device=self.device)
        self.torque_limit = torch.tensor(cfg.torque_limit, device=self.device)
        self.ctrl_limits = torch.tensor(
            self.model.actuator_ctrlrange[self.state.motor_ids.cpu().numpy()],
            device=self.device,
            dtype=torch.float32,
        )
        self.half = 0
        self.failure_weights = torch.ones(
            max(1, (self.nframes - 1) // 100), device=self.device
        )
        self.reset()

    def enable_fast_evaluation(self):
        """Capture the unchanged 10-tick LLC/physics sequence; eval only.

        Capture advances temporary simulation/controller state. The caller must
        reset to the requested reference frame afterward.
        """
        if not self.evaluation:
            raise ValueError("Fast evaluation cannot be enabled during training")
        self._desired_torque = torch.zeros_like(self.previous_action)
        # Fuse the many tiny Torch operations within a controller tick before
        # capturing the ten-tick controller/physics sequence.
        self._controller_update = torch.compile(
            self.controller.update,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        if self.device.type == "cpu":
            # Compile once before timing; restore physical/filter state via reset.
            with torch.inference_mode():
                self._integrate(self._desired_torque)
            self.fast_evaluation = True
            return
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        warp_stream = wp.stream_from_torch(stream)
        with (
            torch.inference_mode(),
            torch.cuda.stream(stream),
            wp.ScopedStream(warp_stream),
        ):
            # Warm up Torch allocations and the already compiled Warp graphs.
            for _ in range(3):
                self._integrate(self._desired_torque)
            torch.cuda.synchronize(self.device)
            self._integration_graph = torch.cuda.CUDAGraph()
            # Capture raw Warp kernels, rather than launching its existing
            # standalone CUDA graph inside another capture.
            with (
                torch.cuda.graph(self._integration_graph, stream=stream),
                wp.ScopedCapture(stream=warp_stream, external=True) as capture,
            ):
                self._integrate(self._desired_torque, capture=True)
            self._warp_integration_capture = capture.graph
        current.wait_stream(stream)
        self.fast_evaluation = True

    def _integrate(self, desired, capture=False):
        for _ in range(10):
            update = getattr(self, "_controller_update", self.controller.update)
            command = update(self.data, desired, 0.001)
            channels = torch.cat(
                (command.feedforward_torque, command.target_velocity), 1
            )
            self.data.ctrl[:, self.state.motor_ids] = channels.clamp(
                self.ctrl_limits[:, 0], self.ctrl_limits[:, 1]
            )
            if capture:
                import mujoco_warp

                mujoco_warp.step(self.sim.model, self.sim.data)
            else:
                self.sim.step()

    def _sync_shadow(self):
        s, d = self.shadow, self.data
        s.qpos[:, self.sq] = d.qpos[:, self.pq]
        s.qvel[:, self.sv] = d.qvel[:, self.pv]
        for hq, hv, cq, cv, sign in self.mapping:
            s.qpos[:, hq] = sign * d.qpos[:, cq]
            s.qvel[:, hv] = sign * d.qvel[:, cv]
        s.qpos[:, self.roll.qposadr[0]] = 0
        s.qvel[:, self.roll.dofadr[0]] = 0
        s.actions.copy_(self.human_action)
        s.time_steps.copy_((self.frame // 2).clamp(max=s.motion.num_frames - 1))
        s.sim.forward()
        s._update_relative_targets()

    def _observations(self):
        human = self.shadow._compute_observations()
        return TensorDict(
            {
                "h1_actor": human["actor"],
                "h1_critic": human["critic"],
                "actor": hip_actor_history(self.history),
                "critic": self.state.critic(
                    self.data, self.frame, self.previous_action
                ),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self):
        return self.obs

    def _reset_ids(self, ids, start_frame=None):
        if not len(ids):
            return
        self.sim.reset(ids)
        self.controller.reset(ids)
        if start_frame is not None:
            starts = torch.full(
                (len(ids),), int(start_frame) * 2, device=self.device, dtype=torch.long
            )
        elif self.evaluation:
            starts = torch.zeros(len(ids), device=self.device, dtype=torch.long)
        else:
            # Failure-biased sampling with 10% uniform probability, reserving
            # enough frames for an uninterrupted full training episode.
            max_start = (self.nframes - 1 - self.max_steps) // 2
            starts50 = torch.arange(max_start + 1, device=self.device)
            bins = (starts50 // 50).clamp(max=len(self.failure_weights) - 1)
            weights = self.failure_weights[bins]
            probability = 0.9 * weights / weights.sum() + 0.1 / len(weights)
            starts = 2 * torch.multinomial(
                probability, len(ids), replacement=True, generator=self.generator
            )
        if (starts < 0).any() or (starts + self.max_steps >= self.nframes).any():
            raise ValueError("Requested start leaves insufficient reference duration")
        self.frame[ids] = starts
        self.data.qpos[ids] = self.reference["qpos"][starts].to(self.data.qpos.dtype)
        self.data.qvel[ids] = self.reference["qvel"][starts].to(self.data.qvel.dtype)
        self.data.ctrl[ids] = 0
        self.episode_length_buf[ids] = 0
        self.previous_action[ids] = 0
        self.human_action[ids] = 0
        self.previous_human_action[ids] = 0
        for values in self.episode_returns.values():
            values[ids] = 0
        self.sim.forward()
        # Raise penetrating starts only; do not move a swing foot down to ground.
        left = [self.model.geom(f"left_foot{i}_collision").id for i in range(1, 8)]
        q = self.data.qpos
        g = self.data.geom_xpos[ids][:, left, 2]
        radii = torch.tensor(self.model.geom_size[left, 0], device=self.device)
        bottom = (g - radii).amin(-1)
        bottom = torch.minimum(bottom, self.state.penetration(self.data)[ids])
        q[ids, 2] += (-bottom).clamp_min(0).float()
        self.sim.forward()
        self._sync_shadow()
        self.loads, self.unsafe = self.state.contact_loads(self.sim)
        sensors = self.state.sensors(self.data, self.previous_action, self.loads)
        self.history[ids] = sensors[ids, None]

    def reset(self, start_frame=None):
        self._reset_ids(torch.arange(self.num_envs, device=self.device), start_frame)
        self.half = 0
        self.obs = self._observations()
        return self.obs

    def _human_reward(self):
        s = self.shadow
        mask = self.tracking_mask
        terms = s._compute_reward_terms()
        actual = s._robot_body_pos()
        rot = s._robot_body_quat()
        lv, av = s._body_velocity(s.tracked_body_ids)
        terms["motion_body_pos"] = torch.exp(
            -(
                (s.body_pos_relative_w[:, mask] - actual[:, mask])
                .square()
                .sum(-1)
                .mean(-1)
            )
            / 0.3**2
        )
        terms["motion_body_ori"] = torch.exp(
            -quat_error_magnitude(s.body_quat_relative_w[:, mask], rot[:, mask])
            .square()
            .mean(-1)
            / 0.4**2
        )
        terms["motion_body_lin_vel"] = torch.exp(
            -(s._target_body_lin_vel()[:, mask] - lv[:, mask]).square().sum(-1).mean(-1)
        )
        terms["motion_body_ang_vel"] = torch.exp(
            -(s._target_body_ang_vel()[:, mask] - av[:, mask]).square().sum(-1).mean(-1)
            / 3.14**2
        )
        terms["action_rate_l2"] = (
            (
                self.human_action[:, self.intact_actions]
                - self.previous_human_action[:, self.intact_actions]
            )
            .square()
            .sum(-1)
        )
        q = s.qpos[:, s.action_qpos_ids][:, self.intact_actions]
        limits = s.soft_joint_limits[self.intact_actions]
        terms["joint_limit"] = (
            (limits[:, 0] - q).clamp_min(0) + (q - limits[:, 1]).clamp_min(0)
        ).sum(-1)
        terms["self_collisions"] = self.unsafe
        return sum(
            self.cfg.human_weights[k] * v for k, v in terms.items()
        ) * 0.01, terms

    @torch.no_grad()
    def step(self, human_action, prosthesis_action):
        if self.half == 0:
            self.human_action.copy_(human_action)
        s = self.shadow
        targets = s.default_action_pos + s.action_scale * self.human_action
        self.data.ctrl[:, self.intact_actuators] = targets[:, self.intact_actions].to(
            self.data.ctrl.dtype
        )
        desired = (prosthesis_action * self.scale).clamp(
            -self.torque_limit, self.torque_limit
        )
        if getattr(self, "fast_evaluation", False) and self.device.type == "cuda":
            self._desired_torque.copy_(desired)
            self._integration_graph.replay()
        else:
            self._integrate(desired)
        self.frame += 1
        self.episode_length_buf += 1
        self.sim.forward()
        self._sync_shadow()
        self.loads, self.unsafe = self.state.contact_loads(self.sim)
        if getattr(self, "fast_evaluation", False) or (
            self.evaluation and self.device.type == "cpu"
        ):
            # Neither critic nor rewards drive inference or termination.
            prost_reward = torch.zeros(self.num_envs, device=self.device)
            human_reward = torch.zeros_like(prost_reward)
            pterms, hterms = {}, {}
        else:
            prost_reward, pterms = self.state.reward(
                self.data,
                self.frame,
                prosthesis_action,
                self.previous_action,
                self.loads,
                self.unsafe,
            )
            human_reward, hterms = self._human_reward()
        self.previous_action.copy_(prosthesis_action)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = self.state.sensors(
            self.data, self.previous_action, self.loads
        )
        self.half = 1 - self.half
        # End episodes only on 50Hz boundaries so both PPO transitions agree.
        timeout = self.episode_length_buf >= self.max_steps
        root = self.data.qpos[:, 2]
        rotation = self.data.xmat[:, self.model.body("pelvis").id].reshape(-1, 3, 3)
        failed = (
            (root < 0.55)
            | (rotation[:, 2, 2] < 0.5)
            | (~torch.isfinite(self.data.qpos).all(-1))
            | (self.unsafe > 0)
        )
        failed = failed if self.half == 0 else torch.zeros_like(failed)
        timeout = timeout if self.half == 0 else torch.zeros_like(timeout)
        done = failed | timeout
        self.metrics = {
            "root_xy_error": torch.linalg.vector_norm(
                self.data.qpos[:, :2] - s._target_body_pos()[:, 0, :2], dim=-1
            ),
            "body_relative_error": torch.linalg.vector_norm(
                s.body_pos_relative_w[:, self.tracking_mask]
                - s._robot_body_pos()[:, self.tracking_mask],
                dim=-1,
            ).mean(-1),
            "prosthesis_foot_error": torch.linalg.vector_norm(
                self.state.sole(self.data)[0] - self.reference["foot_pos"][self.frame],
                dim=-1,
            ),
            "prosthesis_joint_error": (
                self.data.qpos[:, self.state.qids]
                - self.reference["qpos"][self.frame][:, self.state.qids]
            )
            .abs()
            .mean(-1),
        }
        for name, value in [("h1", human_reward), ("prosthesis", prost_reward)]:
            self.episode_returns[name] += value
        logs = {
            name: {f"reward/{k}": v.mean() for k, v in terms.items()}
            for name, terms in [("h1", hterms), ("prosthesis", pterms)]
        }
        ids = done.nonzero().flatten()
        for name, log in logs.items():
            if len(ids):
                log["episode/return"] = self.episode_returns[name][ids].mean()
            log.update({f"metrics/{k}": v.mean() for k, v in self.metrics.items()})
        if self.half == 0:
            self.previous_human_action.copy_(self.human_action)
        if len(ids):
            bins = (self.frame[ids] // 100).clamp(max=len(self.failure_weights) - 1)
            self.failure_weights.mul_(0.999)
            self.failure_weights.scatter_add_(0, bins, failed[ids].float())
            self.failure_weights.clamp_min_(0.01)
            self._reset_ids(ids)
        self.obs = self._observations()
        return (
            self.obs,
            {"h1": human_reward.float(), "prosthesis": prost_reward.float()},
            done,
            {"time_outs": timeout, "logs": logs},
        )

    def close(self):
        self.shadow.close()
