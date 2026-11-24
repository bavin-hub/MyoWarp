"""CUDA-only prosthesis task: one batched real model and one batched shadow."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from myowarp.musclemimic.model import build_model

from .artifacts import frozen_artifacts
from .controller import MissingLowLevelController
from .gpu_controller import write_gpu_motor_command
from .gpu_shadow import FrozenHumanBatch
from .gpu_sim import WarpBatch, require_cuda
from .gpu_state import RealState
from .prosthesis_observations import hip_actor_history
from .reference import load_human_clip, read_reference
from .state import SENSOR_FRAME_DIM


class GPUProsthesisVecEnv(VecEnv):
    num_actions = 2

    def __init__(self, cfg, controller_factory=None, num_envs=1024):
        cfg.validate()
        self.cfg, self.num_envs = cfg, num_envs
        self.device = require_cuda(cfg.device)
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.max_episode_length = round(cfg.episode_seconds / cfg.policy_dt)
        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.model.opt.timestep = cfg.physics_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        human, _ = build_model()
        human.opt.timestep = cfg.physics_dt
        raw_reference = read_reference(cfg)
        self.reference = {
            key: (
                value
                if value.dtype.kind in "USO"
                else torch.as_tensor(
                    value,
                    device=self.device,
                    dtype=torch.bool if value.dtype.kind == "b" else torch.float32,
                )
            )
            for key, value in raw_reference.items()
            if isinstance(value, np.ndarray)
        }
        clip = load_human_clip(cfg.motion_path, human)
        self.sim = WarpBatch(self.model, num_envs, self.device, cfg.nconmax, cfg.njmax)
        self.data = self.sim.view
        self.state = RealState(self.model, self.device, cfg, self.reference)
        artifacts, metadata = frozen_artifacts(cfg.checkpoint_path)
        self.human = FrozenHumanBatch(
            human, self.model, clip, artifacts, metadata, num_envs, self.device, cfg
        )
        # Joint PPO samples only muscles that can affect the physical plant.
        self.trainable_muscle_ids = self.human.shadow_actuator_ids[
            ~torch.isin(self.human.shadow_actuator_ids, self.human.mask)
        ]
        if cfg.train_muscle:
            self.num_actions = 2 + self.trainable_muscle_ids.numel()
        self._muscle_obs = None
        # Native MuJoCo is used only during initialization to filter reset poses,
        # never for per-environment rollout simulation or observations.
        eligible = np.flatnonzero(
            raw_reference["fit_error_m"][:-2] < cfg.max_reset_sole_error
        )
        scratch = mujoco.MjData(self.model)
        gids = [self.model.geom(n).id for n in ("heel", "toe")]
        good = []
        for f in eligible:
            scratch.qpos[:] = raw_reference["qpos"][f]
            mujoco.mj_kinematics(self.model, scratch)
            gap = np.min(
                scratch.geom_xpos[gids, 2]
                - self.model.geom_size[gids, 0]
                - self.state.floor_z
            )
            if gap >= -cfg.max_initial_penetration:
                good.append(f)
        if not good:
            raise ValueError(
                "No adapted reference resets satisfy fit/ground-penetration bounds"
            )
        self.eligible = torch.tensor(good, dtype=torch.long, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.frame = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = torch.zeros_like(self.frame)
        self.episode_returns = torch.zeros(num_envs, device=self.device)
        self.previous_action = torch.zeros((num_envs, 2), device=self.device)
        self.history = torch.zeros(
            (num_envs, cfg.history_steps, SENSOR_FRAME_DIM), device=self.device
        )
        self.pre_substep = SimpleNamespace(
            **{
                k: getattr(self.data, k).clone()
                for k in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
            }
        )
        self.torque_scale = torch.tensor(cfg.torque_scale, device=self.device)
        self.torque_limit = torch.tensor(cfg.torque_limit, device=self.device)
        self.motor_ids = torch.tensor(
            [
                self.model.actuator(n).id
                for n in ("knee_act", "ankle_act", "knee_vel", "ankle_vel")
            ],
            device=self.device,
        )
        self.motor_limits = torch.tensor(
            self.model.actuator_ctrlrange[self.motor_ids.cpu().numpy()].copy(),
            dtype=torch.float32,
            device=self.device,
        )
        # One controller owns independent (N,...) GPU state.
        self.controller = (
            None
            if controller_factory is None
            else controller_factory(sim=self.sim, num_envs=num_envs, device=self.device)
        )
        self.reset()

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        # Empty index sets are valid; partial resets must not alter other histories.
        self.sim.reset(ids)
        picks = torch.randint(
            len(self.eligible),
            (ids.numel(),),
            device=self.device,
            generator=self.generator,
        )
        self.frame[ids] = self.eligible[picks]
        self.data.qpos[ids] = self.reference["qpos"][self.frame[ids]]
        self.data.qvel[ids] = self.reference["qvel"][self.frame[ids]]
        self.data.time[ids] = 0
        self.sim.forward()
        torch._assert_async(
            (
                self.state.penetration(self.data)[ids]
                >= -self.cfg.max_initial_penetration - 1e-5
            ).all(),
            "Reference reset penetrates floor",
        )
        self.episode_length_buf[ids] = 0
        self.episode_returns[ids] = 0
        self.previous_action[ids] = 0
        for k in vars(self.pre_substep):
            getattr(self.pre_substep, k)[ids] = getattr(self.data, k)[ids]
        self.human.reset(ids)
        self._muscle_obs = None
        if self.controller is not None:
            self.controller.reset(ids)
        load, _ = self.state.contact_loads(self.sim)
        self.history[ids] = self.state.sensors(self.data, self.previous_action, load)[
            ids, None, :
        ]
        self.reset_extra_history(ids)
        return self.get_observations()

    def get_observations(self):
        values = {
            "actor": hip_actor_history(self.history),
            "critic": self.state.critic(self.data, self.frame, self.previous_action),
        }
        if self.cfg.train_muscle:
            if self._muscle_obs is None:
                with torch.no_grad():
                    self._muscle_obs = self.human.normalized_observation(
                        self.data, self.pre_substep, self.frame
                    ).clone()
            values["muscle"] = self._muscle_obs.clone()
        return TensorDict(
            values,
            batch_size=[self.num_envs],
            device=self.device,
        )

    @torch.no_grad()
    def human_action(self):
        return self.human.action(self.data, self.pre_substep, self.frame)

    @torch.no_grad()
    def step(self, actions):
        if self.controller is None:
            raise MissingLowLevelController(
                "No low-level controller was supplied; forward stepping requires a batched GPU controller."
            )
        if (
            actions.shape != (self.num_envs, self.num_actions)
            or actions.device != self.data.qpos.device
        ):
            raise ValueError(
                f"Actions must be (num_envs,{self.num_actions}) tensors on the simulation CUDA device"
            )
        torch._assert_async(
            torch.isfinite(actions).all(), "Nonfinite prosthesis action"
        )
        action = actions[:, :2].detach().clamp(-1, 1)
        desired = torch.maximum(
            torch.minimum(action * self.torque_scale, self.torque_limit),
            -self.torque_limit,
        )
        if self.cfg.train_muscle:
            muscle_action = torch.zeros(
                (self.num_envs, self.human.actor.act_dim), device=self.device
            )
            muscle_action[:, self.trainable_muscle_ids] = actions[:, 2:].clamp(-1, 1)
        else:
            muscle_action = self.human_action()
        self.human.write_muscles(self.data, muscle_action)
        old_time = self.data.time.clone()
        for substep in range(10):
            command = self.controller.update(self.data, desired, self.cfg.physics_dt)
            write_gpu_motor_command(
                self.data, command, self.motor_ids, self.motor_limits, self.num_envs
            )
            if substep == 9:
                for k in vars(self.pre_substep):
                    getattr(self.pre_substep, k).copy_(getattr(self.data, k))
            self.sim.step()
        torch._assert_async(
            torch.isfinite(self.data.qpos).all() & torch.isfinite(self.data.qvel).all(),
            "Simulator instability",
        )
        torch._assert_async(
            (self.data.time > old_time).all(), "Simulator failed to advance time"
        )
        self.frame += 1
        self._muscle_obs = None
        self.episode_length_buf += 1
        self.sim.forward()
        load, unsafe = self.state.contact_loads(self.sim)
        reward, terms = self.state.reward(
            self.data, self.frame, action, self.previous_action, load, unsafe
        )
        torch._assert_async(torch.isfinite(reward).all(), "Nonfinite real-state reward")
        terminated = self.termination(terms, unsafe)
        truncated = (self.episode_length_buf >= self.max_episode_length) | (
            self.frame >= len(self.reference["qpos"]) - 1
        )
        done = terminated | truncated
        self.previous_action.copy_(action)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = self.state.sensors(self.data, action, load)
        self.step_extra_history()
        self.episode_returns += reward
        log = {f"reward/{k}": v.mean() for k, v in terms.items()}
        # Logging and checkpointing may synchronize; simulation/sensing/rewards do not copy to CPU.
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel():
            log["episode/return"] = self.episode_returns[ids].mean()
            log["episode/seconds"] = (
                self.episode_length_buf[ids].float().mean() * self.cfg.policy_dt
            )
            self.reset(ids)
        return (
            self.get_observations(),
            reward,
            done.long(),
            {"time_outs": truncated & ~terminated, "log": log},
        )

    def close(self):
        pass

    def reset_extra_history(self, ids):
        """Optional subclass sensor histories, initialized before reset observations."""

    def step_extra_history(self):
        """Optional subclass sensor histories, sampled once per policy step."""

    def termination(self, terms, unsafe):
        height_error = (
            self.data.xpos[:, self.model.body("pelvis").id, 2]
            - self.reference["body_pos"][self.frame, 0, 2]
        ).abs()
        return (
            (height_error > 0.3)
            | (terms["root_ori"] < np.exp(-(1.2**2) / 0.4**2))
            | (unsafe > 0)
        )
