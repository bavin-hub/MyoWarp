"""CPU numerical oracle for tests only; the supported runtime is CUDA env.py."""

from pathlib import Path

import mujoco
import numpy as np

from myowarp.musclemimic.model import build_model
from myowarp.musclemimic.policy import (
    FullbodyObsAdapter,
    LocalPolicyRunner,
    fullbody_history_settings_from_metadata,
    fullbody_obs_adapter_params_from_metadata,
)

from .artifacts import frozen_artifacts
from .controller import MissingLowLevelController, write_motor_command
from .geometry import sole
from .reference import load_human_clip, read_reference
from .shadow import FooledShadow, snapshot
from .state import sensor_frame, tracking_reward


class CPUReferenceOracle:
    num_actions = 2

    def __init__(self, cfg, controller=None, *, artifacts=None, metadata=None):
        cfg.validate()
        self.cfg, self.controller = cfg, controller
        self.rng = np.random.default_rng(cfg.seed)
        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.model.opt.timestep = cfg.physics_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self.human_model, _ = build_model()
        self.clip = load_human_clip(cfg.motion_path, self.human_model)
        self.reference = read_reference(cfg)
        self.shadow = FooledShadow(
            self.human_model, self.model, self.clip, cfg.right_leg_obs
        )
        self.artifacts, self.metadata = artifacts, metadata
        self.human_runner = None
        self.previous_action = np.zeros(2)
        self.steps = 0
        self.eligible = np.flatnonzero(
            self.reference["fit_error_m"][:-2] < cfg.max_reset_sole_error
        )
        if not len(self.eligible):
            raise ValueError(
                "No reference starts satisfy max_reset_sole_error; inspect fit diagnostics"
            )
        self.reset()

    def reset(self, frame=None):
        candidates = (
            [int(frame)]
            if frame is not None
            else self.rng.permutation(self.eligible)[:100]
        )
        for f in candidates:
            if f < 0 or f >= len(self.reference["qpos"]) - 1:
                raise ValueError("Reset frame outside adapted reference")
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.reference["qpos"][f]
            self.data.qvel[:] = self.reference["qvel"][f]
            mujoco.mj_forward(self.model, self.data)
            floor = self.model.geom("floor").id
            gap = min(
                mujoco.mj_geomDistance(
                    self.model, self.data, floor, self.model.geom(n).id, 1.0, None
                )
                for n in ("heel", "toe")
            )
            if gap >= -self.cfg.max_initial_penetration:
                self.frame = f
                break
        else:
            raise ValueError(
                "No nonpenetrating reset found; inspect adapted reference instead of lifting the body"
            )
        self.steps = 0
        self.previous_action[:] = 0
        self.pre_substep = snapshot(self.data)
        if self.human_runner is not None:
            self.human_runner.reset()
        if self.controller is not None:
            self.controller.reset(self.model, self.data)
        current = sensor_frame(self.model, self.data, self.previous_action, self.cfg)
        self.history = np.tile(current, (self.cfg.history_steps, 1))
        return self.observations()

    def observations(self):
        # Privileged critic reads the real physical state and reward targets.
        p, _ = sole(self.model, self.data)
        critic = np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                self.data.act,
                self.reference["qpos"][self.frame],
                self.reference["qvel"][self.frame],
                p - self.reference["foot_pos"][self.frame],
                self.previous_action,
            ]
        ).astype(np.float32)
        return {"actor": self.history.reshape(-1).copy(), "critic": critic}

    def human_action(self):
        if self.human_runner is None:
            checkpoint = Path(self.cfg.checkpoint_path).resolve()
            if self.artifacts is None or self.metadata is None:
                self.artifacts, self.metadata = frozen_artifacts(str(checkpoint))
            self.human_runner = LocalPolicyRunner(
                artifacts=self.artifacts,
                stochastic=False,
                seed=self.cfg.seed,
                obs_adapter=FullbodyObsAdapter(
                    self.human_model,
                    self.clip,
                    goal_params=fullbody_obs_adapter_params_from_metadata(
                        self.metadata
                    ),
                ),
                **fullbody_history_settings_from_metadata(self.metadata),
            )
        if self.metadata.get("normalization_mode") == "fixed":
            # Native numerical oracle for the fixed-normalizer RSL actor.
            from myowarp.musclemimic.policy import _actor_forward

            obs = self.human_runner.obs_adapter.build(
                self.shadow.observe(self.data, self.frame, self.pre_substep), self.frame
            )
            normalized = (obs - self.artifacts.obs_mean) / self.metadata["obs_std"]
            return np.clip(_actor_forward(self.artifacts.params, normalized), -1, 1)
        return self.human_runner.action_for(
            data=self.shadow.observe(self.data, self.frame, self.pre_substep),
            clip=self.clip,
            frame_idx=self.frame,
        )

    def step(self, action):
        if self.controller is None:
            raise MissingLowLevelController(
                "Native oracle steps require an explicit test controller; production uses the CUDA SEA controller"
            )
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected finite two-action prosthesis command")
        if self.frame >= len(self.reference["qpos"]) - 1:
            raise RuntimeError("Reset after reference exhaustion")
        action = np.clip(action, -1, 1)
        torques = np.clip(
            action * self.cfg.torque_scale,
            -np.array(self.cfg.torque_limit),
            self.cfg.torque_limit,
        )
        self.shadow.write_muscles(self.data, self.human_action())
        for substep in range(10):
            command = self.controller.update(
                self.model, self.data, torques.copy(), self.cfg.physics_dt
            )
            write_motor_command(self.model, self.data, command)
            if substep == 9:
                self.pre_substep = snapshot(self.data)
            old_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= old_time or not (
                np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
            ):
                raise FloatingPointError(
                    "Simulator instability: reset/non-finite state"
                )
        self.frame += 1
        self.steps += 1
        mujoco.mj_forward(self.model, self.data)
        reward, terms = tracking_reward(
            self.model,
            self.data,
            self.reference,
            self.frame,
            action,
            self.previous_action,
            self.cfg,
        )
        if not np.isfinite(reward):
            raise FloatingPointError("Non-finite reward")
        pelvis = self.model.body("pelvis").id
        target_height = self.reference["body_pos"][self.frame, 0, 2]
        terminated = (
            abs(self.data.xpos[pelvis, 2] - target_height) > 0.3
            or terms["root_ori"] < np.exp(-(1.2**2) / 0.4**2)
            or terms["unsafe_contact"] > 0
        )
        truncated = (
            self.steps >= round(self.cfg.episode_seconds / self.cfg.policy_dt)
            or self.frame >= len(self.reference["qpos"]) - 1
        )
        self.previous_action[:] = action
        self.history[:-1] = self.history[1:]
        self.history[-1] = sensor_frame(self.model, self.data, action, self.cfg)
        return self.observations(), reward, bool(terminated), bool(truncated), terms
