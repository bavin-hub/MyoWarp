"""Separate intact H1-2 topology, with task-specific tracking reward weights.

Reward terms, termination, randomization and adaptive sampling remain in the original
OSL implementation. No global monkeypatching or mutation of that task occurs.
"""

import mujoco
import torch
from tensordict import TensorDict

from myowarp.tasks.h1_with_osl_tracking.env import (
    REWARD_WEIGHTS as OSL_REWARD_WEIGHTS,
)
from myowarp.tasks.h1_with_osl_tracking.env import (
    TRACKED_BODY_NAMES as OSL_TRACKED_BODY_NAMES,
)
from myowarp.tasks.h1_with_osl_tracking.env import (
    H1WithOslTrackingEnv,
    _ids_by_name,
    _uniform,
)
from myowarp.tasks.h1_with_osl_tracking.reference import MotionReference
from myowarp.tasks.h1_with_osl_velocity.math import (
    matrix_from_quat,
    subtract_frame_transforms,
)
from myowarp.tasks.h1_with_osl_velocity.model import names_for
from myowarp.tasks.h1_with_osl_velocity.sim import StandaloneMujocoWarpSim

from .config import H1TrackingCfg
from .model import build_h1_tracking_model, make_h1_home_qpos

TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_BODY_NAME = "torso_link"
END_EFFECTOR_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
# Shared base methods use only the tracked-body COUNT, never OSL body names.
assert len(TRACKED_BODY_NAMES) == len(OSL_TRACKED_BODY_NAMES) == 14

REWARD_WEIGHTS = {
    **OSL_REWARD_WEIGHTS,
    "motion_global_root_pos": 2.0,
    "motion_global_root_ori": 1.0,
}


class H1TrackingEnv(H1WithOslTrackingEnv):
    """150 actor inputs, 276 critic inputs and 27 joint-position actions."""

    def __init__(self, cfg: H1TrackingCfg) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.num_actions = 27
        self.step_dt = cfg.step_dt
        self.physics_dt = cfg.physics_dt
        self.max_episode_length = cfg.max_episode_length
        self.common_step_counter = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed)

        model = build_h1_tracking_model(
            cfg.model_path,
            physics_dt=cfg.physics_dt,
            solver_iterations=cfg.solver_iterations,
            solver_ls_iterations=cfg.solver_ls_iterations,
            ccd_iterations=cfg.ccd_iterations,
        )
        self.sim = StandaloneMujocoWarpSim(
            model,
            num_envs=cfg.num_envs,
            device=cfg.device,
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
        )
        self.model = model
        self._build_indices()
        self.motion = MotionReference.load(
            cfg.motion_path,
            expected_joint_names=tuple(self.joint_names),
            selected_body_names=TRACKED_BODY_NAMES,
            expected_fps=1.0 / self.step_dt,
            device=self.device,
        )
        self._allocate_buffers()
        self._apply_startup_randomization()
        self._obs = TensorDict({}, batch_size=[self.num_envs], device=self.device)
        self.reset()

    def _build_indices(self) -> None:
        model = self.model
        all_joint_names = names_for(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)[1:]
        actuator_names = names_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
        action_names = [
            name for name in all_joint_names if not name.startswith("socket_")
        ]
        if len(all_joint_names) != 27 or len(action_names) != 27:
            raise RuntimeError("Intact H1-2 tracking requires 27 joints/actions")
        self.joint_names = all_joint_names
        self.action_joint_names = action_names
        self.actuator_joint_names = actuator_names
        joint_ids = _ids_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, all_joint_names)
        action_joint_ids = _ids_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, action_names)
        self.ctrl_from_action_indices = torch.tensor(
            [action_names.index(name) for name in actuator_names],
            device=self.device,
            dtype=torch.long,
        )
        self.joint_qpos_ids = torch.tensor(
            model.jnt_qposadr[joint_ids], device=self.device, dtype=torch.long
        )
        self.joint_qvel_ids = torch.tensor(
            model.jnt_dofadr[joint_ids], device=self.device, dtype=torch.long
        )
        self.action_qpos_ids = torch.tensor(
            model.jnt_qposadr[action_joint_ids], device=self.device, dtype=torch.long
        )
        self.action_joint_indices = torch.tensor(
            [all_joint_names.index(name) for name in action_names],
            device=self.device,
            dtype=torch.long,
        )

        self.root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.anchor_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY_NAME
        )
        self.tracked_body_ids = torch.tensor(
            _ids_by_name(model, mujoco.mjtObj.mjOBJ_BODY, TRACKED_BODY_NAMES),
            device=self.device,
            dtype=torch.long,
        )
        self.anchor_motion_index = TRACKED_BODY_NAMES.index(ANCHOR_BODY_NAME)
        self.end_effector_indices = torch.tensor(
            [TRACKED_BODY_NAMES.index(name) for name in END_EFFECTOR_BODY_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        foot_geom_names = tuple(f"left_foot{i}_collision" for i in range(1, 8)) + tuple(
            f"right_foot{i}_collision" for i in range(1, 8)
        )
        self.foot_geom_ids = torch.tensor(
            _ids_by_name(model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_names),
            device=self.device,
            dtype=torch.long,
        )

        self._sensor_slices: dict[str, slice] = {}
        for name in ("imu_lin_vel", "imu_ang_vel", "self_collision_force"):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            start = int(model.sensor_adr[sensor_id])
            self._sensor_slices[name] = slice(
                start, start + int(model.sensor_dim[sensor_id])
            )

        actuator_scales = (
            0.25 * model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]
        )
        action_scales = actuator_scales[
            [actuator_names.index(name) for name in action_names]
        ]
        self.action_scale = torch.tensor(
            action_scales, device=self.device, dtype=torch.float32
        )
        home_qpos = make_h1_home_qpos(model)
        self.default_joint_pos = torch.tensor(
            home_qpos[self.joint_qpos_ids.cpu().numpy()],
            device=self.device,
            dtype=torch.float32,
        )
        self.default_action_pos = torch.tensor(
            home_qpos[self.action_qpos_ids.cpu().numpy()],
            device=self.device,
            dtype=torch.float32,
        )
        limits = torch.tensor(
            model.jnt_range[joint_ids], device=self.device, dtype=torch.float32
        )
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        width = limits[:, 1] - limits[:, 0]
        self.soft_joint_limits = torch.stack(
            (center - 0.45 * width, center + 0.45 * width), dim=-1
        )

    def _compute_observations(self) -> TensorDict:
        command = torch.cat(
            (self._target_joint_pos(), self._target_joint_vel()), dim=-1
        )
        target_positions = self._target_body_pos()
        target_orientations = self._target_body_quat()
        robot_positions = self._robot_body_pos()
        robot_orientations = self._robot_body_quat()
        robot_anchor_position = robot_positions[:, self.anchor_motion_index]
        robot_anchor_orientation = robot_orientations[:, self.anchor_motion_index]
        anchor_position_b, anchor_orientation_b = subtract_frame_transforms(
            robot_anchor_position,
            robot_anchor_orientation,
            target_positions[:, self.anchor_motion_index],
            target_orientations[:, self.anchor_motion_index],
        )
        anchor_orientation_6d = matrix_from_quat(anchor_orientation_b)[..., :2].reshape(
            self.num_envs, -1
        )
        count = len(TRACKED_BODY_NAMES)
        body_position_b, body_orientation_b = subtract_frame_transforms(
            robot_anchor_position[:, None, :].expand(-1, count, -1),
            robot_anchor_orientation[:, None, :].expand(-1, count, -1),
            robot_positions,
            robot_orientations,
        )
        body_orientation_6d = matrix_from_quat(body_orientation_b)[..., :2].reshape(
            self.num_envs, -1
        )
        body_position_b = body_position_b.reshape(self.num_envs, -1)
        base_linear_velocity = self._sensor("imu_lin_vel")
        base_angular_velocity = self._sensor("imu_ang_vel")
        joint_position = self.qpos[:, self.joint_qpos_ids] - self.default_joint_pos
        joint_velocity = self.qvel[:, self.joint_qvel_ids]

        actor_parts = [
            command,
            anchor_position_b,
            anchor_orientation_6d,
            base_linear_velocity,
            base_angular_velocity,
            joint_position + self.encoder_bias,
            joint_velocity,
            self.actions,
        ]
        if self.cfg.enable_observation_noise:
            noise = self.cfg.observation_noise
            magnitudes = (
                0.0,
                noise.anchor_position,
                noise.anchor_orientation,
                noise.base_linear_velocity,
                noise.base_angular_velocity,
                noise.joint_position,
                noise.joint_velocity,
                0.0,
            )
            actor_parts = [
                value
                if magnitude == 0.0
                else value
                + _uniform(
                    -magnitude,
                    magnitude,
                    value.shape,
                    device=self.device,
                    generator=self._generator,
                )
                for value, magnitude in zip(actor_parts, magnitudes, strict=True)
            ]
        actor = torch.cat(actor_parts, dim=-1)
        critic = torch.cat(
            (
                command,
                anchor_position_b,
                anchor_orientation_6d,
                body_position_b,
                body_orientation_6d,
                base_linear_velocity,
                base_angular_velocity,
                joint_position,
                joint_velocity,
                self.actions,
            ),
            dim=-1,
        )
        if actor.shape[1] != 150 or critic.shape[1] != 276:
            raise RuntimeError(
                f"Observation contract changed: actor={actor.shape}, critic={critic.shape}"
            )
        return TensorDict(
            {"actor": actor, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _allocate_buffers(self):
        super()._allocate_buffers()
        self.actions = torch.zeros((self.num_envs, 27), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.encoder_bias = torch.zeros((self.num_envs, 27), device=self.device)

    def _compute_metrics(self):
        super()._compute_metrics()
        # Capture physical-state metrics before inherited automatic resets.
        self._evaluation_snapshot = super().evaluation_metrics()

    def _compute_rewards(self):
        terms = self._compute_reward_terms()
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, raw_value in terms.items():
            weighted = torch.nan_to_num(
                raw_value * REWARD_WEIGHTS[name] * self.step_dt,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            reward += weighted
            self.episode_reward_sums[name] += weighted
            self._last_reward_terms[name] = raw_value
        return reward

    def evaluation_metrics(self):
        if hasattr(self, "_evaluation_snapshot"):
            return {k: v.clone() for k, v in self._evaluation_snapshot.items()}
        return super().evaluation_metrics()

    def load_training_state(self, state):
        with torch.inference_mode(False):
            if self.bin_failed_count.is_inference():
                self.bin_failed_count = self.bin_failed_count.clone()
        return super().load_training_state(state)
