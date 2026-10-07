"""Standalone MuJoCo-Warp environment for H1-with-OSL motion tracking."""

from __future__ import annotations

import math
from collections.abc import Iterable

import mujoco
import torch
import torch.nn.functional as functional
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from myowarp.tasks.h1_with_osl_velocity.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    subtract_frame_transforms,
    yaw_component,
)
from myowarp.tasks.h1_with_osl_velocity.model import (
    build_h1_with_osl_tracking_model,
    make_h1_with_osl_home_qpos,
    names_for,
)
from myowarp.tasks.h1_with_osl_velocity.sim import StandaloneMujocoWarpSim

from .config import H1WithOslTrackingCfg
from .reference import MotionReference


TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "osl_knee_assembly",
    "osl_foot_assembly",
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
    "osl_foot_assembly",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

REWARD_WEIGHTS = {
    "motion_global_root_pos": 0.5,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
    "action_rate_l2": -0.1,
    "joint_limit": -10.0,
    "self_collisions": -10.0,
}


def _ids_by_name(
    model: mujoco.MjModel,
    objtype: mujoco.mjtObj,
    names: Iterable[str],
) -> list[int]:
    ids = []
    for name in names:
        object_id = mujoco.mj_name2id(model, objtype, name)
        if object_id < 0:
            raise ValueError(f"{objtype.name} named '{name}' was not found")
        ids.append(object_id)
    return ids


def _uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator) * (high - low) + low


class H1WithOslTrackingEnv(VecEnv):
    """Batched reference-motion task with no MJLab runtime dependency."""

    cfg: H1WithOslTrackingCfg

    def __init__(self, cfg: H1WithOslTrackingCfg) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.num_actions = 26
        self.step_dt = cfg.step_dt
        self.physics_dt = cfg.physics_dt
        self.max_episode_length = cfg.max_episode_length
        self.common_step_counter = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed)

        model = build_h1_with_osl_tracking_model(
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
        if len(all_joint_names) != 30 or len(action_names) != 26:
            raise RuntimeError("Motion tracking requires 30 joints and 26 actions")
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
            f"right_foot{i}_collision" for i in range(1, 4)
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
            self._sensor_slices[name] = slice(start, start + int(model.sensor_dim[sensor_id]))

        actuator_scales = (
            0.25 * model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]
        )
        action_scales = actuator_scales[
            [actuator_names.index(name) for name in action_names]
        ]
        self.action_scale = torch.tensor(action_scales, device=self.device, dtype=torch.float32)
        home_qpos = make_h1_with_osl_home_qpos(model)
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
        limits = torch.tensor(model.jnt_range[joint_ids], device=self.device, dtype=torch.float32)
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        width = limits[:, 1] - limits[:, 0]
        self.soft_joint_limits = torch.stack(
            (center - 0.45 * width, center + 0.45 * width), dim=-1
        )

    def _allocate_buffers(self) -> None:
        n, d = self.num_envs, self.device
        self.episode_length_buf = torch.zeros(n, device=d, dtype=torch.long)
        self.actions = torch.zeros((n, 26), device=d)
        self.prev_actions = torch.zeros_like(self.actions)
        self.encoder_bias = torch.zeros((n, 30), device=d)
        self.time_steps = torch.zeros(n, device=d, dtype=torch.long)
        self.body_pos_relative_w = torch.zeros((n, len(TRACKED_BODY_NAMES), 3), device=d)
        self.body_quat_relative_w = torch.zeros((n, len(TRACKED_BODY_NAMES), 4), device=d)
        self.body_quat_relative_w[..., 0] = 1.0
        self.self_collision_hits = torch.zeros(n, device=d)
        self.push_time_left = torch.zeros(n, device=d)
        self.bin_count = int(self.motion.num_frames // (1.0 / self.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, device=d)
        self._current_bin_failed = torch.zeros(self.bin_count, device=d)
        kernel = torch.tensor(
            [self.cfg.motion.adaptive_lambda**i for i in range(self.cfg.motion.adaptive_kernel_size)],
            device=d,
        )
        self.adaptive_kernel = kernel / kernel.sum()
        self.episode_reward_sums = {
            name: torch.zeros(n, device=d) for name in REWARD_WEIGHTS
        }
        self.metric_step_count = torch.zeros(n, device=d, dtype=torch.long)
        self.metric_sums = {
            name: torch.zeros(n, device=d)
            for name in (
                "error_anchor_pos",
                "error_anchor_rot",
                "error_anchor_lin_vel",
                "error_anchor_ang_vel",
                "error_body_pos",
                "error_body_rot",
                "error_body_lin_vel",
                "error_body_ang_vel",
                "error_joint_pos",
                "error_joint_vel",
            )
        }
        self._last_reward_terms = {
            name: torch.zeros(n, device=d) for name in REWARD_WEIGHTS
        }

    def _apply_startup_randomization(self) -> None:
        randomization = self.cfg.domain_randomization
        if not randomization.enabled:
            return
        self.sim.expand_model_fields(
            (
                "geom_friction",
                "body_ipos",
                "body_subtreemass",
                "dof_invweight0",
                "body_invweight0",
                "tendon_length0",
                "tendon_invweight0",
            )
        )
        friction = self.sim.model_tensor("geom_friction")
        samples = _uniform(
            *randomization.foot_friction_range,
            (self.num_envs, 1),
            device=self.device,
            generator=self._generator,
        )
        friction[:, self.foot_geom_ids, 0] = samples
        body_ipos = self.sim.model_tensor("body_ipos")
        default_ipos = torch.tensor(
            self.model.body_ipos[self.anchor_body_id],
            device=self.device,
            dtype=torch.float32,
        )
        offsets = torch.stack(
            [
                _uniform(*value_range, (self.num_envs,), device=self.device, generator=self._generator)
                for value_range in (
                    randomization.torso_com_offset_x,
                    randomization.torso_com_offset_y,
                    randomization.torso_com_offset_z,
                )
            ],
            dim=-1,
        )
        body_ipos[:, self.anchor_body_id] = default_ipos + offsets
        self.sim.recompute_constants()
        self.encoder_bias.uniform_(
            randomization.encoder_bias_range[0],
            randomization.encoder_bias_range[1],
            generator=self._generator,
        )

    @property
    def qpos(self) -> torch.Tensor:
        return self.sim.tensor("qpos")

    @property
    def qvel(self) -> torch.Tensor:
        return self.sim.tensor("qvel")

    @property
    def sensordata(self) -> torch.Tensor:
        return self.sim.tensor("sensordata")

    def _sensor(self, name: str) -> torch.Tensor:
        return self.sensordata[:, self._sensor_slices[name]]

    def _root_quat(self) -> torch.Tensor:
        return self.sim.tensor("xquat")[:, self.root_body_id]

    def _body_velocity(self, body_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.sim.tensor("xpos")[:, body_ids]
        cvel = self.sim.tensor("cvel")[:, body_ids]
        subtree_com = self.sim.tensor("subtree_com")[:, self.root_body_id].unsqueeze(1)
        angular = cvel[..., :3]
        linear = cvel[..., 3:6] - torch.cross(angular, subtree_com - positions, dim=-1)
        return linear, angular

    def _root_link_velocity_w(self) -> torch.Tensor:
        body_ids = torch.tensor([self.root_body_id], device=self.device)
        linear, angular = self._body_velocity(body_ids)
        return torch.cat((linear[:, 0], angular[:, 0]), dim=-1)

    def _target_joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    def _target_joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    def _target_body_pos(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps]

    def _target_body_quat(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    def _target_body_lin_vel(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    def _target_body_ang_vel(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    def _robot_body_pos(self) -> torch.Tensor:
        return self.sim.tensor("xpos")[:, self.tracked_body_ids]

    def _robot_body_quat(self) -> torch.Tensor:
        return self.sim.tensor("xquat")[:, self.tracked_body_ids]

    def _update_relative_targets(self) -> None:
        target_position = self._target_body_pos()
        target_orientation = self._target_body_quat()
        target_anchor_position = target_position[:, self.anchor_motion_index]
        target_anchor_orientation = target_orientation[:, self.anchor_motion_index]
        robot_anchor_position = self.sim.tensor("xpos")[:, self.anchor_body_id]
        robot_anchor_orientation = self.sim.tensor("xquat")[:, self.anchor_body_id]
        count = len(TRACKED_BODY_NAMES)
        delta_position = robot_anchor_position[:, None, :].repeat(1, count, 1)
        delta_position[..., 2] = target_anchor_position[:, None, 2]
        delta_orientation = yaw_component(
            quat_mul(robot_anchor_orientation, quat_inverse(target_anchor_orientation))
        )
        delta_orientation = delta_orientation[:, None, :].expand(-1, count, -1)
        self.body_quat_relative_w = quat_mul(delta_orientation, target_orientation)
        self.body_pos_relative_w = delta_position + quat_apply(
            delta_orientation,
            target_position - target_anchor_position[:, None, :],
        )

    def _record_adaptive_failures(
        self, env_ids: torch.Tensor, terminated: torch.Tensor | None
    ) -> None:
        if terminated is None or env_ids.numel() == 0:
            return
        failed = terminated[env_ids]
        if not torch.any(failed):
            return
        bin_indices = torch.clamp(
            (self.time_steps[env_ids] * self.bin_count) // max(self.motion.num_frames, 1),
            0,
            self.bin_count - 1,
        )
        failed_bins = bin_indices[failed]
        self._current_bin_failed += torch.bincount(
            failed_bins, minlength=self.bin_count
        ).float()

    def _sampling_probabilities(self) -> torch.Tensor:
        probabilities = (
            self.bin_failed_count
            + self.cfg.motion.adaptive_uniform_ratio / float(self.bin_count)
        )
        if self.cfg.motion.adaptive_kernel_size > 1:
            probabilities = functional.pad(
                probabilities.view(1, 1, -1),
                (0, self.cfg.motion.adaptive_kernel_size - 1),
                mode="replicate",
            )
            probabilities = functional.conv1d(
                probabilities, self.adaptive_kernel.view(1, 1, -1)
            ).view(-1)
        return probabilities / probabilities.sum()

    def _sample_frames(self, env_ids: torch.Tensor) -> None:
        mode = self.cfg.motion.sampling_mode
        if mode == "start":
            self.time_steps[env_ids] = 0
        elif mode == "uniform":
            self.time_steps[env_ids] = torch.randint(
                0,
                self.motion.num_frames,
                (len(env_ids),),
                device=self.device,
                generator=self._generator,
            )
        elif mode == "adaptive":
            probabilities = self._sampling_probabilities()
            sampled_bins = torch.multinomial(
                probabilities,
                len(env_ids),
                replacement=True,
                generator=self._generator,
            )
            within_bin = torch.rand(
                len(env_ids), device=self.device, generator=self._generator
            )
            self.time_steps[env_ids] = (
                (sampled_bins + within_bin)
                / self.bin_count
                * (self.motion.num_frames - 1)
            ).long()
        else:
            raise ValueError(f"Unsupported motion sampling mode: {mode}")

    def _write_sampled_motion_state(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        motion_cfg = self.cfg.motion
        root_position = self._target_body_pos()[env_ids, 0].clone()
        root_orientation = self._target_body_quat()[env_ids, 0].clone()
        pose_ranges = (
            motion_cfg.pose_x,
            motion_cfg.pose_y,
            motion_cfg.pose_z,
            motion_cfg.pose_roll,
            motion_cfg.pose_pitch,
            motion_cfg.pose_yaw,
        )
        pose_noise = torch.stack(
            [
                _uniform(*value_range, (count,), device=self.device, generator=self._generator)
                for value_range in pose_ranges
            ],
            dim=-1,
        )
        root_position += pose_noise[:, :3]
        root_orientation = quat_mul(
            quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
            root_orientation,
        )

        root_linear_velocity = self._target_body_lin_vel()[env_ids, 0].clone()
        root_angular_velocity = self._target_body_ang_vel()[env_ids, 0].clone()
        velocity_ranges = (
            motion_cfg.velocity_x,
            motion_cfg.velocity_y,
            motion_cfg.velocity_z,
            motion_cfg.velocity_roll,
            motion_cfg.velocity_pitch,
            motion_cfg.velocity_yaw,
        )
        velocity_noise = torch.stack(
            [
                _uniform(*value_range, (count,), device=self.device, generator=self._generator)
                for value_range in velocity_ranges
            ],
            dim=-1,
        )
        root_linear_velocity += velocity_noise[:, :3]
        root_angular_velocity += velocity_noise[:, 3:]

        joint_position = self._target_joint_pos()[env_ids].clone()
        joint_position += _uniform(
            *motion_cfg.joint_position_range,
            joint_position.shape,
            device=self.device,
            generator=self._generator,
        )
        joint_position = torch.clamp(
            joint_position,
            self.soft_joint_limits[:, 0],
            self.soft_joint_limits[:, 1],
        )
        self.qpos[env_ids, :3] = root_position
        self.qpos[env_ids, 3:7] = root_orientation
        self.qpos[env_ids[:, None], self.joint_qpos_ids] = joint_position
        self.qvel[env_ids, :3] = root_linear_velocity
        self.qvel[env_ids, 3:6] = quat_apply_inverse(
            root_orientation, root_angular_velocity
        )
        self.qvel[env_ids[:, None], self.joint_qvel_ids] = self._target_joint_vel()[env_ids]

    def _resample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated: torch.Tensor | None = None,
    ) -> None:
        if self.cfg.motion.sampling_mode == "adaptive":
            self._record_adaptive_failures(env_ids, terminated)
        self._sample_frames(env_ids)
        self._write_sampled_motion_state(env_ids)

    def _advance_motion(self) -> bool:
        self.time_steps += 1
        ended = (self.time_steps >= self.motion.num_frames).nonzero(as_tuple=False).flatten()
        if ended.numel() > 0:
            self._resample_motion(ended)
        if self.cfg.motion.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.motion.adaptive_alpha * self._current_bin_failed
                + (1.0 - self.cfg.motion.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()
        return ended.numel() > 0

    def _apply_pushes(self) -> bool:
        randomization = self.cfg.domain_randomization
        if not randomization.enabled or not randomization.enable_pushes:
            return False
        self.push_time_left -= self.step_dt
        env_ids = (self.push_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return False
        velocity = self._root_link_velocity_w()[env_ids].clone()
        ranges = (
            randomization.push_lin_vel_x,
            randomization.push_lin_vel_y,
            randomization.push_lin_vel_z,
            randomization.push_ang_vel_x,
            randomization.push_ang_vel_y,
            randomization.push_ang_vel_z,
        )
        for axis, value_range in enumerate(ranges):
            velocity[:, axis] += _uniform(
                *value_range,
                (len(env_ids),),
                device=self.device,
                generator=self._generator,
            )
        self.qvel[env_ids, :3] = velocity[:, :3]
        self.qvel[env_ids, 3:6] = quat_apply_inverse(
            self._root_quat()[env_ids], velocity[:, 3:]
        )
        self.push_time_left[env_ids] = _uniform(
            *randomization.push_interval_s,
            (len(env_ids),),
            device=self.device,
            generator=self._generator,
        )
        return True

    def _compute_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        target_positions = self._target_body_pos()
        target_orientations = self._target_body_quat()
        robot_positions = self._robot_body_pos()
        robot_orientations = self._robot_body_quat()
        anchor_z_error = torch.abs(
            target_positions[:, self.anchor_motion_index, 2]
            - robot_positions[:, self.anchor_motion_index, 2]
        )
        gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(
            self.num_envs, 3
        )
        target_gravity = quat_apply_inverse(
            target_orientations[:, self.anchor_motion_index], gravity
        )
        robot_gravity = quat_apply_inverse(
            robot_orientations[:, self.anchor_motion_index], gravity
        )
        orientation_error = torch.abs(target_gravity[:, 2] - robot_gravity[:, 2])
        end_effector_z_error = torch.abs(
            self.body_pos_relative_w[:, self.end_effector_indices, 2]
            - robot_positions[:, self.end_effector_indices, 2]
        )
        terminated = (
            (anchor_z_error > 0.25)
            | (orientation_error > 0.8)
            | torch.any(end_effector_z_error > 0.25, dim=1)
        )
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _compute_reward_terms(self) -> dict[str, torch.Tensor]:
        target_positions = self._target_body_pos()
        target_orientations = self._target_body_quat()
        robot_positions = self._robot_body_pos()
        robot_orientations = self._robot_body_quat()
        target_anchor_position = target_positions[:, self.anchor_motion_index]
        target_anchor_orientation = target_orientations[:, self.anchor_motion_index]
        robot_anchor_position = robot_positions[:, self.anchor_motion_index]
        robot_anchor_orientation = robot_orientations[:, self.anchor_motion_index]
        robot_linear_velocity, robot_angular_velocity = self._body_velocity(
            self.tracked_body_ids
        )
        position_error = torch.sum(
            (self.body_pos_relative_w - robot_positions) ** 2, dim=-1
        ).mean(dim=-1)
        orientation_error = quat_error_magnitude(
            self.body_quat_relative_w, robot_orientations
        ).square().mean(dim=-1)
        linear_velocity_error = torch.sum(
            (self._target_body_lin_vel() - robot_linear_velocity) ** 2, dim=-1
        ).mean(dim=-1)
        angular_velocity_error = torch.sum(
            (self._target_body_ang_vel() - robot_angular_velocity) ** 2, dim=-1
        ).mean(dim=-1)
        joint_position = self.qpos[:, self.joint_qpos_ids]
        below = torch.clamp(self.soft_joint_limits[:, 0] - joint_position, min=0.0)
        above = torch.clamp(joint_position - self.soft_joint_limits[:, 1], min=0.0)
        return {
            "motion_global_root_pos": torch.exp(
                -torch.sum((target_anchor_position - robot_anchor_position) ** 2, dim=-1)
                / 0.3**2
            ),
            "motion_global_root_ori": torch.exp(
                -quat_error_magnitude(target_anchor_orientation, robot_anchor_orientation).square()
                / 0.4**2
            ),
            "motion_body_pos": torch.exp(-position_error / 0.3**2),
            "motion_body_ori": torch.exp(-orientation_error / 0.4**2),
            "motion_body_lin_vel": torch.exp(-linear_velocity_error / 1.0**2),
            "motion_body_ang_vel": torch.exp(-angular_velocity_error / 3.14**2),
            "action_rate_l2": torch.sum((self.actions - self.prev_actions) ** 2, dim=-1),
            "joint_limit": torch.sum(below + above, dim=-1),
            "self_collisions": self.self_collision_hits,
        }

    def _compute_rewards(self) -> torch.Tensor:
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

    def _compute_metrics(self) -> None:
        target_positions = self._target_body_pos()
        target_orientations = self._target_body_quat()
        robot_positions = self._robot_body_pos()
        robot_orientations = self._robot_body_quat()
        robot_linear_velocity, robot_angular_velocity = self._body_velocity(
            self.tracked_body_ids
        )
        anchor = self.anchor_motion_index
        values = {
            "error_anchor_pos": torch.linalg.norm(
                target_positions[:, anchor] - robot_positions[:, anchor], dim=-1
            ),
            "error_anchor_rot": quat_error_magnitude(
                target_orientations[:, anchor], robot_orientations[:, anchor]
            ),
            "error_anchor_lin_vel": torch.linalg.norm(
                self._target_body_lin_vel()[:, anchor] - robot_linear_velocity[:, anchor],
                dim=-1,
            ),
            "error_anchor_ang_vel": torch.linalg.norm(
                self._target_body_ang_vel()[:, anchor] - robot_angular_velocity[:, anchor],
                dim=-1,
            ),
            "error_body_pos": torch.linalg.norm(
                self.body_pos_relative_w - robot_positions, dim=-1
            ).mean(dim=-1),
            "error_body_rot": quat_error_magnitude(
                self.body_quat_relative_w, robot_orientations
            ).mean(dim=-1),
            "error_body_lin_vel": torch.linalg.norm(
                self._target_body_lin_vel() - robot_linear_velocity, dim=-1
            ).mean(dim=-1),
            "error_body_ang_vel": torch.linalg.norm(
                self._target_body_ang_vel() - robot_angular_velocity, dim=-1
            ).mean(dim=-1),
            "error_joint_pos": torch.linalg.norm(
                self._target_joint_pos() - self.qpos[:, self.joint_qpos_ids], dim=-1
            ),
            "error_joint_vel": torch.linalg.norm(
                self._target_joint_vel() - self.qvel[:, self.joint_qvel_ids], dim=-1
            ),
        }
        for name, value in values.items():
            self.metric_sums[name] += value
        self.metric_step_count += 1

    def _compute_observations(self) -> TensorDict:
        command = torch.cat((self._target_joint_pos(), self._target_joint_vel()), dim=-1)
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
        if actor.shape[1] != 161 or critic.shape[1] != 287:
            raise RuntimeError(
                f"Observation contract changed: actor={actor.shape}, critic={critic.shape}"
            )
        return TensorDict(
            {"actor": actor, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _episode_logs(
        self,
        env_ids: torch.Tensor,
        terminated: torch.Tensor | None,
        time_out: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if env_ids.numel() == 0:
            return {}
        logs: dict[str, torch.Tensor] = {}
        for name, values in self.episode_reward_sums.items():
            logs[f"Episode_Reward/{name}"] = values[env_ids].mean() / self.cfg.episode_length_s
        counts = torch.clamp(self.metric_step_count[env_ids].float(), min=1.0)
        for name, values in self.metric_sums.items():
            logs[f"Metrics/motion/{name}"] = (values[env_ids] / counts).mean()
        probabilities = self._sampling_probabilities()
        entropy = -(probabilities * (probabilities + 1.0e-12).log()).sum()
        logs["Metrics/motion/sampling_entropy"] = entropy / math.log(self.bin_count)
        logs["Metrics/motion/sampling_top1_prob"] = probabilities.max()
        logs["Metrics/motion/sampling_top1_bin"] = (
            probabilities.argmax().float() / self.bin_count
        )
        if terminated is not None:
            logs["Episode_Termination/reference_error"] = terminated[env_ids].float().sum()
        if time_out is not None:
            logs["Episode_Termination/time_out"] = time_out[env_ids].float().sum()
        return logs

    def _reset_idx(
        self,
        env_ids: torch.Tensor,
        *,
        terminated: torch.Tensor | None = None,
        time_out: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if env_ids.numel() == 0:
            return {}
        logs = self._episode_logs(env_ids, terminated, time_out)
        self.sim.reset(env_ids)
        self._resample_motion(env_ids, terminated=terminated)
        self.sim.tensor("ctrl")[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.self_collision_hits[env_ids] = 0.0
        for values in self.episode_reward_sums.values():
            values[env_ids] = 0.0
        for values in self.metric_sums.values():
            values[env_ids] = 0.0
        self.metric_step_count[env_ids] = 0
        randomization = self.cfg.domain_randomization
        if randomization.enabled:
            self.push_time_left[env_ids] = _uniform(
                *randomization.push_interval_s,
                (len(env_ids),),
                device=self.device,
                generator=self._generator,
            )
        return logs

    def reset(self) -> tuple[TensorDict, dict]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        logs = self._reset_idx(env_ids)
        self.sim.forward()
        # Initialize relative targets immediately; MJLab's evaluator performs the
        # same command update before its first policy step.
        self._update_relative_targets()
        self._obs = self._compute_observations()
        return self._obs, {"log": logs}

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Expected actions {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}"
            )
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.to(self.device))
        action_bias = self.encoder_bias[:, self.action_joint_indices]
        targets = self.default_action_pos + self.action_scale * self.actions - action_bias
        controls = targets[:, self.ctrl_from_action_indices]
        self.self_collision_hits.zero_()
        for _ in range(self.cfg.decimation):
            self.sim.tensor("ctrl").copy_(controls)
            self.sim.step()
            self_force = self._sensor("self_collision_force")
            self.self_collision_hits += (
                torch.linalg.norm(self_force, dim=-1) > 10.0
            ).float()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        terminated, time_out = self._compute_terminations()
        rewards = self._compute_rewards()
        self._compute_metrics()
        dones_bool = terminated | time_out
        reset_ids = dones_bool.nonzero(as_tuple=False).flatten()
        logs = self._reset_idx(reset_ids, terminated=terminated, time_out=time_out)
        self.sim.forward()
        if self._advance_motion():
            self.sim.forward()
        self._update_relative_targets()
        if self._apply_pushes():
            self.sim.forward()
        self._obs = self._compute_observations()
        extras = {"time_outs": time_out, "terminated": terminated, "log": logs}
        return self._obs, rewards, dones_bool.long(), extras

    def reward_terms(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self._last_reward_terms.items()}

    def evaluation_metrics(self) -> dict[str, torch.Tensor]:
        robot_positions = self._robot_body_pos()
        robot_orientations = self._robot_body_quat()
        reference_relative = self.body_pos_relative_w
        reference_root_relative = (
            self._target_body_pos()
            - self._target_body_pos()[:, self.anchor_motion_index : self.anchor_motion_index + 1]
        )
        robot_root_relative = (
            robot_positions
            - robot_positions[:, self.anchor_motion_index : self.anchor_motion_index + 1]
        )
        return {
            "mpkpe": torch.linalg.norm(reference_relative - robot_positions, dim=-1).mean(-1),
            "r_mpkpe": torch.linalg.norm(
                reference_root_relative - robot_root_relative, dim=-1
            ).mean(-1),
            "joint_vel_error": torch.linalg.norm(
                self._target_joint_vel() - self.qvel[:, self.joint_qvel_ids], dim=-1
            ),
            "ee_pos_error": torch.linalg.norm(
                reference_relative[:, self.end_effector_indices]
                - robot_positions[:, self.end_effector_indices],
                dim=-1,
            ).mean(-1),
            "ee_ori_error": quat_error_magnitude(
                self.body_quat_relative_w[:, self.end_effector_indices],
                robot_orientations[:, self.end_effector_indices],
            ).mean(-1),
        }

    def set_motion_frame(self, frame: int) -> None:
        if not 0 <= frame < self.motion.num_frames:
            raise ValueError(f"Motion frame must be in [0, {self.motion.num_frames - 1}]")
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.sim.reset(env_ids)
        self.time_steps.fill_(frame)
        self._write_sampled_motion_state(env_ids)
        self.episode_length_buf.zero_()
        self.actions.zero_()
        self.prev_actions.zero_()
        self.sim.forward()
        self._update_relative_targets()
        self._obs = self._compute_observations()

    def training_state(self) -> dict:
        return {
            "common_step_counter": self.common_step_counter,
            "bin_failed_count": self.bin_failed_count.detach().cpu(),
        }

    def load_training_state(self, state: dict | None) -> None:
        if state is None:
            return
        self.common_step_counter = int(state.get("common_step_counter", 0))
        bin_failed_count = state.get("bin_failed_count")
        if bin_failed_count is not None:
            value = torch.as_tensor(bin_failed_count, device=self.device)
            if value.shape == self.bin_failed_count.shape:
                self.bin_failed_count.copy_(value)

    def cpu_state(self, env_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qpos[env_idx].detach().cpu(), self.qvel[env_idx].detach().cpu()

    def reference_qpos(self, env_idx: int = 0) -> torch.Tensor:
        frame = self.time_steps[env_idx]
        result = torch.zeros(self.model.nq, device=self.device)
        result[:3] = self.motion.body_pos_w[frame, 0]
        result[3:7] = self.motion.body_quat_w[frame, 0]
        result[self.joint_qpos_ids] = self.motion.joint_pos[frame]
        return result.detach().cpu()

    def close(self) -> None:
        pass
