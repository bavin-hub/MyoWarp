"""Standalone MuJoCo-Warp/RSL-RL environment for H1 with an OSL right leg."""

from __future__ import annotations

import math
from collections.abc import Iterable

import mujoco
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .config import H1WithOslVelocityCfg
from .math import quat_apply, quat_apply_inverse, quat_mul, wrap_to_pi, yaw_quat
from .model import (
    SOCKET_JOINT_PREFIX,
    build_h1_with_osl_velocity_model,
    make_h1_with_osl_home_qpos,
    names_for,
)
from .sim import StandaloneMujocoWarpSim


REWARD_WEIGHTS = {
    "track_linear_velocity": 1.0,
    "track_angular_velocity": 1.0,
    "pose": 1.0,
    "body_ang_vel": -0.05,
    "angular_momentum": -0.025,
    "action_rate_l2": -0.05,
    "foot_clearance": -1.0,
    "foot_slip": -0.25,
    "soft_landing": -1.0e-3,
    "body_orientation_l2": -1.0,
    "is_terminated": -200.0,
    "joint_acc_l2": -2.5e-7,
    "joint_pos_limits": -10.0,
    "foot_gait": 0.5,
    "stand_still": -1.0,
    "self_collisions": -1.0,
}


def _ids_by_name(
    model: mujoco.MjModel,
    objtype: mujoco.mjtObj,
    names: Iterable[str],
) -> list[int]:
    result = []
    for name in names:
        object_id = mujoco.mj_name2id(model, objtype, name)
        if object_id < 0:
            raise ValueError(f"{objtype.name} named '{name}' was not found")
        result.append(object_id)
    return result


def _uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator) * (high - low) + low


class H1WithOslVelocityEnv(VecEnv):
    """Batched flat-terrain velocity task with no MJLab runtime dependency."""

    cfg: H1WithOslVelocityCfg

    def __init__(self, cfg: H1WithOslVelocityCfg) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.max_episode_length = cfg.max_episode_length
        self.num_actions = 26
        self.step_dt = cfg.step_dt
        self.physics_dt = cfg.physics_dt
        self.common_step_counter = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed)

        model = build_h1_with_osl_velocity_model(
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
        self._allocate_buffers()
        self._apply_startup_randomization()
        self._obs = TensorDict({}, batch_size=[self.num_envs], device=self.device)
        self.reset()

    def _build_indices(self) -> None:
        model = self.model
        joint_names = names_for(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        actuator_names = names_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
        self.joint_names = joint_names[1:]
        self.actuated_joint_names = [
            name for name in self.joint_names if not name.startswith(SOCKET_JOINT_PREFIX)
        ]
        # MJLab exposes policy actions in robot joint order, independently of
        # the order in which actuator groups were added to MuJoCo.
        self.action_joint_names = self.actuated_joint_names
        self.actuator_joint_names = actuator_names
        if len(self.action_joint_names) != 26 or len(self.actuator_joint_names) != 26:
            raise RuntimeError("The standalone task requires exactly 26 actuated joints")

        joint_ids = _ids_by_name(
            model, mujoco.mjtObj.mjOBJ_JOINT, self.actuated_joint_names
        )
        action_joint_ids = _ids_by_name(
            model, mujoco.mjtObj.mjOBJ_JOINT, self.action_joint_names
        )
        self.ctrl_from_action_indices = torch.tensor(
            [self.action_joint_names.index(name) for name in self.actuator_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self.joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self.joint_qpos_ids = torch.tensor(
            model.jnt_qposadr[joint_ids], device=self.device, dtype=torch.long
        )
        self.joint_qvel_ids = torch.tensor(
            model.jnt_dofadr[joint_ids], device=self.device, dtype=torch.long
        )
        self.action_qpos_ids = torch.tensor(
            model.jnt_qposadr[action_joint_ids], device=self.device, dtype=torch.long
        )
        self.action_qvel_ids = torch.tensor(
            model.jnt_dofadr[action_joint_ids], device=self.device, dtype=torch.long
        )
        self.all_joint_qpos_ids = torch.tensor(
            model.jnt_qposadr[1:], device=self.device, dtype=torch.long
        )
        self.all_joint_qvel_ids = torch.tensor(
            model.jnt_dofadr[1:], device=self.device, dtype=torch.long
        )

        self.root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.foot_site_ids = torch.tensor(
            _ids_by_name(model, mujoco.mjtObj.mjOBJ_SITE, ("left_foot", "right_foot")),
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
        for name in (
            "imu_ang_vel",
            "imu_lin_vel",
            "root_angmom",
            "feet_ground_left_found",
            "feet_ground_left_force",
            "feet_ground_right_found",
            "feet_ground_right_force",
            "self_collision_found",
            "self_collision_force",
        ):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            start = int(model.sensor_adr[sensor_id])
            self._sensor_slices[name] = slice(start, start + int(model.sensor_dim[sensor_id]))

        actuator_scales = (
            0.25 * model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]
        )
        action_scales = actuator_scales[
            [self.actuator_joint_names.index(name) for name in self.action_joint_names]
        ]
        self.action_scale = torch.tensor(action_scales, device=self.device, dtype=torch.float32)
        self.home_qpos = torch.tensor(
            make_h1_with_osl_home_qpos(model),
            device=self.device,
            dtype=torch.float32,
        )
        self.default_joint_pos = self.home_qpos[self.joint_qpos_ids]
        self.default_action_pos = self.home_qpos[self.action_qpos_ids]

        limits = torch.tensor(model.jnt_range[joint_ids], device=self.device, dtype=torch.float32)
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        width = limits[:, 1] - limits[:, 0]
        self.soft_joint_limits = torch.stack(
            (center - 0.45 * width, center + 0.45 * width), dim=-1
        )
        self.pose_std_standing = torch.full((26,), 0.05, device=self.device)
        self.pose_std_walking = self._pose_std(running=False)
        self.pose_std_running = self._pose_std(running=True)

    def _pose_std(self, *, running: bool) -> torch.Tensor:
        values = []
        for name in self.actuated_joint_names:
            if "hip_pitch" in name or "knee" in name:
                value = 0.5
            elif "hip_yaw" in name or "hip_roll" in name:
                value = 0.25 if running else 0.15
            elif "ankle_pitch" in name or name.startswith("osl_ankle"):
                value = 0.25 if running else 0.15
            elif "ankle_roll" in name:
                value = 0.1
            elif "torso" in name:
                value = 0.25 if running else 0.15
            elif "shoulder_pitch" in name:
                value = 0.25 if running else 0.15
            else:
                value = 0.1
            values.append(value)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _allocate_buffers(self) -> None:
        n = self.num_envs
        d = self.device
        self.episode_length_buf = torch.zeros(n, device=d, dtype=torch.long)
        self.actions = torch.zeros((n, 26), device=d)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)
        self.encoder_bias = torch.zeros_like(self.actions)
        self.commands = torch.zeros((n, 3), device=d)
        self._fixed_command: tuple[float, float, float] | None = None
        self.heading_target = torch.zeros(n, device=d)
        self.is_heading_env = torch.zeros(n, device=d, dtype=torch.bool)
        self.is_standing_env = torch.zeros(n, device=d, dtype=torch.bool)
        self.command_time_left = torch.zeros(n, device=d)
        self.push_time_left = torch.zeros(n, device=d)
        self.current_air_time = torch.zeros((n, 2), device=d)
        self.current_contact_time = torch.zeros((n, 2), device=d)
        self.previous_contact = torch.zeros((n, 2), device=d, dtype=torch.bool)
        self.first_contact = torch.zeros((n, 2), device=d, dtype=torch.bool)
        self.self_collision_hits = torch.zeros(n, device=d)
        self.peak_foot_height = torch.zeros((n, 2), device=d)
        self.episode_reward_sums = {
            name: torch.zeros(n, device=d) for name in REWARD_WEIGHTS
        }
        self.metric_action_acc_sum = torch.zeros(n, device=d)
        self.metric_step_count = torch.zeros(n, device=d, dtype=torch.long)
        self.command_error_xy_sum = torch.zeros(n, device=d)
        self.command_error_yaw_sum = torch.zeros(n, device=d)
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
            self.model.body_ipos[self.torso_body_id], device=self.device, dtype=torch.float32
        )
        offsets = torch.stack(
            [
                _uniform(*axis_range, (self.num_envs,), device=self.device, generator=self._generator)
                for axis_range in (
                    randomization.torso_com_offset_x,
                    randomization.torso_com_offset_y,
                    randomization.torso_com_offset_z,
                )
            ],
            dim=-1,
        )
        body_ipos[:, self.torso_body_id] = default_ipos + offsets
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
    def qacc(self) -> torch.Tensor:
        return self.sim.tensor("qacc")

    @property
    def sensordata(self) -> torch.Tensor:
        return self.sim.tensor("sensordata")

    def _sensor(self, name: str) -> torch.Tensor:
        return self.sensordata[:, self._sensor_slices[name]]

    def _root_quat(self) -> torch.Tensor:
        return self.sim.tensor("xquat")[:, self.root_body_id]

    def _body_link_velocity(self, body_id: int, position: torch.Tensor) -> torch.Tensor:
        cvel = self.sim.tensor("cvel")[:, body_id]
        subtree_com = self.sim.tensor("subtree_com")[:, self.root_body_id]
        linear = cvel[:, 3:6] - torch.cross(cvel[:, 0:3], subtree_com - position, dim=-1)
        return torch.cat((linear, cvel[:, 0:3]), dim=-1)

    def _root_link_velocity_w(self) -> torch.Tensor:
        position = self.sim.tensor("xpos")[:, self.root_body_id]
        return self._body_link_velocity(self.root_body_id, position)

    def _root_link_velocity_b(self) -> torch.Tensor:
        velocity = self._root_link_velocity_w()
        quat = self._root_quat()
        return torch.cat(
            (quat_apply_inverse(quat, velocity[:, :3]), quat_apply_inverse(quat, velocity[:, 3:])),
            dim=-1,
        )

    def _projected_gravity(self, body_id: int | None = None) -> torch.Tensor:
        body_id = self.root_body_id if body_id is None else body_id
        quat = self.sim.tensor("xquat")[:, body_id]
        gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(self.num_envs, 3)
        return quat_apply_inverse(quat, gravity)

    def _site_velocity(self, site_ids: torch.Tensor) -> torch.Tensor:
        positions = self.sim.tensor("site_xpos")[:, site_ids]
        body_ids = torch.tensor(
            self.model.site_bodyid[site_ids.cpu().numpy()], device=self.device, dtype=torch.long
        )
        cvel = self.sim.tensor("cvel")[:, body_ids]
        subtree_com = self.sim.tensor("subtree_com")[:, self.root_body_id].unsqueeze(1)
        linear = cvel[..., 3:6] - torch.cross(cvel[..., :3], subtree_com - positions, dim=-1)
        return torch.cat((linear, cvel[..., :3]), dim=-1)

    def _foot_contact_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        found = torch.cat(
            (
                self._sensor("feet_ground_left_found"),
                self._sensor("feet_ground_right_found"),
            ),
            dim=1,
        )
        forces = torch.stack(
            (
                self._sensor("feet_ground_left_force"),
                self._sensor("feet_ground_right_force"),
            ),
            dim=1,
        )
        return found > 0.0, forces

    def _update_contact_state(self) -> None:
        contact, _ = self._foot_contact_data()
        self.first_contact |= contact & ~self.previous_contact
        self.current_air_time = torch.where(
            contact,
            torch.zeros_like(self.current_air_time),
            self.current_air_time + self.physics_dt,
        )
        self.current_contact_time = torch.where(
            contact,
            self.current_contact_time + self.physics_dt,
            torch.zeros_like(self.current_contact_time),
        )
        self.previous_contact.copy_(contact)
        self_force = self._sensor("self_collision_force")
        self.self_collision_hits += (torch.linalg.norm(self_force, dim=-1) > 10.0).float()

    def _sample_command_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        command_cfg = self.cfg.command
        if self.common_step_counter > command_cfg.full_range_after_steps:
            return command_cfg.lin_vel_x, command_cfg.lin_vel_y
        return command_cfg.initial_lin_vel_x, command_cfg.initial_lin_vel_y

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._fixed_command is not None:
            self.commands[env_ids] = torch.tensor(
                self._fixed_command, device=self.device, dtype=self.commands.dtype
            )
            self.is_heading_env[env_ids] = False
            self.is_standing_env[env_ids] = False
            self.command_time_left[env_ids] = float("inf")
            return
        count = len(env_ids)
        command_cfg = self.cfg.command
        x_range, y_range = self._sample_command_ranges()
        self.commands[env_ids, 0] = _uniform(
            *x_range, (count,), device=self.device, generator=self._generator
        )
        self.commands[env_ids, 1] = _uniform(
            *y_range, (count,), device=self.device, generator=self._generator
        )
        self.commands[env_ids, 2] = _uniform(
            *command_cfg.ang_vel_z, (count,), device=self.device, generator=self._generator
        )
        self.heading_target[env_ids] = _uniform(
            *command_cfg.heading, (count,), device=self.device, generator=self._generator
        )
        self.is_heading_env[env_ids] = torch.rand(
            count, device=self.device, generator=self._generator
        ) <= command_cfg.heading_probability
        self.is_standing_env[env_ids] = torch.rand(
            count, device=self.device, generator=self._generator
        ) <= command_cfg.standing_probability
        self.command_time_left[env_ids] = _uniform(
            *command_cfg.resampling_time_s,
            (count,),
            device=self.device,
            generator=self._generator,
        )

    def _update_commands(self) -> None:
        self.command_time_left -= self.step_dt
        expired = (self.command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        self._resample_commands(expired)
        quat = self._root_quat()
        forward = quat_apply(quat, torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(self.num_envs, 3))
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_error = wrap_to_pi(self.heading_target - heading)
        yaw_command = torch.clamp(
            self.cfg.command.heading_stiffness * heading_error,
            self.cfg.command.ang_vel_z[0],
            self.cfg.command.ang_vel_z[1],
        )
        self.commands[:, 2] = torch.where(self.is_heading_env, yaw_command, self.commands[:, 2])
        self.commands[self.is_standing_env] = 0.0

    def _apply_pushes(self) -> bool:
        randomization = self.cfg.domain_randomization
        if not randomization.enabled:
            return False
        self.push_time_left -= self.step_dt
        env_ids = (self.push_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return False
        count = len(env_ids)
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
                *value_range, (count,), device=self.device, generator=self._generator
            )
        self.qvel[env_ids, :3] = velocity[:, :3]
        self.qvel[env_ids, 3:6] = quat_apply_inverse(
            self._root_quat()[env_ids], velocity[:, 3:]
        )
        self.push_time_left[env_ids] = _uniform(
            *randomization.push_interval_s,
            (count,),
            device=self.device,
            generator=self._generator,
        )
        return True

    def _command_active(self) -> torch.Tensor:
        magnitude = torch.linalg.norm(self.commands[:, :2], dim=1) + torch.abs(self.commands[:, 2])
        return magnitude > 0.1

    def _compute_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        angle = torch.acos(torch.clamp(-self._projected_gravity()[:, 2], -1.0, 1.0)).abs()
        terminated = angle > math.radians(70.0)
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _compute_reward_terms(self, terminated: torch.Tensor) -> dict[str, torch.Tensor]:
        root_velocity_b = self._root_link_velocity_b()
        linear_error = torch.sum((self.commands[:, :2] - root_velocity_b[:, :2]) ** 2, dim=1)
        linear_error += 2.0 * root_velocity_b[:, 2] ** 2
        angular_error = (self.commands[:, 2] - root_velocity_b[:, 5]) ** 2
        angular_error += 0.05 * torch.sum(root_velocity_b[:, 3:5] ** 2, dim=1)

        joint_pos = self.qpos[:, self.joint_qpos_ids]
        joint_vel = self.qvel[:, self.joint_qvel_ids]
        joint_error = joint_pos - self.default_joint_pos
        command_speed = torch.linalg.norm(self.commands[:, :2], dim=1) + torch.abs(self.commands[:, 2])
        std = torch.where(
            (command_speed < 0.1).unsqueeze(1),
            self.pose_std_standing,
            torch.where(
                (command_speed < 1.5).unsqueeze(1),
                self.pose_std_walking,
                self.pose_std_running,
            ),
        )

        torso_cvel = self.sim.tensor("cvel")[:, self.torso_body_id]
        angular_momentum = self._sensor("root_angmom")
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        foot_position = self.sim.tensor("site_xpos")[:, self.foot_site_ids]
        foot_velocity = self._site_velocity(self.foot_site_ids)
        active = self._command_active().float()
        foot_clearance = torch.sum(
            torch.abs(foot_position[..., 2] - 0.1)
            * torch.linalg.norm(foot_velocity[..., :2], dim=-1),
            dim=1,
        ) * active
        contact, contact_forces = self._foot_contact_data()
        foot_slip = torch.sum(
            torch.sum(foot_velocity[..., :2] ** 2, dim=-1) * contact.float(), dim=1
        ) * active
        soft_landing = torch.sum(
            torch.linalg.norm(contact_forces, dim=-1) * self.first_contact.float(), dim=1
        ) * active
        torso_gravity = self._projected_gravity(self.torso_body_id)
        below = torch.clamp(self.soft_joint_limits[:, 0] - joint_pos, min=0.0)
        above = torch.clamp(joint_pos - self.soft_joint_limits[:, 1], min=0.0)
        phase = (self.episode_length_buf.float() * self.step_dt / 0.6).unsqueeze(1)
        offsets = torch.tensor((0.0, 0.5), device=self.device).unsqueeze(0)
        desired_stance = torch.remainder(phase + offsets, 1.0) < 0.56
        foot_gait = (desired_stance == contact).float().mean(dim=1) * active
        stand_still = torch.sum(joint_error**2, dim=1) * (~self._command_active()).float()

        return {
            "track_linear_velocity": torch.exp(-linear_error / 0.5**2),
            "track_angular_velocity": torch.exp(-angular_error / math.sqrt(0.5) ** 2),
            "pose": torch.exp(-torch.mean(joint_error**2 / std**2, dim=1)),
            "body_ang_vel": torch.sum(torso_cvel[:, :2] ** 2, dim=1),
            "angular_momentum": torch.sum(angular_momentum**2, dim=1),
            "action_rate_l2": action_rate,
            "foot_clearance": foot_clearance,
            "foot_slip": foot_slip,
            "soft_landing": soft_landing,
            "body_orientation_l2": torch.sum(torso_gravity[:, :2] ** 2, dim=1),
            "is_terminated": terminated.float(),
            "joint_acc_l2": torch.sum(self.qacc[:, self.joint_qvel_ids] ** 2, dim=1),
            "joint_pos_limits": torch.sum(below + above, dim=1),
            "foot_gait": foot_gait,
            "stand_still": stand_still,
            "self_collisions": self.self_collision_hits,
        }

    def _compute_rewards(self, terminated: torch.Tensor) -> torch.Tensor:
        terms = self._compute_reward_terms(terminated)
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

    def _compute_observations(self) -> TensorDict:
        joint_pos = self.qpos[:, self.joint_qpos_ids] - self.default_joint_pos
        joint_vel = self.qvel[:, self.joint_qvel_ids]
        projected_gravity = self._projected_gravity()
        base_ang_vel = self._sensor("imu_ang_vel")
        phase_scalar = torch.remainder(
            self.episode_length_buf.float() * self.step_dt, 0.6
        ) / 0.6
        phase = torch.stack(
            (torch.sin(2.0 * torch.pi * phase_scalar), torch.cos(2.0 * torch.pi * phase_scalar)),
            dim=1,
        )
        phase = torch.where(self._command_active().unsqueeze(1), phase, torch.zeros_like(phase))
        critic_common = (
            base_ang_vel,
            projected_gravity,
            self.commands,
            phase,
            joint_pos,
            joint_vel,
            self.actions,
        )
        actor_parts = list(critic_common)
        if self.cfg.enable_observation_noise:
            noise = self.cfg.observation_noise
            actor_parts[0] = actor_parts[0] + _uniform(
                -noise.base_ang_vel,
                noise.base_ang_vel,
                actor_parts[0].shape,
                device=self.device,
                generator=self._generator,
            )
            actor_parts[1] = actor_parts[1] + _uniform(
                -noise.projected_gravity,
                noise.projected_gravity,
                actor_parts[1].shape,
                device=self.device,
                generator=self._generator,
            )
            actor_parts[4] = actor_parts[4] + _uniform(
                -noise.joint_pos,
                noise.joint_pos,
                actor_parts[4].shape,
                device=self.device,
                generator=self._generator,
            )
            actor_parts[5] = actor_parts[5] + _uniform(
                -noise.joint_vel,
                noise.joint_vel,
                actor_parts[5].shape,
                device=self.device,
                generator=self._generator,
            )
        _, contact_forces = self._foot_contact_data()
        critic = torch.cat(
            (
                *critic_common,
                self._sensor("imu_lin_vel"),
                self.sim.tensor("site_xpos")[:, self.foot_site_ids, 2],
                self.current_air_time,
                self.previous_contact.float(),
                torch.sign(contact_forces.flatten(start_dim=1))
                * torch.log1p(torch.abs(contact_forces.flatten(start_dim=1))),
            ),
            dim=1,
        )
        actor = torch.cat(tuple(actor_parts), dim=1)
        if actor.shape[1] != 89 or critic.shape[1] != 104:
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
        logs["Episode_Metrics/mean_action_acc"] = (
            self.metric_action_acc_sum[env_ids] / counts
        ).mean()
        logs["Metrics/twist/error_vel_xy"] = (
            self.command_error_xy_sum[env_ids] / counts
        ).mean()
        logs["Metrics/twist/error_vel_yaw"] = (
            self.command_error_yaw_sum[env_ids] / counts
        ).mean()
        if terminated is not None:
            logs["Episode_Termination/fell_over"] = terminated[env_ids].float().sum()
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
        count = len(env_ids)
        qpos = self.qpos
        qvel = self.qvel
        qpos[env_ids] = self.home_qpos
        qpos[env_ids, 0] += _uniform(-0.5, 0.5, (count,), device=self.device, generator=self._generator)
        qpos[env_ids, 1] += _uniform(-0.5, 0.5, (count,), device=self.device, generator=self._generator)
        yaw = _uniform(-3.14, 3.14, (count,), device=self.device, generator=self._generator)
        qpos[env_ids, 3:7] = quat_mul(
            qpos[env_ids, 3:7], yaw_quat(yaw)
        )
        qvel[env_ids] = 0.0
        self.sim.tensor("ctrl")[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.previous_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.self_collision_hits[env_ids] = 0.0
        self.peak_foot_height[env_ids] = 0.0
        for values in self.episode_reward_sums.values():
            values[env_ids] = 0.0
        self.metric_action_acc_sum[env_ids] = 0.0
        self.metric_step_count[env_ids] = 0
        self.command_error_xy_sum[env_ids] = 0.0
        self.command_error_yaw_sum[env_ids] = 0.0
        self._resample_commands(env_ids)
        randomization = self.cfg.domain_randomization
        if randomization.enabled:
            self.push_time_left[env_ids] = _uniform(
                *randomization.push_interval_s,
                (count,),
                device=self.device,
                generator=self._generator,
            )
        return logs

    def reset(self) -> tuple[TensorDict, dict]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        logs = self._reset_idx(env_ids)
        self.sim.forward()
        self._update_commands()
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
        self.prev_prev_actions.copy_(self.prev_actions)
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.to(self.device))
        targets = self.default_action_pos + self.action_scale * self.actions - self.encoder_bias
        controls = targets[:, self.ctrl_from_action_indices]
        self.first_contact.zero_()
        self.self_collision_hits.zero_()
        for _ in range(self.cfg.decimation):
            self.sim.tensor("ctrl").copy_(controls)
            self.sim.step()
            self._update_contact_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        terminated, time_out = self._compute_terminations()
        rewards = self._compute_rewards(terminated)
        action_acc = self.actions - 2.0 * self.prev_actions + self.prev_prev_actions
        self.metric_action_acc_sum += torch.mean(torch.abs(action_acc), dim=1)
        self.metric_step_count += 1
        root_velocity_b = self._root_link_velocity_b()
        self.command_error_xy_sum += torch.linalg.norm(
            self.commands[:, :2] - root_velocity_b[:, :2], dim=1
        )
        self.command_error_yaw_sum += torch.abs(self.commands[:, 2] - root_velocity_b[:, 5])

        dones_bool = terminated | time_out
        reset_ids = dones_bool.nonzero(as_tuple=False).flatten()
        logs = self._reset_idx(
            reset_ids,
            terminated=terminated,
            time_out=time_out,
        )
        self.sim.forward()
        self._update_commands()
        if self._apply_pushes():
            self.sim.forward()
        self._obs = self._compute_observations()
        extras = {"time_outs": time_out, "log": logs}
        return self._obs, rewards, dones_bool.long(), extras

    def reward_terms(self) -> dict[str, torch.Tensor]:
        """Return the most recently computed unweighted reward terms."""
        return {name: value.clone() for name, value in self._last_reward_terms.items()}

    def set_command(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        """Hold a fixed velocity command across episode resets."""
        self._fixed_command = (linear_x, linear_y, angular_z)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self._obs = self._compute_observations()

    def training_state(self) -> dict[str, int]:
        """Return task-level state that must survive an RSL-RL resume."""
        return {"common_step_counter": self.common_step_counter}

    def load_training_state(self, state: dict | None) -> None:
        """Restore task-level state from a checkpoint when it is available."""
        if state is not None:
            self.common_step_counter = int(state.get("common_step_counter", 0))

    def cpu_state(self, env_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy one world's generalized state for evaluation rendering."""
        return self.qpos[env_idx].detach().cpu(), self.qvel[env_idx].detach().cpu()

    def close(self) -> None:
        pass
