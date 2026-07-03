from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import torch

from myowarp.backends import MujocoWarpBackend
from myowarp.config import TrainConfig
from myowarp.io import ReferenceData, load_reference_data
from myowarp.utils import ModelIndex, build_model_index


class MyoAssistLegWarpEnv:
    """Batched MuJoCo Warp port of the MyoAssist leg imitation env.

    The API is intentionally tensor-native:

        obs = env.reset()
        obs, reward, done, info = env.step(actions)

    `actions` must have shape `[num_envs, model.nu]`.
    """

    def __init__(self, config: TrainConfig, root_dir: str | Path):
        self.config = config
        self.root_dir = Path(root_dir)
        self.env_params = config.env_params
        self.num_envs = int(config.warp_params.num_envs)
        self.device = config.warp_params.device
        self.model_path = self.root_dir / self.env_params.model_path
        self.control_dt = 1.0 / float(self.env_params.control_framerate)
        self.frame_skip = int(self.env_params.physics_sim_framerate // self.env_params.control_framerate)

        self.cpu_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.cpu_model.opt.timestep = 1.0 / float(self.env_params.physics_sim_framerate)
        ctrlrange = torch.as_tensor(self.cpu_model.actuator_ctrlrange, dtype=torch.float32, device=self.device)
        self.ctrl_mid = torch.mean(ctrlrange, dim=-1)
        self.ctrl_half_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        self.model_index = build_model_index(
            self.cpu_model,
            qpos_keys=list(self.env_params.observation_joint_pos_keys),
            qvel_keys=list(self.env_params.observation_joint_vel_keys),
            sensor_keys=list(self.env_params.observation_sensor_keys),
        )

        self.reference_data = load_reference_data(
            self._resolve_reference_path(self.env_params.reference_data_path),
            control_framerate=self.env_params.control_framerate,
            device=self.device,
        )

        self.backend = MujocoWarpBackend(
            model_path=self.model_path,
            num_envs=self.num_envs,
            device=self.device,
            physics_timestep=1.0 / float(self.env_params.physics_sim_framerate),
        )

        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.target_velocity = torch.full(
            (self.num_envs,),
            float(self.env_params.min_target_velocity),
            dtype=torch.float32,
            device=self.device,
        )
        self.reference_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_muscle_activation = torch.zeros(
            (self.num_envs, len(self.model_index.muscle_actuator_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        self.has_prev_muscle_activation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reward_scalar_weights = self._make_reward_scalar_weights()

        self.velocity_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.starting_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.target_velocity_period = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_step_changed_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.r_heel_history = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.l_heel_history = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_heel_strike_foot = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.footstep_delta_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.delta_velocity_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.activation_square_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reward_muscle_activation_penalty_per_step = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.reward_average_velocity_per_step = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reward_footstep_delta_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _resolve_reference_path(self, path: str) -> Path:
        direct = self.root_dir / path
        if direct.exists():
            return direct
        legacy_prefix = "rl_train/reference_data/"
        if path.startswith(legacy_prefix):
            migrated = self.root_dir / "reference_data" / Path(path).name
            if migrated.exists():
                return migrated
        return direct

    @property
    def action_size(self) -> int:
        return self.model_index.nu

    @property
    def observation_size(self) -> int:
        return (
            len(self.model_index.qpos)
            + len(self.model_index.qvel)
            + len(self.model_index.muscle_actuator_ids)
            + sum(sensor.dim for sensor in self.model_index.sensors.values())
            + 1
        )

    def reset(self) -> torch.Tensor:
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.get_obs()

    def step(self, actions: torch.Tensor):
        self._modulate_target_velocity()
        actions = actions.to(device=self.device, dtype=torch.float32)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.backend.set_ctrl(self._normalized_action_to_ctrl(actions))
        self.backend.step(self.frame_skip)

        self.step_count += 1
        next_reference_index = self.reference_index + 1
        out_of_reference = next_reference_index >= self.reference_data.length
        self.reference_index = torch.clamp(next_reference_index, max=self.reference_data.length - 1)

        obs = self.get_obs()
        reward, reward_info = self.get_reward(obs)
        height_done, trajectory_done = self._termination_causes()
        terminated = height_done | trajectory_done
        truncated = self.step_count >= int(self.env_params.custom_max_episode_steps)
        done = terminated | truncated | out_of_reference
        reward = torch.where(out_of_reference, torch.zeros_like(reward), reward)
        info: dict[str, Any] = {
            "reward": reward_info,
            "terminated": terminated,
            "height_done": height_done,
            "trajectory_done": trajectory_done,
            "truncated": truncated | out_of_reference,
            "out_of_reference": out_of_reference,
        }
        if torch.any(done):
            self._reset_envs(done)
            obs = self.get_obs()
        return obs, reward, done, info

    def get_obs(self) -> torch.Tensor:
        data = self.backend.state.data
        qpos = self._gather_qpos(data)
        qvel = self._gather_qvel(data)
        act = self._gather_muscle_activation(data)
        sensor = self._gather_sensors(data)
        target = self.target_velocity[:, None]
        return torch.cat([qpos, qvel, act, sensor, target], dim=-1)

    def get_reward(self, obs: torch.Tensor):
        components = self._reward_components()
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for key, value in components.items():
            reward += self.reward_scalar_weights.get(key, 0.0) * value
        return reward, components

    def _reward_components(self) -> dict[str, torch.Tensor]:
        qpos_reward = self._qpos_imitation_reward()
        qvel_reward = self._qvel_imitation_reward()
        end_effector_reward = torch.full((self.num_envs,), self.control_dt, dtype=torch.float32, device=self.device)
        forward_reward = self._forward_reward()
        act = self._gather_muscle_activation(self.backend.state.data)
        activation_penalty = -self.control_dt * torch.mean(act, dim=-1)
        activation_diff_penalty = self._muscle_activation_diff_penalty(act)
        foot_force_penalty = self._foot_force_penalty()
        joint_constraint_force_penalty = self._joint_constraint_force_penalty()
        per_step_rewards = self._reward_per_step(act)
        self.prev_muscle_activation = act.detach()
        self.has_prev_muscle_activation[:] = True
        components = {
            "qpos_imitation_rewards": qpos_reward,
            "qvel_imitation_rewards": qvel_reward,
            "end_effector_imitation_reward": end_effector_reward,
            "forward_reward": forward_reward,
            "muscle_activation_penalty": activation_penalty,
            "muscle_activation_diff_penalty": activation_diff_penalty,
            "foot_force_penalty": foot_force_penalty,
            "joint_constraint_force_penalty": joint_constraint_force_penalty,
        }
        components.update(per_step_rewards)
        return components

    def get_done(self) -> torch.Tensor:
        height_done, trajectory_done = self._termination_causes()
        return height_done | trajectory_done

    def _termination_causes(self) -> tuple[torch.Tensor, torch.Tensor]:
        pelvis_ty = self._joint_qpos("pelvis_ty")
        out_of_height = pelvis_ty < float(self.env_params.safe_height)
        qpos_diff = self._qpos_diff_tensor()
        out_of_trajectory = torch.any(torch.abs(qpos_diff) > float(self.env_params.out_of_trajectory_threshold), dim=-1)
        return out_of_height, out_of_trajectory

    def _make_reward_scalar_weights(self) -> dict[str, float]:
        weights = {}
        for key, value in self.env_params.reward_keys_and_weights.items():
            if isinstance(value, dict):
                weights[key] = float(sum(value.values()))
            else:
                weights[key] = float(value)
        return weights

    def _reset_envs(self, mask: torch.Tensor) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if not torch.any(mask):
            return
        count = int(mask.sum().item())
        self.step_count[mask] = 0
        self.reference_index[mask] = self._sample_reference_indices(count)
        qpos = self._tensor_field("qpos").clone()
        qvel = self._tensor_field("qvel").clone()
        reset_qpos, reset_qvel = self._reference_state(
            self.reference_index[mask],
            target_velocity=self.target_velocity[mask],
        )
        self._reset_target_velocity(mask)
        reset_qvel[:, self._joint_qvel_adr("pelvis_tx")] = self.target_velocity[mask]
        qpos[mask] = reset_qpos
        qvel[mask] = reset_qvel
        self.backend.set_state(qpos, qvel)

        ctrl = self._tensor_field("ctrl").clone()
        ctrl[mask] = 0.0
        self.backend.set_field("ctrl", ctrl)
        if self.model_index.na > 0 and hasattr(self.backend.state.data, "act"):
            act = self._tensor_field("act").clone()
            act[mask] = 0.0
            self.backend.set_field("act", act)
        if hasattr(self.backend.state.data, "time"):
            time = self._tensor_field("time").clone()
            time[mask] = 0.0
            self.backend.set_field("time", time)

        self.prev_muscle_activation[mask] = 0.0
        self.has_prev_muscle_activation[mask] = False
        self.r_heel_history[mask] = 0.0
        self.l_heel_history[mask] = 0.0
        self.last_heel_strike_foot[mask] = 0
        self._reset_reward_state(mask)
        self.backend.forward()

    def _normalized_action_to_ctrl(self, actions: torch.Tensor) -> torch.Tensor:
        return self.ctrl_mid + actions * self.ctrl_half_range

    def _reset_target_velocity(self, mask: torch.Tensor) -> None:
        count = int(mask.sum().item())
        min_velocity = float(self.env_params.min_target_velocity)
        max_velocity = float(self.env_params.max_target_velocity)
        min_period = float(self.env_params.min_target_velocity_period)
        max_period = float(self.env_params.max_target_velocity_period)
        self.velocity_mode[mask] = torch.randint(0, 3, (count,), device=self.device)
        self.starting_phase[mask] = 2.0 * math.pi * torch.rand(count, dtype=torch.float32, device=self.device)
        self.target_velocity_period[mask] = min_period + (max_period - min_period) * torch.rand(
            count, dtype=torch.float32, device=self.device
        )
        self.prev_step_changed_time[mask] = 0.0
        self.target_velocity[mask] = min_velocity + (max_velocity - min_velocity) * torch.rand(
            count, dtype=torch.float32, device=self.device
        )
        sinusoidal_mask = mask & (self.velocity_mode == 1)
        if torch.any(sinusoidal_mask):
            self.target_velocity[sinusoidal_mask] = self._sinusoidal_target_velocity(sinusoidal_mask)

    def _modulate_target_velocity(self) -> None:
        sinusoidal_mask = self.velocity_mode == 1
        if torch.any(sinusoidal_mask):
            self.target_velocity[sinusoidal_mask] = self._sinusoidal_target_velocity(sinusoidal_mask)

        step_mask = self.velocity_mode == 2
        if torch.any(step_mask):
            time = self.step_count.to(torch.float32) * self.control_dt
            change_mask = step_mask & ((time - self.prev_step_changed_time) > self.target_velocity_period)
            count = int(change_mask.sum().item())
            if count:
                min_velocity = float(self.env_params.min_target_velocity)
                max_velocity = float(self.env_params.max_target_velocity)
                self.target_velocity[change_mask] = min_velocity + (max_velocity - min_velocity) * torch.rand(
                    count, dtype=torch.float32, device=self.device
                )
                self.prev_step_changed_time[change_mask] = time[change_mask]

    def _sinusoidal_target_velocity(self, mask: torch.Tensor) -> torch.Tensor:
        min_velocity = float(self.env_params.min_target_velocity)
        max_velocity = float(self.env_params.max_target_velocity)
        time = self.step_count[mask].to(torch.float32) * self.control_dt
        phase = self.starting_phase[mask] + 2.0 * math.pi * time / self.target_velocity_period[mask]
        return min_velocity + (max_velocity - min_velocity) * (torch.sin(phase) + 1.0) / 2.0

    def _sample_reference_indices(self, count: int) -> torch.Tensor:
        if self.env_params.flag_random_ref_index:
            high = max(1, int(self.reference_data.length * 0.8))
            return torch.randint(high, (count,), device=self.device)
        return torch.zeros(count, dtype=torch.long, device=self.device)

    def _reference_state(
        self,
        ref_index: torch.Tensor,
        target_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_velocity is None:
            target_velocity = self.target_velocity
        count = int(ref_index.numel())
        qpos = torch.as_tensor(self.cpu_model.key_qpos[0], dtype=torch.float32, device=self.device).repeat(
            count, 1
        )
        qvel = torch.as_tensor(self.cpu_model.key_qvel[0], dtype=torch.float32, device=self.device).repeat(
            count, 1
        )
        ref_pelvis_vx = self.reference_data.series_data["dq_pelvis_tx"][ref_index]
        speed_ratio = target_velocity / torch.clamp(ref_pelvis_vx, min=1e-5)

        for key in self.env_params.reference_data_keys:
            q_key = f"q_{key}"
            dq_key = f"dq_{key}"
            if q_key in self.reference_data.series_data:
                qpos[:, self._joint_qpos_adr(key)] = self.reference_data.series_data[q_key][ref_index]
            if dq_key in self.reference_data.series_data:
                qvel[:, self._joint_qvel_adr(key)] = self.reference_data.series_data[dq_key][ref_index] * speed_ratio

        qpos[:, self._joint_qpos_adr("pelvis_tx")] = 0.0
        qvel[:, self._joint_qvel_adr("pelvis_tx")] = target_velocity
        return qpos, qvel

    def _tensor_field(self, name: str) -> torch.Tensor:
        return self.backend.get_tensor(name).to(device=self.device, dtype=torch.float32)

    def _gather_qpos(self, data: Any) -> torch.Tensor:
        qpos = self._tensor_field("qpos")
        return torch.stack([qpos[:, adr] for adr in self.model_index.qpos.values()], dim=-1)

    def _gather_qvel(self, data: Any) -> torch.Tensor:
        qvel = self._tensor_field("qvel")
        return torch.stack([qvel[:, adr] for adr in self.model_index.qvel.values()], dim=-1)

    def _gather_muscle_activation(self, data: Any) -> torch.Tensor:
        if self.model_index.na > 0 and hasattr(data, "act"):
            act = self._tensor_field("act")
            return act[:, : len(self.model_index.muscle_actuator_ids)]
        ctrl = self._tensor_field("ctrl")
        return ctrl[:, self.model_index.muscle_actuator_ids]

    def _gather_sensors(self, data: Any) -> torch.Tensor:
        sensordata = self._tensor_field("sensordata")
        parts = []
        body_mass = torch.tensor(float(self.cpu_model.body_mass.sum()), device=self.device)
        weight = body_mass * 9.81
        for name, sensor in self.model_index.sensors.items():
            values = sensordata[:, sensor.adr : sensor.adr + sensor.dim]
            if "foot" in name or "toes" in name:
                values = values / weight
            parts.append(values)
        if not parts:
            return torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return torch.cat(parts, dim=-1)

    def _joint_qpos(self, name: str) -> torch.Tensor:
        qpos = self._tensor_field("qpos")
        return qpos[:, self.model_index.qpos.get(name, self._joint_qpos_adr(name))]

    def _joint_qvel(self, name: str) -> torch.Tensor:
        qvel = self._tensor_field("qvel")
        return qvel[:, self.model_index.qvel.get(name, self._joint_qvel_adr(name))]

    def _joint_qpos_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.cpu_model.jnt_qposadr[joint_id])

    def _joint_qvel_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.cpu_model.jnt_dofadr[joint_id])

    def _qpos_imitation_reward(self) -> torch.Tensor:
        weights = self.env_params.reward_keys_and_weights.get("qpos_imitation_rewards", {})
        total = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for key, weight in weights.items():
            current = self._joint_qpos(key)
            ref = self.reference_data.series_data[f"q_{key}"][self.reference_index]
            total += float(weight) * self.control_dt * torch.exp(-8.0 * torch.square(current - ref))
        return total

    def _qvel_imitation_reward(self) -> torch.Tensor:
        weights = self.env_params.reward_keys_and_weights.get("qvel_imitation_rewards", {})
        total = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        ref_pelvis = self.reference_data.series_data["dq_pelvis_tx"][self.reference_index]
        speed_ratio = self.target_velocity / torch.clamp(ref_pelvis, min=1e-5)
        for key, weight in weights.items():
            current = self._joint_qvel(key)
            ref = self.reference_data.series_data[f"dq_{key}"][self.reference_index] * speed_ratio
            total += float(weight) * self.control_dt * torch.exp(-8.0 * torch.square(current - ref))
        return total

    def _forward_reward(self) -> torch.Tensor:
        pelvis_vx = self._joint_qvel("pelvis_tx")
        return self.control_dt * torch.exp(-5.0 * torch.square(pelvis_vx - self.target_velocity))

    def _qpos_diff_tensor(self) -> torch.Tensor:
        diffs = []
        weights = self.env_params.reward_keys_and_weights.get("qpos_imitation_rewards", {})
        for key in weights.keys():
            current = self._joint_qpos(key)
            ref = self.reference_data.series_data[f"q_{key}"][self.reference_index]
            diffs.append(current - ref)
        if not diffs:
            return torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return torch.stack(diffs, dim=-1)

    def _muscle_activation_diff_penalty(self, act: torch.Tensor) -> torch.Tensor:
        value = self.control_dt * torch.mean(
            torch.exp(-4.0 * torch.square(self.prev_muscle_activation - act)),
            dim=-1,
        )
        return torch.where(self.has_prev_muscle_activation, value, torch.zeros_like(value))

    def _foot_force_penalty(self) -> torch.Tensor:
        body_mass = torch.tensor(float(self.cpu_model.body_mass.sum()), device=self.device)
        model_weight = body_mass * 9.81
        normalized_foot_force_sum = (torch.abs(self._foot_force("r")) + torch.abs(self._foot_force("l"))) / model_weight
        return -self.control_dt * torch.clamp(normalized_foot_force_sum - 1.2, min=0.0)

    def _joint_constraint_force_penalty(self) -> torch.Tensor:
        if not self.env_params.joint_limit_sensor_keys:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        sensor_values = [
            torch.abs(self._sensor_values(sensor_name)).amax(dim=-1)
            for sensor_name in self.env_params.joint_limit_sensor_keys
        ]
        max_constraint_force = torch.stack(sensor_values, dim=-1).amax(dim=-1)
        body_mass = torch.tensor(float(self.cpu_model.body_mass.sum()), device=self.device)
        return -self.control_dt * max_constraint_force / (body_mass * 9.81)

    def _reward_per_step(self, act: torch.Tensor) -> dict[str, torch.Tensor]:
        self.footstep_delta_time += self.control_dt
        self.delta_velocity_sum += self.control_dt * (self._joint_qvel("pelvis_tx") - self.target_velocity)
        self.activation_square_sum += self.control_dt * torch.sum(torch.square(act), dim=-1)

        right_force = self._foot_force("r")
        left_force = self._foot_force("l")
        self.r_heel_history = torch.roll(self.r_heel_history, shifts=-1, dims=1)
        self.l_heel_history = torch.roll(self.l_heel_history, shifts=-1, dims=1)
        self.r_heel_history[:, -1] = right_force
        self.l_heel_history[:, -1] = left_force

        right_strike = (torch.amin(self.r_heel_history, dim=-1) > 0.1) & (self.last_heel_strike_foot != 1)
        left_strike = (torch.amin(self.l_heel_history, dim=-1) > 0.1) & (self.last_heel_strike_foot != 2)
        strike = right_strike | left_strike

        self.reward_muscle_activation_penalty_per_step = torch.where(
            strike,
            -self.control_dt * self.activation_square_sum,
            self.reward_muscle_activation_penalty_per_step,
        )
        self.reward_average_velocity_per_step = torch.where(
            strike,
            -self.control_dt * torch.abs(self.delta_velocity_sum),
            self.reward_average_velocity_per_step,
        )
        self.reward_footstep_delta_time = torch.where(
            strike,
            self.control_dt * self.footstep_delta_time,
            self.reward_footstep_delta_time,
        )

        self.last_heel_strike_foot = torch.where(right_strike, torch.ones_like(self.last_heel_strike_foot), self.last_heel_strike_foot)
        self.last_heel_strike_foot = torch.where(left_strike, torch.full_like(self.last_heel_strike_foot, 2), self.last_heel_strike_foot)
        if torch.any(strike):
            self._reset_reward_accumulators(strike)

        return {
            "muscle_activation_penalty_per_step": self.reward_muscle_activation_penalty_per_step,
            "average_velocity_per_step": self.reward_average_velocity_per_step,
            "footstep_delta_time": self.reward_footstep_delta_time,
        }

    def _reset_reward_state(self, mask: torch.Tensor) -> None:
        self._reset_reward_accumulators(mask)
        self.reward_muscle_activation_penalty_per_step[mask] = 0.0
        self.reward_average_velocity_per_step[mask] = 0.0
        self.reward_footstep_delta_time[mask] = 0.0

    def _reset_reward_accumulators(self, mask: torch.Tensor) -> None:
        self.footstep_delta_time[mask] = 0.0
        self.delta_velocity_sum[mask] = 0.0
        self.activation_square_sum[mask] = 0.0

    def _foot_force(self, foot_side: str) -> torch.Tensor:
        return self._sensor_values(f"{foot_side}_foot")[:, 0] + self._sensor_values(f"{foot_side}_toes")[:, 0]

    def _sensor_values(self, name: str) -> torch.Tensor:
        sensor_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise KeyError(f"Sensor not found in model: {name}")
        adr = int(self.cpu_model.sensor_adr[sensor_id])
        dim = int(self.cpu_model.sensor_dim[sensor_id])
        sensordata = self._tensor_field("sensordata")
        return sensordata[:, adr : adr + dim]
