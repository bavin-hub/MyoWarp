from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import torch

from myowarp.config import TrainConfig
from myowarp.io import load_reference_data
from myowarp.utils import build_model_index


class MyoAssistLegCpuEnv:
    """Single-env MuJoCo CPU evaluator with the same obs/action surface as the Warp env."""

    def __init__(self, config: TrainConfig, root_dir: str | Path):
        self.config = config
        self.root_dir = Path(root_dir)
        self.env_params = config.env_params
        self.num_envs = 1
        self.device = "cpu"
        self.model_path = self.root_dir / self.env_params.model_path
        self.control_dt = 1.0 / float(self.env_params.control_framerate)
        self.frame_skip = int(self.env_params.physics_sim_framerate // self.env_params.control_framerate)

        self.cpu_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.cpu_model.opt.timestep = 1.0 / float(self.env_params.physics_sim_framerate)
        self.data = mujoco.MjData(self.cpu_model)
        ctrlrange = torch.as_tensor(self.cpu_model.actuator_ctrlrange, dtype=torch.float32)
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
            device="cpu",
        )
        self.reward_scalar_weights = self._make_reward_scalar_weights()
        self.prev_muscle_activation = torch.zeros(len(self.model_index.muscle_actuator_ids), dtype=torch.float32)
        self.has_prev_muscle_activation = False
        self.r_heel_history = torch.zeros(3, dtype=torch.float32)
        self.l_heel_history = torch.zeros(3, dtype=torch.float32)
        self.last_heel_strike_foot = 0
        self.reward_muscle_activation_penalty_per_step = 0.0
        self.reward_average_velocity_per_step = 0.0
        self.reward_footstep_delta_time = 0.0
        self.footstep_delta_time = 0.0
        self.delta_velocity_sum = 0.0
        self.activation_square_sum = 0.0
        self.step_count = 0
        self.target_velocity = float(self.env_params.min_target_velocity)
        self.reference_index = 0
        self.velocity_mode = 0
        self.starting_phase = 0.0
        self.target_velocity_period = 1.0
        self.prev_step_changed_time = 0.0
        torch.manual_seed(int(self.env_params.seed))

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
        self._reset_env()
        return self.get_obs()

    def step(self, actions: torch.Tensor):
        self._modulate_target_velocity()
        action = torch.clamp(actions.detach().cpu().reshape(-1), -1.0, 1.0)
        self.data.ctrl[:] = self._normalized_action_to_ctrl(action).numpy()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.cpu_model, self.data)

        self.step_count += 1
        next_reference_index = self.reference_index + 1
        out_of_reference = next_reference_index >= self.reference_data.length
        self.reference_index = min(next_reference_index, self.reference_data.length - 1)

        obs = self.get_obs()
        reward, reward_info = self.get_reward(obs)
        height_done, trajectory_done = self._termination_causes()
        terminated = height_done or trajectory_done
        truncated = self.step_count >= int(self.env_params.custom_max_episode_steps)
        done = terminated or truncated or out_of_reference
        if out_of_reference:
            reward = torch.zeros_like(reward)
        info: dict[str, Any] = {
            "reward": reward_info,
            "terminated": torch.tensor([terminated], dtype=torch.bool),
            "height_done": torch.tensor([height_done], dtype=torch.bool),
            "trajectory_done": torch.tensor([trajectory_done], dtype=torch.bool),
            "truncated": torch.tensor([truncated or out_of_reference], dtype=torch.bool),
            "out_of_reference": torch.tensor([out_of_reference], dtype=torch.bool),
        }
        if done:
            self._reset_env()
            obs = self.get_obs()
        return obs, reward, torch.tensor([done], dtype=torch.bool), info

    def get_obs(self) -> torch.Tensor:
        qpos = torch.tensor([self.data.qpos[adr] for adr in self.model_index.qpos.values()], dtype=torch.float32)
        qvel = torch.tensor([self.data.qvel[adr] for adr in self.model_index.qvel.values()], dtype=torch.float32)
        act = self._gather_muscle_activation()
        sensor = self._gather_sensors()
        target = torch.tensor([self.target_velocity], dtype=torch.float32)
        return torch.cat([qpos, qvel, act, sensor, target], dim=-1).unsqueeze(0)

    def get_reward(self, obs: torch.Tensor):
        components = self._reward_components()
        reward = torch.zeros(1, dtype=torch.float32)
        for key, value in components.items():
            reward += self.reward_scalar_weights.get(key, 0.0) * value
        return reward, components

    def _reward_components(self) -> dict[str, torch.Tensor]:
        qpos_reward = self._qpos_imitation_reward()
        qvel_reward = self._qvel_imitation_reward()
        end_effector_reward = torch.tensor([self.control_dt], dtype=torch.float32)
        forward_reward = self._forward_reward()
        act = self._gather_muscle_activation()
        activation_penalty = torch.tensor([-self.control_dt * torch.mean(act).item()], dtype=torch.float32)
        activation_diff_penalty = self._muscle_activation_diff_penalty(act)
        foot_force_penalty = self._foot_force_penalty()
        joint_constraint_force_penalty = self._joint_constraint_force_penalty()
        per_step_rewards = self._reward_per_step(act)
        self.prev_muscle_activation = act.detach()
        self.has_prev_muscle_activation = True
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

    def _reset_env(self) -> None:
        self.step_count = 0
        self.reference_index = self._sample_reference_index()
        reset_qpos, reset_qvel = self._reference_state(self.reference_index, target_velocity=self.target_velocity)
        self._reset_target_velocity()
        reset_qvel[self._joint_qvel_adr("pelvis_tx")] = self.target_velocity
        self.data.qpos[:] = reset_qpos.numpy()
        self.data.qvel[:] = reset_qvel.numpy()
        self.data.ctrl[:] = 0.0
        if self.cpu_model.na > 0:
            self.data.act[:] = 0.0
        self.data.time = 0.0
        self.prev_muscle_activation[:] = 0.0
        self.has_prev_muscle_activation = False
        self.r_heel_history[:] = 0.0
        self.l_heel_history[:] = 0.0
        self.last_heel_strike_foot = 0
        self._reset_reward_state()
        mujoco.mj_forward(self.cpu_model, self.data)

    def _normalized_action_to_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        return self.ctrl_mid + action * self.ctrl_half_range

    def _reset_target_velocity(self) -> None:
        min_velocity = float(self.env_params.min_target_velocity)
        max_velocity = float(self.env_params.max_target_velocity)
        min_period = float(self.env_params.min_target_velocity_period)
        max_period = float(self.env_params.max_target_velocity_period)
        self.velocity_mode = int(torch.randint(0, 3, ()).item())
        self.starting_phase = float((2.0 * math.pi * torch.rand(())).item())
        self.target_velocity_period = float((min_period + (max_period - min_period) * torch.rand(())).item())
        self.prev_step_changed_time = 0.0
        self.target_velocity = float((min_velocity + (max_velocity - min_velocity) * torch.rand(())).item())
        if self.velocity_mode == 1:
            self.target_velocity = self._sinusoidal_target_velocity()

    def _modulate_target_velocity(self) -> None:
        if self.velocity_mode == 1:
            self.target_velocity = self._sinusoidal_target_velocity()
        elif self.velocity_mode == 2:
            time = self.step_count * self.control_dt
            if (time - self.prev_step_changed_time) > self.target_velocity_period:
                min_velocity = float(self.env_params.min_target_velocity)
                max_velocity = float(self.env_params.max_target_velocity)
                self.target_velocity = float((min_velocity + (max_velocity - min_velocity) * torch.rand(())).item())
                self.prev_step_changed_time = time

    def _sinusoidal_target_velocity(self) -> float:
        min_velocity = float(self.env_params.min_target_velocity)
        max_velocity = float(self.env_params.max_target_velocity)
        time = self.step_count * self.control_dt
        phase = self.starting_phase + 2.0 * math.pi * time / self.target_velocity_period
        return min_velocity + (max_velocity - min_velocity) * (math.sin(phase) + 1.0) / 2.0

    def _sample_reference_index(self) -> int:
        if self.env_params.flag_random_ref_index:
            high = max(1, int(self.reference_data.length * 0.8))
            return int(torch.randint(high, ()).item())
        return 0

    def _reference_state(self, ref_index: int, target_velocity: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if target_velocity is None:
            target_velocity = self.target_velocity
        qpos = torch.as_tensor(self.cpu_model.key_qpos[0], dtype=torch.float32).clone()
        qvel = torch.as_tensor(self.cpu_model.key_qvel[0], dtype=torch.float32).clone()
        ref_pelvis_vx = self.reference_data.series_data["dq_pelvis_tx"][ref_index]
        speed_ratio = target_velocity / max(float(ref_pelvis_vx.item()), 1e-5)
        for key in self.env_params.reference_data_keys:
            q_key = f"q_{key}"
            dq_key = f"dq_{key}"
            if q_key in self.reference_data.series_data:
                qpos[self._joint_qpos_adr(key)] = self.reference_data.series_data[q_key][ref_index]
            if dq_key in self.reference_data.series_data:
                qvel[self._joint_qvel_adr(key)] = self.reference_data.series_data[dq_key][ref_index] * speed_ratio
        qpos[self._joint_qpos_adr("pelvis_tx")] = 0.0
        qvel[self._joint_qvel_adr("pelvis_tx")] = target_velocity
        return qpos, qvel

    def _make_reward_scalar_weights(self) -> dict[str, float]:
        weights = {}
        for key, value in self.env_params.reward_keys_and_weights.items():
            if isinstance(value, dict):
                weights[key] = float(sum(value.values()))
            else:
                weights[key] = float(value)
        return weights

    def _gather_muscle_activation(self) -> torch.Tensor:
        if self.model_index.na > 0:
            return torch.as_tensor(self.data.act[: len(self.model_index.muscle_actuator_ids)], dtype=torch.float32)
        return torch.as_tensor(self.data.ctrl[self.model_index.muscle_actuator_ids], dtype=torch.float32)

    def _gather_sensors(self) -> torch.Tensor:
        parts = []
        weight = float(self.cpu_model.body_mass.sum()) * 9.81
        for name, sensor in self.model_index.sensors.items():
            values = torch.as_tensor(self.data.sensordata[sensor.adr : sensor.adr + sensor.dim], dtype=torch.float32)
            if "foot" in name or "toes" in name:
                values = values / weight
            parts.append(values)
        if not parts:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(parts, dim=-1)

    def _joint_qpos(self, name: str) -> float:
        return float(self.data.qpos[self.model_index.qpos.get(name, self._joint_qpos_adr(name))])

    def _joint_qvel(self, name: str) -> float:
        return float(self.data.qvel[self.model_index.qvel.get(name, self._joint_qvel_adr(name))])

    def _joint_qpos_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.cpu_model.jnt_qposadr[joint_id])

    def _joint_qvel_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.cpu_model.jnt_dofadr[joint_id])

    def _termination_causes(self) -> tuple[bool, bool]:
        out_of_height = self._joint_qpos("pelvis_ty") < float(self.env_params.safe_height)
        out_of_trajectory = any(
            abs(diff) > float(self.env_params.out_of_trajectory_threshold)
            for diff in self._qpos_diff_list()
        )
        return out_of_height, out_of_trajectory

    def _qpos_imitation_reward(self) -> torch.Tensor:
        total = 0.0
        weights = self.env_params.reward_keys_and_weights.get("qpos_imitation_rewards", {})
        for key, weight in weights.items():
            current = self._joint_qpos(key)
            ref = float(self.reference_data.series_data[f"q_{key}"][self.reference_index].item())
            total += float(weight) * self.control_dt * math.exp(-8.0 * (current - ref) ** 2)
        return torch.tensor([total], dtype=torch.float32)

    def _qvel_imitation_reward(self) -> torch.Tensor:
        total = 0.0
        weights = self.env_params.reward_keys_and_weights.get("qvel_imitation_rewards", {})
        ref_pelvis = float(self.reference_data.series_data["dq_pelvis_tx"][self.reference_index].item())
        speed_ratio = self.target_velocity / max(ref_pelvis, 1e-5)
        for key, weight in weights.items():
            current = self._joint_qvel(key)
            ref = float(self.reference_data.series_data[f"dq_{key}"][self.reference_index].item()) * speed_ratio
            total += float(weight) * self.control_dt * math.exp(-8.0 * (current - ref) ** 2)
        return torch.tensor([total], dtype=torch.float32)

    def _forward_reward(self) -> torch.Tensor:
        pelvis_vx = self._joint_qvel("pelvis_tx")
        return torch.tensor([self.control_dt * math.exp(-5.0 * (pelvis_vx - self.target_velocity) ** 2)], dtype=torch.float32)

    def _qpos_diff_list(self) -> list[float]:
        diffs = []
        weights = self.env_params.reward_keys_and_weights.get("qpos_imitation_rewards", {})
        for key in weights.keys():
            current = self._joint_qpos(key)
            ref = float(self.reference_data.series_data[f"q_{key}"][self.reference_index].item())
            diffs.append(current - ref)
        return diffs

    def _muscle_activation_diff_penalty(self, act: torch.Tensor) -> torch.Tensor:
        if not self.has_prev_muscle_activation:
            return torch.zeros(1, dtype=torch.float32)
        value = self.control_dt * torch.mean(torch.exp(-4.0 * torch.square(self.prev_muscle_activation - act)))
        return value.reshape(1)

    def _foot_force_penalty(self) -> torch.Tensor:
        model_weight = float(self.cpu_model.body_mass.sum()) * 9.81
        normalized_foot_force_sum = (abs(self._foot_force("r")) + abs(self._foot_force("l"))) / model_weight
        return torch.tensor([-self.control_dt * max(normalized_foot_force_sum - 1.2, 0.0)], dtype=torch.float32)

    def _joint_constraint_force_penalty(self) -> torch.Tensor:
        if not self.env_params.joint_limit_sensor_keys:
            return torch.zeros(1, dtype=torch.float32)
        max_constraint_force = max(
            float(torch.abs(self._sensor_values(sensor_name)).amax().item())
            for sensor_name in self.env_params.joint_limit_sensor_keys
        )
        return torch.tensor([-self.control_dt * max_constraint_force / (float(self.cpu_model.body_mass.sum()) * 9.81)], dtype=torch.float32)

    def _reward_per_step(self, act: torch.Tensor) -> dict[str, torch.Tensor]:
        self.footstep_delta_time += self.control_dt
        self.delta_velocity_sum += self.control_dt * (self._joint_qvel("pelvis_tx") - self.target_velocity)
        self.activation_square_sum += self.control_dt * float(torch.sum(torch.square(act)).item())

        right_force = self._foot_force("r")
        left_force = self._foot_force("l")
        self.r_heel_history = torch.roll(self.r_heel_history, shifts=-1)
        self.l_heel_history = torch.roll(self.l_heel_history, shifts=-1)
        self.r_heel_history[-1] = right_force
        self.l_heel_history[-1] = left_force

        right_strike = bool(torch.amin(self.r_heel_history).item() > 0.1 and self.last_heel_strike_foot != 1)
        left_strike = bool(torch.amin(self.l_heel_history).item() > 0.1 and self.last_heel_strike_foot != 2)
        strike = right_strike or left_strike

        if strike:
            self.reward_muscle_activation_penalty_per_step = -self.control_dt * self.activation_square_sum
            self.reward_average_velocity_per_step = -self.control_dt * abs(self.delta_velocity_sum)
            self.reward_footstep_delta_time = self.control_dt * self.footstep_delta_time
            self.last_heel_strike_foot = 1 if right_strike else 2
            self._reset_reward_accumulators()

        return {
            "muscle_activation_penalty_per_step": torch.tensor([self.reward_muscle_activation_penalty_per_step], dtype=torch.float32),
            "average_velocity_per_step": torch.tensor([self.reward_average_velocity_per_step], dtype=torch.float32),
            "footstep_delta_time": torch.tensor([self.reward_footstep_delta_time], dtype=torch.float32),
        }

    def _reset_reward_state(self) -> None:
        self._reset_reward_accumulators()
        self.reward_muscle_activation_penalty_per_step = 0.0
        self.reward_average_velocity_per_step = 0.0
        self.reward_footstep_delta_time = 0.0

    def _reset_reward_accumulators(self) -> None:
        self.footstep_delta_time = 0.0
        self.delta_velocity_sum = 0.0
        self.activation_square_sum = 0.0

    def _foot_force(self, foot_side: str) -> float:
        return float((self._sensor_values(f"{foot_side}_foot")[0] + self._sensor_values(f"{foot_side}_toes")[0]).item())

    def _sensor_values(self, name: str) -> torch.Tensor:
        sensor_id = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise KeyError(f"Sensor not found in model: {name}")
        adr = int(self.cpu_model.sensor_adr[sensor_id])
        dim = int(self.cpu_model.sensor_dim[sensor_id])
        return torch.as_tensor(self.data.sensordata[adr : adr + dim], dtype=torch.float32)
