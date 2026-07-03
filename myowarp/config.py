from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class EnvParams:
    env_id: str = "myoAssistLegImitationExo-v0"
    num_envs: int = 1
    seed: int = 1234
    safe_height: float = 0.7
    out_of_trajectory_threshold: float = 0.2
    flag_random_ref_index: bool = True
    control_framerate: int = 30
    physics_sim_framerate: int = 1200
    min_target_velocity: float = 1.25
    max_target_velocity: float = 1.25
    min_target_velocity_period: float = 2.0
    max_target_velocity_period: float = 10.0
    enable_lumbar_joint: bool = False
    lumbar_joint_fixed_angle: float = -0.13
    lumbar_joint_damping_value: float = 0.05
    observation_joint_pos_keys: list[str] = field(default_factory=list)
    observation_joint_vel_keys: list[str] = field(default_factory=list)
    observation_sensor_keys: list[str] = field(default_factory=list)
    joint_limit_sensor_keys: list[str] = field(default_factory=list)
    terrain_type: str = "flat"
    terrain_params: str = ""
    reward_keys_and_weights: dict[str, Any] = field(default_factory=dict)
    custom_max_episode_steps: int = 1000
    model_path: str = "models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml"
    reference_data_path: str = "reference_data/short_reference_gait.npz"
    reference_data_keys: list[str] = field(default_factory=list)
    prev_trained_policy_path: str | None = None


@dataclass
class WarpParams:
    num_envs: int = 4096
    device: str = "cuda"
    backend: str = "mujoco_warp"


@dataclass
class TrainConfig:
    total_timesteps: float = 3e7
    env_params: EnvParams = field(default_factory=EnvParams)
    policy_params: dict[str, Any] = field(default_factory=dict)
    ppo_params: dict[str, Any] = field(default_factory=dict)
    logger_params: dict[str, Any] = field(default_factory=dict)
    evaluate_param_list: list[dict[str, Any]] = field(default_factory=list)
    warp_params: WarpParams = field(default_factory=WarpParams)


def _dataclass_from_dict(cls: type, data: dict[str, Any]):
    field_names = set(cls.__dataclass_fields__.keys())
    kwargs = {name: data[name] for name in field_names if name in data}
    return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    """Load a MyoAssist JSON config into a light Warp-friendly dataclass."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    env_params = _dataclass_from_dict(EnvParams, raw.get("env_params", {}))
    warp_raw = raw.get("warp_params", {})
    warp_params = _dataclass_from_dict(WarpParams, warp_raw)
    if "num_envs" in raw.get("env_params", {}):
        warp_params.num_envs = int(raw["env_params"]["num_envs"])

    return TrainConfig(
        total_timesteps=raw.get("total_timesteps", 3e7),
        env_params=env_params,
        policy_params=raw.get("policy_params", {}),
        ppo_params=raw.get("ppo_params", {}),
        logger_params=raw.get("logger_params", {}),
        evaluate_param_list=raw.get("evaluate_param_list", []),
        warp_params=warp_params,
    )
