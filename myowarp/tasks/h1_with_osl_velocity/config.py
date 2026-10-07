"""Configuration for the standalone H1-with-OSL velocity task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "assets/robots/h1_with_osl/h1_with_osl.xml"


@dataclass
class ObservationNoiseCfg:
    base_ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5


@dataclass
class DomainRandomizationCfg:
    enabled: bool = True
    foot_friction_range: tuple[float, float] = (0.3, 1.6)
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    torso_com_offset_x: tuple[float, float] = (-0.05, 0.05)
    torso_com_offset_y: tuple[float, float] = (-0.05, 0.05)
    torso_com_offset_z: tuple[float, float] = (-0.05, 0.05)
    push_interval_s: tuple[float, float] = (5.0, 6.0)
    push_lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    push_lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    push_lin_vel_z: tuple[float, float] = (-0.4, 0.4)
    push_ang_vel_x: tuple[float, float] = (-0.52, 0.52)
    push_ang_vel_y: tuple[float, float] = (-0.52, 0.52)
    push_ang_vel_z: tuple[float, float] = (-0.78, 0.78)


@dataclass
class CommandCfg:
    resampling_time_s: tuple[float, float] = (3.0, 8.0)
    standing_probability: float = 0.05
    heading_probability: float = 1.0
    heading_stiffness: float = 0.5
    initial_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    initial_lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    lin_vel_x: tuple[float, float] = (-1.0, 2.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    heading: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    full_range_after_steps: int = 5000 * 24


@dataclass
class H1WithOslVelocityCfg:
    """Task parameters matching ``MyoWarp-H1-With-OSL-Flat`` in MJLab 1.2."""

    model_path: str = str(DEFAULT_MODEL_PATH)
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42
    physics_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 20.0
    nconmax: int = 64
    njmax: int = 300
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    ccd_iterations: int = 50
    enable_observation_noise: bool = True
    observation_noise: ObservationNoiseCfg = field(default_factory=ObservationNoiseCfg)
    domain_randomization: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)
    command: CommandCfg = field(default_factory=CommandCfg)

    @property
    def step_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def max_episode_length(self) -> int:
        return round(self.episode_length_s / self.step_dt)

    def to_dict(self) -> dict:
        return asdict(self)


def make_rsl_rl_cfg(
    *,
    max_iterations: int = 10001,
    save_interval: int = 100,
    run_name: str = "",
) -> dict:
    """Return a native RSL-RL 5.x configuration without MJLab config classes."""
    return {
        "seed": 42,
        "num_steps_per_env": 24,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
        "experiment_name": "h1_with_osl_velocity_standalone",
        "run_name": run_name,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "optimizer": "adam",
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "share_cnn_encoders": False,
        },
        "multi_gpu": None,
        "check_for_nan": True,
    }
