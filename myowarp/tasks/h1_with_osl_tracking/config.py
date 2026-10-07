"""Configuration for standalone H1-with-OSL motion tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "assets/robots/h1_with_osl/h1_with_osl.xml"
DEFAULT_MOTION_PATH = (
    REPO_ROOT / "assets/reference_clips/short_reference_gait_h1_with_osl.npz"
)


@dataclass
class TrackingObservationNoiseCfg:
    anchor_position: float = 0.25
    anchor_orientation: float = 0.05
    base_linear_velocity: float = 0.5
    base_angular_velocity: float = 0.2
    joint_position: float = 0.01
    joint_velocity: float = 0.5


@dataclass
class TrackingDomainRandomizationCfg:
    enabled: bool = True
    enable_pushes: bool = True
    foot_friction_range: tuple[float, float] = (0.3, 1.2)
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    torso_com_offset_x: tuple[float, float] = (-0.025, 0.025)
    torso_com_offset_y: tuple[float, float] = (-0.05, 0.05)
    torso_com_offset_z: tuple[float, float] = (-0.05, 0.05)
    push_interval_s: tuple[float, float] = (1.0, 3.0)
    push_lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    push_lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    push_lin_vel_z: tuple[float, float] = (-0.2, 0.2)
    push_ang_vel_x: tuple[float, float] = (-0.52, 0.52)
    push_ang_vel_y: tuple[float, float] = (-0.52, 0.52)
    push_ang_vel_z: tuple[float, float] = (-0.78, 0.78)


@dataclass
class MotionCommandCfg:
    sampling_mode: str = "adaptive"
    pose_x: tuple[float, float] = (-0.05, 0.05)
    pose_y: tuple[float, float] = (-0.05, 0.05)
    pose_z: tuple[float, float] = (-0.01, 0.01)
    pose_roll: tuple[float, float] = (-0.1, 0.1)
    pose_pitch: tuple[float, float] = (-0.1, 0.1)
    pose_yaw: tuple[float, float] = (-0.2, 0.2)
    velocity_x: tuple[float, float] = (-0.5, 0.5)
    velocity_y: tuple[float, float] = (-0.5, 0.5)
    velocity_z: tuple[float, float] = (-0.2, 0.2)
    velocity_roll: tuple[float, float] = (-0.52, 0.52)
    velocity_pitch: tuple[float, float] = (-0.52, 0.52)
    velocity_yaw: tuple[float, float] = (-0.78, 0.78)
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001


@dataclass
class H1WithOslTrackingCfg:
    """Task parameters matching the H1-with-OSL MJLab tracking task."""

    model_path: str = str(DEFAULT_MODEL_PATH)
    motion_path: str = str(DEFAULT_MOTION_PATH)
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42
    physics_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 10.0
    nconmax: int = 64
    njmax: int = 300
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    ccd_iterations: int = 50
    enable_observation_noise: bool = True
    observation_noise: TrackingObservationNoiseCfg = field(
        default_factory=TrackingObservationNoiseCfg
    )
    domain_randomization: TrackingDomainRandomizationCfg = field(
        default_factory=TrackingDomainRandomizationCfg
    )
    motion: MotionCommandCfg = field(default_factory=MotionCommandCfg)

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
    max_iterations: int = 30_000,
    save_interval: int = 500,
    run_name: str = "",
) -> dict:
    """Return the native RSL-RL configuration used for motion tracking."""
    return {
        "seed": 42,
        "num_steps_per_env": 24,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
        "experiment_name": "h1_with_osl_tracking_standalone",
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
            "entropy_coef": 0.005,
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
