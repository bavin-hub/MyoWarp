"""Configuration for independent, two-rate H1/CMU PPO."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Exp4Cfg:
    human_checkpoint: str = str(
        ROOT / "logs/rsl_rl/h1_2_tracking_standalone/2026-10-23_22-21-41/model_best.pt"
    )
    motion_path: str = str(ROOT / "assets/reference_clips/h1_2_straight_walk_120s.npz")
    reference_path: str = str(
        ROOT / "assets/reference_clips/h1_2_cmu_straight_walk_120s.npz"
    )
    num_envs: int = 1024
    device: str = "cuda:0"
    seed: int = 42
    episode_seconds: float = 30.0
    steps_per_env: int = 24  # H1 steps; CMU collects twice this count
    save_interval: int = 100
    human_learning_rate: float = 3e-5
    human_lr_min: float = 1e-6
    human_lr_max: float = 1e-4
    critic_warmup: int = 20
    human_std_scale: float = 0.5
    torque_scale: tuple = (60.0, 80.0)
    torque_limit: tuple = (100.0, 150.0)
    history_steps: int = 5
    sensor_contact_threshold: float = 25.0
    nconmax: int = 128
    njmax: int = 512
    policy_dt: float = 0.01
    physics_dt: float = 0.001
    human_weights: dict = field(
        default_factory=lambda: {
            "motion_global_root_pos": 2.0,
            "motion_global_root_ori": 1.0,
            "motion_body_pos": 1.0,
            "motion_body_ori": 1.0,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 1.0,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
            "self_collisions": -10.0,
        }
    )
    weights: dict = field(
        default_factory=lambda: {
            "foot_pos": 1.0,
            "foot_ori": 0.5,
            "foot_vel": 0.5,
            "contact": 0.25,
            "support": 0.25,
            "slip": -0.05,
            "impact": -0.01,
            "action_rate": -0.1,
            "effort": -0.001,
            "saturation": -0.01,
            "joint_limit": -10.0,
        }
    )

    def validate(self, evaluation=False):
        if self.device == "cpu":
            if not evaluation or self.num_envs != 1:
                raise ValueError(
                    "CPU is supported only for single-environment evaluation"
                )
        elif not self.device.startswith("cuda"):
            raise ValueError("Expected CUDA training or CPU evaluation")
        if self.num_envs < 1 or self.steps_per_env < 1 or self.history_steps != 5:
            raise ValueError("Positive counts and five sensor history frames required")
        if (
            not 0 < self.episode_seconds <= 120
            or self.save_interval < 1
            or self.critic_warmup < 0
        ):
            raise ValueError("Invalid episode or training settings")
        if not 0 < self.human_lr_min <= self.human_learning_rate <= self.human_lr_max:
            raise ValueError("Invalid H1 learning-rate bounds")
        if self.policy_dt != 0.01 or self.physics_dt != 0.001:
            raise ValueError("CMU policy must run at 100Hz and LLC/physics at 1kHz")
        if (
            len(self.torque_scale) != 2
            or len(self.torque_limit) != 2
            or min(*self.torque_scale, *self.torque_limit) <= 0
        ):
            raise ValueError("Two positive torque scales and limits required")
