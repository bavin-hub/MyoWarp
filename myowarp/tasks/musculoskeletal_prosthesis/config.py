"""Independent configuration; H1 task implementations are left unchanged."""

from dataclasses import dataclass, field
from pathlib import Path

from .ppo_config import make_rsl_rl_cfg as h1_ppo_config

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ProsthesisCfg:
    motion_path: str = str(
        ROOT / "assets/musclemimic/motions/loops/walking_medium10_poses_30s.npz"
    )
    reference_path: str = str(
        ROOT / "assets/musclemimic/references/prosthesis_walk.npz"
    )
    checkpoint_path: str = str(ROOT / "vyvas_pi_h/model_1499.pt")
    model_path: str = str(
        ROOT / "assets/robots/myofullbody_cmu_leg/myofullbody_cmu_leg.xml"
    )
    device: str = "cuda:0"
    nconmax: int = 128
    njmax: int = 1024
    seed: int = 42
    policy_dt: float = 0.01
    physics_dt: float = 0.001
    history_steps: int = 5
    right_leg_obs: str = "fooled"
    train_muscle: bool = False
    muscle_learning_rate: float = 1e-5
    episode_seconds: float = 10.0
    torque_scale: tuple[float, float] = (60.0, 80.0)
    # Joint torque command limits, not the motor actuator force limits.
    torque_limit: tuple[float, float] = (100.0, 150.0)
    sensor_contact_threshold: float = 25.0
    max_reset_sole_error: float = 0.20
    max_initial_penetration: float = 0.02
    weights: dict = field(
        default_factory=lambda: {
            "root_pos": 0.5,
            "root_ori": 0.5,
            "root_vel": 0.5,
            "body_pos": 1.0,
            "body_ori": 1.0,
            "body_vel": 1.0,
            "body_ang_vel": 0.5,
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
            "unsafe_contact": -10.0,
        }
    )

    def select_anatomy(self, anatomy):
        if anatomy not in ("constrained", "removed"):
            raise ValueError("anatomy must be constrained or removed")
        folder = (
            "myofullbody_cmu_leg"
            if anatomy == "constrained"
            else "myofullbody_cmu_leg_no_lower_leg"
        )
        self.model_path = str(ROOT / "assets/robots" / folder / (folder + ".xml"))
        cache = (
            "prosthesis_walk.npz"
            if anatomy == "constrained"
            else "prosthesis_walk_no_lower_leg.npz"
        )
        self.reference_path = str(ROOT / "assets/musclemimic/references" / cache)

    def validate(self):
        if self.muscle_learning_rate <= 0:
            raise ValueError("muscle_learning_rate must be positive")
        if self.right_leg_obs not in ("fooled", "virtual"):
            raise ValueError("right_leg_obs must be fooled or virtual")
        if self.policy_dt != 0.01 or self.physics_dt != 0.001:
            raise ValueError(
                "This contract is 100 Hz policies and 1 kHz low-level physics"
            )
        if self.nconmax < 1 or self.njmax < 1:
            raise ValueError("GPU contact/constraint capacities must be positive")
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")


def make_agent_cfg(save_interval=100):
    cfg = h1_ppo_config(save_interval=save_interval)
    cfg["experiment_name"] = "musculoskeletal_prosthesis"
    cfg["actor"]["hidden_dims"] = [128, 128]
    cfg["actor"]["distribution_cfg"]["init_std"] = 0.2
    return cfg
