"""Configuration for dual-policy full-observation velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass

from myowarp.tasks.h1_with_osl_velocity.config import (
    H1WithOslVelocityCfg,
    make_rsl_rl_cfg as make_single_actor_rsl_rl_cfg,
)


OSL_ACTION_NAMES = ("osl_knee_angle_r", "osl_ankle_angle_r")
OSL_ACTION_INDICES = (9, 10)
H1_ACTION_INDICES = tuple(index for index in range(26) if index not in OSL_ACTION_INDICES)


@dataclass
class H1WithOslDualFullVelocityCfg(H1WithOslVelocityCfg):
    """Task #3: two actors, full observations, and MJLab velocity rewards."""


def make_rsl_rl_cfg(
    *,
    max_iterations: int = 10001,
    save_interval: int = 100,
    run_name: str = "",
) -> dict:
    """Return the joint-PPO configuration for the H1 and OSL actors."""
    cfg = make_single_actor_rsl_rl_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
        run_name=run_name,
    )
    cfg["experiment_name"] = "h1_with_osl_dual_full_velocity_standalone"
    cfg["obs_groups"] = {
        "actor": ["actor"],
        "h1_actor": ["actor"],
        "osl_actor": ["actor"],
        "critic": ["critic"],
    }
    cfg["actor"] = {
        "class_name": "rsl_rl.models:DualMLPModel",
        "primary_action_indices": list(H1_ACTION_INDICES),
        "secondary_action_indices": list(OSL_ACTION_INDICES),
        "primary_obs_set": "h1_actor",
        "secondary_obs_set": "osl_actor",
        "primary_hidden_dims": [512, 256, 128],
        "secondary_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    return cfg
