"""Configuration for dual-policy full-observation motion tracking."""

from __future__ import annotations

from dataclasses import dataclass

from myowarp.tasks.h1_with_osl_dual_full_velocity import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
)
from myowarp.tasks.h1_with_osl_tracking.config import (
    H1WithOslTrackingCfg,
    make_rsl_rl_cfg as make_single_actor_tracking_rsl_rl_cfg,
)


@dataclass
class H1WithOslDualFullTrackingCfg(H1WithOslTrackingCfg):
    """Task #5: two full-observation actors with motion tracking."""


def make_rsl_rl_cfg(
    *,
    max_iterations: int = 30_000,
    save_interval: int = 500,
    run_name: str = "",
) -> dict:
    """Return the joint-PPO configuration for full-observation tracking."""
    cfg = make_single_actor_tracking_rsl_rl_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
        run_name=run_name,
    )
    cfg["experiment_name"] = "h1_with_osl_dual_full_tracking_standalone"
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
