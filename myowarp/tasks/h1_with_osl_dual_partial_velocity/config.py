"""Configuration for dual-policy partial-observation velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass

from myowarp.tasks.h1_with_osl_dual_full_velocity.config import (
    H1WithOslDualFullVelocityCfg,
    make_rsl_rl_cfg as make_dual_full_rsl_rl_cfg,
)


@dataclass
class H1WithOslDualPartialVelocityCfg(H1WithOslDualFullVelocityCfg):
    """Task #4: full H1 observations and encoder-only OSL observations."""


def make_rsl_rl_cfg(
    *,
    max_iterations: int = 10001,
    save_interval: int = 100,
    run_name: str = "",
) -> dict:
    """Return Task #3's joint PPO configuration with a partial OSL input."""
    cfg = make_dual_full_rsl_rl_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
        run_name=run_name,
    )
    cfg["experiment_name"] = "h1_with_osl_dual_partial_velocity_standalone"
    cfg["obs_groups"]["osl_actor"] = ["osl_actor"]
    return cfg
