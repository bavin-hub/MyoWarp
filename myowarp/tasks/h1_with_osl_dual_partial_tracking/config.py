"""Configuration for dual-policy partial-observation motion tracking."""

from __future__ import annotations

from dataclasses import dataclass

from myowarp.tasks.h1_with_osl_dual_full_tracking.config import (
    H1WithOslDualFullTrackingCfg,
    make_rsl_rl_cfg as make_dual_full_tracking_rsl_rl_cfg,
)


@dataclass
class H1WithOslDualPartialTrackingCfg(H1WithOslDualFullTrackingCfg):
    """Task #6: full H1 observations and encoder-only OSL observations."""


def make_rsl_rl_cfg(
    *,
    max_iterations: int = 30_000,
    save_interval: int = 500,
    run_name: str = "",
) -> dict:
    """Return Task #5's joint PPO configuration with a partial OSL input."""
    cfg = make_dual_full_tracking_rsl_rl_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
        run_name=run_name,
    )
    cfg["experiment_name"] = "h1_with_osl_dual_partial_tracking_standalone"
    cfg["obs_groups"]["osl_actor"] = ["osl_actor"]
    return cfg
