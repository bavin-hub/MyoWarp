"""Dual-policy partial-observation H1-with-OSL motion tracking task."""

from myowarp.tasks.h1_with_osl_dual_full_velocity import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
)

from .config import H1WithOslDualPartialTrackingCfg, make_rsl_rl_cfg
from .env import H1WithOslDualPartialTrackingEnv
from .runner import H1DualPartialTrackingRunner

__all__ = [
    "H1_ACTION_INDICES",
    "OSL_ACTION_INDICES",
    "OSL_ACTION_NAMES",
    "H1DualPartialTrackingRunner",
    "H1WithOslDualPartialTrackingCfg",
    "H1WithOslDualPartialTrackingEnv",
    "make_rsl_rl_cfg",
]
