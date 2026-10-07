"""Dual-policy full-observation H1-with-OSL motion tracking task."""

from myowarp.tasks.h1_with_osl_dual_full_velocity import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
)

from .config import H1WithOslDualFullTrackingCfg, make_rsl_rl_cfg
from .env import H1WithOslDualFullTrackingEnv
from .runner import H1DualFullTrackingRunner

__all__ = [
    "H1_ACTION_INDICES",
    "OSL_ACTION_INDICES",
    "OSL_ACTION_NAMES",
    "H1DualFullTrackingRunner",
    "H1WithOslDualFullTrackingCfg",
    "H1WithOslDualFullTrackingEnv",
    "make_rsl_rl_cfg",
]
