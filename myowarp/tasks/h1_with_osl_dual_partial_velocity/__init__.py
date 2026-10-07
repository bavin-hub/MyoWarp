"""Dual-policy partial-observation H1-with-OSL velocity tracking task."""

from myowarp.tasks.h1_with_osl_dual_full_velocity import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
)

from .config import H1WithOslDualPartialVelocityCfg, make_rsl_rl_cfg
from .env import H1WithOslDualPartialVelocityEnv
from .runner import H1DualPartialVelocityRunner

__all__ = [
    "H1_ACTION_INDICES",
    "OSL_ACTION_INDICES",
    "OSL_ACTION_NAMES",
    "H1DualPartialVelocityRunner",
    "H1WithOslDualPartialVelocityCfg",
    "H1WithOslDualPartialVelocityEnv",
    "make_rsl_rl_cfg",
]
