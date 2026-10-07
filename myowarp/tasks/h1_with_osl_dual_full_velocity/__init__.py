"""Dual-policy full-observation H1-with-OSL velocity tracking task."""

from .config import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
    H1WithOslDualFullVelocityCfg,
    make_rsl_rl_cfg,
)
from .env import H1WithOslDualFullVelocityEnv
from .runner import H1DualFullVelocityRunner

__all__ = [
    "H1_ACTION_INDICES",
    "OSL_ACTION_INDICES",
    "OSL_ACTION_NAMES",
    "H1DualFullVelocityRunner",
    "H1WithOslDualFullVelocityCfg",
    "H1WithOslDualFullVelocityEnv",
    "make_rsl_rl_cfg",
]
