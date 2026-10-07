"""Dual-policy full-observation variant of standalone motion tracking."""

from __future__ import annotations

import torch

from myowarp.tasks.h1_with_osl_dual_full_velocity import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
)
from myowarp.tasks.h1_with_osl_tracking.env import H1WithOslTrackingEnv

from .config import H1WithOslDualFullTrackingCfg


class H1WithOslDualFullTrackingEnv(H1WithOslTrackingEnv):
    """Task #2 tracking with explicit H1 and OSL policy action partitions."""

    cfg: H1WithOslDualFullTrackingCfg

    def __init__(self, cfg: H1WithOslDualFullTrackingCfg) -> None:
        super().__init__(cfg)
        actual_osl_names = tuple(self.action_joint_names[index] for index in OSL_ACTION_INDICES)
        if actual_osl_names != OSL_ACTION_NAMES:
            raise RuntimeError(
                f"OSL action partition mismatch: expected {OSL_ACTION_NAMES}, got {actual_osl_names}"
            )
        self.h1_action_indices = torch.tensor(
            H1_ACTION_INDICES, device=self.device, dtype=torch.long
        )
        self.osl_action_indices = torch.tensor(
            OSL_ACTION_INDICES, device=self.device, dtype=torch.long
        )
