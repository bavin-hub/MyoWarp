"""Dual-policy full-observation variant of the standalone velocity task."""

from __future__ import annotations

import torch

from myowarp.tasks.h1_with_osl_velocity.env import H1WithOslVelocityEnv

from .config import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
    H1WithOslDualFullVelocityCfg,
)


class H1WithOslDualFullVelocityEnv(H1WithOslVelocityEnv):
    """Task #1 dynamics and rewards with explicit H1/OSL action partitions."""

    cfg: H1WithOslDualFullVelocityCfg

    def __init__(self, cfg: H1WithOslDualFullVelocityCfg) -> None:
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
