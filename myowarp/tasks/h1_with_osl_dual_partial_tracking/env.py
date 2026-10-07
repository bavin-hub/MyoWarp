"""Dual-policy motion tracking with an encoder-only OSL actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from myowarp.tasks.h1_with_osl_dual_full_tracking.env import (
    H1WithOslDualFullTrackingEnv,
)
from myowarp.tasks.h1_with_osl_dual_full_velocity import OSL_ACTION_INDICES

from .config import H1WithOslDualPartialTrackingCfg


ACTOR_JOINT_POSITION_OFFSET = 75
ACTOR_JOINT_VELOCITY_OFFSET = 105


class H1WithOslDualPartialTrackingEnv(H1WithOslDualFullTrackingEnv):
    """Expose only OSL knee/ankle positions and velocities to the OSL actor."""

    cfg: H1WithOslDualPartialTrackingCfg

    def _compute_observations(self) -> TensorDict:
        observations = super()._compute_observations()
        actor = observations["actor"]
        osl_start, osl_stop = OSL_ACTION_INDICES[0], OSL_ACTION_INDICES[-1] + 1
        joint_indices = self.action_joint_indices[osl_start:osl_stop]
        osl_joint_positions = actor.index_select(
            1, joint_indices + ACTOR_JOINT_POSITION_OFFSET
        )
        osl_joint_velocities = actor.index_select(
            1, joint_indices + ACTOR_JOINT_VELOCITY_OFFSET
        )
        observations["osl_actor"] = torch.cat(
            (osl_joint_positions, osl_joint_velocities), dim=1
        )
        return observations
