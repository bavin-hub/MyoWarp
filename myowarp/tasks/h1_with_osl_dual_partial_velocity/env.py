"""Dual-policy velocity task with an encoder-only OSL actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from myowarp.tasks.h1_with_osl_dual_full_velocity import OSL_ACTION_INDICES
from myowarp.tasks.h1_with_osl_dual_full_velocity.env import (
    H1WithOslDualFullVelocityEnv,
)

from .config import H1WithOslDualPartialVelocityCfg


ACTOR_JOINT_POSITION_OFFSET = 11
ACTOR_JOINT_VELOCITY_OFFSET = 37


class H1WithOslDualPartialVelocityEnv(H1WithOslDualFullVelocityEnv):
    """Expose only OSL knee/ankle positions and velocities to the OSL actor."""

    cfg: H1WithOslDualPartialVelocityCfg

    def _compute_observations(self) -> TensorDict:
        observations = super()._compute_observations()
        actor = observations["actor"]
        osl_start, osl_stop = OSL_ACTION_INDICES[0], OSL_ACTION_INDICES[-1] + 1
        osl_joint_positions = actor[
            :, ACTOR_JOINT_POSITION_OFFSET + osl_start : ACTOR_JOINT_POSITION_OFFSET + osl_stop
        ]
        osl_joint_velocities = actor[
            :, ACTOR_JOINT_VELOCITY_OFFSET + osl_start : ACTOR_JOINT_VELOCITY_OFFSET + osl_stop
        ]
        observations["osl_actor"] = torch.cat(
            (osl_joint_positions, osl_joint_velocities), dim=1
        )
        return observations
