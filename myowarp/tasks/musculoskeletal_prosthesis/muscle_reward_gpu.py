"""Batched CUDA equivalent of ProSim's masked MuscleMimic tracking reward.

The native oracle is used once for reference preprocessing, never for rollouts.
"""

import numpy as np
import torch

from myowarp.musclemimic.fullbody_obs_torch import _relative_site_quantities

from .muscle_reward_native import MimicTracking


def quaternion_angle(a, b):
    # Stable equivalent of 2*acos(abs(dot(normalize(a), normalize(b)))).
    a = torch.nn.functional.normalize(a.double(), dim=-1)
    b = torch.nn.functional.normalize(b.double(), dim=-1)
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    return (
        4
        * torch.atan2(
            torch.linalg.vector_norm(a - b, dim=-1),
            torch.linalg.vector_norm(a + b, dim=-1),
        )
    ).float()


def root_local_velocity(qpos, qvel, root_q, root_v):
    quat = torch.nn.functional.normalize(qpos[:, root_q + 3 : root_q + 7], dim=-1)
    v = qvel[:, root_v : root_v + 3]
    cross = torch.cross(quat[:, 1:], v, dim=-1)
    local = v + 2 * (torch.cross(quat[:, 1:], cross, dim=-1) - quat[:, :1] * cross)
    return torch.cat((local, qvel[:, root_v + 3 : root_v + 6]), -1)


class MuscleMimicRewardBatch:
    def __init__(self, model, clip, device):
        native = MimicTracking(model, clip.qpos, clip.qvel, mask=True)
        self.native = native
        self.root_q, self.root_v = native.root_q, native.root_v

        def index(x):
            return torch.tensor(np.asarray(x), dtype=torch.long, device=device)

        self.site_ids = index(native.site_ids)
        self.site_bodyid = index(model.site_bodyid)
        self.body_rootid = index(model.body_rootid)
        self.pelvis_site = native.pelvis_site
        self.qpos_ids, self.qvel_ids = index(native.qpos_ind), index(native.qvel_ind)
        self.is_quat = torch.tensor(native.is_quat, device=device)
        self.qref = torch.tensor(
            np.asarray(clip.qpos), dtype=torch.float32, device=device
        )
        self.vref = torch.tensor(
            np.asarray(clip.qvel), dtype=torch.float32, device=device
        )
        self.features = {
            name: torch.tensor(
                np.stack([f[name] for f in native.ref]),
                dtype=torch.float32,
                device=device,
            )
            for name in native.ref[0]
        }

    def site_quantities(self, data):
        return _relative_site_quantities(
            site_ids=self.site_ids,
            site_xpos=data.site_xpos,
            site_xmat=data.site_xmat,
            cvel_parent=data.cvel,
            subtree_com_root=data.subtree_com,
            site_bodyid=self.site_bodyid,
            body_rootid=self.body_rootid,
        )

    def reward(self, data, frame, action):
        ref, w, s = self.features, self.native.w, self.native.s
        q, rq = data.qpos[:, self.qpos_ids], self.qref[frame][:, self.qpos_ids]
        v, rv = data.qvel[:, self.qvel_ids], self.vref[frame][:, self.qvel_ids]
        average_squared = lambda a: a.square().flatten(1).mean(-1)
        angle = quaternion_angle(q[:, self.is_quat], rq[:, self.is_quat])
        qdist = average_squared(q[:, ~self.is_quat] - rq[:, ~self.is_quat]) + angle
        rpos, rangles, rvel = self.site_quantities(data)
        a = self.root_q
        terms = {
            "qpos": w.qpos * torch.exp(-s.qpos * qdist),
            "qvel": w.qvel * torch.exp(-s.qvel * average_squared(v - rv)),
            "root_pos": w.root_pos
            * torch.exp(
                -s.root_pos
                * average_squared(data.qpos[:, a : a + 3] - self.qref[frame, a : a + 3])
            ),
            "root_vel": w.root_vel
            * torch.exp(
                -s.root_vel
                * average_squared(
                    root_local_velocity(data.qpos, data.qvel, self.root_q, self.root_v)
                    - ref["root_vel"][frame]
                )
            ),
            "rpos": w.rpos
            * torch.exp(-s.rpos * average_squared(rpos - ref["rpos"][frame])),
            "rquat": w.rquat
            * torch.exp(-s.rquat * average_squared(rangles - ref["rangles"][frame])),
            "rvel_rot": w.rvel
            * torch.exp(
                -s.rvel * average_squared(rvel[:, :, :3] - ref["rvel"][frame][:, :, :3])
            ),
            "rvel_lin": w.rvel
            * torch.exp(
                -s.rvel * average_squared(rvel[:, :, 3:] - ref["rvel"][frame][:, :, 3:])
            ),
            "out_of_bounds": (
                -0.01 * (action.abs() - 1).clamp_min(0).square().mean(-1)
            ).clamp_min(-1),
        }
        reward = torch.nan_to_num(sum(terms.values()).clamp_min(0), nan=0.0)
        terminated = (
            (torch.linalg.vector_norm(rpos - ref["rpos"][frame], dim=-1).mean(-1) > 1)
            | (
                torch.linalg.vector_norm(
                    data.site_xpos[:, self.pelvis_site] - ref["pelvis_site"][frame],
                    dim=-1,
                )
                > 1
            )
            | (
                quaternion_angle(
                    data.qpos[:, a + 3 : a + 7], self.qref[frame, a + 3 : a + 7]
                )
                > 1
            )
        )
        return reward, terms, terminated
