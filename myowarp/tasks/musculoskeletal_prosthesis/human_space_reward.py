"""Exp8: CMU leg -> virtual human mapping and lower-right MuscleMimic reward.

Position formulas ported from cmu_leg walking/retargeting.py and
neuromuscular_stance.py (_to_human_angles). No sibling-repo runtime dependency.
"""

import numpy as np
import torch

from .gpu_sim import WarpBatch
from .muscle_reward_gpu import MuscleMimicRewardBatch
from .muscle_reward_native import Scales, Weights

# CMU LEG configuration, metres. The virtual human has equal-length segments.
CMU_THIGH = 0.66
CMU_SHANK = 0.215
LOWER_LEG_TRACKING_TERMS = ("qpos", "qvel", "rpos", "rquat", "rvel_rot", "rvel_lin")


def leg_to_human(q, v):
    """Batched [knee, ankle] radians/rad/s -> CMU virtual-human coordinates.

    Knee flexion and ankle dorsiflexion are positive. Signed half-angle form
    avoids acos cancellation at a straight knee. Velocities use the analytic
    derivative, including its continuous straight-knee limit (the upstream
    torque helper's identity shortcut is not the derivative of this map).
    """
    knee, ankle = q.unbind(-1)
    kd, ad = v.unbind(-1)
    lt, ls = CMU_THIGH, CMU_SHANK
    ratio = 2 * np.sqrt(lt * ls) / (lt + ls)
    half_sin = torch.sin(knee / 2)
    arg = ratio * half_sin
    kh = 2 * torch.asin(arg.clamp(-1, 1))
    dkh = ratio * torch.cos(knee / 2) / (1 - arg.square()).clamp_min(1e-12).sqrt()
    chord = (lt * lt + ls * ls + 2 * lt * ls * knee.cos()).clamp_min(1e-12).sqrt()
    u = ls * knee.sin() / chord
    t1 = torch.asin(u.clamp(-1, 1))
    dl = -lt * ls * knee.sin() / chord
    du = ls * (knee.cos() * chord - knee.sin() * dl) / chord.square()
    dt1 = du / (1 - u.square()).clamp_min(1e-12).sqrt()
    ah = ankle + kh / 2 - knee + t1
    return torch.stack((kh, ah), -1), torch.stack(
        (dkh * kd, ad + (dkh / 2 - 1 + dt1) * kd), -1
    )


class LowerLegMimicReward(MuscleMimicRewardBatch):
    """Reference features only at the biological right ankle/toes.

    Root terms are deliberately absent. Joint terms use only knee/ankle;
    unactuated subtalar/toes are fixed to zero, never filled from reference.
    """

    def __init__(self, model, clip, device, cfg, num_envs):
        # The full oracle is used offline solely to obtain the original
        # MuscleMimic relative-site reference definitions and coefficients.
        from .muscle_reward_native import MimicTracking

        native = MimicTracking(model, clip.qpos, clip.qvel, mask=False)
        indices = [
            native.site_ids.tolist().index(model.site(n).id) - 1
            for n in ("right_ankle_mimic", "right_toes_mimic")
        ]
        tensor = lambda x: torch.tensor(
            np.asarray(x), device=device, dtype=torch.float32
        )
        self.features = {
            k: tensor(np.stack([f[k][indices] for f in native.ref]))
            for k in ("rpos", "rangles", "rvel")
        }
        self.site_ids = torch.tensor(
            [
                model.site(n).id
                for n in ("pelvis_mimic", "right_ankle_mimic", "right_toes_mimic")
            ],
            device=device,
        )
        self.site_bodyid = torch.tensor(
            model.site_bodyid.copy(), device=device, dtype=torch.long
        )
        self.body_rootid = torch.tensor(
            model.body_rootid.copy(), device=device, dtype=torch.long
        )
        self.qids = torch.tensor(
            [model.joint(n).qposadr[0] for n in ("knee_angle_r", "ankle_angle_r")],
            device=device,
        )
        self.vids = torch.tensor(
            [model.joint(n).dofadr[0] for n in ("knee_angle_r", "ankle_angle_r")],
            device=device,
        )
        self.qref = tensor(clip.qpos)[:, self.qids]
        self.vref = tensor(clip.qvel)[:, self.vids]
        self.sim = WarpBatch(model, num_envs, device, cfg.nconmax, cfg.njmax)

    @torch.no_grad()
    def reward(self, real, human, frame):
        d = self.sim.view
        # Current body state, never the reference body state. Own storage means
        # this reward transform cannot modify the policy's virtual shadow.
        d.qpos.copy_(human.data.qpos)
        d.qvel.copy_(human.data.qvel)
        mapped_q, mapped_v = leg_to_human(
            real.qpos[:, human.prosthesis_q], real.qvel[:, human.prosthesis_v]
        )
        d.qpos[:, self.qids] = mapped_q
        d.qvel[:, self.vids] = mapped_v
        d.qpos[:, human.view_q[2:]] = 0
        d.qvel[:, human.view_v[2:]] = 0
        q, v = mapped_q[:, 0], mapped_v[:, 0]
        for qi, vi, c in human.followers:
            d.qpos[:, qi] = sum(float(c[k]) * q**k for k in range(5))
            d.qvel[:, vi] = sum(k * float(c[k]) * q ** (k - 1) for k in range(1, 5)) * v
        self.sim.forward()
        rpos, rangles, rvel = self.site_quantities(d)
        mse = lambda x: x.square().flatten(1).mean(-1)
        w, s, ref = Weights(), Scales(), self.features
        return {
            "qpos": w.qpos * torch.exp(-s.qpos * mse(mapped_q - self.qref[frame])),
            "qvel": w.qvel * torch.exp(-s.qvel * mse(mapped_v - self.vref[frame])),
            "rpos": w.rpos * torch.exp(-s.rpos * mse(rpos - ref["rpos"][frame])),
            "rquat": w.rquat
            * torch.exp(-s.rquat * mse(rangles - ref["rangles"][frame])),
            "rvel_rot": w.rvel
            * torch.exp(-s.rvel * mse(rvel[..., :3] - ref["rvel"][frame][..., :3])),
            "rvel_lin": w.rvel
            * torch.exp(-s.rvel * mse(rvel[..., 3:] - ref["rvel"][frame][..., 3:])),
        }
