# Standalone port from ProSim commit 5c58afd (prosim/rl/mimic_reward.py).
# Upstream reward formulas: MuscleMimic and loco-mujoco (Apache-2.0).
# Native implementation is an offline reference builder and numerical oracle only.
"""MuscleMimic's tracking reward and termination (numpy, one env), with the
right leg below the hip left out: the prosthesis replaces it.

Ported from MuscleMimic (github.com/amathislab/musclemimic, Apache-2.0):
  - musclemimic/core/reward/trajectory_based.py  MimicReward.__call__, the
    use_mean_exp_reward=False branch (what mm-10m-2 trained with)
  - loco_mujoco/core/utils/math.py  calc_site_velocities, transform_motion,
    calculate_relative_site_quantities, calculate_relative_velocity_in_local_frame
  - musclemimic/core/terminal_state_handler/enhanced_fullbody.py
    MeanRelativeSiteDeviationWithRootTerminalStateHandler
The *_w_sum weights are mm-10m-2's config; the *_w_exp scales and the
out-of-bounds action coefficient are MimicReward's defaults (the config leaves
them unset).  MyoSuite's own myosuite/terms/mimic_reward.py is a different,
simplified reward -- not what the checkpoint's critic learned.

Their sim resets to the trajectory's XY and subtracts that offset from the
reference; ours starts at the clip's own XY, so the offset is zero and dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# mm-10m-2's reward_params.sites_for_mimic, in its order (index 0 is the main site)
SITES = (
    "pelvis_mimic",
    "upper_body_mimic",
    "head_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
)
# Below the right hip the body is the prosthesis (or the strapped-up shank).
# The right knee site stays: it marks where the thigh is, which the human drives.
MASKED_SITES = ("right_ankle_mimic", "right_toes_mimic")
MASKED_JOINT_PREFIXES = ("knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle")


@dataclass(frozen=True)
class Weights:  # *_w_sum
    qpos: float = 0.1
    qvel: float = 0.1
    root_pos: float = 0.1
    root_vel: float = 0.1
    rpos: float = 0.6
    rquat: float = 0.01
    rvel: float = 0.1  # applied to both the rotational and linear parts


@dataclass(frozen=True)
class Scales:  # *_w_exp
    qpos: float = 10.0
    qvel: float = 2.0
    root_pos: float = 10.0
    root_vel: float = 10.0
    rpos: float = 100.0
    rquat: float = 10.0
    rvel: float = 0.1


OUT_OF_BOUNDS_COEFF = 0.01
# terminal handler thresholds (mm-10m-2 terminal_state_params)
SITE_DEV_MAX = 1.0  # m, mean root-relative site deviation
ROOT_DEV_MAX = 1.0  # m, pelvis site position
ROOT_ORIENT_MAX = 1.0  # rad


def masked_joint(name: str) -> bool:
    return name.endswith("_r") and name.startswith(MASKED_JOINT_PREFIXES)


class MimicTracking:
    """Reward and termination for one model (the MyoFullBody the policy sees)
    against one clip.  The reference side is computed once per frame up front.

        mt = MimicTracking(shadow_model, clip_qpos, clip_qvel)
        r, terms = mt.reward(ds, frame, action)
        done, why = mt.terminated(ds, frame)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        mask: bool = True,
        w: Weights | None = None,
        s: Scales | None = None,
    ):
        w = Weights() if w is None else w
        s = Scales() if s is None else s
        self.m, self.w, self.s = model, w, s
        sites = [n for n in SITES if not (mask and n in MASKED_SITES)]
        self.site_ids = np.array([model.site(n).id for n in sites])
        self.body_ids = model.site_bodyid[self.site_ids]
        self.root_body_ids = model.body_rootid[self.body_ids]
        self.pelvis_site = model.site("pelvis_mimic").id
        qpos_ind, qvel_ind, quat = [], [], []
        for j in range(model.njnt):
            if mask and masked_joint(model.joint(j).name):
                continue
            a, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_ind += range(a, a + 7)
                qvel_ind += range(v, v + 6)
                quat += range(a + 3, a + 7)
                self.root_q, self.root_v = a, v
            else:
                nq = 4 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
                assert nq == 1, "ball joints: not in MyoFullBody"
                qpos_ind.append(a)
                qvel_ind.append(v)
        self.qpos_ind, self.qvel_ind = np.array(qpos_ind), np.array(qvel_ind)
        self.is_quat = np.isin(self.qpos_ind, quat)
        self.n_masked_joints = (
            sum(masked_joint(model.joint(j).name) for j in range(model.njnt))
            if mask
            else 0
        )
        # the reference, frame by frame
        d = mujoco.MjData(model)
        self.ref = []
        for f in range(len(qpos)):
            d.qpos[:], d.qvel[:] = qpos[f], qvel[f]
            mujoco.mj_forward(model, d)
            self.ref.append(self._features(d))
        self.ref_qpos, self.ref_qvel = np.asarray(qpos), np.asarray(qvel)

    # ------------------------------------------------------------------ helpers
    def _site_quantities(self, d: mujoco.MjData):
        """calculate_relative_site_quantities: positions (world-frame offsets),
        rotation vectors and 6D velocities of every site relative to site 0."""
        xpos = d.site_xpos[self.site_ids]
        xmat = d.site_xmat[self.site_ids].reshape(-1, 3, 3)
        # calc_site_velocities -> transform_motion(flg_local=False)
        cvel = d.cvel[self.body_ids]
        rot, lin = cvel[:, :3], cvel[:, 3:]
        lin = lin - np.cross(xpos - d.subtree_com[self.root_body_ids], rot)
        xvel = np.hstack([rot, lin])
        main_vel, vel = xvel[0], xvel[1:]
        main_mat, mats = xmat[0], xmat[1:]
        rel_rot = np.einsum("ik,nkj->nij", main_mat.T, mats)
        rpos = xpos[1:] - xpos[0]
        rangles = Rotation.from_matrix(rel_rot).as_rotvec()
        # calculate_relative_velocity_in_local_frame, exactly as upstream writes it
        rel_lin = np.einsum("jk,ik->ij", main_mat, main_vel[3:] - vel[:, 3:])
        ang_b = np.einsum("ikj,ik->ij", rel_rot, vel[:, :3])
        rvel = np.hstack([ang_b - main_vel[:3], rel_lin])
        return rpos, rangles, rvel

    def _root_local_vel(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, qpos[self.root_q + 3 : self.root_q + 7])
        lin = R.reshape(3, 3).T @ qvel[self.root_v : self.root_v + 3]
        return np.concatenate([lin, qvel[self.root_v + 3 : self.root_v + 6]])

    def _features(self, d: mujoco.MjData) -> dict:
        rpos, rangles, rvel = self._site_quantities(d)
        return {
            "rpos": rpos,
            "rangles": rangles,
            "rvel": rvel,
            "root_vel": self._root_local_vel(d.qpos, d.qvel),
            "pelvis_site": d.site_xpos[self.pelvis_site].copy(),
        }

    @staticmethod
    def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
        # == scipy (R(q1).inv() * R(q2)).magnitude(); invariant to the
        # scalar-first/last order upstream mixes up
        q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
        return float(2.0 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))

    # ------------------------------------------------------------------ public
    def reward(
        self, d: mujoco.MjData, frame: int, action: np.ndarray | None = None
    ) -> tuple[float, dict]:
        """MimicReward for state `d` against reference `frame` (upstream: the
        trajectory index after the step).  `d` must look like upstream's data
        after mj_step -- BypassWalk.shadow() does."""
        w, s, ref = self.w, self.s, self.ref[frame]
        q, v = d.qpos[self.qpos_ind], d.qvel[self.qvel_ind]
        rq, rv = (
            self.ref_qpos[frame][self.qpos_ind],
            self.ref_qvel[frame][self.qvel_ind],
        )
        qpos_dist = np.mean(np.square(q[~self.is_quat] - rq[~self.is_quat]))
        qpos_dist += self._quat_angle(q[self.is_quat], rq[self.is_quat])
        qvel_dist = np.mean(np.square(v - rv))
        rpos, rangles, rvel = self._site_quantities(d)
        rpos_dist = np.mean(np.square(rpos - ref["rpos"]))
        rangles_dist = np.mean(np.square(rangles - ref["rangles"]))
        rvel_rot_dist = np.mean(np.square(rvel[:, :3] - ref["rvel"][:, :3]))
        rvel_lin_dist = np.mean(np.square(rvel[:, 3:] - ref["rvel"][:, 3:]))
        a = self.root_q
        root_pos_dist = np.mean(
            np.square(d.qpos[a : a + 3] - self.ref_qpos[frame][a : a + 3])
        )
        root_vel_dist = np.mean(
            np.square(self._root_local_vel(d.qpos, d.qvel) - ref["root_vel"])
        )
        terms = {
            "qpos": w.qpos * np.exp(-s.qpos * qpos_dist),
            "qvel": w.qvel * np.exp(-s.qvel * qvel_dist),
            "root_pos": w.root_pos * np.exp(-s.root_pos * root_pos_dist),
            "root_vel": w.root_vel * np.exp(-s.root_vel * root_vel_dist),
            "rpos": w.rpos * np.exp(-s.rpos * rpos_dist),
            "rquat": w.rquat * np.exp(-s.rquat * rangles_dist),
            "rvel_rot": w.rvel * np.exp(-s.rvel * rvel_rot_dist),
            "rvel_lin": w.rvel * np.exp(-s.rvel * rvel_lin_dist),
        }
        pen = 0.0
        if action is not None:  # ctrlrange is [-1, 1] for every muscle
            a_ = np.asarray(action)
            pen = (
                -float(np.sum(np.square(np.clip(np.abs(a_) - 1.0, 0.0, None))))
                / a_.shape[0]
            )
        terms["out_of_bounds"] = max(OUT_OF_BOUNDS_COEFF * pen, -1.0)
        r = float(np.nan_to_num(max(sum(terms.values()), 0.0), nan=0.0))
        return r, {k: float(x) for k, x in terms.items()}

    def terminated(self, d: mujoco.MjData, frame: int) -> tuple[bool, str]:
        """The three upstream checks (thresholds 1.0), over the kept sites."""
        ref = self.ref[frame]
        rpos, _, _ = self._site_quantities(d)
        if np.mean(np.linalg.norm(rpos - ref["rpos"], axis=-1)) > SITE_DEV_MAX:
            return True, "site"
        if (
            np.linalg.norm(d.site_xpos[self.pelvis_site] - ref["pelvis_site"])
            > ROOT_DEV_MAX
        ):
            return True, "root"
        a = self.root_q
        if (
            self._quat_angle(d.qpos[a + 3 : a + 7], self.ref_qpos[frame][a + 3 : a + 7])
            > ROOT_ORIENT_MAX
        ):
            return True, "orient"
        return False, ""
