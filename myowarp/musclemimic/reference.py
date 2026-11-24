"""Steady gait-cycle looping ported from prosim/reference.py.

Cycle selection, seam corrections and forward-kinematics export are local;
this module has no prosim or MyoSuite runtime dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

KINEMATIC_KEYS = ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")

def _apply_couplings(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """Put every two-joint equality's follower on its polynomial, so a pose
    written into qpos starts on the constraint manifold instead of snapping."""
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        a1 = m.jnt_qposadr[m.eq_obj1id[i]]
        if m.eq_obj2id[i] < 0:
            d.qpos[a1] = m.eq_data[i][0]
        else:
            x = d.qpos[m.jnt_qposadr[m.eq_obj2id[i]]]
            d.qpos[a1] = sum(m.eq_data[i][k] * x**k for k in range(5))


def _write(z, qpos: np.ndarray, qvel: np.ndarray, model: mujoco.MjModel,
           dst: str | os.PathLike, fix=None) -> Path:
    """Write `dst`: clip `z` with qpos/qvel replaced and every kinematic
    array recomputed from them by forward kinematics.  `fix(d, frame)` may
    edit d.qpos/d.qvel before each forward."""
    sids = [model.site(n).id for n in z["site_names"]]
    out = {k: z[k] for k in z.files}
    qpos, qvel = qpos.copy(), qvel.copy()
    n = qpos.shape[0]
    kin = {k: np.empty((n,) + z[k].shape[1:], z[k].dtype) for k in KINEMATIC_KEYS}
    d = mujoco.MjData(model)
    for f in range(n):
        d.qpos[:] = qpos[f]
        d.qvel[:] = qvel[f]
        if fix is not None:
            fix(d, f)
        mujoco.mj_forward(model, d)
        qpos[f], qvel[f] = d.qpos, d.qvel
        got = {"xpos": d.xpos, "xquat": d.xquat, "cvel": d.cvel,
               "subtree_com": d.subtree_com, "site_xpos": d.site_xpos[sids],
               "site_xmat": d.site_xmat[sids]}
        for k in KINEMATIC_KEYS:
            kin[k][f] = got[k]
    out["qpos"], out["qvel"] = qpos.astype(np.float32), qvel.astype(np.float32)
    out.update(kin)
    if "split_points" in out:
        out["split_points"] = np.array([0, n], dtype=out["split_points"].dtype)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename, so processes relocking the same clip at once never
    # read each other's half-written file
    tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.npz")
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    return dst


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def best_cycle(model: mujoco.MjModel, qpos: np.ndarray, hz: float,
               t_min: float = 0.8, t_max: float = 1.6, speed_frac: float = 0.85):
    """(start, length) of the gait cycle inside the clip's steady locomotion whose
    last frame comes back closest to its first: joint-angle RMS plus a little
    joint-velocity RMS, over every hinge joint.  The root is not compared --
    loop_clip() carries it forward."""
    if not (np.isfinite(t_min) and np.isfinite(t_max) and 0 < t_min <= t_max):
        raise ValueError("Cycle duration bounds must be finite, positive and ordered")
    v = np.linalg.norm(np.gradient(qpos[:, :2], 1 / hz, axis=0), axis=1)
    k = max(1, int(hz))
    v = np.convolve(v, np.ones(k) / k, mode="same")
    steady = np.nonzero(v > speed_frac * v.max())[0]
    if not len(steady):
        raise ValueError("No steady locomotion region found in this clip")
    a, b = int(steady[0]), int(steady[-1])
    adr = [model.jnt_qposadr[j] for j in range(model.njnt)
           if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    J = np.degrees(qpos[:, adr])
    dJ = np.gradient(J, 1 / hz, axis=0)
    best = None
    for i in range(a, b):
        for L in range(max(2, int(np.ceil(t_min * hz))), int(np.floor(t_max * hz)) + 1):
            if i + L > b:
                break
            cost = (np.sqrt(np.mean((J[i + L] - J[i]) ** 2))
                    + 0.02 * np.sqrt(np.mean((dJ[i + L] - dJ[i]) ** 2)))
            if best is None or cost < best[0]:
                best = (cost, i, L)
    if best is None:
        raise ValueError("Steady locomotion region is too short for a complete gait cycle")
    return best[1], best[2]


def loop_clip(src: str | os.PathLike, dst: str | os.PathLike, model: mujoco.MjModel,
              seconds: float, start: int | None = None, length: int | None = None,
              *, t_min: float = 0.8, t_max: float = 1.6) -> Path:
    """Write `dst`: one gait cycle of `src` repeated for `seconds`, a constant-
    speed straight locomotion reference of any length.

    The cycle is frames [start, start+length]; by default best_cycle().  Its
    small end-to-start mismatch is spread linearly over the cycle so it closes
    exactly: every joint by its own gap (about 1 deg at the legs), the root
    heading by its drift.  Each repeat is shifted by the cycle's displacement.
    qvel is recomputed by central differences on the looped path, so it is
    continuous across the seams; everything else by forward kinematics."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be positive and finite")
    if (start is None) != (length is None):
        raise ValueError("Specify both start and length, or neither")
    z = np.load(src, allow_pickle=True)
    if [model.joint(i).name for i in range(model.njnt)] != list(z["joint_names"]):
        raise ValueError("clip joint order does not match the model")
    q = z["qpos"].astype(np.float64)
    hz = float(z["frequency"])
    if not np.isfinite(hz) or hz <= 0 or len(q) < 3:
        raise ValueError("Clip needs a positive frequency and at least three frames")
    if start is None or length is None:
        start, length = best_cycle(model, q, hz, t_min=t_min, t_max=t_max)
    L = length
    if start < 0 or L < 2 or start + L >= len(q):
        raise ValueError("Cycle start/end must lie within the source clip, with length >= 2")
    cyc = q[start:start + L + 1].copy()
    t = np.arange(L + 1)[:, None] / L

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert free == [0] and model.jnt_qposadr[0] == 0, "expects one free root joint at qpos[0:7]"
    # joints: close the gap
    cyc[:, 7:] -= t * (cyc[-1, 7:] - cyc[0, 7:])
    # root heading: remove the drift, turning about the cycle's start
    drift = _yaw(cyc[-1, 3:7]) - _yaw(cyc[0, 3:7])
    p0 = cyc[0, :2].copy()
    for f in range(L + 1):
        a = -t[f, 0] * drift
        c, s_ = np.cos(a), np.sin(a)
        cyc[f, :2] = p0 + np.array([[c, -s_], [s_, c]]) @ (cyc[f, :2] - p0)
        rq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
        out = np.empty(4)
        mujoco.mju_mulQuat(out, rq, cyc[f, 3:7])
        cyc[f, 3:7] = out / np.linalg.norm(out)
    step = cyc[-1, :3] - cyc[0, :3]
    step[2] = 0.0

    n = int(round(seconds * hz)) + 1
    qpos = np.empty((n, q.shape[1]))
    for f in range(n):
        k, r = divmod(f, L)
        qpos[f] = cyc[r]
        qpos[f, :3] += k * step

    # velocities from the looped path itself (seams included)
    qvel = np.empty((n, model.nv))
    dt = 1.0 / hz
    for f in range(n):
        a, b = max(f - 1, 0), min(f + 1, n - 1)
        mujoco.mj_differentiatePos(model, qvel[f], (b - a) * dt, qpos[a], qpos[b])

    def fix(d, f):
        _apply_couplings(model, d)
    out = _write(z, qpos, qvel, model, dst, fix)
    print(f"loop_clip: {Path(src).stem} frames {start}->{start + L} ({L / hz:.2f} s cycle), "
          f"{np.linalg.norm(step):.2f} m per cycle, heading drift removed {np.degrees(drift):+.2f} deg, "
          f"{n / hz:.1f} s")
    return out

