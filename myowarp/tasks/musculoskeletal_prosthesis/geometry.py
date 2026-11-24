"""CMU linkage kinematics ported from prosim/sim_leg.py; no controller logic."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

_A = np.array([0.00034, -0.115])
_B = np.array([0.00062, -0.2131])
_C = np.array([-0.06959, -0.11819])
_D = np.array([-0.06948, -0.20319])
_L2, _L3, _L4 = (
    np.linalg.norm(_C - _A),
    np.linalg.norm(_D - _C),
    np.linalg.norm(_D - _B),
)
_TH2, _TH3, _TH4 = (
    np.arctan2(*(_C - _A)[::-1]),
    np.arctan2(*(_D - _C)[::-1]),
    np.arctan2(*(_D - _B)[::-1]),
)


def _fourbar(d_drive: float) -> tuple[float, float]:
    """Closed-form four-bar: input crank rotation -> (coupler, ankle) rotations,
    all relative to the as-built pose.  Same solve that was validated against
    MuJoCo to ~1e-13 rad over +/-20 deg."""
    th2 = _TH2 + d_drive
    C = _A + _L2 * np.array([np.cos(th2), np.sin(th2)])
    r = _B - C
    dist = np.linalg.norm(r)
    cg = (dist**2 + _L4**2 - _L3**2) / (2 * dist * _L4)
    g = np.arccos(np.clip(cg, -1.0, 1.0))
    beta = np.arctan2(r[1], r[0])
    for sgn in (+1, -1):  # pick the assembly branch
        th4 = beta + sgn * g + np.pi
        D = _B + _L4 * np.array([np.cos(th4), np.sin(th4)])
        if abs(np.linalg.norm(D - C) - _L3) < 1e-9:
            d4 = np.arctan2(np.sin(th4 - _TH4), np.cos(th4 - _TH4))
            if abs(d4) < np.pi / 2:
                th3 = np.arctan2(*(D - C)[::-1])
                return np.arctan2(np.sin(th3 - _TH3), np.cos(th3 - _TH3)), d4
    raise RuntimeError(f"four-bar does not close at drive={d_drive:.4f} rad")


def _drive_for_ankle(ankle: float) -> float:
    """Invert the four-bar by bisection; the map is monotonic over the range."""
    lo, hi = np.radians(-45.0), np.radians(45.0)
    if not _fourbar(lo)[1] <= ankle <= _fourbar(hi)[1]:
        raise ValueError(f"ankle {np.degrees(ankle):.2f} deg out of four-bar range")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _fourbar(mid)[1] < ankle:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def set_prosthesis(model, data, knee, drive):
    coupler, ankle = _fourbar(float(drive))
    for name, value in {
        "knee": knee,
        "knee_motor": knee,
        "ankle_drive": drive,
        "ankle_motor": drive,
        "ankle_coupler": coupler - drive,
        "ankle": ankle,
    }.items():
        data.qpos[model.joint(name).qposadr[0]] = value


def sole(model, data):
    """Midpoint of heel/toe sphere undersides, in the real prosthetic foot frame."""
    ids = [model.geom(n).id for n in ["heel", "toe"]]
    body = model.body("ankle_foot").id
    rotation = data.xmat[body].reshape(3, 3)
    point = np.mean(data.geom_xpos[ids], axis=0) - rotation[:, 1] * np.mean(
        model.geom_size[ids, 0]
    )
    return point, rotation


def joint_map(source, target, *, allow_missing=False):
    src_q, dst_q, src_v, dst_v = [], [], [], []
    for i in range(source.njnt):
        try:
            j = target.joint(source.joint(i).name).id
        except KeyError:
            if allow_missing:
                continue
            raise
        nq = {0: 7, 1: 4}.get(int(source.jnt_type[i]), 1)
        nv = {0: 6, 1: 3}.get(int(source.jnt_type[i]), 1)
        src_q.extend(range(source.jnt_qposadr[i], source.jnt_qposadr[i] + nq))
        dst_q.extend(range(target.jnt_qposadr[j], target.jnt_qposadr[j] + nq))
        src_v.extend(range(source.jnt_dofadr[i], source.jnt_dofadr[i] + nv))
        dst_v.extend(range(target.jnt_dofadr[j], target.jnt_dofadr[j] + nv))
    return tuple(np.array(x, dtype=int) for x in (src_q, dst_q, src_v, dst_v))


def muscle_map(source, target):
    """Shared actuator and activation indices, preserving full shadow layout."""
    source_ids, target_ids = [], []
    for i in range(source.nu):
        try:
            j = target.actuator(source.actuator(i).name).id
        except KeyError:
            continue
        source_ids.append(i)
        target_ids.append(j)
    source_ids, target_ids = (
        np.array(source_ids, dtype=int),
        np.array(target_ids, dtype=int),
    )
    return (
        source_ids,
        target_ids,
        source.actuator_actadr[source_ids].copy(),
        target.actuator_actadr[target_ids].copy(),
    )


def body_velocities(model, data, ids):
    # World velocities at link origins, consistent with xpos tracking.
    result = []
    for body in ids:
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, int(body))
        result.append(np.r_[jacp @ data.qvel, jacr @ data.qvel])
    return np.array(result)


def angular_velocities(rotations, dt):
    rotations = np.asarray(rotations)
    out = np.empty(rotations.shape[:-2] + (3,))
    for f in range(len(rotations)):
        a, b = max(0, f - 1), min(len(rotations) - 1, f + 1)
        delta = rotations[b] @ np.swapaxes(rotations[a], -1, -2)
        out[f] = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(
            out[f].shape
        ) / ((b - a) * dt)
    return out
