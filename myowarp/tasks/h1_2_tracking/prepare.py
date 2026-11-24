"""Convert already-retargeted H1-2 LAFAN1 CSV into the tracking NPZ contract."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .config import H1TrackingCfg
from .model import build_h1_tracking_model


def prepare_motion(source, output=None):
    cfg = H1TrackingCfg()
    model = build_h1_tracking_model(cfg.model_path)
    output = Path(output or cfg.motion_path)
    raw = np.loadtxt(source, delimiter=",")
    if (
        raw.ndim != 2
        or raw.shape[1] != 34
        or len(raw) < 3
        or not np.isfinite(raw).all()
    ):
        raise ValueError("Expected finite 34-column H1-2 CSV")
    source_time = np.arange(len(raw)) / 30.0
    time = np.arange(int(np.floor(source_time[-1] / cfg.step_dt)) + 1) * cfg.step_dt
    qpos = np.stack([np.interp(time, source_time, raw[:, i]) for i in range(34)], -1)
    qpos[:, 3:7] = Slerp(source_time, Rotation.from_quat(raw[:, 3:7]))(time).as_quat()[
        :, [3, 0, 1, 2]
    ]
    # Rebase XY only; keep the recorded global path and heading thereafter.
    qpos[:, :2] -= qpos[0, :2]
    data = mujoco.MjData(model)
    foot = [
        model.geom(f"{side}_foot{i}_collision").id
        for side in ("left", "right")
        for i in range(1, 8)
    ]
    lowest = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        lowest.append(
            min(
                data.geom_xpos[i, 2]
                - model.geom_size[i, 0]
                - model.geom_size[i, 1] * abs(data.geom_xmat[i, 8])
                for i in foot
            )
        )
    # Constant correction preserves root velocity and the recorded path.
    offset = max(0.0, -float(min(lowest)))
    qpos[:, 2] += offset
    qvel = np.empty((len(time), model.nv))
    for f in range(len(time)):
        a, b = max(0, f - 1), min(len(time) - 1, f + 1)
        mujoco.mj_differentiatePos(
            model, qvel[f], (b - a) * cfg.step_dt, qpos[a], qpos[b]
        )
    body_ids = [i for i in range(1, model.nbody) if model.body(i).name != "terrain"]
    roots = model.body_rootid[body_ids]
    arrays = {
        k: np.empty((len(time), len(body_ids), d), np.float32)
        for k, d in [
            ("body_pos_w", 3),
            ("body_quat_w", 4),
            ("body_lin_vel_w", 3),
            ("body_ang_vel_w", 3),
        ]
    }
    for f, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = qvel[f]
        mujoco.mj_forward(model, data)
        arrays["body_pos_w"][f] = data.xpos[body_ids]
        arrays["body_quat_w"][f] = data.xquat[body_ids]
        c = data.cvel[body_ids]
        arrays["body_ang_vel_w"][f] = c[:, :3]
        arrays["body_lin_vel_w"][f] = c[:, 3:] + np.cross(
            c[:, :3], data.xpos[body_ids] - data.subtree_com[roots]
        )
        if f % 3000 == 0:
            print(f"Preparing H1-2 tracking reference: {f + 1}/{len(time)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=[1 / cfg.step_dt],
        joint_pos=qpos[:, 7:].astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        joint_names=[model.joint(i).name for i in range(1, model.njnt)],
        body_names=[model.body(i).name for i in body_ids],
        source_file=str(Path(source).resolve()),
        source_fps=[30.0],
        ground_offset_m=offset,
        **arrays,
    )
    print(
        f"Saved {len(time)} frames ({time[-1]:.2f} s), constant ground correction {offset:.4f} m: {output}",
        flush=True,
    )
    return output
