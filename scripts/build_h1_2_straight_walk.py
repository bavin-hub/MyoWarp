"""Build a smooth periodic straight walk from the selected LAFAN1 segment."""

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from myowarp.tasks.h1_2_tracking.model import build_h1_tracking_model


def periodic_curve(values, dt):
    """Correct endpoint pose/velocity over the last 0.4s, then fit periodically."""
    values = values.copy()
    width = min(20, len(values) - 2)
    duration = width * dt
    u = np.linspace(0, 1, width + 1)[:, None]
    delta = values[-1] - values[0]
    dv = (values[-1] - values[-2] - values[1] + values[0]) / dt
    # Cubic Hermite correction has zero value/slope at the blend entrance.
    correction = (3 * u * u - 2 * u * u * u) * delta + (
        u * u * u - u * u
    ) * duration * dv
    values[-width - 1 :] -= correction
    values[-1] = values[0]
    return CubicSpline(np.arange(len(values)) * dt, values, bc_type="periodic")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets/reference_clips/h1_2_straight_walk_120s.npz",
    )
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error("--seconds must be positive and finite")
    source = ROOT / "assets/reference_clips/walk1_subject1_h1_2.npz"
    with np.load(source) as ref:
        fps = float(ref["fps"].item())
        joint = ref["joint_pos"].astype(float)
        velocity = ref["joint_vel"].astype(float)
        root = ref["body_pos_w"][:, 0].astype(float)
        quat = ref["body_quat_w"][:, 0].astype(float)
        names = ref["joint_names"].copy()
    dt = 1 / fps
    candidates = []
    for a in range(round(202.5 * fps), round(204.5 * fps)):
        for b in range(a + round(3 * fps), round(208.5 * fps)):
            score = np.mean((joint[a] - joint[b]) ** 2) + 0.0025 * np.mean(
                (velocity[a] - velocity[b]) ** 2
            )
            candidates.append((score, a, b))
    _, a, b = min(candidates)
    period = (b - a) * dt
    displacement = root[b, :2] - root[a, :2]
    heading = np.arctan2(displacement[1], displacement[0])
    rotation = Rotation.from_euler("z", -heading)
    positions = rotation.apply(root[a : b + 1] - np.r_[root[a, :2], 0])
    orientations = rotation * Rotation.from_quat(quat[a : b + 1][:, [1, 2, 3, 0]])
    angles = np.unwrap(orientations.as_euler("xyz"), axis=0)
    fraction = np.linspace(0, 1, b - a + 1)[:, None]
    speed = np.linalg.norm(displacement) / period
    positions[:, :2] -= fraction * positions[-1, :2]
    # Remove the small net turn while preserving within-stride yaw variation.
    angles[:, 2] -= fraction[:, 0] * (angles[-1, 2] - angles[0, 2])
    features = np.c_[positions, angles, joint[a : b + 1]]
    curve = periodic_curve(features, dt)
    t = np.arange(round(args.seconds * fps) + 1) * dt
    phase = np.remainder(t, period)
    values = curve(phase)
    qpos = np.c_[
        values[:, :3],
        Rotation.from_euler("xyz", values[:, 3:6]).as_quat()[:, [3, 0, 1, 2]],
        values[:, 6:],
    ]
    qpos[:, 0] += speed * t
    model = build_h1_tracking_model(ROOT / "assets/robots/unitree_h1_2/xmls/h1_2.xml")
    data = mujoco.MjData(model)
    feet = [
        model.geom(f"{side}_foot{i}_collision").id
        for side in ("left", "right")
        for i in range(1, 8)
    ]
    lowest = float("inf")
    for q in qpos[: b - a + 1]:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        lowest = min(
            lowest,
            min(
                data.geom_xpos[i, 2]
                - model.geom_size[i, 0]
                - model.geom_size[i, 1] * abs(data.geom_xmat[i, 8])
                for i in feet
            ),
        )
    correction = max(0.0, -lowest)
    qpos[:, 2] += correction
    qvel = np.empty((len(t), model.nv))
    for i in range(len(t)):
        lo, hi = max(0, i - 1), min(len(t) - 1, i + 1)
        mujoco.mj_differentiatePos(model, qvel[i], (hi - lo) * dt, qpos[lo], qpos[hi])
    ids = [i for i in range(1, model.nbody) if model.body(i).name != "terrain"]
    roots = model.body_rootid[ids]
    arrays = {
        key: np.empty((len(t), len(ids), dim), np.float32)
        for key, dim in [
            ("body_pos_w", 3),
            ("body_quat_w", 4),
            ("body_lin_vel_w", 3),
            ("body_ang_vel_w", 3),
        ]
    }
    for i in range(len(t)):
        data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, data)
        arrays["body_pos_w"][i] = data.xpos[ids]
        arrays["body_quat_w"][i] = data.xquat[ids]
        arrays["body_ang_vel_w"][i] = data.cvel[ids, :3]
        arrays["body_lin_vel_w"][i] = data.cvel[ids, 3:] + np.cross(
            data.cvel[ids, :3], data.xpos[ids] - data.subtree_com[roots]
        )
    # Periodic spline guarantees identical pose and first/second derivatives.
    for order in (0, 1, 2):
        np.testing.assert_allclose(curve(0, order), curve(period, order), atol=1e-8)
    assert all(np.isfinite(v).all() for v in arrays.values())
    report = dict(
        source_file=str(source),
        source_start_s=a * dt,
        source_end_s=b * dt,
        cycle_seconds=period,
        blend_seconds=0.4,
        speed_m_s=speed,
        duration_seconds=t[-1],
        ground_offset_m=correction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fps=[fps],
        joint_pos=qpos[:, 7:].astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        joint_names=names,
        body_names=[model.body(i).name for i in ids],
        **arrays,
        **report,
    )
    print(report)
    print(f"Saved {len(t)} frames: {args.output}")


if __name__ == "__main__":
    main()
