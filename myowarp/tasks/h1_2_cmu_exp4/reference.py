"""Constrained foot-pose reference adaptation using the actual H1/CMU geometry."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from myowarp.tasks.h1_2_tracking.model import build_h1_tracking_model
from myowarp.tasks.musculoskeletal_prosthesis.geometry import (
    _drive_for_ankle,
    joint_map,
    set_prosthesis,
    sole,
)

from .config import ROOT
from .model import build_model


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_fingerprint():
    return hashlib.sha256(
        (ROOT / "assets/robots/h1_with_cmu_leg/h1_with_cmu_leg.xml").read_bytes()
        + Path(__file__).with_name("model.py").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()


def prepare_reference(cfg):
    human = build_h1_tracking_model(ROOT / "assets/robots/unitree_h1_2/xmls/h1_2.xml")
    combined = build_model()
    hd = mujoco.MjData(human)
    cd = mujoco.MjData(combined)
    with np.load(cfg.motion_path) as src:
        source_fps = float(src["fps"].item())
        n = len(src["joint_pos"])
        t0 = np.arange(n) / source_fps
        t = np.arange(round(t0[-1] / cfg.policy_dt) + 1) * cfg.policy_dt
        original = np.c_[
            src["body_pos_w"][:, 0], src["body_quat_w"][:, 0], src["joint_pos"]
        ]
    hq = np.stack([np.interp(t, t0, original[:, i]) for i in range(human.nq)], axis=1)
    hq[:, 3:7] = Slerp(t0, Rotation.from_quat(original[:, [4, 5, 6, 3]]))(t).as_quat()[
        :, [3, 0, 1, 2]
    ]
    sq, dq, _, _ = joint_map(human, combined, allow_missing=True)
    hfoot = human.body("right_ankle_roll_link").id
    mujoco.mj_forward(human, hd)
    mujoco.mj_forward(combined, cd)
    right = [human.geom(f"right_foot{i}_collision").id for i in range(1, 8)]
    points = hd.geom_xpos[right].copy()
    points[:, 2] -= human.geom_size[right, 0] + human.geom_size[right, 1] * abs(
        hd.geom_xmat[right, 8]
    )
    offset = hd.xmat[hfoot].reshape(3, 3).T @ (points.mean(0) - hd.xpos[hfoot])
    align = hd.xmat[hfoot].reshape(3, 3).T @ sole(combined, cd)[1]
    klimits = combined.joint("knee").range.copy()
    al = combined.joint("ankle").range
    dlimits = np.array([_drive_for_ankle(float(x)) for x in al])
    lo = np.array([klimits[0], dlimits[0]])
    hi = np.array([klimits[1], dlimits[1]])
    body_names = [
        human.body(i).name
        for i in range(1, human.nbody)
        if human.body(i).name
        not in (
            "terrain",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
        )
    ]
    ids = [combined.body(name).id for name in body_names]
    qpos = np.empty((len(t), combined.nq))
    foot_pos = []
    foot_rot = []
    position_errors = []
    rotation_errors = []
    contact = []
    source_foot_positions = []
    limits_hit = []
    seed = np.array([0.3, 0.0])
    solutions = {}
    for f in range(len(t)):
        hd.qpos[:] = hq[f]
        mujoco.mj_kinematics(human, hd)
        hr = hd.xmat[hfoot].reshape(3, 3)
        target_pos = hd.xpos[hfoot] + hr @ offset
        target_rot = hr @ align
        cd.qpos[:] = combined.qpos0
        cd.qpos[dq] = hq[f, sq]
        previous = seed.copy()

        def residual(
            x, target_pos=target_pos, target_rot=target_rot, previous=previous
        ):
            set_prosthesis(combined, cd, *x)
            mujoco.mj_kinematics(combined, cd)
            pos, rot = sole(combined, cd)
            return np.r_[
                (pos - target_pos) / 0.1,
                Rotation.from_matrix(rot @ target_rot.T).as_rotvec() / 0.5,
                0.02 * (x - previous),
            ]

        # Warm starts preserve the assembly branch and temporal continuity.
        # Translating an identical gait pose does not change its optimal angles.
        key = np.round(hq[f, 2:], 6).tobytes()
        if key not in solutions:
            result = least_squares(
                residual,
                np.clip(seed, lo + 1e-9, hi - 1e-9),
                bounds=(lo, hi),
                max_nfev=40,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            solutions[key] = result.x.copy()
        seed = solutions[key]
        residual(seed)
        pos, rot = sole(combined, cd)
        qpos[f] = cd.qpos
        foot_pos.append(pos)
        foot_rot.append(rot)
        position_errors.append(np.linalg.norm(pos - target_pos))
        rotation_errors.append(
            np.linalg.norm(Rotation.from_matrix(rot @ target_rot.T).as_rotvec())
        )
        limits_hit.append(bool(np.any(np.minimum(seed - lo, hi - seed) < 1e-4)))
        bottoms = (
            hd.geom_xpos[right, 2]
            - human.geom_size[right, 0]
            - human.geom_size[right, 1] * abs(hd.geom_xmat[right, 8])
        )
        contact.append(float(bottoms.min()))
        source_foot_positions.append(target_pos.copy())
        if f % 2000 == 0:
            print(f"Reference adaptation: {f + 1}/{len(t)}", flush=True)
    foot_pos = np.asarray(foot_pos)
    foot_rot = np.asarray(foot_rot)
    # Kinematic references have no measured GRFs and may float above the floor.
    # Infer stance from low foot speed and low height relative to this recording.
    heights = np.asarray(contact)
    contact_height = float(np.percentile(heights, 10) + 0.04)
    source_velocity = np.gradient(source_foot_positions, cfg.policy_dt, axis=0)
    contact = (heights < contact_height) & (
        np.linalg.norm(source_velocity[:, :2], axis=1) < 0.6
    )
    qvel = np.empty((len(t), combined.nv))
    for f in range(len(t)):
        a, b = max(0, f - 1), min(len(t) - 1, f + 1)
        mujoco.mj_differentiatePos(
            combined, qvel[f], (b - a) * cfg.policy_dt, qpos[a], qpos[b]
        )
    arrays = {
        k: np.empty((len(t), len(ids), d), np.float32)
        for k, d in [
            ("body_pos", 3),
            ("body_rot", 9),
            ("body_vel", 3),
            ("body_ang_vel", 3),
        ]
    }
    roots = combined.body_rootid[ids]
    for f in range(len(t)):
        cd.qpos[:], cd.qvel[:] = qpos[f], qvel[f]
        mujoco.mj_forward(combined, cd)
        arrays["body_pos"][f] = cd.xpos[ids]
        arrays["body_rot"][f] = cd.xmat[ids]
        arrays["body_ang_vel"][f] = cd.cvel[ids, :3]
        arrays["body_vel"][f] = cd.cvel[ids, 3:] + np.cross(
            cd.cvel[ids, :3], cd.xpos[ids] - cd.subtree_com[roots]
        )
    arrays["body_rot"] = arrays["body_rot"].reshape(len(t), len(ids), 3, 3)
    report = {
        "frames": len(t),
        "duration_seconds": float(t[-1]),
        "mean_foot_position_error_m": float(np.mean(position_errors)),
        "p95_foot_position_error_m": float(np.percentile(position_errors, 95)),
        "max_foot_position_error_m": float(np.max(position_errors)),
        "mean_foot_orientation_error_deg": float(np.degrees(np.mean(rotation_errors))),
        "max_foot_orientation_error_deg": float(np.degrees(np.max(rotation_errors))),
        "joint_limit_fraction": float(np.mean(limits_hit)),
        "position_scale_m": 0.1,
        "orientation_scale_rad": 0.5,
        "inferred_stance_fraction": float(contact.mean()),
        "contact_height_threshold_m": contact_height,
        "contact_speed_threshold_m_s": 0.6,
    }
    output = Path(cfg.reference_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        foot_pos=foot_pos.astype(np.float32),
        foot_rot=foot_rot.astype(np.float32),
        foot_vel=np.gradient(foot_pos, cfg.policy_dt, axis=0).astype(np.float32),
        contact=contact,
        body_names=body_names,
        fps=[1 / cfg.policy_dt],
        source_sha256=fingerprint(cfg.motion_path),
        model_sha256=model_fingerprint(),
        fit_position_errors=position_errors,
        fit_orientation_errors=rotation_errors,
        **arrays,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report
