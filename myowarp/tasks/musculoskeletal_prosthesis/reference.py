"""Offline constrained prosthetic-foot reference fit; original human clip stays unchanged."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from myowarp.musclemimic.bypass import _apply_couplings
from myowarp.musclemimic.model import FULLBODY_BODY2SITES_FOR_MIMIC
from myowarp.musclemimic.trajectory import expand_motion_clip_to_model, load_motion_clip

from .geometry import (
    _drive_for_ankle,
    angular_velocities,
    joint_map,
    set_prosthesis,
    sole,
)

EXCLUDED = {"tibia_r", "talus_r", "toes_r"}
TRACKED = tuple(n for n in FULLBODY_BODY2SITES_FOR_MIMIC if n not in EXCLUDED)


def adapter_hash():
    return hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("geometry.py").read_bytes()
    ).hexdigest()


def load_human_clip(path, model):
    clip = expand_motion_clip_to_model(
        load_motion_clip(
            Path(path).resolve(), expected_nq=model.nq, expected_nv=model.nv
        ),
        model,
    )
    if clip.frequency_hz is None or not np.isclose(clip.frequency_hz, 100):
        raise ValueError("Reference must specify 100 Hz")
    return clip


def biological_sole_offset(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    points = []
    for g in range(model.ngeom):
        if (
            model.body(model.geom_bodyid[g]).name in {"calcn_r", "toes_r"}
            and model.geom_contype[g]
        ):
            segment = np.zeros(6)
            mujoco.mj_geomDistance(model, data, floor, g, 100.0, segment)
            points.append(segment[3:])
    body = model.body("calcn_r").id
    return data.xmat[body].reshape(3, 3).T @ (np.mean(points, axis=0) - data.xpos[body])


def prepare_reference(cfg, human, combined, *, max_frames=None):
    clip = load_human_clip(cfg.motion_path, human)
    n = len(clip.qpos) if max_frames is None else min(len(clip.qpos), max_frames)
    if n < 3:
        raise ValueError("Need at least three reference frames")
    hd, cd = mujoco.MjData(human), mujoco.MjData(combined)
    sq, dq, _, _ = joint_map(human, combined, allow_missing=True)
    ids = [human.body(name).id for name in TRACKED]
    biological_foot = human.body("calcn_r").id
    offset = biological_sole_offset(human)
    mujoco.mj_forward(human, hd)
    mujoco.mj_resetDataKeyframe(combined, cd, combined.key("bypass_stand").id)
    mujoco.mj_forward(combined, cd)
    alignment = hd.xmat[biological_foot].reshape(3, 3).T @ sole(combined, cd)[1]
    qpos, positions, rotations, foot_pos, foot_rot, errors, knee_drive = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    contact = []
    seed = np.array([0.3, 0.0])
    knee_limits = combined.jnt_range[combined.joint("knee").id]
    ankle_limits = combined.jnt_range[combined.joint("ankle").id]
    drive_limits = [_drive_for_ankle(float(angle)) for angle in ankle_limits]
    floor = human.geom("floor").id
    foot_geoms = [
        g
        for g in range(human.ngeom)
        if human.geom_contype[g]
        and human.body(human.geom_bodyid[g]).name in {"calcn_r", "toes_r"}
    ]
    for frame in range(n):
        hd.qpos[:] = clip.qpos[frame]
        hd.qvel[:] = 0 if clip.qvel is None else clip.qvel[frame]
        mujoco.mj_forward(human, hd)
        target_rot = hd.xmat[biological_foot].reshape(3, 3) @ alignment
        target_pos = (
            hd.xpos[biological_foot] + hd.xmat[biological_foot].reshape(3, 3) @ offset
        )
        cd.qpos[:] = combined.qpos0
        cd.qpos[dq] = hd.qpos[sq]
        _apply_couplings(combined, cd)
        _apply_couplings(combined, cd)

        def residual(x, target_pos=target_pos, target_rot=target_rot):
            set_prosthesis(combined, cd, *x)
            mujoco.mj_forward(combined, cd)
            p, r = sole(combined, cd)
            return np.r_[
                (p - target_pos) / 0.1,
                Rotation.from_matrix(r @ target_rot.T).as_rotvec() / 0.5,
            ]

        result = least_squares(
            residual,
            seed,
            bounds=(
                [knee_limits[0], drive_limits[0]],
                [knee_limits[1], drive_limits[1]],
            ),
            max_nfev=30,
        )
        seed = result.x
        residual(seed)
        p, r = sole(combined, cd)
        qpos.append(cd.qpos.copy())
        positions.append(hd.xpos[ids].copy())
        rotations.append(hd.xmat[ids].reshape(-1, 3, 3).copy())
        foot_pos.append(p.copy())
        foot_rot.append(r.copy())
        errors.append(np.linalg.norm(p - target_pos))
        knee_drive.append(seed.copy())
        contact.append(
            any(
                mujoco.mj_geomDistance(human, hd, floor, g, 0.1, None) < 0.02
                for g in foot_geoms
            )
        )
        if frame % 500 == 0:
            print(f"Reference adaptation: {frame + 1}/{n} frames", flush=True)
    positions, rotations = np.array(positions), np.array(rotations)
    foot_pos, foot_rot, qpos = np.array(foot_pos), np.array(foot_rot), np.array(qpos)
    qvel = np.empty((n, combined.nv))
    for f in range(n):
        a, b = max(f - 1, 0), min(f + 1, n - 1)
        mujoco.mj_differentiatePos(
            combined, qvel[f], (b - a) * cfg.policy_dt, qpos[a], qpos[b]
        )
    payload = {
        "qpos": qpos,
        "qvel": qvel,
        "body_pos": positions,
        "body_rot": rotations,
        "body_vel": np.gradient(positions, cfg.policy_dt, axis=0),
        "body_ang_vel": angular_velocities(rotations, cfg.policy_dt),
        "foot_pos": foot_pos,
        "foot_rot": foot_rot,
        "foot_vel": np.gradient(foot_pos, cfg.policy_dt, axis=0),
        "foot_ang_vel": angular_velocities(foot_rot, cfg.policy_dt),
        "contact": np.array(contact),
        "fit_error_m": np.array(errors),
        "knee_drive": np.array(knee_drive),
        "body_names": np.array(TRACKED),
        "source_sha256": hashlib.sha256(Path(cfg.motion_path).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest(),
        "adapter_sha256": adapter_hash(),
        "policy_dt": cfg.policy_dt,
    }
    output = Path(cfg.reference_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return payload


def read_reference(cfg):
    with np.load(cfg.reference_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    for key, file in [
        ("source_sha256", cfg.motion_path),
        ("model_sha256", cfg.model_path),
    ]:
        if str(data[key]) != hashlib.sha256(Path(file).read_bytes()).hexdigest():
            raise ValueError("Reference cache is stale; rerun preparation")
    if not np.isclose(float(data["policy_dt"]), cfg.policy_dt):
        raise ValueError("Reference timestep mismatch")
    if str(data.get("adapter_sha256", "")) != adapter_hash():
        raise ValueError("Reference adapter changed; rerun preparation")
    return data
