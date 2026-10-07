"""Retarget the MyoAssist human gait clip to H1 with a right OSL leg.

The output follows MJLab's motion-tracking NPZ layout.  This is a joint-space
retarget: human anatomical coordinates are mapped to the closest robot DOFs,
then target body states are generated with MuJoCo forward kinematics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "myoassist/rl_train/reference_data/short_reference_gait.npz"
DEFAULT_OUTPUT = ROOT / "assets/reference_clips/short_reference_gait_h1_with_osl.npz"
MODEL_XML = ROOT / "assets/robots/h1_with_osl/h1_with_osl.xml"


def _resample(values: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    duration = (len(values) - 1) / input_fps
    input_time = np.arange(len(values), dtype=np.float64) / input_fps
    output_time = np.arange(0.0, duration, 1.0 / output_fps)
    return np.interp(output_time, input_time, values).astype(np.float64)


def _quat_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Return MuJoCo-order (w, x, y, z) quaternions."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.column_stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"Joint not found in target model: {name}")
    return joint_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    source = np.load(args.source, allow_pickle=True)
    metadata = source["metadata"].item()
    human_raw = source["series_data"].item()
    input_fps = float(metadata["sample_rate"])
    human = {
        key: _resample(np.asarray(value, dtype=np.float64), input_fps, args.fps)
        for key, value in human_raw.items()
        if key.startswith("q_")
    }
    frames = len(next(iter(human.values())))

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    qpos = np.tile(model.qpos0, (frames, 1)).astype(np.float64)

    # Root translation: OpenSim/MyoAssist uses tx forward, tz lateral, ty up.
    qpos[:, 0] = human["q_pelvis_tx"] - human["q_pelvis_tx"][0]
    qpos[:, 1] = human["q_pelvis_tz"] - human["q_pelvis_tz"][0]
    qpos[:, 2] = human["q_pelvis_ty"]
    qpos[:, 3:7] = _quat_from_rpy(
        human["q_pelvis_list"],
        human["q_pelvis_tilt"],
        human["q_pelvis_rotation"] - human["q_pelvis_rotation"][0],
    )

    # Anatomical-coordinate mapping.  Signs account for the robot joint axes.
    mapping: dict[str, tuple[str, float, float]] = {
        "left_hip_yaw_joint": ("q_hip_rotation_l", -1.0, 0.0),
        "left_hip_pitch_joint": ("q_hip_flexion_l", -1.0, 0.0),
        "left_hip_roll_joint": ("q_hip_adduction_l", -1.0, 0.0),
        "left_knee_joint": ("q_knee_angle_l", -1.0, 0.0),
        "left_ankle_pitch_joint": ("q_ankle_angle_l", -1.0, 0.0),
        "left_ankle_roll_joint": ("q_subtalar_angle_l", 1.0, 0.0),
        "right_hip_yaw_joint": ("q_hip_rotation_r", 1.0, 0.0),
        "right_hip_pitch_joint": ("q_hip_flexion_r", -1.0, 0.0),
        "right_hip_roll_joint": ("q_hip_adduction_r", 1.0, 0.0),
        "osl_knee_angle_r": ("q_knee_angle_r", -1.0, 0.0),
        "osl_ankle_angle_r": ("q_ankle_angle_r", 1.0, 0.0),
        "torso_joint": ("q_lumbar_rotation", 1.0, 0.0),
        "left_shoulder_pitch_joint": ("q_arm_flex_l", -1.0, 0.0),
        "left_shoulder_roll_joint": ("q_arm_add_l", -1.0, 0.0),
        "left_shoulder_yaw_joint": ("q_arm_rot_l", -1.0, 0.0),
        "left_elbow_joint": ("q_elbow_flex_l", 1.0, 0.0),
        "left_wrist_roll_joint": ("q_pro_sup_l", -1.0, np.pi / 2),
        "left_wrist_pitch_joint": ("q_wrist_flex_l", 1.0, 0.0),
        "left_wrist_yaw_joint": ("q_wrist_dev_l", 1.0, 0.0),
        "right_shoulder_pitch_joint": ("q_arm_flex_r", -1.0, 0.0),
        "right_shoulder_roll_joint": ("q_arm_add_r", 1.0, 0.0),
        "right_shoulder_yaw_joint": ("q_arm_rot_r", 1.0, 0.0),
        "right_elbow_joint": ("q_elbow_flex_r", 1.0, 0.0),
        "right_wrist_roll_joint": ("q_pro_sup_r", 1.0, -np.pi / 2),
        "right_wrist_pitch_joint": ("q_wrist_flex_r", 1.0, 0.0),
        "right_wrist_yaw_joint": ("q_wrist_dev_r", 1.0, 0.0),
    }

    for target_name, (source_name, scale, offset) in mapping.items():
        joint_id = _joint_id(model, target_name)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        values = scale * human[source_name] + offset
        if model.jnt_limited[joint_id]:
            values = np.clip(values, *model.jnt_range[joint_id])
        qpos[:, qpos_adr] = values

    # Passive socket-compliance joints remain neutral.
    passive_names = (
        "socket_piston",
        "socket_rotation_1",
        "socket_rotation_2",
        "socket_rotation_3",
    )
    for name in passive_names:
        qpos[:, int(model.jnt_qposadr[_joint_id(model, name)])] = 0.0

    # Shift the full clip just enough that neither target foot penetrates the plane.
    data = mujoco.MjData(model)
    foot_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            *(f"left_foot{i}_collision" for i in range(1, 8)),
            *(f"right_foot{i}_collision" for i in range(1, 4)),
        )
    ]
    lowest = np.empty(frames, dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        geom_bottoms = []
        for geom_id in foot_geom_ids:
            radius, half_length = model.geom_size[geom_id, :2]
            segment_vertical = abs(data.geom_xmat[geom_id, 8]) * half_length
            geom_bottoms.append(data.geom_xpos[geom_id, 2] - segment_vertical - radius)
        lowest[frame] = min(geom_bottoms)
    ground_offset = max(0.0, -float(lowest.min()))
    qpos[:, 2] += ground_offset

    dt = 1.0 / args.fps
    qvel = np.empty((frames, model.nv), dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel[0], dt, qpos[0], qpos[1])
    mujoco.mj_differentiatePos(model, qvel[-1], dt, qpos[-2], qpos[-1])
    for frame in range(1, frames - 1):
        mujoco.mj_differentiatePos(model, qvel[frame], 2.0 * dt, qpos[frame - 1], qpos[frame + 1])

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]
    body_pos_w = np.empty((frames, len(body_names), 3), dtype=np.float32)
    body_quat_w = np.empty((frames, len(body_names), 4), dtype=np.float32)
    body_lin_vel_w = np.empty((frames, len(body_names), 3), dtype=np.float32)
    body_ang_vel_w = np.empty((frames, len(body_names), 3), dtype=np.float32)
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        body_pos_w[frame] = data.xpos[1:]
        body_quat_w[frame] = data.xquat[1:]
        for body_index, body_id in enumerate(range(1, model.nbody)):
            angular = data.cvel[body_id, :3]
            offset = data.subtree_com[root_body_id] - data.xpos[body_id]
            body_ang_vel_w[frame, body_index] = angular
            body_lin_vel_w[frame, body_index] = (
                data.cvel[body_id, 3:] - np.cross(angular, offset)
            )

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    ]
    mapping_metadata = {
        target: {"source": source_name, "scale": scale, "offset": offset}
        for target, (source_name, scale, offset) in mapping.items()
    }
    mapping_metadata["root"] = {
        "position": "[pelvis_tx-relative, pelvis_tz-relative, pelvis_ty+ground_offset]",
        "orientation": "rpy=[pelvis_list, pelvis_tilt, pelvis_rotation-relative]",
        "ground_offset_m": ground_offset,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fps=np.asarray([args.fps], dtype=np.float64),
        joint_pos=qpos[:, 7:].astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        source_file=np.asarray(str(args.source)),
        source_fps=np.asarray([input_fps], dtype=np.float64),
        passive_joint_names=np.asarray(passive_names),
        mapping_json=np.asarray(json.dumps(mapping_metadata, sort_keys=True)),
    )
    print(f"Saved {frames} frames at {args.fps:g} Hz to {args.output}")
    print(f"Joint/body shapes: {qpos[:, 7:].shape} / {body_pos_w.shape}")
    print(f"Applied root height offset: {ground_offset:.4f} m")


if __name__ == "__main__":
    main()
