"""Replay the retargeted H1-with-OSL reference motion in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets.robots.h1_with_osl.h1_with_osl_constants import use_h1_with_osl_spec
from assets.robots.unitree_h1_2.h1_2_constants import get_h1_2_robot_cfg

from _view_mujoco_scene import _build_scene


DEFAULT_CLIP = (
    ROOT / "assets/reference_clips/short_reference_gait_h1_with_osl.npz"
)


def _load_clip(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as clip:
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
        }
        missing = required.difference(clip.files)
        if missing:
            raise ValueError(f"Clip is missing fields: {sorted(missing)}")
        return {key: clip[key].copy() for key in required}


def _apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: dict[str, np.ndarray],
    frame: int,
    pelvis_index: int,
) -> None:
    joint_pos = clip["joint_pos"]
    joint_vel = clip["joint_vel"]
    if joint_pos.shape[1] != model.nq - 7:
        raise ValueError(
            f"Clip has {joint_pos.shape[1]} joints, model expects {model.nq - 7}"
        )

    data.qpos[:3] = clip["body_pos_w"][frame, pelvis_index]
    data.qpos[3:7] = clip["body_quat_w"][frame, pelvis_index]
    data.qpos[7:] = joint_pos[frame]
    data.qvel[:3] = clip["body_lin_vel_w"][frame, pelvis_index]
    data.qvel[3:6] = clip["body_ang_vel_w"][frame, pelvis_index]
    data.qvel[6:] = joint_vel[frame]
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    clip = _load_clip(args.clip)
    frames = len(clip["joint_pos"])
    if not 0 <= args.start_frame < frames:
        parser.error(f"--start-frame must be between 0 and {frames - 1}")

    body_names = [str(name) for name in clip["body_names"]]
    try:
        pelvis_index = body_names.index("pelvis")
    except ValueError as exc:
        raise ValueError("Clip body_names does not contain 'pelvis'") from exc

    robot_cfg = use_h1_with_osl_spec(get_h1_2_robot_cfg())
    model = _build_scene(robot_cfg)
    data = mujoco.MjData(model)
    _apply_frame(model, data, clip, args.start_frame, pelvis_index)

    fps = float(clip["fps"][0])
    print(
        f"clip={args.clip} frames={frames} fps={fps:g} "
        f"duration={(frames - 1) / fps:.2f}s"
    )

    if args.headless:
        for offset in range(args.steps):
            frame = (args.start_frame + offset) % frames
            _apply_frame(model, data, clip, frame, pelvis_index)
        print(f"headless replay passed ({args.steps} frames)")
        return

    from mujoco import viewer

    controls = {"paused": False, "restart": False}

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            controls["paused"] = not controls["paused"]
        elif keycode in (ord("R"), ord("r")):
            controls["restart"] = True

    frame = args.start_frame
    frame_period = 1.0 / (fps * args.speed)
    with viewer.launch_passive(model, data, key_callback=key_callback) as handle:
        pelvis_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis"
        )
        if pelvis_id >= 0:
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            handle.cam.trackbodyid = pelvis_id
            handle.cam.distance = 3.0
            handle.cam.azimuth = 135.0
            handle.cam.elevation = -15.0

        next_frame_time = time.perf_counter()
        while handle.is_running():
            if controls["restart"]:
                frame = args.start_frame
                controls["restart"] = False
                next_frame_time = time.perf_counter()

            if controls["paused"]:
                handle.sync()
                time.sleep(0.01)
                next_frame_time = time.perf_counter()
                continue

            _apply_frame(model, data, clip, frame, pelvis_index)
            handle.sync()
            frame += 1
            if frame >= frames:
                if args.no_loop:
                    break
                frame = 0

            next_frame_time += frame_period
            time.sleep(max(0.0, next_frame_time - time.perf_counter()))


if __name__ == "__main__":
    main()
