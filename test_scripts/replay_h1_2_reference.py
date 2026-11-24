"""View an existing H1-2 reference segment kinematically, without a policy."""

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=202.5)
    parser.add_argument("--end", type=float, default=208.5)
    parser.add_argument(
        "--motion", type=Path, help="Replay a separate tracking NPZ in full"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with np.load(
        args.motion or ROOT / "assets/reference_clips/walk1_subject1_h1_2.npz"
    ) as ref:
        fps = float(ref["fps"].item())
        if args.motion:
            args.start, args.end = 0, (len(ref["joint_pos"]) - 1) / fps
        if not 0 <= args.start < args.end <= (len(ref["joint_pos"]) - 1) / fps:
            parser.error("Segment must lie within the reference")
        segment = slice(round(args.start * fps), round(args.end * fps))
        joints = ref["joint_pos"][segment].copy()
        root_pos = ref["body_pos_w"][segment, 0].copy()
        root_quat = ref["body_quat_w"][segment, 0].copy()
        names = ref["joint_names"].tolist()
    root_pos[:, :2] -= root_pos[0, :2].copy()
    scene = mujoco.MjSpec.from_file(str(ROOT / "assets/unitree_viewer_scene.xml"))
    ground = scene.texture("ground_grid")
    ground.rgb1 = [0.65, 0.65, 0.65]
    ground.rgb2 = [0.4, 0.4, 0.4]
    scene.visual.headlight.ambient = [0.5, 0.5, 0.5]
    scene.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    scene.worldbody.add_light(
        pos=[0, -3, 4],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.6, 0.6, 0.6],
        castshadow=False,
    )
    robot = mujoco.MjSpec.from_file(
        str(ROOT / "assets/robots/unitree_h1_2/xmls/h1_2.xml")
    )
    scene.attach(robot, prefix="", frame=scene.worldbody.add_frame())
    model = scene.compile()
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in names]

    def set_frame(frame):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[addresses] = joints[frame]
        mujoco.mj_forward(model, data)

    if args.check:
        for frame in range(len(joints)):
            set_frame(frame)
            if not np.isfinite(data.xpos).all():
                raise RuntimeError("Non-finite reference pose")
        print(
            f"Validated {len(joints)} frames ({len(joints) / fps:.1f}s); no reference files changed"
        )
        return
    from mujoco import viewer

    set_frame(0)
    with viewer.launch_passive(model, data) as window:
        window.cam.distance = 3.5
        window.cam.azimuth = 135
        window.cam.elevation = -15
        start = time.monotonic()
        while window.is_running():
            tick = time.monotonic()
            frame = int((tick - start) * fps) % len(joints)
            set_frame(frame)
            window.cam.lookat[:] = data.qpos[:3]
            window.sync()
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))


if __name__ == "__main__":
    main()
