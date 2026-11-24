"""Static rotatable H1-2 + CMU leg view; preserves both source leg dimensions."""

import argparse
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/robots/h1_with_cmu_leg/h1_with_cmu_leg.xml")
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Place both soles on the floor (the XML includes the mounting adapter).
    foot = model.geom("left_foot4_collision").id
    root = model.joint("floating_base_joint").qposadr[0]
    data.qpos[root + 2] -= data.geom_xpos[foot, 2] - model.geom_size[foot, 0]
    mujoco.mj_forward(model, data)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint") == -1
    print(
        f"H1 + CMU: {model.nbody - 1} bodies, {model.njnt} joints, {model.nu} actuators"
    )
    soles = [
        data.geom_xpos[model.geom(n).id, 2] - model.geom_size[model.geom(n).id, 0]
        for n in ("heel", "toe", "left_foot4_collision")
    ]
    if any(abs(z) > 1e-7 for z in soles):
        raise RuntimeError(f"Feet are not at ground level: {soles}")
    print("Both H1 left foot and CMU heel/toe touch the ground in this pose.")
    if args.headless:
        return
    from mujoco import viewer as mj_viewer

    print("Static view: left drag rotates, right drag pans, scroll zooms.")
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.95]
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
