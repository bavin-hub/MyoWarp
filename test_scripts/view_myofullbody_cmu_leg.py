"""Static, rotatable view of the whole-body musculoskeletal model with CMU leg."""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anatomy", choices=("constrained", "removed"), default="constrained"
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    import mujoco

    xml = (
        Path(__file__).resolve().parents[1]
        / "assets/robots/myofullbody_cmu_leg/myofullbody_cmu_leg.xml"
    )
    if args.anatomy == "removed":
        xml = (
            Path(__file__).resolve().parents[1]
            / "assets/robots/myofullbody_cmu_leg_no_lower_leg/myofullbody_cmu_leg_no_lower_leg.xml"
        )
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("bypass_stand").id)
    mujoco.mj_forward(model, data)
    print(f"Whole-body + CMU leg: {model.nbody - 1} bodies, {model.nu} actuators")
    if args.headless:
        return
    import mujoco.viewer

    print("Static model: left drag rotates, right drag pans, scroll zooms.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.xpos[model.body("pelvis").id]
        viewer.cam.distance = 3
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
