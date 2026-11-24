"""View the retained intact H1-2 model with native MuJoCo."""

import argparse
import sys
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from myowarp.tasks.h1_2_tracking.model import build_h1_tracking_model, make_h1_home_qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model = build_h1_tracking_model(ROOT / "assets/robots/unitree_h1_2/xmls/h1_2.xml")
    data = mujoco.MjData(model)
    data.qpos[:] = make_h1_home_qpos(model)
    mujoco.mj_forward(model, data)
    print(f"H1-2: nq={model.nq}, nv={model.nv}, actuators={model.nu}")
    if args.headless:
        return
    from mujoco import viewer

    with viewer.launch_passive(model, data) as window:
        window.cam.lookat[:] = [0, 0, 0.9]
        window.cam.distance = 3
        window.cam.azimuth = 135
        window.cam.elevation = -15
        while window.is_running():
            window.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
