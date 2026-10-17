"""Load the CMU prosthetic leg in MuJoCo with an interactive free camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "assets" / "robots" / "cmu_leg" / "prosthesis.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="load and step without a viewer"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="simulation steps in headless mode"
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"loaded: {MODEL_XML}\nnq={model.nq} nv={model.nv} nu={model.nu}")

    if args.headless:
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError("simulation produced non-finite joint state")
        print(f"headless smoke test passed ({args.steps} steps)")
        return

    from mujoco import viewer

    with viewer.launch_passive(model, data) as window:
        mujoco.mjv_defaultFreeCamera(model, window.cam)
        while window.is_running():
            started = time.monotonic()
            # Advance a display frame at a time; the model uses a 0.2 ms timestep.
            for _ in range(max(1, round(1 / (60 * model.opt.timestep)))):
                mujoco.mj_step(model, data)
            window.sync()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
