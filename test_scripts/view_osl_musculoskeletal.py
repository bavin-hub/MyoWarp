"""Open the OSL with the full musculoskeletal model in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "assets" / "robots" / "osl" / "myolegs_OSL_KA.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="load and step the model without opening the viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of simulation steps in headless mode (default: 10)",
    )
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    stand_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, stand_key)
    mujoco.mj_forward(model, data)

    print(
        f"loaded: {MODEL_XML}\n"
        f"nq={model.nq} nv={model.nv} nu={model.nu} "
        f"muscles={model.nu - 2} osl_motors=2"
    )

    if args.headless:
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        print(f"headless smoke test passed ({args.steps} steps)")
        return

    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
    main()
