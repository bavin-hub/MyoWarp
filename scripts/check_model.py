from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml")
    args = parser.parse_args()

    path = Path(args.model)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"loaded: {path}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} na={model.na} nsensor={model.nsensor}")
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    print(f"first actuators: {actuator_names[:8]}")
    print(f"last actuators: {actuator_names[-4:]}")


if __name__ == "__main__":
    main()
