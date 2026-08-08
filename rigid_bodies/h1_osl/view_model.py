#!/usr/bin/env python3
"""Open the H1-OSL MuJoCo model in the interactive viewer."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


DEFAULT_XML = Path(__file__).with_name("scene.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "xml",
        nargs="?",
        type=Path,
        default=DEFAULT_XML,
        help="MuJoCo XML file to open (default: h1_osl/scene.xml)",
    )
    args = parser.parse_args()

    xml_path = args.xml.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"Opening {xml_path}")
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
