from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
    model_path = Path(__file__).with_name("scene.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
