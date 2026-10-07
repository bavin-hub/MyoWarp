"""Build an MJLab-configured robot and ground plane for MuJoCo viewing."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg


_SCENE_XML = Path(__file__).resolve().parents[1] / "assets" / "unitree_viewer_scene.xml"


def _build_scene(robot_cfg: EntityCfg) -> mujoco.MjModel:
    """Compose an MJLab-configured entity into the viewer scene."""
    robot = robot_cfg.build()
    scene_spec = mujoco.MjSpec.from_file(str(_SCENE_XML))

    key_qpos: np.ndarray | None = None
    key_ctrl: np.ndarray | None = None
    if robot.spec.keys:
        key = robot.spec.keys[0]
        key_qpos = np.array(key.qpos)
        key_ctrl = np.array(key.ctrl)
        robot.spec.delete(key)

    frame = scene_spec.worldbody.add_frame()
    scene_spec.attach(robot.spec, prefix="robot/", frame=frame)

    if key_qpos is not None and key_ctrl is not None:
        scene_spec.add_key(
            name="init_state",
            qpos=key_qpos.tolist(),
            ctrl=key_ctrl.tolist(),
        )

    return scene_spec.compile()


def view_scene(
    robot_cfg_factory: Callable[[], EntityCfg], robot_name: str
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Open the Unitree {robot_name} scene in MuJoCo."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="load and step the model without opening the viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of simulation steps to run in headless mode (default: 10)",
    )
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative")

    model = _build_scene(robot_cfg_factory())
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    if ground_id < 0 or model.geom_type[ground_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise RuntimeError("MJLab scene does not contain the expected ground plane")

    print(
        f"robot={robot_name} nq={model.nq} nv={model.nv} nu={model.nu} "
        f"mjlab_actuators=yes ground_plane=yes"
    )

    if args.headless:
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        print(f"headless smoke test passed ({args.steps} steps)")
        return

    from mujoco import viewer

    viewer.launch(model, data)
