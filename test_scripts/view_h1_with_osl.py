"""Open the MJLab-configured H1-2 with an OSL right leg."""

from pathlib import Path
import sys

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets.robots.h1_with_osl.h1_with_osl_constants import use_h1_with_osl_spec
from assets.robots.unitree_h1_2.h1_2_constants import get_h1_2_robot_cfg

from _view_mujoco_scene import view_scene


def get_upright_robot_cfg():
    cfg = use_h1_with_osl_spec(get_h1_2_robot_cfg())
    cfg.init_state.pos = (0.0, 0.0, 0.0)
    robot_spec_fn = cfg.spec_fn

    def get_pinned_spec() -> mujoco.MjSpec:
        spec = robot_spec_fn()
        free_joint = next(
            joint
            for joint in spec.joints
            if joint.type == mujoco.mjtJoint.mjJNT_FREE
        )
        spec.delete(free_joint)
        return spec

    cfg.spec_fn = get_pinned_spec
    return cfg


if __name__ == "__main__":
    view_scene(
        get_upright_robot_cfg,
        "H1-2 with OSL",
    )
