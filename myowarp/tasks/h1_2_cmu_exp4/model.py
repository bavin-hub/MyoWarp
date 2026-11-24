"""Combined H1/CMU model with 24 H1 position and four CMU motor channels."""

import re
from pathlib import Path

import mujoco

from myowarp.tasks.h1_with_osl_velocity.model import (
    ACTUATOR_GROUPS,
    _add_position_actuator,
)

ROOT = Path(__file__).resolve().parents[3]


def build_model():
    robot = mujoco.MjSpec.from_file(
        str(ROOT / "assets/robots/h1_with_cmu_leg/h1_with_cmu_leg.xml")
    )
    robot.delete(robot.geom("floor"))
    for light in list(robot.lights):
        robot.delete(light)
    scene = mujoco.MjSpec.from_file(str(ROOT / "assets/unitree_viewer_scene.xml"))
    scene.attach(robot, prefix="", frame=scene.worldbody.add_frame())
    for joint in list(scene.joints):
        for group in ACTUATOR_GROUPS:
            if any(re.fullmatch(pattern, joint.name) for pattern in group.patterns):
                _add_position_actuator(scene, joint.name, group)
    # Keep CMU collision defaults; match H1 tracking contact settings on H1 geoms.
    for geom in scene.geoms:
        if geom.name.endswith("_collision"):
            geom.condim = 3 if geom.name.startswith("left_foot") else 1
            geom.contype = geom.conaffinity = 1
            if geom.name.startswith("left_foot"):
                geom.friction[0] = 0.6
    model = scene.compile()
    model.opt.timestep = 0.001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    assert model.nu == 28  # 24 H1 position actuators + four LLC motor channels
    return model
