"""Intact 27-actuator H1-2 model using existing MJLab-equivalent helpers."""

import re
from dataclasses import replace
from pathlib import Path

import mujoco

from myowarp.tasks.h1_with_osl_velocity.model import (
    ACTUATOR_GROUPS,
    _add_position_actuator,
    _add_task_contact_sensors,
    _configure_collisions,
    _make_flat_scene_spec,
)


def build_h1_tracking_model(
    model_path,
    *,
    physics_dt=0.005,
    solver_iterations=10,
    solver_ls_iterations=20,
    ccd_iterations=50,
):
    spec = _make_flat_scene_spec(
        model_path,
        viewer_scene_path=Path(__file__).resolve().parents[3]
        / "assets/unitree_viewer_scene.xml",
    )
    groups = list(ACTUATOR_GROUPS)
    groups[1] = replace(groups[1], patterns=(r".*_knee_joint",))
    joints = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    for group in groups:
        for name in joints:
            if any(re.fullmatch(pattern, name) for pattern in group.patterns):
                _add_position_actuator(spec, name, group)
    _configure_collisions(spec)
    _add_task_contact_sensors(spec, include_feet=False)
    model = spec.compile()
    model.opt.timestep = physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.impratio = 1.0
    model.opt.gravity[:] = [0, 0, -9.81]
    model.opt.iterations = solver_iterations
    model.opt.tolerance = 1e-8
    model.opt.ls_iterations = solver_ls_iterations
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = ccd_iterations
    if model.nu != 27 or model.nq != 34:
        raise RuntimeError("Expected intact H1-2 with 27 actuators")
    return model


def make_h1_home_qpos(model):
    home = model.qpos0.copy()
    home[:7] = [0, 0, 1.02, 1, 0, 0, 0]
    patterns = {
        r".*_hip_pitch_joint": -0.2,
        r".*_knee_joint": 0.5,
        r".*_ankle_pitch_joint": -0.3,
        r".*_shoulder_pitch_joint": 0.28,
        r".*_elbow_joint": 0.52,
    }
    for j in range(1, model.njnt):
        name = model.joint(j).name
        for pattern, value in patterns.items():
            if re.fullmatch(pattern, name):
                home[model.jnt_qposadr[j]] = value
                break
    return home
