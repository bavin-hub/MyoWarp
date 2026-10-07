"""Build the task's MuJoCo model without depending on MJLab."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


SOCKET_JOINT_PREFIX = "socket_"


@dataclass(frozen=True)
class ActuatorGroup:
    patterns: tuple[str, ...]
    stiffness: float
    damping: float
    effort_limit: float
    armature: float


ACTUATOR_GROUPS = (
    ActuatorGroup(
        patterns=(r".*_hip_yaw.*", r".*_hip_pitch.*", r".*_hip_roll.*", r"torso_joint"),
        stiffness=98.7,
        damping=6.3,
        effort_limit=200.0,
        armature=0.025,
    ),
    ActuatorGroup(
        patterns=(r"left_knee_joint", r"osl_knee_angle_r"),
        stiffness=157.7,
        damping=10.1,
        effort_limit=300.0,
        armature=0.04,
    ),
    ActuatorGroup(
        patterns=(
            r".*_ankle_pitch.*",
            r".*_ankle_roll.*",
            r"osl_ankle_angle_r",
            r".*_shoulder_pitch.*",
            r".*_shoulder_roll.*",
        ),
        stiffness=19.7,
        damping=1.3,
        effort_limit=40.0,
        armature=0.005,
    ),
    ActuatorGroup(
        patterns=(
            r".*_shoulder_yaw.*",
            r".*_elbow.*",
            r".*_wrist_pitch.*",
            r".*_wrist_roll.*",
            r".*_wrist_yaw.*",
        ),
        stiffness=7.9,
        damping=0.5,
        effort_limit=18.0,
        armature=0.002,
    ),
)


HOME_JOINT_POS = {
    ".*_hip_pitch_joint": -0.2,
    "left_knee_joint": 0.5,
    "osl_knee_angle_r": 0.5,
    ".*_ankle_pitch_joint": -0.3,
    "osl_ankle_angle_r": 0.3,
    ".*_shoulder_pitch_joint": 0.28,
    ".*_elbow_joint": 0.52,
}


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.fullmatch(pattern, name) for pattern in patterns)


def _add_position_actuator(
    spec: mujoco.MjSpec,
    joint_name: str,
    group: ActuatorGroup,
) -> None:
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = group.stiffness
    actuator.biasprm[1] = -group.stiffness
    actuator.biasprm[2] = -group.damping
    actuator.inheritrange = 0.0
    actuator.ctrllimited = False
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-group.effort_limit, group.effort_limit])
    joint = spec.joint(joint_name)
    delta = group.effort_limit / group.stiffness
    actuator.ctrlrange[:] = np.array(
        [joint.range[0] - delta, joint.range[1] + delta]
    )
    joint.armature = group.armature
    joint.frictionloss = 0.0


def _configure_collisions(spec: mujoco.MjSpec) -> None:
    for geom in spec.geoms:
        name = geom.name or ""
        if name == "terrain":
            geom.contype = 1
            geom.conaffinity = 1
            geom.condim = 3
            continue
        if not name.endswith("_collision"):
            geom.contype = 0
            geom.conaffinity = 0
            continue
        geom.contype = 1
        geom.conaffinity = 1
        is_foot = bool(re.fullmatch(r"(left|right)_foot[1-7]_collision", name))
        geom.condim = 3 if is_foot else 1
        geom.priority = 1 if is_foot else 0
        if is_foot:
            geom.friction[0] = 0.6


def _add_contact_sensor(
    spec: mujoco.MjSpec,
    *,
    name: str,
    objtype: mujoco.mjtObj,
    objname: str,
    reftype: mujoco.mjtObj,
    refname: str,
    field_index: int,
    reduce_mode: int,
) -> None:
    spec.add_sensor(
        name=name,
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=objtype,
        objname=objname,
        reftype=reftype,
        refname=refname,
        intprm=[1 << field_index, reduce_mode, 1],
    )


def _add_task_contact_sensors(spec: mujoco.MjSpec, *, include_feet: bool) -> None:
    if include_feet:
        for side, body in (
            ("left", "left_ankle_roll_link"),
            ("right", "osl_foot_assembly"),
        ):
            _add_contact_sensor(
                spec,
                name=f"feet_ground_{side}_found",
                objtype=mujoco.mjtObj.mjOBJ_XBODY,
                objname=body,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,
                refname="terrain",
                field_index=0,
                reduce_mode=3,
            )
            _add_contact_sensor(
                spec,
                name=f"feet_ground_{side}_force",
                objtype=mujoco.mjtObj.mjOBJ_XBODY,
                objname=body,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,
                refname="terrain",
                field_index=1,
                reduce_mode=3,
            )
    for field_name, field_index in (("found", 0), ("force", 1)):
        _add_contact_sensor(
            spec,
            name=f"self_collision_{field_name}",
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname="pelvis",
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname="pelvis",
            field_index=field_index,
            reduce_mode=0,
        )


def _home_position(name: str) -> float:
    for pattern, value in HOME_JOINT_POS.items():
        if re.fullmatch(pattern, name):
            return value
    return 0.0


def make_h1_with_osl_home_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Return the task home pose without changing MuJoCo's joint references."""
    home_qpos = model.qpos0.copy()
    home_qpos[:7] = np.array([0.0, 0.0, 1.02, 1.0, 0.0, 0.0, 0.0])
    for joint_id in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        home_qpos[model.jnt_qposadr[joint_id]] = _home_position(name)
    return home_qpos


def _make_flat_scene_spec(model_path: str | Path) -> mujoco.MjSpec:
    """Assemble the robot after a fixed terrain body, matching MJLab's scene."""
    robot_path = Path(model_path).resolve()
    sibling_robot_path = robot_path.with_name("h1_with_osl.xml")
    if robot_path.name == "scene.xml" and sibling_robot_path.exists():
        robot_path = sibling_robot_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))

    viewer_scene_path = robot_path.parents[2] / "unitree_viewer_scene.xml"
    if viewer_scene_path.exists():
        scene_spec = mujoco.MjSpec.from_file(str(viewer_scene_path))
        terrain_geom = next(
            (geom for geom in scene_spec.geoms if geom.name == "terrain"), None
        )
        if terrain_geom is not None:
            scene_spec.delete(terrain_geom)
        terrain_body = scene_spec.worldbody.add_body(name="terrain")
        terrain_body.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.01),
            material="ground_grid",
            condim=3,
        )
    else:
        scene_spec = mujoco.MjSpec()
        terrain_body = scene_spec.worldbody.add_body(name="terrain")
        terrain_body.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0.0, 0.0, 0.01),
            condim=3,
        )
    scene_spec.worldbody.add_site(
        name="env_origin_0",
        pos=(0.0, 0.0, 0.0),
        size=(0.3, 0.3, 0.3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=(0.2, 0.6, 0.2, 0.3),
        group=4,
    )
    frame = scene_spec.worldbody.add_frame()
    scene_spec.attach(robot_spec, prefix="", frame=frame)
    return scene_spec


def _build_h1_with_osl_model(
    model_path: str | Path,
    *,
    include_foot_sensors: bool,
    physics_dt: float = 0.005,
    solver_iterations: int = 10,
    solver_ls_iterations: int = 20,
    ccd_iterations: int = 50,
) -> mujoco.MjModel:
    """Compile the same flat-terrain model previously assembled by MJLab."""
    spec = _make_flat_scene_spec(model_path)
    joint_names = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    for group in ACTUATOR_GROUPS:
        for name in joint_names:
            if _matches_any(name, group.patterns):
                _add_position_actuator(spec, name, group)

    _configure_collisions(spec)
    _add_task_contact_sensors(spec, include_feet=include_foot_sensors)
    model = spec.compile()
    model.opt.timestep = physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.impratio = 1.0
    model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
    model.opt.iterations = solver_iterations
    model.opt.tolerance = 1.0e-8
    model.opt.ls_iterations = solver_ls_iterations
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = ccd_iterations

    if model.nu != 26:
        raise RuntimeError(f"Expected 26 policy actuators, compiled model has {model.nu}")

    return model


def build_h1_with_osl_velocity_model(
    model_path: str | Path,
    *,
    physics_dt: float = 0.005,
    solver_iterations: int = 10,
    solver_ls_iterations: int = 20,
    ccd_iterations: int = 50,
) -> mujoco.MjModel:
    """Compile the flat-terrain model with velocity-task contact sensors."""
    return _build_h1_with_osl_model(
        model_path,
        include_foot_sensors=True,
        physics_dt=physics_dt,
        solver_iterations=solver_iterations,
        solver_ls_iterations=solver_ls_iterations,
        ccd_iterations=ccd_iterations,
    )


def build_h1_with_osl_tracking_model(
    model_path: str | Path,
    *,
    physics_dt: float = 0.005,
    solver_iterations: int = 10,
    solver_ls_iterations: int = 20,
    ccd_iterations: int = 50,
) -> mujoco.MjModel:
    """Compile the flat-terrain model with the tracking self-contact sensor."""
    return _build_h1_with_osl_model(
        model_path,
        include_foot_sensors=False,
        physics_dt=physics_dt,
        solver_iterations=solver_iterations,
        solver_ls_iterations=solver_ls_iterations,
        ccd_iterations=ccd_iterations,
    )


def names_for(model: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj, i) or "" for i in range(count)]
