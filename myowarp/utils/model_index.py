from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class JointIndex:
    name: str
    joint_id: int
    qpos_adr: int
    qvel_adr: int


@dataclass(frozen=True)
class ActuatorIndex:
    name: str
    actuator_id: int


@dataclass(frozen=True)
class SensorIndex:
    name: str
    sensor_id: int
    adr: int
    dim: int


@dataclass(frozen=True)
class ModelIndex:
    qpos: dict[str, int]
    qvel: dict[str, int]
    actuators: dict[str, int]
    sensors: dict[str, SensorIndex]
    muscle_actuator_ids: list[int]
    exo_actuator_ids: list[int]
    nu: int
    nq: int
    nv: int
    na: int


ROOT_FREEJOINT_ALIASES = {
    "pelvis_tx": (0, 0),
    "pelvis_tz": (1, 1),
    "pelvis_ty": (2, 2),
}


def apply_torso_lean(
    model: mujoco.MjModel,
    enable_lumbar_joint: bool,
    lumbar_joint_fixed_angle: float,
) -> None:
    """Lean the torso to match ``MyoAssistLegBase._setup``.

    When the lumbar joint is disabled and the model has no ``lumbar_extension``
    joint (the case for both the tutorial-22 and OSL_KA legs used here),
    myoassist sets ``body("torso").quat = [1, 0, 0, lumbar_joint_fixed_angle]``.
    The raw (unnormalized) value is kept identical so MuJoCo/MJWarp kinematics
    reproduce the same posture as the source pipeline.
    """
    if enable_lumbar_joint:
        return
    has_lumbar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lumbar_extension") >= 0
    if has_lumbar:
        # Neither model shipped here has a lumbar joint, so myoassist's
        # angle/range/damping fixing branch is intentionally not reproduced.
        return
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
        return
    model.body_quat[torso_id] = [1.0, 0.0, 0.0, float(lumbar_joint_fixed_angle)]


def _joint_index(model: mujoco.MjModel, name: str) -> JointIndex:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        if name in ROOT_FREEJOINT_ALIASES:
            qpos_adr, qvel_adr = ROOT_FREEJOINT_ALIASES[name]
            return JointIndex(name=name, joint_id=-1, qpos_adr=qpos_adr, qvel_adr=qvel_adr)
        raise KeyError(f"Joint not found in model: {name}")
    return JointIndex(
        name=name,
        joint_id=int(joint_id),
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        qvel_adr=int(model.jnt_dofadr[joint_id]),
    )


def _sensor_index(model: mujoco.MjModel, name: str) -> SensorIndex:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"Sensor not found in model: {name}")
    return SensorIndex(
        name=name,
        sensor_id=int(sensor_id),
        adr=int(model.sensor_adr[sensor_id]),
        dim=int(model.sensor_dim[sensor_id]),
    )


def build_model_index(
    model: mujoco.MjModel,
    qpos_keys: list[str],
    qvel_keys: list[str],
    sensor_keys: list[str],
) -> ModelIndex:
    qpos = {name: _joint_index(model, name).qpos_adr for name in qpos_keys}
    qvel = {name: _joint_index(model, name).qvel_adr for name in qvel_keys}
    sensors = {name: _sensor_index(model, name) for name in sensor_keys}
    actuators = {}
    muscle_actuator_ids: list[int] = []
    exo_actuator_ids: list[int] = []

    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name is None:
            name = f"actuator_{actuator_id}"
        actuators[name] = actuator_id
        if model.na > 0:
            if actuator_id < model.na:
                muscle_actuator_ids.append(actuator_id)
            else:
                exo_actuator_ids.append(actuator_id)
        elif name.startswith("Exo_") or name.startswith("osl_"):
            exo_actuator_ids.append(actuator_id)
        else:
            muscle_actuator_ids.append(actuator_id)

    return ModelIndex(
        qpos=qpos,
        qvel=qvel,
        actuators=actuators,
        sensors=sensors,
        muscle_actuator_ids=muscle_actuator_ids,
        exo_actuator_ids=exo_actuator_ids,
        nu=int(model.nu),
        nq=int(model.nq),
        nv=int(model.nv),
        na=int(model.na),
    )
