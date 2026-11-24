"""Real-state sensing and tracking reward; never reads the fooled shadow."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import body_velocities, sole

SENSOR_FRAME_DIM = 29

SENSOR_FIELDS = (
    "pelvis_orientation_6d",
    "thigh_orientation_6d",
    "pelvis_angular_velocity",
    "thigh_angular_velocity",
    "pelvis_gravity",
    "knee_ankle_position_velocity",
    "prosthetic_toe_heel_contacts",
    "previous_normalized_actions",
)


def contacts(model, data):
    floor = model.geom("floor").id
    load = np.zeros(3)  # left foot, prosthetic toe, prosthetic heel
    unsafe = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
        if g is None:
            unsafe += 1
            continue
        name = model.geom(g).name
        body = model.body(model.geom_bodyid[g]).name
        slot = (
            0
            if body in {"calcn_l", "toes_l"}
            else 1
            if name == "toe"
            else 2
            if name == "heel"
            else None
        )
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if slot is not None:
            load[slot] += max(float(force[0]), 0.0)
        elif force[0] > 5:
            unsafe += 1
    return load, unsafe


def sensor_frame(model, data, previous_action, cfg):
    ids = [model.body(n).id for n in ("pelvis", "femur_r")]
    velocity = body_velocities(model, data, ids)
    rotations = data.xmat[ids].reshape(2, 3, 3)
    q = [data.qpos[model.joint(n).qposadr[0]] for n in ("knee", "ankle")]
    v = [data.qvel[model.joint(n).dofadr[0]] for n in ("knee", "ankle")]
    load, _ = contacts(model, data)
    return np.concatenate(
        [
            rotations[0, :, :2].reshape(-1),
            rotations[1, :, :2].reshape(-1),
            velocity[:, 3:].reshape(-1),
            rotations[0].T @ np.array([0.0, 0.0, -1.0]),
            q,
            v,
            (load[1:] > cfg.sensor_contact_threshold).astype(float),
            previous_action,
        ]
    ).astype(np.float32)


def angle_error(a, b):
    delta = np.asarray(a) @ np.swapaxes(b, -1, -2)
    return np.linalg.norm(
        Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
    )


def tracking_reward(model, data, reference, frame, action, previous_action, cfg):
    ids = [model.body(str(name)).id for name in reference["body_names"]]
    positions = data.xpos[ids]
    rotations = data.xmat[ids].reshape(-1, 3, 3)
    velocity = body_velocities(model, data, ids)
    p, r = sole(model, data)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, p, model.body("ankle_foot").id)
    foot_velocity = jacp @ data.qvel
    target_p, target_r = reference["body_pos"][frame], reference["body_rot"][frame]
    # Root translation remains global, while segment positions are pelvis-relative.
    rel_error = (positions - positions[:1]) - (target_p - target_p[:1])
    load, unsafe = contacts(model, data)
    bw = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    expected_contact = bool(reference["contact"][frame])
    real_contact = bool(load[1:].sum() > cfg.sensor_contact_threshold)
    below, above = [], []
    for name in ("knee", "ankle"):
        joint = model.joint(name).id
        q = data.qpos[model.jnt_qposadr[joint]]
        lo, hi = model.jnt_range[joint]
        below.append(max(lo - q, 0.0))
        above.append(max(q - hi, 0.0))
    motors = [
        model.actuator(n).id for n in ("knee_act", "ankle_act", "knee_vel", "ankle_vel")
    ]
    force = np.abs(data.actuator_force[motors])
    limits = np.max(np.abs(model.actuator_forcerange[motors]), axis=1)
    power = abs(float(data.actuator_force[motors] @ data.actuator_velocity[motors]))
    terms = {
        "root_pos": np.exp(-np.sum((positions[0] - target_p[0]) ** 2) / 0.3**2),
        "root_ori": np.exp(
            -(angle_error(rotations[:1], target_r[:1])[0] ** 2) / 0.4**2
        ),
        "root_vel": np.exp(
            -np.sum((velocity[0, :3] - reference["body_vel"][frame, 0]) ** 2)
        ),
        "body_pos": np.exp(-np.mean(np.sum(rel_error**2, axis=-1)) / 0.3**2),
        "body_ori": np.exp(-np.mean(angle_error(rotations, target_r) ** 2) / 0.4**2),
        "body_vel": np.exp(
            -np.mean(
                np.sum((velocity[:, :3] - reference["body_vel"][frame]) ** 2, axis=-1)
            )
        ),
        "body_ang_vel": np.exp(
            -np.mean(
                np.sum(
                    (velocity[:, 3:] - reference["body_ang_vel"][frame]) ** 2, axis=-1
                )
            )
            / np.pi**2
        ),
        "foot_pos": np.exp(-np.sum((p - reference["foot_pos"][frame]) ** 2) / 0.15**2),
        "foot_ori": np.exp(
            -(angle_error(r[None], reference["foot_rot"][frame][None])[0] ** 2) / 0.5**2
        ),
        "foot_vel": np.exp(
            -np.sum((foot_velocity - reference["foot_vel"][frame]) ** 2)
        ),
        "contact": float(real_contact == expected_contact),
        "support": float(expected_contact)
        * min(load[1:].sum() / max(0.3 * bw, 1.0), 1.0),
        "slip": float(real_contact) * np.sum(foot_velocity[:2] ** 2),
        "impact": max(load[1:].sum() / bw - 1.5, 0.0) ** 2,
        "action_rate": np.sum((action - previous_action) ** 2),
        "effort": power / 100.0,
        "saturation": np.sum((force / np.maximum(limits, 1.0)) > 0.95),
        "joint_limit": sum(below) + sum(above),
        "unsafe_contact": unsafe,
    }
    reward = (
        sum(cfg.weights[name] * value for name, value in terms.items()) * cfg.policy_dt
    )
    return float(reward), {k: float(v) for k, v in terms.items()}
