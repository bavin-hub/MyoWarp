"""Native virtual/fooled shadows, preserving prosim's last-substep convention."""

import mujoco
import numpy as np

from .geometry import joint_map, muscle_map

LOCK_ONLY_MUSCLES = (
    "bfsh_r",
    "edl_r",
    "ehl_r",
    "fdl_r",
    "fhl_r",
    "gaslat_r",
    "gasmed_r",
    "perbrev_r",
    "perlong_r",
    "soleus_r",
    "tibant_r",
    "tibpost_r",
    "vasint_r",
    "vaslat_r",
    "vasmed_r",
)
VIEW_JOINTS = ("knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r")


def snapshot(data):
    return {
        key: getattr(data, key).copy()
        for key in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
    }


class FooledShadow:
    def __init__(self, human, combined, clip, mode="fooled"):
        if mode not in ("fooled", "virtual"):
            raise ValueError("Shadow mode must be fooled or virtual")
        self.mode = mode
        self.model, self.real_model, self.clip = human, combined, clip
        self.data = mujoco.MjData(human)
        self.sq, self.dq, self.sv, self.dv = joint_map(
            human, combined, allow_missing=True
        )
        (
            self.shadow_actuator_ids,
            self.actuator_ids,
            self.shadow_activation_ids,
            self.activation_ids,
        ) = muscle_map(human, combined)
        self.mask = np.array([human.actuator(name).id for name in LOCK_ONLY_MUSCLES])

    def _replace_leg(self, frame, source):
        m, d = self.model, self.data
        for i, name in enumerate(VIEW_JOINTS):
            j = m.joint(name).id
            if self.mode == "virtual":
                if i < 2:
                    real_joint = self.real_model.joint(("knee", "ankle")[i])
                    q, v = (
                        source["qpos"][real_joint.qposadr[0]],
                        source["qvel"][real_joint.dofadr[0]],
                    )
                else:
                    q, v = 0.0, 0.0
            else:
                q = self.clip.qpos[frame, m.jnt_qposadr[j]]
                v = (
                    0.0
                    if self.clip.qvel is None
                    else self.clip.qvel[frame, m.jnt_dofadr[j]]
                )
            d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]] = q, v
        knee = m.joint("knee_angle_r").id
        q, v = d.qpos[m.jnt_qposadr[knee]], d.qvel[m.jnt_dofadr[knee]]
        for i in range(m.neq):
            if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[i] == knee:
                j, coefficients = m.eq_obj1id[i], m.eq_data[i, :5]
                d.qpos[m.jnt_qposadr[j]] = sum(coefficients[k] * q**k for k in range(5))
                d.qvel[m.jnt_dofadr[j]] = (
                    sum(k * coefficients[k] * q ** (k - 1) for k in range(1, 5)) * v
                )

    def observe(self, real, frame, pre_substep):
        d = self.data
        for source in (pre_substep, snapshot(real)):
            d.qpos[:] = self.clip.qpos[frame]
            d.qvel[:] = 0 if self.clip.qvel is None else self.clip.qvel[frame]
            d.qpos[self.sq] = source["qpos"][self.dq]
            d.qvel[self.sv] = source["qvel"][self.dv]
            d.act[:] = 0
            d.ctrl[:] = 0
            d.act[self.shadow_activation_ids] = source["act"][self.activation_ids]
            d.ctrl[self.shadow_actuator_ids] = source["ctrl"][self.actuator_ids]
            self._replace_leg(frame, source)
            if source is pre_substep:
                d.qacc_warmstart[:] = 0
                d.qacc_warmstart[self.sv] = source["qacc_warmstart"][self.dv]
                mujoco.mj_forward(self.model, d)
        d.time = real.time
        return d

    def write_muscles(self, real, commands):
        u = np.asarray(commands, dtype=float).copy()
        if u.shape != (self.model.nu,) or not np.isfinite(u).all():
            raise ValueError(
                "Frozen muscle actor must return finite 354-action commands"
            )
        u = np.clip(
            u, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        u[self.mask] = 0.0
        real.ctrl[self.actuator_ids] = u[self.shadow_actuator_ids]
