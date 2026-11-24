"""Observation-only ablations on a fully actuated biological body.

Physical muscles are never masked. The same 15 activation/excitation entries
that vanish in the bypass pipeline are suppressed only in the shadow.
"""

import mujoco

from myowarp.tasks.musculoskeletal_prosthesis.shadow import VIEW_JOINTS, FooledShadow


class WholeBodyObservationShadow(FooledShadow):
    def __init__(self, model, clip, mode):
        if mode not in ("baseline", "fooled", "virtual"):
            raise ValueError("Mode must be baseline, fooled or virtual")
        super().__init__(model, model, clip, mode="fooled")
        self.experiment_mode = mode

    def _replace_leg(self, frame, source):
        if self.experiment_mode == "baseline":
            return
        if self.experiment_mode == "fooled":
            super()._replace_leg(frame, source)
        else:
            # Knee/ankle already contain current biological state via joint_map.
            # Match bypass virtual mode's zero subtalar and toe coordinates.
            for name in VIEW_JOINTS[2:]:
                joint = self.model.joint(name)
                self.data.qpos[joint.qposadr[0]] = 0
                self.data.qvel[joint.dofadr[0]] = 0
            knee = self.model.joint("knee_angle_r")
            q, v = self.data.qpos[knee.qposadr[0]], self.data.qvel[knee.dofadr[0]]
            for i in range(self.model.neq):
                if (
                    self.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
                    and self.model.eq_obj2id[i] == knee.id
                ):
                    joint = self.model.joint(self.model.eq_obj1id[i])
                    c = self.model.eq_data[i, :5]
                    self.data.qpos[joint.qposadr[0]] = sum(
                        c[k] * q**k for k in range(5)
                    )
                    self.data.qvel[joint.dofadr[0]] = (
                        sum(k * c[k] * q ** (k - 1) for k in range(1, 5)) * v
                    )
        # Suppress before forward (derived force uses zero activation) and again
        # when current raw state is copied. Never write to physical MjData.
        self.data.act[self.mask] = 0
        self.data.ctrl[self.mask] = 0
