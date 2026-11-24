"""Real H1+CMU state; reusable CMU sensor/contact/reward arithmetic."""

import numpy as np
import torch

from myowarp.tasks.musculoskeletal_prosthesis.gpu_state import RealState


class H1CMUState(RealState):
    def __init__(self, model, device, cfg, reference):
        self.cfg, self.model, self.device, self.reference = (
            cfg,
            model,
            device,
            reference,
        )
        index = lambda a: torch.tensor(a, dtype=torch.long, device=device)
        self.sensor_ids = index(
            [model.body(n).id for n in ("pelvis", "right_hip_roll_link")]
        )
        self.sensor_roots = index(model.body_rootid[self.sensor_ids.cpu().numpy()])
        self.body_ids = index([model.body(str(n)).id for n in reference["body_names"]])
        self.body_roots = index(model.body_rootid[self.body_ids.cpu().numpy()])
        self.foot = model.body("ankle_foot").id
        self.foot_root = index([model.body_rootid[self.foot]])
        self.foot_ids = index([self.foot])
        self.sole_geoms = index([model.geom(n).id for n in ("heel", "toe")])
        self.radius = float(np.mean(model.geom_size[self.sole_geoms.cpu().numpy(), 0]))
        self.radii = torch.tensor(
            model.geom_size[self.sole_geoms.cpu().numpy(), 0].copy(), device=device
        )
        self.qids = index([model.joint(n).qposadr[0] for n in ("knee", "ankle")])
        self.vids = index([model.joint(n).dofadr[0] for n in ("knee", "ankle")])
        self.ranges = torch.tensor(
            np.array([model.joint(n).range for n in ("knee", "ankle")]), device=device
        )
        self.motor_ids = index(
            [
                model.actuator(n).id
                for n in ("knee_act", "ankle_act", "knee_vel", "ankle_vel")
            ]
        )
        self.force_limits = torch.tensor(
            np.abs(model.actuator_forcerange[self.motor_ids.cpu().numpy()]).max(1),
            device=device,
        ).clamp_min(1)
        self.bw = float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity))
        self.floor = model.geom("terrain").id
        self.floor_z = float(model.geom_pos[self.floor, 2])
        slots = np.full(model.ngeom, -1, dtype=np.int64)
        for i in range(1, 8):
            slots[model.geom(f"left_foot{i}_collision").id] = 0
        slots[model.geom("toe").id] = 1
        slots[model.geom("heel").id] = 2
        self.slots = index(slots)
        self.loads = None

    def reward(self, data, frame, action, previous, load, unsafe):
        # Inherited formula computes all terms; select ONLY prosthesis terms.
        # Supply zero weights for unrelated computations without rewarding them.
        from types import SimpleNamespace

        cfg = self.cfg
        weights = {
            key: 0.0
            for key in (
                "root_pos",
                "root_ori",
                "root_vel",
                "body_pos",
                "body_ori",
                "body_vel",
                "body_ang_vel",
                "unsafe_contact",
            )
        }
        weights.update(cfg.weights)
        self.cfg = SimpleNamespace(**{**vars(cfg), "weights": weights})
        try:
            _, terms = super().reward(data, frame, action, previous, load, unsafe)
        finally:
            self.cfg = cfg
        selected = {name: terms[name] for name in cfg.weights}
        result = (
            sum(cfg.weights[name] * value for name, value in selected.items())
            * cfg.policy_dt
        )
        return result.float(), selected
