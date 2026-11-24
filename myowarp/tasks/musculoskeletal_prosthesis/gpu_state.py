"""Vectorized real-state sensors/rewards. All rollout arrays stay on device."""

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from myowarp.musclemimic.math_torch import rotation_error


def body_velocity(data, ids, root_ids, points=None):
    """World linear/angular velocities at link origins or supplied points."""
    cvel = data.cvel[:, ids]
    points = data.xpos[:, ids] if points is None else points
    linear = cvel[..., 3:] + torch.cross(
        cvel[..., :3], points - data.subtree_com[:, root_ids], dim=-1
    )
    return torch.cat((linear, cvel[..., :3]), -1)


class RealState:
    def __init__(self, model, device, cfg, reference):
        self.cfg, self.model, self.device, self.reference = (
            cfg,
            model,
            device,
            reference,
        )

        def index(values):
            return torch.as_tensor(values, dtype=torch.long, device=device)

        self.sensor_ids = index([model.body(n).id for n in ("pelvis", "femur_r")])
        self.sensor_roots = index(model.body_rootid[self.sensor_ids.cpu().numpy()])
        names = reference["body_names"]
        self.body_ids = index([model.body(str(n)).id for n in names])
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
            np.max(
                np.abs(model.actuator_forcerange[self.motor_ids.cpu().numpy()]), axis=1
            ),
            device=device,
        ).clamp_min(1)
        self.bw = float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity))
        self.floor = model.geom("floor").id
        self.floor_z = float(model.geom_pos[self.floor, 2])
        slots = np.full(model.ngeom, -1, dtype=np.int64)
        for g in range(model.ngeom):
            if model.body(model.geom_bodyid[g]).name in ("calcn_l", "toes_l"):
                slots[g] = 0
        slots[model.geom("toe").id], slots[model.geom("heel").id] = 1, 2
        self.slots = index(slots)
        self.loads = None

    def sole(self, d):
        r = d.xmat[:, self.foot].reshape(-1, 3, 3)
        p = d.geom_xpos[:, self.sole_geoms].mean(1) - r[:, :, 1] * self.radius
        return p, r

    def penetration(self, d):
        return (d.geom_xpos[:, self.sole_geoms, 2] - self.radii - self.floor_z).amin(1)

    def contact_loads(self, sim):
        # Use MuJoCo-Warp's own pyramidal/elliptic force decoder.
        d = sim.data
        capacity = d.contact.worldid.shape[0]
        torch._assert_async(
            (wp.to_torch(d.nacon) <= capacity).all(),
            "GPU contact capacity exceeded; increase nconmax",
        )
        torch._assert_async(
            (wp.to_torch(d.nefc) <= d.njmax).all(),
            "GPU constraint capacity exceeded; increase njmax",
        )
        if self.loads is None:
            with wp.ScopedStream(sim.stream):
                self.contact_ids_wp = wp.array(
                    np.arange(capacity, dtype=np.int32), dtype=int
                )
                self.force_wp = wp.zeros(capacity, dtype=wp.spatial_vector)
            self.contact_ids = wp.to_torch(self.contact_ids_wp)
            self.force = wp.to_torch(self.force_wp)
            self.load = torch.zeros((sim.num_envs, 3), device=self.device)
            self.unsafe = torch.zeros(sim.num_envs, device=self.device)
            self.loads = True
        self.force.zero_()
        with wp.ScopedStream(sim.stream):
            mjw.contact_force(sim.model, d, self.contact_ids_wp, False, self.force_wp)
        valid = self.contact_ids < wp.to_torch(d.nacon)[0]
        worlds = wp.to_torch(d.contact.worldid).long().clamp(0, sim.num_envs - 1)
        geoms = wp.to_torch(d.contact.geom).long()
        floor_contact = (geoms[:, 0] == self.floor) | (geoms[:, 1] == self.floor)
        other = torch.where(geoms[:, 0] == self.floor, geoms[:, 1], geoms[:, 0]).clamp(
            0, self.model.ngeom - 1
        )
        slots = self.slots[other]
        normal = self.force[:, 0].clamp_min(0)
        supported = valid & floor_contact & (slots >= 0)
        self.load.zero_()
        self.load.view(-1).scatter_add_(
            0, worlds * 3 + slots.clamp_min(0), normal * supported
        )
        unsafe = valid & (~floor_contact | ((slots < 0) & (normal > 5)))
        self.unsafe.zero_()
        self.unsafe.scatter_add_(0, worlds, unsafe.float())
        return self.load, self.unsafe

    def sensors(self, d, action, load):
        r = d.xmat[:, self.sensor_ids].reshape(-1, 2, 3, 3)
        v = body_velocity(d, self.sensor_ids, self.sensor_roots)
        return torch.cat(
            (
                r[:, 0, :, :2].flatten(1),
                r[:, 1, :, :2].flatten(1),
                v[:, :, 3:].flatten(1),
                -r[:, 0, 2, :],
                d.qpos[:, self.qids],
                d.qvel[:, self.vids],
                (load[:, 1:] > self.cfg.sensor_contact_threshold).float(),
                action,
            ),
            1,
        ).float()

    def critic(self, d, frame, previous):
        p, _ = self.sole(d)
        return torch.cat(
            (
                d.qpos,
                d.qvel,
                d.act,
                self.reference["qpos"][frame],
                self.reference["qvel"][frame],
                p - self.reference["foot_pos"][frame],
                previous,
            ),
            1,
        ).float()

    def reward(self, d, frame, action, previous, load, unsafe):
        ref, cfg = self.reference, self.cfg
        p, r = self.sole(d)
        pos = d.xpos[:, self.body_ids]
        rot = d.xmat[:, self.body_ids].reshape(-1, len(self.body_ids), 3, 3)
        v = body_velocity(d, self.body_ids, self.body_roots)
        fv = body_velocity(d, self.foot_ids, self.foot_root, p[:, None])[:, 0, :3]
        tp, tr = ref["body_pos"][frame], ref["body_rot"][frame]
        rel = (pos - pos[:, :1]) - (tp - tp[:, :1])
        squared = lambda t: t.square().sum(-1)
        load_r = load[:, 1:].sum(-1)
        contact = load_r > cfg.sensor_contact_threshold
        expected = ref["contact"][frame].bool()
        q = d.qpos[:, self.qids]
        f = d.actuator_force[:, self.motor_ids]
        terms = {
            "root_pos": torch.exp(-squared(pos[:, 0] - tp[:, 0]) / 0.3**2),
            "root_ori": torch.exp(
                -rotation_error(rot[:, 0], tr[:, 0]).square() / 0.4**2
            ),
            "root_vel": torch.exp(-squared(v[:, 0, :3] - ref["body_vel"][frame, 0])),
            "body_pos": torch.exp(-squared(rel).mean(-1) / 0.3**2),
            "body_ori": torch.exp(-rotation_error(rot, tr).square().mean(-1) / 0.4**2),
            "body_vel": torch.exp(
                -squared(v[:, :, :3] - ref["body_vel"][frame]).mean(-1)
            ),
            "body_ang_vel": torch.exp(
                -squared(v[:, :, 3:] - ref["body_ang_vel"][frame]).mean(-1)
                / torch.pi**2
            ),
            "foot_pos": torch.exp(-squared(p - ref["foot_pos"][frame]) / 0.15**2),
            "foot_ori": torch.exp(
                -rotation_error(r, ref["foot_rot"][frame]).square() / 0.5**2
            ),
            "foot_vel": torch.exp(-squared(fv - ref["foot_vel"][frame])),
            "contact": (contact == expected).float(),
            "support": expected.float() * (load_r / max(0.3 * self.bw, 1)).clamp(max=1),
            "slip": contact.float() * squared(fv[:, :2]),
            "impact": (load_r / self.bw - 1.5).clamp_min(0).square(),
            "action_rate": squared(action - previous),
            "effort": (f * d.actuator_velocity[:, self.motor_ids]).sum(-1).abs() / 100,
            "saturation": (f.abs() / self.force_limits > 0.95).float().sum(-1),
            "joint_limit": (
                (self.ranges[:, 0] - q).clamp_min(0)
                + (q - self.ranges[:, 1]).clamp_min(0)
            ).sum(-1),
            "unsafe_contact": unsafe,
        }
        reward = sum(cfg.weights[k] * v for k, v in terms.items()) * cfg.policy_dt
        return reward.float(), terms
