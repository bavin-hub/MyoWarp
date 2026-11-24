"""One batched biological shadow and frozen Torch muscle actor on CUDA."""

import mujoco
import numpy as np
import torch

from myowarp.musclemimic.actor_torch import MimicActorModule
from myowarp.musclemimic.fullbody_obs_torch import TorchFullbodyObsAdapter
from myowarp.musclemimic.policy import (
    FullbodyObsAdapter,
    fullbody_history_settings_from_metadata,
    fullbody_obs_adapter_params_from_metadata,
)
from myowarp.musclemimic.running_stats import torch_running_mean_std_update_per_env

from .geometry import joint_map, muscle_map
from .gpu_sim import WarpBatch
from .shadow import LOCK_ONLY_MUSCLES, VIEW_JOINTS


class FrozenHumanBatch:
    def __init__(self, human, real, clip, artifacts, metadata, num_envs, device, cfg):
        history = fullbody_history_settings_from_metadata(metadata)
        if history["len_obs_history"] != 1 or history["split_goal"]:
            raise ValueError(
                "GPU frozen actor currently requires the mm-10m-2 single-frame, unsplit checkpoint"
            )
        self.sim = WarpBatch(human, num_envs, device, cfg.nconmax, cfg.njmax)
        self.data, self.device = self.sim.view, device
        self.mode = cfg.right_leg_obs
        self.actor = (
            MimicActorModule.from_artifacts(artifacts)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.fixed_normalization = metadata.get("normalization_mode") == "fixed"
        if self.fixed_normalization:
            # Strict load includes the exact stored std (no var/std round trip).
            self.actor.load_state_dict(metadata["actor_state_dict"], strict=True)
        cpu_adapter = FullbodyObsAdapter(
            human, clip, goal_params=fullbody_obs_adapter_params_from_metadata(metadata)
        )
        self.adapter = TorchFullbodyObsAdapter(cpu_adapter, device=torch.device(device))

        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        self.sq, self.dq, self.sv, self.dv = [
            indices(x) for x in joint_map(human, real, allow_missing=True)
        ]
        shared, real_ids, shadow_act, real_act = muscle_map(human, real)
        self.shadow_actuator_ids = indices(shared)
        self.actuator_ids = indices(real_ids)
        self.shadow_activation_ids = indices(shadow_act)
        self.activation_ids = indices(real_act)
        self.mask = indices([human.actuator(n).id for n in LOCK_ONLY_MUSCLES])
        self.view_q = indices([human.joint(n).qposadr[0] for n in VIEW_JOINTS])
        self.view_v = indices([human.joint(n).dofadr[0] for n in VIEW_JOINTS])
        self.prosthesis_q = indices(
            [real.joint(n).qposadr[0] for n in ("knee", "ankle")]
        )
        self.prosthesis_v = indices(
            [real.joint(n).dofadr[0] for n in ("knee", "ankle")]
        )
        self.clip_q = torch.tensor(clip.qpos, dtype=torch.float32, device=device)
        self.clip_v = torch.tensor(
            clip.qvel
            if clip.qvel is not None
            else np.zeros((len(clip.qpos), human.nv)),
            dtype=torch.float32,
            device=device,
        )
        self.followers = []
        knee = human.joint("knee_angle_r").id
        self.knee_q, self.knee_v = (
            int(human.jnt_qposadr[knee]),
            int(human.jnt_dofadr[knee]),
        )
        for i in range(human.neq):
            if (
                human.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
                and human.eq_obj2id[i] == knee
            ):
                j = human.eq_obj1id[i]
                self.followers.append(
                    (
                        int(human.jnt_qposadr[j]),
                        int(human.jnt_dofadr[j]),
                        human.eq_data[i, :5].copy(),
                    )
                )
        self.ctrl_range = torch.tensor(
            human.actuator_ctrlrange.copy(), dtype=torch.float32, device=device
        )
        self.initial_mean = torch.tensor(artifacts.obs_mean, device=device)
        self.initial_var = torch.tensor(artifacts.obs_var, device=device)
        self.initial_count = float(np.asarray(artifacts.obs_count).item())
        self.mean = self.initial_mean.repeat(num_envs, 1)
        self.var = self.initial_var.repeat(num_envs, 1)
        self.count = torch.full((num_envs,), self.initial_count, device=device)

    def reset(self, ids):
        self.mean[ids] = self.initial_mean
        self.var[ids] = self.initial_var
        self.count[ids] = self.initial_count

    def _copy(self, source, frame):
        d = self.data
        # Missing physical joints exist only in the full biological shadow.
        d.qpos[:] = self.clip_q[frame]
        d.qvel[:] = self.clip_v[frame]
        d.qpos[:, self.sq] = source.qpos[:, self.dq]
        d.qvel[:, self.sv] = source.qvel[:, self.dv]
        d.act.zero_()
        d.ctrl.zero_()
        d.act[:, self.shadow_activation_ids] = source.act[:, self.activation_ids]
        d.ctrl[:, self.shadow_actuator_ids] = source.ctrl[:, self.actuator_ids]
        if self.mode == "virtual":
            d.qpos[:, self.view_q[:2]] = source.qpos[:, self.prosthesis_q]
            d.qvel[:, self.view_v[:2]] = source.qvel[:, self.prosthesis_v]
            d.qpos[:, self.view_q[2:]] = 0
            d.qvel[:, self.view_v[2:]] = 0
        else:
            d.qpos[:, self.view_q] = self.clip_q[frame][:, self.view_q]
            d.qvel[:, self.view_v] = self.clip_v[frame][:, self.view_v]
        q, v = d.qpos[:, self.knee_q], d.qvel[:, self.knee_v]
        for qi, vi, c in self.followers:
            d.qpos[:, qi] = sum(float(c[k]) * q**k for k in range(5))
            d.qvel[:, vi] = sum(k * float(c[k]) * q ** (k - 1) for k in range(1, 5)) * v

    def observe(self, real, pre_substep, frame):
        self._copy(pre_substep, frame)
        self.data.qacc_warmstart.zero_()
        self.data.qacc_warmstart[:, self.sv] = pre_substep.qacc_warmstart[:, self.dv]
        self.sim.forward()
        # Preserve prosim's convention: derived values at pre-last-substep,
        # raw q/qvel/act/ctrl at the current state. Never modify real physics.
        self._copy(real, frame)
        self.data.time[:] = real.time
        return self.adapter.build(self.data, frame)

    @torch.no_grad()
    def normalized_observation(self, real, pre_substep, frame):
        obs = self.observe(real, pre_substep, frame)
        if obs.shape[1] != self.actor.obs_dim:
            raise ValueError("Frozen human observation/checkpoint dimension mismatch")
        if self.fixed_normalization:
            return self.actor.normalize(obs)
        obs, self.mean, self.var, self.count = torch_running_mean_std_update_per_env(
            obs, self.mean, self.var, self.count
        )
        return obs

    @torch.no_grad()
    def action(self, real, pre_substep, frame):
        obs = self.normalized_observation(real, pre_substep, frame)
        return self.actor.forward_normalized(obs).clamp(-1, 1)

    def write_muscles(self, real, action):
        torch._assert_async(
            torch.isfinite(action).all(), "Nonfinite frozen muscle output"
        )
        u = torch.maximum(
            torch.minimum(action, self.ctrl_range[:, 1]), self.ctrl_range[:, 0]
        ).clone()
        u[:, self.mask] = 0
        real.ctrl[:, self.actuator_ids] = u[:, self.shadow_actuator_ids]
