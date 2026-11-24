"""Native CPU MuJoCo adapters for single-environment evaluation only."""

from dataclasses import replace

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from myowarp.tasks.h1_2_tracking.env import TRACKED_BODY_NAMES, H1TrackingEnv
from myowarp.tasks.h1_2_tracking.model import build_h1_tracking_model
from myowarp.tasks.h1_with_osl_tracking.reference import MotionReference

from .state import H1CMUState


class NativeSim:
    def __init__(self, model, *, num_envs=1, **kwargs):
        if num_envs != 1:
            raise ValueError("Native CPU evaluation supports one environment")
        self.mj_model = self.cpu_model = model
        self.native_data = mujoco.MjData(model)
        self.num_envs = 1
        self.device = "cpu"
        mujoco.mj_forward(model, self.native_data)

    def tensor(self, name):
        return torch.from_numpy(getattr(self.native_data, name))[None]

    def step(self):
        mujoco.mj_step(self.mj_model, self.native_data)

    def forward(self):
        mujoco.mj_forward(self.mj_model, self.native_data)

    def reset(self, env_ids=None):
        if env_ids is None or len(env_ids):
            mujoco.mj_resetData(self.mj_model, self.native_data)


class NativeH1Shadow(H1TrackingEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.num_actions = 27
        self.step_dt = cfg.step_dt
        self.physics_dt = cfg.physics_dt
        self.max_episode_length = cfg.max_episode_length
        self.common_step_counter = 0
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self.model = build_h1_tracking_model(cfg.model_path)
        self.sim = NativeSim(self.model)
        self._build_indices()
        self.motion = MotionReference.load(
            cfg.motion_path,
            expected_joint_names=tuple(self.joint_names),
            selected_body_names=TRACKED_BODY_NAMES,
            expected_fps=1 / self.step_dt,
            device="cpu",
        )
        self.motion = replace(
            self.motion,
            **{
                name: value.double()
                for name, value in vars(self.motion).items()
                if isinstance(value, torch.Tensor)
            },
        )
        self._allocate_buffers()
        self._obs = TensorDict({}, batch_size=[1], device="cpu")
        self.sim.forward()
        self._update_relative_targets()
        self._obs = self._compute_observations()

    def _compute_observations(self):
        # Native MuJoCo is float64; pretrained network inputs remain float32.
        return super()._compute_observations().to(dtype=torch.float32)


class NativeState(H1CMUState):
    def contact_loads(self, sim):
        load = np.zeros((1, 3), dtype=np.float32)
        unsafe = np.zeros(1, dtype=np.float32)
        force = np.empty(6)
        slots = self.slots.numpy()
        for i in range(sim.native_data.ncon):
            contact = sim.native_data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            mujoco.mj_contactForce(self.model, sim.native_data, i, force)
            floor_contact = g1 == self.floor or g2 == self.floor
            other = g2 if g1 == self.floor else g1
            slot = slots[other]
            normal = max(0.0, force[0])
            if floor_contact and slot >= 0:
                load[0, slot] += normal
            if not floor_contact or (slot < 0 and normal > 5):
                unsafe[0] += 1
        return torch.from_numpy(load), torch.from_numpy(unsafe)
