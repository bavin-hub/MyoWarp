"""MuJoCo-Warp 3.11 storage shared with Torch. Production runtime requires CUDA."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


def require_cuda(device):
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This pipeline requires CUDA and a working NVIDIA driver; no CPU fallback. Offline prepare and tensor parity tests can run on CPU."
        )
    if mujoco.__version__ != "3.11.0":
        raise RuntimeError(
            "Use requirements-prosthesis-rl.txt in the isolated environment (MuJoCo/Warp 3.11.0)."
        )
    return str(
        torch.device(
            "cuda",
            target.index if target.index is not None else torch.cuda.current_device(),
        )
    )


class WarpBatch:
    def __init__(self, model, num_envs, device, nconmax=128, njmax=1024):
        self.device = require_cuda(device)
        self.cpu_model = model
        self.num_envs = num_envs
        wp.init()
        self.wp_device = wp.get_device(self.device)
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.stream):
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(
                model,
                mujoco.MjData(model),
                nworld=num_envs,
                nconmax=nconmax,
                njmax=njmax,
            )
            self.reset_mask_wp = wp.zeros(num_envs, dtype=bool)
        self.reset_mask = wp.to_torch(self.reset_mask_wp)
        # All Warp operations use the Torch stream for zero-copy access.
        self.view = SimpleNamespace(
            **{
                name: wp.to_torch(getattr(self.data, name))
                for name in (
                    "qpos",
                    "qvel",
                    "act",
                    "ctrl",
                    "qacc_warmstart",
                    "time",
                    "actuator_length",
                    "actuator_velocity",
                    "actuator_force",
                    "sensordata",
                    "site_xpos",
                    "site_xmat",
                    "cvel",
                    "subtree_com",
                    "xpos",
                    "xmat",
                    "geom_xpos",
                )
            }
        )

    def forward(self):
        with wp.ScopedStream(self.stream):
            mjw.forward(self.model, self.data)

    def step(self):
        with wp.ScopedStream(self.stream):
            mjw.step(self.model, self.data)

    def reset(self, ids):
        self.reset_mask.zero_()
        self.reset_mask[ids] = True
        with wp.ScopedStream(self.stream):
            mjw.reset_data(self.model, self.data, self.reset_mask_wp)
