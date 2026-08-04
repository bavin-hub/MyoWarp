from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import mujoco
import torch
import warp as wp


@dataclass
class WarpState:
    """Thin state wrapper so env code does not depend on raw MJWarp objects."""

    data: Any


class MujocoWarpBackend:
    """Small adapter around MuJoCo Warp.

    This module is the only place that should know about `mujoco_warp`.
    The public examples currently use:

        import mujoco_warp as mjw
        mj_model = mujoco.MjModel.from_xml_path(...)
        m = mjw.put_model(mj_model)
        d = mjw.make_data(mj_model)
        mjw.step(m, d)

    Keep this adapter narrow so API changes in MJWarp stay contained.
    """

    def __init__(
        self,
        model_path: str | Path,
        num_envs: int,
        device: str = "cuda",
        physics_timestep: float | None = None,
        model_mutator: Optional[Callable[[mujoco.MjModel], None]] = None,
    ):
        self.model_path = Path(model_path)
        self.num_envs = int(num_envs)
        self.device = device
        self.cpu_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if physics_timestep is not None:
            self.cpu_model.opt.timestep = float(physics_timestep)
        # Apply any model edits (e.g. torso lean) before uploading to Warp, since
        # put_model snapshots the model and later edits would not take effect.
        if model_mutator is not None:
            model_mutator(self.cpu_model)

        try:
            import mujoco_warp as mjw  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mujoco_warp is not installed. Install it in the myowarp env before "
                "running the GPU backend."
            ) from exc

        self.mjw = mjw
        self.model = self._put_model(self.cpu_model)
        self.state = WarpState(data=self._make_data())

    def _put_model(self, model: mujoco.MjModel):
        if hasattr(self.mjw, "put_model"):
            return self.mjw.put_model(model)
        raise AttributeError("mujoco_warp.put_model was not found")

    def _make_data(self):
        if hasattr(self.mjw, "make_data"):
            return self.mjw.make_data(self.cpu_model, nworld=self.num_envs)
        raise AttributeError("mujoco_warp.make_data was not found")

    def set_ctrl(self, ctrl: torch.Tensor) -> None:
        """Write `[num_envs, nu]` controls into Warp data."""
        if ctrl.shape != (self.num_envs, self.cpu_model.nu):
            raise ValueError(f"Expected ctrl shape {(self.num_envs, self.cpu_model.nu)}, got {tuple(ctrl.shape)}")
        self.set_field("ctrl", ctrl)

    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """Write batched generalized positions and velocities into Warp data."""
        if qpos.shape != (self.num_envs, self.cpu_model.nq):
            raise ValueError(f"Expected qpos shape {(self.num_envs, self.cpu_model.nq)}, got {tuple(qpos.shape)}")
        if qvel.shape != (self.num_envs, self.cpu_model.nv):
            raise ValueError(f"Expected qvel shape {(self.num_envs, self.cpu_model.nv)}, got {tuple(qvel.shape)}")
        self._assign(self.state.data.qpos, qpos)
        self._assign(self.state.data.qvel, qvel)

    def set_field(self, name: str, value: torch.Tensor) -> None:
        if not hasattr(self.state.data, name):
            raise AttributeError(f"Warp data object has no writable {name} field")
        target = getattr(self.state.data, name)
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Expected {name} shape {tuple(target.shape)}, got {tuple(value.shape)}")
        self._assign(target, value)

    def forward(self) -> None:
        maybe_new_data = self.mjw.forward(self.model, self.state.data)
        if maybe_new_data is not None:
            self.state.data = maybe_new_data

    def step(self, frame_skip: int) -> None:
        for _ in range(frame_skip):
            maybe_new_data = self.mjw.step(self.model, self.state.data)
            if maybe_new_data is not None:
                self.state.data = maybe_new_data

    def get_tensor(self, name: str) -> torch.Tensor:
        value = getattr(self.state.data, name)
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "to_torch"):
            return value.to_torch()
        if hasattr(value, "numpy"):
            return torch.as_tensor(value.numpy(), device=self.device)
        return torch.as_tensor(value, device=self.device)

    def _assign(self, target, value: torch.Tensor) -> None:
        value = value.to(device=self.device, dtype=torch.float32).contiguous()
        target.assign(wp.from_torch(value, dtype=wp.float32))
