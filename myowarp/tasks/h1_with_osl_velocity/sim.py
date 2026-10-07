"""Small MuJoCo-Warp adapter used by the standalone task."""

from __future__ import annotations

from typing import Any

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp


@wp.kernel(module="myowarp_h1_velocity")
def _repeat_array_kernel(
    src: wp.array(dtype=Any),  # type: ignore[type-arg]
    elements_per_world: int,
    dst: wp.array(dtype=Any),  # type: ignore[type-arg]
):
    tid = wp.tid()
    dst[tid] = src[tid % elements_per_world]


def _tile_first_dimension(array: wp.array, num_envs: int):
    new_shape = list(array.shape)
    new_shape[0] = num_envs
    array_type = {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[
        len(new_shape)
    ]
    tiled = array_type(shape=new_shape, dtype=array.dtype, device=array.device)
    source_flat = array.flatten()
    target_flat = tiled.flatten()
    elements_per_world = target_flat.shape[0] // num_envs
    wp.launch(
        _repeat_array_kernel,
        dim=target_flat.shape[0],
        inputs=[source_flat, elements_per_world],
        outputs=[target_flat],
        device=array.device,
    )
    return tiled


class StandaloneMujocoWarpSim:
    """Owns a batched MuJoCo-Warp model and exposes zero-copy Torch views."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        num_envs: int,
        device: str,
        nconmax: int,
        njmax: int,
    ) -> None:
        self.mj_model = model
        self.num_envs = num_envs
        self.device = device
        self.wp_device = wp.get_device(device)
        cpu_data = mujoco.MjData(model)
        mujoco.mj_forward(model, cpu_data)
        with wp.ScopedDevice(self.wp_device):
            self.model = mjwarp.put_model(model)
            self.model.opt.ls_parallel = True
            self.model.opt.contact_sensor_maxmatch = 64
            self.data = mjwarp.put_data(
                model,
                cpu_data,
                nworld=num_envs,
                nconmax=nconmax,
                njmax=njmax,
            )
            self._reset_mask_wp = wp.zeros(num_envs, dtype=bool)
        self._reset_mask = wp.to_torch(self._reset_mask_wp)
        self._graphs: dict[str, Any] = {}
        self._capture_graphs()

    def tensor(self, name: str) -> torch.Tensor:
        return wp.to_torch(getattr(self.data, name))

    def model_tensor(self, name: str) -> torch.Tensor:
        return wp.to_torch(getattr(self.model, name))

    def expand_model_fields(self, fields: tuple[str, ...]) -> None:
        if self.num_envs == 1:
            return
        with wp.ScopedDevice(self.wp_device):
            for field in fields:
                array = getattr(self.model, field)
                if array.shape[0] == self.num_envs:
                    continue
                setattr(self.model, field, _tile_first_dimension(array, self.num_envs))
        self._capture_graphs()

    def recompute_constants(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            mjwarp.set_const(self.model, self.data)
        self._capture_graphs()

    def _capture_graphs(self) -> None:
        self._graphs.clear()
        if not self.wp_device.is_cuda or not wp.is_mempool_enabled(self.wp_device):
            return
        with wp.ScopedDevice(self.wp_device):
            with wp.ScopedCapture() as capture:
                mjwarp.step(self.model, self.data)
            self._graphs["step"] = capture.graph
            with wp.ScopedCapture() as capture:
                mjwarp.forward(self.model, self.data)
            self._graphs["forward"] = capture.graph
            with wp.ScopedCapture() as capture:
                mjwarp.reset_data(self.model, self.data, reset=self._reset_mask_wp)
            self._graphs["reset"] = capture.graph

    def step(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            graph = self._graphs.get("step")
            if graph is None:
                mjwarp.step(self.model, self.data)
            else:
                wp.capture_launch(graph)

    def forward(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            graph = self._graphs.get("forward")
            if graph is None:
                mjwarp.forward(self.model, self.data)
            else:
                wp.capture_launch(graph)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._reset_mask.zero_()
        if env_ids is None:
            self._reset_mask.fill_(True)
        else:
            self._reset_mask[env_ids] = True
        with wp.ScopedDevice(self.wp_device):
            graph = self._graphs.get("reset")
            if graph is None:
                mjwarp.reset_data(self.model, self.data, reset=self._reset_mask_wp)
            else:
                wp.capture_launch(graph)

