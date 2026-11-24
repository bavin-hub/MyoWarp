"""Batched CUDA motor interface; the local control law is in sea_controller.py."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class GPUMotorCommand:
    feedforward_torque: (
        torch.Tensor
    )  # (N,2), torque-actuator input = LLC FF + drive I, Nm
    target_velocity: torch.Tensor  # (N,2), rad/s


class GPULowLevelController(Protocol):
    def reset(self, env_ids: torch.Tensor) -> None: ...
    def update(
        self, state, desired_joint_torque: torch.Tensor, dt: float
    ) -> GPUMotorCommand: ...


def write_gpu_motor_command(data, command, motor_ids, limits, num_envs):
    """Validate device/shape and write all four channels without host copies."""
    for t in (command.feedforward_torque, command.target_velocity):
        if (
            not isinstance(t, torch.Tensor)
            or t.shape != (num_envs, 2)
            or t.device != data.ctrl.device
        ):
            raise ValueError(
                "GPU controller must return (num_envs,2) CUDA tensors on the simulation device"
            )
        torch._assert_async(
            torch.isfinite(t).all(), "Nonfinite low-level motor command"
        )
    values = torch.cat((command.feedforward_torque, command.target_velocity), 1)
    data.ctrl[:, motor_ids] = torch.maximum(
        torch.minimum(values, limits[:, 1]), limits[:, 0]
    )
