"""Legacy CPU-oracle interfaces and controller factory loading."""

import importlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MissingLowLevelController(RuntimeError):
    pass


@dataclass(frozen=True)
class MotorCommand:
    feedforward_torque: np.ndarray  # knee_motor, ankle_motor; Nm
    target_velocity: np.ndarray  # knee_motor, ankle_motor; rad/s


class LowLevelController(Protocol):
    def reset(self, model, data) -> None: ...
    def update(
        self, model, data, desired_joint_torque: np.ndarray, dt: float
    ) -> MotorCommand: ...


def load_controller_factory(spec):
    if not spec:
        raise MissingLowLevelController(
            "No controller factory was specified. "
            "Provide --controller module:factory. "
            "No direct-torque fallback is provided."
        )
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def write_motor_command(model, data, command):
    ff = np.asarray(command.feedforward_torque, dtype=float)
    velocity = np.asarray(command.target_velocity, dtype=float)
    if (
        ff.shape != (2,)
        or velocity.shape != (2,)
        or not np.isfinite(np.r_[ff, velocity]).all()
    ):
        raise ValueError(
            "Controller must return finite (2,) motor torque and velocity arrays"
        )
    for i, joint in enumerate(("knee", "ankle")):
        for suffix, value in (("act", ff[i]), ("vel", velocity[i])):
            actuator = model.actuator(f"{joint}_{suffix}").id
            data.ctrl[actuator] = np.clip(value, *model.actuator_ctrlrange[actuator])
