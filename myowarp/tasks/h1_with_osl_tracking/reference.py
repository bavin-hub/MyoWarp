"""Reference-motion loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionReference:
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_joint_names: tuple[str, ...],
        selected_body_names: tuple[str, ...],
        expected_fps: float,
        device: torch.device,
    ) -> "MotionReference":
        with np.load(Path(path)) as data:
            required = {
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "joint_names",
                "body_names",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Motion file is missing fields: {sorted(missing)}")
            fps = float(data["fps"][0])
            joint_names = tuple(str(name) for name in data["joint_names"])
            body_names = tuple(str(name) for name in data["body_names"])
            if joint_names != expected_joint_names:
                raise ValueError("Motion joint order does not match the simulation model")
            if abs(fps - expected_fps) > 1.0e-6:
                raise ValueError(
                    f"Motion FPS {fps:g} must equal control frequency {expected_fps:g}"
                )
            try:
                body_indices = [body_names.index(name) for name in selected_body_names]
            except ValueError as exc:
                raise ValueError(f"Motion file is missing a tracked body: {exc}") from exc

            def tensor(name: str) -> torch.Tensor:
                return torch.as_tensor(
                    data[name].copy(), device=device, dtype=torch.float32
                )

            joint_pos = tensor("joint_pos")
            joint_vel = tensor("joint_vel")
            body_pos_w = tensor("body_pos_w")[:, body_indices]
            body_quat_w = tensor("body_quat_w")[:, body_indices]
            body_lin_vel_w = tensor("body_lin_vel_w")[:, body_indices]
            body_ang_vel_w = tensor("body_ang_vel_w")[:, body_indices]

        if joint_pos.ndim != 2 or joint_pos.shape != joint_vel.shape:
            raise ValueError("Motion joint position/velocity arrays are inconsistent")
        if joint_pos.shape[1] != len(expected_joint_names):
            raise ValueError("Motion file has the wrong number of joints")
        expected_body_shape = (joint_pos.shape[0], len(selected_body_names))
        if body_pos_w.shape[:2] != expected_body_shape:
            raise ValueError("Motion body arrays are inconsistent with joint frames")
        return cls(
            fps=fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_names=joint_names,
            body_names=selected_body_names,
        )
