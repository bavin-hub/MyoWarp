"""Torch quaternion helpers using MuJoCo's wxyz convention."""

from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inverse(q: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return quat_conjugate(q) / torch.sum(q * q, dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    return torch.stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ),
        dim=-1,
    )


def yaw_component(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw_quat(yaw)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / torch.sum(q * q, dim=-1)
    values = torch.stack(
        (
            1.0 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1.0 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1.0 - two_s * (x * x + y * y),
        ),
        dim=-1,
    )
    return values.reshape(q.shape[:-1] + (3, 3))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    difference = quat_mul(q1, quat_conjugate(q2))
    difference = torch.where(difference[..., :1] < 0.0, -difference, difference)
    vector_norm = torch.linalg.norm(difference[..., 1:], dim=-1)
    return 2.0 * torch.atan2(vector_norm, difference[..., 0])


def subtract_frame_transforms(
    position_a: torch.Tensor,
    orientation_a: torch.Tensor,
    position_b: torch.Tensor,
    orientation_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_a = quat_inverse(orientation_a)
    return (
        quat_apply(inverse_a, position_b - position_a),
        quat_mul(inverse_a, orientation_b),
    )


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi
