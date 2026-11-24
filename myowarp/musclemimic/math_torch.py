"""Batched rotation conversions; scalar-first quaternions, no host transfers."""

import torch


def quat_from_matrix(m):
    """Stable largest-component conversion, including rotations near pi."""
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    qabs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), -1
            ),
            min=0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (
                    qabs[..., 0] ** 2,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    qabs[..., 1] ** 2,
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    qabs[..., 2] ** 2,
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    qabs[..., 3] ** 2,
                ),
                -1,
            ),
        ),
        -2,
    ) / (2 * qabs.clamp_min(0.1)[..., :, None])
    index = qabs.argmax(-1)[..., None, None].expand(*qabs.shape[:-1], 1, 4)
    q = candidates.gather(-2, index).squeeze(-2)
    return torch.where(q[..., :1] < 0, -q, q)


def axis_angle_from_quat(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    half = torch.atan2(
        torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True), q[..., :1]
    )
    return q[..., 1:] * (2 / torch.sinc(half / torch.pi))


def rotation_error(a, b):
    return torch.linalg.vector_norm(
        axis_angle_from_quat(quat_from_matrix(a @ b.transpose(-1, -2))), dim=-1
    )
