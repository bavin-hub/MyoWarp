"""Optional actor inputs for experiments 6/7; angles are in radians."""

import torch


def relative_hip_rpy(pelvis_rotation, thigh_rotation):
    """Relative thigh orientation: R_rel = R_pelvis.T @ R_thigh.

    Returns [roll, pitch, yaw] with R_rel = Rz(yaw) Ry(pitch) Rx(roll).
    These use the model's pelvis/body axes, not anatomical generalized coordinates.
    At gimbal lock, choose roll=0 and the equivalent combined yaw.
    """
    relative = pelvis_rotation.transpose(-1, -2) @ thigh_rotation
    cosine_pitch = torch.linalg.vector_norm(relative[..., :2, 0], dim=-1)
    pitch = torch.atan2(-relative[..., 2, 0], cosine_pitch)
    roll = torch.atan2(relative[..., 2, 1], relative[..., 2, 2])
    yaw = torch.atan2(relative[..., 1, 0], relative[..., 0, 0])
    singular = cosine_pitch < 1e-6
    roll = torch.where(singular, torch.zeros_like(roll), roll)
    yaw = torch.where(
        singular, torch.atan2(-relative[..., 0, 1], relative[..., 1, 1]), yaw
    )
    return torch.stack((roll, pitch, yaw), -1)


def hip_actor_history(history):
    """Replace pelvis orientation/angular velocity with hip RPY per frame.

    Internal sensor history retains its old layout for reward/debug consumers.
    Actor frames: hip RPY, thigh orientation, thigh angular velocity,
    pelvis-frame gravity, joint q/qvel, contacts, previous actions (23 values).
    """

    def rotation(values):
        columns = values.reshape(*values.shape[:-1], 3, 2)
        third = torch.cross(columns[..., 0], columns[..., 1], dim=-1)
        return torch.cat((columns, third.unsqueeze(-1)), -1)

    hip = relative_hip_rpy(rotation(history[..., :6]), rotation(history[..., 6:12]))
    return torch.cat((hip, history[..., 6:12], history[..., 15:]), -1).flatten(1)
