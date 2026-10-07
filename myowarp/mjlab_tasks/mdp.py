"""Unitree velocity-task MDP terms not provided by stock MJLab 1.2."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import *  # noqa: F401, F403
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403
from mjlab.utils.lab_api.math import quat_apply_inverse


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def phase(env, period: float, command_name: str) -> torch.Tensor:
  """Return sine/cosine gait phase, zeroed for standing commands."""
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  value = torch.stack(
    (torch.sin(global_phase * 2.0 * torch.pi), torch.cos(global_phase * 2.0 * torch.pi)),
    dim=1,
  )
  command = env.command_manager.get_command(command_name)
  assert command is not None
  standing = torch.linalg.norm(command, dim=1) < 0.1
  return torch.where(standing.unsqueeze(1), torch.zeros_like(value), value)


def track_linear_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track commanded planar velocity while suppressing vertical velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return torch.exp(-(xy_error + 2.0 * z_error) / std**2)


def track_angular_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track yaw rate while lightly suppressing roll and pitch rates."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(z_error + 0.05 * xy_error) / std**2)


def body_orientation_l2(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize tilt of the configured body relative to gravity."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
  else:
    projected_gravity_b = asset.data.projected_gravity_b
  return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)


def feet_gait(
  env,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward alternating foot contacts according to a fixed gait phase."""
  sensor: ContactSensor = env.scene[sensor_name]
  is_contact = sensor.data.current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
  is_stance = ((global_phase + offsets) % 1.0) < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return reward * (command_magnitude > command_threshold).float()


def stand_still(
  env,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from the home pose when the command is near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
    :, asset_cfg.joint_ids
  ]
  cost = torch.sum(torch.square(error), dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return cost * (command_magnitude <= command_threshold).float()

