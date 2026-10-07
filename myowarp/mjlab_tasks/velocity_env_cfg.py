"""Unitree-style velocity task built on MJLab's maintained base config."""

import math

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg as _mjlab_velocity_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp


def make_velocity_env_cfg():
  """Create the velocity task used by Unitree's H1-2 training setup."""
  cfg = _mjlab_velocity_cfg()

  # Actor excludes privileged linear velocity and includes an explicit gait phase.
  old_actor = cfg.observations["actor"].terms
  base_lin_vel = old_actor["base_lin_vel"]
  actor_terms = {
    "base_ang_vel": old_actor["base_ang_vel"],
    "projected_gravity": old_actor["projected_gravity"],
    "command": old_actor["command"],
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": old_actor["joint_pos"],
    "joint_vel": old_actor["joint_vel"],
    "actions": old_actor["actions"],
    "height_scan": old_actor["height_scan"],
  }
  cfg.observations["actor"].terms = actor_terms
  cfg.observations["actor"].history_length = 1

  old_critic = cfg.observations["critic"].terms
  cfg.observations["critic"].terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": old_critic["height_scan"],
    "foot_height": old_critic["foot_height"],
    "foot_air_time": old_critic["foot_air_time"],
    "foot_contact": old_critic["foot_contact"],
    "foot_contact_forces": old_critic["foot_contact_forces"],
  }
  cfg.observations["critic"].history_length = 1
  del base_lin_vel

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_standing_envs = 0.05
  command.rel_heading_envs = 1.0
  command.ranges.lin_vel_x = (-1.0, 2.0)
  command.ranges.lin_vel_y = (-1.0, 1.0)
  command.ranges.ang_vel_z = (-1.0, 1.0)

  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
  cfg.events["push_robot"].interval_range_s = (5.0, 6.0)
  cfg.events["foot_friction"].params["ranges"] = (0.3, 1.6)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }

  rewards = cfg.rewards
  rewards["track_linear_velocity"].func = mdp.track_linear_velocity
  rewards["track_linear_velocity"].weight = 1.0
  rewards["track_angular_velocity"].func = mdp.track_angular_velocity
  rewards["track_angular_velocity"].weight = 1.0

  rewards.pop("upright")
  rewards["body_orientation_l2"] = RewardTermCfg(
    func=mdp.body_orientation_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
  )

  rewards["pose"].params["walking_threshold"] = 0.1
  rewards["body_ang_vel"].weight = -0.05
  rewards["angular_momentum"].weight = -0.025
  rewards.pop("dof_pos_limits")
  rewards.pop("air_time")
  rewards.pop("foot_swing_height")
  rewards["is_terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
  rewards["joint_acc_l2"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
  rewards["joint_pos_limits"] = RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0)
  rewards["action_rate_l2"].weight = -0.05
  rewards["foot_gait"] = RewardTermCfg(
    func=mdp.feet_gait,
    weight=0.5,
    params={
      "period": 0.6,
      "offset": [0.0, 0.5],
      "threshold": 0.56,
      "command_threshold": 0.1,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
    },
  )
  rewards["foot_clearance"].weight = -1.0
  rewards["foot_clearance"].params.update(
    target_height=0.10,
    command_threshold=0.1,
  )
  rewards["foot_slip"].weight = -0.25
  rewards["foot_slip"].params["command_threshold"] = 0.1
  rewards["soft_landing"].weight = -1.0e-3
  rewards["soft_landing"].params["command_threshold"] = 0.1
  rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.5, 1.0),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-1.0, 1.0),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-1.0, 2.0),
      "lin_vel_y": (-1.0, 1.0),
    },
  ]

  # Unitree uses this scale as its generic default; H1-2 overrides it per actuator.
  cfg.actions["joint_pos"].scale = 0.25
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(70.0)
  return cfg

