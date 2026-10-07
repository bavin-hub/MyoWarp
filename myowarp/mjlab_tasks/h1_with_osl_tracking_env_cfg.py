"""MJLab motion-tracking task for H1-2 with an OSL right leg."""

from pathlib import Path

from assets.robots.h1_with_osl.h1_with_osl_constants import use_h1_with_osl_spec
from assets.robots.unitree_h1_2.h1_2_constants import get_h1_2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from .h1_with_osl_env_cfg import H1_WITH_OSL_ACTION_SCALE


REFERENCE_CLIP = (
  Path(__file__).resolve().parents[2]
  / "assets/reference_clips/short_reference_gait_h1_with_osl.npz"
)


def h1_with_osl_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the flat-terrain H1-with-OSL motion-tracking environment."""
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {
    "robot": use_h1_with_osl_spec(get_h1_2_robot_cfg()),
  }

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = H1_WITH_OSL_ACTION_SCALE

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = str(REFERENCE_CLIP)
  motion.anchor_body_name = "torso_link"
  motion.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "osl_knee_assembly",
    "osl_foot_assembly",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  foot_geom_names = tuple(f"left_foot{i}_collision" for i in range(1, 8)) + tuple(
    f"right_foot{i}_collision" for i in range(1, 4)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "osl_foot_assembly",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  cfg.viewer.body_name = "torso_link"

  # Match the already-tested flat H1-with-OSL simulation/contact allocation.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.sampling_mode = "start"

  return cfg
