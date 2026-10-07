"""H1-2 with an OSL right leg using the unchanged H1-2 velocity task."""

from dataclasses import replace

from assets.robots.h1_with_osl.h1_with_osl_constants import use_h1_with_osl_spec
from assets.robots.unitree_h1_2.h1_2_constants import (
  H1_2_ACTION_SCALE,
  get_h1_2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg

from .h1_2_env_cfg import h1_2_flat_env_cfg, h1_2_rough_env_cfg


H1_WITH_OSL_ACTION_SCALE = {
  **H1_2_ACTION_SCALE,
  # osl_knee_angle_r already matches the existing ".*_knee.*" scale.
  "osl_ankle_angle_r": H1_2_ACTION_SCALE[".*_ankle_pitch.*"],
}

# Exclude the four passive socket-compliance joints from policy observations and
# joint-dependent rewards. They remain active in the physics model.
def _actuated_joints_cfg() -> SceneEntityCfg:
  """Return a fresh selector because managers resolve these objects in place."""
  return SceneEntityCfg("robot", joint_names=(r"^(?!socket_).*$",))


def _use_h1_with_osl(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Retarget an H1-2 task without changing its MDP structure."""
  cfg.scene.entities["robot"] = use_h1_with_osl_spec(get_h1_2_robot_cfg())

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = H1_WITH_OSL_ACTION_SCALE

  for group_name in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      cfg.observations[group_name].terms[term_name].params["asset_cfg"] = (
        _actuated_joints_cfg()
      )

  for reward_name in ("pose", "joint_acc_l2", "joint_pos_limits", "stand_still"):
    cfg.rewards[reward_name].params["asset_cfg"] = _actuated_joints_cfg()

  # The left foot keeps seven H1 collision geoms; the OSL foot has three.
  foot_geom_names = tuple(f"left_foot{i}_collision" for i in range(1, 8)) + tuple(
    f"right_foot{i}_collision" for i in range(1, 4)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names

  sensors = []
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "feet_ground_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor = replace(
        sensor,
        primary=replace(
          sensor.primary,
          pattern=r"^(left_ankle_roll_link|osl_foot_assembly)$",
        ),
      )
    sensors.append(sensor)
  cfg.scene.sensors = tuple(sensors)

  # Preserve the H1 ankle-pitch posture widths for the renamed OSL ankle joint.
  cfg.rewards["pose"].params["std_walking"][r"osl_ankle.*"] = 0.15
  cfg.rewards["pose"].params["std_running"][r"osl_ankle.*"] = 0.25
  return cfg


def h1_with_osl_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _use_h1_with_osl(h1_2_rough_env_cfg(play=play))


def h1_with_osl_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _use_h1_with_osl(h1_2_flat_env_cfg(play=play))
