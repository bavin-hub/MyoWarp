"""MJLab task registrations provided by MyoWarp."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .h1_2_env_cfg import h1_2_flat_env_cfg, h1_2_rough_env_cfg
from .h1_with_osl_env_cfg import (
  h1_with_osl_flat_env_cfg,
  h1_with_osl_rough_env_cfg,
)
from .h1_with_osl_tracking_env_cfg import h1_with_osl_tracking_env_cfg
from .rl_cfg import (
  h1_2_ppo_runner_cfg,
  h1_with_osl_ppo_runner_cfg,
  h1_with_osl_tracking_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="MyoWarp-H1_2-Rough",
  env_cfg=h1_2_rough_env_cfg(),
  play_env_cfg=h1_2_rough_env_cfg(play=True),
  rl_cfg=h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="MyoWarp-H1_2-Flat",
  env_cfg=h1_2_flat_env_cfg(),
  play_env_cfg=h1_2_flat_env_cfg(play=True),
  rl_cfg=h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="MyoWarp-H1-With-OSL-Rough",
  env_cfg=h1_with_osl_rough_env_cfg(),
  play_env_cfg=h1_with_osl_rough_env_cfg(play=True),
  rl_cfg=h1_with_osl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="MyoWarp-H1-With-OSL-Flat",
  env_cfg=h1_with_osl_flat_env_cfg(),
  play_env_cfg=h1_with_osl_flat_env_cfg(play=True),
  rl_cfg=h1_with_osl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="MyoWarp-H1-With-OSL-Tracking",
  env_cfg=h1_with_osl_tracking_env_cfg(),
  play_env_cfg=h1_with_osl_tracking_env_cfg(play=True),
  rl_cfg=h1_with_osl_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
