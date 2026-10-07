"""Contract and rollout tests for standalone H1-with-OSL motion tracking."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/myowarp-test-cache")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/myowarp-test-warp-cache")

import mujoco
import numpy as np
import torch

from myowarp.tasks.h1_with_osl_tracking import (
    H1WithOslTrackingCfg,
    H1WithOslTrackingEnv,
    make_rsl_rl_cfg,
)
from myowarp.tasks.h1_with_osl_tracking.env import REWARD_WEIGHTS, TRACKED_BODY_NAMES
from myowarp.tasks.h1_with_osl_velocity.model import build_h1_with_osl_tracking_model
from tests.test_h1_with_osl_velocity import EXPECTED_ACTUATORS, EXPECTED_POLICY_ACTIONS


def test_motion_pipeline_has_no_mjlab_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sources = list((repo_root / "myowarp/tasks/h1_with_osl_tracking").glob("*.py"))
    sources += [repo_root / "scripts/train.py", repo_root / "scripts/eval.py"]
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
        assert all(module.split(".", 1)[0] != "mjlab" for module in imported_modules)


def test_motion_model_reference_and_training_contract() -> None:
    cfg = H1WithOslTrackingCfg(num_envs=1, device="cpu")
    model = build_h1_with_osl_tracking_model(cfg.model_path)
    assert (model.nq, model.nv, model.nu) == (37, 36, 26)
    assert (model.nsensor, model.nsensordata) == (9, 23)
    assert cfg.step_dt == 0.02
    assert cfg.max_episode_length == 500
    assert len(REWARD_WEIGHTS) == 9

    agent_cfg = make_rsl_rl_cfg()
    assert agent_cfg["num_steps_per_env"] == 24
    assert agent_cfg["max_iterations"] == 30_000
    assert agent_cfg["save_interval"] == 500
    assert agent_cfg["algorithm"]["entropy_coef"] == 0.005

    # Guard the distinction between the policy home pose and MuJoCo's joint
    # reference configuration: changing model.qpos0 shifts all link kinematics.
    with np.load(cfg.motion_path) as motion:
        data = mujoco.MjData(model)
        data.qpos[:3] = motion["body_pos_w"][0, 0]
        data.qpos[3:7] = motion["body_quat_w"][0, 0]
        data.qpos[7:] = motion["joint_pos"][0]
        mujoco.mj_forward(model, data)
        body_names = tuple(str(name) for name in motion["body_names"])
        for name in TRACKED_BODY_NAMES:
            model_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            motion_id = body_names.index(name)
            np.testing.assert_allclose(
                data.xpos[model_id], motion["body_pos_w"][0, motion_id], atol=1.0e-6
            )


def test_motion_cpu_rollout_randomization_terminations_and_sampler() -> None:
    cfg = H1WithOslTrackingCfg(
        num_envs=2,
        device="cpu",
        seed=11,
        enable_observation_noise=False,
    )
    env = H1WithOslTrackingEnv(cfg)
    try:
        assert env.motion.num_frames == 4490
        assert env.motion.fps == 50.0
        assert env.motion.body_pos_w.shape == (4490, len(TRACKED_BODY_NAMES), 3)
        observations = env.get_observations()
        assert env.action_joint_names == EXPECTED_POLICY_ACTIONS
        assert [
            env.action_joint_names[index]
            for index in env.ctrl_from_action_indices.cpu().tolist()
        ] == EXPECTED_ACTUATORS
        assert observations["actor"].shape == (2, 161)
        assert observations["critic"].shape == (2, 287)

        friction = env.sim.model_tensor("geom_friction")[:, env.foot_geom_ids, 0]
        assert torch.allclose(friction, friction[:, :1].expand_as(friction))
        assert not torch.allclose(friction[0], friction[1])

        observations, rewards, dones, extras = env.step(torch.zeros((2, 26)))
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(observations["critic"]).all()
        assert torch.isfinite(rewards).all()
        assert set(extras) == {"time_outs", "terminated", "log"}
        assert set(env.reward_terms()) == set(REWARD_WEIGHTS)
        assert all(torch.isfinite(value).all() for value in env.reward_terms().values())

        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, dones, extras = env.step(torch.zeros((2, 26)))
        assert dones.bool().all()
        assert extras["time_outs"].all()

        env.body_pos_relative_w[:, env.end_effector_indices, 2] += 10.0
        previous_failures = env.bin_failed_count.sum().clone()
        _, _, dones, extras = env.step(torch.zeros((2, 26)))
        assert dones.bool().all()
        assert extras["terminated"].all()
        assert env.bin_failed_count.sum() > previous_failures
    finally:
        env.close()
