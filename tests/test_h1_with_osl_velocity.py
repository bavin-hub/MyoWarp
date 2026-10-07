"""Contract and rollout tests for the standalone H1-with-OSL velocity task."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/myowarp-test-cache")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/myowarp-test-warp-cache")

import mujoco
import torch

from myowarp.tasks.h1_with_osl_velocity import (
    H1WithOslVelocityCfg,
    H1WithOslVelocityEnv,
    build_h1_with_osl_velocity_model,
    make_rsl_rl_cfg,
)
from myowarp.tasks.h1_with_osl_velocity.env import REWARD_WEIGHTS
from myowarp.tasks.h1_with_osl_velocity.model import names_for


EXPECTED_ACTUATORS = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "torso_joint",
    "left_knee_joint",
    "osl_knee_angle_r",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "osl_ankle_angle_r",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

EXPECTED_POLICY_ACTIONS = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "osl_knee_angle_r",
    "osl_ankle_angle_r",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def test_standalone_pipeline_has_no_mjlab_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sources = list((repo_root / "myowarp/tasks/h1_with_osl_velocity").glob("*.py"))
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


def test_model_and_training_contract() -> None:
    cfg = H1WithOslVelocityCfg(num_envs=1, device="cpu")
    model = build_h1_with_osl_velocity_model(cfg.model_path)

    assert (model.nq, model.nv, model.nu) == (37, 36, 26)
    assert names_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu) == EXPECTED_ACTUATORS
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    actuated_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in EXPECTED_POLICY_ACTIONS
    ]
    assert (model.dof_frictionloss[model.jnt_dofadr[actuated_joint_ids]] == 0.0).all()
    assert model.nsensordata == 31
    assert cfg.step_dt == 0.02
    assert cfg.max_episode_length == 1000
    assert len(REWARD_WEIGHTS) == 16

    agent_cfg = make_rsl_rl_cfg()
    assert agent_cfg["obs_groups"] == {"actor": ["actor"], "critic": ["critic"]}
    assert agent_cfg["num_steps_per_env"] == 24
    assert agent_cfg["algorithm"]["class_name"] == "rsl_rl.algorithms:PPO"


def test_cpu_rollout_and_domain_randomization() -> None:
    cfg = H1WithOslVelocityCfg(
        num_envs=2,
        device="cpu",
        seed=7,
        enable_observation_noise=False,
    )
    env = H1WithOslVelocityEnv(cfg)
    try:
        observations = env.get_observations()
        assert env.action_joint_names == EXPECTED_POLICY_ACTIONS
        assert [
            env.action_joint_names[index]
            for index in env.ctrl_from_action_indices.cpu().tolist()
        ] == EXPECTED_ACTUATORS
        assert observations["actor"].shape == (2, 89)
        assert observations["critic"].shape == (2, 104)

        friction = env.sim.model_tensor("geom_friction")[:, env.foot_geom_ids, 0]
        assert torch.allclose(friction, friction[:, :1].expand_as(friction))
        assert not torch.allclose(friction[0], friction[1])

        env.set_command(0.5, 0.0, 0.0)
        observations, rewards, dones, extras = env.step(torch.zeros((2, 26)))
        assert observations["actor"].shape == (2, 89)
        assert observations["critic"].shape == (2, 104)
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(observations["critic"]).all()
        assert torch.isfinite(rewards).all()
        assert dones.shape == (2,)
        assert set(extras) == {"time_outs", "log"}
        assert set(env.reward_terms()) == set(REWARD_WEIGHTS)
        assert all(torch.isfinite(value).all() for value in env.reward_terms().values())

        env.episode_length_buf[:] = env.max_episode_length - 1
        env.step(torch.zeros((2, 26)))
        assert torch.allclose(
            env.commands, torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        )
    finally:
        env.close()
