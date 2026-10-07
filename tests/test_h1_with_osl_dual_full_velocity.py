"""Contracts for dual-policy full-observation velocity tracking."""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/myowarp-test-cache")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/myowarp-test-warp-cache")

import torch
from tensordict import TensorDict

from myowarp.tasks.h1_with_osl_dual_full_velocity import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
    H1WithOslDualFullVelocityCfg,
    H1WithOslDualFullVelocityEnv,
    make_rsl_rl_cfg,
)
from myowarp.tasks.h1_with_osl_velocity.env import REWARD_WEIGHTS
from rsl_rl.models import DualMLPModel


def test_dual_pipeline_has_no_mjlab_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sources = list(
        (repo_root / "myowarp/tasks/h1_with_osl_dual_full_velocity").glob("*.py")
    )
    sources.append(repo_root / "rsl_rl/rsl_rl/models/dual_mlp_model.py")
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
        assert all(module.split(".", 1)[0] != "mjlab" for module in imported_modules)


def _make_actor(num_envs: int = 4) -> tuple[DualMLPModel, TensorDict]:
    observations = TensorDict(
        {
            "actor": torch.randn(num_envs, 89),
            "critic": torch.randn(num_envs, 104),
        },
        batch_size=[num_envs],
    )
    cfg = make_rsl_rl_cfg()["actor"]
    cfg.pop("class_name")
    cfg["primary_hidden_dims"] = [32, 16]
    cfg["secondary_hidden_dims"] = [32, 16]
    actor = DualMLPModel(
        observations,
        {
            "actor": ["actor"],
            "h1_actor": ["actor"],
            "osl_actor": ["actor"],
        },
        "actor",
        26,
        **copy.deepcopy(cfg),
    )
    return actor, observations


def test_dual_task_configuration() -> None:
    cfg = make_rsl_rl_cfg()
    actor_cfg = cfg["actor"]

    assert actor_cfg["class_name"] == "rsl_rl.models:DualMLPModel"
    assert tuple(actor_cfg["primary_action_indices"]) == H1_ACTION_INDICES
    assert tuple(actor_cfg["secondary_action_indices"]) == OSL_ACTION_INDICES
    assert cfg["obs_groups"]["h1_actor"] == ["actor"]
    assert cfg["obs_groups"]["osl_actor"] == ["actor"]
    assert cfg["obs_groups"]["critic"] == ["critic"]
    assert cfg["algorithm"]["class_name"] == "rsl_rl.algorithms:PPO"


def test_dual_actor_routing_joint_probability_and_gradients() -> None:
    actor, observations = _make_actor()

    deterministic_actions = actor(observations)
    primary_actions = actor.primary_actor(observations)
    secondary_actions = actor.secondary_actor(observations)
    assert deterministic_actions.shape == (4, 26)
    assert torch.allclose(
        deterministic_actions[:, actor.primary_action_indices], primary_actions
    )
    assert torch.allclose(
        deterministic_actions[:, actor.secondary_action_indices], secondary_actions
    )

    actor(observations, stochastic_output=True)
    actions = torch.zeros(4, 26)
    joint_log_probability = actor.get_output_log_prob(actions)
    expected_log_probability = actor.primary_actor.get_output_log_prob(
        actions[:, actor.primary_action_indices]
    ) + actor.secondary_actor.get_output_log_prob(
        actions[:, actor.secondary_action_indices]
    )
    assert torch.allclose(joint_log_probability, expected_log_probability)
    assert torch.allclose(
        actor.output_entropy,
        actor.primary_actor.output_entropy + actor.secondary_actor.output_entropy,
    )

    (-joint_log_probability.mean()).backward()
    primary_gradients = [
        parameter.grad for parameter in actor.primary_actor.mlp.parameters()
    ]
    secondary_gradients = [
        parameter.grad for parameter in actor.secondary_actor.mlp.parameters()
    ]
    assert any(gradient is not None and torch.any(gradient != 0) for gradient in primary_gradients)
    assert any(gradient is not None and torch.any(gradient != 0) for gradient in secondary_gradients)


def test_dual_actor_export_matches_deterministic_policy() -> None:
    actor, observations = _make_actor(num_envs=2)
    actor.eval()
    exported = torch.jit.script(actor.as_jit())

    expected = actor(observations)
    actual = exported(observations["actor"])
    assert torch.allclose(actual, expected, atol=1.0e-5)


def test_dual_environment_reuses_velocity_task_contract() -> None:
    cfg = H1WithOslDualFullVelocityCfg(
        num_envs=1,
        device="cpu",
        seed=5,
        enable_observation_noise=False,
    )
    cfg.domain_randomization.enabled = False
    env = H1WithOslDualFullVelocityEnv(cfg)
    try:
        observations = env.get_observations()
        assert observations["actor"].shape == (1, 89)
        assert observations["critic"].shape == (1, 104)
        assert tuple(env.action_joint_names[index] for index in OSL_ACTION_INDICES) == OSL_ACTION_NAMES
        assert env.h1_action_indices.tolist() == list(H1_ACTION_INDICES)
        assert env.osl_action_indices.tolist() == list(OSL_ACTION_INDICES)

        observations, rewards, dones, extras = env.step(torch.zeros(1, 26))
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(rewards).all()
        assert dones.shape == (1,)
        assert set(extras) == {"time_outs", "log"}
        assert set(env.reward_terms()) == set(REWARD_WEIGHTS)
    finally:
        env.close()
