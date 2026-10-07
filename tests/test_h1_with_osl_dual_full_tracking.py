"""Contracts for dual-policy full-observation motion tracking."""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/myowarp-test-cache")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/myowarp-test-warp-cache")

import torch
from tensordict import TensorDict

from myowarp.tasks.h1_with_osl_dual_full_tracking import (
    H1_ACTION_INDICES,
    OSL_ACTION_INDICES,
    OSL_ACTION_NAMES,
    H1WithOslDualFullTrackingCfg,
    H1WithOslDualFullTrackingEnv,
    make_rsl_rl_cfg,
)
from myowarp.tasks.h1_with_osl_tracking.env import REWARD_WEIGHTS
from rsl_rl.models import DualMLPModel


def test_dual_tracking_pipeline_has_no_mjlab_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sources = list(
        (repo_root / "myowarp/tasks/h1_with_osl_dual_full_tracking").glob("*.py")
    )
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
            "actor": torch.randn(num_envs, 161),
            "critic": torch.randn(num_envs, 287),
        },
        batch_size=[num_envs],
    )
    agent_cfg = make_rsl_rl_cfg()
    actor_cfg = agent_cfg["actor"]
    actor_cfg.pop("class_name")
    actor_cfg["primary_hidden_dims"] = [32, 16]
    actor_cfg["secondary_hidden_dims"] = [32, 16]
    actor = DualMLPModel(
        observations,
        agent_cfg["obs_groups"],
        "actor",
        26,
        **copy.deepcopy(actor_cfg),
    )
    return actor, observations


def test_dual_tracking_configuration_and_routing() -> None:
    cfg = make_rsl_rl_cfg()
    actor_cfg = cfg["actor"]
    assert cfg["experiment_name"] == "h1_with_osl_dual_full_tracking_standalone"
    assert cfg["obs_groups"]["h1_actor"] == ["actor"]
    assert cfg["obs_groups"]["osl_actor"] == ["actor"]
    assert tuple(actor_cfg["primary_action_indices"]) == H1_ACTION_INDICES
    assert tuple(actor_cfg["secondary_action_indices"]) == OSL_ACTION_INDICES
    assert cfg["algorithm"]["entropy_coef"] == 0.005

    actor, observations = _make_actor()
    assert actor.primary_actor.obs_dim == 161
    assert actor.secondary_actor.obs_dim == 161
    actions = actor(observations)
    assert torch.allclose(
        actions[:, actor.primary_action_indices], actor.primary_actor(observations)
    )
    assert torch.allclose(
        actions[:, actor.secondary_action_indices], actor.secondary_actor(observations)
    )


def test_dual_tracking_export_matches_deterministic_policy() -> None:
    actor, observations = _make_actor(num_envs=2)
    actor.eval()
    exported = torch.jit.script(actor.as_jit())

    expected = actor(observations)
    actual = exported(observations["actor"])
    assert torch.allclose(actual, expected, atol=1.0e-5)

    onnx_model = actor.as_onnx(verbose=False)
    assert onnx_model.input_names == ["obs"]
    assert onnx_model.get_dummy_inputs()[0].shape == (1, 161)


def test_dual_tracking_environment_contract() -> None:
    cfg = H1WithOslDualFullTrackingCfg(
        num_envs=2,
        device="cpu",
        seed=11,
        enable_observation_noise=False,
    )
    cfg.domain_randomization.enabled = False
    cfg.motion.sampling_mode = "start"
    env = H1WithOslDualFullTrackingEnv(cfg)
    try:
        observations = env.get_observations()
        assert observations["actor"].shape == (2, 161)
        assert observations["critic"].shape == (2, 287)
        assert tuple(env.action_joint_names[index] for index in OSL_ACTION_INDICES) == OSL_ACTION_NAMES
        assert env.h1_action_indices.tolist() == list(H1_ACTION_INDICES)
        assert env.osl_action_indices.tolist() == list(OSL_ACTION_INDICES)

        observations, rewards, dones, extras = env.step(torch.zeros(2, 26))
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(rewards).all()
        assert dones.shape == (2,)
        assert set(extras) == {"time_outs", "terminated", "log"}
        assert set(env.reward_terms()) == set(REWARD_WEIGHTS)
        assert all(
            torch.isfinite(value).all() for value in env.evaluation_metrics().values()
        )

        training_state = env.training_state()
        assert set(training_state) == {
            "common_step_counter",
            "bin_failed_count",
        }
    finally:
        env.close()
