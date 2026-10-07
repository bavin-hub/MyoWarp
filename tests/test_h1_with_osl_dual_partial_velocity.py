"""Contracts for dual-policy partial-observation velocity tracking."""

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
)
from myowarp.tasks.h1_with_osl_dual_partial_velocity import (
    H1WithOslDualPartialVelocityCfg,
    H1WithOslDualPartialVelocityEnv,
    make_rsl_rl_cfg,
)
from myowarp.tasks.h1_with_osl_dual_partial_velocity.env import (
    ACTOR_JOINT_POSITION_OFFSET,
    ACTOR_JOINT_VELOCITY_OFFSET,
)
from myowarp.tasks.h1_with_osl_velocity.env import REWARD_WEIGHTS
from rsl_rl.models import DualMLPModel


def test_partial_dual_pipeline_has_no_mjlab_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sources = list(
        (repo_root / "myowarp/tasks/h1_with_osl_dual_partial_velocity").glob("*.py")
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
            "actor": torch.randn(num_envs, 89),
            "osl_actor": torch.randn(num_envs, 4),
            "critic": torch.randn(num_envs, 104),
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


def test_partial_dual_configuration_and_actor_routing() -> None:
    cfg = make_rsl_rl_cfg()
    assert cfg["experiment_name"] == "h1_with_osl_dual_partial_velocity_standalone"
    assert cfg["obs_groups"]["h1_actor"] == ["actor"]
    assert cfg["obs_groups"]["osl_actor"] == ["osl_actor"]
    assert tuple(cfg["actor"]["primary_action_indices"]) == H1_ACTION_INDICES
    assert tuple(cfg["actor"]["secondary_action_indices"]) == OSL_ACTION_INDICES

    actor, observations = _make_actor()
    assert actor.primary_actor.obs_dim == 89
    assert actor.secondary_actor.obs_dim == 4
    actions = actor(observations)
    assert torch.allclose(
        actions[:, actor.primary_action_indices],
        actor.primary_actor(observations),
    )
    assert torch.allclose(
        actions[:, actor.secondary_action_indices],
        actor.secondary_actor(observations),
    )


def test_partial_dual_export_uses_separate_observations() -> None:
    actor, observations = _make_actor(num_envs=2)
    actor.eval()
    exported = torch.jit.script(actor.as_jit())

    expected = actor(observations)
    actual = exported(observations["actor"], observations["osl_actor"])
    assert torch.allclose(actual, expected, atol=1.0e-5)

    onnx_model = actor.as_onnx(verbose=False)
    assert onnx_model.input_names == ["primary_obs", "secondary_obs"]
    assert onnx_model.get_dummy_inputs()[0].shape == (1, 89)
    assert onnx_model.get_dummy_inputs()[1].shape == (1, 4)


def test_partial_environment_observation_and_velocity_contract() -> None:
    cfg = H1WithOslDualPartialVelocityCfg(
        num_envs=2,
        device="cpu",
        seed=9,
        enable_observation_noise=True,
    )
    cfg.domain_randomization.enabled = False
    env = H1WithOslDualPartialVelocityEnv(cfg)
    try:
        observations = env.get_observations()
        assert observations["actor"].shape == (2, 89)
        assert observations["osl_actor"].shape == (2, 4)
        assert observations["critic"].shape == (2, 104)

        osl_start, osl_stop = OSL_ACTION_INDICES[0], OSL_ACTION_INDICES[-1] + 1
        expected_partial = torch.cat(
            (
                observations["actor"][
                    :,
                    ACTOR_JOINT_POSITION_OFFSET
                    + osl_start : ACTOR_JOINT_POSITION_OFFSET
                    + osl_stop,
                ],
                observations["actor"][
                    :,
                    ACTOR_JOINT_VELOCITY_OFFSET
                    + osl_start : ACTOR_JOINT_VELOCITY_OFFSET
                    + osl_stop,
                ],
            ),
            dim=1,
        )
        assert torch.equal(observations["osl_actor"], expected_partial)

        observations, rewards, dones, extras = env.step(torch.zeros(2, 26))
        assert torch.isfinite(observations["osl_actor"]).all()
        assert torch.isfinite(rewards).all()
        assert dones.shape == (2,)
        assert set(extras) == {"time_outs", "log"}
        assert set(env.reward_terms()) == set(REWARD_WEIGHTS)
    finally:
        env.close()
