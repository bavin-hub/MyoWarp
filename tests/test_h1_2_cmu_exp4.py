"""Functional checks for geometry adaptation and independent PPO updates."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from myowarp.tasks.h1_2_cmu_exp4.config import Exp4Cfg
from myowarp.tasks.h1_2_cmu_exp4.model import build_model
from myowarp.tasks.h1_2_cmu_exp4.reference import fingerprint, model_fingerprint


def test_adapted_reference_geometry():
    cfg = Exp4Cfg()
    if not Path(cfg.reference_path).exists():
        pytest.skip("Prepare local H1/CMU reference first")
    model = build_model()
    data = mujoco.MjData(model)
    with np.load(cfg.reference_path) as ref:
        assert str(ref["source_sha256"]) == fingerprint(cfg.motion_path)
        assert str(ref["model_sha256"]) == model_fingerprint()
        assert ref["qpos"].shape == (12001, model.nq)
        assert np.isfinite(ref["qvel"]).all()
        for name in ("knee", "ankle"):
            index = model.joint(name).qposadr[0]
            assert (ref["qpos"][:, index] >= model.joint(name).range[0] - 1e-6).all()
            assert (ref["qpos"][:, index] <= model.joint(name).range[1] + 1e-6).all()
        assert 0.3 < ref["contact"].mean() < 0.8
        for frame in (0, 137, 999, 12000):
            data.qpos[:] = ref["qpos"][frame]
            mujoco.mj_forward(model, data)
            # Closed linkage remains consistent after saving float32 poses.
            assert np.max(np.abs(data.efc_pos[:3])) < 1e-6
            ids = [model.body(str(n)).id for n in ref["body_names"]]
            np.testing.assert_allclose(
                data.xpos[ids], ref["body_pos"][frame], atol=2e-5
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_independent_updates_warmup_and_paired_reload(tmp_path):
    from myowarp.tasks.h1_2_cmu_exp4.env import Exp4Env
    from myowarp.tasks.h1_2_cmu_exp4.runner import Exp4Runner

    cfg = Exp4Cfg(num_envs=4, steps_per_env=4, critic_warmup=1)
    if not Path(cfg.reference_path).exists():
        pytest.skip("Prepare local reference first")
    env = Exp4Env(cfg)
    try:
        runner = Exp4Runner(env)
        h, p = runner.learners["h1"], runner.learners["prosthesis"]
        assert env.obs["h1_actor"].shape == (4, 150)
        assert env.obs["h1_critic"].shape == (4, 276)
        assert env.obs["actor"].shape == (4, 115)
        assert env.obs["critic"].shape == (4, 151)
        assert h.alg.storage.num_transitions_per_env == 4
        assert p.alg.storage.num_transitions_per_env == 8
        assert not (
            {id(x) for x in h.alg.actor.parameters()}
            & {id(x) for x in p.alg.actor.parameters()}
        )
        actor = {k: v.clone() for k, v in h.alg._raw_actor.state_dict().items()}
        critic = {k: v.clone() for k, v in h.alg._raw_critic.state_dict().items()}
        prost = {k: v.clone() for k, v in p.alg._raw_actor.state_dict().items()}
        runner.learn(1)
        for k, v in actor.items():
            torch.testing.assert_close(
                v, h.alg._raw_actor.state_dict()[k], rtol=0, atol=0
            )
        assert any(
            not torch.equal(v, h.alg._raw_critic.state_dict()[k])
            for k, v in critic.items()
        )
        assert any(
            not torch.equal(v, p.alg._raw_actor.state_dict()[k])
            for k, v in prost.items()
        )
        runner.learn(2)
        assert any(
            not torch.equal(v, h.alg._raw_actor.state_dict()[k])
            for k, v in actor.items()
        )
        for k, v in actor.items():
            if k.startswith("obs_normalizer"):
                torch.testing.assert_close(
                    v, h.alg._raw_actor.state_dict()[k], rtol=0, atol=0
                )
        assert cfg.human_lr_min <= h.alg.learning_rate <= cfg.human_lr_max
        for hq, hv, cq, cv, sign in env.mapping:
            torch.testing.assert_close(
                env.shadow.qpos[:, hq], sign * env.data.qpos[:, cq]
            )
        assert not env.shadow.qpos[:, env.roll.qposadr[0]].any()
        saved = tmp_path / "paired.pt"
        runner.save(saved)
        before = {
            name: {k: v.clone() for k, v in r.alg._raw_actor.state_dict().items()}
            for name, r in runner.learners.items()
        }
        with torch.no_grad():
            next(h.alg.actor.parameters()).add_(1)
        runner.load(saved)
        assert h.alg.n_updates == 2
        for name, r in runner.learners.items():
            for k, v in before[name].items():
                torch.testing.assert_close(
                    v, r.alg._raw_actor.state_dict()[k], rtol=0, atol=0
                )
        assert env.half == 0
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_evaluation_preserves_physics():
    from myowarp.tasks.h1_2_cmu_exp4.env import Exp4Env

    cfg = Exp4Cfg(num_envs=1)
    env = Exp4Env(cfg, evaluation=True)
    try:
        h = torch.zeros(1, 27, device=env.device)
        p = torch.tensor([[0.1, -0.1]], device=env.device)

        def rollout():
            env.reset(0)
            positions = []
            velocities = []
            commands = []
            with torch.inference_mode():
                for _ in range(20):
                    obs, _, _, _ = env.step(h, p)
                    positions.append(env.data.qpos.clone())
                    velocities.append(env.data.qvel.clone())
                    commands.append(env.data.ctrl.clone())
                    assert torch.isfinite(obs["actor"]).all()
            return tuple(torch.stack(a) for a in (positions, velocities, commands))

        slow = rollout()
        env.enable_fast_evaluation()
        fast = rollout()
        # Fused arithmetic and GPU contact reductions are not bitwise identical.
        torch.testing.assert_close(slow[0], fast[0], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(slow[1], fast[1], atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(slow[2], fast[2], atol=0.1, rtol=3e-3)
    finally:
        env.close()


def test_native_cpu_evaluation_does_not_need_cuda(monkeypatch):
    from myowarp.tasks.h1_2_cmu_exp4.env import Exp4Env
    from myowarp.tasks.h1_2_cmu_exp4.runner import Exp4Runner

    cfg = Exp4Cfg(num_envs=1, device="cpu")

    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU evaluation attempted to use CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", no_cuda)
    monkeypatch.setattr(torch.cuda, "current_stream", no_cuda)
    torch.set_num_threads(1)
    with pytest.raises(ValueError, match="CPU"):
        cfg.validate()  # CPU remains disallowed for training.
    env = Exp4Env(cfg, evaluation=True)
    try:
        runner = Exp4Runner(env)
        obs = env.reset(0)
        with torch.inference_mode():
            for _ in range(4):
                actions = runner.inference_actions(obs)
                obs, _, _, _ = env.step(actions["h1"], actions["prosthesis"])
        assert env.data.qpos.device.type == "cpu"
        assert next(runner.learners["h1"].alg.actor.parameters()).device.type == "cpu"
        assert (
            next(runner.learners["prosthesis"].alg.actor.parameters()).device.type
            == "cpu"
        )
        assert env.controller.previous_tau.device.type == "cpu"
        assert env.sim.native_data.time == pytest.approx(0.04)
        assert obs["h1_actor"].dtype == torch.float32
        assert torch.isfinite(obs["actor"]).all()
        assert (
            env.controller.ticks.item()
            == env.controller.common["startup_mute_ticks"] + 40
        )
    finally:
        env.close()
