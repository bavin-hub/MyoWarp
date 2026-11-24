"""Intact H1-2 adaptation and isolation from the existing OSL task."""

import copy
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from myowarp.tasks.h1_2_tracking import (
    H1TrackingCfg,
    H1TrackingEnv,
    H1TrackingRunner,
    make_rsl_rl_cfg,
)
from myowarp.tasks.h1_2_tracking.env import TRACKED_BODY_NAMES
from myowarp.tasks.h1_2_tracking.model import build_h1_tracking_model
from myowarp.tasks.h1_with_osl_tracking import (
    H1WithOslTrackingCfg,
    H1WithOslTrackingEnv,
)
from myowarp.tasks.h1_with_osl_velocity.model import ACTUATOR_GROUPS

MOTION = str(
    Path(__file__).resolve().parents[1]
    / "assets/reference_clips/h1_2_straight_walk_120s.npz"
)


def test_intact_model_motion_and_source_isolation():
    original_cfg = asdict(H1WithOslTrackingCfg())
    original_groups = copy.deepcopy(ACTUATOR_GROUPS)
    cfg = H1TrackingCfg(motion_path=MOTION)
    model = build_h1_tracking_model(cfg.model_path)
    assert (model.nq, model.nv, model.nu) == (34, 33, 27)
    assert len(TRACKED_BODY_NAMES) == 14
    assert all("osl" not in model.joint(i).name for i in range(model.njnt))
    assert ACTUATOR_GROUPS == original_groups
    assert asdict(H1WithOslTrackingCfg()) == original_cfg
    for name in ("left_knee_joint", "right_knee_joint"):
        a = model.actuator(name).id
        assert model.actuator_gainprm[a, 0] == 157.7
        assert model.actuator_forcerange[a, 1] == 300
    if not Path(cfg.motion_path).exists():
        pytest.skip("Prepare the local H1 walking reference first")
    with np.load(cfg.motion_path) as motion:
        assert len(motion["joint_pos"]) == 6001
        assert motion["joint_pos"].shape[1] == 27
        assert float(motion["fps"][0]) == 50
        assert tuple(motion["joint_names"]) == tuple(
            model.joint(i).name for i in range(1, model.njnt)
        )
        names = list(motion["body_names"])
        for frame in (0, 500, 6000):
            d = mujoco.MjData(model)
            pelvis = names.index("pelvis")
            d.qpos[:3] = motion["body_pos_w"][frame, pelvis]
            d.qpos[3:7] = motion["body_quat_w"][frame, pelvis]
            d.qpos[7:] = motion["joint_pos"][frame]
            mujoco.mj_forward(model, d)
            for name in TRACKED_BODY_NAMES:
                np.testing.assert_allclose(
                    d.xpos[model.body(name).id],
                    motion["body_pos_w"][frame, names.index(name)],
                    atol=2e-6,
                )


def test_unchanged_reward_sampling_and_ppo_recipe():
    from myowarp.tasks.h1_with_osl_tracking import make_rsl_rl_cfg as original_agent

    cfg = make_rsl_rl_cfg()
    old = original_agent()
    assert cfg.pop("experiment_name") != old.pop("experiment_name")
    assert cfg == old
    for name in (
        "_compute_terminations",
        "_resample_motion",
        "_apply_startup_randomization",
        "step",
    ):
        assert getattr(H1TrackingEnv, name) is getattr(H1WithOslTrackingEnv, name)
    from myowarp.tasks.h1_2_tracking.env import REWARD_WEIGHTS
    from myowarp.tasks.h1_with_osl_tracking.env import REWARD_WEIGHTS as source_weights

    differences = {k: v for k, v in REWARD_WEIGHTS.items() if v != source_weights[k]}
    assert differences == {"motion_global_root_pos": 2.0, "motion_global_root_ori": 1.0}


def test_nested_checkpoint_configuration_roundtrip():
    from myowarp.tasks.h1_2_tracking.config import config_from_dict

    original = H1TrackingCfg()
    original.domain_randomization.enabled = False
    original.motion.pose_x = (-0.02, 0.03)
    restored = config_from_dict(asdict(original))
    assert asdict(restored) == asdict(original)
    assert restored.motion.pose_x == (-0.02, 0.03)
    assert not restored.domain_randomization.enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_feedback_training_and_checkpoint(tmp_path):
    cfg = H1TrackingCfg(num_envs=16, motion_path=MOTION)
    if not Path(cfg.motion_path).exists():
        pytest.skip("Prepare the local H1 walking reference first")
    cfg.domain_randomization.enabled = False
    cfg.enable_observation_noise = False
    env = H1TrackingEnv(cfg)
    try:
        obs = env.get_observations()
        assert obs["actor"].shape == (16, 150)
        assert obs["critic"].shape == (16, 276)
        assert env.actions.shape == (16, 27)
        assert env.foot_geom_ids.numel() == 14
        env.set_motion_frame(500)
        initial = env._compute_observations()["actor"][:, 54:57].clone()
        env.qpos[:, 0] += 0.2
        env.sim.forward()
        changed = env._compute_observations()["actor"][:, 54:57]
        assert bool((torch.linalg.vector_norm(changed - initial, dim=-1) > 0.19).all())
        env.reset()
        agent = make_rsl_rl_cfg(max_iterations=1, save_interval=1)
        agent["num_steps_per_env"] = 4
        agent["algorithm"]["num_learning_epochs"] = 1
        agent["algorithm"]["num_mini_batches"] = 1
        runner = H1TrackingRunner(env, agent, str(tmp_path), device=cfg.device)
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        checkpoint = tmp_path / "model_0.pt"
        assert checkpoint.exists()
        runner.load(checkpoint, map_location=cfg.device)
        with torch.inference_mode():
            action = runner.get_inference_policy()(env.get_observations())
            assert action.shape == (16, 27)
            _, reward, _, _ = env.step(action)
            assert bool(torch.isfinite(reward).all())
        runner.logger.rewbuffer.extend([1.0] * 100)
        # Save best through the same production path, and ensure strict identity.
        runner.save(tmp_path / "model_best.pt")
        saved = torch.load(
            tmp_path / "model_best.pt", weights_only=True, map_location="cpu"
        )
        assert saved["infos"]["pipeline"] == "h1_2_tracking"
    finally:
        env.close()
