"""Reference conditioning and pelvis-relative hip history contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from myowarp.tasks.musculoskeletal_prosthesis.prosthesis_observations import (
    hip_actor_history,
    relative_hip_rpy,
)
from myowarp.tasks.musculoskeletal_prosthesis.split_config import SplitTrainingCfg


class HipRotationTests(unittest.TestCase):
    def test_relative_angles_ignore_global_body_rotation(self):
        angles = np.array([[0.2, -0.3, 0.4], [-0.5, 0.1, -0.2]])
        relative = Rotation.from_euler("xyz", angles).as_matrix()
        pelvis = Rotation.from_euler(
            "xyz", [[0.4, 0.3, -0.6], [-0.1, 0.2, 0.7]]
        ).as_matrix()
        actual = relative_hip_rpy(torch.tensor(pelvis), torch.tensor(pelvis @ relative))
        np.testing.assert_allclose(actual.numpy(), angles, atol=1e-12)
        global_rotation = Rotation.from_euler("xyz", [0.9, -0.2, 0.4]).as_matrix()
        rotated = relative_hip_rpy(
            torch.tensor(global_rotation @ pelvis),
            torch.tensor(global_rotation @ pelvis @ relative),
        )
        torch.testing.assert_close(actual, rotated)

    def test_hip_replaces_only_pelvis_channels(self):
        history = torch.randn(2, 5, 29)
        rotations = torch.tensor(
            Rotation.random(20, random_state=12).as_matrix(), dtype=torch.float32
        ).reshape(2, 5, 2, 3, 3)
        history[..., :6] = rotations[..., 0, :, :2].flatten(-2)
        history[..., 6:12] = rotations[..., 1, :, :2].flatten(-2)
        actual = hip_actor_history(history).reshape(2, 5, 23)
        torch.testing.assert_close(
            actual[..., :3],
            relative_hip_rpy(rotations[..., 0, :, :], rotations[..., 1, :, :]),
        )
        torch.testing.assert_close(actual[..., 3:9], history[..., 6:12])
        torch.testing.assert_close(actual[..., 9:], history[..., 15:])
        history[..., 12:15] += 123  # Removed pelvis angular velocity.
        torch.testing.assert_close(actual.flatten(1), hip_actor_history(history))

    def test_singular_pose_is_finite_and_reconstructs_rotation(self):
        relative = Rotation.from_euler(
            "xyz", [[0.3, np.pi / 2, 0.5], [0.3, -np.pi / 2, 0.5]]
        ).as_matrix()
        angles = relative_hip_rpy(
            torch.eye(3, dtype=torch.float64).repeat(2, 1, 1), torch.tensor(relative)
        ).numpy()
        self.assertTrue(np.isfinite(angles).all())
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", angles).as_matrix(), relative, atol=1e-12
        )

    def test_variants_keep_separate_rewards(self):
        for variant, experiment in (
            ("reference", 6),
            ("muscle_reference", 6),
            ("hip_rpy", 7),
        ):
            cfg = SplitTrainingCfg(prosthesis_observation=variant)
            cfg.validate()
            self.assertEqual(cfg.experiment, experiment)
            cfg.reward_mode = "shared"
            with self.assertRaises(ValueError):
                cfg.validate()
        self.assertEqual(SplitTrainingCfg().experiment, 4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ObservationVariantsGPUTests(unittest.TestCase):
    def test_real_history_reference_and_ppo_roundtrip(self):
        from myowarp.tasks.musculoskeletal_prosthesis.config import ROOT, ProsthesisCfg
        from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
            create_controller,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_env import (
            SplitProsthesisEnv,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_runner import SplitRunner

        for variant, width in (
            ("reference", 119),
            ("muscle_reference", 136),
            ("hip_rpy", 115),
        ):
            with self.subTest(variant=variant):
                cfg = ProsthesisCfg(
                    checkpoint_path=str(
                        ROOT / "assets/musclemimic/checkpoints/mm-10m-2"
                    ),
                    train_muscle=True,
                    right_leg_obs="virtual",
                    muscle_learning_rate=3e-5,
                )
                cfg.select_anatomy("removed")
                training = SplitTrainingCfg(
                    prosthesis_observation=variant, steps_per_env=2
                )
                env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=2)
                env.eligible = env.eligible[
                    env.eligible < len(env.reference["qpos"]) - 100
                ]
                env.reset()
                obs = env.get_observations()
                self.assertEqual(obs["actor"].shape, (2, width))
                self.assertEqual(obs["critic"].shape, (2, 671))
                self.assertEqual(obs["muscle"].shape, (2, 2418))
                self.assertEqual(env.history.shape, (2, 5, 29))
                torch.testing.assert_close(
                    obs["actor"][:, :115], hip_actor_history(env.history)
                )
                if variant == "reference":
                    torch.testing.assert_close(
                        obs["actor"][:, 115:117],
                        env.reference["qpos"][env.frame][:, env.state.qids],
                    )
                    torch.testing.assert_close(
                        obs["actor"][:, 117:119],
                        env.reference["qvel"][env.frame][:, env.state.vids],
                    )
                elif variant == "muscle_reference":
                    adapter = env.human.adapter
                    frames = (env.frame[:, None] + adapter._lookahead).clamp(
                        max=adapter._traj_len - 1
                    )
                    expected = torch.cat(
                        (
                            env.reference["qpos"][frames][..., env.state.qids]
                            - env.data.qpos[:, None, env.state.qids],
                            env.reference["qvel"][frames][..., env.state.vids]
                            - env.data.qvel[:, None, env.state.vids],
                        ),
                        -1,
                    ).flatten(1)
                    torch.testing.assert_close(obs["actor"][:, 115:135], expected)
                    # Perturb the real joints without changing reference or phase:
                    # every future target must shift by minus that perturbation.
                    old_q = env.data.qpos[:, env.state.qids].clone()
                    old_v = env.data.qvel[:, env.state.vids].clone()
                    env.data.qpos[:, env.state.qids] = old_q + 0.125
                    env.data.qvel[:, env.state.vids] = old_v - 0.25
                    changed = env.get_observations()["actor"][:, 115:135]
                    delta = (changed - expected).reshape(2, 5, 4)
                    torch.testing.assert_close(
                        delta[..., :2], torch.full_like(delta[..., :2], -0.125)
                    )
                    torch.testing.assert_close(
                        delta[..., 2:], torch.full_like(delta[..., 2:], 0.25)
                    )
                    env.data.qpos[:, env.state.qids] = old_q
                    env.data.qvel[:, env.state.vids] = old_v
                    torch.testing.assert_close(
                        obs["actor"][:, -1], env.frame.float() / adapter._traj_len
                    )
                    self.assertEqual(adapter._lookahead.tolist(), [0, 20, 40, 60, 80])
                    # Reading augmented observations must leave the existing
                    # muscle actor input unchanged.
                    with torch.inference_mode():
                        first = obs["muscle"].clone()
                        again = env.get_observations()["muscle"]
                        torch.testing.assert_close(first, again, rtol=0, atol=0)
                else:
                    current = env.hip_rpy()
                    torch.testing.assert_close(
                        env.hip_history, current[:, None].expand(-1, 5, -1)
                    )
                    before = env.hip_history.clone()
                    env.get_observations()
                    torch.testing.assert_close(before, env.hip_history)
                    with torch.inference_mode():
                        _, _, done, _ = env.step(
                            torch.zeros((2, 334), device=env.device)
                        )
                    for i in range(2):
                        if not bool(done[i]):
                            torch.testing.assert_close(
                                env.hip_history[i, :-1], before[i, 1:]
                            )
                    torch.testing.assert_close(env.hip_history[:, -1], env.hip_rpy())
                    untouched = env.hip_history[1].clone()
                    with torch.inference_mode():
                        env.reset(torch.tensor([0], device=env.device))
                    torch.testing.assert_close(untouched, env.hip_history[1])
                    torch.testing.assert_close(
                        env.hip_history[0], env.hip_rpy()[0, None].expand(5, -1)
                    )
                with tempfile.TemporaryDirectory() as tmp:
                    runner = SplitRunner(env, training, tmp)
                    for learner in runner.learners.values():
                        learner.alg.num_learning_epochs = 1
                        learner.alg.num_mini_batches = 1
                    runner.learn(1)
                    path = Path(tmp) / "model_0.pt"
                    runner.load(path)
                    with torch.inference_mode():
                        action = runner.inference_policy()(env.get_observations())
                        self.assertEqual(action.shape, (2, 334))
                        _, reward, _, _ = env.step(action)
                        self.assertTrue(bool(torch.isfinite(reward).all()))
                env.close()
