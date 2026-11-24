"""Experiment 4: source reward parity and independent two-critic GPU learning."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from myowarp.musclemimic.model import build_model
from myowarp.tasks.musculoskeletal_prosthesis.muscle_reward_gpu import (
    MuscleMimicRewardBatch,
)
from myowarp.tasks.musculoskeletal_prosthesis.split_config import SplitTrainingCfg
from myowarp.tasks.musculoskeletal_prosthesis.split_env import PROSTHESIS_REWARD_TERMS


class SplitRewardTests(unittest.TestCase):
    def test_masked_reward_matches_recorded_prosim_oracle(self):
        fixture = np.load(Path(__file__).parent / "fixtures/exp4_mimic_reward.npz")
        model, _ = build_model()
        clip = SimpleNamespace(
            qpos=fixture["reference_qpos"], qvel=fixture["reference_qvel"]
        )
        reward = MuscleMimicRewardBatch(model, clip, "cpu")
        data = SimpleNamespace(
            **{
                key: torch.tensor(fixture[key], dtype=torch.float32)
                for key in (
                    "qpos",
                    "qvel",
                    "site_xpos",
                    "site_xmat",
                    "cvel",
                    "subtree_com",
                )
            }
        )
        actual, terms, terminated = reward.reward(
            data,
            torch.tensor(fixture["frames"]),
            torch.tensor(fixture["actions"], dtype=torch.float32),
        )
        np.testing.assert_allclose(
            actual.numpy(), fixture["rewards"], atol=2e-5, rtol=3e-4
        )
        np.testing.assert_allclose(
            torch.stack([terms[k] for k in fixture["term_names"]], -1).numpy(),
            fixture["terms"],
            atol=2e-5,
            rtol=3e-4,
        )
        np.testing.assert_array_equal(terminated.numpy(), fixture["terminated"])
        masked = {
            model.joint(j).name
            for j in range(model.njnt)
            if model.jnt_qposadr[j] in set(reward.native.qpos_ind)
        }
        self.assertNotIn("knee_angle_r", masked)
        self.assertNotIn("knee_angle_beta_rotation1_r", masked)
        self.assertIn("hip_flexion_r", masked)
        self.assertNotIn(model.site("right_ankle_mimic").id, reward.native.site_ids)
        self.assertIn(model.site("right_knee_mimic").id, reward.native.site_ids)

    def test_prosthesis_reward_excludes_body_and_muscle_terms(self):
        self.assertFalse(
            set(PROSTHESIS_REWARD_TERMS)
            & {
                "root_pos",
                "root_ori",
                "root_vel",
                "body_pos",
                "body_ori",
                "body_vel",
                "body_ang_vel",
            }
        )
        self.assertIn("foot_pos", PROSTHESIS_REWARD_TERMS)
        self.assertIn("effort", PROSTHESIS_REWARD_TERMS)

    def test_finetune_configuration_validation(self):
        SplitTrainingCfg().validate()
        for cfg in (
            SplitTrainingCfg(critic_warmup=-1),
            SplitTrainingCfg(muscle_lr_max=1e-6),
            SplitTrainingCfg(muscle_std_scale=0),
        ):
            with self.assertRaises(ValueError):
                cfg.validate()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SplitGPUTests(unittest.TestCase):
    def test_shared_reward_keeps_independent_critics_and_ppo_buffers(self):
        from myowarp.tasks.musculoskeletal_prosthesis.config import ROOT, ProsthesisCfg
        from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
            create_controller,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_env import (
            SplitProsthesisEnv,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_runner import SplitRunner

        cfg = ProsthesisCfg(
            checkpoint_path=str(ROOT / "assets/musclemimic/checkpoints/mm-10m-2"),
            train_muscle=True,
            right_leg_obs="virtual",
            muscle_learning_rate=3e-5,
        )
        cfg.select_anatomy("removed")
        training = SplitTrainingCfg(
            reward_mode="shared", critic_warmup=0, steps_per_env=2
        )
        env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=2)
        env.eligible = env.eligible[env.eligible < len(env.reference["qpos"]) - 100]
        env.reset()
        with tempfile.TemporaryDirectory() as tmp:
            runner = SplitRunner(env, training, tmp)
            for learner in runner.learners.values():
                learner.alg.num_learning_epochs = 1
                learner.alg.num_mini_batches = 1
            with torch.inference_mode():
                policy = runner.inference_policy()
                _, shared, _, extras = env.step(policy(env.get_observations()))
                torch.testing.assert_close(
                    shared,
                    extras["muscle_component_reward"]
                    + extras["prosthesis_component_reward"],
                )
                torch.testing.assert_close(
                    shared, extras["muscle_reward"], rtol=0, atol=0
                )
            muscle_before = runner.muscle.alg.actor.mlp.output.weight.detach().clone()
            prost_before = runner.prosthesis.alg.actor.mlp[0].weight.detach().clone()
            runner.learn(1)
            self.assertIsNot(
                runner.muscle.alg.optimizer, runner.prosthesis.alg.optimizer
            )
            self.assertNotEqual(
                runner.muscle.alg.storage.advantages.data_ptr(),
                runner.prosthesis.alg.storage.advantages.data_ptr(),
            )
            torch.testing.assert_close(
                runner.muscle.alg.storage.rewards,
                runner.prosthesis.alg.storage.rewards,
                rtol=0,
                atol=0,
            )
            self.assertFalse(
                torch.equal(
                    runner.muscle.alg.storage.values,
                    runner.prosthesis.alg.storage.values,
                )
            )
            self.assertFalse(
                torch.equal(muscle_before, runner.muscle.alg.actor.mlp.output.weight)
            )
            self.assertFalse(
                torch.equal(prost_before, runner.prosthesis.alg.actor.mlp[0].weight)
            )
            checkpoint = Path(tmp) / "model_0.pt"
            training.reward_mode = "separate"
            with self.assertRaisesRegex(ValueError, "reward mode"):
                runner.load(checkpoint)
            training.reward_mode = "shared"
            runner.load(checkpoint)
            self.assertEqual(runner.muscle.alg.n_updates, 1)
        env.close()

    def test_pretrained_critic_warmup_separate_rewards_and_roundtrip(self):
        from myowarp.musclemimic.policy import _actor_forward
        from myowarp.tasks.musculoskeletal_prosthesis.artifacts import frozen_artifacts
        from myowarp.tasks.musculoskeletal_prosthesis.config import ROOT, ProsthesisCfg
        from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
            create_controller,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_env import (
            SplitProsthesisEnv,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_runner import SplitRunner

        cfg = ProsthesisCfg(
            checkpoint_path=str(ROOT / "assets/musclemimic/checkpoints/mm-10m-2"),
            train_muscle=True,
            right_leg_obs="virtual",
            episode_seconds=0.01,
            muscle_learning_rate=3e-5,
        )
        cfg.select_anatomy("removed")
        training = SplitTrainingCfg(critic_warmup=1, steps_per_env=2)
        env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=64)
        with tempfile.TemporaryDirectory() as tmp:
            runner = SplitRunner(env, training, tmp)
            # Keep update tests short; production defaults are unchanged.
            for learner in runner.learners.values():
                learner.alg.num_learning_epochs = 1
                learner.alg.num_mini_batches = 1
            obs = env.get_observations()
            artifacts, _ = frozen_artifacts(cfg.checkpoint_path)
            norm = (obs["muscle"].cpu().numpy() - artifacts.obs_mean) / np.sqrt(
                artifacts.obs_var + 1e-8
            )
            reference_critic = _actor_forward(
                {"actor": artifacts.params["critic"]}, norm
            )
            with torch.no_grad():
                actual_critic = runner.muscle.alg.critic(
                    runner.muscle.env.observations(obs)
                )
            np.testing.assert_allclose(
                actual_critic.cpu(), reference_critic, rtol=2e-4, atol=2e-4
            )
            actor_before = runner.muscle.alg.actor.mlp.output.weight.detach().clone()
            std_before = (
                runner.muscle.alg.actor.distribution.log_std_param.detach().clone()
            )
            critic_before = runner.muscle.alg.critic.mlp.output.weight.detach().clone()
            prost_before = runner.prosthesis.alg.actor.mlp[0].weight.detach().clone()
            muscle_parameters = {
                id(p)
                for m in (runner.muscle.alg.actor, runner.muscle.alg.critic)
                for p in m.parameters()
            }
            prost_parameters = {
                id(p)
                for m in (runner.prosthesis.alg.actor, runner.prosthesis.alg.critic)
                for p in m.parameters()
            }
            self.assertFalse(muscle_parameters & prost_parameters)
            runner.learn(1)
            torch.testing.assert_close(
                actor_before, runner.muscle.alg.actor.mlp.output.weight, rtol=0, atol=0
            )
            torch.testing.assert_close(
                std_before,
                runner.muscle.alg.actor.distribution.log_std_param,
                rtol=0,
                atol=0,
            )
            self.assertFalse(
                torch.equal(critic_before, runner.muscle.alg.critic.mlp.output.weight)
            )
            self.assertFalse(
                torch.equal(prost_before, runner.prosthesis.alg.actor.mlp[0].weight)
            )
            self.assertEqual(runner.muscle.alg.n_updates, 1)
            self.assertIsNotNone(runner.best["muscle"])
            self.assertTrue((Path(tmp) / "model_best_muscle.pt").is_file())
            self.assertTrue((Path(tmp) / "model_best_prosthesis.pt").is_file())
            runner.learn(2)
            self.assertFalse(
                torch.equal(actor_before, runner.muscle.alg.actor.mlp.output.weight)
            )
            self.assertEqual(runner.muscle.alg.n_updates, 2)
            for lr in (1.0, 1e-10):
                runner.muscle.alg.learning_rate = lr
                self.assertTrue(1e-6 <= runner.muscle.alg.learning_rate <= 1e-4)
            path = Path(tmp) / "model_1.pt"
            expected = runner.muscle.alg.actor.mlp.output.weight.detach().clone()
            with torch.no_grad():
                runner.muscle.alg.actor.mlp.output.weight.zero_()
            runner.load(path)
            torch.testing.assert_close(
                expected, runner.muscle.alg.actor.mlp.output.weight
            )
            self.assertEqual(runner.iteration, 1)
            self.assertEqual(runner.muscle.alg.n_updates, 2)
            with torch.inference_mode():
                policy = runner.inference_policy()
                action = policy(env.get_observations())
                self.assertEqual(action.shape, (64, 334))
                _, prosthesis_reward, _, extras = env.step(action)
                self.assertTrue(bool(torch.isfinite(extras["muscle_reward"]).all()))
                self.assertFalse(
                    torch.equal(prosthesis_reward, extras["muscle_reward"])
                )
        env.close()
