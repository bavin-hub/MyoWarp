"""Best checkpoint selection uses a finite, full completed-episode window."""

from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

import torch

from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner


class BestCheckpointTests(TestCase):
    def runner(self):
        runner = ProsthesisRunner.__new__(ProsthesisRunner)
        runner.logger = SimpleNamespace(
            writer=Mock(),
            log_dir="/tmp/best-checkpoint-test",
            rewbuffer=deque(maxlen=100),
        )
        runner.best_checkpoint = None
        runner.save = Mock()
        return runner

    def test_only_full_finite_improvements_are_saved(self):
        runner = self.runner()
        runner.logger.rewbuffer.extend([2.0] * 99)
        runner._save_best_checkpoint(1)
        runner.save.assert_not_called()
        runner.logger.rewbuffer.append(2.0)
        runner._save_best_checkpoint(2)
        runner.save.assert_called_once_with("/tmp/best-checkpoint-test/model_best.pt")
        self.assertEqual(runner.best_checkpoint["iteration"], 2)
        self.assertEqual(runner.best_checkpoint["mean_episode_return"], 2.0)
        for score in (2.0, 1.0, float("nan"), float("inf")):
            runner.logger.rewbuffer.clear()
            runner.logger.rewbuffer.extend([score] * 100)
            runner._save_best_checkpoint(3)
        self.assertEqual(runner.save.call_count, 1)
        runner.logger.rewbuffer.clear()
        runner.logger.rewbuffer.extend([3.0] * 100)
        runner._save_best_checkpoint(4)
        self.assertEqual(runner.save.call_count, 2)
        self.assertEqual(runner.best_checkpoint["iteration"], 4)

    def test_every_iteration_hook_including_non_save_iterations(self):
        logger = SimpleNamespace(log=Mock())

        def initialize(runner, *args, **kwargs):
            runner.logger = logger

        with patch("rsl_rl.runners.OnPolicyRunner.__init__", initialize):
            runner = ProsthesisRunner()
        runner._save_best_checkpoint = Mock()
        logger.log(it=37)
        runner._save_best_checkpoint.assert_called_once_with(37)

    def test_restored_best_score_is_respected_and_logging_can_be_disabled(self):
        runner = self.runner()
        runner.best_checkpoint = {
            "mean_episode_return": 10.0,
            "iteration": 100,
            "window_episodes": 100,
        }
        runner.logger.rewbuffer.extend([9.0] * 100)
        runner._save_best_checkpoint(101)
        runner.save.assert_not_called()
        runner.logger.writer = None
        runner.logger.rewbuffer.clear()
        runner.logger.rewbuffer.extend([11.0] * 100)
        runner._save_best_checkpoint(102)
        runner.save.assert_not_called()


@skipUnless(torch.cuda.is_available(), "CUDA required")
class BestCheckpointGPUTests(TestCase):
    def test_training_saves_and_restores_best_independently_of_periodic_interval(self):
        from myowarp.tasks.musculoskeletal_prosthesis.config import (
            ProsthesisCfg,
            make_agent_cfg,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv
        from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
            create_controller,
        )

        cfg = ProsthesisCfg(right_leg_obs="virtual", episode_seconds=0.01)
        cfg.select_anatomy("removed")
        env = GPUProsthesisVecEnv(cfg, create_controller, num_envs=128)
        agent = make_agent_cfg(save_interval=100)
        agent["num_steps_per_env"] = 2
        agent["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
        with TemporaryDirectory() as tmp:
            runner = ProsthesisRunner(env, agent, log_dir=tmp, device=env.device)
            runner.learn(2)
            path = str(Path(tmp) / "model_best.pt")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                saved["infos"]["best_checkpoint"]["iteration"], saved["iter"]
            )
            self.assertEqual(saved["infos"]["best_checkpoint"]["window_episodes"], 100)
            self.assertIn("optimizer_state_dict", saved)
            self.assertTrue((Path(tmp) / "model_0.pt").is_file())
            self.assertTrue((Path(tmp) / "model_1.pt").is_file())
            runner.load(path, map_location=env.device)
            self.assertEqual(runner.best_checkpoint, saved["infos"]["best_checkpoint"])
        env.close()
