"""Joint-action routing, actual PPO updates, and checkpoint round trip."""

import tempfile
import unittest
from pathlib import Path

import torch

from myowarp.tasks.musculoskeletal_prosthesis.config import (
    ROOT,
    ProsthesisCfg,
    make_agent_cfg,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CooptTests(unittest.TestCase):
    def test_independent_frozen_and_joint_updates(self):
        from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv
        from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner
        from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
            create_controller,
        )

        default = str(ROOT / "assets/musclemimic/checkpoints/mm-10m-2")
        vyvas = str(ROOT / "vyvas_pi_h/model_1499.pt")
        for checkpoint, coopt in ((default, False), (vyvas, False), (default, True)):
            with self.subTest(checkpoint=checkpoint, coopt=coopt):
                cfg = ProsthesisCfg(
                    checkpoint_path=checkpoint,
                    train_muscle=coopt,
                    right_leg_obs="virtual",
                )
                cfg.select_anatomy("removed")
                env = GPUProsthesisVecEnv(cfg, create_controller, num_envs=2)
                train = make_agent_cfg(save_interval=1)
                train["num_steps_per_env"] = 4
                train["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
                if coopt:
                    train["algorithm"]["class_name"] = (
                        "myowarp.tasks.musculoskeletal_prosthesis.coopt:CooptPPO"
                    )
                frozen = env.human.actor.output.weight.detach().clone()
                with tempfile.TemporaryDirectory() as tmp:
                    runner = ProsthesisRunner(
                        env, train, log_dir=tmp, device=env.device
                    )
                    if coopt:
                        obs = env.get_observations()
                        torch.testing.assert_close(
                            obs["muscle"], env.get_observations()["muscle"]
                        )
                        with torch.no_grad():
                            joint = runner.alg.actor(obs)
                            expected = env.human.actor.forward_normalized(
                                obs["muscle"]
                            ).clamp(-1, 1)
                        torch.testing.assert_close(
                            joint[:, 2:], expected[:, env.trainable_muscle_ids]
                        )
                        self.assertEqual(joint.shape, (2, 334))
                        muscle_before = (
                            runner.alg.actor.muscle.output.weight.detach().clone()
                        )
                        prosthesis_before = (
                            runner.alg.actor.mlp[0].weight.detach().clone()
                        )
                        self.assertEqual(runner.alg.schedule, "fixed")
                        self.assertEqual(
                            [g["lr"] for g in runner.alg.optimizer.param_groups],
                            [1e-3, 1e-5],
                        )
                    runner.learn(2, init_at_random_ep_len=False)
                    # The shadow's inference copy is never optimized, even in coopt.
                    torch.testing.assert_close(frozen, env.human.actor.output.weight)
                    if coopt:
                        self.assertFalse(
                            torch.equal(
                                muscle_before, runner.alg.actor.muscle.output.weight
                            )
                        )
                        self.assertFalse(
                            torch.equal(
                                prosthesis_before, runner.alg.actor.mlp[0].weight
                            )
                        )
                        limits = env.human.ctrl_range[env.human.shadow_actuator_ids]
                        commands = env.data.ctrl[:, env.human.actuator_ids]
                        self.assertTrue(bool((commands >= limits[:, 0]).all()))
                        self.assertTrue(bool((commands <= limits[:, 1]).all()))
                        saved = runner.alg.actor.muscle.output.weight.detach().clone()
                        path = str(Path(tmp) / "roundtrip.pt")
                        runner.save(path)
                        with torch.no_grad():
                            runner.alg.actor.muscle.output.weight.zero_()
                        runner.load(path, map_location=env.device)
                        torch.testing.assert_close(
                            saved, runner.alg.actor.muscle.output.weight
                        )
                        # Deterministic eval also returns and steps the full joint action.
                        with torch.inference_mode():
                            policy = runner.get_inference_policy()
                            _, reward, _, _ = env.step(policy(env.get_observations()))
                        self.assertTrue(bool(torch.isfinite(reward).all()))
                env.close()


if __name__ == "__main__":
    unittest.main()
