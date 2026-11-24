"""CMU convention parity, analytic velocity mapping, and Exp8 GPU integration."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from myowarp.tasks.musculoskeletal_prosthesis.human_space_reward import (
    CMU_SHANK,
    CMU_THIGH,
    leg_to_human,
)
from myowarp.tasks.musculoskeletal_prosthesis.split_config import SplitTrainingCfg


def cmu_position_oracle(knee, ankle):
    """Original scalar CMU formulas; independently evaluate law of cosines."""
    if knee == 0:
        return [0.0, ankle]
    lt, ls = CMU_THIGH, CMU_SHANK
    lh = (lt + ls) / 2
    chord = math.sqrt(lt**2 + ls**2 + 2 * lt * ls * math.cos(knee))
    ck = (2 * lh**2 - chord**2) / (2 * lh**2)
    kh = math.copysign(math.pi - math.acos(np.clip(ck, -1, 1)), knee)
    t1 = math.asin(np.clip(ls * math.sin(knee) / chord, -1, 1))
    return [kh, ankle + kh / 2 - knee + t1]


class HumanSpaceMappingTests(unittest.TestCase):
    def test_positions_match_cmu_convention(self):
        q = torch.tensor(
            [[k, a] for k in [-0.15, 0, 0.01, 0.4, 1.0, 1.48] for a in [-0.3, 0, 0.3]],
            dtype=torch.float64,
        )
        actual, _ = leg_to_human(q, torch.zeros_like(q))
        expected = [cmu_position_oracle(*row) for row in q.tolist()]
        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_velocity_is_derivative_including_straight_knee(self):
        q = torch.tensor(
            [[-0.05, 0.1], [0.0, 0.0], [1e-8, -0.1], [0.8, 0.2], [1.48, -0.3]],
            dtype=torch.float64,
        )
        v = torch.tensor([[0.7, -0.2]], dtype=torch.float64).expand_as(q)
        _, actual = leg_to_human(q, v)
        eps = 1e-6
        plus, _ = leg_to_human(q + eps * v, v)
        minus, _ = leg_to_human(q - eps * v, v)
        torch.testing.assert_close(
            actual, (plus - minus) / (2 * eps), atol=1e-9, rtol=1e-7
        )
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_exp8_is_separate_baseline_only(self):
        cfg = SplitTrainingCfg(prosthesis_reward="human_space")
        cfg.validate()
        self.assertEqual(cfg.experiment, 8)
        for kwargs in (
            {"reward_mode": "shared"},
            {"prosthesis_observation": "muscle_reference"},
        ):
            with self.assertRaises(ValueError):
                SplitTrainingCfg(prosthesis_reward="human_space", **kwargs).validate()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Exp8GPUTests(unittest.TestCase):
    def test_reward_mapping_isolation_native_features_and_ppo(self):
        import mujoco

        from myowarp.tasks.musculoskeletal_prosthesis.config import ROOT, ProsthesisCfg
        from myowarp.tasks.musculoskeletal_prosthesis.muscle_reward_native import (
            MimicTracking,
            Scales,
            Weights,
        )
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
        )
        cfg.select_anatomy("removed")
        training = SplitTrainingCfg(prosthesis_reward="human_space", steps_per_env=2)
        env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=2)
        with torch.inference_mode():
            obs = env.get_observations()
            self.assertEqual(obs["actor"].shape, (2, 115))
            # Nontrivial positions and velocities, including the straight knee.
            env.data.qpos[:, env.state.qids] = torch.tensor(
                [[0.0, 0.1], [0.8, -0.2]], device=env.device
            )
            env.data.qvel[:, env.state.vids] = torch.tensor(
                [[0.7, -0.3], [-0.2, 0.5]], device=env.device
            )
            env._muscle_obs = None
            env.get_observations()
            muscle_before = env.human.data.qpos.clone()
            real_before = env.data.qpos.clone()
            reward_model = env.lower_leg_tracking
            terms = reward_model.reward(env.data, env.human, env.frame)
            torch.testing.assert_close(
                muscle_before, env.human.data.qpos, rtol=0, atol=0
            )
            torch.testing.assert_close(real_before, env.data.qpos, rtol=0, atol=0)
            # Native MuJoCo oracle checks lower-leg derived site features and
            # MuscleMimic's actual frame conventions against the GPU result.
            model = env.human.sim.cpu_model
            native = object.__new__(MimicTracking)
            native.site_ids = reward_model.site_ids.cpu().numpy()
            native.body_ids = model.site_bodyid[native.site_ids]
            native.root_body_ids = model.body_rootid[native.body_ids]
            for i in range(2):
                d = mujoco.MjData(model)
                d.qpos[:] = reward_model.sim.view.qpos[i].cpu().numpy()
                d.qvel[:] = reward_model.sim.view.qvel[i].cpu().numpy()
                mujoco.mj_forward(model, d)
                rp, ra, rv = native._site_quantities(d)
                f = int(env.frame[i])
                w, s = Weights(), Scales()
                pairs = {
                    "rpos": (rp, "rpos", w.rpos, s.rpos),
                    "rquat": (ra, "rangles", w.rquat, s.rquat),
                    "rvel_rot": (rv[:, :3], "rvel", w.rvel, s.rvel),
                    "rvel_lin": (rv[:, 3:], "rvel", w.rvel, s.rvel),
                }
                for key, (actual, refkey, weight, scale) in pairs.items():
                    ref = reward_model.features[refkey][f].cpu().numpy()
                    if key == "rvel_rot":
                        ref = ref[:, :3]
                    elif key == "rvel_lin":
                        ref = ref[:, 3:]
                    expected = weight * np.exp(-scale * np.square(actual - ref).mean())
                    self.assertAlmostEqual(float(terms[key][i]), expected, delta=2e-5)
        with tempfile.TemporaryDirectory() as tmp:
            runner = SplitRunner(env, training, tmp)
            for learner in runner.learners.values():
                learner.alg.num_learning_epochs = 1
                learner.alg.num_mini_batches = 1
            runner.learn(1)
            path = Path(tmp) / "model_0.pt"
            runner.load(path)
            self.assertEqual(
                runner.training.experiment
                if hasattr(runner, "training")
                else training.experiment,
                8,
            )
            with torch.inference_mode():
                _, reward, _, extras = env.step(
                    runner.inference_policy()(env.get_observations())
                )
                self.assertTrue(bool(torch.isfinite(reward).all()))
                self.assertTrue(bool(torch.isfinite(extras["muscle_reward"]).all()))
                self.assertIn("reward/human_qpos", extras["log"])
                self.assertNotIn("reward/foot_pos", extras["log"])
        env.close()
