"""Removed-leg XML, full biological shadow and GPU task contracts."""

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from myowarp.musclemimic.model import build_model
from myowarp.tasks.musculoskeletal_prosthesis.config import (
    ProsthesisCfg,
    make_agent_cfg,
)
from myowarp.tasks.musculoskeletal_prosthesis.reference import load_human_clip
from myowarp.tasks.musculoskeletal_prosthesis.shadow import FooledShadow, snapshot


def removed_cfg():
    cfg = ProsthesisCfg()
    cfg.select_anatomy("removed")
    return cfg


class RemovedLegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = removed_cfg()
        cls.model = mujoco.MjModel.from_xml_path(cls.cfg.model_path)
        cls.human, _ = build_model()
        cls.clip = load_human_clip(cls.cfg.motion_path, cls.human)

    def test_structure_and_remaining_mass(self):
        m = self.model
        for name in ("tibia_r", "talus_r", "calcn_r", "toes_r", "patella_r"):
            with self.assertRaises(KeyError):
                m.body(name)
        self.assertGreater(m.body("kneeStator").id, 0)
        self.assertEqual(m.body("kneeStator").parentid[0], m.body("femur_r").id)
        self.assertEqual(m.na, 332)
        self.assertEqual(m.nu, 336)
        constrained = mujoco.MjModel.from_xml_path(ProsthesisCfg().model_path)
        for i in range(1, m.nbody):
            name = m.body(i).name
            self.assertAlmostEqual(m.body_mass[i], constrained.body(name).mass[0])
        self.assertLess(m.body_mass.sum(), constrained.body_mass.sum())
        for i in range(m.npair):
            for g in (m.pair_geom1[i], m.pair_geom2[i]):
                self.assertGreaterEqual(g, 0)

    def test_native_shadow_keeps_full_shape_and_zeros_missing_muscles(self):
        d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, d, self.model.key("bypass_stand").id)
        d.act[:] = 0.2
        d.ctrl[: self.model.na] = 0.3
        mujoco.mj_forward(self.model, d)
        before = snapshot(d)
        for mode in ("fooled", "virtual"):
            shadow = FooledShadow(self.human, self.model, self.clip, mode=mode)
            obs = shadow.observe(d, 20, before)
            self.assertEqual(obs.act.shape, (354,))
            self.assertEqual(len(shadow.actuator_ids), 332)
            missing = np.setdiff1d(np.arange(354), shadow.shadow_actuator_ids)
            self.assertEqual(len(missing), 22)
            np.testing.assert_array_equal(obs.act[missing], 0)
            np.testing.assert_array_equal(obs.ctrl[missing], 0)
            np.testing.assert_allclose(obs.act[shadow.shadow_activation_ids], 0.2)
            if mode == "fooled":
                j = self.human.joint("knee_angle_r").qposadr[0]
                self.assertEqual(obs.qpos[j], self.clip.qpos[20, j])
            else:
                self.assertEqual(
                    obs.qpos[self.human.joint("knee_angle_r").qposadr[0]],
                    d.qpos[self.model.joint("knee").qposadr[0]],
                )
            shadow.write_muscles(d, np.ones(354) * 0.5)
            np.testing.assert_allclose(d.ctrl[shadow.actuator_ids], 0.5)
            d.ctrl[:] = before["ctrl"]
            for k, v in before.items():
                np.testing.assert_array_equal(getattr(d, k), v)


@unittest.skipUnless(
    torch.cuda.is_available() and Path(removed_cfg().reference_path).is_file(),
    "CUDA and removed-leg reference required",
)
class RemovedLegGPUTests(unittest.TestCase):
    def test_gpu_both_shadow_modes_and_ppo_checkpoint(self):
        from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv
        from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner
        from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
            create_controller,
        )

        cfg = removed_cfg()
        env = GPUProsthesisVecEnv(cfg, create_controller, num_envs=2)
        for mode in ("fooled", "virtual"):
            env.human.mode = mode
            env.reset()
            before = env.data.qpos.clone()
            actions = env.human_action()
            self.assertEqual(actions.shape, (2, 354))
            missing = torch.ones(354, dtype=torch.bool, device=env.device)
            missing[env.human.shadow_actuator_ids] = False
            self.assertTrue(bool((env.human.data.act[:, missing] == 0).all()))
            self.assertTrue(bool((env.human.data.ctrl[:, missing] == 0).all()))
            torch.testing.assert_close(before, env.data.qpos)
            obs, reward, _, _ = env.step(torch.zeros((2, 2), device=env.device))
            self.assertEqual(obs["actor"].shape, (2, 115))
            self.assertEqual(obs["critic"].shape, (2, 671))
            self.assertTrue(bool(torch.isfinite(reward).all()))
        env.human.mode = cfg.right_leg_obs
        train = make_agent_cfg(save_interval=1)
        train["num_steps_per_env"] = 2
        train["algorithm"]["num_learning_epochs"] = 1
        train["algorithm"]["num_mini_batches"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            runner = ProsthesisRunner(env, train, log_dir=tmp, device=env.device)
            runner.learn(1, init_at_random_ep_len=False)
            checkpoint = str(Path(tmp) / "resume.pt")
            runner.save(checkpoint)
            with torch.inference_mode():
                runner.load(checkpoint, map_location=env.device)
        env.close()


if __name__ == "__main__":
    unittest.main()
