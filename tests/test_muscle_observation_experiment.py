"""Check observation-only mutations never suppress physical muscle control."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from myowarp.musclemimic.model import build_model
from myowarp.musclemimic.observation_experiment import WholeBodyObservationShadow
from myowarp.tasks.musculoskeletal_prosthesis.config import ProsthesisCfg
from myowarp.tasks.musculoskeletal_prosthesis.reference import load_human_clip
from myowarp.tasks.musculoskeletal_prosthesis.shadow import snapshot


@unittest.skipUnless(Path(ProsthesisCfg().motion_path).is_file(), "Local clip required")
class ObservationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = build_model(model_version="1.0.5")
        cls.clip = load_human_clip(ProsthesisCfg().motion_path, cls.model)

    def state(self):
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.clip.qpos[10]
        d.qvel[:] = self.clip.qvel[10]
        d.act[:] = 0.25
        d.ctrl[:] = 0.4
        mujoco.mj_forward(m, d)
        return d

    def test_baseline_retains_all_activation_observations(self):
        d = self.state()
        shadow = WholeBodyObservationShadow(self.model, self.clip, "baseline")
        observed = shadow.observe(d, 10, snapshot(d))
        np.testing.assert_array_equal(observed.act, d.act)
        np.testing.assert_array_equal(observed.ctrl, d.ctrl)
        np.testing.assert_array_equal(observed.qpos, d.qpos)

    def test_fooled_zeroes_shadow_only_and_uses_reference(self):
        d = self.state()
        shadow = WholeBodyObservationShadow(self.model, self.clip, "fooled")
        original = snapshot(d)
        observed = shadow.observe(d, 20, original)
        np.testing.assert_array_equal(observed.act[shadow.mask], 0)
        np.testing.assert_array_equal(observed.ctrl[shadow.mask], 0)
        for key, value in original.items():
            np.testing.assert_array_equal(getattr(d, key), value)
        for name in (
            "knee_angle_r",
            "ankle_angle_r",
            "subtalar_angle_r",
            "mtp_angle_r",
        ):
            i = self.model.joint(name).qposadr[0]
            self.assertAlmostEqual(observed.qpos[i], self.clip.qpos[20, i])
        keep = self.model.actuator("recfem_r").id
        self.assertEqual(observed.act[keep], 0.25)

    def test_virtual_retains_actual_knee_ankle_with_zero_shadow_activation(self):
        d = self.state()
        for name, value in [("knee_angle_r", 0.6), ("ankle_angle_r", 0.1)]:
            d.qpos[self.model.joint(name).qposadr[0]] = value
        shadow = WholeBodyObservationShadow(self.model, self.clip, "virtual")
        original = snapshot(d)
        observed = shadow.observe(d, 20, original)
        for name in ("knee_angle_r", "ankle_angle_r"):
            j = self.model.joint(name)
            self.assertEqual(observed.qpos[j.qposadr[0]], d.qpos[j.qposadr[0]])
            self.assertEqual(observed.qvel[j.dofadr[0]], d.qvel[j.dofadr[0]])
        for name in ("subtalar_angle_r", "mtp_angle_r"):
            j = self.model.joint(name)
            self.assertEqual(observed.qpos[j.qposadr[0]], 0)
            self.assertEqual(observed.qvel[j.dofadr[0]], 0)
        np.testing.assert_array_equal(observed.act[shadow.mask], 0)
        for key, value in original.items():
            np.testing.assert_array_equal(getattr(d, key), value)


if __name__ == "__main__":
    unittest.main()
