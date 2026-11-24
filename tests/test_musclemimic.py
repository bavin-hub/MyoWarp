"""Model and checkpoint conformance checks; generated poses are not walking validation."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from myowarp.musclemimic.model import build_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "assets/musclemimic/checkpoints/mm-10m-2"


@unittest.skipUnless(tuple(map(int, mujoco.__version__.split('.')[:2])) >= (3, 11),
                     "Run with the separate MuJoCo 3.11 evaluation environment")
class MuscleMimicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = build_model()

    def test_full_biological_body_and_muscle_control(self):
        model = self.model
        self.assertEqual((model.nq, model.nv, model.nu), (89, 88, 354))
        np.testing.assert_array_equal(model.actuator_dyntype, mujoco.mjtDyn.mjDYN_MUSCLE)
        self.assertGreater(model.body("head").id, 0)
        self.assertGreater(model.body("toes_r").id, 0)
        self.assertGreater(model.body("toes_l").id, 0)

    def test_checkpoint_specific_knee_coupling(self):
        eq = self.model.equality("knee_angle_translation2_constraint_l").id
        np.testing.assert_allclose(self.model.eq_data[eq, :5],
                                   [-7.69254e-11, -0.00587971, 0.00125622, 2.61846e-06, -6.24355e-07])

    @unittest.skipUnless((CHECKPOINT / "train_state").is_dir(), "Download checkpoint to test inference")
    def test_checkpoint_observation_history_and_forward_dynamics(self):
        from myowarp.musclemimic.policy import (
            FullbodyObsAdapter, LocalPolicyRunner,
            fullbody_history_settings_from_metadata, fullbody_obs_adapter_params_from_metadata,
            load_local_policy_artifacts, read_checkpoint_config_metadata,
        )
        from myowarp.musclemimic.trajectory import expand_motion_clip_to_model, load_motion_clip

        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # A stationary synthetic target exercises interfaces only; no claim that
        # the pretrained locomotion policy can balance against this target.
        fields = {key: np.repeat(np.asarray(getattr(data, key))[None], 120, axis=0)
                  for key in ["qpos", "qvel", "site_xpos", "site_xmat", "cvel", "subtree_com"]}
        fields.update(site_names=np.array([model.site(i).name or "" for i in range(model.nsite)]),
                      site_bodyid=model.site_bodyid, body_rootid=model.body_rootid, frequency_hz=100.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.npz"
            np.savez(path, **fields)
            clip = expand_motion_clip_to_model(
                load_motion_clip(path, expected_nq=model.nq, expected_nv=model.nv), model)
            meta = read_checkpoint_config_metadata(CHECKPOINT)
            runner = LocalPolicyRunner(
                artifacts=load_local_policy_artifacts(CHECKPOINT), stochastic=False, seed=0,
                obs_adapter=FullbodyObsAdapter(model, clip,
                    goal_params=fullbody_obs_adapter_params_from_metadata(meta)),
                **fullbody_history_settings_from_metadata(meta))
            initial = data.qpos.copy()
            for frame in range(3):
                action = runner.action_for(data=data, clip=clip, frame_idx=frame)
                self.assertEqual(action.shape, (354,))
                self.assertTrue(np.isfinite(action).all())
                runner.step(model, data, action)
            self.assertTrue(np.isfinite(data.qpos).all())
            self.assertAlmostEqual(data.time, .03)
            self.assertFalse(np.array_equal(initial, data.qpos))
            np.testing.assert_array_equal(clip.qpos[0], initial)


if __name__ == "__main__":
    unittest.main()
