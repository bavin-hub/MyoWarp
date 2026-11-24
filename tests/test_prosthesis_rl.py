"""Contract checks; test controllers are never exposed by the training CLI."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from myowarp.musclemimic.model import build_model
from myowarp.tasks.musculoskeletal_prosthesis.config import (
    ProsthesisCfg,
    make_agent_cfg,
)
from myowarp.tasks.musculoskeletal_prosthesis.controller import (
    MissingLowLevelController,
    MotorCommand,
    load_controller_factory,
)
from myowarp.tasks.musculoskeletal_prosthesis.cpu_oracle import (
    CPUReferenceOracle as ProsthesisEnv,
)
from myowarp.tasks.musculoskeletal_prosthesis.reference import (
    prepare_reference,
    read_reference,
)
from myowarp.tasks.musculoskeletal_prosthesis.state import sensor_frame, tracking_reward


class ControllerBoundaryTests(unittest.TestCase):
    def test_missing_controller_is_an_explicit_error(self):
        with self.assertRaises(MissingLowLevelController):
            load_controller_factory(None)

    def test_only_single_prosthesis_actor_is_configured(self):
        config = make_agent_cfg()
        self.assertEqual(
            config["obs_groups"], {"actor": ["actor"], "critic": ["critic"]}
        )
        self.assertNotIn("Dual", config["actor"]["class_name"])
        self.assertGreater(config["save_interval"], 0)


@unittest.skipUnless(
    tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 11),
    "Needs native MuJoCo 3.11",
)
class ProsthesisTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(ProsthesisCfg().motion_path).is_file():
            raise unittest.SkipTest("Generate/download the walking reference first")
        cls.directory = tempfile.TemporaryDirectory()
        cls.cfg = replace(
            ProsthesisCfg(),
            reference_path=str(Path(cls.directory.name) / "reference.npz"),
        )
        prepare_reference(
            cls.cfg,
            build_model()[0],
            mujoco.MjModel.from_xml_path(cls.cfg.model_path),
            max_frames=30,
        )
        cls.env = ProsthesisEnv(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.env.controller = None
        self.env.reset(frame=5)

    def test_shadow_fooling_does_not_change_real_body(self):
        env = self.env
        original = env.data.qpos.copy()
        shadow = env.shadow.observe(env.data, env.frame, env.pre_substep)
        human_q = env.human_model.joint("knee_angle_r").qposadr[0]
        self.assertAlmostEqual(shadow.qpos[human_q], env.clip.qpos[env.frame, human_q])
        np.testing.assert_array_equal(env.data.qpos, original)
        self.assertAlmostEqual(
            env.data.qpos[env.model.joint("knee_angle_r").qposadr[0]],
            np.pi / 2,
            places=4,
        )

    def test_mask_preserves_hip_muscle_and_prosthetic_channels(self):
        env = self.env
        motor_ids = [
            env.model.actuator(n).id
            for n in ("knee_act", "ankle_act", "knee_vel", "ankle_vel")
        ]
        env.data.ctrl[motor_ids] = 0.17
        env.shadow.write_muscles(env.data, np.full(354, 0.5))
        np.testing.assert_array_equal(
            env.data.ctrl[env.shadow.actuator_ids[env.shadow.mask]], 0.0
        )
        self.assertEqual(env.data.ctrl[env.model.actuator("recfem_r").id], 0.5)
        np.testing.assert_array_equal(env.data.ctrl[motor_ids], 0.17)

    def test_actor_excludes_sea_and_muscle_signals(self):
        env = self.env
        first = sensor_frame(env.model, env.data, np.zeros(2), env.cfg)
        env.data.qpos[env.model.joint("knee_motor").qposadr[0]] += 0.1
        env.data.qvel[env.model.joint("ankle_motor").dofadr[0]] += 2
        env.data.act[:] = 0.7
        env.data.ctrl[:] = 0.9
        second = sensor_frame(env.model, env.data, np.zeros(2), env.cfg)
        self.assertEqual(first.shape, (29,))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(env.observations()["actor"].shape, (145,))

    def test_reward_reads_actual_body_even_when_shadow_is_fooled(self):
        env = self.env
        args = (
            env.model,
            env.data,
            env.reference,
            env.frame,
            np.zeros(2),
            np.zeros(2),
            env.cfg,
        )
        _, before = tracking_reward(*args)
        env.data.qpos[0] += 0.5
        mujoco.mj_forward(env.model, env.data)
        env.shadow.observe(env.data, env.frame, env.pre_substep)
        _, after = tracking_reward(*args)
        self.assertLess(after["root_pos"], before["root_pos"])
        self.assertLess(after["foot_pos"], before["foot_pos"])

    def test_controller_is_required_before_forward_motion(self):
        env = self.env
        qpos = env.data.qpos.copy()
        with self.assertRaises(MissingLowLevelController):
            env.step(np.zeros(2))
        np.testing.assert_array_equal(qpos, env.data.qpos)

    def test_exactly_ten_low_level_updates_and_torque_scaling(self):
        env = self.env

        class RecordingController:
            def __init__(self):
                self.requests = []

            def reset(self, model, data):
                self.requests.clear()

            def update(self, model, data, desired_joint_torque, dt):
                self.requests.append((desired_joint_torque, dt))
                return MotorCommand(np.array([0.1, 0.2]), np.zeros(2))

        controller = RecordingController()
        env.controller = controller
        # A deliberate test double isolates timing/routing, not gait performance.
        original_action = env.human_action
        env.human_action = lambda: np.zeros(354)
        try:
            _, reward, _, _, _ = env.step(np.array([0.5, -0.5]))
        finally:
            env.human_action = original_action
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(len(controller.requests), 10)
        np.testing.assert_allclose(controller.requests[0][0], [30.0, -40.0])
        self.assertEqual(controller.requests[0][1], 0.001)
        self.assertAlmostEqual(env.data.time, 0.01)

    def test_linkage_and_reference_derived_fields(self):
        env = self.env
        rows = (env.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(
            env.data.efc_id,
            np.flatnonzero(env.model.eq_type == mujoco.mjtEq.mjEQ_CONNECT),
        )
        self.assertLess(float(np.max(np.abs(env.data.efc_pos[rows]))), 1e-5)
        ref = read_reference(env.cfg)
        self.assertTrue(np.isfinite(ref["qvel"]).all())
        for name in ("knee", "ankle"):
            j = env.model.joint(name).id
            q = ref["qpos"][:, env.model.jnt_qposadr[j]]
            self.assertGreaterEqual(q.min(), env.model.jnt_range[j, 0] - 1e-7)
            self.assertLessEqual(q.max(), env.model.jnt_range[j, 1] + 1e-7)
        np.testing.assert_allclose(
            ref["foot_vel"], np.gradient(ref["foot_pos"], 0.01, axis=0)
        )


if __name__ == "__main__":
    unittest.main()
