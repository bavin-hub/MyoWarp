"""Check advancing periodic references, derived kinematics and invalid inputs."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from myowarp.musclemimic.reference import KINEMATIC_KEYS, best_cycle, loop_clip


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.src = Path(self.directory.name) / "source.npz"
        self.dst = Path(self.directory.name) / "loop.npz"
        self.model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody><body name="pelvis" pos="0 0 1"><freejoint name="root"/>
            <geom size=".1"/><body><joint name="knee"/><geom size=".05"/>
            <site name="foot" pos="0 0 -.4"/></body></body></worldbody></mujoco>''')
        m = self.model
        d = mujoco.MjData(m)
        n = 401
        qpos = np.tile(m.qpos0, (n, 1))
        t = np.arange(n) / 100
        qpos[:, 0] = t
        qpos[:, 7] = .3 * np.sin(2 * np.pi * t)
        fields = {key: [] for key in KINEMATIC_KEYS}
        for q in qpos:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            for key in fields:
                fields[key].append(np.array(getattr(d, key)))
        np.savez(self.src, **{key: np.array(value) for key, value in fields.items()},
                 qpos=qpos, qvel=np.zeros((n, m.nv)), frequency=100.,
                 joint_names=np.array([m.joint(j).name for j in range(m.njnt)]),
                 site_names=np.array(["foot"]), split_points=np.array([0, n]))

    def test_advances_each_cycle_and_recomputes_kinematics(self):
        loop_clip(self.src, self.dst, self.model, 3., start=100, length=100)
        with np.load(self.dst) as out:
            self.assertEqual(len(out["qpos"]), 301)
            np.testing.assert_allclose(out["qpos"][100, :3] - out["qpos"][0, :3], [1, 0, 0])
            np.testing.assert_allclose(out["qpos"][::100, 7], 0, atol=1e-6)
            np.testing.assert_allclose(out["qvel"][99:102, 0], 1, atol=1e-6)
            np.testing.assert_array_equal(out["split_points"], [0, 301])
            d = mujoco.MjData(self.model)
            for frame in [1, 99, 100, 101, 299]:
                d.qpos[:] = out["qpos"][frame]
                d.qvel[:] = out["qvel"][frame]
                mujoco.mj_forward(self.model, d)
                for key in KINEMATIC_KEYS:
                    np.testing.assert_allclose(out[key][frame], getattr(d, key), atol=1e-6)

    def test_cycle_search_finds_one_period(self):
        with np.load(self.src) as source:
            start, length = best_cycle(self.model, source["qpos"], 100.)
        self.assertEqual(length, 100)
        self.assertGreaterEqual(start, 0)

    def test_invalid_parameters_are_rejected(self):
        for options in [dict(seconds=-1), dict(seconds=1, start=1),
                        dict(seconds=1, start=-1, length=100),
                        dict(seconds=1, start=350, length=100)]:
            with self.assertRaises(ValueError):
                loop_clip(self.src, self.dst, self.model, **options)

    def test_short_running_cycle(self):
        with np.load(self.src) as source:
            qpos = source["qpos"].copy()
        qpos[:, 7] = .3 * np.sin(4 * np.pi * np.arange(len(qpos)) / 100)
        _start, length = best_cycle(self.model, qpos, 100., t_min=.4, t_max=.7)
        self.assertEqual(length, 50)
        with self.assertRaises(ValueError):
            best_cycle(self.model, qpos, 100., t_min=.7, t_max=.4)


if __name__ == "__main__":
    unittest.main()
