"""CPU parity tests plus real CUDA integration (skipped if CUDA/assets absent)."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from myowarp.musclemimic.math_torch import axis_angle_from_quat, quat_from_matrix
from myowarp.musclemimic.running_stats import (
    numpy_running_mean_std_update,
    torch_running_mean_std_update_per_env,
)
from myowarp.tasks.musculoskeletal_prosthesis.config import ProsthesisCfg
from myowarp.tasks.musculoskeletal_prosthesis.controller import (
    MissingLowLevelController,
)
from myowarp.tasks.musculoskeletal_prosthesis.gpu_controller import (
    GPUMotorCommand,
    write_gpu_motor_command,
)
from myowarp.tasks.musculoskeletal_prosthesis.gpu_sim import require_cuda


class TensorParityTests(unittest.TestCase):
    def test_legacy_eval_has_zero_placeholder_and_preserves_sensors(self):
        from tensordict import TensorDict

        from myowarp.tasks.musculoskeletal_prosthesis.eval_adapter import (
            pad_unavailable_left_contact,
        )

        values = torch.arange(2 * 145).reshape(2, 145).float()
        obs = TensorDict(
            {"actor": values, "critic": torch.zeros((2, 737))}, batch_size=[2]
        )
        padded = pad_unavailable_left_contact(obs, 5)["actor"].reshape(2, 5, 30)
        self.assertTrue(bool((padded[:, :, 25] == 0).all()))
        restored = torch.cat((padded[:, :, :25], padded[:, :, 26:]), -1).flatten(1)
        torch.testing.assert_close(restored, values)
        self.assertEqual(obs["actor"].shape, (2, 145))

    def test_eval_can_swap_only_the_muscle_checkpoint(self):
        from dataclasses import asdict

        from myowarp.tasks.musculoskeletal_prosthesis.runner import (
            validate_checkpoint_config,
        )

        current = asdict(ProsthesisCfg())
        saved = dict(current, checkpoint_path="old_muscle_checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint_path"):
            validate_checkpoint_config(saved, current)
        changes = validate_checkpoint_config(saved, current, allow_muscle_override=True)
        self.assertEqual(set(changes), {"checkpoint_path"})
        saved["physics_dt"] = 0.002
        with self.assertRaisesRegex(ValueError, "physics_dt"):
            validate_checkpoint_config(saved, current, allow_muscle_override=True)

    def test_rotations_at_identity_and_near_pi(self):
        vectors = np.r_[
            np.zeros((1, 3)),
            np.eye(3) * np.pi,
            Rotation.random(100, random_state=8).as_rotvec(),
        ]
        r = Rotation.from_rotvec(vectors).as_matrix()
        got = axis_angle_from_quat(
            quat_from_matrix(torch.tensor(r, dtype=torch.float64))
        ).numpy()
        np.testing.assert_allclose(Rotation.from_rotvec(got).as_matrix(), r, atol=1e-12)

    def test_normalizers_are_independent_and_match_numpy(self):
        obs = torch.tensor([[1.0, 2.0, 3.0], [-5.0, 6.0, 8.0]])
        mean, var, count = (
            torch.zeros_like(obs),
            torch.ones_like(obs),
            torch.full((2,), 10.0),
        )
        for _ in range(3):
            expected = [
                numpy_running_mean_std_update(
                    obs=obs[i].numpy(),
                    mean=mean[i].numpy(),
                    var=var[i].numpy(),
                    count=count[i].numpy(),
                )
                for i in range(2)
            ]
            out, mean, var, count = torch_running_mean_std_update_per_env(
                obs, mean, var, count
            )
            for i in range(2):
                np.testing.assert_allclose(out[i].numpy(), expected[i][0], atol=1e-6)
                np.testing.assert_allclose(mean[i].numpy(), expected[i][1], atol=1e-6)
                np.testing.assert_allclose(var[i].numpy(), expected[i][2], atol=1e-6)

    def test_cpu_runtime_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            require_cuda("cpu")

    def test_four_motor_channels_and_clipping(self):
        d = SimpleNamespace(ctrl=torch.zeros((2, 8)))
        cmd = GPUMotorCommand(
            torch.tensor([[2.0, -3.0], [4.0, -5.0]]),
            torch.tensor([[6.0, -7.0], [8.0, -9.0]]),
        )
        ids = torch.tensor([4, 5, 6, 7])
        limits = torch.tensor([[-3.0, 3.0]] * 4)
        write_gpu_motor_command(d, cmd, ids, limits, 2)
        torch.testing.assert_close(d.ctrl[:, :4], torch.zeros((2, 4)))
        torch.testing.assert_close(
            d.ctrl[:, 4:],
            torch.tensor([[2.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0]]),
        )
        with self.assertRaises(ValueError):
            write_gpu_motor_command(
                d, GPUMotorCommand(torch.zeros(2), torch.zeros(2)), ids, limits, 2
            )


@unittest.skipUnless(
    torch.cuda.is_available() and Path(ProsthesisCfg().reference_path).is_file(),
    "CUDA and prepared MuscleMimic assets required",
)
class CUDAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv

        cls.env = GPUProsthesisVecEnv(ProsthesisCfg(), num_envs=2)

    def setUp(self):
        self.env.controller = None
        self.env.reset()

    def test_real_shadow_obs_inference_and_reward_parity(self):
        from myowarp.musclemimic.policy import (
            FullbodyObsAdapter,
            fullbody_obs_adapter_params_from_metadata,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.cpu_oracle import (
            CPUReferenceOracle as ProsthesisEnv,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.state import (
            sensor_frame,
            tracking_reward,
        )

        env = self.env
        cpu = ProsthesisEnv(env.cfg)
        frame = int(env.frame[0].item())
        cpu.reset(frame=frame)
        # Compare at exactly the same physical state, with native derived values.
        cpu.data.qpos[:] = env.data.qpos[0].cpu().numpy()
        cpu.data.qvel[:] = env.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(cpu.model, cpu.data)
        cpu.pre_substep = {
            k: getattr(cpu.data, k).copy() for k in vars(env.pre_substep)
        }
        original = env.data.qpos.clone()
        gpu_obs = env.human.observe(env.data, env.pre_substep, env.frame)
        shadow = cpu.shadow.observe(cpu.data, frame, cpu.pre_substep)
        from myowarp.tasks.musculoskeletal_prosthesis.artifacts import frozen_artifacts

        artifacts, metadata = frozen_artifacts(env.cfg.checkpoint_path)
        adapter = FullbodyObsAdapter(
            cpu.human_model,
            cpu.clip,
            goal_params=fullbody_obs_adapter_params_from_metadata(metadata),
        )
        expected = adapter.build(shadow, frame)
        np.testing.assert_allclose(
            gpu_obs[0].cpu().numpy(), expected, atol=0.03, rtol=0.003
        )
        torch.testing.assert_close(env.data.qpos, original)
        load, unsafe = env.state.contact_loads(env.sim)
        sensors = env.state.sensors(env.data, env.previous_action, load)
        np.testing.assert_allclose(
            sensors[0].cpu().numpy(),
            sensor_frame(cpu.model, cpu.data, np.zeros(2), cpu.cfg),
            atol=0.001,
            rtol=0.001,
        )
        reward, terms = env.state.reward(
            env.data, env.frame, env.previous_action, env.previous_action, load, unsafe
        )
        expected_reward, expected_terms = tracking_reward(
            cpu.model, cpu.data, cpu.reference, frame, np.zeros(2), np.zeros(2), cpu.cfg
        )
        for key, value in expected_terms.items():
            self.assertAlmostEqual(float(terms[key][0]), value, delta=0.01, msg=key)
        self.assertAlmostEqual(float(reward[0]), expected_reward, delta=0.001)
        actions = env.human_action()
        from myowarp.musclemimic.policy import _actor_forward

        if metadata.get("normalization_mode") == "fixed":
            norm = (expected - artifacts.obs_mean) / metadata["obs_std"]
            torch.testing.assert_close(
                env.human.actor.obs_std.cpu(), metadata["actor_state_dict"]["obs_std"]
            )
            torch.testing.assert_close(env.human.mean[0], env.human.initial_mean)
            self.assertEqual(float(env.human.count[0]), env.human.initial_count)
        else:
            norm = numpy_running_mean_std_update(
                obs=expected,
                mean=artifacts.obs_mean,
                var=artifacts.obs_var,
                count=artifacts.obs_count,
            )[0]
        np.testing.assert_allclose(
            actions[0].cpu().numpy(),
            np.clip(_actor_forward(artifacts.params, norm), -1, 1),
            atol=0.01,
            rtol=0.01,
        )
        self.assertFalse(any(p.requires_grad for p in env.human.actor.parameters()))
        self.assertEqual(env.get_observations()["actor"].shape, (2, 115))
        self.assertEqual(env.get_observations()["critic"].shape, (2, 737))

    def test_virtual_shadow_uses_prosthesis_state_and_copies_activations(self):
        from myowarp.tasks.musculoskeletal_prosthesis.shadow import FooledShadow

        env = self.env
        human = env.human
        old_mode = human.mode
        try:
            human.mode = "virtual"
            env.data.qpos[:, human.prosthesis_q] = torch.tensor(
                [0.7, -0.12], device=env.device
            )
            env.data.qvel[:, human.prosthesis_v] = torch.tensor(
                [0.4, -0.3], device=env.device
            )
            for key in vars(env.pre_substep):
                getattr(env.pre_substep, key).copy_(getattr(env.data, key))
            physical = env.data.qpos.clone()
            human.observe(env.data, env.pre_substep, env.frame)
            torch.testing.assert_close(
                human.data.qpos[:, human.view_q[:2]],
                env.data.qpos[:, human.prosthesis_q],
            )
            torch.testing.assert_close(
                human.data.qvel[:, human.view_v[:2]],
                env.data.qvel[:, human.prosthesis_v],
            )
            self.assertTrue(bool((human.data.qpos[:, human.view_q[2:]] == 0).all()))
            torch.testing.assert_close(
                human.data.act, env.data.act[:, human.actuator_ids]
            )
            torch.testing.assert_close(physical, env.data.qpos)
            import mujoco

            from myowarp.musclemimic.model import build_model
            from myowarp.tasks.musculoskeletal_prosthesis.reference import (
                load_human_clip,
            )

            model, _ = build_model()
            shadow = FooledShadow(
                model,
                env.model,
                load_human_clip(env.cfg.motion_path, model),
                mode="virtual",
            )
            data = mujoco.MjData(env.model)
            for key in vars(env.pre_substep):
                getattr(data, key)[:] = getattr(env.data, key)[0].cpu().numpy()
            pre = {key: getattr(data, key).copy() for key in vars(env.pre_substep)}
            expected = shadow.observe(data, int(env.frame[0]), pre)
            np.testing.assert_allclose(
                human.data.site_xpos[0].cpu().numpy(), expected.site_xpos, atol=2e-5
            )
        finally:
            human.mode = old_mode

    def test_left_contact_is_excluded_from_actor(self):
        env = self.env
        load = torch.zeros((2, 3), device=env.device)
        first = env.state.sensors(env.data, env.previous_action, load)
        load[:, 0] = 1000
        second = env.state.sensors(env.data, env.previous_action, load)
        torch.testing.assert_close(first, second)
        load[:, 1:] = 1000
        third = env.state.sensors(env.data, env.previous_action, load)
        torch.testing.assert_close(
            third[:, -4:-2], torch.ones((2, 2), device=env.device)
        )
        self.assertEqual(first.shape, (2, 29))

    def test_step_is_blocked_and_partial_reset_preserves_other_world(self):
        env = self.env
        before = env.data.qpos.clone()
        with self.assertRaises(MissingLowLevelController):
            env.step(torch.zeros((2, 2), device=env.device))
        torch.testing.assert_close(before, env.data.qpos)
        env.history[1] = 123
        env.human.mean[1] = 456
        env.reset(torch.tensor([0], device=env.device))
        torch.testing.assert_close(before[1], env.data.qpos[1])
        self.assertTrue(bool((env.history[1] == 123).all()))
        self.assertTrue(bool((env.human.mean[1] == 456).all()))

    def test_gpu_actor_excludes_sea_and_rewards_read_real_body(self):
        env = self.env
        load, unsafe = env.state.contact_loads(env.sim)
        before = env.state.sensors(env.data, env.previous_action, load).clone()
        motor = env.model.joint("knee_motor").qposadr[0]
        saved = env.data.qpos[:, motor].clone()
        env.data.qpos[:, motor] += 0.1
        # The actor reads joint encoders, never the motor/SEA difference.
        after = env.state.sensors(env.data, env.previous_action, load)
        torch.testing.assert_close(before, after)
        env.data.qpos[:, motor] = saved
        _, terms = env.state.reward(
            env.data, env.frame, env.previous_action, env.previous_action, load, unsafe
        )
        env.data.qpos[:, 0] += 0.5
        env.sim.forward()
        changed_load, changed_unsafe = env.state.contact_loads(env.sim)
        _, changed = env.state.reward(
            env.data,
            env.frame,
            env.previous_action,
            env.previous_action,
            changed_load,
            changed_unsafe,
        )
        self.assertTrue(bool((changed["root_pos"] < terms["root_pos"]).all()))

    def test_testonly_controller_timing_and_gpu_ppo_checkpoint(self):
        # Wiring test only: no functioning SEA controller and no learned walking.
        env = self.env

        class RecordingController:
            def __init__(self):
                self.calls = []

            def reset(self, ids):
                pass

            def update(self, state, desired, dt):
                self.calls.append((desired.clone(), dt))
                return GPUMotorCommand(
                    torch.zeros_like(desired), torch.zeros_like(desired)
                )

        controller = RecordingController()
        env.controller = controller
        env.step(torch.tensor([[0.5, -0.5], [0.2, -0.2]], device=env.device))
        self.assertEqual(len(controller.calls), 10)
        torch.testing.assert_close(
            controller.calls[0][0][0], torch.tensor([30.0, -40.0], device=env.device)
        )
        self.assertTrue(all(dt == 0.001 for _, dt in controller.calls))
        from myowarp.tasks.musculoskeletal_prosthesis.config import make_agent_cfg
        from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner

        cfg = make_agent_cfg(save_interval=1)
        cfg["num_steps_per_env"] = 2
        cfg["algorithm"]["num_learning_epochs"] = 1
        cfg["algorithm"]["num_mini_batches"] = 1
        cfg["actor"]["hidden_dims"] = [32, 32]
        cfg["critic"]["hidden_dims"] = [32, 32]
        with tempfile.TemporaryDirectory() as tmp:
            runner = ProsthesisRunner(env, cfg, log_dir=tmp, device=env.device)
            runner.learn(1, init_at_random_ep_len=False)
            checkpoint = str(Path(tmp) / "resume.pt")
            runner.save(checkpoint)
            # Loading under no_grad accommodates RSL's inference-mode normalizer buffers.
            with torch.inference_mode():
                info = runner.load(checkpoint, map_location=env.device)
            self.assertIn("cuda_rng", info)
        env.controller = None


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    unittest.main()
