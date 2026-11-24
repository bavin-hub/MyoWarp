"""Recorded original-controller parity and CUDA closed-loop integration."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from myowarp.tasks.musculoskeletal_prosthesis.config import (
    ProsthesisCfg,
    make_agent_cfg,
)
from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
    SEATorqueController,
    create_controller,
)


class SEAParityTests(unittest.TestCase):
    def replay(self, device):
        path = Path(__file__).parent / "fixtures/sea_controller_oracle.npz"
        with np.load(path) as pack:
            fixture = {k: pack[k] for k in pack.files}
        model = mujoco.MjModel.from_xml_path(ProsthesisCfg().model_path)
        controller = SEATorqueController(
            sim=SimpleNamespace(cpu_model=model),
            num_envs=3,
            device=device,
            startup_ready=False,
        )
        controller.prev_enabled[2] = True
        controller.ticks[2] = 1000
        data = SimpleNamespace(
            qpos=torch.zeros((3, model.nq), dtype=torch.float64, device=device),
            qvel=torch.zeros((3, model.nv), dtype=torch.float64, device=device),
            ctrl=torch.zeros((3, model.nu), dtype=torch.float64, device=device),
        )
        tensors = {k: torch.as_tensor(v, device=device) for k, v in fixture.items()}
        outputs = {
            k: torch.empty_like(tensors[k])
            for k in ("raw_ff", "velocity", "conditioned", "total_ff")
        }
        for t in range(len(fixture["q"])):
            data.qpos[:, controller.qids] = tensors["q"][t]
            data.qvel[:, controller.vids] = tensors["v"][t]
            command = controller.compute(
                data,
                tensors["desired"][t],
                0.001,
                enabled=tensors["enabled"][t],
                actuate=tensors["actuate"][t],
                standing=tensors["standing"][t],
            )
            outputs["raw_ff"][t] = controller.raw_feedforward
            outputs["velocity"][t] = command.target_velocity
            outputs["conditioned"][t] = controller.commanded_torque
            outputs["total_ff"][t] = command.feedforward_torque
        for k, actual in outputs.items():
            np.testing.assert_allclose(
                actual.cpu().numpy(), fixture[k], atol=2e-8, rtol=2e-9, err_msg=k
            )
        # Partial reset clears only requested environment state.
        before = controller.z1[1:].clone()
        integral = controller.drive_integral[1:].clone()
        controller.reset(torch.tensor([0], device=device))
        torch.testing.assert_close(before, controller.z1[1:])
        torch.testing.assert_close(integral, controller.drive_integral[1:])
        self.assertTrue(bool((controller.z1[0] == 0).all()))
        self.assertTrue(bool((controller.drive_integral[0] == 0).all()))

    def test_cpu_against_recorded_original_controller(self):
        self.replay("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_against_recorded_original_controller(self):
        self.replay("cuda:0")


@unittest.skipUnless(
    torch.cuda.is_available() and Path(ProsthesisCfg().reference_path).is_file(),
    "CUDA and prepared motion required",
)
class SEAIntegrationTests(unittest.TestCase):
    def test_real_controller_rollout_ppo_and_resume(self):
        from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv
        from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner

        env = GPUProsthesisVecEnv(ProsthesisCfg(), create_controller, num_envs=4)
        self.assertIsInstance(env.controller, SEATorqueController)
        for _ in range(50):
            _, rewards, _, _ = env.step(torch.zeros((4, 2), device=env.device))
            self.assertTrue(bool(torch.isfinite(rewards).all()))
        self.assertTrue(bool((env.controller.ticks > 1000).all()))
        self.assertTrue(bool(torch.isfinite(env.data.qpos).all()))
        cfg = make_agent_cfg(save_interval=1)
        cfg["num_steps_per_env"] = 4
        cfg["algorithm"]["num_learning_epochs"] = 1
        cfg["algorithm"]["num_mini_batches"] = 1
        cfg["actor"]["hidden_dims"] = [32, 32]
        cfg["critic"]["hidden_dims"] = [32, 32]
        with tempfile.TemporaryDirectory() as tmp:
            runner = ProsthesisRunner(env, cfg, log_dir=tmp, device=env.device)
            frozen = [p.detach().clone() for p in env.human.actor.parameters()]
            runner.learn(1, init_at_random_ep_len=False)
            runner.save(str(Path(tmp) / "resume.pt"))
            with torch.inference_mode():
                runner.load(str(Path(tmp) / "resume.pt"), map_location=env.device)
            for before, after in zip(frozen, env.human.actor.parameters(), strict=True):
                torch.testing.assert_close(before, after)
            self.assertTrue(bool((env.controller.drive_integral == 0).all()))
        env.close()


if __name__ == "__main__":
    unittest.main()
