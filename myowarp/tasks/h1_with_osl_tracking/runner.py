"""RSL-RL runner and motion-bundled ONNX export."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from torch import nn


class _OnnxMotionModel(nn.Module):
    def __init__(self, actor, motion) -> None:
        super().__init__()
        self.policy = actor.as_onnx(verbose=False)
        self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
        self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
        self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
        self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
        self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
        self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
        self.time_step_total = motion.num_frames

    def forward(self, observations, time_step):
        frame = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.policy(observations),
            self.joint_pos[frame],
            self.joint_vel[frame],
            self.body_pos_w[frame],
            self.body_quat_w[frame],
            self.body_lin_vel_w[frame],
            self.body_ang_vel_w[frame],
        )


class _OnnxSeparateObsMotionModel(_OnnxMotionModel):
    def forward(self, primary_observations, secondary_observations, time_step):
        frame = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.policy(primary_observations, secondary_observations),
            self.joint_pos[frame],
            self.joint_vel[frame],
            self.body_pos_w[frame],
            self.body_quat_w[frame],
            self.body_lin_vel_w[frame],
            self.body_ang_vel_w[frame],
        )


class H1TrackingRunner(OnPolicyRunner):
    """Persist adaptive sampling state and bundle motion data in ONNX exports."""

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_infos = dict(infos or {})
        checkpoint_infos["env_state"] = self.env.training_state()
        super().save(path, checkpoint_infos)
        run_directory = Path(path).parent
        try:
            self.export_policy_to_onnx(
                str(run_directory), filename=f"{run_directory.name}.onnx"
            )
        except Exception as error:
            print(f"[WARN] Motion ONNX export failed; training continues: {error}")

    def load(
        self,
        path: str | Path,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(str(path), load_cfg, strict, map_location) or {}
        self.env.load_training_state(infos.get("env_state"))
        return infos

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        actor = self.alg.get_policy()
        policy = actor.as_onnx(verbose=False)
        if len(policy.input_names) == 1:
            model = _OnnxMotionModel(actor, self.env.motion)
        elif len(policy.input_names) == 2:
            model = _OnnxSeparateObsMotionModel(actor, self.env.motion)
        else:
            raise ValueError(
                f"Motion export supports one or two policy inputs, got {policy.input_names}"
            )
        model.to("cpu")
        model.eval()
        time_step = torch.zeros(1, 1)
        dummy_inputs = (*model.policy.get_dummy_inputs(), time_step)
        torch.onnx.export(
            model,
            dummy_inputs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=[*model.policy.input_names, "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
            dynamo=False,
        )
