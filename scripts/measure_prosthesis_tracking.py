"""Matched deterministic tracking evaluation of removed-leg experiments."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from myowarp.musclemimic.math_torch import rotation_error
from myowarp.tasks.musculoskeletal_prosthesis.config import (
    ProsthesisCfg,
    make_agent_cfg,
)
from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv
from myowarp.tasks.musculoskeletal_prosthesis.gpu_state import body_velocity
from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner
from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import create_controller

METRICS = (
    "root_xy_m",
    "root_3d_m",
    "body_root_relative_m",
    "prosthesis_sole_global_m",
    "prosthesis_sole_root_relative_m",
    "root_velocity_m_s",
    "knee_abs_deg",
    "ankle_abs_deg",
    "root_orientation_deg",
    "sole_orientation_deg",
)


@torch.inference_mode()
def evaluate(
    folder, num_envs, steps, expected_starts, output, selection="model_1199.pt"
):
    split = folder.name.startswith(("04_", "05_", "06_", "07_", "08_"))
    filename = (
        ("model_best_prosthesis.pt" if split else "model_best.pt")
        if selection == "best"
        else selection
    )
    checkpoint = folder / filename
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_iteration = int(saved["iter"])
    cfg = ProsthesisCfg(**saved["infos"]["prosthesis_config"])
    cfg.validate()
    if split:
        from myowarp.tasks.musculoskeletal_prosthesis.split_config import (
            SplitTrainingCfg,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_env import (
            SplitProsthesisEnv,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_runner import SplitRunner

        training = SplitTrainingCfg(**saved["infos"]["training_config"])
        env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=num_envs)
        runner = SplitRunner(env, training)
        runner.load(checkpoint)
        policy = runner.inference_policy()
    else:
        env = GPUProsthesisVecEnv(cfg, create_controller, num_envs=num_envs)
        train = make_agent_cfg()
        if cfg.train_muscle:
            train["algorithm"]["class_name"] = (
                "myowarp.tasks.musculoskeletal_prosthesis.coopt:CooptPPO"
            )
        runner = ProsthesisRunner(env, train, log_dir=None, device=env.device)
        runner.load(str(checkpoint), map_location=env.device)
        policy = runner.get_inference_policy()
    del saved
    env.eligible = env.eligible[env.eligible + steps < len(env.reference["qpos"])]
    if env.eligible.numel() == 0:
        raise ValueError("No reset frames with the full evaluation horizon remaining")
    env.generator.manual_seed(12345)
    env.reset()
    starts = env.frame.cpu().numpy().copy()
    if expected_starts is not None:
        np.testing.assert_array_equal(starts, expected_starts)
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(active)
    records = []
    original_reward = env.state.reward

    def measure(d, frame, *args):
        result = original_reward(d, frame, *args)
        state, ref = env.state, env.reference
        pos = d.xpos[:, state.body_ids]
        target = ref["body_pos"][frame]
        root_error = pos[:, 0] - target[:, 0]
        sole, sole_rot = state.sole(d)
        relative = (pos - pos[:, :1]) - (target - target[:, :1])
        sole_relative = (sole - pos[:, 0]) - (ref["foot_pos"][frame] - target[:, 0])
        vel = body_velocity(d, state.body_ids, state.body_roots)[:, 0, :3]
        qerr = (d.qpos[:, state.qids] - ref["qpos"][frame][:, state.qids]).abs()
        norm = lambda x: torch.linalg.vector_norm(x, dim=-1)
        values = torch.stack(
            (
                norm(root_error[:, :2]),
                norm(root_error),
                norm(relative).mean(-1),
                norm(sole - ref["foot_pos"][frame]),
                norm(sole_relative),
                norm(vel - ref["body_vel"][frame, 0]),
                torch.rad2deg(qerr[:, 0]),
                torch.rad2deg(qerr[:, 1]),
                torch.rad2deg(
                    rotation_error(
                        d.xmat[:, state.body_ids[0]].reshape(-1, 3, 3),
                        ref["body_rot"][frame, 0],
                    )
                ),
                torch.rad2deg(rotation_error(sole_rot, ref["foot_rot"][frame])),
            ),
            -1,
        )
        # Called at the advanced physical state, before any automatic resets.
        records.append(values.masked_fill(~active[:, None], float("nan")).clone())
        return result

    env.state.reward = measure
    for step in range(steps):
        _, _, done, extra = env.step(policy(env.get_observations()))
        failed |= active & done.bool() & ~extra["time_outs"]
        active &= ~done.bool()
        if (step + 1) % 100 == 0:
            print(
                f"{folder.name}: {step + 1}/{steps}, still active={active.sum().item()}",
                flush=True,
            )
        if not bool(active.any()):
            break
    values = torch.stack(records).cpu().numpy()
    valid = np.isfinite(values[:, :, 0])
    counts = valid.sum(0)
    completed = (counts == steps) & ~failed.cpu().numpy()
    final = np.stack([values[counts[i] - 1, i] for i in range(num_envs)])
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": checkpoint_iteration,
        "start_frames": starts.tolist(),
        "num_episodes": num_envs,
        "horizon_seconds": steps * cfg.policy_dt,
        "completed_full_horizon": int(completed.sum()),
        "mean_episode_seconds": float(counts.mean() * cfg.policy_dt),
        "metrics_sample_weighted_mean": dict(
            zip(METRICS, np.nanmean(values, axis=(0, 1)).tolist(), strict=True)
        ),
        "metrics_mean_per_episode": dict(
            zip(METRICS, np.nanmean(values, axis=0).mean(0).tolist(), strict=True)
        ),
        "metrics_mean_final": dict(zip(METRICS, final.mean(0).tolist(), strict=True)),
        "measurement": "Euclidean position/velocity distances and absolute joint angle errors; real physical state before automatic reset, including first terminal state. No samples after first termination. Deterministic actions, no training.",
    }
    np.savez_compressed(
        output / f"{folder.name}.npz",
        metrics=values,
        names=METRICS,
        start_frames=starts,
        valid_steps=counts,
        completed=completed,
    )
    (output / f"{folder.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    env.close()
    return summary, starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--checkpoint",
        default="model_1199.pt",
        help="best, or a checkpoint filename shared across all runs",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("num-envs and steps must be positive")
    output = args.output or args.run / (
        "tracking_evaluation_best"
        if args.checkpoint == "best"
        else "tracking_evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    summaries, starts = {}, None
    names = [
        name
        for name in (
            "01_default_frozen",
            "02_vyvas_frozen",
            "03_default_coopt",
            "04_independent",
            "05_shared",
            "06_reference",
            "06_muscle_reference",
            "07_hip_rpy",
            "08_human_space",
        )
        if (args.run / name).is_dir()
    ]
    if not names:
        parser.error("No experiment folders found in the run directory")
    for name in names:
        summaries[name], starts = evaluate(
            args.run / name, args.num_envs, args.steps, starts, output, args.checkpoint
        )
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    # Compare identical time samples up to the earliest failure for each start.
    arrays = []
    for name in names:
        with np.load(output / f"{name}.npz") as data:
            values = data["metrics"]
            padded = np.full((args.steps, args.num_envs, len(METRICS)), np.nan)
            padded[: len(values)] = values
            arrays.append(padded)
    common = np.logical_and.reduce([np.isfinite(x[:, :, 0]) for x in arrays])
    paired = {
        "description": "Identical time samples before the earliest termination among policies for each start",
        "mean_compared_seconds_per_episode": float(common.sum() / args.num_envs * 0.01),
        "metrics": {
            name: dict(zip(METRICS, data[common].mean(0).tolist(), strict=True))
            for name, data in zip(names, arrays, strict=True)
        },
    }
    (output / "paired_window_means.json").write_text(
        json.dumps(paired, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
