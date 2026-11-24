"""H1-2 + CMU Exp4: geometry adaptation and independent GPU PPO learners."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rsl_rl")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "check", "train", "eval"))
    p.add_argument("--num-envs", type=int)
    p.add_argument(
        "--iterations",
        type=int,
        default=2500,
        help="Total iteration limit including resumed updates",
    )
    p.add_argument(
        "--checkpoint", type=Path, help="Paired Exp4 checkpoint for resume/eval"
    )
    p.add_argument(
        "--human-checkpoint", type=Path, help="Intact H1 pretrained actor and critic"
    )
    p.add_argument("--motion-file", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--episode-seconds", type=float)
    p.add_argument("--critic-warmup", type=int)
    p.add_argument("--save-interval", type=int)
    p.add_argument("--steps", type=int, default=12000, help="100Hz evaluation steps")
    p.add_argument(
        "--start-frame", type=int, default=0, help="Start in original 50Hz H1 reference"
    )
    p.add_argument("--viewer", action="store_true")
    p.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        help="Evaluation defaults to native CPU; training stays GPU",
    )
    p.add_argument("--viewer-fps", type=float, default=30, help="Viewer refresh rate")
    p.add_argument(
        "--slow-eval",
        action="store_true",
        help="Use original uncaptured evaluation for comparison",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    from myowarp.tasks.h1_2_cmu_exp4.config import Exp4Cfg

    cfg = Exp4Cfg()
    import torch

    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        from myowarp.tasks.h1_2_cmu_exp4.runner import PIPELINE

        if saved["infos"]["pipeline"] != PIPELINE:
            p.error("Expected paired H1+CMU Exp4 checkpoint")
        cfg = Exp4Cfg(**saved["infos"]["config"])
    for key, value in [
        ("human_checkpoint", args.human_checkpoint),
        ("motion_path", args.motion_file),
        ("reference_path", args.reference),
    ]:
        if value is not None:
            setattr(cfg, key, str(value.resolve()))
    cfg.num_envs = args.num_envs or (1024 if args.mode == "train" else 1)
    backend = args.backend or ("cpu" if args.mode == "eval" else "gpu")
    if backend == "cpu":
        if args.mode != "eval":
            p.error("CPU backend is evaluation only")
        cfg.device = "cpu"
        torch.set_num_threads(1)
    elif not cfg.device.startswith("cuda"):
        cfg.device = "cuda:0"
    if args.episode_seconds is not None:
        cfg.episode_seconds = args.episode_seconds
    elif args.mode == "eval":
        cfg.episode_seconds = min(120.0, args.steps * 0.01)
    if args.critic_warmup is not None:
        cfg.critic_warmup = args.critic_warmup
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    cfg.validate(evaluation=args.mode == "eval")
    if args.steps < 1 or args.iterations < 1:
        p.error("Positive counts required")
    if not 0 < args.viewer_fps <= 100:
        p.error("--viewer-fps must be in (0, 100]")
    if args.viewer and (args.mode != "eval" or cfg.num_envs != 1):
        p.error("Viewer requires eval with one environment")
    if args.mode == "eval" and not args.checkpoint:
        p.error("eval requires --checkpoint")
    if args.mode == "prepare":
        from myowarp.tasks.h1_2_cmu_exp4.reference import prepare_reference

        prepare_reference(cfg)
        return
    if backend == "gpu" and not torch.cuda.is_available():
        p.error("CUDA is required")
    torch.manual_seed(cfg.seed)
    from myowarp.tasks.h1_2_cmu_exp4.env import Exp4Env
    from myowarp.tasks.h1_2_cmu_exp4.runner import Exp4Runner

    output = None
    if args.mode == "train":
        output = args.output or ROOT / "logs/h1_2_cmu_exp4" / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        if output.exists() and any(output.glob("model_*.pt")) and not args.checkpoint:
            p.error("Choose a fresh output folder or resume explicitly")
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    env = Exp4Env(cfg, evaluation=args.mode == "eval")
    try:
        if args.mode == "check":
            obs = env.get_observations()
            for _ in range(2):
                obs, reward, done, extras = env.step(
                    torch.zeros(cfg.num_envs, 27, device=env.device),
                    torch.zeros(cfg.num_envs, 2, device=env.device),
                )
            print(
                json.dumps(
                    {
                        "observations": {k: list(v.shape) for k, v in obs.items()},
                        "finite": all(
                            bool(torch.isfinite(v).all()) for v in obs.values()
                        ),
                        "rewards": {k: float(v.mean()) for k, v in reward.items()},
                        "h1_hz": 50,
                        "prosthesis_hz": 100,
                        "llc_hz": 1000,
                    },
                    indent=2,
                )
            )
            return
        runner = Exp4Runner(env, output)
        if args.checkpoint:
            runner.load(args.checkpoint, evaluation=args.mode == "eval")
        if args.mode == "train":
            print(f"Paired PPO logs/checkpoints: {output}", flush=True)
            runner.learn(args.iterations)
            return
        if not args.slow_eval:
            env.enable_fast_evaluation()
        env.reset(args.start_frame)
        import mujoco

        window = None
        display = mujoco.MjData(env.model)
        if args.viewer:
            from mujoco import viewer

            display.qpos[:] = env.data.qpos[0].cpu().numpy()
            mujoco.mj_forward(env.model, display)
            window = viewer.launch_passive(env.model, display)
            window.cam.distance = 3.5
            window.cam.azimuth = 135
            window.cam.elevation = -15
        obs = env.get_observations()
        active = torch.ones(cfg.num_envs, dtype=torch.bool, device=env.device)
        counts = torch.zeros(cfg.num_envs, device=env.device)
        sums = {}
        final = {}
        trace = []
        timeouts = torch.zeros_like(active)
        rollout_start = time.monotonic()
        next_render = 0.0
        render_seconds = 0.0
        try:
            with torch.inference_mode():
                for step in range(args.steps):
                    actions = runner.inference_actions(obs)
                    obs, _rewards, done, extras = env.step(
                        actions["h1"], actions["prosthesis"]
                    )
                    counts += active
                    for key, value in env.metrics.items():
                        sums[key] = sums.get(
                            key, torch.zeros_like(value)
                        ) + torch.where(active, value, 0.0)
                        final[key] = torch.where(
                            active, value, final.get(key, torch.zeros_like(value))
                        )
                    trace.append({k: v.detach() for k, v in env.metrics.items()})
                    timeouts |= active & extras["time_outs"]
                    active &= ~done
                    if window:
                        if not window.is_running():
                            break
                        if step * 0.01 >= next_render:
                            render_start = time.monotonic()
                            display.qpos[:] = env.data.qpos[0].cpu().numpy()
                            # Only positions are needed for rendering.
                            mujoco.mj_kinematics(env.model, display)
                            mujoco.mj_camlight(env.model, display)
                            window.cam.lookat[:] = display.qpos[:3]
                            window.sync()
                            render_seconds += time.monotonic() - render_start
                            next_render += 1 / args.viewer_fps
                        # Pace against the whole rollout clock, allowing cheap
                        # steps to catch up after a more expensive render frame.
                        time.sleep(
                            max(
                                0.0,
                                (step + 1) * 0.01 - (time.monotonic() - rollout_start),
                            )
                        )
                    if not active.any():
                        break
        finally:
            if window:
                window.close()
        if backend == "gpu":
            torch.cuda.synchronize(env.device)
        wall_seconds = time.monotonic() - rollout_start
        # Transfer the whole trace after simulation, rather than synchronizing
        # the GPU four times on every policy step.
        trace_columns = {
            key: torch.stack([row[key] for row in trace]).cpu().tolist() for key in sums
        }
        trace = [
            {key: values[i] for key, values in trace_columns.items()}
            for i in range(len(trace))
        ]
        report = {
            "checkpoint": str(args.checkpoint),
            "num_envs": cfg.num_envs,
            "start_frame": args.start_frame,
            "mean_duration_seconds": float(counts.mean() * 0.01),
            "completed": int(timeouts.sum()),
            "rollout_wall_seconds": wall_seconds,
            "viewer_render_seconds": render_seconds,
            "simulation_to_wall_ratio": float(counts.mean() * 0.01) / wall_seconds,
            "fast_evaluation": not args.slow_eval,
            "backend": backend,
            "mean_errors": {
                k: float((v / counts.clamp_min(1)).mean()) for k, v in sums.items()
            },
            "final_errors": {k: float(v.mean()) for k, v in final.items()},
            "units": "positions: metres; prosthesis joint error: radians",
        }
        destination = args.output or args.checkpoint.parent / "eval_exp4.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        destination.with_suffix(".trace.json").write_text(json.dumps(trace) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
