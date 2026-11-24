"""Independent intact H1-2 walking training/evaluation using the MJLab-parity task."""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "rsl_rl"):
    sys.path.insert(0, str(source))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "check", "train", "eval"))
    p.add_argument("--num-envs", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-iterations", type=int, default=30000)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--motion-file", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--episode-seconds", type=float, help="Override evaluation episode duration")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--disable-domain-randomization", action="store_true")
    p.add_argument("--disable-observation-noise", action="store_true")
    args = p.parse_args()
    from myowarp.tasks.h1_2_tracking import (
        H1TrackingCfg,
        H1TrackingEnv,
        H1TrackingRunner,
        make_rsl_rl_cfg,
    )

    if args.mode == "prepare":
        from myowarp.tasks.h1_2_tracking.prepare import prepare_motion

        prepare_motion(
            args.motion_file or ROOT / "assets/h1_amass/motions/walk1_subject1.csv",
            args.output,
        )
        return
    import mujoco
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        p.error("This training/replay pipeline requires CUDA")
    num_envs = args.num_envs or (1024 if args.mode == "train" else 1)
    if num_envs < 1 or args.steps < 1 or args.max_iterations < 1:
        p.error("Counts must be positive")
    if args.viewer and (args.mode != "eval" or num_envs != 1):
        p.error("--viewer requires eval with one env")
    if args.mode == "eval" and not args.checkpoint:
        p.error("eval requires --checkpoint")
    if args.episode_seconds is not None:
        import math

        if args.mode != "eval" or not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0:
            p.error("--episode-seconds requires eval and a positive finite duration")
    cfg = H1TrackingCfg(
        num_envs=num_envs,
        device=args.device,
        seed=args.seed if args.seed is not None else 42,
    )
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("infos", {}).get("pipeline") != "h1_2_tracking":
            p.error("Expected an intact H1-2 tracking checkpoint")
        from myowarp.tasks.h1_2_tracking.config import config_from_dict

        cfg = config_from_dict(saved["infos"]["env_config"])
        cfg.num_envs = num_envs
        cfg.device = args.device
        if args.seed is not None:
            cfg.seed = args.seed
        del saved
    if args.motion_file:
        cfg.motion_path = str(args.motion_file.resolve())
    torch.manual_seed(cfg.seed)
    cfg.enable_observation_noise = (
        cfg.enable_observation_noise and not args.disable_observation_noise
    )
    cfg.domain_randomization.enabled = (
        cfg.domain_randomization.enabled and not args.disable_domain_randomization
    )
    if args.mode in ("eval", "check"):
        cfg.enable_observation_noise = False
        cfg.domain_randomization.enabled = False
        cfg.domain_randomization.enable_pushes = False
        cfg.motion.sampling_mode = "start"
        for key in vars(cfg.motion):
            if key.startswith(("pose_", "velocity_")) or key == "joint_position_range":
                setattr(cfg.motion, key, (0.0, 0.0))
    if args.episode_seconds is not None:
        cfg.episode_length_s = args.episode_seconds
    env = H1TrackingEnv(cfg)
    agent = make_rsl_rl_cfg(
        max_iterations=args.max_iterations, save_interval=args.save_interval
    )
    agent["seed"] = cfg.seed
    try:
        if args.mode == "check":
            obs = env.get_observations()
            with torch.inference_mode():
                _, r, _, _ = env.step(torch.zeros((num_envs, 27), device=env.device))
            print(
                json.dumps(
                    {
                        "actor": list(obs["actor"].shape),
                        "critic": list(obs["critic"].shape),
                        "actions": env.num_actions,
                        "finite": bool(torch.isfinite(r).all()),
                        "frames": env.motion.num_frames,
                        "reference": cfg.motion_path,
                    },
                    indent=2,
                )
            )
            return
        output = None
        if args.mode == "train":
            output = (
                args.output
                or ROOT
                / "logs/rsl_rl/h1_2_tracking_standalone"
                / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            if (
                output.exists()
                and any(output.glob("model_*.pt"))
                and not args.checkpoint
            ):
                p.error(
                    "Output contains checkpoints; use a fresh path or --checkpoint to resume"
                )
            (output / "params").mkdir(parents=True, exist_ok=True)
            (output / "params/env.json").write_text(
                json.dumps(cfg.to_dict(), indent=2) + "\n"
            )
            (output / "params/agent.json").write_text(
                json.dumps(agent, indent=2) + "\n"
            )
        runner = H1TrackingRunner(
            env,
            copy.deepcopy(agent),
            log_dir=str(output) if output else None,
            device=args.device,
        )
        if args.checkpoint:
            runner.load(args.checkpoint, map_location=args.device)
        if args.mode == "train":
            print(
                f"Intact H1-2 tracking: {num_envs} GPU envs; output={output}",
                flush=True,
            )
            runner.learn(
                num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
            )
            return
        env.set_motion_frame(args.start_frame)
        policy = runner.get_inference_policy(device=env.device)
        window = None
        display = mujoco.MjData(env.model)
        if args.viewer:
            from mujoco import viewer

            window = viewer.launch_passive(env.model, display)
            window.cam.distance = 3
        active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        sums = {}
        counts = torch.zeros(num_envs, device=env.device)
        obs = env.get_observations()
        for step in range(args.steps):
            tick = time.monotonic()
            with torch.inference_mode():
                obs, _, done, _extras = env.step(policy(obs))
            counts += active
            for k, v in env.evaluation_metrics().items():
                sums[k] = sums.get(k, torch.zeros_like(v)) + torch.where(active, v, 0.0)
            active &= ~done.bool()
            if window:
                if not window.is_running():
                    break
                q, v = env.cpu_state()
                display.qpos[:] = q
                display.qvel[:] = v
                mujoco.mj_forward(env.model, display)
                window.cam.lookat[:] = display.xpos[env.anchor_body_id]
                window.sync()
                time.sleep(max(0.0, env.step_dt - (time.monotonic() - tick)))
            if not bool(active.any()):
                break
        if window:
            window.close()
        print(
            json.dumps(
                {
                    "mean_duration_seconds": float(counts.mean() * env.step_dt),
                    "metrics": {
                        k: float((v / counts.clamp_min(1)).mean())
                        for k, v in sums.items()
                    },
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
