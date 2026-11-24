"""Offline reference preparation and CUDA-only prosthesis PPO/check/evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "rsl_rl"):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "train", "eval"))
    parser.add_argument(
        "--anatomy",
        choices=("constrained", "removed"),
        default="constrained",
        help="Retain the constrained biological lower leg or remove it",
    )
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--human-checkpoint", type=Path)
    parser.add_argument(
        "--train-muscle",
        action="store_true",
        help="Joint PPO: fine-tune the muscle actor and train the prosthesis actor",
    )
    parser.add_argument("--muscle-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--controller",
        default="myowarp.tasks.musculoskeletal_prosthesis.sea_controller:create_controller",
        help="Batched GPU controller module:factory (default: local CMU SEA controller)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="CUDA device; CPU runtime is not supported"
    )
    parser.add_argument(
        "--right-leg-obs",
        choices=("fooled", "virtual"),
        help="Shadow leg observations; eval defaults to virtual, training/check to fooled",
    )
    parser.add_argument("--frames", type=int, help="Preparation prefix only")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/musculoskeletal_prosthesis"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Copy one GPU world to native viewer for rendering only",
    )
    args = parser.parse_args()
    os.environ.setdefault(
        "JAX_PLATFORMS", "cpu"
    )  # Orbax loading only; actor inference is Torch/CUDA.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from myowarp.tasks.musculoskeletal_prosthesis.config import (
        ProsthesisCfg,
        make_agent_cfg,
    )
    from myowarp.tasks.musculoskeletal_prosthesis.controller import (
        MissingLowLevelController,
        load_controller_factory,
    )

    if args.num_envs < 1 or args.save_interval < 1:
        parser.error("num-envs and save-interval must be positive")
    cfg = ProsthesisCfg(device=args.device, seed=args.seed)
    cfg.train_muscle = args.train_muscle
    cfg.muscle_learning_rate = args.muscle_learning_rate
    cfg.select_anatomy(args.anatomy)
    cfg.right_leg_obs = args.right_leg_obs or (
        "virtual" if args.mode == "eval" else "fooled"
    )
    for value, field in (
        (args.motion, "motion_path"),
        (args.reference, "reference_path"),
        (args.human_checkpoint, "checkpoint_path"),
    ):
        if value:
            setattr(cfg, field, str(value.resolve()))
    cfg.validate()
    if args.mode == "prepare":
        import mujoco
        import numpy as np

        from myowarp.musclemimic.model import build_model
        from myowarp.tasks.musculoskeletal_prosthesis.reference import prepare_reference

        ref = prepare_reference(
            cfg,
            build_model()[0],
            mujoco.MjModel.from_xml_path(cfg.model_path),
            max_frames=args.frames,
        )
        print(
            json.dumps(
                {
                    "path": cfg.reference_path,
                    "frames": len(ref["qpos"]),
                    "fit_error_mean_m": float(np.mean(ref["fit_error_m"])),
                    "fit_error_max_m": float(np.max(ref["fit_error_m"])),
                },
                indent=2,
            )
        )
        return
    try:
        factory = (
            load_controller_factory(args.controller)
            if args.mode in ("check", "train", "eval")
            else None
        )
    except MissingLowLevelController as error:
        parser.error(str(error))
    import torch

    from myowarp.tasks.musculoskeletal_prosthesis.gpu_sim import require_cuda

    try:
        cfg.device = require_cuda(cfg.device)
    except RuntimeError as error:
        parser.error(str(error))
    from myowarp.tasks.musculoskeletal_prosthesis.runner import (
        ProsthesisRunner,
        ProsthesisVecEnv,
        validate_checkpoint_config,
    )

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if args.checkpoint:
        saved = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=True
        )
        try:
            validate_checkpoint_config(
                saved["infos"]["prosthesis_config"],
                asdict(cfg),
                allow_muscle_override=args.mode == "eval",
            )
        except ValueError as error:
            parser.error(
                f"{error}. Select the checkpoint's matching --anatomy and reference; train a new policy for a different physical model."
            )
        del saved
    env = ProsthesisVecEnv(cfg, factory, args.num_envs)
    if args.mode == "check":
        with torch.no_grad():
            obs = env.get_observations()
            human = env.human_action()
            load, unsafe = env.state.contact_loads(env.sim)
            reward, _ = env.state.reward(
                env.data,
                env.frame,
                env.previous_action,
                env.previous_action,
                load,
                unsafe,
            )
        print(
            json.dumps(
                {
                    "device": env.device,
                    "num_envs": env.num_envs,
                    "actor_shape": list(obs["actor"].shape),
                    "critic_shape": list(obs["critic"].shape),
                    "frozen_actions": list(human.shape),
                    "muscle_checkpoint": cfg.checkpoint_path,
                    "anatomy": args.anatomy,
                    "model_path": cfg.model_path,
                    "right_leg_obs": cfg.right_leg_obs,
                    "muscle_normalization": "fixed"
                    if env.human.fixed_normalization
                    else "running",
                    "finite": bool(
                        torch.isfinite(human).all() & torch.isfinite(reward).all()
                    ),
                    "mean_reward": float(reward.mean()),
                    "physics": "MuJoCo-Warp 3.11, CUDA",
                    "low_level_status": "local CMU SEA controller + simulated drive integral, 1 kHz",
                    "sea_deflection_in_actor": False,
                },
                indent=2,
            )
        )
        return
    if args.mode == "eval" and not args.checkpoint:
        parser.error("Evaluation needs --checkpoint")
    if args.viewer and (args.mode != "eval" or args.num_envs != 1):
        parser.error("Viewer requires eval --num-envs 1")
    agent_cfg = make_agent_cfg(args.save_interval)
    if cfg.train_muscle:
        agent_cfg["algorithm"]["class_name"] = (
            "myowarp.tasks.musculoskeletal_prosthesis.coopt:CooptPPO"
        )
    agent_cfg.update(seed=cfg.seed, max_iterations=args.iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    if args.mode == "eval":
        from myowarp.tasks.musculoskeletal_prosthesis.eval_adapter import (
            adapt_legacy_checkpoint,
        )

        if not cfg.train_muscle:
            env = adapt_legacy_checkpoint(env, args.checkpoint)
    runner = ProsthesisRunner(
        env, agent_cfg, log_dir=str(args.output), device=cfg.device
    )
    if args.checkpoint:
        runner.load(
            str(args.checkpoint.resolve()),
            map_location=cfg.device,
            allow_muscle_override=args.mode == "eval",
        )
    if args.mode == "train":
        runner.learn(args.iterations, init_at_random_ep_len=False)
        return
    policy = runner.get_inference_policy(device=cfg.device)
    viewer = None
    if args.viewer:
        import mujoco
        import mujoco.viewer
        import mujoco_warp as mjw

        display_data = mujoco.MjData(env.model)
        viewer = mujoco.viewer.launch_passive(env.model, display_data)
    try:
        for i in range(args.steps):
            with torch.inference_mode():
                _, rewards, dones, _ = env.step(policy(env.get_observations()))
            if i % 100 == 0:
                print(
                    f"step={i} mean_reward={rewards.mean().item():.4f} resets={dones.sum().item()}",
                    flush=True,
                )
            if viewer:
                if not viewer.is_running():
                    break
                # Rendering only: this host copy never feeds training/simulation.
                mjw.get_data_into(display_data, env.model, env.sim.data, world_id=0)
                viewer.cam.lookat[:] = display_data.xpos[env.model.body("pelvis").id]
                viewer.sync()
    finally:
        if viewer:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
