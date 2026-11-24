"""Experiments 4–8: separate muscle/prosthesis PPOs on CUDA."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "eval"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1200,
        help="Total iteration limit, including resumed updates",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--human-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--steps-per-env", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--muscle-lr", type=float)
    parser.add_argument("--muscle-lr-min", type=float)
    parser.add_argument("--muscle-lr-max", type=float)
    parser.add_argument("--critic-warmup", type=int)
    parser.add_argument("--muscle-std-scale", type=float)
    parser.add_argument(
        "--prosthesis-reward", choices=("prosthesis_space", "human_space")
    )
    parser.add_argument("--reward-mode", choices=("separate", "shared"))
    parser.add_argument(
        "--prosthesis-observation",
        choices=("baseline", "reference", "muscle_reference", "hip_rpy"),
        help="muscle_reference: knee/ankle targets at muscle lookahead times + phase; reference: legacy current joint targets",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    import torch

    from myowarp.tasks.musculoskeletal_prosthesis.config import ProsthesisCfg
    from myowarp.tasks.musculoskeletal_prosthesis.runner import (
        validate_checkpoint_config,
    )
    from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
        create_controller,
    )
    from myowarp.tasks.musculoskeletal_prosthesis.split_config import SplitTrainingCfg
    from myowarp.tasks.musculoskeletal_prosthesis.split_env import (
        PROSTHESIS_REWARD_TERMS,
        SplitProsthesisEnv,
    )
    from myowarp.tasks.musculoskeletal_prosthesis.split_runner import (
        PIPELINE,
        SplitRunner,
    )

    if args.mode == "eval" and args.checkpoint is None:
        parser.error("eval requires --checkpoint")
    num_envs = (
        args.num_envs
        if args.num_envs is not None
        else {"train": 1024, "check": 2, "eval": 1}[args.mode]
    )
    if num_envs < 1 or args.iterations < 1 or args.steps < 1:
        parser.error(
            "Environment, iteration and evaluation-step counts must be positive"
        )
    if args.viewer and (args.mode != "eval" or num_envs != 1):
        parser.error("--viewer requires eval --num-envs 1")
    cfg = ProsthesisCfg(
        checkpoint_path=str(ROOT / "assets/musclemimic/checkpoints/mm-10m-2"),
        train_muscle=True,
        right_leg_obs="virtual",
    )
    cfg.select_anatomy("removed")
    training = SplitTrainingCfg()
    saved_environment, saved_training = None, None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("infos", {}).get("pipeline") != PIPELINE:
            parser.error(
                "Use an Experiment 4–8 checkpoint; experiments 1–3 have different architectures"
            )
        cfg = ProsthesisCfg(**saved["infos"]["prosthesis_config"])
        training = SplitTrainingCfg(**saved["infos"]["training_config"])
        saved_environment = saved["infos"]["prosthesis_config"]
        saved_training = asdict(
            training
        )  # Older checkpoints default to separate rewards and baseline observations.
        del saved
    if args.device:
        cfg.device = args.device
    if args.human_checkpoint:
        cfg.checkpoint_path = str(args.human_checkpoint.resolve())
    for field, value in (
        ("steps_per_env", args.steps_per_env),
        ("save_interval", args.save_interval),
        ("muscle_learning_rate", args.muscle_lr),
        ("muscle_lr_min", args.muscle_lr_min),
        ("muscle_lr_max", args.muscle_lr_max),
        ("critic_warmup", args.critic_warmup),
        ("muscle_std_scale", args.muscle_std_scale),
        ("reward_mode", args.reward_mode),
        ("prosthesis_reward", args.prosthesis_reward),
        ("prosthesis_observation", args.prosthesis_observation),
    ):
        if value is not None:
            setattr(training, field, value)
    cfg.muscle_learning_rate = training.muscle_learning_rate
    cfg.validate()
    training.validate()
    if args.checkpoint:
        try:
            validate_checkpoint_config(saved_environment, asdict(cfg))
            if saved_training != asdict(training):
                raise ValueError(
                    "Independent-PPO training overrides differ from the saved checkpoint"
                )
        except ValueError as error:
            parser.error(str(error))
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    output = None
    if args.mode == "train":
        output = args.output or ROOT / "logs/musculoskeletal_prosthesis" / (
            {
                4: "04_independent_",
                5: "05_shared_",
                6: "06_muscle_reference_"
                if training.prosthesis_observation == "muscle_reference"
                else "06_reference_",
                7: "07_hip_rpy_",
                8: "08_human_space_",
            }[training.experiment]
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if (
            output.exists()
            and any(output.glob("model_*.pt"))
            and args.checkpoint is None
        ):
            parser.error(
                "Output already contains checkpoints; choose a new folder or resume with --checkpoint"
            )
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(
                {
                    "environment": asdict(cfg),
                    "training": asdict(training),
                    "prosthesis_reward_terms": (
                        [
                            k
                            for k in PROSTHESIS_REWARD_TERMS
                            if k not in ("foot_pos", "foot_ori", "foot_vel")
                        ]
                        + [
                            f"human_{k}"
                            for k in (
                                "qpos",
                                "qvel",
                                "rpos",
                                "rquat",
                                "rvel_rot",
                                "rvel_lin",
                            )
                        ]
                        if training.prosthesis_reward == "human_space"
                        else list(PROSTHESIS_REWARD_TERMS)
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=num_envs)
    runner = SplitRunner(env, training, output)
    if args.checkpoint:
        runner.load(args.checkpoint)
    if args.mode == "train":
        experiment = training.experiment
        print(
            f"Experiment {experiment}: {num_envs} CUDA envs; output={output}",
            flush=True,
        )
        try:
            runner.learn(args.iterations)
        finally:
            env.close()
        return
    policy = runner.inference_policy()
    if args.mode == "check":
        with torch.inference_mode():
            obs = env.get_observations()
            action = policy(obs)
            _, prosthesis_reward, _, extras = env.step(action)
        print(
            json.dumps(
                {
                    "device": env.device,
                    "num_envs": num_envs,
                    "muscle_actor_and_critic_observation": list(obs["muscle"].shape),
                    "prosthesis_actor_observation": list(obs["actor"].shape),
                    "prosthesis_critic_observation": list(obs["critic"].shape),
                    "joint_action_shape": list(action.shape),
                    "muscle_reward_mean": float(extras["muscle_reward"].mean()),
                    "prosthesis_reward_mean": float(prosthesis_reward.mean()),
                    "finite": bool(
                        torch.isfinite(prosthesis_reward).all()
                        & torch.isfinite(extras["muscle_reward"]).all()
                    ),
                    "training": asdict(training),
                },
                indent=2,
            )
        )
        env.close()
        return
    viewer = None
    if args.viewer:
        import mujoco
        import mujoco.viewer
        import mujoco_warp as mjw

        display = mujoco.MjData(env.model)
        viewer = mujoco.viewer.launch_passive(env.model, display)
    try:
        with torch.inference_mode():
            for i in range(args.steps):
                _, reward, done, extras = env.step(policy(env.get_observations()))
                if i % 100 == 0:
                    print(
                        f"step={i} muscle_reward={extras['muscle_reward'].mean().item():.4f} prosthesis_reward={reward.mean().item():.4f} resets={done.sum().item()}",
                        flush=True,
                    )
                if viewer:
                    if not viewer.is_running():
                        break
                    mjw.get_data_into(display, env.model, env.sim.data, world_id=0)
                    viewer.cam.lookat[:] = display.xpos[env.model.body("pelvis").id]
                    viewer.sync()
    finally:
        if viewer:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
