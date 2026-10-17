"""Train LocoMuJoCo's SkeletonTorque walking reference with upstream PPOJax."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "loco-mujoco"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("warp", "mjx"), default="warp")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=int, default=300_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/locomujoco/skeleton_walk")
    parser.add_argument("--smoke-test", action="store_true", help="Two tiny PPO updates; does not learn walking")
    parser.add_argument("--check-only", action="store_true", help="Load model, reference and agent config without training")
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    os.environ.setdefault("HF_HOME", str(ROOT / "logs/locomujoco/huggingface"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.backend == "warp" and args.device == "cpu":
        parser.error("Use --backend mjx for CPU checks; Warp training requires a CUDA GPU")
    import jax
    import numpy as np
    from loco_mujoco import ImitationFactory
    from loco_mujoco.algorithms import PPOJax
    from omegaconf import OmegaConf

    print(f"JAX devices: {jax.devices()}", flush=True)
    config = OmegaConf.load(ROOT / "loco-mujoco/examples/training_examples/jax_rl_mimic/conf.yaml")
    experiment = config.experiment
    experiment.task_factory.params = {"default_dataset_conf": {"task": "walk", "dataset_type": "mocap"}}
    experiment.env_params.env_name = "MjxSkeletonTorque"
    experiment.env_params.use_mjwarp = args.backend == "warp"
    experiment.env_params.nconmax = 128 * args.num_envs
    experiment.env_params.goal_params.visualize_goal = False
    experiment.num_envs = args.num_envs
    experiment.total_timesteps = args.total_timesteps
    experiment.n_seeds = 1
    # Keep upstream PPO and mimic reward settings; omit expensive optional
    # trajectory-distance validation during training. Metrics still get saved.
    experiment.validation.active = False
    experiment.validation.num = 1
    if args.smoke_test:
        experiment.num_envs = 2
        experiment.num_steps = 4
        experiment.total_timesteps = 16
        experiment.num_minibatches = 2
        experiment.update_epochs = 1
        experiment.hidden_layers = [32, 32]
        experiment.env_params.nconmax = 256
    if experiment.num_envs < 1 or experiment.total_timesteps < experiment.num_envs * experiment.num_steps:
        parser.error("Request at least one rollout batch of training")
    if experiment.num_envs * experiment.num_steps % experiment.num_minibatches:
        parser.error("Rollout batch must be divisible by the number of minibatches")
    env = ImitationFactory.make(**experiment.env_params, **experiment.task_factory.params)
    agent_conf = PPOJax.init_agent_conf(env, config)
    print(f"SkeletonTorque: {env._model.nu} actuators; {experiment.num_updates} PPO updates; "
          f"reference=walk, control=DefaultControl (torque)", flush=True)
    if args.check_only:
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    started = time.monotonic()
    train = jax.jit(PPOJax.build_train_fn(env, agent_conf))
    result = train(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(result)
    checkpoint = PPOJax.save_agent(output, agent_conf, result["agent_state"])
    metrics = result["training_metrics"]
    np.savez_compressed(output / "training_metrics.npz", **{
        name: np.asarray(getattr(metrics, name)) for name in metrics.__dataclass_fields__
    })
    (output / "run.json").write_text(json.dumps({
        "seed": args.seed, "backend": args.backend, "smoke_test": args.smoke_test,
        "wall_seconds": time.monotonic() - started, "checkpoint": str(checkpoint),
        "timesteps": int(experiment.num_updates * experiment.num_envs * experiment.num_steps),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
