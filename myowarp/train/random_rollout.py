from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myowarp.config import load_config
from myowarp.envs import MyoAssistLegWarpEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imitation_tutorial_22_separated_net_partial_obs.json")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(root / args.config)
    if args.num_envs is not None:
        config.warp_params.num_envs = args.num_envs
    if args.device is not None:
        config.warp_params.device = args.device

    env = MyoAssistLegWarpEnv(config=config, root_dir=root)
    obs = env.reset()
    print(f"obs shape: {tuple(obs.shape)}")
    for step in range(args.steps):
        actions = torch.zeros((env.num_envs, env.action_size), device=env.device)
        obs, reward, done, info = env.step(actions)
        print(
            f"step={step} obs={tuple(obs.shape)} "
            f"reward_mean={reward.mean().item():.6f} done={done.float().mean().item():.3f}"
        )


if __name__ == "__main__":
    main()
