from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myowarp.config import load_config
from myowarp.policies import GaussianActorCritic


def _count_info(info: dict, key: str) -> int:
    value = info.get(key)
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().bool().sum().item())
    return int(bool(value))


def _cause_counts(info: dict) -> dict[str, int]:
    return {
        "trajectory": _count_info(info, "trajectory_done"),
        "height": _count_info(info, "height_done"),
        "truncated": _count_info(info, "truncated"),
        "out_of_reference": _count_info(info, "out_of_reference"),
    }


def _cause_text(counts: dict[str, int]) -> str:
    active = [f"{key}={value}" for key, value in counts.items() if value]
    return " ".join(active) if active else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["cpu", "warp"], default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_path = args.config or checkpoint.get("config_path", "configs/imitation_tutorial_22_separated_net_partial_obs.json")
    config = load_config(root / config_path)
    policy_device = torch.device(args.device or "cpu")

    if args.backend == "warp":
        from myowarp.envs.myoassist_leg_warp import MyoAssistLegWarpEnv

        config.warp_params.num_envs = args.num_envs
        config.env_params.num_envs = args.num_envs
        config.warp_params.device = str(policy_device)
        config.ppo_params["device"] = str(policy_device)
        env = MyoAssistLegWarpEnv(config=config, root_dir=root)
    else:
        from myowarp.envs.myoassist_leg_cpu import MyoAssistLegCpuEnv

        config.warp_params.num_envs = 1
        config.env_params.num_envs = 1
        config.warp_params.device = "cpu"
        env = MyoAssistLegCpuEnv(config=config, root_dir=root)

    model = GaussianActorCritic(
        obs_size=env.observation_size,
        action_size=env.action_size,
        policy_params=config.policy_params,
    ).to(policy_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    obs = env.reset()
    returns = torch.zeros(env.num_envs, dtype=torch.float32)
    lengths = torch.zeros(env.num_envs, dtype=torch.float32)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    total_cause_counts = {"trajectory": 0, "height": 0, "truncated": 0, "out_of_reference": 0}

    with torch.no_grad():
        for step in range(args.steps):
            action, _logprob, _value = model.act(obs.to(policy_device), deterministic=not args.stochastic)
            obs, reward, done, info = env.step(action.detach().cpu())
            reward_cpu = reward.detach().cpu()
            done_cpu = done.detach().cpu()
            returns += reward_cpu
            lengths += 1.0
            if torch.any(done_cpu):
                counts = _cause_counts(info)
                for key, value in counts.items():
                    total_cause_counts[key] += value
                episode_returns = returns[done_cpu].tolist()
                episode_lengths = lengths[done_cpu].tolist()
                completed_returns.extend(returns[done_cpu].tolist())
                completed_lengths.extend(lengths[done_cpu].tolist())
                if len(episode_returns) <= 8:
                    for episode_return, episode_length in zip(episode_returns, episode_lengths, strict=False):
                        print(
                            f"episode_done step={step} return={episode_return:.6f} "
                            f"len={episode_length:.0f} cause={_cause_text(counts)}"
                        )
                else:
                    print(
                        f"episodes_done step={step} count={len(episode_returns)} "
                        f"return_mean={sum(episode_returns) / len(episode_returns):.6f} "
                        f"len_mean={sum(episode_lengths) / len(episode_lengths):.2f} "
                        f"cause={_cause_text(counts)}"
                    )
                returns[done_cpu] = 0.0
                lengths[done_cpu] = 0.0
            if step % 100 == 0 or step == args.steps - 1:
                print(
                    f"step={step} reward_mean={reward_cpu.mean().item():.6f} "
                    f"done_frac={done_cpu.float().mean().item():.3f}"
                )

    if completed_returns:
        print(f"episodes={len(completed_returns)} return_mean={sum(completed_returns) / len(completed_returns):.6f}")
        print(f"length_mean={sum(completed_lengths) / len(completed_lengths):.2f}")
        print(f"termination_counts={total_cause_counts}")
    else:
        print(f"no completed episodes; partial_return_mean={returns.mean().item():.6f}")


if __name__ == "__main__":
    main()
