from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from myowarp.config import TrainConfig, load_config
from myowarp.envs import MyoAssistLegWarpEnv
from myowarp.policies import GaussianActorCritic


def _ppo(config: TrainConfig, key: str, default: Any) -> Any:
    return config.ppo_params.get(key, default)


def _save_checkpoint(
    path: Path,
    *,
    model: GaussianActorCritic,
    optimizer: torch.optim.Optimizer,
    config_path: str,
    global_step: int,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config_path": config_path,
            "global_step": global_step,
            "update": update,
        },
        path,
    )


def train(config: TrainConfig, *, root: Path, config_path: str, log_dir: Path) -> None:
    torch.manual_seed(int(config.env_params.seed))
    env = MyoAssistLegWarpEnv(config=config, root_dir=root)
    device = torch.device(config.warp_params.device)

    model = GaussianActorCritic(
        obs_size=env.observation_size,
        action_size=env.action_size,
        policy_params=config.policy_params,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(_ppo(config, "learning_rate", 1e-4)), eps=1e-5)

    n_steps = int(_ppo(config, "n_steps", 1024))
    num_envs = env.num_envs
    rollout_size = n_steps * num_envs
    batch_size = min(int(_ppo(config, "batch_size", rollout_size)), rollout_size)
    n_epochs = int(_ppo(config, "n_epochs", 10))
    gamma = float(_ppo(config, "gamma", 0.99))
    gae_lambda = float(_ppo(config, "gae_lambda", 0.95))
    clip_range = float(_ppo(config, "clip_range", 0.2))
    clip_range_vf = _ppo(config, "clip_range_vf", None)
    clip_range_vf = None if clip_range_vf is None else float(clip_range_vf)
    ent_coef = float(_ppo(config, "ent_coef", 0.0))
    vf_coef = float(_ppo(config, "vf_coef", 0.5))
    max_grad_norm = float(_ppo(config, "max_grad_norm", 0.5))
    target_kl = _ppo(config, "target_kl", None)
    target_kl = None if target_kl is None else float(target_kl)
    total_timesteps = int(float(config.total_timesteps))
    total_updates = max(1, math.ceil(total_timesteps / rollout_size))
    save_every_updates = int(config.logger_params.get("logging_frequency", 8))

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "trained_models"
    metrics_path = log_dir / "train_log.jsonl"

    obs = env.reset()
    global_step = 0
    current_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(num_envs, dtype=torch.float32, device=device)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []

    print(f"training on {device} | envs={num_envs} n_steps={n_steps} rollout={rollout_size} batch={batch_size}")
    for update in range(1, total_updates + 1):
        obs_buf = torch.zeros((n_steps, num_envs, env.observation_size), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((n_steps, num_envs, env.action_size), dtype=torch.float32, device=device)
        logprobs_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        values_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        reward_components: dict[str, float] = {}
        done_components = {"height": 0.0, "trajectory": 0.0, "reference": 0.0}

        for step in range(n_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                action, logprob, value = model.act(obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            obs, reward, done, info = env.step(action)
            rewards_buf[step] = reward
            dones_buf[step] = done.float()
            global_step += num_envs

            current_returns += reward
            current_lengths += 1.0
            if torch.any(done):
                completed_returns.extend(current_returns[done].detach().cpu().tolist())
                completed_lengths.extend(current_lengths[done].detach().cpu().tolist())
                current_returns[done] = 0.0
                current_lengths[done] = 0.0

            for key, value in info["reward"].items():
                reward_components[key] = reward_components.get(key, 0.0) + float(value.mean().detach().cpu())
            done_components["height"] += float(info["height_done"].float().mean().detach().cpu())
            done_components["trajectory"] += float(info["trajectory_done"].float().mean().detach().cpu())
            done_components["reference"] += float(info["out_of_reference"].float().mean().detach().cpu())

        with torch.no_grad():
            _next_action, _next_logprob, next_value = model.act(obs, deterministic=True)

        advantages = torch.zeros_like(rewards_buf)
        last_gae = torch.zeros(num_envs, dtype=torch.float32, device=device)
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_nonterminal = 1.0 - dones_buf[step]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones_buf[step]
                next_values = values_buf[step + 1]
            delta = rewards_buf[step] + gamma * next_values * next_nonterminal - values_buf[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, env.observation_size))
        b_actions = actions_buf.reshape((-1, env.action_size))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        approx_kl = torch.tensor(0.0, device=device)
        clip_fraction = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        indices = torch.arange(rollout_size, device=device)
        for _epoch in range(n_epochs):
            permutation = indices[torch.randperm(rollout_size, device=device)]
            for start in range(0, rollout_size, batch_size):
                mb_idx = permutation[start : start + batch_size]
                new_logprob, entropy, new_value = model.evaluate_actions(b_obs[mb_idx], b_actions[mb_idx])
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()

                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                if clip_range_vf is None:
                    value_pred = new_value
                else:
                    value_pred = b_values[mb_idx] + torch.clamp(new_value - b_values[mb_idx], -clip_range_vf, clip_range_vf)
                value_loss = 0.5 * torch.square(value_pred - b_returns[mb_idx]).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > 1.5 * target_kl:
                break

        recent_returns = completed_returns[-100:]
        recent_lengths = completed_lengths[-100:]
        metrics = {
            "update": update,
            "global_step": global_step,
            "mean_reward": float(torch.mean(rewards_buf).detach().cpu()),
            "episode_return_mean": float(sum(recent_returns) / len(recent_returns)) if recent_returns else None,
            "episode_length_mean": float(sum(recent_lengths) / len(recent_lengths)) if recent_lengths else None,
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "clip_fraction": float(clip_fraction.detach().cpu()),
            "reward_components": {key: value / n_steps for key, value in reward_components.items()},
            "done_components": {key: value / n_steps for key, value in done_components.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        print(
            f"update={update}/{total_updates} step={global_step} "
            f"reward={metrics['mean_reward']:.4f} "
            f"ep_return={metrics['episode_return_mean']} "
            f"ep_len={metrics['episode_length_mean']} "
            f"done_traj={metrics['done_components']['trajectory']:.3f} "
            f"done_height={metrics['done_components']['height']:.3f} "
            f"kl={metrics['approx_kl']:.5f}",
            flush=True,
        )

        if update % save_every_updates == 0 or update == total_updates:
            _save_checkpoint(
                ckpt_dir / f"step_{global_step}.pt",
                model=model,
                optimizer=optimizer,
                config_path=config_path,
                global_step=global_step,
                update=update,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imitation_tutorial_22_separated_net_partial_obs.json")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--total-timesteps", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(root / args.config)
    if args.num_envs is not None:
        config.warp_params.num_envs = args.num_envs
        config.env_params.num_envs = args.num_envs
    if args.device is not None:
        config.warp_params.device = args.device
        config.ppo_params["device"] = args.device
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.n_steps is not None:
        config.ppo_params["n_steps"] = args.n_steps
    if args.batch_size is not None:
        config.ppo_params["batch_size"] = args.batch_size
    if args.iterations is not None:
        n_steps = int(config.ppo_params.get("n_steps", 1024))
        config.total_timesteps = int(args.iterations) * int(config.warp_params.num_envs) * n_steps

    log_dir = Path(args.log_dir) if args.log_dir else root / "results" / f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train(config, root=root, config_path=args.config, log_dir=log_dir)


if __name__ == "__main__":
    main()
