from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch

from myowarp.config import load_config
from myowarp.envs import MyoAssistLegCpuEnv, MyoAssistLegWarpEnv
from myowarp.policies import GaussianActorCritic


def _count_info(info: dict, key: str) -> int:
    value = info.get(key)
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().bool().sum().item())
    return int(bool(value))


def _cause_text(info: dict) -> str:
    counts = {
        "trajectory": _count_info(info, "trajectory_done"),
        "height": _count_info(info, "height_done"),
        "truncated": _count_info(info, "truncated"),
        "out_of_reference": _count_info(info, "out_of_reference"),
    }
    active = [f"{key}={value}" for key, value in counts.items() if value]
    return " ".join(active) if active else "unknown"


def _copy_world_to_viewer(env: MyoAssistLegWarpEnv, data: mujoco.MjData, world_id: int = 0) -> None:
    qpos = env.backend.get_tensor("qpos")[world_id].detach().cpu().numpy()
    qvel = env.backend.get_tensor("qvel")[world_id].detach().cpu().numpy()
    ctrl = env.backend.get_tensor("ctrl")[world_id].detach().cpu().numpy()
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(env.cpu_model, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["cpu", "warp"], default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until the viewer is closed")
    args = parser.parse_args()

    root = Path.cwd()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_path = args.config or checkpoint.get("config_path", "configs/imitation_tutorial_22_separated_net_partial_obs.json")
    config = load_config(root / config_path)
    policy_device = torch.device(args.device or "cpu")

    if args.backend == "warp":
        config.warp_params.num_envs = 1
        config.env_params.num_envs = 1
        config.warp_params.device = str(policy_device)
        config.ppo_params["device"] = str(policy_device)
        env = MyoAssistLegWarpEnv(config=config, root_dir=root)
    else:
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
    viewer_data = env.data if isinstance(env, MyoAssistLegCpuEnv) else mujoco.MjData(env.cpu_model)
    sleep_dt = env.control_dt / max(args.speed, 1e-6)
    episode_return = 0.0
    episode_length = 0
    step = 0

    with mujoco.viewer.launch_passive(env.cpu_model, viewer_data) as viewer:
        while viewer.is_running():
            if args.steps > 0 and step >= args.steps:
                break
            frame_start = time.time()
            with torch.no_grad():
                action, _logprob, _value = model.act(obs.to(policy_device), deterministic=not args.stochastic)
                obs, reward, done, info = env.step(action.detach().cpu())
            episode_return += float(reward.detach().cpu().mean().item())
            episode_length += 1
            if bool(done.detach().cpu().any().item()):
                print(
                    f"episode_done step={step} return={episode_return:.6f} "
                    f"len={episode_length} cause={_cause_text(info)}"
                )
                episode_return = 0.0
                episode_length = 0
            if isinstance(env, MyoAssistLegWarpEnv):
                _copy_world_to_viewer(env, viewer_data)
            viewer.sync()
            step += 1
            elapsed = time.time() - frame_start
            if elapsed < sleep_dt:
                time.sleep(sleep_dt - elapsed)


if __name__ == "__main__":
    main()
