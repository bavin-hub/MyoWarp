#!/usr/bin/env python3
"""Evaluate standalone H1-with-OSL velocity or motion-tracking tasks."""

import argparse
import copy
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, REPO_ROOT / "rsl_rl"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import mujoco  # noqa: E402
import torch  # noqa: E402

from myowarp.tasks.h1_with_osl_velocity import (  # noqa: E402
    H1VelocityRunner,
    H1WithOslVelocityCfg,
    H1WithOslVelocityEnv,
    make_rsl_rl_cfg as make_velocity_agent_cfg,
)
from myowarp.tasks.h1_with_osl_dual_full_velocity import (  # noqa: E402
    H1DualFullVelocityRunner,
    H1WithOslDualFullVelocityCfg,
    H1WithOslDualFullVelocityEnv,
    make_rsl_rl_cfg as make_dual_full_velocity_agent_cfg,
)
from myowarp.tasks.h1_with_osl_dual_partial_velocity import (  # noqa: E402
    H1DualPartialVelocityRunner,
    H1WithOslDualPartialVelocityCfg,
    H1WithOslDualPartialVelocityEnv,
    make_rsl_rl_cfg as make_dual_partial_velocity_agent_cfg,
)
from myowarp.tasks.h1_with_osl_dual_full_tracking import (  # noqa: E402
    H1DualFullTrackingRunner,
    H1WithOslDualFullTrackingCfg,
    H1WithOslDualFullTrackingEnv,
    make_rsl_rl_cfg as make_dual_full_tracking_agent_cfg,
)
from myowarp.tasks.h1_with_osl_dual_partial_tracking import (  # noqa: E402
    H1DualPartialTrackingRunner,
    H1WithOslDualPartialTrackingCfg,
    H1WithOslDualPartialTrackingEnv,
    make_rsl_rl_cfg as make_dual_partial_tracking_agent_cfg,
)
from myowarp.tasks.h1_with_osl_tracking import (  # noqa: E402
    H1TrackingRunner,
    H1WithOslTrackingCfg,
    H1WithOslTrackingEnv,
    make_rsl_rl_cfg as make_tracking_agent_cfg,
)


TASK_ALIASES = {
    "h1_with_osl_velocity": "h1_with_osl_velocity",
    "MyoWarp-H1-With-OSL-Flat": "h1_with_osl_velocity",
    "h1_with_osl_tracking": "h1_with_osl_tracking",
    "MyoWarp-H1-With-OSL-Tracking": "h1_with_osl_tracking",
    "h1_with_osl_dual_full_velocity": "h1_with_osl_dual_full_velocity",
    "MyoWarp-H1-With-OSL-Dual-Full-Flat": "h1_with_osl_dual_full_velocity",
    "h1_with_osl_dual_partial_velocity": "h1_with_osl_dual_partial_velocity",
    "MyoWarp-H1-With-OSL-Dual-Partial-Flat": "h1_with_osl_dual_partial_velocity",
    "h1_with_osl_dual_full_tracking": "h1_with_osl_dual_full_tracking",
    "MyoWarp-H1-With-OSL-Dual-Full-Tracking": "h1_with_osl_dual_full_tracking",
    "h1_with_osl_dual_partial_tracking": "h1_with_osl_dual_partial_tracking",
    "MyoWarp-H1-With-OSL-Dual-Partial-Tracking": "h1_with_osl_dual_partial_tracking",
}

VELOCITY_TASKS = {
    "h1_with_osl_velocity",
    "h1_with_osl_dual_full_velocity",
    "h1_with_osl_dual_partial_velocity",
}
TRACKING_TASKS = {
    "h1_with_osl_tracking",
    "h1_with_osl_dual_full_tracking",
    "h1_with_osl_dual_partial_tracking",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", nargs="?", default="h1_with_osl_velocity", choices=TASK_ALIASES
    )
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="disable motion-reset, startup, and observation randomization",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    args = parser.parse_args()
    if args.agent == "trained" and args.checkpoint_file is None:
        parser.error("--checkpoint-file is required when --agent=trained")
    task = TASK_ALIASES[args.task]
    if args.num_envs is None:
        args.num_envs = 1024 if task in TRACKING_TASKS and not args.viewer else 1
    if args.steps is None and task in VELOCITY_TASKS:
        args.steps = 1000
    if args.viewer and args.num_envs != 1:
        parser.error("--viewer requires --num-envs 1")
    if task in TRACKING_TASKS and args.command is not None:
        parser.error("--command applies only to velocity tracking")
    if task in VELOCITY_TASKS and args.deterministic:
        parser.error("--deterministic applies only to motion tracking")
    return args


def _policy(args: argparse.Namespace, env, runner_class, make_agent_cfg):
    if args.agent == "zero":
        return lambda observations: torch.zeros(
            (env.num_envs, env.num_actions), device=env.device
        )
    if args.agent == "random":
        return lambda observations: torch.randn(
            (env.num_envs, env.num_actions), device=env.device
        )
    runner = runner_class(
        env,
        copy.deepcopy(make_agent_cfg()),
        log_dir=None,
        device=args.device,
    )
    runner.load(
        args.checkpoint_file,
        load_cfg={"actor": True, "critic": False, "optimizer": False},
        map_location=args.device,
    )
    return runner.get_inference_policy(device=args.device)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    task = TASK_ALIASES[args.task]
    if task == "h1_with_osl_velocity":
        env_cfg = H1WithOslVelocityCfg(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            enable_observation_noise=False,
        )
        env_class = H1WithOslVelocityEnv
        runner_class = H1VelocityRunner
        make_agent_cfg = make_velocity_agent_cfg
    elif task == "h1_with_osl_dual_full_velocity":
        env_cfg = H1WithOslDualFullVelocityCfg(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            enable_observation_noise=False,
        )
        env_class = H1WithOslDualFullVelocityEnv
        runner_class = H1DualFullVelocityRunner
        make_agent_cfg = make_dual_full_velocity_agent_cfg
    elif task == "h1_with_osl_dual_partial_velocity":
        env_cfg = H1WithOslDualPartialVelocityCfg(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            enable_observation_noise=False,
        )
        env_class = H1WithOslDualPartialVelocityEnv
        runner_class = H1DualPartialVelocityRunner
        make_agent_cfg = make_dual_partial_velocity_agent_cfg
    else:
        if task == "h1_with_osl_dual_full_tracking":
            env_cfg = H1WithOslDualFullTrackingCfg(
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                enable_observation_noise=not args.deterministic,
            )
            env_class = H1WithOslDualFullTrackingEnv
            runner_class = H1DualFullTrackingRunner
            make_agent_cfg = make_dual_full_tracking_agent_cfg
        elif task == "h1_with_osl_dual_partial_tracking":
            env_cfg = H1WithOslDualPartialTrackingCfg(
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                enable_observation_noise=not args.deterministic,
            )
            env_class = H1WithOslDualPartialTrackingEnv
            runner_class = H1DualPartialTrackingRunner
            make_agent_cfg = make_dual_partial_tracking_agent_cfg
        else:
            env_cfg = H1WithOslTrackingCfg(
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                enable_observation_noise=not args.deterministic,
            )
            env_class = H1WithOslTrackingEnv
            runner_class = H1TrackingRunner
            make_agent_cfg = make_tracking_agent_cfg
        if args.motion_file is not None:
            env_cfg.motion_path = str(args.motion_file.resolve())
        env_cfg.motion.sampling_mode = "start"
        env_cfg.domain_randomization.enable_pushes = False
        if args.deterministic:
            env_cfg.motion.pose_x = (0.0, 0.0)
            env_cfg.motion.pose_y = (0.0, 0.0)
            env_cfg.motion.pose_z = (0.0, 0.0)
            env_cfg.motion.pose_roll = (0.0, 0.0)
            env_cfg.motion.pose_pitch = (0.0, 0.0)
            env_cfg.motion.pose_yaw = (0.0, 0.0)
            env_cfg.motion.velocity_x = (0.0, 0.0)
            env_cfg.motion.velocity_y = (0.0, 0.0)
            env_cfg.motion.velocity_z = (0.0, 0.0)
            env_cfg.motion.velocity_roll = (0.0, 0.0)
            env_cfg.motion.velocity_pitch = (0.0, 0.0)
            env_cfg.motion.velocity_yaw = (0.0, 0.0)
            env_cfg.motion.joint_position_range = (0.0, 0.0)
    if task in VELOCITY_TASKS or args.deterministic:
        env_cfg.domain_randomization.enabled = False
    env = env_class(env_cfg)
    viewer = None
    try:
        policy = _policy(args, env, runner_class, make_agent_cfg)
        if args.command is not None:
            env.set_command(*args.command)
        if task in TRACKING_TASKS:
            env.set_motion_frame(args.start_frame)

        render_data = None
        if args.viewer:
            import mujoco.viewer

            render_data = mujoco.MjData(env.model)
            viewer = mujoco.viewer.launch_passive(env.model, render_data)

        observations = env.get_observations()
        returns = torch.zeros(env.num_envs, device=env.device)
        episode_returns: list[float] = []
        falls = 0
        tracking_successes = 0
        tracking_episodes = 0
        tracking_metric_sums: dict[str, torch.Tensor] = {}
        tracking_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        tracking_active_steps = torch.zeros(env.num_envs, device=env.device)
        start = time.perf_counter()
        executed_steps = 0
        while args.steps is None or executed_steps < args.steps:
            active_tracking = ~tracking_done
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, extras = env.step(actions)
            executed_steps += 1
            if task in TRACKING_TASKS:
                tracking_active_steps += active_tracking.float()
                for name, value in env.evaluation_metrics().items():
                    tracking_metric_sums[name] = tracking_metric_sums.get(
                        name, torch.zeros(env.num_envs, device=env.device)
                    ) + torch.where(active_tracking, value, 0.0)
            returns += rewards
            finished = dones.bool()
            newly_done = finished
            if task in TRACKING_TASKS:
                newly_done = finished & ~tracking_done
            episode_returns.extend(returns[newly_done].detach().cpu().tolist())
            returns[newly_done] = 0.0
            fall_count = extras["log"].get("Episode_Termination/fell_over", 0)
            falls += int(fall_count.item() if isinstance(fall_count, torch.Tensor) else fall_count)
            if task in TRACKING_TASKS:
                terminated = extras["terminated"].bool()
                time_outs = extras["time_outs"].bool()
                tracking_successes += int(
                    (newly_done & time_outs & ~terminated).sum().item()
                )
                tracking_episodes += int(newly_done.sum().item())
                tracking_done |= newly_done

            if viewer is not None and render_data is not None:
                qpos, qvel = env.cpu_state()
                render_data.qpos[:] = qpos.numpy()
                render_data.qvel[:] = qvel.numpy()
                mujoco.mj_forward(env.model, render_data)
                viewer.sync()
                target_step_time = env.step_dt / max(args.realtime_factor, 1.0e-6)
                time.sleep(max(0.0, target_step_time))
                if not viewer.is_running():
                    break
            if task in TRACKING_TASKS and tracking_done.all():
                break
        elapsed = time.perf_counter() - start
        samples = max(1, executed_steps * args.num_envs)
        mean_return = (
            sum(episode_returns) / len(episode_returns)
            if episode_returns
            else float(returns.mean())
        )
        print(f"Mean return: {mean_return:.4f}")
        print(f"Completed episodes: {len(episode_returns)}; falls: {falls}")
        print(f"Throughput: {samples / elapsed:.1f} environment steps/s")
        if tracking_metric_sums:
            success_rate = tracking_successes / max(tracking_episodes, 1)
            print(f"success_rate: {success_rate:.4f}")
            for name, value in tracking_metric_sums.items():
                per_env = value / torch.clamp(tracking_active_steps, min=1.0)
                print(f"{name}: {float(per_env.mean()):.4f}")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
