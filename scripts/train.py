#!/usr/bin/env python3
"""Train standalone H1-with-OSL tasks with MuJoCo-Warp and local RSL-RL."""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, REPO_ROOT / "rsl_rl"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", nargs="?", default="h1_with_osl_velocity", choices=TASK_ALIASES
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--log-dir",
        type=Path,
    )
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument("--disable-observation-noise", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)

    task = TASK_ALIASES[args.task]
    if task == "h1_with_osl_velocity":
        max_iterations = args.max_iterations if args.max_iterations is not None else 10001
        save_interval = args.save_interval if args.save_interval is not None else 100
        env_cfg = H1WithOslVelocityCfg(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            enable_observation_noise=not args.disable_observation_noise,
        )
        agent_cfg = make_velocity_agent_cfg(
            max_iterations=max_iterations,
            save_interval=save_interval,
            run_name=args.run_name,
        )
        env_class = H1WithOslVelocityEnv
        runner_class = H1VelocityRunner
        experiment_name = "h1_with_osl_velocity_standalone"
    elif task == "h1_with_osl_dual_full_velocity":
        max_iterations = args.max_iterations if args.max_iterations is not None else 10001
        save_interval = args.save_interval if args.save_interval is not None else 100
        env_cfg = H1WithOslDualFullVelocityCfg(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            enable_observation_noise=not args.disable_observation_noise,
        )
        agent_cfg = make_dual_full_velocity_agent_cfg(
            max_iterations=max_iterations,
            save_interval=save_interval,
            run_name=args.run_name,
        )
        env_class = H1WithOslDualFullVelocityEnv
        runner_class = H1DualFullVelocityRunner
        experiment_name = "h1_with_osl_dual_full_velocity_standalone"
    elif task == "h1_with_osl_dual_partial_velocity":
        max_iterations = args.max_iterations if args.max_iterations is not None else 10001
        save_interval = args.save_interval if args.save_interval is not None else 100
        env_cfg = H1WithOslDualPartialVelocityCfg(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            enable_observation_noise=not args.disable_observation_noise,
        )
        agent_cfg = make_dual_partial_velocity_agent_cfg(
            max_iterations=max_iterations,
            save_interval=save_interval,
            run_name=args.run_name,
        )
        env_class = H1WithOslDualPartialVelocityEnv
        runner_class = H1DualPartialVelocityRunner
        experiment_name = "h1_with_osl_dual_partial_velocity_standalone"
    else:
        max_iterations = args.max_iterations if args.max_iterations is not None else 30_000
        save_interval = args.save_interval if args.save_interval is not None else 500
        if task == "h1_with_osl_dual_full_tracking":
            env_cfg = H1WithOslDualFullTrackingCfg(
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                enable_observation_noise=not args.disable_observation_noise,
            )
            agent_cfg = make_dual_full_tracking_agent_cfg(
                max_iterations=max_iterations,
                save_interval=save_interval,
                run_name=args.run_name,
            )
            env_class = H1WithOslDualFullTrackingEnv
            runner_class = H1DualFullTrackingRunner
            experiment_name = "h1_with_osl_dual_full_tracking_standalone"
        elif task == "h1_with_osl_dual_partial_tracking":
            env_cfg = H1WithOslDualPartialTrackingCfg(
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                enable_observation_noise=not args.disable_observation_noise,
            )
            agent_cfg = make_dual_partial_tracking_agent_cfg(
                max_iterations=max_iterations,
                save_interval=save_interval,
                run_name=args.run_name,
            )
            env_class = H1WithOslDualPartialTrackingEnv
            runner_class = H1DualPartialTrackingRunner
            experiment_name = "h1_with_osl_dual_partial_tracking_standalone"
        else:
            env_cfg = H1WithOslTrackingCfg(
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                enable_observation_noise=not args.disable_observation_noise,
            )
            agent_cfg = make_tracking_agent_cfg(
                max_iterations=max_iterations,
                save_interval=save_interval,
                run_name=args.run_name,
            )
            env_class = H1WithOslTrackingEnv
            runner_class = H1TrackingRunner
            experiment_name = "h1_with_osl_tracking_standalone"
        if args.motion_file is not None:
            env_cfg.motion_path = str(args.motion_file.resolve())
    env_cfg.domain_randomization.enabled = not args.disable_domain_randomization
    agent_cfg["seed"] = args.seed
    agent_cfg["num_steps_per_env"] = args.num_steps_per_env

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.run_name}" if args.run_name else ""
    log_root = args.log_dir or REPO_ROOT / "logs/rsl_rl" / experiment_name
    run_dir = log_root / f"{timestamp}{suffix}"
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=False)
    (params_dir / "env.json").write_text(json.dumps(env_cfg.to_dict(), indent=2) + "\n")
    (params_dir / "agent.json").write_text(json.dumps(agent_cfg, indent=2) + "\n")

    print(f"Task: {args.task} (standalone MuJoCo-Warp)")
    print(f"Run directory: {run_dir}")
    env = env_class(env_cfg)
    try:
        runner = runner_class(env, copy.deepcopy(agent_cfg), str(run_dir), device=args.device)
        runner.add_git_repo_to_log(str(REPO_ROOT))
        if args.resume is not None:
            runner.load(args.resume, map_location=args.device)
            print(f"Resumed from: {args.resume}")
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
        runner.export_policy_to_onnx(str(run_dir / "exported"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
