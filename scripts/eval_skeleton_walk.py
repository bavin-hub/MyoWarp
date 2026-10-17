"""Evaluate a SkeletonTorque PPOJax checkpoint with native MuJoCo physics."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "loco-mujoco"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("HF_HOME", str(ROOT / "logs/locomujoco/huggingface"))
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import jax
    import numpy as np
    from loco_mujoco import ImitationFactory
    from loco_mujoco.algorithms import PPOJax

    agent_conf, agent_state = PPOJax.load_agent(str(args.checkpoint.resolve()))
    config = agent_conf.config
    if config.experiment.env_params.env_name != "MjxSkeletonTorque":
        parser.error("This evaluator expects a SkeletonTorque checkpoint")
    params = dict(config.experiment.env_params)
    params["env_name"] = "SkeletonTorque"
    params["headless"] = args.headless
    params.pop("use_mjwarp", None)
    params.pop("nconmax", None)
    env = ImitationFactory.make(**params, **config.experiment.task_factory.params)
    @jax.jit
    def action_for(obs):
        (distribution, _value), _updates = agent_conf.network.apply(
            {"params": agent_state.train_state.params,
             "run_stats": agent_state.train_state.run_stats},
            obs, mutable=["run_stats"],
        )
        return distribution.mean()

    # Upstream play_policy_mujoco renders even with render=False. Drive its
    # environment API directly so headless evaluation remains display-free.
    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env._model, env._data)
        viewer.cam.distance = 3
    obs = env.reset()
    episodes, episode_return, completed_returns = 0, 0., []
    try:
        for _ in range(args.steps):
            started = time.monotonic()
            action = np.asarray(action_for(obs))
            if not np.isfinite(action).all():
                raise FloatingPointError("Non-finite policy action")
            obs, reward, absorbing, done, _info = env.step(action)
            episode_return += float(reward)
            if not (np.isfinite(env._data.qpos).all() and np.isfinite(env._data.qvel).all()):
                raise FloatingPointError("Non-finite simulator state")
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.cam.lookat[:] = env._data.subtree_com[0]
                viewer.sync()
                time.sleep(max(0, env.dt - (time.monotonic() - started)))
            if done or absorbing:
                episodes += 1
                completed_returns.append(episode_return)
                episode_return = 0.
                obs = env.reset()
    finally:
        if viewer is not None:
            viewer.close()
        env.stop()
    print(f"Completed episodes: {episodes}; mean episode return: "
          f"{np.mean(completed_returns) if completed_returns else episode_return:.3f}")


if __name__ == "__main__":
    main()
