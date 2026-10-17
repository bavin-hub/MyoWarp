"""Inspect LocoMuJoCo models in a rotatable MuJoCo viewer (no simulation)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="SkeletonTorque", help="Registered LocoMuJoCo model name")
    parser.add_argument("--list", action="store_true", help="List available model names")
    parser.add_argument("--headless", action="store_true", help="Compile and inspect without opening a window")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1] / "loco-mujoco"
    if not (checkout / "loco_mujoco").is_dir():
        parser.error("Missing submodule: run git submodule update --init loco-mujoco")
    sys.path.insert(0, str(checkout))
    import loco_mujoco
    import mujoco

    if not hasattr(loco_mujoco, "get_registered_envs"):
        parser.error("LocoMuJoCo dependencies are missing; use .venv-locomujoco/bin/python")
    registered = loco_mujoco.get_registered_envs()
    if args.list:
        print("\n".join(sorted(registered)))
        return
    if args.model not in registered:
        parser.error(f"Unknown model {args.model!r}; use --list")
    if args.model.startswith("Mjx"):
        parser.error("Use the standard model name (without Mjx) for static viewing")
    env = loco_mujoco.RLFactory.make(
        args.model, headless=True, goal_type="NoGoal", reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
    )
    env.reset()
    # LocoMuJoCo exposes its compiled native model/data through these fields.
    model, data = env._model, env._data
    mujoco.mj_forward(model, data)
    print(f"{args.model}: {model.nbody - 1} bodies, nq={model.nq}, nv={model.nv}, "
          f"{model.nu} actuators, mass={model.body_mass.sum():.2f} kg")
    if args.headless:
        return
    import mujoco.viewer

    print("Static model. Drag left mouse to rotate; scroll to zoom; right mouse to pan.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.subtree_com[0]
        viewer.cam.distance = max(2.5, 2 * model.stat.extent)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
