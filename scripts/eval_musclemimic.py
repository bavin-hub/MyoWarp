"""Evaluate the repository-local MuscleMimic policy on the full biological body."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "assets/musclemimic/checkpoints/mm-10m-2")
    parser.add_argument("--clip", type=Path,
                        default=ROOT / "assets/musclemimic/motions/MyoFullBody/gmr/KIT/314/walking_medium09_poses.npz")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--view-model", action="store_true", help="Static inspection; no policy or clip needed")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/musclemimic/eval.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from myowarp.musclemimic.model import build_model
    from myowarp.musclemimic.policy import (
        FullbodyObsAdapter, LocalPolicyRunner,
        fullbody_history_settings_from_metadata,
        fullbody_obs_adapter_params_from_metadata,
        load_local_policy_artifacts, read_checkpoint_config_metadata,
    )
    from myowarp.musclemimic.trajectory import expand_motion_clip_to_model, load_motion_clip

    model, _spec = build_model(model_version="1.0.5")
    data = mujoco.MjData(model)
    print(f"MyoFullBody 1.0.5: nq={model.nq}, nv={model.nv}, muscles={model.nu}", flush=True)
    if args.view_model:
        mujoco.mj_forward(model, data)
        if args.headless:
            return
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3
            viewer.cam.lookat[:] = data.xpos[model.body("pelvis").id]
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        return
    if not args.clip.is_file():
        parser.error(f"Motion clip missing: {args.clip}. Request dataset access, sign in with hf auth login, "
                     "then run scripts/fetch_musclemimic.py")
    if not args.checkpoint.is_dir():
        parser.error("Checkpoint missing; run scripts/fetch_musclemimic.py")
    clip = expand_motion_clip_to_model(
        load_motion_clip(args.clip.resolve(), expected_nq=model.nq, expected_nv=model.nv), model,
    )
    if clip.frequency_hz is not None and not np.isclose(clip.frequency_hz, 100):
        parser.error(f"Expected a 100 Hz retargeted clip, found {clip.frequency_hz} Hz")
    metadata = read_checkpoint_config_metadata(args.checkpoint.resolve())
    runner = LocalPolicyRunner(
        artifacts=load_local_policy_artifacts(args.checkpoint.resolve()), stochastic=False,
        seed=args.seed, obs_adapter=FullbodyObsAdapter(
            model=model, clip=clip, goal_params=fullbody_obs_adapter_params_from_metadata(metadata),
        ), **fullbody_history_settings_from_metadata(metadata),
    )
    # The reference initializes physics once. Subsequent states come only from
    # muscle actions and forward dynamics, with no root assistance or pose resets.
    data.qpos[:] = clip.qpos[0]
    if clip.qvel is not None:
        data.qvel[:] = clip.qvel[0]
    mujoco.mj_forward(model, data)
    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 3
    rows, states, actions = [], [], []
    failure = None
    count = min(args.steps, len(clip.qpos))
    pelvis = model.body("pelvis").id
    try:
        for frame in range(count):
            started = time.monotonic()
            action = runner.action_for(data=data, clip=clip, frame_idx=frame)
            if not np.isfinite(action).all():
                failure = "Non-finite policy action"
                break
            before = data.time
            runner.step(model, data, action)
            if data.time <= before or not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                failure = "Non-finite state or simulator automatic reset"
                break
            mujoco.mj_forward(model, data)
            # State after action frame i is compared with reference frame i+1.
            target = clip.qpos[min(frame + 1, len(clip.qpos) - 1)]
            row = {"step": frame + 1, "time": float(data.time),
                   "pelvis_height_m": float(data.xpos[pelvis, 2]),
                   "root_horizontal_error_m": float(np.linalg.norm(data.qpos[:2] - target[:2])),
                   "mean_muscle_activation": float(data.act.mean())}
            rows.append(row)
            states.append(data.qpos.copy())
            actions.append(action.copy())
            if frame % 100 == 0:
                print(row, flush=True)
            if row["pelvis_height_m"] < 0.5:
                failure = "Pelvis below 0.5 m (fall watchdog)"
                break
            if viewer is not None:
                if not viewer.is_running():
                    failure = "Viewer closed"
                    break
                viewer.cam.lookat[:] = data.xpos[pelvis]
                viewer.sync()
                time.sleep(max(0, runner.frame_skip * model.opt.timestep - (time.monotonic() - started)))
    finally:
        if viewer is not None:
            viewer.close()
    report = {"checkpoint": str(args.checkpoint.resolve()), "clip": str(args.clip.resolve()),
              "model_version": "1.0.5", "muscles": model.nu,
              "requested_steps": count, "completed_steps": len(rows), "failure": failure,
              "completed": failure is None and len(rows) == count, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), qpos=np.array(states), muscle_actions=np.array(actions))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
