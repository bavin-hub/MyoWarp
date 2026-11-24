"""Record actual best-checkpoint rollouts with a rear-to-front whole-body orbit."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rsl_rl")]

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

RUN = ROOT / "logs/musculoskeletal_prosthesis/eight_experiments_20261021_231020"
EXPERIMENTS = {
    1: ("01_default_frozen", "Exp1: Frozen default muscle policy"),
    3: ("03_default_coopt", "Exp3: Joint muscle and prosthesis PPO"),
    4: ("04_independent", "Exp4: Separate PPOs and rewards"),
    5: ("05_shared", "Exp5: Separate PPOs, shared reward"),
    7: ("07_hip_rpy", "Exp7: Hip RPY history"),
    8: ("08_human_space", "Exp8: Separate PPOs, human-space prosthesis reward"),
    2: ("02_vyvas_frozen", "Exp2: Frozen Vyvas muscle policy"),
    6: ("06_muscle_reference", "Exp6: Separate PPOs, prosthesis reference"),
}


def rollout(exp, seconds, output, run=RUN):
    from myowarp.tasks.musculoskeletal_prosthesis.config import (
        ProsthesisCfg,
        make_agent_cfg,
    )
    from myowarp.tasks.musculoskeletal_prosthesis.gpu_env import GPUProsthesisVecEnv
    from myowarp.tasks.musculoskeletal_prosthesis.runner import ProsthesisRunner
    from myowarp.tasks.musculoskeletal_prosthesis.sea_controller import (
        create_controller,
    )

    folder, title = EXPERIMENTS[exp]
    split = exp >= 4
    checkpoint = (
        run / folder / ("model_best_prosthesis.pt" if split else "model_best.pt")
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_iteration = int(saved["iter"])
    cfg = ProsthesisCfg(**saved["infos"]["prosthesis_config"])
    torch.manual_seed(42)
    if split:
        from myowarp.tasks.musculoskeletal_prosthesis.split_config import (
            SplitTrainingCfg,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_env import (
            SplitProsthesisEnv,
        )
        from myowarp.tasks.musculoskeletal_prosthesis.split_runner import SplitRunner

        training = SplitTrainingCfg(**saved["infos"]["training_config"])
        env = SplitProsthesisEnv(cfg, training, create_controller, num_envs=1)
        runner = SplitRunner(env, training)
        runner.load(checkpoint)
        policy = runner.inference_policy()
    else:
        env = GPUProsthesisVecEnv(cfg, create_controller, num_envs=1)
        agent = make_agent_cfg()
        if cfg.train_muscle:
            agent["algorithm"]["class_name"] = (
                "myowarp.tasks.musculoskeletal_prosthesis.coopt:CooptPPO"
            )
        runner = ProsthesisRunner(env, agent, log_dir=None, device=env.device)
        runner.load(str(checkpoint), map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
    del saved
    env.eligible = env.eligible[
        env.eligible + env.max_episode_length < len(env.reference["qpos"])
    ]
    env.generator.manual_seed(12345)
    with torch.inference_mode():
        env.reset()
    states = [env.data.qpos[0].cpu().numpy().copy()]
    episodes = [1]
    events = []
    original_reset = env.reset
    terminal = {}

    def capture_reset(ids=None):
        if ids is not None and ids.numel():
            terminal["qpos"] = env.data.qpos[0].cpu().numpy().copy()
        return original_reset(ids)

    env.reset = capture_reset
    episode = 1
    start_frame = int(env.frame[0])
    tick = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(round(seconds / cfg.policy_dt)):
                terminal.clear()
                _, _, done, extra = env.step(policy(env.get_observations()))
                states.append(
                    terminal.get("qpos", env.data.qpos[0].cpu().numpy().copy())
                )
                episodes.append(episode)
                if bool(done[0]):
                    events.append(
                        {
                            "time_seconds": (step + 1) * cfg.policy_dt,
                            "episode": episode,
                            "reason": "episode timeout / reference end"
                            if bool(extra["time_outs"][0])
                            else "fall / tracking termination",
                        }
                    )
                    episode += 1
                if (step + 1) % 500 == 0:
                    print(
                        f"Exp{exp}: simulated {(step + 1) * cfg.policy_dt:.1f}/{seconds:.1f}s, resets={len(events)}, wall={time.monotonic() - tick:.1f}s",
                        flush=True,
                    )
    finally:
        env.close()
    np.savez_compressed(
        output / f"exp{exp}_rollout.npz",
        qpos=np.asarray(states),
        episode=episodes,
        dt=cfg.policy_dt,
    )
    meta = {
        "experiment": exp,
        "title": title,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "model_path": cfg.model_path,
        "seconds": seconds,
        "start_frame": start_frame,
        "seed": 12345,
        "events": events,
        "episode_limit_seconds": cfg.episode_seconds,
        "policy": "deterministic best checkpoint; automatic resets retained",
        "wall_seconds": time.monotonic() - tick,
    }
    (output / f"exp{exp}_recording.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def render(exp, output, fps, width, height, *, model_override=None, stem=None):
    prefix = stem or f"exp{exp}"
    meta = json.loads((output / f"{prefix}_recording.json").read_text())
    with np.load(output / f"{prefix}_rollout.npz") as pack:
        poses = pack["qpos"]
        episodes = pack["episode"]
        dt = float(pack["dt"])
    model = model_override if model_override is not None else mujoco.MjModel.from_xml_path(meta["model_path"])
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.quality.shadowsize = 2048
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.elevation = -12
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    data.qpos[:] = poses[0]
    mujoco.mj_forward(model, data)
    pelvis = model.body("pelvis").id
    forward = (
        data.geom_xpos[model.geom("toe").id] - data.geom_xpos[model.geom("heel").id]
    )
    local_forward = data.xmat[pelvis].reshape(3, 3).T @ forward
    geoms = np.flatnonzero(
        (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
        & (model.geom_group < 3)
        & (model.geom_rgba[:, 3] > 0.01)
    )
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    titlefont = ImageFont.truetype(fontpath, 26)
    font = ImageFont.truetype(fontpath, 23)
    destination = output / f"{prefix}_best_rear_to_front.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        stdin=subprocess.PIPE,
    )
    renderer = mujoco.Renderer(model, height=height, width=width)
    heading_previous = None
    center_previous = None
    distance_previous = None
    try:
        for frame in range(round(meta["seconds"] * fps)):
            t = frame / fps
            index = min(round(t / dt), len(poses) - 1)
            data.qpos[:] = poses[index]
            mujoco.mj_forward(model, data)
            direction = data.xmat[pelvis].reshape(3, 3) @ local_forward
            heading = np.degrees(np.arctan2(direction[1], direction[0]))
            if heading_previous is not None:
                heading = (
                    heading_previous
                    + ((heading - heading_previous + 180) % 360 - 180) * 0.12
                )
            heading_previous = heading
            # Bound all visible mesh/primitive geometry to keep the whole body in shot.
            rotations = data.geom_xmat[geoms].reshape(-1, 3, 3)
            bounds = model.geom_aabb[geoms]
            centers = data.geom_xpos[geoms] + np.einsum(
                "nij,nj->ni", rotations, bounds[:, :3]
            )
            half = np.einsum("nij,nj->ni", np.abs(rotations), bounds[:, 3:])
            low = (centers - half).min(0)
            high = (centers + half).max(0)
            center = (low + high) / 2
            radius = np.linalg.norm((high - low) / 2)
            distance = max(
                3.3, radius / np.sin(np.radians(model.vis.global_.fovy / 2)) * 1.18
            )
            if (
                center_previous is not None
                and episodes[index] == episodes[max(0, index - 4)]
            ):
                center = 0.8 * center_previous + 0.2 * center
                distance = max(distance, 0.9 * distance_previous + 0.1 * distance)
            center_previous = center
            distance_previous = distance
            camera.lookat[:] = center
            camera.distance = distance
            phase = np.clip((t - 6) / 20, 0, 1)
            orbit = 180 * (0.5 - 0.5 * np.cos(np.pi * phase))
            # MuJoCo's azimuth zero places the camera opposite its horizontal
            # viewing direction: heading is the rear view, heading+180 front.
            camera.azimuth = heading + orbit
            renderer.update_scene(data, camera=camera, scene_option=options)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, width - 12, 58), fill=(245, 245, 245))
            draw.text((25, 20), meta["title"], font=titlefont, fill="black")
            reset_count = sum(e["time_seconds"] <= t for e in meta["events"])
            label = f"{t:05.1f} s / {meta['seconds']:.0f} s   |   Episode {int(episodes[index])}   |   Resets: {reset_count}"
            draw.rectangle(
                (12, height - 48, width - 12, height - 12), fill=(245, 245, 245)
            )
            draw.text((25, height - 44), label, font=font, fill="black")
            recent = [e for e in meta["events"] if 0 <= t - e["time_seconds"] < 1]
            if recent:
                draw.text(
                    (25, 70),
                    "RESET: " + recent[-1]["reason"],
                    font=font,
                    fill=(180, 20, 20),
                )
            process.stdin.write(np.asarray(image).tobytes())
            if frame in (
                0,
                round(6 * fps),
                round(16 * fps),
                round(26 * fps),
                round(meta["seconds"] * fps) - 1,
            ):
                image.save(output / f"{prefix}_frame_{t:04.1f}s.png")
            if frame % 300 == 0:
                print(
                    f"Exp{exp}: rendered {frame}/{round(meta['seconds'] * fps)} frames",
                    flush=True,
                )
    finally:
        renderer.close()
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    meta.update(
        video=str(destination),
        fps=fps,
        resolution=[width, height],
        camera="rear hold 0-6s; eased side orbit 6-26s; front hold 26-30s; whole-body bounds",
        reset_labels=True,
    )
    (output / f"{prefix}_recording.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {destination}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiments", type=int, nargs="+", default=[8, 2, 6], choices=sorted(EXPERIMENTS)
    )
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--output", type=Path)
    p.add_argument("--render-only", action="store_true")
    args = p.parse_args()
    args.output = args.output or args.run / "videos_best_rear_to_front"
    args.output.mkdir(parents=True, exist_ok=True)
    for exp in args.experiments:
        if not args.render_only:
            rollout(exp, args.seconds, args.output, args.run)
        render(exp, args.output, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
