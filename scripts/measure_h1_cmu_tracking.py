"""Evaluate independent CPU H1/CMU episodes from seeded reference starts."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]

import numpy as np
import torch

from myowarp.tasks.h1_2_cmu_exp4.config import Exp4Cfg
from myowarp.tasks.h1_2_cmu_exp4.env import Exp4Env
from myowarp.tasks.h1_2_cmu_exp4.runner import Exp4Runner, PIPELINE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.seconds <= 120:
        parser.error('Positive episode count and duration in (0,120] required')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['infos']['pipeline'] != PIPELINE:
        parser.error('Expected paired H1/CMU Exp4 checkpoint')
    cfg = Exp4Cfg(**saved['infos']['config'])
    cfg.device = 'cpu'; cfg.num_envs = 1; cfg.episode_seconds = args.seconds
    torch.set_num_threads(1); torch.manual_seed(cfg.seed)
    env = Exp4Env(cfg, evaluation=True)
    runner = Exp4Runner(env); runner.load(args.checkpoint, evaluation=True)
    env.enable_fast_evaluation()
    max_start = (env.nframes - 1 - round(args.seconds / .01)) // 2
    starts = np.random.default_rng(args.seed).integers(0, max_start + 1, args.episodes)
    records = []; episodes = []; started = time.monotonic()
    try:
        with torch.inference_mode():
            for episode, start in enumerate(starts):
                obs = env.reset(int(start)); values = []; completed = False
                for _ in range(round(args.seconds / .01)):
                    action = runner.inference_actions(obs)
                    obs, _, done, extra = env.step(action['h1'], action['prosthesis'])
                    values.append([float(v[0]) for v in env.metrics.values()])
                    if bool(done[0]):
                        completed = bool(extra['time_outs'][0]); break
                values = np.asarray(values)
                records.append(values)
                episodes.append(dict(start_frame_50hz=int(start), duration_seconds=len(values)*.01,
                                     completed=completed, mean=values.mean(0).tolist(), final=values[-1].tolist()))
                print(f'Episode {episode+1}/{args.episodes}: {len(values)*.01:.2f}s, completed={completed}', flush=True)
    finally:
        env.close()
    names = list(env.metrics)
    mean = np.concatenate(records).mean(0)
    final = np.stack([a[-1] for a in records]).mean(0)
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_iteration=int(saved['iter']),
                  backend='native CPU', seed=args.seed, num_episodes=args.episodes,
                  horizon_seconds=args.seconds, completed_full_horizon=sum(e['completed'] for e in episodes),
                  mean_episode_seconds=float(np.mean([e['duration_seconds'] for e in episodes])),
                  metrics_sample_weighted_mean=dict(zip(names, mean.tolist())),
                  metrics_mean_final=dict(zip(names, final.tolist())), episodes=episodes,
                  wall_seconds=time.monotonic()-started,
                  measurement='Real-state errors before first termination; no samples after reset. Different motion/model from musculoskeletal experiments.')
    output = args.output or args.checkpoint.parent / f'eval_cpu_{args.episodes}x{args.seconds:g}s.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    lengths = np.array([len(v) for v in records])
    np.savez_compressed(output.with_suffix('.npz'), values=np.concatenate(records), lengths=lengths,
                        metric_names=names, start_frames_50hz=starts, dt=.01)
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2),flush=True)


if __name__ == '__main__':
    main()
