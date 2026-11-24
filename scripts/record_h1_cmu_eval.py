"""Record the latest paired H1/CMU policy using CPU physics and a camera orbit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'rsl_rl'),str(ROOT/'scripts')]

import numpy as np
import torch
from myowarp.tasks.h1_2_cmu_exp4.config import Exp4Cfg
from myowarp.tasks.h1_2_cmu_exp4.env import Exp4Env
from myowarp.tasks.h1_2_cmu_exp4.runner import Exp4Runner,PIPELINE
from record_prosthesis_evals import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.checkpoint is None:
        choices=list((ROOT/'logs/h1_2_cmu_exp4').glob('*/model_best_prosthesis.pt'))
        if not choices:p.error('No best H1/CMU checkpoint found')
        args.checkpoint=max(choices,key=lambda path:path.stat().st_mtime)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved['infos']['pipeline']!=PIPELINE:p.error('Expected paired H1/CMU Exp4 checkpoint')
    cfg=Exp4Cfg(**saved['infos']['config']);cfg.device='cpu';cfg.num_envs=1;cfg.episode_seconds=120
    if not 0<args.seconds<=120:p.error('Duration must be in (0,120]')
    output=args.output or args.checkpoint.parent/'videos_best_rear_to_front'
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(cfg.seed)
    env=Exp4Env(cfg,evaluation=True);runner=Exp4Runner(env);runner.load(args.checkpoint,evaluation=True)
    env.enable_fast_evaluation();env.reset(0)
    states=[env.data.qpos[0].numpy().copy()];episodes=[1];events=[];episode=1
    original_reset=env._reset_ids;terminal={}
    def capture_reset(ids,start_frame=None):
        if len(ids):terminal['qpos']=env.data.qpos[0].numpy().copy()
        return original_reset(ids,start_frame)
    env._reset_ids=capture_reset
    started=time.monotonic()
    try:
        obs=env.get_observations()
        with torch.inference_mode():
            for step in range(round(args.seconds/.01)):
                terminal.clear()
                action=runner.inference_actions(obs)
                obs,_,done,extra=env.step(action['h1'],action['prosthesis'])
                states.append(terminal.get('qpos',env.data.qpos[0].numpy().copy()));episodes.append(episode)
                if bool(done[0]):
                    events.append(dict(time_seconds=(step+1)*.01,episode=episode,reason='episode timeout / reference end' if bool(extra['time_outs'][0]) else 'fall / tracking termination'))
                    episode+=1
                if (step+1)%500==0:print(f'H1+CMU: {(step+1)*.01:.1f}/{args.seconds}s, resets={len(events)}',flush=True)
    finally:env.close()
    np.savez_compressed(output/'h1_cmu_rollout.npz',qpos=np.asarray(states),episode=episodes,dt=.01)
    meta=dict(experiment='H1-2 + CMU Exp4',title='H1-2 + CMU: Exp4, best checkpoint',checkpoint=str(args.checkpoint.resolve()),checkpoint_iteration=int(saved['iter']),model_path=str(ROOT/'assets/robots/h1_with_cmu_leg/h1_with_cmu_leg.xml'),seconds=args.seconds,start_frame=0,events=events,episode_limit_seconds=120,backend='native CPU physics, CPU actors and 1kHz SEA controller',wall_seconds=time.monotonic()-started)
    (output/'h1_cmu_recording.json').write_text(json.dumps(meta,indent=2)+'\n')
    render('H1+CMU',output,30,1280,720,model_override=env.model,stem='h1_cmu')

if __name__=='__main__':main()
