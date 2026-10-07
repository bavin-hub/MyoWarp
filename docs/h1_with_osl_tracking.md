# H1 with OSL: standalone motion tracking

This task runs entirely through MuJoCo-Warp and the repository's local RSL-RL
checkout. It does not import or call MJLab during training or evaluation.

## Reference and task contract

- Reference: `assets/reference_clips/short_reference_gait_h1_with_osl.npz`.
- 4,490 frames at 50 Hz (89.8 seconds).
- 30 simulated robot joints, including four passive socket-compliance joints.
- 26 position-control policy actions.
- 14 tracked bodies with `torso_link` as the motion anchor.
- Physics timestep: 0.005 s; action decimation: 4; control timestep: 0.02 s.
- Episode duration: 10 seconds (500 policy steps).
- Actor observation: 161 values.
- Asymmetric critic observation: 287 values.

The actor observes the 60-value reference joint command, anchor position and
6D orientation errors, base linear/angular velocity, all 30 joint positions and
velocities, and 26 previous actions. The critic additionally observes the
positions and 6D orientations of all 14 tracked robot bodies relative to the
current torso anchor. Encoder bias and uniform observation corruption apply only
where they do in the source actor contract.

## Rewards

All terms are multiplied by the 0.02 s control timestep.

| Term | Weight | Kernel standard deviation |
|---|---:|---:|
| global anchor position | 0.5 | 0.3 |
| global anchor orientation | 0.5 | 0.4 |
| anchor-aligned body positions | 1.0 | 0.3 |
| anchor-aligned body orientations | 1.0 | 0.4 |
| global body linear velocities | 1.0 | 1.0 |
| global body angular velocities | 1.0 | 3.14 |
| action-rate L2 | -0.1 | — |
| soft joint-position limits | -10.0 | — |
| self-collision substeps | -10.0 | — |

## Resets, sampling, and termination

Training starts each environment from an adaptively sampled reference frame.
Reference root pose, root velocity, and joint positions are perturbed using the
same ranges as the source task. Failed-frame bins are exponentially accumulated
and sampled more frequently; these statistics are saved in RSL-RL checkpoints.

Episodes terminate when the torso anchor height differs from the reference by
more than 0.25 m, torso tilt differs beyond the 0.8 projected-gravity threshold,
or a tracked ankle/wrist height differs by more than 0.25 m. Episodes also time
out after 10 seconds.

## Train and evaluate

```bash
python scripts/train.py h1_with_osl_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000
```

```bash
python scripts/eval.py h1_with_osl_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer
```

Evaluation reports success rate, MPKPE, root-relative MPKPE, joint-velocity
error, and end-effector position/orientation error. Motion ONNX exports include
the actor and the reference arrays needed by downstream playback.

Headless evaluation follows the source evaluation protocol by default: 1,024
environments each run one episode from the start of the motion, actor noise and
startup/reset randomization remain enabled, and interval pushes are disabled.
Use `--num-envs` to change the batch size, `--steps` to cap the rollout, or
`--deterministic` for clean playback and debugging.
