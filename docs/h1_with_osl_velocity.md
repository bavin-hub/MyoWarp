# H1 with OSL: standalone velocity tracking

This task runs directly on MuJoCo-Warp and the repository's `rsl_rl` checkout.
The training and evaluation entry points do not import MJLab. The legacy
`myowarp/mjlab_tasks` implementation is retained as a parity reference only.

## Task contract

- 26 position-control actions; the four socket-compliance joints remain passive.
- Physics timestep: 0.005 s; action decimation: 4; control timestep: 0.02 s.
- Episode duration: 20 s (1000 policy steps).
- Actor observation: 89 values.
- Asymmetric critic observation: 104 values.
- Flat terrain, heading-based velocity commands, staged command ranges, pushes,
  foot-friction randomization, torso COM randomization, encoder bias, and actor
  observation noise.

The actor receives base angular velocity, projected gravity, the three-value
velocity command, two-value gait phase, 26 joint positions, 26 joint velocities,
and the previous 26 raw actions. The critic additionally receives base linear
velocity, two foot heights, two air times, two contact flags, and six transformed
foot-contact force values.

## Reward port

All reward values are multiplied by the 0.02 s control timestep before they are
summed. The terms and weights are:

| Term | Weight |
|---|---:|
| planar/vertical linear velocity tracking | 1.0 |
| yaw/roll/pitch angular velocity tracking | 1.0 |
| speed-dependent default-pose tracking | 1.0 |
| torso roll/pitch angular velocity | -0.05 |
| whole-body angular momentum | -0.025 |
| action-rate L2 | -0.05 |
| foot clearance | -1.0 |
| foot slip | -0.25 |
| landing impact | -0.001 |
| torso orientation | -1.0 |
| non-timeout termination | -200.0 |
| joint acceleration L2 | -2.5e-7 |
| soft joint-position limits | -10.0 |
| alternating foot gait | 0.5 |
| non-default pose while standing | -1.0 |
| self-collision substeps | -1.0 |

## Train and evaluate

Install the local RSL-RL checkout and this package, then launch training:

```bash
uv pip install -e ./rsl_rl
uv pip install -e .
python scripts/train.py h1_with_osl_velocity --device cuda:0 --num-envs 4096
```

Runs are stored below `logs/rsl_rl/h1_with_osl_velocity_standalone/`. Each run
contains RSL-RL checkpoints, TensorBoard events, JSON environment/agent configs,
the Git diff, and an exported ONNX actor.

```bash
python scripts/eval.py h1_with_osl_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer
```
