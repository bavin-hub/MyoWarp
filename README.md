# MyoWarp

Standalone MuJoCo-Warp reinforcement-learning tasks for assisted humanoid
locomotion. Simulation, rewards, training, checkpointing, and evaluation live in
this repository; RSL-RL is used from the local `rsl_rl` checkout.

## Implemented tasks

Task 1, `h1_with_osl_velocity`:

- H1 with a passive-compliance OSL right leg and 26 policy actions.
- MJLab velocity-task reward semantics and weights, ported locally.
- 89-value actor and 104-value asymmetric critic observations.
- Heading-based velocity commands and staged command curriculum.
- Friction, torso-COM, encoder-bias, push, and observation randomization.
- Batched MuJoCo-Warp simulation and native RSL-RL 5.x PPO.
- Checkpoint/resume (including task curriculum state), TensorBoard logging, ONNX
  export, headless evaluation, and native MuJoCo viewing.

Task 2, `h1_with_osl_tracking`:

- Tracks the 4,490-frame, 50 Hz retargeted H1-with-OSL motion reference.
- 161-value actor and 287-value asymmetric critic observations.
- Nine reference-motion reward terms with the MJLab weights and kernels.
- Torso-anchor and ankle/wrist reference-error terminations.
- Adaptive failure-weighted frame sampling with checkpoint persistence.
- Randomized reference-state resets, pushes, friction, torso COM, encoder bias,
  and observation corruption.
- Motion metrics and ONNX exports containing both actor and reference data.

Task 3, `h1_with_osl_dual_full_velocity`:

- Two independent full-observation actors trained by one joint PPO objective.
- A 24-action H1 actor and two-action OSL knee/ankle actor.
- One shared asymmetric critic and combined log-probability, entropy, and KL.
- Task #1 rewards, curriculum, contacts, terminations, and randomization unchanged.
- Additive local RSL-RL `DualMLPModel`; existing single-actor tasks are unaffected.

Task 4, `h1_with_osl_dual_partial_velocity`:

- The H1 actor retains its full 89-value observation and 24 actions.
- The OSL actor receives only knee/ankle positions and velocities and produces
  the two OSL actions.
- The shared critic and Task #1 velocity-training contract remain unchanged.

Task 5, `h1_with_osl_dual_full_tracking`:

- Independent 24-action H1 and two-action OSL actors both receive the full
  161-value motion-tracking observation.
- A shared 287-value privileged critic trains with both actors through joint PPO.
- Task #2 motion rewards, sampler, terminations, metrics, and export are unchanged.

Task 6, `h1_with_osl_dual_partial_tracking`:

- The H1 actor retains its full 161-value tracking observation and 24 actions.
- The OSL actor sees only knee/ankle positions and velocities and produces two actions.
- The shared critic and Task #2 motion-tracking contract remain unchanged.

The train/eval pipeline does not import or call MJLab. The existing
`myowarp/mjlab_tasks` package is retained only as the parity reference while the
remaining planned tasks are ported.

See [the task specification](docs/h1_with_osl_velocity.md) for the complete
observation and reward contract.
The motion contract is in
[docs/h1_with_osl_tracking.md](docs/h1_with_osl_tracking.md).
The dual-policy contract is in
[docs/h1_with_osl_dual_full_velocity.md](docs/h1_with_osl_dual_full_velocity.md).
The partial-observation contract is in
[docs/h1_with_osl_dual_partial_velocity.md](docs/h1_with_osl_dual_partial_velocity.md).
The dual-policy motion contract is in
[docs/h1_with_osl_dual_full_tracking.md](docs/h1_with_osl_dual_full_tracking.md).
The partial-observation motion contract is in
[docs/h1_with_osl_dual_partial_tracking.md](docs/h1_with_osl_dual_partial_tracking.md).

## Install

From the repository root, with the project environment active:

```bash
uv pip install -e ./rsl_rl
uv pip install -e .
```

The scripts also prepend the local RSL-RL source directory, ensuring that the
checked-out version is selected when they are launched from this repository.

## Train

```bash
python scripts/train.py h1_with_osl_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001
```

Motion tracking:

```bash
python scripts/train.py h1_with_osl_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000
```

Dual-policy full-observation velocity tracking:

```bash
python scripts/train.py h1_with_osl_dual_full_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001
```

Dual-policy partial-observation velocity tracking:

```bash
python scripts/train.py h1_with_osl_dual_partial_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001
```

Dual-policy full-observation motion tracking:

```bash
python scripts/train.py h1_with_osl_dual_full_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000
```

Dual-policy partial-observation motion tracking:

```bash
python scripts/train.py h1_with_osl_dual_partial_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000
```

Training output is written beneath:

```text
logs/rsl_rl/h1_with_osl_velocity_standalone/<timestamp>[_run-name]/
```

Motion runs use the corresponding `h1_with_osl_tracking_standalone` directory.
Dual-policy runs use `h1_with_osl_dual_full_velocity_standalone`.
Partial-observation runs use `h1_with_osl_dual_partial_velocity_standalone`.
Dual motion runs use `h1_with_osl_dual_full_tracking_standalone`.
Partial dual motion runs use `h1_with_osl_dual_partial_tracking_standalone`.

Resume from a checkpoint with `--resume path/to/model_ITERATION.pt`. The old task
alias `MyoWarp-H1-With-OSL-Flat` is also accepted for convenience.

## Evaluate

```bash
python scripts/eval.py h1_with_osl_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer
```

Motion-tracking evaluation starts from a reference frame instead of taking a
velocity command:

```bash
python scripts/eval.py h1_with_osl_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer
```

Without `--deterministic`, motion evaluation matches the source evaluator: it
keeps reset/startup/actor randomization, disables pushes, and runs one episode
per environment (1,024 environments by default when no viewer is requested).

For a checkpoint-free CPU smoke test:

```bash
python scripts/eval.py h1_with_osl_velocity \
  --agent zero \
  --device cpu \
  --steps 10
```

More command examples are collected in [docs/cmds.md](docs/cmds.md).

## Test

```bash
python -m pytest -q \
  tests/test_h1_with_osl_velocity.py \
  tests/test_h1_with_osl_tracking.py \
  tests/test_h1_with_osl_dual_full_velocity.py \
  tests/test_h1_with_osl_dual_partial_velocity.py \
  tests/test_h1_with_osl_dual_full_tracking.py \
  tests/test_h1_with_osl_dual_partial_tracking.py
```

The tests validate the model/action contract, actor and critic observation
dimensions, domain randomization, reward-term coverage, and a finite CPU rollout.
