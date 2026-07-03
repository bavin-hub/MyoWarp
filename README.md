# myowarp

Standalone MuJoCo Warp port of the MyoAssist leg imitation/exo training setup.

This repo is intentionally separate from `myoassist`. Training uses MuJoCo Warp
for batched GPU simulation, while policy evaluation and visualization can use
plain CPU MuJoCo for faster single-policy debugging.

## What Is Ported

- Tutorial leg imitation config:
  `configs/imitation_tutorial_22_separated_net_partial_obs.json`
- MuJoCo model:
  `models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml`
- Reference gait data:
  `reference_data/short_reference_gait.npz`
- Batched Warp env:
  `myowarp/envs/myoassist_leg_warp.py`
- Single-env CPU MuJoCo eval env:
  `myowarp/envs/myoassist_leg_cpu.py`
- Custom PyTorch PPO trainer:
  `myowarp/train/train_ppo.py`
- CPU/Warp metric eval:
  `myowarp/train/eval_policy.py`
- CPU/Warp GUI viewer:
  `myowarp/train/view_policy.py`
- Separated human/exo actor network structure from the MyoAssist config.

## Install

If the `myo_warp` uv environment already exists and is activated:

```bash
cd /home/bavin/my_ws/legged_systems/myowarp
python -m pip install -e .
python -m pip install -r requirements.txt
```

If using uv from scratch:

```bash
cd /home/bavin/my_ws/legged_systems/myowarp
uv venv myo_warp --python 3.11
source myo_warp/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

## Smoke Checks

Check that the XML loads with CPU MuJoCo:

```bash
python scripts/check_model.py --model models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml
```

Run a short random rollout through MuJoCo Warp:

```bash
python -m myowarp.train.random_rollout \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --num-envs 64 \
  --steps 10
```

## Train

Training uses the Warp/GPU environment.

```bash
cd /home/bavin/my_ws/legged_systems/myowarp
python -m myowarp.train.train_ppo \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json
```

Useful overrides:

```bash
python -m myowarp.train.train_ppo \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --num-envs 128 \
  --n-steps 64 \
  --batch-size 8192 \
  --iterations 100 \
  --device cuda
```

`total_timesteps` is total simulation samples, not PPO update count:

```text
updates = total_timesteps / (num_envs * n_steps)
```

For example, `1024 envs * 64 n_steps = 65,536 samples/update`, so
`30,000,000 timesteps` is about `458` PPO updates.

Training prints one line per PPO update:

```text
update=... step=... reward=... ep_return=... ep_len=... done_traj=... done_height=... kl=...
```

Checkpoints and logs are written to:

```text
results/train_session_YYYYMMDD-HHMMSS/
results/train_session_YYYYMMDD-HHMMSS/trained_models/step_XXXX.pt
results/train_session_YYYYMMDD-HHMMSS/train_log.jsonl
```

## Metric Eval

By default, eval uses CPU MuJoCo, not Warp. This is usually faster and simpler
for checking one policy.

```bash
python -m myowarp.train.eval_policy \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --checkpoint results/train_session_YYYYMMDD-HHMMSS/trained_models/step_XXXX.pt \
  --steps 1000 \
  --device cpu
```

Eval prints episode return, episode length, and termination cause:

```text
episode_done step=... return=... len=... cause=trajectory=1
termination_counts={'trajectory': ..., 'height': ..., 'truncated': ..., 'out_of_reference': ...}
```

To force Warp eval:

```bash
python -m myowarp.train.eval_policy \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --checkpoint results/train_session_YYYYMMDD-HHMMSS/trained_models/step_XXXX.pt \
  --steps 1000 \
  --backend warp \
  --num-envs 64 \
  --device cuda
```

## GUI Eval

By default, the viewer also uses CPU MuJoCo.

```bash
python -m myowarp.train.view_policy \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --checkpoint results/train_session_YYYYMMDD-HHMMSS/trained_models/step_XXXX.pt \
  --device cpu
```

Run for a fixed number of control steps:

```bash
python -m myowarp.train.view_policy \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --checkpoint results/train_session_YYYYMMDD-HHMMSS/trained_models/step_XXXX.pt \
  --steps 1000 \
  --device cpu
```

Use `--speed 0.5` or `--speed 2.0` to slow down or speed up playback.

## Important Differences From MyoAssist

- Training uses a custom PyTorch PPO loop, not Stable-Baselines3 PPO.
- Checkpoints are `.pt`, not Stable-Baselines3 `.zip`.
- Training simulation runs through MuJoCo Warp; CPU eval/view runs through
  regular MuJoCo.
- The MyoSuite `MujocoEnv` wrapper is not used here.
- The active model is intact musculoskeletal legs plus two ankle exo actuators
  (`Exo_R`, `Exo_L`), not an amputee prosthetic replacement model.
- Exact CPU MuJoCo and MuJoCo Warp dynamics may differ slightly.

## GitHub / Repo Hygiene

Do not commit large generated folders:

```text
myo_warp/
results/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.warp/
```

Commit the source and small required assets:

```text
myowarp/
configs/
models/
reference_data/
scripts/
README.md
requirements.txt
pyproject.toml
uv.lock
```

Use Git LFS if you intentionally want to version large trained `.pt`
checkpoints.
