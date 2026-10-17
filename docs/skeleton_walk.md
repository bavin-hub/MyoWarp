# SkeletonTorque walking

The project scripts call LocoMuJoCo's ImitationFactory and PPOJax APIs. They use
the default `SkeletonTorque/walk.npz` motion reference, a trajectory-conditioned
goal, MimicReward, and the skeleton's 27 torque actuators. Training uses
MjxSkeletonTorque; evaluation uses native MuJoCo's SkeletonTorque. No training
code is changed inside the submodule. Weights & Biases is not required.

From the project root, install the isolated environment:

```bash
uv pip install --python .venv-locomujoco/bin/python -r requirements-locomujoco.txt
# On a CUDA 12 machine, add the matching JAX GPU packages:
uv pip install --python .venv-locomujoco/bin/python 'jax[cuda12]==0.9.2'
```

Train on a CUDA GPU:

```bash
.venv-locomujoco/bin/python scripts/train_skeleton_walk.py \
  --device gpu --backend warp --num-envs 2048 --total-timesteps 300000000 \
  --output logs/locomujoco/skeleton_walk
```

The first run downloads the motion into `logs/locomujoco/huggingface`. Training
compiles the upstream JAX function and saves `PPOJax_saved.pkl`, `config.yaml`,
`training_metrics.npz`, and `run.json` after completion. A run at the same
output path overwrites its previous results; use a new path to retain runs.
There are no intermediate checkpoints or resume support in this wrapper.

Evaluate with an interactive camera and actual forward physics:

```bash
.venv-locomujoco/bin/python scripts/eval_skeleton_walk.py \
  --checkpoint logs/locomujoco/skeleton_walk/PPOJax_saved.pkl --steps 3000
```

Add `--headless` to evaluate without a window. Evaluation uses deterministic
actions, retains the checkpoint's observation normalization statistics, and
resets when the upstream environment terminates an episode.

For a small CPU test of two PPO updates, saving, and headless replay:

```bash
.venv-locomujoco/bin/python scripts/train_skeleton_walk.py \
  --device cpu --backend mjx --smoke-test \
  --output logs/locomujoco/skeleton_walk_smoke
.venv-locomujoco/bin/python scripts/eval_skeleton_walk.py \
  --checkpoint logs/locomujoco/skeleton_walk_smoke/PPOJax_saved.pkl \
  --headless --steps 20
```

The smoke checkpoint is not a learned walking policy. CUDA/Warp training must
be validated on the machine providing the GPU. If the experimental upstream
Warp backend fails, use `--backend mjx --device gpu` for standard MJX training.
