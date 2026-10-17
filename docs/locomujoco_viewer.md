# LocoMuJoCo model viewer

The upstream repository is pinned as the `loco-mujoco` Git submodule. After
cloning this project, fetch it with `git submodule update --init loco-mujoco`.
LocoMuJoCo's model package is installed by its dependencies; motion datasets
are not needed for this viewer.

Use the separate environment to leave the existing training environment intact:

```bash
uv venv .venv-locomujoco --python python3.12
uv pip install --python .venv-locomujoco/bin/python -r requirements-locomujoco.txt
.venv-locomujoco/bin/python test_scripts/view_locomujoco.py --list
.venv-locomujoco/bin/python test_scripts/view_locomujoco.py --model SkeletonTorque
.venv-locomujoco/bin/python test_scripts/view_locomujoco.py --model SkeletonMuscle
```

The script creates the model using LocoMuJoCo's API and holds its initial pose
in MuJoCo's native viewer. Left drag rotates, right drag pans, and scrolling
zooms. It does not advance physics, load trajectories, or run a policy.

For a display-free check, add `--headless`. Use standard model names without
the `Mjx` prefix; the native viewer does not need a GPU. MyoSkeleton additionally
requires upstream's model installer and license acceptance before viewing.
