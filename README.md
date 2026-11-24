# MyoWarp

Musculoskeletal and H1-2 locomotion with a CMU knee/ankle prosthesis. This branch
contains eight muscle/prosthesis learning experiments, a standalone intact H1-2
walking tracker, and independent H1/CMU policy fine-tuning. Model assets, reference
adaptation, policies, low-level control, training and evaluation run locally.

## Documentation

- [Commands](docs/commands.md): train each experiment, run all eight sequentially,
  evaluate best checkpoints, measure tracking and record videos.
- [Eight muscle/prosthesis experiments](docs/musculo_prosthesis_8_experiments.md):
  policy initialization, observations, rewards and optimization differences.
- [Musculoskeletal pipeline](docs/musculoskeletal_prosthesis_rl.md) and
  [anatomy variants](docs/prosthesis_anatomy_variants.md).
- [MuscleMimic assets and motion clips](docs/musclemimic.md).
- [Intact H1-2 tracking](docs/h1_2_tracking.md).
- [H1-2 + CMU Exp4](docs/h1_2_cmu_exp4.md): independent policy learning and CPU/GPU evaluation.

## Environments

Use two separate Python environments because the simulation versions differ.
Python 3.12 is recommended. From the repository root, using `uv`:

```bash
# Intact H1-2 and H1 + CMU: MuJoCo/Warp 3.5.
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# MuscleMimic and muscle/prosthesis experiments: MuJoCo/Warp 3.11.
uv venv --python 3.12 .venv-musclemimic
uv pip install --python .venv-musclemimic/bin/python -r requirements-prosthesis-rl.txt
```

Run scripts from the repository root with the appropriate interpreter. The
local `rsl_rl/` source is installed by the requirements files. Training requires
CUDA. Muscle/prosthesis evaluation also uses CUDA. H1 + CMU supports native CPU
evaluation, including both actors and the 1 kHz controller, without CUDA.
Video recording requires `ffmpeg` and offscreen EGL graphics support.

## Models and data

Retained robot assets are under `assets/robots/`:

- `myofullbody/`: full biological model used by the pretrained muscle policy.
- `myofullbody_cmu_leg/`: constrained biological-leg bypass model.
- `myofullbody_cmu_leg_no_lower_leg/`: removed biological lower leg plus CMU.
- `unitree_h1_2/`: intact 27-action H1-2.
- `h1_with_cmu_leg/`: H1-2 with the CMU leg.
- `cmu_leg/`: source prosthesis meshes and linkage/SEA model.

Combined models reference the base muscle, H1 and CMU asset folders; keep those
folders together. Preserve bundled licenses and provenance notices.

Downloaded pretrained policies and source muscle motions are **not committed**.
Follow [the MuscleMimic setup](docs/musclemimic.md); gated Hugging Face motion data
requires approved access. The Vyvas checkpoint is also a local artifact.
Regenerate the stitched muscle clip and adapted reference as described in the
docs. The H1 120-second reference and its adapted CMU reference are under
`assets/reference_clips/`. The optional original H1 CSV, retained locally under
`assets/h1_amass/motions/walk1_subject1.csv`, is ignored by Git; that folder name
is historical, not an active AMASS-policy integration.

`logs/`, learned checkpoints, virtual environments and caches are ignored by
Git. A fresh clone does not include trained policies or recorded videos. To
reproduce existing evaluations, supply the local pretrained artifacts and best
checkpoints at the paths in [commands.md](docs/commands.md), or adjust those paths
to your copies. H1 + CMU training initializes from an intact H1 actor and critic;
train intact H1 first if that checkpoint is unavailable.

## Quick commands

```bash
# Run all eight muscle/prosthesis experiments independently and sequentially.
bash scripts/run_eight_prosthesis_experiments.sh

# H1 + CMU independent policy training.
.venv/bin/python scripts/h1_2_cmu_exp4.py train \
  --num-envs 1024 --iterations 2500 --episode-seconds 30

# Evaluate a paired H1/CMU best checkpoint using CPU simulation and control.
.venv/bin/python scripts/h1_2_cmu_exp4.py eval \
  --checkpoint logs/h1_2_cmu_exp4/RUN/model_best_prosthesis.pt \
  --backend cpu --num-envs 1 --viewer --episode-seconds 120 --steps 12000
```

Replace `RUN` with the actual run folder. Best snapshots selected by muscle/H1
return and prosthesis return each contain both policies from the same instant;
do not combine policies from different snapshots. Training-return selection is
not a held-out tracking score.

The musculoskeletal reference is 30 seconds; its training episodes are capped
at 10 seconds. The H1 reference is 120 seconds; H1/CMU training defaults to
30-second episodes. Episode limits and reference lengths are distinct.

## Development checks

Install `pytest` in a development environment, then run tests with its matching
simulation version:

```bash
.venv/bin/python -m pytest tests/test_h1_2_tracking.py tests/test_h1_2_cmu_exp4.py -q
.venv-musclemimic/bin/python -m pytest tests/test_musclemimic*.py tests/test_prosthesis*.py tests/test_removed_lower_leg.py tests/test_sea_controller.py -q
```

Some tests need local model/motion/checkpoint artifacts; GPU integration tests
also require CUDA. Controller and reward parity fixtures are documented in
[tests/fixtures/README.md](tests/fixtures/README.md).

The folders named `h1_with_osl_tracking` and `h1_with_osl_velocity` under
`myowarp/tasks/` remain as shared implementation dependencies of the H1 tasks.
The old OSL assets, entry points, dual-policy tasks, MJLab registrations and
pretrained AMASS policy port are not supported workflows on this branch.
