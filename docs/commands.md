# Training, evaluation and recording commands

Run these commands in Bash from the repository root. The commands below do not
require activating a virtual environment. Musculoskeletal runs use
`.venv-musclemimic`; H1 runs use `.venv`.

Experiment definitions and reward/observation differences are explained in
[musculo_prosthesis_8_experiments.md](musculo_prosthesis_8_experiments.md).

## Shell setup

Run this block once in each terminal:

```bash
MUSCLE_PY="$PWD/.venv-musclemimic/bin/python"
H1_PY="$PWD/.venv/bin/python"
DEFAULT_MUSCLE="$PWD/assets/musclemimic/checkpoints/mm-10m-2"
VYVAS_MUSCLE="$PWD/vyvas_pi_h/model_1499.pt"

# Existing successful runs: use these for evaluation and recording.
MUSCLE_RUN="$PWD/logs/musculoskeletal_prosthesis/eight_experiments_20261021_231020"
H1_RUN="$PWD/logs/h1_2_cmu_exp4/2026-10-23_23-31-52"
H1_PRETRAINED="$PWD/logs/rsl_rl/h1_2_tracking_standalone/2026-10-23_22-21-41/model_best.pt"

# Fresh output locations for new training. Do not create these folders manually.
MUSCLE_TRAIN_RUN="$PWD/logs/musculoskeletal_prosthesis/eight_individual_$(date +%Y%m%d_%H%M%S)"
H1_TRAIN_RUN="$PWD/logs/h1_2_cmu_exp4/$(date +%Y-%m-%d_%H-%M-%S)"
MUSCLE_COMMON=(--device cuda:0 --num-envs 1024 --save-interval 100)
```

Training requires CUDA. Musculoskeletal evaluation also requires CUDA. H1 + CMU
evaluation defaults to native CPU physics, CPU policies and the CPU 1 kHz SEA
controller; only its viewer uses graphics hardware.

## Musculoskeletal reference preparation

The 30-second stitched walking clip and adapted removed-leg reference already
exist. Run preparation if the reference needs rebuilding:

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py prepare \
  --anatomy removed --right-leg-obs virtual \
  --human-checkpoint "$DEFAULT_MUSCLE"
```

## Train the eight experiments individually

Each run starts independently; none loads another experiment's learned policy.
All use the removed lower-leg model, virtual observations and 1,024 environments.
The iteration limits match the retained eight-experiment batch: **2,500 for
Exp4/Exp8, 800 for the others**.

| Experiment | Setup | Output subfolder |
| --- | --- | --- |
| 1 | Frozen default muscle policy; train prosthesis | `01_default_frozen` |
| 2 | Frozen Vyvas muscle policy; train prosthesis | `02_vyvas_frozen` |
| 3 | Joint muscle/prosthesis PPO | `03_default_coopt` |
| 4 | Separate actors, critics, rewards and PPO updates | `04_independent` |
| 5 | Separate PPO learners, shared reward | `05_shared` |
| 6 | Exp4 plus prosthesis reference inputs | `06_muscle_reference` |
| 7 | Historical hip-RPY observation variant | `07_hip_rpy` |
| 8 | Exp4 with human-space prosthesis tracking reward | `08_human_space` |

Exp7's final actor observations match Exp4 after hip RPY became common to the
baseline. The historical experiment flag remains available.

### Exp1: frozen default muscle policy

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 800 --seed 42 --anatomy removed --right-leg-obs virtual \
  --human-checkpoint "$DEFAULT_MUSCLE" \
  --output "$MUSCLE_TRAIN_RUN/01_default_frozen"
```

### Exp2: frozen Vyvas muscle policy

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 800 --seed 42 --anatomy removed --right-leg-obs virtual \
  --human-checkpoint "$VYVAS_MUSCLE" \
  --output "$MUSCLE_TRAIN_RUN/02_vyvas_frozen"
```

### Exp3: joint muscle/prosthesis PPO

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 800 --seed 42 --anatomy removed --right-leg-obs virtual \
  --human-checkpoint "$DEFAULT_MUSCLE" \
  --train-muscle --muscle-learning-rate 1e-5 \
  --output "$MUSCLE_TRAIN_RUN/03_default_coopt"
```

### Exp4: independent learners and rewards

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 2500 --human-checkpoint "$DEFAULT_MUSCLE" \
  --reward-mode separate \
  --output "$MUSCLE_TRAIN_RUN/04_independent"
```

### Exp5: independent learners, shared reward

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 800 --human-checkpoint "$DEFAULT_MUSCLE" \
  --reward-mode shared \
  --output "$MUSCLE_TRAIN_RUN/05_shared"
```

### Exp6: reference-conditioned prosthesis actor

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 800 --human-checkpoint "$DEFAULT_MUSCLE" \
  --reward-mode separate --prosthesis-observation muscle_reference \
  --output "$MUSCLE_TRAIN_RUN/06_muscle_reference"
```

### Exp7: hip-RPY variant

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 800 --human-checkpoint "$DEFAULT_MUSCLE" \
  --reward-mode separate --prosthesis-observation hip_rpy \
  --output "$MUSCLE_TRAIN_RUN/07_hip_rpy"
```

### Exp8: human-space prosthesis reward

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py train "${MUSCLE_COMMON[@]}" \
  --iterations 2500 --human-checkpoint "$DEFAULT_MUSCLE" \
  --reward-mode separate --prosthesis-reward human_space \
  --output "$MUSCLE_TRAIN_RUN/08_human_space"
```

## Train all eight sequentially with one shell script

Use the existing [eight-experiment launcher](../scripts/run_eight_prosthesis_experiments.sh)
instead of the individual commands. It prepares the reference, runs Exp1 through
Exp8 sequentially, and stops on failure. Its output folder must not already exist.

```bash
ALL_EIGHT_RUN="$PWD/logs/musculoskeletal_prosthesis/eight_experiments_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="$ALL_EIGHT_RUN" bash scripts/run_eight_prosthesis_experiments.sh
```

After new training, point `MUSCLE_RUN` to the new output before evaluating it:

```bash
MUSCLE_RUN="$ALL_EIGHT_RUN"
```

For individually trained experiments, use `MUSCLE_RUN="$MUSCLE_TRAIN_RUN"` instead.

## Evaluate each muscle/prosthesis best checkpoint

Run one viewer command at a time. At 100 Hz, `--steps 3000` requests 30 seconds
of total simulation, including automatic resets. Each episode is capped at
10 seconds or ends earlier on failure; this is not a guaranteed uninterrupted
30-second walk.

Exp1–3 use `model_best.pt`. Exp4–8 use `model_best_prosthesis.pt`, which contains
both policies from the same training instant. `model_best_muscle.pt` is an
alternative paired snapshot selected by muscle return; do not mix policies
from separately selected snapshots.

### Exp1

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py eval \
  --checkpoint "$MUSCLE_RUN/01_default_frozen/model_best.pt" \
  --anatomy removed --right-leg-obs virtual --human-checkpoint "$DEFAULT_MUSCLE" \
  --num-envs 1 --viewer --steps 3000 \
  --output "$MUSCLE_RUN/01_default_frozen/eval_viewer"
```

### Exp2

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py eval \
  --checkpoint "$MUSCLE_RUN/02_vyvas_frozen/model_best.pt" \
  --anatomy removed --right-leg-obs virtual --human-checkpoint "$VYVAS_MUSCLE" \
  --num-envs 1 --viewer --steps 3000 \
  --output "$MUSCLE_RUN/02_vyvas_frozen/eval_viewer"
```

### Exp3

```bash
"$MUSCLE_PY" scripts/prosthesis_rl.py eval \
  --checkpoint "$MUSCLE_RUN/03_default_coopt/model_best.pt" \
  --anatomy removed --right-leg-obs virtual --human-checkpoint "$DEFAULT_MUSCLE" \
  --train-muscle --muscle-learning-rate 1e-5 \
  --num-envs 1 --viewer --steps 3000 \
  --output "$MUSCLE_RUN/03_default_coopt/eval_viewer"
```

### Exp4

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py eval \
  --checkpoint "$MUSCLE_RUN/04_independent/model_best_prosthesis.pt" \
  --num-envs 1 --viewer --steps 3000
```

### Exp5

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py eval \
  --checkpoint "$MUSCLE_RUN/05_shared/model_best_prosthesis.pt" \
  --num-envs 1 --viewer --steps 3000
```

### Exp6

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py eval \
  --checkpoint "$MUSCLE_RUN/06_muscle_reference/model_best_prosthesis.pt" \
  --num-envs 1 --viewer --steps 3000
```

### Exp7

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py eval \
  --checkpoint "$MUSCLE_RUN/07_hip_rpy/model_best_prosthesis.pt" \
  --num-envs 1 --viewer --steps 3000
```

### Exp8

```bash
"$MUSCLE_PY" scripts/prosthesis_split_rl.py eval \
  --checkpoint "$MUSCLE_RUN/08_human_space/model_best_prosthesis.pt" \
  --num-envs 1 --viewer --steps 3000
```

## Measure all eight best checkpoints: 32 starts each

This runs headlessly, measures only the first episode per environment, and uses
the same 32 sampled starts across experiments with a 10-second horizon.

```bash
"$MUSCLE_PY" scripts/measure_prosthesis_tracking.py "$MUSCLE_RUN" \
  --checkpoint best --num-envs 32 --steps 1000 \
  --output "$MUSCLE_RUN/tracking_evaluation_best"
```

Results include `summary.json`, individual experiment reports/raw metrics and
`paired_window_means.json`. No deleted periodic checkpoint is required.

## Record muscle/prosthesis policies

The recorder accepts all eight experiment IDs and automatically selects the
correct best checkpoint. Record one policy, for example Exp8:

```bash
"$MUSCLE_PY" scripts/record_prosthesis_evals.py \
  --run "$MUSCLE_RUN" --experiments 8 --seconds 30
```

Replace `8` with any ID from `1` through `8`. Record the selected three, or all
eight sequentially:

```bash
"$MUSCLE_PY" scripts/record_prosthesis_evals.py \
  --run "$MUSCLE_RUN" --experiments 8 2 6 --seconds 30

"$MUSCLE_PY" scripts/record_prosthesis_evals.py \
  --run "$MUSCLE_RUN" --experiments 1 2 3 4 5 6 7 8 --seconds 30
```

Output defaults to `$MUSCLE_RUN/videos_best_rear_to_front/`. Videos are 1280×720
at 30 FPS: rear hold for 6 seconds, eased orbit to the front over 20 seconds,
then front hold for 4 seconds. The whole body stays in frame and resets are
labelled. Rollouts, metadata and preview frames are saved beside the MP4s.
Recording uses offscreen EGL and requires working graphics access and `ffmpeg`.

To change rendering without rerunning physics:

```bash
"$MUSCLE_PY" scripts/record_prosthesis_evals.py \
  --run "$MUSCLE_RUN" --experiments 8 2 6 --render-only
```

## Intact H1-2 walking policy

The retained intact H1 best checkpoint initializes H1 + CMU. These optional
commands rebuild references or train a replacement; existing evaluations do not
require running them again.

```bash
# Original already-retargeted H1 source -> 50 Hz tracking reference.
"$H1_PY" scripts/h1_2_tracking.py prepare \
  --motion-file assets/h1_amass/motions/walk1_subject1.csv \
  --output assets/reference_clips/walk1_subject1_h1_2.npz

# Build the separate smooth 120-second straight-walking clip.
"$H1_PY" scripts/build_h1_2_straight_walk.py --seconds 120

# Train intact H1 from scratch in a fresh output folder.
"$H1_PY" scripts/h1_2_tracking.py train \
  --motion-file assets/reference_clips/h1_2_straight_walk_120s.npz \
  --num-envs 2048 --max-iterations 3000 --save-interval 100 \
  --output "logs/rsl_rl/h1_2_tracking_standalone/$(date +%Y-%m-%d_%H-%M-%S)"
```

Evaluate the retained intact H1 checkpoint (this pipeline uses GPU physics):

```bash
"$H1_PY" scripts/h1_2_tracking.py eval \
  --checkpoint "$H1_PRETRAINED" --num-envs 1 --viewer \
  --episode-seconds 120 --steps 6000
```

View the reference kinematically, without a policy:

```bash
"$H1_PY" test_scripts/replay_h1_2_reference.py \
  --motion assets/reference_clips/h1_2_straight_walk_120s.npz
```

## H1 + CMU Exp4: prepare and train

H1 keeps its pretrained actor and critic, then fine-tunes with critic warmup and
a low bounded learning rate. The CMU actor and critic start from scratch. PPOs
and rewards are separate. H1 acts at 50 Hz, CMU at 100 Hz, and LLC/physics at 1 kHz.
Training uses CUDA and 30-second episodes.

The adapted H1/CMU reference already exists. Rebuild it when its source or model
changes, then train in a fresh output folder:

```bash
"$H1_PY" scripts/h1_2_cmu_exp4.py prepare

"$H1_PY" scripts/h1_2_cmu_exp4.py train \
  --human-checkpoint "$H1_PRETRAINED" \
  --num-envs 1024 --iterations 2500 --episode-seconds 30 \
  --output "$H1_TRAIN_RUN"
```

After new training, use `H1_RUN="$H1_TRAIN_RUN"` for its evaluation/recording.
For a different intact H1 initialization, change `H1_PRETRAINED` before training.

## H1 + CMU: evaluate and measure

Both `model_best_h1.pt` and `model_best_prosthesis.pt` contain the complete paired
policies. The following uses the prosthesis-return-selected best pair, as in the
recorded results. CPU evaluation runs both actors, native MuJoCo and the 1 kHz
controller on CPU; first-run controller compilation adds startup time.

```bash
"$H1_PY" scripts/h1_2_cmu_exp4.py eval \
  --checkpoint "$H1_RUN/model_best_prosthesis.pt" \
  --backend cpu --num-envs 1 --viewer --episode-seconds 120 --steps 12000
```

For headless tracking measurements, omit `--viewer`:

```bash
"$H1_PY" scripts/h1_2_cmu_exp4.py eval \
  --checkpoint "$H1_RUN/model_best_prosthesis.pt" \
  --backend cpu --num-envs 1 --episode-seconds 120 --steps 12000 \
  --output "$H1_RUN/eval_cpu_120s.json"
```

Evaluate 32 independently sampled starts, each up to 10 seconds:

```bash
"$H1_PY" scripts/measure_h1_cmu_tracking.py \
  --checkpoint "$H1_RUN/model_best_prosthesis.pt" \
  --episodes 32 --seconds 10 --seed 12345 \
  --output "$H1_RUN/eval_cpu_32x10s.json"
```

Optional GPU evaluation uses `--backend gpu`. `--viewer-fps 15` reduces viewer
refresh overhead. All evaluations stop each measured episode at its first
failure or horizon; a requested 120 seconds is not a guarantee of survival.

## H1 + CMU: record the best pair

```bash
"$H1_PY" scripts/record_h1_cmu_eval.py \
  --checkpoint "$H1_RUN/model_best_prosthesis.pt" --seconds 30
```

Outputs go to `$H1_RUN/videos_best_rear_to_front/`, including
`h1_cmu_best_rear_to_front.mp4`. It uses CPU policies/controller/physics and
renders the same whole-body rear-to-front 30-second camera path. Omitting
`--checkpoint` automatically selects the latest best prosthesis checkpoint
under `logs/h1_2_cmu_exp4/`.

## Existing result locations

- Eight experiments: `logs/musculoskeletal_prosthesis/eight_experiments_20261021_231020/`
- Muscle videos: `videos_best_rear_to_front/` inside that run.
- H1 + CMU: `logs/h1_2_cmu_exp4/2026-10-23_23-31-52/`
- H1 + CMU video: `videos_best_rear_to_front/h1_cmu_best_rear_to_front.mp4` inside that run.
- Comparison image: `docs/figures/h1_cmu_vs_musculo_exp8.png`.

Only best checkpoints were retained during cleanup. Use the best filenames in
this document rather than deleted `model_N.pt` or `model_last.pt` files.
