#!/usr/bin/env bash
# Eight independent experiments, run sequentially; stop on any failure.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv-musclemimic/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/musculoskeletal_prosthesis/eight_experiments_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_MUSCLE="$ROOT/assets/musclemimic/checkpoints/mm-10m-2"
VYVAS_MUSCLE="$ROOT/vyvas_pi_h/model_1499.pt"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
[[ -d "$DEFAULT_MUSCLE" && -f "$VYVAS_MUSCLE" ]] || {
    echo "Missing default or Vyvas muscle checkpoint" >&2; exit 1;
}
mkdir -p "$(dirname -- "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT" || { echo "Use a fresh OUTPUT_ROOT" >&2; exit 1; }
export PYTHONUNBUFFERED=1
COMMON=(--device "$DEVICE" --num-envs 1024 --save-interval 100)

# Shared reference geometry only; no learned checkpoint passes between runs.
"$PYTHON" scripts/prosthesis_rl.py prepare --anatomy removed \
    2>&1 | tee "$OUTPUT_ROOT/prepare.log"

run_original() {
    local name="$1" human="$2"
    shift 2
    echo "Starting $name ($(date -Is))"
    "$PYTHON" scripts/prosthesis_rl.py train "${COMMON[@]}" --iterations 800 \
        --anatomy removed --right-leg-obs virtual --seed 42 \
        --human-checkpoint "$human" --output "$OUTPUT_ROOT/$name" "$@" \
        2>&1 | tee "$OUTPUT_ROOT/$name.console.log"
}

run_split() {
    local name="$1" iterations="$2" reward_mode="$3"
    shift 3
    echo "Starting $name: $iterations iterations ($(date -Is))"
    "$PYTHON" scripts/prosthesis_split_rl.py train "${COMMON[@]}" \
        --iterations "$iterations" --human-checkpoint "$DEFAULT_MUSCLE" \
        --reward-mode "$reward_mode" --output "$OUTPUT_ROOT/$name" "$@" \
        2>&1 | tee "$OUTPUT_ROOT/$name.console.log"
}

run_original 01_default_frozen "$DEFAULT_MUSCLE"
run_original 02_vyvas_frozen "$VYVAS_MUSCLE"
run_original 03_default_coopt "$DEFAULT_MUSCLE" \
    --train-muscle --muscle-learning-rate 1e-5
run_split 04_independent 2500 separate
run_split 05_shared 800 shared
run_split 06_muscle_reference 800 separate --prosthesis-observation muscle_reference
run_split 07_hip_rpy 800 separate --prosthesis-observation hip_rpy
run_split 08_human_space 2500 separate --prosthesis-reward human_space
echo "All eight experiments finished. Results: $OUTPUT_ROOT"
