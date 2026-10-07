# MyoWarp Commands

Run these commands from the repository root with the MyoWarp environment active.

## Model viewers

```bash
# Unitree G1
python test_scripts/view_g1.py

# Unitree H1-2
python test_scripts/view_h1.py

# Unitree H1 with OSL right leg
python test_scripts/view_h1_with_osl.py

# Standalone Open Source Leg
python test_scripts/view_osl.py

# Open Source Leg with the musculoskeletal model
python test_scripts/view_osl_musculoskeletal.py
```

## Headless viewer smoke tests

```bash
python test_scripts/view_g1.py --headless --steps 10
python test_scripts/view_h1.py --headless --steps 10
python test_scripts/view_h1_with_osl.py --headless --steps 10
python test_scripts/view_osl.py --headless --steps 10
python test_scripts/view_osl_musculoskeletal.py --headless --steps 10
```

## Model checks

```bash
python scripts/check_model.py \
  --model assets/robots/osl/osl.xml

python scripts/check_model.py \
  --model assets/robots/osl/myolegs_OSL_KA.xml
```

## Random rollouts

```bash
# CPU OSL rollout
python -m myowarp.train.random_rollout \
  --config configs/imitation_osl80_default.json \
  --backend cpu \
  --num-envs 1 \
  --steps 5

# Parallel MuJoCo Warp rollout
python -m myowarp.train.random_rollout \
  --config configs/imitation_tutorial_22_separated_net_partial_obs.json \
  --backend warp \
  --num-envs 64 \
  --steps 10
```

## Training

```bash
# Standalone H1 with OSL velocity training (MuJoCo-Warp + local RSL-RL)
python scripts/train.py h1_with_osl_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001

# Small CPU smoke run
python scripts/train.py h1_with_osl_velocity \
  --device cpu \
  --num-envs 4 \
  --num-steps-per-env 4 \
  --max-iterations 1 \
  --disable-domain-randomization \
  --disable-observation-noise

# Continue training; the command-range curriculum counter is restored too
python scripts/train.py h1_with_osl_velocity \
  --resume logs/rsl_rl/h1_with_osl_velocity_standalone/RUN/model_ITERATION.pt

# Standalone H1 with OSL motion tracking
python scripts/train.py h1_with_osl_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000

# Dual-policy full-observation velocity tracking
python scripts/train.py h1_with_osl_dual_full_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001

# Dual-policy partial-observation velocity tracking
python scripts/train.py h1_with_osl_dual_partial_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001

# Dual-policy full-observation motion tracking
python scripts/train.py h1_with_osl_dual_full_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000

# Dual-policy partial-observation motion tracking
python scripts/train.py h1_with_osl_dual_partial_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000

# Motion tracking with another compatible reference clip
python scripts/train.py h1_with_osl_tracking \
  --motion-file assets/reference_clips/short_reference_gait_h1_with_osl.npz
```

## Reference-clip retargeting

```bash
# Retarget the MyoAssist human gait clip to H1 with the OSL right leg
python scripts/retarget_h1_with_osl_reference.py

# Replay the retargeted motion (Space: pause, R: restart)
python test_scripts/replay_h1_with_osl_motion.py
```

## Policy evaluation

```bash
# Headless evaluation at a fixed forward-speed command
python scripts/eval.py h1_with_osl_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --steps 1000 \
  --num-envs 1 \
  --device cuda:0

# Native MuJoCo viewer
python scripts/eval.py h1_with_osl_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer

# Zero-action model check, with no checkpoint required
python scripts/eval.py h1_with_osl_velocity \
  --agent zero \
  --device cpu \
  --steps 100

# Motion tracking from the beginning of the reference
python scripts/eval.py h1_with_osl_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer

# Dual-policy full-observation velocity evaluation
python scripts/eval.py h1_with_osl_dual_full_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_full_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer

# Dual-policy partial-observation velocity evaluation
python scripts/eval.py h1_with_osl_dual_partial_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_partial_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer

# Dual-policy full-observation motion evaluation
python scripts/eval.py h1_with_osl_dual_full_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_full_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer

# Dual-policy partial-observation motion evaluation
python scripts/eval.py h1_with_osl_dual_partial_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_partial_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer

# Motion-tracking zero-action and metric smoke test
python scripts/eval.py h1_with_osl_tracking \
  --agent zero \
  --device cpu \
  --num-envs 1 \
  --deterministic \
  --steps 100
```
