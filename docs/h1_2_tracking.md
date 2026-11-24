# Intact H1-2 walking: independent MJLab-parity training

This is a separate 27-joint H1-2 task. It reuses the existing standalone
MJLab-equivalent motion-tracking PPO, reward, randomization, termination and
adaptive reference-sampling implementations. Existing H1-with-OSL task
implementations, references and training/evaluation entry points are unchanged.

The policy and critic start **from scratch**. This pipeline does not load the
ProtoMotions checkpoint or use its per-step XY reference realignment.

## Commands

Use the main `.venv` environment (MuJoCo/Warp 3.5). The pretrained AMASS replay
continues using its separate `.venv-musclemimic` environment.

```bash
# Reference already converted in this workspace. Recreate if needed:
.venv/bin/python scripts/h1_2_tracking.py prepare

# Train on the long walking clip:
.venv/bin/python scripts/h1_2_tracking.py train \
  --num-envs 1024 --max-iterations 30000 --save-interval 500

# Evaluate a saved policy (substitute the actual run path):
.venv/bin/python scripts/h1_2_tracking.py eval \
  --checkpoint logs/rsl_rl/h1_2_tracking_standalone/RUN/model_best.pt \
  --num-envs 1 --viewer --steps 500 --start-frame 500

# GPU shape/physics check:
.venv/bin/python scripts/h1_2_tracking.py check --num-envs 2
```

Only short smoke training is run during integration, not the full training job.
A new timestamped run folder is used unless `--output` is supplied. Reusing a
folder containing checkpoints requires `--checkpoint` to resume explicitly.
Resume uses the existing RSL-RL additional-iteration semantics.

`model_best.pt` is saved when the rolling mean return over 100 completed
training episodes improves. It includes both networks and optimizer states;
this is training-return selection, not an independently measured tracking best.
Periodic and final checkpoints and motion-bundled ONNX export use the existing
tracking runner. Best saving does not repeatedly export the long motion ONNX.
TensorBoard logs and JSON configs are stored in the same run folder.

Evaluation is deterministic: no observation noise, motion-reset perturbations,
startup randomization or pushes. It reports the first episode only; subsequent
automatic resets are excluded. Physical-state metrics are captured before
resets. Default episodes are 10 seconds; `--steps` caps evaluation length and
`--start-frame` chooses the reference frame at 50 Hz.

## Converted reference

- Input: already-retargeted Unitree H1-2 `walk1_subject1.csv` at 30 FPS.
- Output: `assets/reference_clips/walk1_subject1_h1_2.npz`.
- 13,066 frames at 50 Hz, spanning 261.30 seconds.
- CSV joint positions are not retargeted again. Quaternion slerp resamples the
  root orientation; `mj_differentiatePos` gives velocities.
- Body poses, quaternions and velocities are generated using forward kinematics
  of the actual intact H1-2 training model, rather than reusing the pretrained
  checkpoint's box-foot model features. Quaternions are MuJoCo WXYZ.
- Initial XY is rebased to zero once; the recorded path and heading are retained.
- A constant **4.02 cm upward root correction** removes the maximum foot-capsule
  penetration in the source clip; this preserves root velocity and relative
  joint motion. This is not a framewise tracking correction.
- Names and ordering are saved and validated. The NPZ uses the same required
  fields as the MJLab tracking loader and our standalone tracking reference.

The CSV is one long walking recording with direction/style changes, not a
promise that every frame is straight-line constant-speed walking.

## Task contract

| Item | Intact H1-2 |
| --- | --- |
| Actions | 27 joint-position targets |
| Actor inputs | 150 |
| Critic inputs | 276 |
| Tracked bodies | 14 |
| Motion anchor | torso_link |
| End effectors | both ankles and wrists |
| Physics / policy timestep | 5 ms / 20 ms |
| Default episode duration | 10 s |

Actor inputs: reference joint q/qvel (54), anchor position error (3), relative
anchor orientation in 6D (6), base linear/angular velocities (6), measured joint
positions relative to home (27), joint velocities (27), previous actions (27).
The critic adds root-relative positions/orientations for 14 tracked bodies
(126 values). Actor corruption and encoder bias follow the original task.

The reference anchor position is expressed relative to the **actual current
anchor**, giving explicit lag/drift feedback. The global-anchor position reward
uses the fixed reference path; only the body-relative tracking terms align their
XY anchor to the robot, exactly as in the existing MJLab-parity reward.

Rewards and PPO hyperparameters are unchanged: global anchor position/orientation,
anchor-aligned body position/orientation, global body linear/angular velocities,
action-rate penalty, soft joint limits and self-collision penalty. Model gains,
home pose and action scaling use the H1-2 recipe, including both knee actuators.

## Isolation and implementation

- New runtime package: `myowarp/tasks/h1_2_tracking/`.
- New dedicated CLI: `scripts/h1_2_tracking.py`.
- Standalone runtime; legacy MJLab registration/configuration files were removed
  during branch cleanup.
- New reference and log paths; checkpoints carry a separate `h1_2_tracking`
  pipeline identifier, and unrelated checkpoints are rejected.

The runtime subclasses the original tracking environment and overrides only
construction, model/body/action indices, buffer widths, observation widths,
pre-reset evaluation-metric capture, and the two global tracking reward weights.
Shared methods do not receive mutated
module constants or monkeypatches. There are no MJLab runtime imports in the
standalone pipeline. Both use the same 14-body count; the subclass explicitly
guards that count where shared base methods rely on it.
