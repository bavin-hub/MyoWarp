# H1-2 + CMU Exp4

This separate pipeline fine-tunes the intact H1-2 walking actor and critic while
training a CMU prosthesis actor and critic from scratch. The two learners have
independent rewards, advantages, PPO updates and optimizers. It uses the
120-second straight-walking clip. Existing H1-with-OSL, intact H1 tracking and
musculoskeletal pipelines are unchanged.

## Commands

The adapted reference is already generated. To rebuild after changing the
source clip, model, or adaptation implementation:

```bash
.venv/bin/python scripts/h1_2_cmu_exp4.py prepare
```

Train a new experiment (no long training job is started by setup):

```bash
.venv/bin/python scripts/h1_2_cmu_exp4.py train \
  --num-envs 1024 --iterations 2500 --episode-seconds 30
```

Logs and paired checkpoints go into `logs/h1_2_cmu_exp4/TIMESTAMP/`.
TensorBoard has separate `h1/` and `prosthesis/` event files. Best files
`model_best_h1.pt` and `model_best_prosthesis.pt` each contain BOTH learners
from the same instant. Select `model_best_prosthesis.pt` for prosthesis-return
selection; `model_best_h1.pt` selects H1 return. These are rolling training
returns (up to 100 completed episodes), not held-out tracking measurements.
Periodic `model_N.pt` and `model_last.pt` also contain both policies.
`model_last.pt` is saved on normal completion or a handled interruption.

Evaluate an entire 120-second episode from reference frame zero:

```bash
.venv/bin/python scripts/h1_2_cmu_exp4.py eval \
  --checkpoint logs/h1_2_cmu_exp4/TIMESTAMP/model_best_prosthesis.pt \
  --num-envs 1 --viewer --episode-seconds 120 --steps 12000
```

Omit `--viewer` for headless evaluation. The CLI writes `eval_exp4.json` and a
per-step trace beside the checkpoint. Statistics include only the first
episode per environment, before automatic resets. `--start-frame` uses 50Hz
source frames and requires enough remaining motion for the requested episode.
`--output` selects the report path. Resume both optimizers and warmup counters:

```bash
.venv/bin/python scripts/h1_2_cmu_exp4.py train \
  --checkpoint logs/h1_2_cmu_exp4/TIMESTAMP/model_last.pt \
  --output logs/h1_2_cmu_exp4/TIMESTAMP --num-envs 1024 --iterations 2500
```

`--iterations` is a total limit, including resumed updates. A resume resets
physical states, controller filters and sensor histories; it does not restore
an interrupted rollout. Configurations and reference/model hashes are checked.

## Real model and timing

The real combined model has 24 H1 position actuators and four CMU motor
channels. The original H1 actor still emits 27 actions; its right knee,
ankle-pitch and ankle-roll actions are ignored. They are not converted into
CMU torques in this experiment. The new prosthesis policy emits two desired
joint torques, scaled by 60/80 Nm per action and limited to 100/150 Nm.

H1 acts at **50Hz**, preserving its pretrained timing. CMU acts at **100Hz**;
the SEA low-level controller and MuJoCo-Warp physics run at **1kHz** on CUDA.
During each H1 interval, its action is held for two CMU policy intervals. Each
PPO update collects 24 H1 transitions and 48 CMU transitions, covering the same
0.48 seconds. H1 rewards sum the two 10ms contributions. Episodes end only at
20ms boundaries so both PPOs receive consistent terminals.

Training episodes last up to **30 seconds** and reset earlier for falls,
large tilt, nonfinite state, or unsafe contacts. Reference start frames use
failure-biased sampling with 10% uniform probability. Starts reserve enough
reference frames for a full episode; the finite recording never wraps during
an episode. There is no artificial global XY realignment or per-step pose
correction. Evaluation starts deterministically and can run all 120 seconds.
Noise, random pushes and physical parameter randomization are disabled in
this initial separate setup to isolate coupled-policy learning.

## H1 actor, critic and observations

Both networks and their normalization statistics initialize from
`logs/rsl_rl/h1_2_tracking_standalone/2026-10-23_22-21-41/model_best.pt`.
The actor has 150 inputs; the critic has 276. A full H1 virtual model copies
the real floating-base state and all 24 intact joint states. CMU knee maps to
H1 knee with positive sign; CMU ankle maps to H1 ankle pitch with negative
sign. Missing ankle-roll position/velocity are zero. All derived observation
features are recomputed by forwarding this virtual model. Reference inputs
stay in the original H1 motion convention. Previous actions keep the original
27-dimensional checkpoint schema. The virtual model supplies observations,
not the reward or physical contact measurements.

H1 uses its own PPO with 20 critic-only warmup updates. Its actor/critic
normalization statistics remain fixed. The actor exploration standard
deviation starts at half its pretrained value. H1 learning rate starts at
3e-5, bounded to [1e-6, 1e-4], with adaptive KL scheduling. Both actor and critic
fine-tune after warmup. The CMU actor/critic initialize randomly and have an
independent standard PPO optimizer. There is no shared critic or summed reward.

## Prosthesis observations

The actor receives five sensor-history frames of 23 values, total **115**:

- Hip RPY relative to pelvis.
- Right-thigh world orientation (6D) and angular velocity.
- Gravity in the pelvis frame.
- Actual CMU knee/ankle positions and velocities.
- Prosthetic toe/heel contact flags.
- Previous two-dimensional torque action.

It receives no left-foot information, reference input, global position,
explicit pelvis orientation/angular velocity, or SEA deflection. The
privileged critic receives 151 values: full real qpos/qvel, adapted target
qpos/qvel, foot-position error, and previous torque action. SEA internals in
the critic are privileged training information and never enter the actor.

## Separate rewards

Both rewards derive from the real combined state. H1's unchanged existing
reward formulas retain global torso position/orientation tracking (weights
2.0 and 1.0), body pose/velocity tracking, action rate, soft joint limits and
unsafe/self contact penalties. Tracking excludes replaced right knee and foot
bodies; joint/action penalties exclude all three removed H1 joints. The
unchanged body-relative reward geometry is evaluated on the 12 retained real
tracked bodies. Using a synchronized H1 representation for these retained
bodies is geometrically identical to their actual combined-model state;
no virtual lower-leg state contributes to the reward.

CMU's separate reward keeps Exp4's prosthesis terms: adapted foot position,
orientation and velocity; inferred contact timing; support; slip; impact;
action-rate, effort, saturation and joint-limit penalties. It contains no H1
whole-body or muscle reward terms. Both learners share physical failure and
timeout signals.

## Reference adaptation and its limits

The original H1 clip remains unchanged. The new file
`assets/reference_clips/h1_2_cmu_straight_walk_120s.npz` is sampled at 100Hz.
For each pose, shared root/intact joint states remain fixed while bounded
least squares fits CMU knee and ankle-drive angles to the original H1 sole
midpoint and orientation. A neutral-pose calibration aligns the two foot
coordinate frames. The four-bar linkage is closed analytically, joint limits
are enforced, and a small temporal regularizer discourages discontinuities.
Identical repeated poses reuse the same fitted solution. Joint/root velocities
are differentiated consistently; actual forward kinematics supplies attainable
CMU sole and retained-body targets.

The paired JSON report records geometric fitting errors **against H1**, which
are distinct from the policy's future tracking errors **against the adapted
CMU reference**. Current fit over 12,001 frames:

| Measurement | Value |
|---|---:|
| Mean foot-position mismatch | 4.87 cm |
| 95th percentile foot-position mismatch | 14.81 cm |
| Maximum foot-position mismatch | 16.46 cm |
| Mean foot-orientation mismatch | 7.36 degrees |
| Maximum foot-orientation mismatch | 17.58 degrees |
| Frames at a knee/drive limit | 31.0% |

These residuals are material: different knee height/link lengths and missing
ankle roll prevent exact lower-leg replacement. They are not hidden by changing
the original H1 reference or by moving the physical model every step.

The motion contains kinematics, not measured contact forces. Expected stance
is therefore a heuristic: low source-foot XY speed (<0.6m/s) and low sole
height (<10th-percentile height +4cm). Its current right stance fraction is
62.2%. Actual contact flags/support/slip penalties always use decoded real
MuJoCo contact forces, not those reference labels. Source references may float
above the ground; reset height is raised only if collision surfaces penetrate.

## Evaluation performance

GPU evaluation (`--backend gpu`) uses a compiled SEA tick inside a combined Torch/Warp
CUDA graph for each ten-tick (10ms) control interval. This preserves 1kHz
controller/physics and both policy rates. Training continues to use the original
uncaptured path. Fast inference skips unused reward calculations; safety
termination and reported tracking errors remain active. Use `--backend gpu --slow-eval` to
compare against the original GPU arithmetic/physics loop.

The viewer refreshes at 30Hz instead of 100Hz, forwards only rendering
kinematics, and paces playback against the elapsed rollout clock. Adjust it with
`--viewer-fps 15` for lower rendering overhead. Tracking traces stay on the GPU
until evaluation finishes instead of forcing four host transfers each step.
Initial compiler/graph setup can take a few seconds; the cached subsequent
rollout is what the speed measurements describe.

On the RTX 5090 with one environment and the actual best checkpoint, synchronized
stage profiling attributed approximately 47% of runtime to the Torch LLC and
25% to physics; unused reward computation accounted for another 12%. Profiling
synchronization itself adds overhead. An uninstrumented matched 5-second
headless comparison measured **8.58s original versus 4.39s optimized**, about
**1.95x faster**. A 30Hz viewer test measured **6.23s for 5s of motion**, including
1.47s in rendering/readback; full real-time viewer speed is not guaranteed.

Controller fixture parity, unchanged training/warmup behavior, checkpoint reload
and fast/slow physical-state parity tests passed. Compiled fused arithmetic and
GPU contact reductions are not bitwise identical; the short matched-policy
benchmark's maximum qpos difference over 0.3s was 6.6e-5. Longer contact-driven
trajectories can diverge numerically even with equivalent control laws.

## Default CPU evaluation

Evaluation now defaults to **native CPU MuJoCo**. Both actors, the virtual H1
model, physical simulation, contact decoding and the 1kHz controller execute on
CPU. No CUDA device is required. This backend supports one environment and is
not available for training. Training continues to use CUDA exactly as before.

```bash
.venv/bin/python scripts/h1_2_cmu_exp4.py eval \
  --checkpoint logs/h1_2_cmu_exp4/2026-10-23_23-31-52/model_best_prosthesis.pt \
  --backend cpu --num-envs 1 --viewer --episode-seconds 120 --steps 12000
```

The existing paired checkpoint loads directly; no retraining or model conversion
is needed. The CPU controller uses the same local SEA logic compiled into CPU
kernels. Native MuJoCo advances in 1ms steps, CMU acts every 10ms, and H1 acts
every 20ms. The CPU evaluator sets Torch to one thread for these small networks.
`--slow-eval` disables controller compilation for diagnostic comparison. First
compilation can add startup latency; compiler outputs are cached.

CPU and GPU backends use the same reference, observations, history, torque
limits, low-level filters and stopping conditions. Native MuJoCo uses float64
states while MuJoCo-Warp uses float32, so contact trajectories are not bitwise
identical. Actor inputs and checkpoint weights remain float32. CPU contact
loads come from native `mj_contactForce`, with the same supported-foot and
unsafe-contact classification as GPU evaluation.

A matched five-second headless evaluation measured **1.57s CPU versus 4.39s
optimized GPU** (2.8x faster). The CPU viewer completed five seconds in **5.02s**,
providing real-time playback at 30Hz. These timings exclude model loading and
one-time compiler setup. Use `--backend gpu` to explicitly select GPU evaluation.
The JSON report identifies the backend and simulation/wall-clock speed ratio.
