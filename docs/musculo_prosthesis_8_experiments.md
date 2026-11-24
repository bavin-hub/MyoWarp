# Eight musculoskeletal + CMU prosthesis experiments

These experiments test how a pretrained whole-body muscle policy and a learned
CMU knee/ankle policy can control the same combined embodiment. They compare a
frozen muscle policy, joint optimization, independent optimization, prosthesis
reference inputs, and the coordinate space used for prosthesis tracking rewards.

Use [commands.md](commands.md) for individual training/evaluation commands, the
single eight-run shell script, tracking measurements and video recording. This
document explains the experimental differences. H1 + CMU is a separate pipeline,
described in [h1_2_cmu_exp4.md](h1_2_cmu_exp4.md).

## Shared embodiment and control flow

All eight retained runs use the model with the biological right lower leg
removed:

`assets/robots/myofullbody_cmu_leg_no_lower_leg/myofullbody_cmu_leg_no_lower_leg.xml`

The physical model has 332 retained muscle actuators and four CMU motor
channels. The original biological policy has 354 muscle outputs; only the 332
surviving commands can affect the physical model. A full biological shadow model
preserves the pretrained observation layout.

The policies run at **100 Hz**. On every 10 ms policy interval:

1. Construct the muscle observation using the virtual biological model and
   construct the prosthesis sensor/history observation from the real model.
2. Predict muscle commands and two desired prosthesis joint torques.
3. Apply the retained muscle commands to the real combined model.
4. Execute ten 1 ms low-level controller and physics steps. The SEA controller
   converts desired joint torques into motor feedforward-plus-integral torques
   and motor velocity targets; MuJoCo supplies velocity proportional feedback.
5. Measure the new physical state, compute the applicable reward or rewards,
   update histories, and reset completed/failed environments.
6. Collect rollouts and perform the experiment's PPO update or updates.

Training and musculoskeletal evaluation use the local CUDA MuJoCo-Warp pipeline
in `.venv-musclemimic`. The 1 kHz controller is ported locally; there are no runtime
API calls to the sibling ProSim or CMU repositories.

The motion is the **30-second stitched walking reference**
`assets/musclemimic/motions/loops/walking_medium10_poses_30s.npz`. The physical
prosthesis reference is fitted separately for initialization, privileged critic
inputs and the applicable physical-space rewards. Training episodes are capped
at 10 seconds, with sampled starts and earlier failure termination. The motion
length and episode length are different quantities.

## Observations and actions

### Virtual muscle observation

The original full-body observation contains **2,418 values**, including body
state, muscle features, contacts/tracking features and motion-reference targets.
In virtual mode:

- Retained body/joint state comes from the real combined model.
- Biological knee/ankle coordinates and velocities are replaced using actual
  prosthesis kinematics in the biological observation convention.
- Missing right subtalar/toe coordinates and velocities are zero; biological
  knee follower coordinates are projected consistently.
- The 22 missing muscles' activations and excitations are zero. Derived muscle
  and site features are recomputed in the biological shadow.

Zeroing activation does not mean every derived feature is zero. Length, velocity
and related kinematic features still depend on the reconstructed pose.
The full network keeps its 354-output architecture; joint/independent learners
include only the 332 physically retained muscle commands in their likelihoods.

### Prosthesis actor

The baseline actor has **115 inputs**: five consecutive 100 Hz history frames,
oldest to newest (`t-40 ms` through `t`). Each 23-value frame contains:

- Pelvis-relative hip RPY: 3.
- Right-thigh world orientation, represented in 6D: 6.
- Right-thigh world angular velocity: 3.
- Gravity direction in the pelvis frame: 3.
- Prosthesis knee/ankle positions and velocities: 4.
- Prosthetic toe/heel contact flags: 2.
- Previous torque action: 2.

There is no left-foot information, SEA deflection, explicit pelvis world
orientation/angular velocity, or reference input in the baseline prosthesis
actor. Exp6 adds reference information as described below. Hip RPY is computed
from `R_pelvis.T @ R_thigh`; it is not simply a copy of anatomical hip joint
coordinates.

Every prosthesis actor outputs **two desired joint torques**, not motor commands
or joint position targets. Torque scaling/limits and the existing low-level
controller are shared across the experiments.

### Critics

The privileged prosthesis critic receives **671 values** for the removed-leg
model, including whole-body/actuator state and reference information unavailable
to the deployment actor. Exp1–3 use one such critic. Exp4–8 additionally have a
separate pretrained muscle critic receiving the **2,418-value virtual muscle
observation**, with the same masking/reconstruction as the muscle actor.

## Experiment overview

| Exp | Muscle policy | Optimization and critics | Reward delivery | Prosthesis actor |
| --- | --- | --- | --- | --- |
| 1 | Default `mm-10m-2`, frozen | Prosthesis PPO and new critic only | Existing full combined-model reward | Baseline 115 |
| 2 | Vyvas `model_1499.pt`, frozen | Same as Exp1 | Same as Exp1 | Baseline 115 |
| 3 | Default actor, fine-tuned | Joint PPO likelihood, one new shared critic | One combined-model reward and shared advantage | Baseline 115 |
| 4 | Default actor **and critic**, fine-tuned | Independent actors, critics and PPO updates | Separate masked muscle and prosthesis rewards | Baseline 115 |
| 5 | Same initialization as Exp4 | Independent actors, critics and PPO updates | Both learners receive the sum of Exp4 rewards | Baseline 115 |
| 6 | Same as Exp4 | Same as Exp4 | Same separate rewards as Exp4 | Baseline + reference errors and phase: 136 |
| 7 | Same as Exp4 | Same as Exp4 | Same separate rewards as Exp4 | Historical hip-RPY variant; final layout is 115 |
| 8 | Same as Exp4 | Same as Exp4 | Muscle reward unchanged; prosthesis tracking evaluated in human coordinates | Baseline 115 |

Every run starts independently. No experiment inherits learned policy weights
from another experiment. "Same as Exp4" describes the setup, not a resume from
Exp4's checkpoint.

## Exp1: frozen default muscle policy

Keep the original `assets/musclemimic/checkpoints/mm-10m-2` muscle policy frozen.
Train only a randomly initialized prosthesis actor and privileged critic.

The prosthesis learner receives the existing full real-model reward: root/body
tracking, prosthetic foot tracking, contact/support behavior and regularization
terms. Although it controls only the prosthesis, its reward can assess the
whole body's motion. This is the default pretrained-human baseline.

## Exp2: frozen Vyvas muscle policy

Repeat Exp1 with `vyvas_pi_h/model_1499.pt` as the frozen muscle checkpoint.
The prosthesis actor and critic still start from scratch. The embodiment,
prosthesis observation and reward structure stay the same.

This tests whether the alternative pretrained muscle policy works better with
the prosthesis without further muscle fine-tuning.

## Exp3: joint muscle/prosthesis PPO

Initialize the muscle actor from the default checkpoint and train it alongside
a new prosthesis actor. Their actions form one joint Gaussian policy, with
summed log probabilities, one shared new privileged critic, and one shared
combined-model reward/advantage. The pretrained muscle critic is **not** loaded.

The muscle actor keeps its pretrained residual architecture. The 332 retained
muscle outputs and two prosthesis outputs enter the joint likelihood. Muscle
observations used for rollout actions are cached for PPO replay.

The muscle network and exploration parameters use learning rate **1e-5**;
prosthesis actor and critic use **1e-3**. Initial muscle exploration standard
deviation is **0.03**, versus **0.2** for prosthesis actions. Scheduling is fixed
to preserve those optimizer-group rates. Therefore, Exp3 also differs from the
frozen runs in optimization schedule and exploration; it is not an isolated
comparison of muscle updates alone.

## Exp4: independent PPO learners and separate rewards

Initialize both muscle actor and muscle critic from the default checkpoint.
Initialize a separate prosthesis actor and critic randomly. Each pair has its
own optimizer, rollout buffer, returns, advantages and PPO update. The pairs
share physical transitions and reset signals, not a critic or reward scalar.

The muscle pair uses fixed pretrained normalization. The prosthesis pair uses
its existing running observation normalization. Muscle adaptation follows the
locally ported ProSim fine-tuning approach:

- First **20 updates**: freeze muscle actor and its exploration parameters;
  update the muscle critic while the prosthesis learner trains normally.
- After warmup: train both muscle networks with initial learning rate **3e-5**,
  bounded to **[1e-6, 1e-4]**, using adaptive KL scheduling.
- Initialize muscle exploration deviations at **0.5×** pretrained values.
- Muscle PPO uses four epochs, four minibatches, entropy coefficient 0,
  value-loss coefficient 0.5, clipping 0.2, gamma 0.99 and lambda 0.95.
- Default rollout length is 24 steps per environment, or **0.24 seconds** at
  100 Hz, before an update. Episodes continue across updates.

### Muscle reward

Use the ported MuscleMimic reward with right lower-leg joint/follower terms and
right ankle/toe tracking sites excluded. Retained hip/thigh and knee-site
tracking remain. It is evaluated on the biological virtual reconstruction of
the real state; the excluded lower-leg terms do not drive muscle learning.
It is not a separate freely simulated biological rollout.

| Term | Weight | Exponential scale |
| --- | ---: | ---: |
| Joint position | 0.1 | 10 |
| Joint velocity | 0.1 | 2 |
| Root position | 0.1 | 10 |
| Root local velocity | 0.1 | 10 |
| Relative site position | 0.6 | 100 |
| Relative site rotation vector | 0.01 | 10 |
| Relative site angular velocity | 0.1 | 0.1 |
| Relative site linear velocity | 0.1 | 0.1 |

Retain the out-of-bounds penalty on commanded muscles. The nonnegative total is
scaled by **0.0385**, matching the inherited pretrained-critic calibration.
Do not multiply this scaled muscle reward by another policy timestep.

### Prosthesis reward

Keep only the existing real-model terms specific to prosthesis behavior:
`foot_pos`, `foot_ori`, `foot_vel`, `contact`, `support`, `slip`, `impact`,
`action_rate`, `effort`, `saturation`, and `joint_limit`.

These use the fitted prosthesis reference, original weights/scales, and the
**0.01 s** reward multiplier. No whole-body/root tracking reward or muscle
reward is added to this learner. Global safety and masked muscle tracking
termination still reset both policies together.

## Exp5: independent PPO learners with one shared reward

Keep Exp4's networks, initialization, observations, warmup and optimizers.
Change only the delivered scalar reward:

```text
R_shared = R_muscle + R_prosthesis
muscle PPO receives R_shared
prosthesis PPO receives R_shared
```

Each component already includes its own scaling. The sum has unit coefficients;
it is not averaged or scaled again. Both critics learn the same return target
structure from different observations, but compute their own values and
advantages. Log probabilities and PPO updates remain independent. This tests
shared incentives without switching to Exp3's joint PPO architecture.

## Exp6: reference-conditioned prosthesis actor

Keep Exp4's separate rewards and learning setup. Add adapted knee/ankle reference
errors at **0, 0.2, 0.4, 0.6 and 0.8 seconds** ahead:

```text
q_ref(t + offset)    - q_actual(t)
qvel_ref(t + offset) - qvel_actual(t)
```

Each time contributes two position and two velocity errors. Append normalized
motion phase: **115 + 5×4 + 1 = 136 inputs**. Targets use prosthesis-compatible
joint coordinates. The reference window uses the muscle actor's lookahead
timing, but contains only joint errors, not full-body/pelvis-relative site
features. It does not add global pelvis-position error feedback.

This tests whether explicit motion objectives improve prosthesis tracking.
It requires the reference at execution time. Critics and muscle observations
are unchanged. The launcher uses `--prosthesis-observation muscle_reference`.
The legacy `reference` flag instead adds four current absolute joint targets
(119 inputs) and is not the retained Exp6 definition.

## Exp7: hip-RPY history

The original intention was to add pelvis-relative hip RPY history to Exp4's
prosthesis sensor inputs without providing a reference. The later common
observation update replaced pelvis orientation/angular velocity with hip RPY
for **all** experiments.

Consequently, the final Exp7 actor has the same **115 inputs** as Exp4. It does
not append duplicate hip values or add a sixth temporal frame. The
`--prosthesis-observation hip_rpy` flag and independent run remain for experiment
history, but the final runs are not distinct observation ablations. In the
retained eight-run batch, Exp4 also has more training iterations than Exp7.

## Exp8: prosthesis tracking reward in human coordinates

Keep Exp4's actors, critics, separate PPO updates, muscle reward and baseline
prosthesis observation. Replace only the prosthesis tracking component with a
restricted MuscleMimic-style reward in biological coordinates:

```text
actual CMU knee/ankle q/qvel
    -> CMU virtual-human retargeting convention
    -> biological reward-model forward kinematics
    -> comparison with the original biological motion
```

The mapping is ported locally from CMU commit
`046002ba1c655c43b64b893486195527b4a8ce13`: `knee_leg_to_human` in
`retargeting.py` plus the ankle correction in `neuromuscular_stance.py`.
Its fixed source lengths are thigh **0.66 m**, shank **0.215 m**, and equal virtual
segments **0.4375 m**. It preserves planar chord/foot-orientation conventions;
it is not an exact spatial match between the two full-body models. Velocities
use the analytic mapping Jacobian, including its continuous zero-knee limit.
There is no motor-coordinate or torque conversion in this reward mapping.

A separate biological GPU reward model copies retained current body state,
inserts mapped knee/ankle state, projects followers, zeros missing subtalar/toe
coordinates, and recomputes kinematics. This changes neither physical state nor
the muscle actor's virtual observation model.

| Prosthesis tracking term | Weight | Exponential scale |
| --- | ---: | ---: |
| Mapped knee/ankle position | 0.1 | 10 |
| Mapped knee/ankle velocity | 0.1 | 2 |
| Right ankle/toe relative site position | 0.6 | 100 |
| Right ankle/toe relative rotation vector | 0.01 | 10 |
| Right ankle/toe relative angular velocity | 0.1 | 0.1 |
| Right ankle/toe relative linear velocity | 0.1 | 0.1 |

Sum these terms and multiply by **0.0385**. They replace Exp4's `foot_pos`,
`foot_ori`, and `foot_vel`. Retain real-state contact/support, slip, impact,
action-rate, motor effort, saturation and joint-limit terms with their original
weights and timestep scaling. No whole-body/root reward or muscle-specific
term enters this prosthesis tracking objective.

The fitted physical-space reference remains in use for initialization and
privileged critic inputs. Standard evaluation still measures physical-space
errors against that common adapted reference, enabling the eight experiments'
reported metrics to use the same convention. Those evaluation errors are not
Exp8's human-space training reward.

## Training settings, best checkpoints and retained results

The retained batch uses **1,024 environments**, seed 42, 24 policy steps per
rollout, and a checkpoint interval of 100 updates. Exp4 and Exp8 run for
**2,500 iterations**; Exp1/2/3/5/6/7 run for **800**. Earlier three-, five-, and
two-experiment launchers have different historical limits; the authoritative
current commands are in [commands.md](commands.md).

- Exp1–3: `model_best.pt`; Exp3 includes the fine-tuned muscle actor.
- Exp4–8: `model_best_muscle.pt` and `model_best_prosthesis.pt`. Each is a
  complete contemporaneous pair of actors/critics, optimizer state and settings.
  Never mix the muscle policy from one best file with the prosthesis policy from
  another. The labels identify which learner's return selected the snapshot.
- Best selection uses rolling training episode return, not deterministic
  evaluation tracking error. Exp5 uses shared return for both selectors.
- Only best checkpoints remain after cleanup. New runs still save periodic/final
  checkpoints. Resume resets physics, low-level filters and observation history;
  it is not a restoration of an interrupted physical rollout.

Retained runs:
`logs/musculoskeletal_prosthesis/eight_experiments_20261021_231020/`.
Their best-checkpoint evaluation uses **32 matched starting points**, a
**10-second horizon**, and samples through the first terminal state, before automatic reset.
Completion, duration, global drift and local pose error must be read together:
short failed trials have less time to accumulate global drift.

Tracking reports and the results image are under `tracking_evaluation_best/`.
Recorded 30-second Exp8/Exp2/Exp6 videos are under
`videos_best_rear_to_front/`, with actual resets labelled. A 30-second video may
contain multiple shorter episodes; it is not proof of uninterrupted survival.

## Implementation references

- `myowarp/tasks/musculoskeletal_prosthesis/gpu_env.py`: shared physics and resets.
- `coopt.py`: Exp3 joint actor, shared critic/PPO and optimizer groups.
- `split_env.py`, `split_runner.py`: Exp4–8 rewards, observations and independent PPOs.
- `split_models.py`, `split_config.py`: pretrained actor/critic and fine-tuning settings.
- `prosthesis_observations.py`: current hip-based actor history.
- `human_space_reward.py`: Exp8 retargeting and restricted biological reward.
- `sea_controller.py`: local 1 kHz low-level controller and simulated drive integral.

The muscle fine-tuning/reward workflow was ported from ProSim `5c58afd`;
provenance notices remain beside the implementation.
