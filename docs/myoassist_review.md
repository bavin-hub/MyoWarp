# MyoAssist Review

## About

MyoAssist is a reinforcement-learning codebase for controlling lower-body musculoskeletal models, optionally coupled to an exoskeleton. Its custom training stack uses Stable-Baselines3 PPO and focuses on gait imitation and speed-controlled walking. The trained "human" is a musculoskeletal leg model with a simplified pelvis/torso; it is not a whole-body musculoskeletal model.

## Training Tasks

1. **Human-only gait imitation:** reproduce a reference gait using muscle activations.
2. **Human + exoskeleton, full observations:** maintain the reference gait with exoskeleton assistance while reducing human effort. The exoskeleton actor observes joint positions and velocities, muscle activations, foot forces, and target speed.
3. **Human + exoskeleton, partial observations:** pursue the same assisted-gait objective while the exoskeleton actor sees only ankle angles and angular velocities, approximating limited real-world sensing.
4. **Human training with exoskeleton disabled:** train the musculoskeletal gait controller while forcing exoskeleton actuator commands to zero.
5. **Speed-controlled human + exoskeleton:** follow varying commanded walking speeds while minimizing human effort.

## Model and Actions

The 22-muscle model contains 11 muscle actuators per leg. Human-only tasks therefore have 22 muscle actions. Exoskeleton tasks add two ankle actuator commands, producing a combined 24-dimensional action vector. There are no trained upper-body actions.

## Policy Architecture

Exoskeleton tasks use one PPO model containing two independent actor MLPs, rather than a single network split only at its final layer:

- The human actor maps its selected observations to 22 muscle actions.
- The exoskeleton actor maps its selected observations to two exoskeleton actions.
- Their outputs are assembled into one 24-action vector.
- A shared critic evaluates the joint system, and PPO trains both actors jointly.

Human-only training uses only the muscle-control policy. In the exoskeleton-disabled configuration, the exoskeleton outputs are overridden with zeros.

## Question 1

**How is the policy optimized when it has two separate actors? How does the PPO loss propagate into both actors?**

### Discussion

#### Joint PPO optimization

The human and exoskeleton actors are optimized through one joint PPO loss:

1. The human actor produces the means of 22 muscle actions.
2. The exoskeleton actor produces the means of two exoskeleton actions.
3. These means are assembled into one 24-dimensional Gaussian action distribution.
4. PPO calculates one joint log-probability by summing the log-probabilities of all 24 actions.
5. Backpropagation sends gradients into each actor through its own action outputs.

Both actors are therefore driven by the same joint PPO loss and global advantage signal. There are no separate human and exoskeleton PPO losses or separate credit-assignment signals. One optimizer updates both actors and the shared critic.

For example, suppose the aggregate human action log-probability changes from `-1.0` to `-0.9`, while the exoskeleton action log-probability changes from `-0.5` to `-0.45`. The old and new joint log-probabilities are `-1.50` and `-1.35`, so the PPO probability ratio is:

```text
exp(-1.35 - (-1.50)) = 1.16
```

With an advantage of `2.0`, the unclipped joint surrogate objective is `1.16 * 2.0 = 2.32`. Backpropagation sends the human portion of the gradient to the human actor and the exoskeleton portion to the exoskeleton actor.

#### Full- and partial-observation models

| Component | Full-observation task | Partial-observation task |
|---|---:|---:|
| Human actor observation | Full 44-dimensional state | Full 44-dimensional state |
| Exoskeleton actor observation | Full 44-dimensional state | Four-dimensional ankle state |
| Critic observation | Full 44-dimensional state | Full 44-dimensional state |
| Human actor output | 22 muscle activations | 22 muscle activations |
| Exoskeleton actor output | Two ankle torque commands | Two ankle torque commands |

The full 44-dimensional state contains:

- Eight positions: left/right ankle, hip, and knee angles, plus pelvis tilt and height.
- Nine velocities: the corresponding velocities plus forward pelvis velocity.
- Twenty-two muscle activations.
- Four foot-contact force measurements.
- One target walking velocity.

The partial exoskeleton observation contains only the left/right ankle angles and their angular velocities. Only the exoskeleton actor's input is restricted; the human actor and common critic still receive the full state.

#### Shared anatomical and exoskeleton ankle angles

There are no separate exoskeleton joints in the tutorial model. Its two actuators apply torque directly to the musculoskeletal left and right ankle joints. The anatomical ankle angles are therefore also the exoskeleton-controlled angles, and they are included in both the full and partial exoskeleton observations.

#### Policy training target

This depends on the training configuration:

- Human-only, full-observation, partial-observation, and exoskeleton-disabled training use both reference-gait tracking and fixed-velocity tracking at 1.25 m/s. Their reward explicitly combines two distinct tracking errors: **reference-state error**, based on joint-position and joint-velocity differences from the reference gait, and **target-velocity error**, based on pelvis forward-velocity difference from 1.25 m/s. Penalties for muscle activation, excessive foot force, and joint constraint force are added to these tracking terms.
- Speed-controlled training tracks varying commanded velocities. Its reference joint-position and joint-velocity reward weights are zero, so reference-motion tracking does not contribute to its objective.

#### Reference-motion retargeting

The supplied `short_reference_gait.npz` contains full-body joint-position and joint-velocity trajectories. The 22-muscle training environments select only the compatible pelvis, hip, knee, and ankle coordinates from that data. Therefore, the reference is usable in the 22-muscle embodiment's joint-coordinate space, but the repository does not provide an explicit retargeting pipeline or evidence that the motion was retargeted specifically for the combined 22-muscle-plus-exoskeleton model.

There is no separate exoskeleton motion to retarget because the tutorial exoskeleton applies torque directly to the human model's ankle joints. RL learns the 22 muscle activations and two exoskeleton torque commands while tracking the selected reference coordinates.

## Question 2

**Does MyoAssist have any OSL-related training pipelines other than OSL models?**

### Discussion

No. MyoAssist provides MuJoCo models and simulation assets for the Open Source Leg (OSL), but it has no OSL-specific:

- Training environment or pipeline
- Reward definition
- JSON training configuration
- Training command

The existing custom PPO configurations target the tutorial exoskeleton. An OSL training pipeline would therefore need to be created by adapting the environment, observations/actions, rewards, and configuration to an OSL model.

#### Related OSL training repositories

OSL-specific training support is available outside MyoAssist:

- [MyoHub/myosuite](https://github.com/MyoHub/myosuite) provides the `myoChallengeOslRunFixed-v0` and `myoChallengeOslRunRandom-v0` environments, including OSL observations, control logic, termination conditions, and rewards.
- [MyoHub/myochallenge_2024eval](https://github.com/MyoHub/myochallenge_2024eval) provides the MyoChallenge 2024 training/submission scaffold and documents Stable-Baselines3 and DEP-RL support.

The default reward is defined in MyoSuite's [`run_track_v0.py`](https://github.com/MyoHub/myosuite/blob/main/myosuite/envs/myo/myochallenge/run_track_v0.py). Its weighted terms are:

- `sparse` with weight `1`: forward progress, implemented as forward root velocity.
- `solved` with weight `10`: bonus for reaching the end of the track.

The environment also calculates muscle-effort regularization (`act_reg`) and joint-limit pain (`pain`), but neither has a nonzero default reward weight. They must be explicitly enabled or used in a custom reward. The default OSL task does not include reference-motion tracking.

The published first-place MyoChallenge 2024 locomotion solution used PPO, adversarial motion priors, and curriculum training. Its trained networks were referenced publicly, but the paper states that the actual PPO/imitation training environment remained confidential. Therefore, the official MyoSuite repositories provide the strongest available open foundation, but not a complete reproducible winning training recipe.

## Question 3

**What do the three musculoskeletal model families in MyoAssist's `models/` folder represent?**

### Discussion

The numbers in the folder names denote the total biological muscle actuators in each original base model, not degrees of freedom. The three families trade simulation simplicity for anatomical and three-dimensional detail.

| Feature | `22muscle_2D` | `26muscle_3D` | `80muscle` family |
|---|---|---|---|
| Base muscle count | 22 | 26 | 80 |
| Base muscles per leg | 11 | 13 | 40 |
| Locomotion | Planar constrained | Full 3D | Full 3D |
| Anatomical detail | Simplified | Reduced | Highest |
| Lateral balance | Not modeled actively | Required | Required |
| OSL variant | Ankle only | Ankle only | Knee and ankle |
| Muscles remaining in OSL variant | 18 | 22 | 54 |
| OSL electric actuators | One ankle motor | One ankle motor | Knee and ankle motors |
| OSL training configuration | None | None | None |

#### 3D appearance versus 2D locomotion

The `22muscle_2D` models contain three-dimensional bodies and meshes and are displayed in MuJoCo's 3D world. Their lower-body joint structure nevertheless constrains locomotion to a plane: the pelvis can move forward/backward and vertically and can tilt, while the hips, knees, and ankles primarily flex and extend in the sagittal plane. The model cannot actively step sideways, rotate the hips out of plane, or balance laterally. Its appearance is 3D, but its permitted walking dynamics are 2D.

The `26muscle_3D` family adds full pelvis translation/rotation, hip adduction/abduction and rotation, and a hip abductor and adductor on each leg. It therefore supports genuine out-of-plane dynamics and lateral balance. The `80muscle` family is also fully 3D but uses a much more anatomically detailed set of 40 muscles per leg.

#### OSL variants

All three families include an OSL-integrated variant, but not every XML in each family contains an OSL:

- `models/22muscle_2D/myoLeg22_2D_OSL_A.xml` is the planar model with an OSL ankle. Replaced biological structures reduce the model from 22 to 18 muscles, and one electric ankle actuator is added.
- `models/26muscle_3D/myoLeg26_OSL_A.xml` is the reduced 3D model with an OSL ankle. It contains 22 remaining muscles and one electric ankle actuator.
- `models/80muscle/myoLeg80_OSL_KA/myolegs_OSL_KA.xml` is derived from the detailed 80-muscle model and represents a transfemoral amputation. It removes 26 right-leg muscles, leaving 54 biological muscles, and adds electric OSL knee and ankle actuators.

Thus, `80muscle` in the folder name identifies the source model family; the final OSL knee-ankle model has **54 biological muscles plus two OSL motors**, not 80 active muscles. None of the three OSL variants has a dedicated training environment, reward, or JSON configuration in MyoAssist.

## Question 4

**If an OSL policy were trained with the `80muscle` model, how would it be deployed on a physical OSL prosthesis and how would walking begin?**

### Discussion

Only the OSL actor would be deployed. The simulated muscle actor represents the biological human and cannot be deployed; the wearer's nervous system and remaining muscles take its place. For the `80muscle` OSL knee-ankle variant, the deployed actor would output commands for the OSL knee and ankle motors.

MyoAssist does not currently contain an OSL training or hardware-deployment pipeline. The following observation design would therefore have to be implemented during training and reproduced exactly on the hardware.

| Full OSL actor | Partial OSL actor |
|---|---|
| Human lower-body joint angles and velocities | OSL knee and ankle angles |
| Muscle activations | OSL knee and ankle angular velocities |
| Foot-contact forces | — |
| Pelvis state and target velocity | — |

A partial actor analogous to MyoAssist's tutorial design would receive:

```text
[OSL knee angle, OSL ankle angle,
 OSL knee angular velocity, OSL ankle angular velocity]
```

These values would come from the OSL joint encoders and must have the same order, units, normalization, and update rate used during training. The actor's outputs would be converted into bounded knee and ankle torque commands and passed through the hardware's low-level and safety controllers. A full actor is considerably harder to deploy because muscle activation, complete human kinematics, and contact states are not all directly available without additional sensors or estimators.

#### Walking initiation and control flow

The prosthesis controller starts first in a safe standing or impedance-control mode, but the human initiates locomotion. The wearer shifts body weight onto the support leg and moves the residual thigh forward using the hip through the socket. Sensors or an intent detector recognize unloading, thigh motion, or changing prosthesis state and transition the controller into walking control.

```text
Power on and calibrate sensors
→ safe standing controller
→ human shifts weight and initiates a step
→ gait-start detector recognizes intent
→ OSL actor produces knee and ankle commands
→ low-level controller safely applies them
→ repeat the sensing-and-control loop
→ return to standing control when walking stops
```

An actor observing only OSL knee and ankle angles and velocities may recognize gait initiation too late, particularly before the prosthesis has begun moving. A practical system should therefore use a separate gait-start transition controller or include another measurable intent signal such as prosthesis load or thigh IMU motion. A policy trained only for steady-state walking should not be assumed to handle standing, starting, and stopping unless those transitions were included during training.

#### Related OSL publication

[Azocar et al., *Design and clinical implementation of an open-source bionic leg* (2020)](https://www.nature.com/articles/s41551-020-00619-3) demonstrates this general sensor-driven control structure on physical OSL hardware. It uses a standing controller and divides walking into early-to-mid stance, late stance, swing flexion, and swing extension; transitions are determined from mechanical sensors, including joint encoders and a load sensor. The paper uses finite-state impedance control rather than the proposed MyoAssist RL actor, so it supports the hardware walking flow but is not evidence of an existing MyoAssist OSL deployment.
