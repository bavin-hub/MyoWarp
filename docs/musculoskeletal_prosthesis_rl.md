# Musculoskeletal + CMU prosthesis pipeline

This pipeline combines a pretrained whole-body MuscleMimic policy with a learned
CMU knee/ankle policy. Runtime code is local under
`myowarp/tasks/musculoskeletal_prosthesis/` and `myowarp/musclemimic/`; it does not
call the sibling ProSim or CMU repositories.

- [Eight experiments](musculo_prosthesis_8_experiments.md): definitive actor,
  critic, observation, reward and optimization differences.
- [Commands](commands.md): preparation, individual/sequential training,
  best-checkpoint evaluation, measurements and recording.
- [Anatomy variants](prosthesis_anatomy_variants.md): constrained versus removed
  biological lower-leg models.
- [MuscleMimic setup](musclemimic.md): pretrained artifacts and source motions.

## Physical and virtual models

The retained eight experiments use
`assets/robots/myofullbody_cmu_leg_no_lower_leg/myofullbody_cmu_leg_no_lower_leg.xml`.
It contains 332 retained muscle actuators and four CMU motor channels. The
alternative constrained model keeps the biological right knee bent at 90 degrees.
These are different physical plants and require their own adapted references
and prosthesis checkpoints.

A full 354-muscle biological shadow preserves the original muscle-policy
observation contract. It copies retained real body state. Virtual mode maps
actual prosthesis knee/ankle state into the biological observation, projects
knee followers and zeros the missing subtalar/toe state. Missing muscle
activation/excitation observations are zero. Remaining derived muscle/site
features are recomputed rather than forced to zero. The shadow supplies
observations; it does not simulate a second free-running physical person.

Fooled mode substitutes reference lower-leg kinematics instead. The retained
eight experiments use virtual mode. Physical muscle commands always route only
to actuators present in the real model.

## Control and observations

Both policies act at **100 Hz**. Muscle outputs are applied to the retained
muscle actuators; prosthesis outputs are two desired joint torques. Ten **1 ms**
controller/physics ticks follow each policy action. The local SEA controller
supplies motor velocity targets and feedforward-plus-integral torque; MuJoCo's
velocity actuators supply proportional feedback. There is no direct-torque
bypass of the SEA dynamics.

The current baseline prosthesis actor has **115 inputs**: five 23-value sensor
history frames, including hip RPY relative to pelvis, thigh world orientation
and angular velocity, pelvis-frame gravity, CMU knee/ankle q/qvel, prosthetic
contacts and previous actions. It does not observe the left foot, SEA deflection,
explicit pelvis world orientation/angular velocity, or a reference. Exp6 adds
21 reference/phase values, giving 136. Internal/legacy sensor formats are not
the current deployment actor layout.

The muscle observation has **2,418 values**. The privileged prosthesis critic
has **671 values** for the removed-leg model. Separate-PPO experiments also use
a pretrained muscle critic with the full virtual muscle observation.

## Reference and reward

The source stitched walking clip is 30 seconds at 100 Hz. Offline constrained
reference fitting uses the actual prosthesis geometry, joint limits and closed
four-bar linkage. The biological reference remains unchanged. Adapted physical
reference poses and differentiated velocities are used for initialization,
privileged critic inputs and the applicable physical-space reward terms.

Exp1–3 use the combined-model reward with one privileged critic. Exp4–7 use the
masked MuscleMimic reward and prosthesis-specific terms, delivered separately
except for Exp5's shared scalar sum. Exp8 replaces physical-foot tracking with
CMU-to-human mapped lower-leg tracking; its muscle reward remains unchanged.
See the [eight-experiment guide](musculo_prosthesis_8_experiments.md) for exact
weights, mapping conventions and training safeguards.

Measurements use the real physical state before automatic reset. Episode
completion, duration, global drift and local pose tracking answer different
questions and should be inspected together.

## Runtime and artifacts

Use `.venv-musclemimic` with MuJoCo/Warp 3.11, Torch and CUDA. Simulation, shadow
forward passes, policy inference, sensing, rewards, PPO and the 1 kHz controller
run on GPU. Native MuJoCo is used for offline preparation, reset filtering,
viewing and numerical oracle tests. The CPU evaluator for H1 + CMU is a separate
pipeline, not a CPU musculoskeletal backend.

The main entry points are `scripts/prosthesis_rl.py` (Exp1–3) and
`scripts/prosthesis_split_rl.py` (Exp4–8). All-eight training uses
`scripts/run_eight_prosthesis_experiments.sh`. Scripts restore saved configuration
and reject incompatible physical-model or learner changes.

Exp1–3 keep `model_best.pt`; Exp4–8 keep best muscle-selected and
prosthesis-selected paired snapshots. Each paired file contains both learners
from the same instant. Best selection uses training return, not deterministic
tracking error. Only best checkpoints remain after cleanup; all current commands
use these files. Checkpoints, downloaded artifacts and logs are excluded from
Git and must be supplied or regenerated on a fresh machine.

## Validation

The retained tests exercise physical/shadow mapping, reference fitting,
observation schemas, independent/joint PPO, checkpoint compatibility and CPU/GPU
SEA-controller parity. Recorded source reward/controller oracles are described
in [tests/fixtures/README.md](../tests/fixtures/README.md). Only fixture regeneration
reads sibling source repositories; ordinary runtime and tests use local files.

GPU tests need CUDA and local model/motion/checkpoint artifacts. Short tests
validate implementation contracts, not long-horizon learning quality.
