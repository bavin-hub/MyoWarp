# SEA controller numerical oracle

`sea_controller_oracle.npz` records 1600 ticks for three independent instances of
the original scalar `cmu_leg.control.low_level_control.LowLevelControl`.
It includes desired-torque steps, soft/hard joint-limit violations, startup mute,
enable/actuate edges, a standing interval, and an already-enabled walking start.
The drive-integral outputs use the original ProSim update equation and gains
read from the combined model. Units and array layout are documented by the
fixture generator; q/v columns are knee, knee_motor, ankle, ankle_motor,
ankle_drive. Output columns are knee and ankle.

Regenerate explicitly with:

```bash
.venv-musclemimic/bin/python test_scripts/generate_sea_controller_fixture.py \
  --cmu-source ../cmu_leg/src
```

Only this development tool reads the sibling source. Runtime and tests use the
local parameter snapshot and recorded fixture. Source SHA-256 digests are in
`myowarp/tasks/musculoskeletal_prosthesis/sea_parameters.json`.

# Experiment 4 MuscleMimic reward oracle

`exp4_mimic_reward.npz` records five reward/termination evaluations from the
original ProSim `prosim/rl/mimic_reward.py`, commit
`5c58afd545201a417e12a82b5060d6e9da9d6818`, SHA256
`9011572d9ea56aeb28b8611251d4ed10db90f3a2592dd80ad7261ef3f33f69cc`.
It contains the first 40 reference poses/velocities from this repo's walking
loop and native MuJoCo data fields at reference frames 2, 15, 31, 20 and 10.
Cases cover an exact reference pose, two perturbed body states, a 1.5 m root
translation (termination), and changes confined to masked right ankle/toe
coordinates. Random generator seed 2026 generates perturbations and 332 raw
muscle commands, including values outside actuator bounds. `terms` columns are
named by `term_names`; `rewards` are unscaled; `terminated` are original boolean
termination results. The standalone Torch port is compared against these fixed
source results without importing ProSim.
