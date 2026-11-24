# Experiment 4 source provenance

The standalone masked MuscleMimic reward and fine-tuning workflow are ported
from the user's ProSim repository, commit
`5c58afd545201a417e12a82b5060d6e9da9d6818`:

- `prosim/rl/mimic_reward.py` → `muscle_reward_native.py` (offline oracle),
  `muscle_reward_gpu.py` (batched Torch implementation).
- `prosim/rl/policy.py`, `ppo.py`, `env.py` and `scripts/train_bypass.py` →
  local pretrained network loading, fixed normalization, critic warm-up,
  bounded adaptive learning rate and reward scaling.

Original `mimic_reward.py` SHA256:
`9011572d9ea56aeb28b8611251d4ed10db90f3a2592dd80ad7261ef3f33f69cc`.

ProSim attributes the reward math to MuscleMimic's
`musclemimic/core/reward/trajectory_based.py`,
`musclemimic/core/terminal_state_handler/enhanced_fullbody.py`, and loco-mujoco's
`loco_mujoco/core/utils/math.py` (Apache-2.0). The local MyoSuite actor module
retains its existing Apache-2.0 notice and license.

No runtime import or API call targets ProSim, CMU-leg or MyoSuite. The GPU
environment, separate prosthesis objective and two-learner runner are local
implementations. The analytical high-level prosthesis controller is deliberately
replaced by the separately trained prosthesis actor; the existing local GPU
low-level controller remains.
