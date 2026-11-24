# Controller provenance

`sea_controller.py` ports the user's local CMU leg control equations from
`cmu_leg/src/cmu_leg/control/low_level_control.py`, `control/low_level/knee.py`,
`control/low_level/ankle.py`, and `control/utils.py`. The simulated drive integral
comes from `prosim/prosim/sim_leg.py` (`MujocoLeg.apply`).

`sea_parameters.json` snapshots the source parameters and SHA-256 digests.
These sibling repositories are reference sources only, with no runtime imports.

Preserved details include soft/hard limit asymmetries, conditioning defaults,
Butterworth transposed-direct-form-II updates, backward differences and dt
clamps, ankle feedforward without efficiency division, per-world enable/actuate
edges, hardware-enable mute and the simulated drive's clamped integrator.
Hardware encoder/PDO conversions, EtherCAT, estimation and high-level gait
controllers are not part of this simulation controller.

The 16.8 Hz integral corner used for both drives is ProSim's existing assumption:
its comments identify the PI convention as inferred and the knee value as reused
from the ankle. This port preserves that simulation assumption; it does not
establish equivalence to measured motor-drive dynamics.
