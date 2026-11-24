# MyoFullBody + CMU, without the biological right lower leg

`myofullbody_cmu_leg_no_lower_leg.xml` is a separate MuJoCo 3.11 asset derived
from the existing constrained-leg XML. The original XML is not modified.

Removed: biological right tibia, talus, calcaneus, toes and patella; their 11
joint coordinates; dependent tendons, 22 muscle actuators, sensors, equalities
and explicit contact pairs. The right femur/hip and the entire mounted CMU
prosthesis, its four actuators, SEA tendons and four-bar linkage remain.
Surviving body masses/inertias and the prosthesis mount are preserved exactly.

| Quantity | Constrained | Removed |
|---|---:|---:|
| qpos / qvel | 95 / 94 | 84 / 83 |
| Physical muscles | 354 | 332 |
| Total actuators | 358 | 336 |
| Total mass | 89.515 kg | 84.483 kg |

The 22 removed muscles include seven hip-crossing muscles whose tendons attach
to deleted segments. They are removed rather than reattached arbitrarily.
No stump/socket or surgical tendon reconstruction is modeled. `anatomy.json`
records all removed names.

The policy shadow retains the complete original biological model and all 354
muscle observation/output slots. Surviving joints/muscles map by name; absent
muscles have zero activation/excitation in the shadow and their predicted
commands are discarded. Fooled and virtual kinematics work as before.

Regenerate and inspect:

```bash
.venv-musclemimic/bin/python scripts/build_myofullbody_cmu_leg_no_lower_leg.py
.venv-musclemimic/bin/python test_scripts/view_myofullbody_cmu_leg.py --anatomy removed
```

The static viewer also accepts `--anatomy constrained` (default). Mesh assets
are shared through relative paths; no mesh copies or sibling-repo imports are
needed. The `bypass_stand` keyframe is remapped by surviving joint names.
