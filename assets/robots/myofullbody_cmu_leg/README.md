# MyoFullBody with the CMU leg

`myofullbody_cmu_leg.xml` is a directly loadable MuJoCo 3.11 model, exported
from the repository-local assembly in `myowarp/musclemimic/bypass.py`, ported
from prosim's bent-knee bypass model. There are no prosim or MyoSuite imports.

The biological right knee is held at 90 degrees, its ankle/subtalar/toe joints
are held neutral, and its foot contacts are disabled. The CMU leg is mounted
to the right femur. This represents an able-bodied person wearing a bypass;
the biological leg has not been amputated. The mount is approximately 0.5471 m
below the hip, chosen to level the prosthetic sole with the biological left foot.

The model retains 354 biological muscle actuators and adds the prosthesis's
four motor/velocity actuators, SEA tendons, and four-bar constraint. It includes
the `bypass_stand` keyframe used by the static viewer. No policy or controller
is run by the viewer.

```bash
.venv-musclemimic/bin/python test_scripts/view_myofullbody_cmu_leg.py
.venv-musclemimic/bin/python test_scripts/view_myofullbody_cmu_leg.py --headless
```

Left drag rotates, right drag pans, and scrolling zooms. To regenerate:

```bash
.venv-musclemimic/bin/python scripts/build_myofullbody_cmu_leg.py
```

`prosthesis_source.xml` is copied from prosim/models/prosthesis.xml. Its meshes
are shared with `../cmu_leg/meshes/`; whole-body meshes are shared with
`../myofullbody/model/`. The exported XML uses only repository-relative paths.
Keep these asset directories together when moving the model. Whole-body asset
licenses and notices are retained in `../myofullbody/`.

Export explicitly preserves compiled masses/inertias, normalizes attached
default classes, and avoids conflicting legacy memory limits. The build script
reloads the XML and compares mass, inertia, collision dimensions and masks with
the assembled model before reporting success.
