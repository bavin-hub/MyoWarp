# Selecting the physical right-leg anatomy

`--anatomy constrained` (default) keeps the original biological right leg,
knee constrained at 90 degrees, with the mounted CMU prosthesis.
`--anatomy removed` selects the separate XML without the biological right knee
and distal bodies. Both physical models use the full biological policy shadow.

The new asset is
`assets/robots/myofullbody_cmu_leg_no_lower_leg/myofullbody_cmu_leg_no_lower_leg.xml`.
The original remains
`assets/robots/myofullbody_cmu_leg/myofullbody_cmu_leg.xml`.

Name-based mappings copy only surviving physical state to the shadow. Missing
joints are initialized from reference and then passed through the chosen fooled
or virtual leg mapping. Missing muscles' activation/excitation observations are
zero; their full-policy outputs are not sent to physics. All other mapped
muscle outputs remain available, with the existing lower-leg mask preserved.
The shadow still has 354 muscles and 2418 policy observations. The prosthesis
baseline actor has 115 inputs; the critic has 737 for constrained anatomy and 671
for removed anatomy. Separate physical references are cached for each XML.

Removing the lower leg removes 22 distal-attaching muscle actuators, including
seven hip-crossing muscles. Their tendons are not automatically reattached to
the thigh. Total mass drops from 89.515 to 84.483 kg; surviving inertials and
prosthesis geometry/controller parameters are unchanged. This is a different
physical plant and needs its own prosthesis training run/checkpoint.

## Commands

Prepare and check the removed model:

```bash
.venv-musclemimic/bin/python scripts/prosthesis_rl.py prepare --anatomy removed
.venv-musclemimic/bin/python scripts/prosthesis_rl.py check \
  --anatomy removed --right-leg-obs virtual --num-envs 2
```

Train with the selected anatomy and shadow mode:

```bash
.venv-musclemimic/bin/python scripts/prosthesis_rl.py train \
  --anatomy removed --right-leg-obs virtual --device cuda:0 \
  --num-envs 4096 --iterations 10000 --save-interval 100 \
  --output logs/musculoskeletal_prosthesis/removed_virtual
```

Evaluate the corresponding checkpoint:

```bash
.venv-musclemimic/bin/python scripts/prosthesis_rl.py eval \
  --anatomy removed --right-leg-obs virtual --num-envs 1 \
  --checkpoint logs/musculoskeletal_prosthesis/removed_virtual/model_9999.pt \
  --steps 3000 --viewer
```

Replace `removed` with `constrained` and use the matching checkpoint/output
folder for the existing model. `--reference` can explicitly override the cache
path. The default muscle policy remains the configured Vyvas checkpoint; use
`--human-checkpoint assets/musclemimic/checkpoints/mm-10m-2` for the original
checkpoint. Training still defaults to fooled shadow mode unless specified;
evaluation defaults to virtual. Geometry selection does not change those defaults.

Do not reuse a constrained model's critic/optimizer checkpoint for removed-leg
training: its state dimensions and physical dynamics differ. CLI preflight
reports mismatched anatomy/reference configuration before loading weights.

## Validation

Both variants have prepared full 3001-frame walking references. Native tests
verify removed bodies/joints, retained inertials, shadow mappings, zero missing
muscle channels and safe physical command routing. CUDA tests cover both shadow
modes, finite real-controller steps, correct observation dimensions, PPO update
and checkpoint restore for the removed variant, plus existing constrained-task
regressions. These are integration tests, not learned walking validation.
