# Whole-body MuscleMimic evaluation

The implementation is ported into `myowarp/musclemimic/` from MyoSuite 3.0.0.
There are no runtime imports of prosim, MyoSuite, or LocoMuJoCo. The body MJCF,
meshes and textures are bundled under `assets/robots/myofullbody/`, with their
licenses. Actor inference and observation/history handling run locally in NumPy;
JAX/Orbax are used to read the downloaded checkpoint. No training is performed.

The pretrained `amathislab/mm-10m-2` actor controls 354 muscles. The bundled
musclemimic-models 1.0.6 assets are built with the upstream 1.0.5 attribute values
matching this checkpoint. Preserve these knee-coupling and wrap adjustments.
The evaluator preserves the biological legs and uses native MuJoCo forward
dynamics. It initializes the pose/velocity from the reference once, then drives
muscles at 100 Hz, with five 2 ms physics steps per action. It does not overwrite
the simulated pose with reference frames or attach the CMU leg.

Use a separate environment so existing H1-with-OSL dependencies stay intact:

```bash
uv venv .venv-musclemimic --python python3.12
uv pip install --python .venv-musclemimic/bin/python -r requirements-musclemimic.txt
```

Request access to the motion dataset at
https://huggingface.co/datasets/amathislab/musclemimic-retargeted, then authenticate
with the approved account. Do not paste access tokens into chat or source code.

```bash
uvx hf auth login
.venv-musclemimic/bin/python scripts/fetch_musclemimic.py
```

This downloads the policy checkpoint and `KIT/314/walking_medium09_poses`. Use
`--clip KIT/167/walking_medium10_poses` (repeatable) to fetch other references.
Weights and motions are ignored by Git, so each checkout downloads them locally.
Downloads happen only in the fetch script; evaluation uses local paths offline.

```bash
.venv-musclemimic/bin/python scripts/eval_musclemimic.py --steps 1000
.venv-musclemimic/bin/python scripts/eval_musclemimic.py --headless --steps 1000
```

For another reference, pass `--clip <path-to-retargeted-npz>`. This checkpoint
expects the MyoFullBody retargeted format with site transforms, body velocities,
subtree COM data, and name/index mappings. Generic AMASS or skeleton-only clips
are not equivalent inputs. The evaluator writes JSON metrics and an NPZ of
simulated positions/actions under `logs/musclemimic/`. The JSON records falls,
non-finite states and early exits rather than reporting them as successful replay.

Without any motion data, inspect the static whole-body model:

```bash
.venv-musclemimic/bin/python scripts/eval_musclemimic.py --view-model
```

Left drag rotates, right drag pans, and scrolling zooms. Add `--headless` to
check model compilation without a display. Run conformance checks with:

```bash
.venv-musclemimic/bin/python -m unittest discover -s tests -p test_musclemimic.py -v
```

These checks use a generated stationary target to test checkpoint loading,
observations/history, actions and integration. They do not validate walking.

The downloaded `KIT/314/walking_medium09_poses` reference has also been evaluated:
all 773 policy steps (7.73 seconds) completed without falling or non-finite states.
This confirms working forward replay for that clip, not perfect tracking or
validation of every motion. See `logs/musclemimic/eval.json` for per-step root drift
and activation measurements. During development, Orbax restore stalled inside
Codex's filesystem sandbox; the same evaluation completed outside that sandbox.

## Long walking references

`myowarp/musclemimic/reference.py` ports prosim's `best_cycle`, `loop_clip`,
kinematic export and joint-coupling projection. It has no prosim imports.
The generator selects steady walking (excluding acceleration/deceleration),
finds a cycle with closely matching joint positions/velocities, corrects the
boundary and heading drift, and advances root translation on each repetition.
It recomputes joint velocities, body/site poses, velocities and subtree COM
features for the policy rather than concatenating stale reference arrays.

```bash
.venv-musclemimic/bin/python scripts/fetch_musclemimic.py \
  --clip KIT/314/walking_medium10_poses
.venv-musclemimic/bin/python scripts/make_loop_clip.py --seconds 30
.venv-musclemimic/bin/python scripts/eval_musclemimic.py \
  --clip assets/musclemimic/motions/loops/walking_medium10_poses_30s.npz \
  --steps 3000
```

Use `--clip <local-npz>` and `--output <destination-npz>` to select other input
and output files. To choose a cycle manually, provide both `--start <frame>` and
`--length <frames>`. Automatic selection on the default source chooses frames
381–492: a 1.11-second cycle advancing approximately 1.22 m per repetition.
Generated clips remain under the ignored motion-data directory. They are new
cyclic references, not original uninterrupted 30-second motion captures; making
a reference longer does not guarantee that the policy will track it successfully.

The default generated reference was evaluated end-to-end: all 3,001 frames
completed without a fall or non-finite state. Horizontal root error reached
3.25 m by the end, so this demonstrates sustained walking rather than accurate
long-term global trajectory tracking. Metrics are saved in
`logs/musclemimic/loop_medium10_30s.json`.

For running, use shorter cycle-search bounds:

```bash
.venv-musclemimic/bin/python scripts/fetch_musclemimic.py \
  --clip KIT/314/walking_run01_poses
.venv-musclemimic/bin/python scripts/make_loop_clip.py \
  --clip assets/musclemimic/motions/MyoFullBody/gmr/KIT/314/walking_run01_poses.npz \
  --seconds 30 --min-cycle 0.4 --max-cycle 0.9
.venv-musclemimic/bin/python scripts/eval_musclemimic.py \
  --clip assets/musclemimic/motions/loops/walking_run01_poses_30s.npz \
  --steps 3000
```

The selected running cycle is frames 144–217 (0.73 s, approximately 1.70 m per
cycle). The 30-second reference generates successfully, but the current policy
fell after 1.76 seconds in the headless test; this loop is not validated for
sustained running. Results: `logs/musclemimic/running_loop_30s.json`.
