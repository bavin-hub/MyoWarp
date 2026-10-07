# H1 with OSL: dual-policy partial-observation motion tracking

Task #6 uses Task #5's dual motion policy while restricting the OSL actor to
four encoder-derived values.

| Component | Observation | Output |
|---|---:|---:|
| H1 actor | Full 161-value tracking state | 24 non-OSL actions |
| OSL actor | Four OSL joint values | OSL knee and ankle actions |
| Shared critic | Full 287-value privileged state | State value |

The OSL observation order is:

```text
[OSL knee position relative to home,
 OSL ankle position relative to home,
 OSL knee angular velocity,
 OSL ankle angular velocity]
```

The four values are selected from the same noisy, encoder-biased joint readings
inside the H1 actor observation. The OSL actor does not receive the reference
motion command, anchor or base state, other joints, contacts, previous actions,
or critic-only information.

The reference clip, nine motion rewards and weights, self-contact handling,
terminations, adaptive failure-weighted sampler, domain randomization,
evaluation metrics, action permutation, and joint PPO optimization are inherited
unchanged from Tasks #2 and #5.

## Train and evaluate

```bash
python scripts/train.py h1_with_osl_dual_partial_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000
```

Runs are written under
`logs/rsl_rl/h1_with_osl_dual_partial_tracking_standalone/`.

```bash
python scripts/eval.py h1_with_osl_dual_partial_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_partial_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer
```

The motion-bundled ONNX policy exposes `primary_obs` with 161 values,
`secondary_obs` with four values, and `time_step`. Task #5 checkpoints are not
structurally compatible because the OSL actor input changes from 161 to four.
