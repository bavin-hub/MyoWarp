# H1 with OSL: dual-policy full-observation motion tracking

Task #5 combines Task #2 motion tracking with Task #3's dual-actor PPO model.

| Component | Observation | Output |
|---|---:|---:|
| H1 actor | Full 161-value tracking observation | 24 non-OSL actions |
| OSL actor | Full 161-value tracking observation | OSL knee and ankle actions |
| Shared critic | Full 287-value privileged observation | State value |

The two actors have independent observation normalizers, MLPs, Gaussian means,
and learnable standard deviations. Their outputs are assembled into the
canonical 26-joint action vector. RSL-RL uses one PPO ratio based on their summed
log-probabilities and updates both actors and the shared critic using the same
advantage.

The retargeted 4,490-frame reference, nine motion rewards and weights,
self-contact handling, tracking terminations, randomized motion resets, adaptive
failure-weighted frame sampler, domain randomization, evaluation metrics, and
motion-bundled ONNX export are inherited unchanged from
`h1_with_osl_tracking`.

## Train and evaluate

```bash
python scripts/train.py h1_with_osl_dual_full_tracking \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 30000
```

Runs are written under
`logs/rsl_rl/h1_with_osl_dual_full_tracking_standalone/`.

```bash
python scripts/eval.py h1_with_osl_dual_full_tracking \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_full_tracking_standalone/RUN/model_ITERATION.pt \
  --start-frame 0 \
  --deterministic \
  --viewer
```

The ONNX export accepts the 161-value observation and a motion time step, then
returns the assembled actions and bundled reference-motion values. Task #2
single-actor checkpoints are not structurally compatible with this dual actor.
