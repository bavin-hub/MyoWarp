# H1 with OSL: dual-policy partial-observation velocity tracking

Task #4 uses the same environment, action split, shared critic, and joint PPO
optimization as Task #3. Only the OSL actor observation changes.

| Component | Observation | Output |
|---|---:|---:|
| H1 actor | Full 89-value actor state | 24 non-OSL actions |
| OSL actor | Four encoder-derived values | OSL knee and ankle actions |
| Shared critic | Full 104-value privileged state | State value |

The OSL input order is:

```text
[OSL knee position relative to home,
 OSL ankle position relative to home,
 OSL knee angular velocity,
 OSL ankle angular velocity]
```

These values are selected from the same noisy joint observations used by the H1
actor. The OSL actor does not receive the velocity command, base state, gait
phase, other joint states, contacts, previous actions, or critic-only state.

The Task #1 reward terms and weights, command curriculum, contacts,
terminations, domain randomization, control targets, and MuJoCo action
permutation remain unchanged. PPO and rollout storage also remain unchanged;
the additive `DualMLPModel` sums both actors' log-probabilities, entropy, and KL.

## Train and evaluate

```bash
python scripts/train.py h1_with_osl_dual_partial_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001
```

Runs are written under
`logs/rsl_rl/h1_with_osl_dual_partial_velocity_standalone/`.

```bash
python scripts/eval.py h1_with_osl_dual_partial_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_partial_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer
```

The ONNX policy exposes two inputs: `primary_obs` with 89 values and
`secondary_obs` with four values. Its output is the assembled 26-action vector.
Task #3 checkpoints are not structurally compatible because the OSL actor input
dimension changed from 89 to four.
