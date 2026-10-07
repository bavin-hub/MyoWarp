# H1 with OSL: dual-policy full-observation velocity tracking

Task #3 runs the Task #1 MuJoCo-Warp environment with one joint PPO policy that
contains two independent actor MLPs:

- The H1 actor receives the complete 89-value actor observation and outputs the
  24 non-OSL joint actions.
- The OSL actor receives the same complete observation and outputs the OSL right
  knee and ankle actions at canonical policy indices 9 and 10.
- The shared critic receives the 104-value privileged observation.

Each actor owns an independent observation normalizer, MLP, Gaussian mean, and
learnable standard deviation. Their actions are assembled into the existing
26-joint MJLab action order before the environment applies joint offsets/scales
and maps targets into MuJoCo actuator order.

RSL-RL computes a single PPO ratio from the sum of the two actor
log-probabilities. The entropy and KL divergence are also summed, and one
optimizer updates both actors and the shared critic from the same advantage.
Rollout storage and PPO itself are unchanged; `DualMLPModel` is an additional
model implementation in the local RSL-RL checkout.

The reward terms and weights, command curriculum, contacts, terminations,
domain randomization, simulation settings, and observations are inherited
unchanged from `h1_with_osl_velocity`.

## Train and evaluate

```bash
python scripts/train.py h1_with_osl_dual_full_velocity \
  --device cuda:0 \
  --num-envs 4096 \
  --max-iterations 10001
```

Runs are written under
`logs/rsl_rl/h1_with_osl_dual_full_velocity_standalone/`.

```bash
python scripts/eval.py h1_with_osl_dual_full_velocity \
  --checkpoint-file logs/rsl_rl/h1_with_osl_dual_full_velocity_standalone/RUN/model_ITERATION.pt \
  --command 0.5 0.0 0.0 \
  --viewer
```

The exported ONNX policy has one full-observation input and one assembled
26-action output. Checkpoints from the single-actor task are not structurally
compatible with this dual-actor model.
