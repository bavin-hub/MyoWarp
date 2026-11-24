"""Evaluation-only compatibility for policies trained with left-foot contact.

The unavailable sensor is a fixed zero placeholder, never a simulation reading.
This preserves old network dimensions, not the original observation distribution.
New training uses 145 actual inputs and does not use this adapter.
"""

import torch


def pad_unavailable_left_contact(observations, history_steps):
    out = observations.clone()
    actor = out["actor"]
    history = actor.reshape(actor.shape[0], history_steps, 29)
    left_zero = torch.zeros_like(history[:, :, :1])
    out["actor"] = torch.cat(
        (history[:, :, :25], left_zero, history[:, :, 25:]), -1
    ).flatten(1)
    return out


class LegacyEvaluationAdapter:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return pad_unavailable_left_contact(
            self.env.get_observations(), self.cfg.history_steps
        )

    def step(self, actions):
        obs, reward, done, extras = self.env.step(actions)
        return (
            pad_unavailable_left_contact(obs, self.cfg.history_steps),
            reward,
            done,
            extras,
        )


def adapt_legacy_checkpoint(env, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    width = saved["actor_state_dict"]["mlp.0.weight"].shape[1]
    current = env.get_observations()["actor"].shape[1]
    if width == current:
        return env
    if width == env.cfg.history_steps * 30 and current == env.cfg.history_steps * 29:
        print(
            "Legacy 150-input prosthesis policy: unavailable left-foot contact is fixed to zero in every history frame. Evaluation only; retrain the 145-input policy for deployment.",
            flush=True,
        )
        return LegacyEvaluationAdapter(env)
    raise ValueError(
        f"Prosthesis checkpoint expects {width} observations; current task supplies {current}"
    )
