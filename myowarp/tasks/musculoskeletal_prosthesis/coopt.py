"""Two actors, one joint PPO likelihood and a shared real-state critic.

As in H1 dual-policy PPO, log probabilities sum over both actors' actions.
The muscle network retains its pretrained residual architecture. Normalized
muscle observations are stored in the rollout so PPO replays the exact inputs.
"""

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution

from myowarp.musclemimic.actor_torch import MimicActorModule

from .artifacts import frozen_artifacts


class MuscleProsthesisActor(MLPModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        checkpoint_path,
        muscle_indices,
        **kwargs,
    ):
        super().__init__(obs, obs_groups, obs_set, 2, **kwargs)
        artifacts, metadata = frozen_artifacts(checkpoint_path)
        self.muscle = MimicActorModule.from_artifacts(artifacts)
        if metadata.get("normalization_mode") == "fixed":
            self.muscle.load_state_dict(metadata["actor_state_dict"], strict=True)
        self.muscle.requires_grad_(True)
        self.register_buffer(
            "muscle_indices", torch.tensor(muscle_indices, dtype=torch.long)
        )
        if output_dim != 2 + len(muscle_indices):
            raise ValueError(
                "Joint action dimension differs from physical muscle mapping"
            )
        self.distribution = GaussianDistribution(output_dim, init_std=0.2)
        with torch.no_grad():
            self.distribution.std_param[2:] = 0.03

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        prosthesis = self.mlp(self.get_latent(obs, masks, hidden_state))
        muscle = self.muscle.forward_normalized(obs["muscle"]).clamp(-1, 1)
        mean = torch.cat((prosthesis, muscle[:, self.muscle_indices]), dim=-1)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean


class CooptPPO(PPO):
    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        if not env.cfg.train_muscle:
            raise ValueError("CooptPPO requires train_muscle")
        cfg["actor"].update(
            class_name="myowarp.tasks.musculoskeletal_prosthesis.coopt:MuscleProsthesisActor",
            checkpoint_path=env.cfg.checkpoint_path,
            muscle_indices=env.trainable_muscle_ids.cpu().tolist(),
        )
        # Upstream adaptive PPO overwrites every group's LR with one value.
        # Fixed scheduling preserves the explicit low muscle learning rate.
        cfg["algorithm"]["schedule"] = "fixed"
        alg = super().construct_algorithm(obs, env, cfg, device)
        slow = list(alg.actor.muscle.parameters()) + list(
            alg.actor.distribution.parameters()
        )
        slow_ids = {id(p) for p in slow}
        fast = [p for p in alg.actor.parameters() if id(p) not in slow_ids]
        fast += list(alg.critic.parameters())
        alg.optimizer = torch.optim.Adam(
            [
                {
                    "params": fast,
                    "lr": alg.learning_rate,
                    "name": "prosthesis_and_critic",
                },
                {
                    "params": slow,
                    "lr": env.cfg.muscle_learning_rate,
                    "name": "muscle_and_exploration",
                },
            ]
        )
        return alg
