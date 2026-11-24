"""Standalone GPU port of ProSim's pretrained actor/critic and fine-tune PPO.

Workflow source: ProSim 5c58afd, prosim/rl/{policy,ppo}.py.
Uses local MuscleMimic network/artifacts, with no ProSim or MyoSuite imports.
"""

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution

from myowarp.musclemimic.actor_torch import MimicActorModule

from .artifacts import frozen_artifacts


class PretrainedMimicModel(MLPModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        checkpoint_path,
        part,
        muscle_indices=None,
        std_scale=0.5,
        distribution_cfg=None,
    ):
        torch.nn.Module.__init__(self)
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = False
        self.obs_normalizer = torch.nn.Identity()
        artifacts, _ = frozen_artifacts(checkpoint_path)
        if part not in artifacts.params:
            raise ValueError(
                "Muscle fine-tuning requires an original checkpoint containing actor AND critic"
            )
        self.mlp = MimicActorModule(
            {"actor": artifacts.params[part]}, artifacts.obs_mean, artifacts.obs_var
        ).requires_grad_(True)
        if self.obs_dim != self.mlp.obs_dim:
            raise ValueError("Pretrained observation dimension mismatch")
        ids = muscle_indices if part == "actor" else [0]
        if ids is None or len(ids) != output_dim:
            raise ValueError("Pretrained output/physical muscle mapping mismatch")
        self.register_buffer("output_indices", torch.tensor(ids, dtype=torch.long))
        self.distribution = None
        if distribution_cfg is not None:
            self.distribution = GaussianDistribution(output_dim, std_type="log")
            log_std = torch.tensor(np.array(artifacts.params["log_std"], copy=True))
            with torch.no_grad():
                self.distribution.log_std_param.copy_(
                    log_std[self.output_indices] + np.log(std_scale)
                )

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        # Fused checkpoint normalization is fixed for BOTH networks, as in ProSim.
        value = self.mlp(self.get_latent(obs, masks, hidden_state))[
            :, self.output_indices
        ]
        if self.distribution is not None and stochastic_output:
            self.distribution.update(value)
            return self.distribution.sample()
        return value


class MuscleFinetunePPO(PPO):
    def __init__(self, *args, critic_warmup=20, lr_range=(1e-6, 1e-4), **kwargs):
        self.lr_range = tuple(lr_range)
        self.critic_warmup = critic_warmup
        self.n_updates = 0
        super().__init__(*args, **kwargs)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = min(max(float(value), self.lr_range[0]), self.lr_range[1])

    def update(self):
        warmup = self.n_updates < self.critic_warmup
        if warmup:
            self.actor.requires_grad_(False)
        try:
            losses = super().update()
        finally:
            if warmup:
                self.actor.requires_grad_(True)
        self.n_updates += 1
        losses["critic_warmup"] = float(warmup)
        return losses

    def save(self):
        saved = super().save()
        saved["finetune_updates"] = self.n_updates
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        loaded = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("iteration", False):
            self.n_updates = loaded_dict["finetune_updates"]
        return loaded
