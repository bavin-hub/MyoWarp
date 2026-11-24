"""Pretrained normalization, bounded adaptation and critic-only warmup."""

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel


class FixedNormalizationMLP(MLPModel):
    def update_normalization(self, obs):
        # Preserve pretrained actor AND critic statistics during fine-tuning.
        pass


class FinetunePPO(PPO):
    def __init__(self, *args, critic_warmup=20, lr_range=(1e-6, 1e-4), **kwargs):
        self.lr_range = lr_range
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
            self.actor.requires_grad_(True)
        self.n_updates += 1
        losses["critic_warmup"] = float(warmup)
        return losses

    def save(self):
        result = super().save()
        result["finetune_updates"] = self.n_updates
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("iteration", False):
            self.n_updates = loaded_dict.get("finetune_updates", 0)
        return result
