from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class NetworkIndexHandler:
    def __init__(self, net_indexing_info: dict[str, Any], obs_size: int, action_size: int):
        self.net_indexing_info = net_indexing_info
        self.obs_size = obs_size
        self.action_size = action_size

    def observation_num(self, net_name: str) -> int:
        total = 0
        for item in self.net_indexing_info[net_name]["observation"]:
            if item["type"] == "range":
                start, end = item["range"]
                total += end - start
            elif item["type"] == "index":
                total += len(item["index"])
        return total

    def action_num(self, net_name: str) -> int:
        total = 0
        for item in self.net_indexing_info[net_name].get("action", []):
            if item["type"] == "range_mapping":
                start, end = item["range_net"]
                total += end - start
            elif item["type"] == "index_mapping":
                total += len(item["index"])
        return total

    def obs_for(self, obs: torch.Tensor, net_name: str) -> torch.Tensor:
        parts = []
        for item in self.net_indexing_info[net_name]["observation"]:
            if item["type"] == "range":
                start, end = item["range"]
                parts.append(obs[..., start:end])
            elif item["type"] == "index":
                parts.append(obs[..., item["index"]])
        return torch.cat(parts, dim=-1)

    def map_actions(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = torch.zeros(
            (next(iter(outputs.values())).shape[0], self.action_size),
            device=next(iter(outputs.values())).device,
            dtype=next(iter(outputs.values())).dtype,
        )
        for net_name, output in outputs.items():
            for item in self.net_indexing_info[net_name].get("action", []):
                if item["type"] == "range_mapping":
                    net_start, net_end = item["range_net"]
                    action_start, action_end = item["range_action"]
                    actions[:, action_start:action_end] = output[:, net_start:net_end]
                elif item["type"] == "constant":
                    action_start, action_end = item["range_action"]
                    actions[:, action_start:action_end] = item["default_value"]
        return actions


def _mlp(input_dim: int, hidden: list[int], output_dim: int, output_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for dim in hidden:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(nn.Tanh())
        last_dim = dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class HumanExoActorCritic(nn.Module):
    """Separated human/exo actors plus shared critic, matching the JSON config."""

    def __init__(self, obs_size: int, action_size: int, custom_policy_params: dict[str, Any]):
        super().__init__()
        self.net_arch = custom_policy_params["net_arch"]
        self.index = NetworkIndexHandler(
            custom_policy_params["net_indexing_info"],
            obs_size=obs_size,
            action_size=action_size,
        )
        self.human_actor = _mlp(
            self.index.observation_num("human_actor"),
            self.net_arch["human_actor"],
            self.index.action_num("human_actor"),
        )
        self.exo_actor = _mlp(
            self.index.observation_num("exo_actor"),
            self.net_arch["exo_actor"],
            self.index.action_num("exo_actor"),
        )
        self.critic = _mlp(
            self.index.observation_num("common_critic"),
            self.net_arch["common_critic"],
            1,
            output_activation=False,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        human_obs = self.index.obs_for(obs, "human_actor")
        exo_obs = self.index.obs_for(obs, "exo_actor")
        value_obs = self.index.obs_for(obs, "common_critic")
        human_action = self.human_actor(human_obs)
        exo_action = self.exo_actor(exo_obs)
        action = self.index.map_actions({"human_actor": human_action, "exo_actor": exo_action})
        value = self.critic(value_obs).squeeze(-1)
        return action, value


class GaussianActorCritic(nn.Module):
    """Torch-native PPO policy compatible with the original custom policy config."""

    def __init__(self, obs_size: int, action_size: int, policy_params: dict[str, Any]):
        super().__init__()
        custom_params = policy_params.get("custom_policy_params", policy_params)
        net_info = custom_params.get("net_indexing_info", {})
        self.action_size = action_size
        self.log_std = nn.Parameter(
            torch.ones(action_size, dtype=torch.float32) * float(custom_params.get("log_std_init", 0.0))
        )
        if "exo_actor" in net_info:
            self.core: nn.Module = HumanExoActorCritic(obs_size, action_size, custom_params)
        else:
            net_arch = custom_params.get("net_arch", {})
            self.actor = _mlp(obs_size, net_arch.get("human_actor", [64, 64]), action_size)
            self.critic = _mlp(obs_size, net_arch.get("common_critic", [64, 64]), 1, output_activation=False)
            self.core = nn.Identity()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.core, HumanExoActorCritic):
            return self.core(obs)
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std), mean, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, mean, value = self.distribution(obs)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, _mean, value = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
