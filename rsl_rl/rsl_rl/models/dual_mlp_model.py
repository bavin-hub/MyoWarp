# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two independent MLP actors exposed through the standard RSL-RL model API."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import HiddenState

from .mlp_model import MLPModel


class DualMLPModel(nn.Module):
    """Combine two independent stochastic MLP actors into one action vector.

    The two branches may consume different observation sets and control arbitrary,
    disjoint action indices. Their diagonal Gaussian distributions are independent,
    so their summed log-probability, entropy, and KL divergence form the equivalent
    joint action distribution expected by PPO.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        primary_action_indices: list[int] | tuple[int, ...],
        secondary_action_indices: list[int] | tuple[int, ...],
        primary_obs_set: str = "actor",
        secondary_obs_set: str = "actor",
        primary_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        secondary_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        del obs_set  # Required by the common model constructor interface.
        if distribution_cfg is None:
            raise ValueError("DualMLPModel requires a stochastic distribution_cfg")

        primary_indices = [int(index) for index in primary_action_indices]
        secondary_indices = [int(index) for index in secondary_action_indices]
        combined_indices = primary_indices + secondary_indices
        if sorted(combined_indices) != list(range(output_dim)):
            raise ValueError(
                "primary_action_indices and secondary_action_indices must be a "
                f"disjoint partition of [0, {output_dim})"
            )

        self.primary_actor = MLPModel(
            obs,
            obs_groups,
            primary_obs_set,
            len(primary_indices),
            hidden_dims=primary_hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=copy.deepcopy(distribution_cfg),
        )
        self.secondary_actor = MLPModel(
            obs,
            obs_groups,
            secondary_obs_set,
            len(secondary_indices),
            hidden_dims=secondary_hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=copy.deepcopy(distribution_cfg),
        )
        self.primary_obs_set = primary_obs_set
        self.secondary_obs_set = secondary_obs_set
        self.output_dim = output_dim
        self.register_buffer(
            "primary_action_indices",
            torch.tensor(primary_indices, dtype=torch.long),
        )
        self.register_buffer(
            "secondary_action_indices",
            torch.tensor(secondary_indices, dtype=torch.long),
        )
        self.register_buffer(
            "canonical_from_concatenated",
            torch.tensor(
                [combined_indices.index(index) for index in range(output_dim)],
                dtype=torch.long,
            ),
        )

    def _assemble(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((primary, secondary), dim=-1)
        return combined.index_select(-1, self.canonical_from_concatenated)

    def _split(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            outputs.index_select(-1, self.primary_action_indices),
            outputs.index_select(-1, self.secondary_action_indices),
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        primary = self.primary_actor(obs, masks, hidden_state, stochastic_output)
        secondary = self.secondary_actor(obs, masks, hidden_state, stochastic_output)
        return self._assemble(primary, secondary)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.primary_actor.reset(dones, hidden_state)
        self.secondary_actor.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.primary_actor.detach_hidden_state(dones)
        self.secondary_actor.detach_hidden_state(dones)

    @property
    def output_mean(self) -> torch.Tensor:
        return self._assemble(self.primary_actor.output_mean, self.secondary_actor.output_mean)

    @property
    def output_std(self) -> torch.Tensor:
        return self._assemble(self.primary_actor.output_std, self.secondary_actor.output_std)

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.primary_actor.output_entropy + self.secondary_actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return (self.output_mean, self.output_std)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        primary, secondary = self._split(outputs)
        return self.primary_actor.get_output_log_prob(primary) + self.secondary_actor.get_output_log_prob(secondary)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_distribution = torch.distributions.Normal(old_mean, old_std)
        new_distribution = torch.distributions.Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_distribution, new_distribution).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        self.primary_actor.update_normalization(obs)
        self.secondary_actor.update_normalization(obs)

    def as_jit(self) -> nn.Module:
        if self.primary_actor.obs_groups == self.secondary_actor.obs_groups:
            return _TorchSharedObsDualMLPModel(self)
        return _TorchSeparateObsDualMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        if self.primary_actor.obs_groups == self.secondary_actor.obs_groups:
            return _OnnxSharedObsDualMLPModel(self, verbose)
        return _OnnxSeparateObsDualMLPModel(self, verbose)


class _DualExportBase(nn.Module):
    def __init__(self, model: DualMLPModel) -> None:
        super().__init__()
        self.primary_policy = model.primary_actor.as_jit()
        self.secondary_policy = model.secondary_actor.as_jit()
        self.register_buffer(
            "canonical_from_concatenated",
            model.canonical_from_concatenated.detach().cpu().clone(),
        )

    def _assemble(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        return torch.cat((primary, secondary), dim=-1).index_select(
            -1, self.canonical_from_concatenated
        )


class _TorchSharedObsDualMLPModel(_DualExportBase):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._assemble(
            self.primary_policy(observations),
            self.secondary_policy(observations),
        )

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchSeparateObsDualMLPModel(_DualExportBase):
    def forward(
        self, primary_observations: torch.Tensor, secondary_observations: torch.Tensor
    ) -> torch.Tensor:
        return self._assemble(
            self.primary_policy(primary_observations),
            self.secondary_policy(secondary_observations),
        )

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxSharedObsDualMLPModel(_TorchSharedObsDualMLPModel):
    is_recurrent: bool = False

    def __init__(self, model: DualMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.primary_actor.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _OnnxSeparateObsDualMLPModel(_TorchSeparateObsDualMLPModel):
    is_recurrent: bool = False

    def __init__(self, model: DualMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.primary_input_size = model.primary_actor.obs_dim
        self.secondary_input_size = model.secondary_actor.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.primary_input_size),
            torch.zeros(1, self.secondary_input_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["primary_obs", "secondary_obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
