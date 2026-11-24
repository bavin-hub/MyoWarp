# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch re-implementation of the MuscleMimic actor network.

Converts Orbax/JAX parameter trees (loaded via
:func:`~myowarp.musclemimic.policy.load_local_policy_artifacts`)
into a :class:`torch.nn.Module` suitable for GPU-batched inference in mjlab.

Architecture (mirrors ``_actor_forward`` in ``fullbody_local_policy.py``):

    obs  ->  [obs_norm]  ->  ResBlock_*  ->  optional Tail  ->  output

Each ``ResBlock`` contains:
  - layer0: Linear → LayerNorm → SiLU
  - layer1: Linear → LayerNorm
  - optional projection on the residual shortcut
  - scalar gated residual: ``SiLU(shortcut + sigmoid(gate) * y)``

The Tail applies one more Linear → LayerNorm → SiLU before the output linear.
Obs normalisation uses the checkpoint's stored ``obs_mean`` / ``obs_var`` (fixed,
no running update) and is fused into the module so the ONNX export is
self-contained.

Example::

    from myowarp.musclemimic.policy import (
        load_local_policy_artifacts,
    )
    from myowarp.musclemimic.actor_torch import MimicActorModule

    artifacts = load_local_policy_artifacts(checkpoint_root)
    module = MimicActorModule.from_artifacts(artifacts)
    module.eval()

    obs = torch.zeros(N, artifacts.obs_dim)
    actions = module(obs)   # (N, act_dim), values in (-1, 1) approx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from myowarp.musclemimic.policy import (
        LocalPolicyArtifacts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _t(arr: Any) -> torch.Tensor:
    """Convert numpy array to float32 torch tensor."""
    return torch.as_tensor(np.array(arr, dtype=np.float32, copy=True))


@dataclass(frozen=True)
class _ResBlockWeights:
    """Raw numpy weights for one residual block."""

    l0_kernel: np.ndarray  # (in_dim, hidden_dim) JAX convention
    l0_bias: np.ndarray
    l0_ln_scale: np.ndarray
    l0_ln_bias: np.ndarray
    l1_kernel: np.ndarray  # (hidden_dim, hidden_dim)
    l1_bias: np.ndarray
    l1_ln_scale: np.ndarray
    l1_ln_bias: np.ndarray
    proj_kernel: np.ndarray | None  # (in_dim, hidden_dim) or None
    proj_bias: np.ndarray | None
    gate_raw: float  # scalar logit; gate = sigmoid(gate_raw)


def _read_res_block(actor: dict[str, Any], idx: int) -> _ResBlockWeights:
    d0 = actor[f"block{idx}_layer0_dense"]
    ln0 = actor[f"block{idx}_layer0_ln"]
    d1 = actor[f"block{idx}_layer1_dense"]
    ln1 = actor[f"block{idx}_layer1_ln"]
    proj = actor.get(f"block{idx}_proj")
    gate = actor[f"res_gate_{idx}"]
    return _ResBlockWeights(
        l0_kernel=np.asarray(d0["kernel"], dtype=np.float32),
        l0_bias=np.asarray(d0["bias"], dtype=np.float32),
        l0_ln_scale=np.asarray(ln0["scale"], dtype=np.float32),
        l0_ln_bias=np.asarray(ln0["bias"], dtype=np.float32),
        l1_kernel=np.asarray(d1["kernel"], dtype=np.float32),
        l1_bias=np.asarray(d1["bias"], dtype=np.float32),
        l1_ln_scale=np.asarray(ln1["scale"], dtype=np.float32),
        l1_ln_bias=np.asarray(ln1["bias"], dtype=np.float32),
        proj_kernel=(
            np.asarray(proj["kernel"], dtype=np.float32) if proj is not None else None
        ),
        proj_bias=(
            np.asarray(proj["bias"], dtype=np.float32) if proj is not None else None
        ),
        gate_raw=float(np.asarray(gate, dtype=np.float32).reshape(())),
    )


def _residual_block_indices(actor: dict[str, Any]) -> list[int]:
    """Return sorted residual block indices present in an actor tree."""
    indices: list[int] = []
    suffix = "_layer0_dense"
    for key in actor:
        if not key.startswith("block") or not key.endswith(suffix):
            continue
        idx = key[len("block") : -len(suffix)]
        if idx.isdigit():
            indices.append(int(idx))
    return sorted(indices)


def _sorted_prefix_indices(tree: dict[str, Any], prefix: str) -> list[int]:
    """Return sorted integer suffixes for keys like ``f"{prefix}{i}"``."""
    indices: list[int] = []
    for key in tree:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _linear_from_jax(kernel: np.ndarray, bias: np.ndarray) -> nn.Linear:
    """Build nn.Linear from JAX-convention (in, out) kernel."""
    out_dim, in_dim = kernel.T.shape
    layer = nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        layer.weight.copy_(_t(kernel.T))  # PyTorch: (out, in)
        layer.bias.copy_(_t(bias))
    return layer


def _layer_norm_from_jax(scale: np.ndarray, bias: np.ndarray) -> nn.LayerNorm:
    ln = nn.LayerNorm(scale.shape[0], elementwise_affine=True, eps=1e-5)
    with torch.no_grad():
        ln.weight.copy_(_t(scale))
        ln.bias.copy_(_t(bias))
    return ln


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    """One gated residual block of the MuscleMimic actor."""

    def __init__(self, weights: _ResBlockWeights) -> None:
        super().__init__()
        self.layer0 = _linear_from_jax(weights.l0_kernel, weights.l0_bias)
        self.ln0 = _layer_norm_from_jax(weights.l0_ln_scale, weights.l0_ln_bias)
        self.layer1 = _linear_from_jax(weights.l1_kernel, weights.l1_bias)
        self.ln1 = _layer_norm_from_jax(weights.l1_ln_scale, weights.l1_ln_bias)

        self.proj: nn.Linear | None = None
        if weights.proj_kernel is not None:
            assert weights.proj_bias is not None
            self.proj = _linear_from_jax(weights.proj_kernel, weights.proj_bias)

        # Scalar gate stored as a non-trainable parameter so it moves with .to(device)
        self.register_buffer("gate_raw", torch.tensor(weights.gate_raw))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.ln0(self.layer0(h)))
        y = self.ln1(self.layer1(y))
        shortcut = h if self.proj is None else self.proj(h)
        gate = torch.sigmoid(self.gate_raw)
        return F.silu(shortcut + gate * y)


# ---------------------------------------------------------------------------
# Full actor module
# ---------------------------------------------------------------------------


class MimicActorModule(nn.Module):
    """Actor as a PyTorch module with fused obs normalisation.

    Obs normalisation uses the checkpoint's stored ``obs_mean`` / ``obs_var``
    as fixed buffers — no running update at inference time.

    Args:
        params: Param tree from :class:`~...LocalPolicyArtifacts` (already
                converted to numpy via ``_to_numpy_tree``).
        obs_mean: Stored running mean, shape ``(obs_dim,)``.
        obs_var: Stored running variance, shape ``(obs_dim,)``.

    Returns:
        ``(N, act_dim)`` tensor of deterministic mean actions.
    """

    def __init__(
        self,
        params: dict[str, Any],
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
    ) -> None:
        super().__init__()
        actor = params["actor"]
        block_indices = _residual_block_indices(actor)
        if not block_indices:
            raise ValueError("Unsupported actor params: no residual blocks found.")
        self.blocks = nn.ModuleList(
            [_ResidualBlock(_read_res_block(actor, i)) for i in block_indices]
        )

        self.tail: nn.Linear | None = None
        self.tail_ln: nn.LayerNorm | None = None
        tail_d = actor.get("tail_dense")
        tail_ln = actor.get("tail_ln")
        if tail_d is not None:
            if tail_ln is None:
                raise ValueError(
                    "Unsupported actor params: tail_dense without tail_ln."
                )
            self.tail = _linear_from_jax(
                np.asarray(tail_d["kernel"], dtype=np.float32),
                np.asarray(tail_d["bias"], dtype=np.float32),
            )
            self.tail_ln = _layer_norm_from_jax(
                np.asarray(tail_ln["scale"], dtype=np.float32),
                np.asarray(tail_ln["bias"], dtype=np.float32),
            )
        out = actor["output"]
        self.output = _linear_from_jax(
            np.asarray(out["kernel"], dtype=np.float32),
            np.asarray(out["bias"], dtype=np.float32),
        )

        self.register_buffer("obs_mean", _t(obs_mean))
        self.register_buffer("obs_std", torch.sqrt(_t(obs_var) + 1e-8))

        self.obs_dim: int = int(obs_mean.shape[0])
        self.act_dim: int = int(np.asarray(out["bias"]).shape[0])

    @classmethod
    def from_artifacts(cls, artifacts: LocalPolicyArtifacts) -> MimicActorModule:
        """Construct from a loaded :class:`~...LocalPolicyArtifacts` object.

        Args:
            artifacts: Loaded policy artifacts (params + normalisation stats).

        Returns:
            ``MimicActorModule`` in eval mode with weights copied from
            the checkpoint.
        """
        module = cls(artifacts.params, artifacts.obs_mean, artifacts.obs_var)
        module.eval()
        return module

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply checkpoint obs normalisation: ``(obs - mean) / std``.

        Args:
            obs: Raw observations, shape ``(..., obs_dim)``.

        Returns:
            Normalised observations, same shape.
        """
        return (obs - self.obs_mean) / self.obs_std

    def forward_normalized(self, norm_obs: torch.Tensor) -> torch.Tensor:
        """Run the actor on already-normalised observations.

        Useful when normalisation is applied externally (e.g. in a batched
        mjlab pipeline where normalisation is done once per batch).

        Args:
            norm_obs: Pre-normalised observations, shape ``(N, obs_dim)``.

        Returns:
            Deterministic mean actions, shape ``(N, act_dim)``.
        """
        h = norm_obs
        for block in self.blocks:
            h = block(h)
        if self.tail is not None:
            assert self.tail_ln is not None
            h = F.silu(self.tail_ln(self.tail(h)))
        return self.output(h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise obs then run the actor.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``.

        Returns:
            Deterministic mean actions, shape ``(N, act_dim)``.
        """
        return self.forward_normalized(self.normalize(obs))
