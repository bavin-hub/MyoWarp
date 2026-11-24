"""Load frozen Orbax or RSL muscle checkpoints before CUDA rollouts begin."""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from myowarp.musclemimic.policy import (
    LocalPolicyArtifacts,
    load_local_policy_artifacts,
    read_checkpoint_config_metadata,
)


def load_torch_muscle_artifacts(path):
    """Translate the residual actor schema; retain exact fixed checkpoint stats."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["actor_state_dict"]
    actor_state = {
        k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")
    }
    if any(not torch.isfinite(v).all() for v in actor_state.values()):
        raise ValueError(
            "Muscle checkpoint contains nonfinite actor weights/statistics"
        )
    mean, std = actor_state["obs_mean"], actor_state["obs_std"]
    if (
        mean.shape != (2418,)
        or actor_state["output.bias"].shape != (354,)
        or not (std > 0).all()
    ):
        raise ValueError(
            "Expected the 2418-input / 354-muscle fullbody actor with positive normalization std"
        )

    def array(name):
        return actor_state[name].numpy().copy()

    actor = {}
    blocks = sorted(
        {
            int(m.group(1))
            for key in actor_state
            if (m := re.match(r"blocks\.(\d+)\.", key))
        }
    )
    if blocks != list(range(len(blocks))) or not blocks:
        raise ValueError("Expected contiguous residual blocks in muscle checkpoint")
    for i in blocks:
        for layer in (0, 1):
            p = f"blocks.{i}.layer{layer}"
            actor[f"block{i}_layer{layer}_dense"] = {
                "kernel": array(p + ".weight").T,
                "bias": array(p + ".bias"),
            }
            p = f"blocks.{i}.ln{layer}"
            actor[f"block{i}_layer{layer}_ln"] = {
                "scale": array(p + ".weight"),
                "bias": array(p + ".bias"),
            }
        if f"blocks.{i}.proj.weight" in actor_state:
            actor[f"block{i}_proj"] = {
                "kernel": array(f"blocks.{i}.proj.weight").T,
                "bias": array(f"blocks.{i}.proj.bias"),
            }
        actor[f"res_gate_{i}"] = array(f"blocks.{i}.gate_raw")
    if "tail.weight" in actor_state:
        actor["tail_dense"] = {
            "kernel": array("tail.weight").T,
            "bias": array("tail.bias"),
        }
        actor["tail_ln"] = {
            "scale": array("tail_ln.weight"),
            "bias": array("tail_ln.bias"),
        }
    actor["output"] = {"kernel": array("output.weight").T, "bias": array("output.bias")}
    artifacts = LocalPolicyArtifacts(
        params={"actor": actor},
        obs_mean=array("obs_mean"),
        obs_var=np.square(array("obs_std")),
        obs_count=np.asarray(1, dtype=np.float32),
        obs_dim=2418,
        action_dim=354,
    )
    metadata = json.loads(
        Path(__file__).with_name("muscle_observation_metadata.json").read_text()
    )
    metadata.update(
        checkpoint_format="rsl_rl",
        normalization_mode="fixed",
        actor_state_dict=actor_state,
        obs_std=array("obs_std"),
        iteration=checkpoint.get("iter"),
    )
    return artifacts, metadata


@lru_cache(maxsize=2)
def frozen_artifacts(checkpoint):
    path = Path(checkpoint).resolve()
    if path.is_file():
        return load_torch_muscle_artifacts(path)
    return load_local_policy_artifacts(path), read_checkpoint_config_metadata(path)
