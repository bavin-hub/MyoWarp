"""CUDA RSL-RL runner for frozen-muscle or joint muscle/prosthesis PPO."""

from dataclasses import asdict
from functools import wraps
from math import isfinite
from pathlib import Path
from statistics import mean

import torch
from rsl_rl.runners import OnPolicyRunner

from .gpu_env import GPUProsthesisVecEnv as ProsthesisVecEnv


def validate_checkpoint_config(saved, current, *, allow_muscle_override=False):
    defaults = {
        "right_leg_obs": "fooled",
        "train_muscle": False,
        "muscle_learning_rate": 1e-5,
    }
    saved = {**defaults, **saved}
    current = {**defaults, **current}
    differences = {
        key: (saved.get(key), current.get(key))
        for key in saved.keys() | current.keys()
        if saved.get(key) != current.get(key)
    }
    permitted = {"checkpoint_path", "right_leg_obs"} if allow_muscle_override else set()
    rejected = differences.keys() - permitted
    if rejected:
        raise ValueError(
            "Checkpoint task configuration differs: " + ", ".join(sorted(rejected))
        )
    return differences


class ProsthesisRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_checkpoint = None
        original_log = self.logger.log

        @wraps(original_log)
        def log_with_best_checkpoint(*args, **kwargs):
            original_log(*args, **kwargs)
            iteration = kwargs["it"] if "it" in kwargs else args[0]
            self._save_best_checkpoint(iteration)

        # Check after every iteration, independently of periodic saving.
        self.logger.log = log_with_best_checkpoint

    def _save_best_checkpoint(self, iteration):
        logger = self.logger
        if logger.writer is None or logger.log_dir is None:
            return
        # Match the displayed Mean reward; wait for a full 100-episode window.
        if len(logger.rewbuffer) < 100:
            return
        score = mean(logger.rewbuffer)
        if not isfinite(score):
            return
        if (
            self.best_checkpoint is not None
            and score <= self.best_checkpoint["mean_episode_return"]
        ):
            return
        self.best_checkpoint = {
            "mean_episode_return": score,
            "iteration": iteration,
            "window_episodes": 100,
        }
        self.save(str(Path(logger.log_dir) / "model_best.pt"))
        logger.writer.add_scalar("Train/best_mean_episode_return", score, iteration)
        print(
            f"Saved model_best.pt: mean episode return={score:.4f}, iteration={iteration}",
            flush=True,
        )

    def save(self, path, infos=None):
        info = dict(infos or {})
        info["best_checkpoint"] = self.best_checkpoint
        info["prosthesis_config"] = asdict(self.env.cfg)
        info["reference_frame"] = self.env.frame.clone()
        info["resume_semantics"] = (
            "PPO/optimizer/RNG restored; physical/controller/frozen-human histories reset"
        )
        info["env_rng"] = self.env.generator.get_state()
        info["torch_rng"] = torch.get_rng_state()
        info["cuda_rng"] = torch.cuda.get_rng_state(self.env.device)
        super().save(path, info)

    def load(
        self,
        path,
        load_cfg=None,
        strict=True,
        map_location=None,
        *,
        allow_muscle_override=False,
    ):
        # RSL's running normalizer may replace _std inside rollout inference_mode.
        # Make those buffers writable before loading into an already-used runner.
        with torch.inference_mode(False):
            for model in (self.alg.actor, self.alg.critic):
                for module in model.modules():
                    for name, value in module.named_buffers(recurse=False):
                        if torch.is_inference(value):
                            setattr(module, name, value.clone())
        infos = super().load(path, load_cfg, strict, map_location)
        changes = validate_checkpoint_config(
            infos["prosthesis_config"],
            asdict(self.env.cfg),
            allow_muscle_override=allow_muscle_override,
        )
        self.best_checkpoint = infos.get("best_checkpoint")
        if "checkpoint_path" in changes:
            old, new = changes["checkpoint_path"]
            print(
                f"Evaluating saved prosthesis policy with muscle checkpoint: {new} (trained with: {old})",
                flush=True,
            )
        if "right_leg_obs" in changes:
            old, new = changes["right_leg_obs"]
            print(
                f"Evaluation shadow mode: {new} (prosthesis policy trained with: {old})",
                flush=True,
            )
        self.env.generator.set_state(infos["env_rng"].cpu())
        torch.set_rng_state(infos["torch_rng"].cpu())
        torch.cuda.set_rng_state(infos["cuda_rng"].cpu(), self.env.device)
        with torch.inference_mode():
            self.env.reset()
        return infos


__all__ = ["ProsthesisRunner", "ProsthesisVecEnv"]
