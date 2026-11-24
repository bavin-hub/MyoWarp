"""Separate checkpoint identity and best-return saving for intact H1-2."""

import math
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from statistics import mean

import torch
from rsl_rl.runners import OnPolicyRunner

from myowarp.tasks.h1_with_osl_tracking.runner import H1TrackingRunner as BaseRunner

PIPELINE = "h1_2_tracking"


class H1TrackingRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_return = -math.inf
        self.best_iteration = None
        original_log = self.logger.log

        @wraps(original_log)
        def log_with_best(*args, **kwargs):
            original_log(*args, **kwargs)
            if self.logger.log_dir is None or len(self.logger.rewbuffer) < 100:
                return
            score = mean(self.logger.rewbuffer)
            if math.isfinite(score) and score > self.best_return:
                self.best_return = score
                self.best_iteration = kwargs["it"] if "it" in kwargs else args[0]
                self.save(str(Path(self.logger.log_dir) / "model_best.pt"))

        self.logger.log = log_with_best

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos.update(
            pipeline=PIPELINE,
            env_config=asdict(self.env.cfg),
            env_state=self.env.training_state(),
            best_return=self.best_return,
            best_iteration=self.best_iteration,
        )
        # Best snapshots don't repeatedly export a 261-second motion ONNX.
        if Path(path).name == "model_best.pt":
            OnPolicyRunner.save(self, str(path), infos)
        else:
            super().save(str(path), infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        info = saved.get("infos", {})
        if info.get("pipeline") != PIPELINE:
            raise ValueError("Expected an intact H1-2 tracking checkpoint")
        for key in ("model_path", "motion_path", "physics_dt", "decimation"):
            if info["env_config"][key] != getattr(self.env.cfg, key):
                raise ValueError(f"Checkpoint {key} differs")
        del saved
        # RSL updates normalization std under inference_mode during collection.
        # Restore ordinary buffers before an in-place checkpoint copy, without
        # changing the shared OSL runner or RSL implementation.
        with torch.inference_mode(False):
            for model in (self.alg._raw_actor, self.alg._raw_critic):
                for module in model.modules():
                    for name, value in module._buffers.items():
                        if value is not None and value.is_inference():
                            module._buffers[name] = value.clone()
        infos = super().load(path, load_cfg, strict, map_location)
        self.best_return = infos.get("best_return", -math.inf)
        self.best_iteration = infos.get("best_iteration")
        return infos
