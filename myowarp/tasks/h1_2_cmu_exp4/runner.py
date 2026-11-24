"""Independent 50Hz H1 and 100Hz CMU PPO rollouts and paired checkpoints."""

import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from myowarp.tasks.h1_2_tracking.config import make_rsl_rl_cfg

from .reference import fingerprint, model_fingerprint

PIPELINE = "h1_2_cmu_exp4_independent_ppo_v1"


class PolicyView:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.num_actions = 27 if name == "h1" else 2

    def __getattr__(self, key):
        return getattr(self.env, key)

    def observations(self, obs):
        keys = ("h1_actor", "h1_critic") if self.name == "h1" else ("actor", "critic")
        return TensorDict(
            {key: obs[key] for key in keys},
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def get_observations(self):
        return self.observations(self.env.get_observations())


def algorithms(cfg):
    h = make_rsl_rl_cfg(save_interval=cfg.save_interval)
    p = make_rsl_rl_cfg(save_interval=cfg.save_interval)
    h["num_steps_per_env"] = cfg.steps_per_env
    p["num_steps_per_env"] = 2 * cfg.steps_per_env
    h["obs_groups"] = {"actor": ["h1_actor"], "critic": ["h1_critic"]}
    for part in ("actor", "critic"):
        h[part]["class_name"] = (
            "myowarp.tasks.h1_2_cmu_exp4.algorithms:FixedNormalizationMLP"
        )
    h["algorithm"].update(
        class_name="myowarp.tasks.h1_2_cmu_exp4.algorithms:FinetunePPO",
        learning_rate=cfg.human_learning_rate,
        lr_range=(cfg.human_lr_min, cfg.human_lr_max),
        critic_warmup=cfg.critic_warmup,
        entropy_coef=0.0,
        num_learning_epochs=4,
        value_loss_coef=0.5,
    )
    p["actor"]["hidden_dims"] = [128, 128]
    p["actor"]["distribution_cfg"]["init_std"] = 0.2
    h["experiment_name"] = "h1_2_cmu_exp4_h1"
    p["experiment_name"] = "h1_2_cmu_exp4_prosthesis"
    return h, p


class Exp4Runner:
    def __init__(self, env, output=None):
        self.env = env
        self.cfg = env.cfg
        self.output = Path(output) if output else None
        self.learners = {
            name: OnPolicyRunner(
                PolicyView(env, name),
                config,
                log_dir=str(self.output / name) if self.output else None,
                device=env.device,
            )
            for name, config in zip(("h1", "prosthesis"), algorithms(self.cfg))
        }
        self.iteration = -1
        self.best = {name: None for name in self.learners}
        saved = torch.load(
            self.cfg.human_checkpoint, map_location=env.device, weights_only=True
        )
        # Pretrained actor, critic and normalization; fresh optimizer for fine-tuning.
        self.learners["h1"].alg.load(saved, {"actor": True, "critic": True}, True)
        distribution = self.learners["h1"].alg._raw_actor.distribution
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                distribution.std_param.mul_(self.cfg.human_std_scale)
            else:
                distribution.log_std_param.add_(math.log(self.cfg.human_std_scale))

    def inference_actions(self, obs):
        for learner in self.learners.values():
            learner.alg.eval_mode()
        return {
            name: learner.alg.actor(learner.env.observations(obs))
            for name, learner in self.learners.items()
        }

    def _process(self, name, obs, reward, done, extras):
        learner = self.learners[name]
        own = {"time_outs": extras["time_outs"], "log": extras["logs"][name]}
        learner.alg.process_env_step(learner.env.observations(obs), reward, done, own)
        learner.logger.process_env_step(reward, done, own)

    def learn(self, max_iterations):
        for learner in self.learners.values():
            learner.alg.train_mode()
            learner.logger.init_logging_writer()
        obs = self.env.get_observations()
        start = self.iteration + 1
        try:
            for iteration in range(start, max_iterations):
                tick = perf_counter()
                with torch.inference_mode():
                    for _ in range(self.cfg.steps_per_env):
                        human = self.learners["h1"].alg.act(
                            self.learners["h1"].env.observations(obs)
                        )
                        human_reward = torch.zeros(
                            self.env.num_envs, device=self.env.device
                        )
                        for _ in range(2):
                            prost = self.learners["prosthesis"].alg.act(
                                self.learners["prosthesis"].env.observations(obs)
                            )
                            obs, rewards, done, extras = self.env.step(human, prost)
                            check_nan(obs, rewards["prosthesis"], done)
                            human_reward += rewards["h1"]
                            self._process(
                                "prosthesis", obs, rewards["prosthesis"], done, extras
                            )
                        self._process("h1", obs, human_reward, done, extras)
                    for learner in self.learners.values():
                        learner.alg.compute_returns(learner.env.observations(obs))
                collection = perf_counter() - tick
                self.iteration = iteration
                for learner in self.learners.values():
                    tick = perf_counter()
                    losses = learner.alg.update()
                    learner.current_learning_iteration = iteration
                    learner.logger.log(
                        it=iteration,
                        start_it=start,
                        total_it=max_iterations,
                        collect_time=collection,
                        learn_time=perf_counter() - tick,
                        loss_dict=losses,
                        learning_rate=learner.alg.learning_rate,
                        action_std=learner.alg.actor.output_std,
                        rnd_weight=None,
                    )
                self._best_checkpoints()
                if self.output and iteration % self.cfg.save_interval == 0:
                    self.save(self.output / f"model_{iteration}.pt")
        finally:
            if self.output:
                self.save(self.output / "model_last.pt")
            for learner in self.learners.values():
                if learner.logger.writer is not None:
                    learner.logger.stop_logging_writer()

    def _best_checkpoints(self):
        for name, learner in self.learners.items():
            values = learner.logger.rewbuffer
            if not values:
                continue
            score = mean(values)
            if math.isfinite(score) and (
                self.best[name] is None or score > self.best[name]["return"]
            ):
                self.best[name] = {
                    "return": score,
                    "iteration": self.iteration,
                    "episodes": len(values),
                }
                if self.output:
                    self.save(self.output / f"model_best_{name}.pt")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "pipeline": PIPELINE,
            "config": asdict(self.cfg),
            "best": self.best,
            "motion_sha256": fingerprint(self.cfg.motion_path),
            "reference_sha256": fingerprint(self.cfg.reference_path),
            "model_sha256": model_fingerprint(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.env.device),
            "env_rng": self.env.generator.get_state(),
            "failure_weights": self.env.failure_weights,
            "return_windows": {
                name: list(r.logger.rewbuffer) for name, r in self.learners.items()
            },
            "length_windows": {
                name: list(r.logger.lenbuffer) for name, r in self.learners.items()
            },
            "resume_semantics": "Both PPO/optimizer states and warmup counter restored; physics and histories reset",
        }
        torch.save(
            dict(
                iter=self.iteration,
                infos=info,
                **{name: r.alg.save() for name, r in self.learners.items()},
            ),
            path,
        )

    def load(self, path, evaluation=False):
        saved = torch.load(path, map_location=self.env.device, weights_only=True)
        info = saved["infos"]
        if info["pipeline"] != PIPELINE:
            raise ValueError("Wrong checkpoint pipeline")
        current = asdict(self.cfg)
        previous = dict(info["config"])
        for key in ("num_envs", "device") + (
            ("episode_seconds",) if evaluation else ()
        ):
            current.pop(key, None)
            previous.pop(key, None)
        if current != previous:
            raise ValueError("Checkpoint task/training configuration differs")
        for key, actual in [
            ("motion_sha256", fingerprint(self.cfg.motion_path)),
            ("reference_sha256", fingerprint(self.cfg.reference_path)),
            ("model_sha256", model_fingerprint()),
        ]:
            if info[key] != actual:
                raise ValueError(f"Checkpoint {key} differs")
        for name, learner in self.learners.items():
            with torch.inference_mode(False):
                for model in (learner.alg._raw_actor, learner.alg._raw_critic):
                    for module in model.modules():
                        for key, value in list(module._buffers.items()):
                            if value is not None and value.is_inference():
                                module._buffers[key] = value.clone()
            learner.alg.load(saved[name], None, True)
            learner.logger.rewbuffer.clear()
            learner.logger.rewbuffer.extend(info["return_windows"][name])
            learner.logger.lenbuffer.clear()
            learner.logger.lenbuffer.extend(info["length_windows"][name])
        self.iteration = int(saved["iter"])
        self.best = info["best"]
        self.env.failure_weights.copy_(info["failure_weights"])
        if not evaluation:
            self.env.generator.set_state(info["env_rng"].cpu())
            torch.set_rng_state(info["torch_rng"].cpu())
            torch.cuda.set_rng_state(info["cuda_rng"].cpu(), self.env.device)
        self.env.reset()
