"""Two PPO learners with independent critics, advantages and optimizers."""

from dataclasses import asdict
from math import isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from .runner import validate_checkpoint_config
from .split_config import SplitTrainingCfg

PIPELINE = "muscle_prosthesis_independent_ppo_v1"


class PolicyView:
    """Observation/action schema for one learner; physics belongs to the runner."""

    def __init__(self, env, muscle):
        self.env, self.muscle = env, muscle
        self.num_actions = env.trainable_muscle_ids.numel() if muscle else 2

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observations(self, obs):
        values = (
            {"policy": obs["muscle"]}
            if self.muscle
            else {key: obs[key] for key in ("actor", "critic")}
        )
        return TensorDict(values, batch_size=obs.batch_size, device=obs.device)

    def get_observations(self):
        return self.observations(self.env.get_observations())


class SplitRunner:
    def __init__(self, env, training, output=None):
        self.env, self.training = env, training
        self.output = None if output is None else Path(output)
        muscle_cfg, prosthesis_cfg = training.algorithms(env)
        self.muscle = OnPolicyRunner(
            PolicyView(env, True),
            muscle_cfg,
            log_dir=None if self.output is None else str(self.output / "muscle"),
            device=env.device,
        )
        self.prosthesis = OnPolicyRunner(
            PolicyView(env, False),
            prosthesis_cfg,
            log_dir=None if self.output is None else str(self.output / "prosthesis"),
            device=env.device,
        )
        self.learners = {"muscle": self.muscle, "prosthesis": self.prosthesis}
        self.iteration = -1
        self.best = {name: None for name in self.learners}

    def _best_checkpoints(self):
        improved = []
        for name, learner in self.learners.items():
            if len(learner.logger.rewbuffer) < 100:
                continue
            score = mean(learner.logger.rewbuffer)
            if isfinite(score) and (
                self.best[name] is None
                or score > self.best[name]["mean_episode_return"]
            ):
                self.best[name] = {
                    "mean_episode_return": score,
                    "iteration": self.iteration,
                    "window_episodes": 100,
                }
                improved.append(name)
        if self.output is not None:
            for name in improved:
                path = self.output / f"model_best_{name}.pt"
                self.save(path)
                self.learners[name].logger.writer.add_scalar(
                    "Train/best_mean_episode_return",
                    self.best[name]["mean_episode_return"],
                    self.iteration,
                )
                print(f"Saved {path.name}: {self.best[name]}", flush=True)

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
                    for _ in range(self.training.steps_per_env):
                        muscle = self.muscle.alg.act(self.muscle.env.observations(obs))
                        prosthesis = self.prosthesis.alg.act(
                            self.prosthesis.env.observations(obs)
                        )
                        obs, reward, done, extras = self.env.step(
                            torch.cat((prosthesis, muscle), -1)
                        )
                        check_nan(obs, reward, done)
                        for name, learner in self.learners.items():
                            own_reward = (
                                extras["muscle_reward"] if name == "muscle" else reward
                            )
                            own_log = (
                                extras["muscle_log"]
                                if name == "muscle"
                                else extras["log"]
                            )
                            own_extras = {
                                "time_outs": extras["time_outs"],
                                "log": own_log,
                            }
                            own_obs = learner.env.observations(obs)
                            learner.alg.process_env_step(
                                own_obs, own_reward, done, own_extras
                            )
                            learner.logger.process_env_step(
                                own_reward, done, own_extras
                            )
                    for learner in self.learners.values():
                        learner.alg.compute_returns(learner.env.observations(obs))
                collection = perf_counter() - tick
                self.iteration = iteration
                for name, learner in self.learners.items():
                    tick = perf_counter()
                    losses = learner.alg.update()
                    update_time = perf_counter() - tick
                    learner.current_learning_iteration = iteration
                    print(f"\n[{name}]", flush=True)
                    learner.logger.log(
                        it=iteration,
                        start_it=start,
                        total_it=max_iterations,
                        collect_time=collection,
                        learn_time=update_time,
                        loss_dict=losses,
                        learning_rate=learner.alg.learning_rate,
                        action_std=learner.alg.actor.output_std,
                        rnd_weight=None,
                    )
                self._best_checkpoints()
                if (
                    self.output is not None
                    and iteration % self.training.save_interval == 0
                ):
                    self.save(self.output / f"model_{iteration}.pt")
            if self.output is not None and self.iteration >= start:
                self.save(self.output / f"model_{self.iteration}.pt")
        finally:
            for learner in self.learners.values():
                if learner.logger.writer is not None:
                    learner.logger.stop_logging_writer()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        infos = {
            "pipeline": PIPELINE,
            "prosthesis_config": asdict(self.env.cfg),
            "training_config": asdict(self.training),
            "best": self.best,
            "env_rng": self.env.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.env.device),
            "return_windows": {
                name: list(learner.logger.rewbuffer)
                for name, learner in self.learners.items()
            },
            "length_windows": {
                name: list(learner.logger.lenbuffer)
                for name, learner in self.learners.items()
            },
            "resume_semantics": "Both PPOs, optimizers, warm-up counter, RNG and best-score windows restored; physical/controller/observation histories reset.",
        }
        torch.save(
            {
                "iter": self.iteration,
                "infos": infos,
                **{name: learner.alg.save() for name, learner in self.learners.items()},
            },
            path,
        )

    def load(self, path):
        saved = torch.load(path, weights_only=True, map_location=self.env.device)
        if saved["infos"]["pipeline"] != PIPELINE:
            raise ValueError("Not an Experiment 4–7 independent-PPO checkpoint")
        info = saved["infos"]
        validate_checkpoint_config(info["prosthesis_config"], asdict(self.env.cfg))
        saved_training = SplitTrainingCfg(**info["training_config"])
        if asdict(saved_training) != asdict(self.training):
            raise ValueError(
                "Independent-PPO training configuration differs (including reward mode and observations)"
            )
        for name, learner in self.learners.items():
            # RSL normalizers can replace buffers during inference-mode rollouts.
            with torch.inference_mode(False):
                for model in (learner.alg.actor, learner.alg.critic):
                    for module in model.modules():
                        for key, value in module.named_buffers(recurse=False):
                            if torch.is_inference(value):
                                setattr(module, key, value.clone())
            learner.alg.load(saved[name], None, True)
            learner.logger.rewbuffer.clear()
            learner.logger.rewbuffer.extend(info["return_windows"][name])
            learner.logger.lenbuffer.clear()
            learner.logger.lenbuffer.extend(info["length_windows"][name])
            learner.logger.cur_reward_sum.zero_()
            learner.logger.cur_episode_length.zero_()
        self.iteration = int(saved["iter"])
        self.best = info["best"]
        self.env.generator.set_state(info["env_rng"].cpu())
        torch.set_rng_state(info["torch_rng"].cpu())
        torch.cuda.set_rng_state(info["cuda_rng"].cpu(), self.env.device)
        with torch.inference_mode():
            self.env.reset()

    def inference_policy(self):
        for learner in self.learners.values():
            learner.alg.eval_mode()

        def policy(obs):
            muscle = self.muscle.alg.actor(self.muscle.env.observations(obs))
            prosthesis = self.prosthesis.alg.actor(
                self.prosthesis.env.observations(obs)
            )
            return torch.cat((prosthesis, muscle), -1)

        return policy
