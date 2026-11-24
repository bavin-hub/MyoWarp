"""Experiments 4–7: independent PPOs, reward modes and prosthesis actor inputs."""

from dataclasses import dataclass

from .config import make_agent_cfg


@dataclass
class SplitTrainingCfg:
    steps_per_env: int = 24
    save_interval: int = 100
    muscle_learning_rate: float = 3e-5
    muscle_lr_min: float = 1e-6
    muscle_lr_max: float = 1e-4
    critic_warmup: int = 20
    muscle_std_scale: float = 0.5
    muscle_reward_scale: float = 0.0385
    reward_mode: str = "separate"
    prosthesis_reward: str = "prosthesis_space"
    prosthesis_observation: str = "baseline"

    def validate(self):
        if self.prosthesis_reward not in ("prosthesis_space", "human_space"):
            raise ValueError("Unknown prosthesis reward variant")
        if self.prosthesis_reward == "human_space" and (
            self.reward_mode != "separate" or self.prosthesis_observation != "baseline"
        ):
            raise ValueError(
                "Experiment 8 requires baseline observations and separate rewards"
            )
        if self.prosthesis_observation not in (
            "baseline",
            "reference",
            "muscle_reference",
            "hip_rpy",
        ):
            raise ValueError("Unknown prosthesis observation variant")
        if self.prosthesis_observation != "baseline" and self.reward_mode != "separate":
            raise ValueError("Experiments 6/7 use Experiment 4's separate rewards")
        if self.reward_mode not in ("separate", "shared"):
            raise ValueError("reward_mode must be separate or shared")
        if self.steps_per_env < 1 or self.save_interval < 1 or self.critic_warmup < 0:
            raise ValueError("Invalid rollout, checkpoint interval or warm-up count")
        if (
            not 0
            < self.muscle_lr_min
            <= self.muscle_learning_rate
            <= self.muscle_lr_max
        ):
            raise ValueError("Muscle learning rate must be within positive bounds")
        if self.muscle_std_scale <= 0 or self.muscle_reward_scale <= 0:
            raise ValueError("Muscle noise and reward scales must be positive")

    @property
    def experiment(self):
        if self.prosthesis_reward == "human_space":
            return 8
        if self.prosthesis_observation in ("reference", "muscle_reference"):
            return 6
        if self.prosthesis_observation == "hip_rpy":
            return 7
        return 5 if self.reward_mode == "shared" else 4

    def algorithms(self, env):
        self.validate()
        prosthesis = make_agent_cfg(self.save_interval)
        prosthesis["num_steps_per_env"] = self.steps_per_env
        muscle = make_agent_cfg(self.save_interval)
        muscle["num_steps_per_env"] = self.steps_per_env
        muscle["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
        common = {
            "class_name": "myowarp.tasks.musculoskeletal_prosthesis.split_models:PretrainedMimicModel",
            "checkpoint_path": env.cfg.checkpoint_path,
        }
        muscle["actor"] = {
            **common,
            "part": "actor",
            "muscle_indices": env.trainable_muscle_ids.cpu().tolist(),
            "std_scale": self.muscle_std_scale,
            "distribution_cfg": {"class_name": "rsl_rl.modules:GaussianDistribution"},
        }
        muscle["critic"] = {**common, "part": "critic"}
        muscle["algorithm"].update(
            class_name="myowarp.tasks.musculoskeletal_prosthesis.split_models:MuscleFinetunePPO",
            learning_rate=self.muscle_learning_rate,
            lr_range=(self.muscle_lr_min, self.muscle_lr_max),
            critic_warmup=self.critic_warmup,
            num_learning_epochs=4,
            value_loss_coef=0.5,
            entropy_coef=0.0,
        )
        return muscle, prosthesis
