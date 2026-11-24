"""Experiments 4–7: actor input variants and rewards on one real model/shadow."""

import torch
from tensordict import TensorDict

from .gpu_env import GPUProsthesisVecEnv
from .muscle_reward_gpu import MuscleMimicRewardBatch
from .prosthesis_observations import hip_actor_history, relative_hip_rpy
from .reference import load_human_clip

# Keep only terms already used in frozen-muscle runs that describe the CMU leg.
# Whole-body unsafe contacts still terminate both policies, but are not rewarded.
PROSTHESIS_REWARD_TERMS = (
    "foot_pos",
    "foot_ori",
    "foot_vel",
    "contact",
    "support",
    "slip",
    "impact",
    "action_rate",
    "effort",
    "saturation",
    "joint_limit",
)


class SplitProsthesisEnv(GPUProsthesisVecEnv):
    def __init__(self, cfg, training, controller_factory, num_envs=1024):
        training.validate()
        self.training = training
        self.hip_history = None
        if not cfg.train_muscle or cfg.right_leg_obs != "virtual":
            raise ValueError(
                "Independent PPO requires train_muscle and virtual observations"
            )
        super().__init__(cfg, controller_factory, num_envs)
        human = self.human.sim.cpu_model
        clip = load_human_clip(cfg.motion_path, human)
        self.muscle_tracking = MuscleMimicRewardBatch(human, clip, self.device)
        self.lower_leg_tracking = None
        if training.prosthesis_reward == "human_space":
            from .human_space_reward import LowerLegMimicReward

            self.lower_leg_tracking = LowerLegMimicReward(
                human, clip, self.device, cfg, num_envs
            )
        self.full_reward = self.state.reward
        self.state.reward = self._separate_rewards
        self.muscle_terminated = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.pending_muscle_action = None

    def get_observations(self):
        # Unlike exp3, BOTH muscle networks use fixed fused checkpoint stats.
        if self._muscle_obs is None:
            with torch.no_grad():
                self._muscle_obs = self.human.observe(
                    self.data, self.pre_substep, self.frame
                ).clone()
        actor = hip_actor_history(self.history)
        variant = self.training.prosthesis_observation
        if variant == "reference":
            # Current prosthesis-compatible q/qvel targets, not biological joints.
            targets = torch.cat(
                (
                    self.reference["qpos"][self.frame][:, self.state.qids],
                    self.reference["qvel"][self.frame][:, self.state.vids],
                ),
                -1,
            )
            actor = torch.cat((actor, targets), -1)
        elif variant == "muscle_reference":
            # Adapted knee/ankle errors relative to the real current joints.
            adapter = self.human.adapter
            frames = (self.frame[:, None] + adapter._lookahead).clamp(
                max=adapter._traj_len - 1
            )
            targets = torch.cat(
                (
                    self.reference["qpos"][frames][..., self.state.qids]
                    - self.data.qpos[:, None, self.state.qids],
                    self.reference["qvel"][frames][..., self.state.vids]
                    - self.data.qvel[:, None, self.state.vids],
                ),
                -1,
            ).flatten(1)
            phase = self.frame[:, None].float() / max(adapter._traj_len, 1)
            actor = torch.cat((actor, targets, phase), -1)
        return TensorDict(
            {
                "actor": actor,
                "critic": self.state.critic(
                    self.data, self.frame, self.previous_action
                ),
                "muscle": self._muscle_obs.clone(),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def hip_rpy(self):
        rotations = self.data.xmat[:, self.state.sensor_ids].reshape(-1, 2, 3, 3)
        return relative_hip_rpy(rotations[:, 0], rotations[:, 1])

    def reset_extra_history(self, ids):
        if self.training.prosthesis_observation == "hip_rpy":
            if self.hip_history is None:
                self.hip_history = torch.zeros_like(self.history[:, :, :3])
            self.hip_history[ids] = self.hip_rpy()[ids, None]

    def step_extra_history(self):
        if self.training.prosthesis_observation == "hip_rpy":
            self.hip_history[:, :-1] = self.hip_history[:, 1:].clone()
            self.hip_history[:, -1] = self.hip_rpy()

    def _separate_rewards(self, data, frame, *args):
        _, terms = self.full_reward(data, frame, *args)
        prosthesis_terms = {key: terms[key] for key in PROSTHESIS_REWARD_TERMS}
        prosthesis_reward = (
            sum(self.cfg.weights[k] * v for k, v in prosthesis_terms.items())
            * self.cfg.policy_dt
        )
        self.get_observations()  # Refresh virtual shadow once at the new state.
        raw, self.muscle_terms, self.muscle_terminated = self.muscle_tracking.reward(
            self.human.data, frame, self.pending_muscle_action
        )
        if self.lower_leg_tracking is not None:
            mapped_terms = self.lower_leg_tracking.reward(data, self.human, frame)
            # Replace only foot-space tracking; retain real-state contacts and
            # action/effort/safety penalties from Exp4.
            prosthesis_terms = {
                k: v
                for k, v in prosthesis_terms.items()
                if k not in ("foot_pos", "foot_ori", "foot_vel")
            }
            prosthesis_reward = (
                sum(mapped_terms.values()) * self.training.muscle_reward_scale
                + sum(self.cfg.weights[k] * v for k, v in prosthesis_terms.items())
                * self.cfg.policy_dt
            )
            prosthesis_terms.update({f"human_{k}": v for k, v in mapped_terms.items()})
        self.muscle_reward = raw * self.training.muscle_reward_scale
        torch._assert_async(
            torch.isfinite(self.muscle_reward).all(), "Nonfinite muscle reward"
        )
        # Root orientation is used only by the inherited safety termination.
        return prosthesis_reward, {**prosthesis_terms, "root_ori": terms["root_ori"]}

    def termination(self, terms, unsafe):
        return super().termination(terms, unsafe) | self.muscle_terminated

    @torch.no_grad()
    def step(self, actions):
        self.pending_muscle_action = actions[:, 2:].detach()
        obs, reward, done, extras = super().step(actions)
        orientation = extras["log"].pop("reward/root_ori")
        extras["log"]["safety/root_orientation"] = orientation
        extras["muscle_reward"] = self.muscle_reward.clone()
        extras["muscle_log"] = {
            f"reward/{k}": v.mean() for k, v in self.muscle_terms.items()
        }
        extras["muscle_component_reward"] = extras["muscle_reward"].clone()
        extras["prosthesis_component_reward"] = reward.clone()
        if self.training.reward_mode == "shared":
            # Experiment 5: identical scalar rewards, still independent PPOs.
            reward = reward + extras["muscle_reward"]
            extras["muscle_reward"] = reward.clone()
            if "episode/return" in extras["log"]:
                extras["log"]["component/prosthesis_episode_return"] = extras[
                    "log"
                ].pop("episode/return")
            for log in (extras["log"], extras["muscle_log"]):
                log["reward/shared_total"] = reward.mean()
                log["component/muscle_reward"] = extras[
                    "muscle_component_reward"
                ].mean()
                log["component/prosthesis_reward"] = extras[
                    "prosthesis_component_reward"
                ].mean()
        return obs, reward, done, extras
