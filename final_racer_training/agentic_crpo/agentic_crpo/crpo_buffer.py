"""SB3 RolloutBuffer extension with an independent cost GAE."""

from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer


class CRPORolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    critic_taus: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_cost_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    cost_advantages: th.Tensor
    normalized_advantages: th.Tensor
    normalized_cost_advantages: th.Tensor
    returns: th.Tensor
    cost_returns: th.Tensor


class CRPORolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        *args,
        gamma_cost: float = 1.0,
        gae_lambda_cost: float = 0.95,
        **kwargs,
    ) -> None:
        self.gamma_cost = gamma_cost
        self.gae_lambda_cost = gae_lambda_cost
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs)
        self.costs = np.zeros(shape, dtype=np.float32)
        self.cost_values = np.zeros(shape, dtype=np.float32)
        self.cost_advantages = np.zeros(shape, dtype=np.float32)
        self.cost_returns = np.zeros(shape, dtype=np.float32)
        self.normalized_advantages = np.zeros(shape, dtype=np.float32)
        self.normalized_cost_advantages = np.zeros(shape, dtype=np.float32)
        self.critic_taus = np.zeros(shape, dtype=np.float32)
        self.reward_adv_mean = 0.0
        self.reward_adv_std = 0.0
        self.constraint_adv_mean = 0.0
        self.constraint_adv_std = 0.0
        self.action_versions = np.zeros(shape, dtype=np.int64)
        self.policy_versions = np.zeros(shape, dtype=np.int64)
        self.action_ages = np.zeros(shape, dtype=np.float64)
        self.sim_timestamps = np.zeros(shape, dtype=np.float64)
        self.step_ids = np.zeros(shape, dtype=np.int64)
        self.sim_times = np.zeros(shape, dtype=np.float64)
        self.state_versions = np.zeros(shape, dtype=np.int64)
        self.communication_state_versions = np.zeros(shape, dtype=np.int64)
        self.update_in_progress_flags = np.zeros(shape, dtype=bool)
        self.action_wall_time_ns = np.zeros(shape, dtype=np.int64)
        self.state_timestamps = np.zeros(shape, dtype=np.float64)
        self.cost_ready = np.ones(shape, dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        cost: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor | np.ndarray,
        cost_value: th.Tensor | np.ndarray,
        log_prob: th.Tensor | np.ndarray,
        *,
        critic_tau: np.ndarray | float = 0.0,
        action_version: np.ndarray | int = 0,
        policy_version: np.ndarray | int = 0,
        action_age: np.ndarray | float = 0.0,
        sim_timestamp: np.ndarray | float = 0.0,
        step_id: np.ndarray | int = 0,
        sim_time: np.ndarray | float = 0.0,
        state_version: np.ndarray | int = 0,
        communication_state_version: np.ndarray | int = 0,
        update_in_progress: np.ndarray | bool = False,
        action_timestamp: np.ndarray | int = 0,
        state_timestamp: np.ndarray | float = 0.0,
        cost_ready: np.ndarray | bool = True,
    ) -> None:
        position = self.pos
        value_array = (
            value.detach().cpu().numpy()
            if isinstance(value, th.Tensor)
            else np.asarray(value)
        )
        cost_value_array = (
            cost_value.detach().cpu().numpy()
            if isinstance(cost_value, th.Tensor)
            else np.asarray(cost_value)
        )
        log_prob_array = (
            log_prob.detach().cpu().numpy()
            if isinstance(log_prob, th.Tensor)
            else np.asarray(log_prob)
        )
        if log_prob_array.ndim == 0:
            log_prob_array = log_prob_array.reshape(-1, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))
        for name, values in (
            ("observation", obs),
            ("action", action),
            ("reward", reward),
            ("cost", cost),
            ("reward value", value_array),
            ("cost value", cost_value_array),
            ("old log probability", log_prob_array),
            ("critic tau", np.asarray(critic_tau)),
        ):
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    f"rollout {name} contains NaN or Inf"
                )
        self.observations[position] = np.asarray(obs)
        self.actions[position] = np.asarray(action)
        self.rewards[position] = np.asarray(reward)
        self.episode_starts[position] = np.asarray(episode_start)
        self.values[position] = value_array.reshape(-1)
        self.log_probs[position] = log_prob_array
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        self.costs[position] = np.asarray(cost)
        self.cost_values[position] = cost_value_array.reshape(-1)
        self.critic_taus[position] = np.clip(
            np.asarray(critic_tau, dtype=np.float32).reshape(-1), 0.0, 1.0
        )
        self.action_versions[position] = np.asarray(action_version)
        self.policy_versions[position] = np.asarray(policy_version)
        self.action_ages[position] = np.asarray(action_age)
        self.sim_timestamps[position] = np.asarray(sim_timestamp)
        self.step_ids[position] = np.asarray(step_id)
        self.sim_times[position] = np.asarray(sim_time)
        self.state_versions[position] = np.asarray(state_version)
        self.communication_state_versions[position] = np.asarray(
            communication_state_version
        )
        self.update_in_progress_flags[position] = np.asarray(
            update_in_progress
        )
        self.action_wall_time_ns[position] = np.asarray(action_timestamp)
        self.state_timestamps[position] = np.asarray(state_timestamp)
        self.cost_ready[position] = np.asarray(cost_ready)

    def backfill_cost(self, position: int, env_index: int, cost: float) -> None:
        if not (0 <= position < self.valid_size and 0 <= env_index < self.n_envs):
            raise IndexError("invalid rollout cost backfill position")
        if not np.isfinite(cost):
            raise ValueError("backfilled cost must be finite")
        self.costs[position, env_index] = float(cost)
        self.cost_ready[position, env_index] = True

    def assert_costs_ready(self) -> None:
        if not np.all(self.cost_ready[: self.valid_size]):
            pending = np.argwhere(~self.cost_ready[: self.valid_size])
            raise RuntimeError(
                f"CRPO update refused: {len(pending)} rollout costs are pending"
            )

    def assert_finite(self) -> None:
        """Refuse an update before non-finite rollout data reaches gradients."""

        size = self.valid_size
        for name in (
            "observations",
            "actions",
            "rewards",
            "costs",
            "values",
            "cost_values",
            "critic_taus",
            "log_probs",
            "advantages",
            "cost_advantages",
            "normalized_advantages",
            "normalized_cost_advantages",
            "returns",
            "cost_returns",
        ):
            values = self.__dict__[name][:size]
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    f"CRPO update refused: rollout {name} contains NaN or Inf"
                )

    @property
    def valid_size(self) -> int:
        """Number of collected transitions, including a partial final rollout."""

        return self.buffer_size if self.full else self.pos

    def compute_returns_and_advantage(
        self,
        last_values: th.Tensor,
        last_cost_values: th.Tensor,
        dones: np.ndarray,
    ) -> None:
        valid_size = self.valid_size
        if valid_size < 1:
            raise ValueError("cannot compute returns for an empty rollout")
        self.assert_costs_ready()
        next_reward_values = last_values.detach().cpu().numpy().flatten()
        next_cost_values = last_cost_values.detach().cpu().numpy().flatten()
        last_reward_gae = 0.0
        last_cost_gae = 0.0
        for step in reversed(range(valid_size)):
            if step == valid_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                following_reward = next_reward_values
                following_cost = next_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                following_reward = self.values[step + 1]
                following_cost = self.cost_values[step + 1]
            reward_delta = (
                self.rewards[step]
                + self.gamma * following_reward * next_non_terminal
                - self.values[step]
            )
            last_reward_gae = (
                reward_delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_reward_gae
            )
            self.advantages[step] = last_reward_gae
            cost_delta = (
                self.costs[step]
                + self.gamma_cost * following_cost * next_non_terminal
                - self.cost_values[step]
            )
            last_cost_gae = (
                cost_delta
                + self.gamma_cost
                * self.gae_lambda_cost
                * next_non_terminal
                * last_cost_gae
            )
            self.cost_advantages[step] = last_cost_gae
        self.returns[:valid_size] = (
            self.advantages[:valid_size] + self.values[:valid_size]
        )
        self.cost_returns[:valid_size] = (
            self.cost_advantages[:valid_size] + self.cost_values[:valid_size]
        )
        # Normalize the two complete-rollout advantage populations
        # independently. Raw advantages and return targets remain untouched.
        reward_advantages = self.advantages[:valid_size]
        cost_advantages = self.cost_advantages[:valid_size]
        self.reward_adv_mean = float(np.mean(reward_advantages, dtype=np.float64))
        self.reward_adv_std = float(np.std(reward_advantages, dtype=np.float64))
        self.constraint_adv_mean = float(
            np.mean(cost_advantages, dtype=np.float64)
        )
        self.constraint_adv_std = float(
            np.std(cost_advantages, dtype=np.float64)
        )
        self.normalized_advantages[:valid_size] = (
            reward_advantages - self.reward_adv_mean
        ) / (self.reward_adv_std + 1.0e-8)
        self.normalized_cost_advantages[:valid_size] = (
            cost_advantages - self.constraint_adv_mean
        ) / (self.constraint_adv_std + 1.0e-8)

    def get(
        self, batch_size: int | None = None
    ) -> Generator[CRPORolloutBufferSamples, None, None]:
        valid_size = self.valid_size
        if valid_size < 1:
            raise ValueError("cannot sample an empty rollout")
        self.assert_costs_ready()
        sample_count = valid_size * self.n_envs
        indices = np.random.permutation(sample_count)
        if not self.generator_ready:
            for name in (
                "observations",
                "critic_taus",
                "actions",
                "values",
                "cost_values",
                "log_probs",
                "advantages",
                "cost_advantages",
                "normalized_advantages",
                "normalized_cost_advantages",
                "returns",
                "cost_returns",
            ):
                self.__dict__[name] = self.swap_and_flatten(
                    self.__dict__[name][:valid_size]
                )
            self.generator_ready = True
        batch_size = batch_size or sample_count
        for start in range(0, sample_count, batch_size):
            batch = indices[start : start + batch_size]
            data = (
                self.observations[batch],
                self.critic_taus[batch].flatten(),
                self.actions[batch].astype(np.float32, copy=False),
                self.values[batch].flatten(),
                self.cost_values[batch].flatten(),
                self.log_probs[batch].flatten(),
                self.advantages[batch].flatten(),
                self.cost_advantages[batch].flatten(),
                self.normalized_advantages[batch].flatten(),
                self.normalized_cost_advantages[batch].flatten(),
                self.returns[batch].flatten(),
                self.cost_returns[batch].flatten(),
            )
            yield CRPORolloutBufferSamples(*tuple(map(self.to_torch, data)))
