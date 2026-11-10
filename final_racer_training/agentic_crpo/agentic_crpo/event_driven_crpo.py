"""CRPO components for the opt-in event-driven one-shot experiment.

The fixed-clock trainer continues to use :class:`CRPORolloutBuffer`.  This
module keeps the same policy/update equations and only replaces the temporal
part of GAE: each transition's discount and trace decay are scaled by its
measured simulated duration relative to the existing 100 ms decision period.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import GymEnv

from .crpo_buffer import CRPORolloutBuffer
from .llm_prior_crpo_ppo import LLMPriorCRPOPPO


FIXED_CLOCK_REFERENCE_DELTA_T_S = 0.1


class EventDrivenCRPORolloutBuffer(CRPORolloutBuffer):
    """CRPO rollout buffer with duration-aware reward and constraint GAE."""

    def __init__(
        self,
        *args: Any,
        reference_delta_t_s: float = FIXED_CLOCK_REFERENCE_DELTA_T_S,
        **kwargs: Any,
    ) -> None:
        self.reference_delta_t_s = float(reference_delta_t_s)
        if (
            not math.isfinite(self.reference_delta_t_s)
            or self.reference_delta_t_s <= 0.0
        ):
            raise ValueError("reference_delta_t_s must be finite and positive")
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.delta_times_s = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float64
        )

    def add(self, *args: Any, **kwargs: Any) -> None:
        position = self.pos
        start_times = np.asarray(
            kwargs.get("sim_time", 0.0), dtype=np.float64
        ).reshape(-1)
        end_times = np.asarray(
            kwargs.get("sim_timestamp", 0.0), dtype=np.float64
        ).reshape(-1)
        if start_times.size == 1 and self.n_envs > 1:
            start_times = np.repeat(start_times, self.n_envs)
        if end_times.size == 1 and self.n_envs > 1:
            end_times = np.repeat(end_times, self.n_envs)
        if start_times.size != self.n_envs or end_times.size != self.n_envs:
            raise ValueError("event-driven timestamps do not match n_envs")
        delta_times_s = end_times - start_times
        # The deliberately transport-free mock backend uses negative action
        # versions and has no result publication timestamp. Keep the existing
        # ``--mock-smoke`` entry useful without relaxing production checks.
        source_versions = np.asarray(
            kwargs.get("state_version", 0), dtype=np.int64
        ).reshape(-1)
        if source_versions.size == 1 and self.n_envs > 1:
            source_versions = np.repeat(source_versions, self.n_envs)
        mock_without_event_clock = (
            source_versions.size == self.n_envs
        ) & (source_versions < 0) & (delta_times_s <= 0.0)
        delta_times_s = np.where(
            mock_without_event_clock,
            self.reference_delta_t_s,
            delta_times_s,
        )
        if (
            not np.all(np.isfinite(delta_times_s))
            or np.any(delta_times_s <= 0.0)
        ):
            raise FloatingPointError(
                "event-driven rollout delta_t must be finite and positive"
            )
        super().add(*args, **kwargs)
        self.delta_times_s[position] = delta_times_s

    def assert_finite(self) -> None:
        super().assert_finite()
        delta_times_s = self.delta_times_s[: self.valid_size]
        if (
            not np.all(np.isfinite(delta_times_s))
            or np.any(delta_times_s <= 0.0)
        ):
            raise FloatingPointError(
                "CRPO update refused: event delta_t is invalid"
            )

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
        delta_times_s = self.delta_times_s[:valid_size]
        if (
            not np.all(np.isfinite(delta_times_s))
            or np.any(delta_times_s <= 0.0)
        ):
            raise FloatingPointError(
                "cannot compute event-driven GAE with invalid delta_t"
            )

        next_reward_values = last_values.detach().cpu().numpy().flatten()
        next_cost_values = last_cost_values.detach().cpu().numpy().flatten()
        last_reward_gae: float | np.ndarray = 0.0
        last_cost_gae: float | np.ndarray = 0.0
        for step in reversed(range(valid_size)):
            if step == valid_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                following_reward = next_reward_values
                following_cost = next_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                following_reward = self.values[step + 1]
                following_cost = self.cost_values[step + 1]

            duration_ratio = delta_times_s[step] / self.reference_delta_t_s
            reward_gamma = np.power(self.gamma, duration_ratio)
            reward_lambda = np.power(self.gae_lambda, duration_ratio)
            cost_gamma = np.power(self.gamma_cost, duration_ratio)
            cost_lambda = np.power(self.gae_lambda_cost, duration_ratio)

            reward_delta = (
                self.rewards[step]
                + reward_gamma * following_reward * next_non_terminal
                - self.values[step]
            )
            last_reward_gae = (
                reward_delta
                + reward_gamma
                * reward_lambda
                * next_non_terminal
                * last_reward_gae
            )
            self.advantages[step] = last_reward_gae

            cost_delta = (
                self.costs[step]
                + cost_gamma * following_cost * next_non_terminal
                - self.cost_values[step]
            )
            last_cost_gae = (
                cost_delta
                + cost_gamma
                * cost_lambda
                * next_non_terminal
                * last_cost_gae
            )
            self.cost_advantages[step] = last_cost_gae

        self.returns[:valid_size] = (
            self.advantages[:valid_size] + self.values[:valid_size]
        )
        self.cost_returns[:valid_size] = (
            self.cost_advantages[:valid_size]
            + self.cost_values[:valid_size]
        )
        reward_advantages = self.advantages[:valid_size]
        cost_advantages = self.cost_advantages[:valid_size]
        self.reward_adv_mean = float(
            np.mean(reward_advantages, dtype=np.float64)
        )
        self.reward_adv_std = float(
            np.std(reward_advantages, dtype=np.float64)
        )
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


class EventDrivenOneShotLLMPriorCRPOPPO(LLMPriorCRPOPPO):
    """LLM-prior CRPO using the isolated duration-aware rollout buffer."""

    rollout_buffer: EventDrivenCRPORolloutBuffer

    def __init__(
        self,
        *args: Any,
        event_reference_delta_t_s: float = FIXED_CLOCK_REFERENCE_DELTA_T_S,
        **kwargs: Any,
    ) -> None:
        self.event_reference_delta_t_s = float(event_reference_delta_t_s)
        super().__init__(*args, **kwargs)
        if hasattr(self, "n_envs"):
            self._install_event_rollout_buffer()

    def _install_event_rollout_buffer(self) -> None:
        self.rollout_buffer_class = EventDrivenCRPORolloutBuffer
        self.rollout_buffer_kwargs = {
            "gamma_cost": self.gamma_cost,
            "gae_lambda_cost": self.gae_lambda_cost,
            "reference_delta_t_s": self.event_reference_delta_t_s,
        }
        self.rollout_buffer = EventDrivenCRPORolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

    def _bootstrap_discount(
        self, gamma: float, info: dict[str, Any]
    ) -> float:
        delta_t_s = float(info.get("transition_sim_time_s", float("nan")))
        if not math.isfinite(delta_t_s) or delta_t_s <= 0.0:
            raise FloatingPointError(
                "event-driven terminal bootstrap has invalid delta_t"
            )
        return float(
            np.power(
                float(gamma),
                delta_t_s / self.event_reference_delta_t_s,
            )
        )

    @classmethod
    def load(
        cls,
        path: str | Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> "EventDrivenOneShotLLMPriorCRPOPPO":
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        if not isinstance(model, cls):
            raise TypeError("restored model is not the event-driven variant")
        if not hasattr(model, "event_reference_delta_t_s"):
            model.event_reference_delta_t_s = (
                FIXED_CLOCK_REFERENCE_DELTA_T_S
            )
        model._install_event_rollout_buffer()
        return model

    def train(self) -> None:
        size = self.rollout_buffer.valid_size
        if size:
            delta_times_s = self.rollout_buffer.delta_times_s[:size]
            self.logger.record(
                "event/delta_t_mean_s", float(np.mean(delta_times_s))
            )
            self.logger.record(
                "event/delta_t_min_s", float(np.min(delta_times_s))
            )
            self.logger.record(
                "event/delta_t_max_s", float(np.max(delta_times_s))
            )
            self.logger.record(
                "event/reference_delta_t_s",
                self.event_reference_delta_t_s,
            )
        super().train()
