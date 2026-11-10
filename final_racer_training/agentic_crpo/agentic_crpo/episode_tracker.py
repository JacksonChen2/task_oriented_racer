"""Mission-level constraint estimates that survive rollout boundaries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    final_task_loss: float
    time_weighted_task_loss: float
    initial_task_loss: float
    sum_cost: float
    total_bs_resource: float
    episode_length: int
    constraint_elapsed_time_s: float
    constraint_episode_duration_s: float
    telescoping_error: float


class EpisodeCostTracker:
    def __init__(
        self,
        n_envs: int,
        window: int = 20,
        telescoping_tolerance: float = 1.0e-6,
        check_telescoping: bool = True,
    ) -> None:
        if n_envs < 1 or window < 1:
            raise ValueError("invalid episode tracker dimensions")
        self.n_envs = n_envs
        self.telescoping_tolerance = float(telescoping_tolerance)
        self.check_telescoping = bool(check_telescoping)
        if self.telescoping_tolerance < 0.0:
            raise ValueError("telescoping tolerance must be non-negative")
        self.records: deque[EpisodeRecord] = deque(maxlen=window)
        self.costs = np.zeros(n_envs, np.float64)
        self.resources = np.zeros(n_envs, np.float64)
        self.lengths = np.zeros(n_envs, np.int64)
        self.initial_losses = np.zeros(n_envs, np.float64)
        self.constraint_values = np.zeros(n_envs, np.float64)
        self.elapsed_times = np.zeros(n_envs, np.float64)
        self.episode_durations = np.full(n_envs, 300.0, np.float64)

    def reset_active(self) -> None:
        """Discard partial mission state while retaining completed records."""

        self.costs = np.zeros(self.n_envs, np.float64)
        self.resources = np.zeros(self.n_envs, np.float64)
        self.lengths = np.zeros(self.n_envs, np.int64)
        self.initial_losses = np.zeros(self.n_envs, np.float64)
        self.constraint_values = np.zeros(self.n_envs, np.float64)
        self.elapsed_times = np.zeros(self.n_envs, np.float64)
        self.episode_durations = np.full(self.n_envs, 300.0, np.float64)

    def observe(
        self,
        costs: np.ndarray,
        dones: np.ndarray,
        infos: list[dict],
    ) -> None:
        for index in range(self.n_envs):
            info = infos[index]
            if self.lengths[index] == 0:
                self.initial_losses[index] = float(info.get("L_task_initial", 0.0))
                self.constraint_values[index] = self.initial_losses[index]
            self.costs[index] += float(costs[index])
            self.resources[index] += float(info.get("C_BS", 0.0))
            self.lengths[index] += 1
            self.constraint_values[index] = float(
                info.get(
                    "L_task_time_weighted",
                    info.get("constraint_value", 0.0),
                )
            )
            self.elapsed_times[index] = float(
                info.get("constraint_elapsed_time_s", self.elapsed_times[index])
            )
            self.episode_durations[index] = float(
                info.get(
                    "constraint_episode_duration_s",
                    self.episode_durations[index],
                )
            )
            if not dones[index]:
                continue
            final_loss = float(info.get("L_task", self.initial_losses[index] + self.costs[index]))
            time_weighted_loss = float(self.constraint_values[index])
            expected_return = (
                time_weighted_loss
                * self.elapsed_times[index]
                / self.episode_durations[index]
            )
            error = (
                self.costs[index] - expected_return
                if self.check_telescoping
                else 0.0
            )
            if self.check_telescoping and abs(error) > self.telescoping_tolerance:
                warnings.warn(
                    "CRPO task cost failed the telescoping check: "
                    f"error={error:.3e}, tolerance={self.telescoping_tolerance:.3e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.records.append(
                EpisodeRecord(
                    final_task_loss=final_loss,
                    time_weighted_task_loss=time_weighted_loss,
                    initial_task_loss=float(self.initial_losses[index]),
                    sum_cost=float(self.costs[index]),
                    total_bs_resource=float(self.resources[index]),
                    episode_length=int(self.lengths[index]),
                    constraint_elapsed_time_s=float(self.elapsed_times[index]),
                    constraint_episode_duration_s=float(
                        self.episode_durations[index]
                    ),
                    telescoping_error=float(error),
                )
            )
            self.costs[index] = 0.0
            self.resources[index] = 0.0
            self.lengths[index] = 0
            self.initial_losses[index] = 0.0
            self.constraint_values[index] = 0.0
            self.elapsed_times[index] = 0.0
            self.episode_durations[index] = 300.0

    def estimate(
        self, cost_value_estimate: np.ndarray | None = None
    ) -> tuple[float, str]:
        active = self.lengths > 0
        if np.any(active):
            return (
                float(np.mean(self.constraint_values[active])),
                "partial_time_weighted_mean",
            )
        if self.records:
            return (
                float(self.records[-1].time_weighted_task_loss),
                "time_weighted_mean",
            )
        if cost_value_estimate is None:
            return 0.0, "partial_time_weighted_mean"
        estimate = self.initial_losses + self.costs + np.asarray(cost_value_estimate)
        return float(np.mean(estimate)), "critic_estimate"
