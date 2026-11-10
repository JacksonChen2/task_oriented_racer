"""The configured task loss and its time-weighted CRPO constraint."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .schemas import TaskMetrics


@dataclass(frozen=True)
class TaskLossWeights:
    alpha_traj: float = 0.30
    alpha_cov: float = 0.30
    alpha_red: float = 0.10
    alpha_map: float = 0.30

    def __post_init__(self) -> None:
        if min(
            self.alpha_traj,
            self.alpha_cov,
            self.alpha_red,
            self.alpha_map,
        ) < 0.0:
            raise ValueError("task-loss weights must be non-negative")
        total = self.alpha_traj + self.alpha_cov + self.alpha_red + self.alpha_map
        if abs(total - 1.0) > 1.0e-9:
            raise ValueError("task-loss weights must sum to 1")

    def total(self, metrics: TaskMetrics) -> float:
        return float(
            self.alpha_traj * metrics.d_traj
            + self.alpha_cov * metrics.d_cov
            + self.alpha_red * metrics.d_red
            + self.alpha_map * metrics.d_map
        )


class TaskLossTracker:
    """Integrate task loss over simulation time.

    The constraint at each transition is the trapezoidal time average of
    ``L_task`` since reset.  The CRPO stage cost is that interval's trapezoid
    area divided by the fixed episode duration.  Consequently, a complete
    fixed-duration episode return equals its time-weighted mean task loss.
    """

    def __init__(
        self,
        weights: TaskLossWeights,
        episode_duration_s: float = 300.0,
    ) -> None:
        if not math.isfinite(episode_duration_s) or episode_duration_s <= 0.0:
            raise ValueError("episode duration must be finite and positive")
        self.weights = weights
        self.episode_duration_s = float(episode_duration_s)
        self.initial = 0.0
        self.previous = 0.0
        self.previous_time_s = 0.0
        self.integrated_loss = 0.0
        self.elapsed_time_s = 0.0
        self.time_weighted_mean = 0.0
        self.interval_previous_loss = 0.0
        self.interval_delta_time_s = 0.0
        self.interval_mean_loss = 0.0
        self.interval_constraint_cost = 0.0
        self.sum_cost = 0.0

    def reset(self, metrics: TaskMetrics) -> float:
        self.initial = self.weights.total(metrics)
        self.previous = self.initial
        self.previous_time_s = float(metrics.task_time_s)
        self.integrated_loss = 0.0
        self.elapsed_time_s = 0.0
        self.time_weighted_mean = self.initial
        self.interval_previous_loss = self.initial
        self.interval_delta_time_s = 0.0
        self.interval_mean_loss = self.initial
        self.interval_constraint_cost = 0.0
        self.sum_cost = 0.0
        return self.initial

    def update(self, metrics: TaskMetrics) -> tuple[float, float]:
        current = self.weights.total(metrics)
        current_time_s = float(metrics.task_time_s)
        delta_time_s = current_time_s - self.previous_time_s
        if delta_time_s < -1.0e-9:
            raise ValueError(
                "task_time_s must be monotonic for time-weighted task loss: "
                f"previous={self.previous_time_s}, current={current_time_s}"
            )
        delta_time_s = max(0.0, delta_time_s)
        previous_loss = self.previous
        interval_mean = 0.5 * (previous_loss + current)
        constraint_cost = (
            interval_mean * delta_time_s / self.episode_duration_s
        )
        if delta_time_s > 0.0:
            self.integrated_loss += interval_mean * delta_time_s
            self.elapsed_time_s += delta_time_s
            self.time_weighted_mean = (
                self.integrated_loss / self.elapsed_time_s
            )
        self.interval_previous_loss = previous_loss
        self.interval_delta_time_s = delta_time_s
        self.interval_mean_loss = interval_mean
        self.interval_constraint_cost = constraint_cost
        self.previous = current
        self.previous_time_s = current_time_s
        self.sum_cost += constraint_cost
        return current, constraint_cost

    @property
    def constraint_value(self) -> float:
        return float(self.time_weighted_mean)

    @property
    def telescoping_error(self) -> float:
        return float(
            self.sum_cost
            - self.integrated_loss / self.episode_duration_s
        )
