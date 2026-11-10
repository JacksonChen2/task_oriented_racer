"""Task degradations using a perfect reference plus live map synchronization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import TaskMetrics


_EPSILON = 1.0e-8


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("task metric produced NaN or Inf")
    return float(np.clip(value, 0.0, 1.0))


def _nested_statistics(result: dict[str, Any]) -> dict[str, Any]:
    communication = result.get("communication", {})
    if not isinstance(communication, dict):
        return {}
    statistics = communication.get("statistics", {})
    return statistics if isinstance(statistics, dict) else {}


def _first_history(
    containers: Iterable[dict[str, Any]], names: tuple[str, ...]
) -> list[dict[str, Any]]:
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, list) and value:
                return [item for item in value if isinstance(item, dict)]
    return []


def _history_time(item: dict[str, Any]) -> float:
    """Prefer a persisted canonical task time while accepting old references."""

    return float(
        item.get("task_time_s", item.get("time_s", item.get("sim_time_s", 0.0)))
    )


class ReferenceTaskMetricEvaluator:
    """Compute fixed-horizon loss metrics outside both policy observations."""

    def __init__(
        self,
        reference_result: str | None,
        n_uavs: int,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
        redundancy_cell_m: float = 1.0,
        fixed_exploration_time_s: float = 300.0,
        require_auxiliary_reference_metrics: bool = True,
        auxiliary_reference_result: str | None = None,
        require_explicit_task_clock: bool = False,
        variable_task_clock: bool = False,
    ) -> None:
        self.n_uavs = int(n_uavs)
        lower = np.asarray(workspace_min, dtype=np.float64)
        upper = np.asarray(workspace_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("workspace bounds must contain three increasing values")
        self.workspace_diameter_m = float(np.linalg.norm(upper - lower))
        self.redundancy_cell_m = max(float(redundancy_cell_m), 1.0e-6)
        self.fixed_exploration_time_s = max(
            float(fixed_exploration_time_s), 1.0e-6
        )
        self.require_auxiliary_reference_metrics = bool(
            require_auxiliary_reference_metrics
        )
        self.require_explicit_task_clock = bool(require_explicit_task_clock)
        self.variable_task_clock = bool(variable_task_clock)

        self.reference_times = np.empty(0, np.float64)
        self.reference_positions = np.empty((0, self.n_uavs, 3), np.float64)
        self.coverage_times = np.empty(0, np.float64)
        self.coverage_values = np.empty(0, np.float64)
        self.redundancy_times = np.empty(0, np.float64)
        self.redundancy_values = np.empty(0, np.float64)
        self.map_iou_times = np.empty(0, np.float64)
        self.map_iou_values = np.empty(0, np.float64)
        self.reference_elapsed = self.fixed_exploration_time_s
        self.reference_source: Path | None = None
        self.auxiliary_reference_source: Path | None = None
        self.auxiliary_reference_elapsed: float | None = None
        self.auxiliary_metrics_from_reference = False
        if reference_result:
            self._load(reference_result)
        if auxiliary_reference_result:
            self._load_auxiliary(auxiliary_reference_result)
        self.reset()

    def _load(self, path: str) -> None:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError("perfect reference result must be a JSON object")
        self.reference_source = source
        metrics = result.get("metrics", result)
        if not isinstance(metrics, dict):
            metrics = result
        statistics = _nested_statistics(result)
        containers = (metrics, statistics, result)

        trajectory_history = metrics.get("trajectory_history", [])
        valid_trajectory = [
            item
            for item in trajectory_history
            if isinstance(item, dict)
            and len(item.get("positions", [])) == self.n_uavs
        ]
        if valid_trajectory:
            valid_trajectory.sort(key=_history_time)
            self.reference_times = np.asarray(
                [_history_time(item) for item in valid_trajectory], np.float64
            )
            self.reference_positions = np.asarray(
                [item["positions"] for item in valid_trajectory], np.float64
            )
        self.reference_elapsed = max(
            float(metrics.get("elapsed", self.fixed_exploration_time_s)), 1.0e-6
        )

        joint_coverage_history = metrics.get(
            "mapping_coverage_joint_history", []
        )
        if joint_coverage_history:
            self.coverage_times, self.coverage_values = self._scalar_history(
                joint_coverage_history,
                ("ratio", "coverage", "mapping_coverage_joint"),
            )
        else:
            # Legacy results only contain per-UAV scalar histories. Their
            # former "joint" metric was the maximum individual coverage and
            # remains readable for provenance-compatible old checkpoints.
            self.coverage_times, self.coverage_values = (
                self._team_coverage_curve(
                    metrics.get("mapping_coverage_history", [])
                )
            )
        quality_history = _first_history(
            containers,
            (
                "task_quality_history",
                "constraint_reference_history",
                "reference_task_history",
            ),
        )
        redundancy_history = _first_history(
            containers, ("redundancy_history", "redundant_exploration_history")
        )
        map_history = _first_history(
            containers, ("bs_global_map_iou_history", "map_iou_history")
        )
        if quality_history:
            redundancy_history = redundancy_history or quality_history
            map_history = map_history or quality_history
        self.redundancy_times, self.redundancy_values = self._scalar_history(
            redundancy_history,
            ("redundancy", "redundant_exploration_ratio", "R_red"),
        )
        self.map_iou_times, self.map_iou_values = self._scalar_history(
            map_history, ("map_iou", "bs_global_map_iou", "IoU_map")
        )
        # D_map is computed from the live joint/BS coverage pair, so the
        # perfect reference only needs an auxiliary redundancy curve.
        self.auxiliary_metrics_from_reference = bool(
            self.redundancy_times.size
        )

        # Old references did not persist these histories. These reconstructions
        # remain available only for mock/diagnostic compatibility; strict real
        # training rejects them in ``validate_reference``.
        if not self.redundancy_times.size and valid_trajectory:
            self.redundancy_times, self.redundancy_values = (
                self._trajectory_redundancy_curve()
            )
        if not self.map_iou_times.size and self.coverage_times.size:
            self.map_iou_times = self.coverage_times.copy()
            self.map_iou_values = self.coverage_values.copy()

    def _load_auxiliary(self, path: str) -> None:
        """Load only missing loss-only curves from a separately provenanced run.

        The main reference remains authoritative for trajectories, coverage, and
        elapsed time.  This split exists for legacy perfect-communication results
        that predate persistence of exploration-redundancy curves.
        """
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError("auxiliary reference result must be a JSON object")
        metrics = result.get("metrics", result)
        if not isinstance(metrics, dict):
            metrics = result
        statistics = _nested_statistics(result)
        containers = (metrics, statistics, result)
        quality_history = _first_history(
            containers,
            (
                "task_quality_history",
                "constraint_reference_history",
                "reference_task_history",
            ),
        )
        redundancy_history = _first_history(
            containers, ("redundancy_history", "redundant_exploration_history")
        )
        map_history = _first_history(
            containers, ("bs_global_map_iou_history", "map_iou_history")
        )
        if quality_history:
            redundancy_history = redundancy_history or quality_history
            map_history = map_history or quality_history
        redundancy_times, redundancy_values = self._scalar_history(
            redundancy_history,
            ("redundancy", "redundant_exploration_ratio", "R_red"),
        )
        map_iou_times, map_iou_values = self._scalar_history(
            map_history, ("map_iou", "bs_global_map_iou", "IoU_map")
        )
        if not redundancy_times.size:
            raise ValueError(
                "auxiliary reference must contain a redundancy history"
            )
        self.redundancy_times = redundancy_times
        self.redundancy_values = redundancy_values
        if map_iou_times.size:
            self.map_iou_times = map_iou_times
            self.map_iou_values = map_iou_values
        self.auxiliary_reference_source = source
        self.auxiliary_reference_elapsed = max(
            float(metrics.get("elapsed", self.fixed_exploration_time_s)), 1.0e-6
        )
        self.auxiliary_metrics_from_reference = True

    def _team_coverage_curve(
        self, history: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        records = [item for item in history if isinstance(item, dict)]
        if not records:
            return np.empty(0), np.empty(0)
        events: list[tuple[float, int | None, float]] = []
        for item in records:
            try:
                stamp = _history_time(item)
                ratio = _clip01(float(item["ratio"]))
                drone = item.get("drone_id")
                drone_id = None if drone is None else int(drone)
            except (KeyError, TypeError, ValueError):
                continue
            if drone_id is not None and not 0 <= drone_id < self.n_uavs:
                continue
            events.append((stamp, drone_id, ratio))
        if not events:
            return np.empty(0), np.empty(0)
        events.sort(key=lambda item: item[0])
        per_uav = np.zeros(self.n_uavs, np.float64)
        times = [0.0]
        values = [0.0]
        for stamp, drone_id, ratio in events:
            if drone_id is None:
                joint = max(values[-1], ratio)
            else:
                per_uav[drone_id] = max(per_uav[drone_id], ratio)
                joint = max(values[-1], float(np.max(per_uav)))
            if stamp == times[-1]:
                values[-1] = max(values[-1], joint)
            else:
                times.append(stamp)
                values.append(joint)
        return np.asarray(times), np.maximum.accumulate(np.asarray(values))

    @staticmethod
    def _scalar_history(
        history: list[dict[str, Any]], value_names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs: list[tuple[float, float]] = []
        for item in history:
            try:
                stamp = _history_time(item)
            except (TypeError, ValueError):
                continue
            value = next((item[name] for name in value_names if name in item), None)
            if value is None:
                continue
            try:
                pairs.append((stamp, _clip01(float(value))))
            except (TypeError, ValueError):
                continue
        if not pairs:
            return np.empty(0), np.empty(0)
        pairs.sort(key=lambda item: item[0])
        times: list[float] = []
        values: list[float] = []
        for stamp, value in pairs:
            if times and stamp == times[-1]:
                values[-1] = value
            else:
                times.append(stamp)
                values.append(value)
        return np.asarray(times), np.asarray(values)

    def _trajectory_redundancy_curve(self) -> tuple[np.ndarray, np.ndarray]:
        per_uav: list[set[tuple[int, int, int]]] = [
            set() for _ in range(self.n_uavs)
        ]
        values = []
        for positions in self.reference_positions:
            for drone_id, position in enumerate(positions):
                cell = tuple(
                    np.floor(position / self.redundancy_cell_m).astype(int)
                )
                per_uav[drone_id].add(cell)
            denominator = sum(len(items) for items in per_uav)
            union = set().union(*per_uav)
            values.append(
                0.0 if denominator == 0 else 1.0 - len(union) / denominator
            )
        return self.reference_times.copy(), np.asarray(values, dtype=np.float64)

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_times.size and self.coverage_times.size)

    @property
    def has_complete_reference(self) -> bool:
        if not self.has_reference:
            return False
        return (
            not self.require_auxiliary_reference_metrics
            or self.auxiliary_metrics_from_reference
        )

    def validate_reference(self) -> None:
        if not self.has_reference:
            raise FileNotFoundError(
                "perfect reference must contain trajectory_history and "
                "mapping_coverage_history"
            )
        if (
            self.require_auxiliary_reference_metrics
            and not self.auxiliary_metrics_from_reference
        ):
            raise FileNotFoundError(
                "strict fixed-horizon constraint requires perfect-reference "
                "redundancy_history; rerun the perfect-communication reference "
                "with task diagnostics enabled"
            )
        tolerance_s = max(0.1, 0.01 * self.fixed_exploration_time_s)
        if abs(self.reference_elapsed - self.fixed_exploration_time_s) > tolerance_s:
            raise ValueError(
                "perfect reference elapsed time does not match the configured "
                f"fixed horizon: reference={self.reference_elapsed:.3f}s, "
                f"configured={self.fixed_exploration_time_s:.3f}s"
            )
        if (
            self.auxiliary_reference_elapsed is not None
            and abs(
                self.auxiliary_reference_elapsed - self.fixed_exploration_time_s
            )
            > tolerance_s
        ):
            raise ValueError(
                "auxiliary reference elapsed time does not match the configured "
                "fixed horizon: "
                f"reference={self.auxiliary_reference_elapsed:.3f}s, "
                f"configured={self.fixed_exploration_time_s:.3f}s"
            )

    def reset(self) -> None:
        self.trajectory_distance_sum_m = 0.0
        self.trajectory_position_count = 0
        self.coverage_deficit_area = 0.0
        self.coverage_reference_area = 0.0
        self.last_metric_time: float | None = None
        self.last_actual_coverage = 0.0
        self.last_task_step: int | None = None
        self.last_task_time_s: float | None = None
        self.last_task_metrics: TaskMetrics | None = None
        self.actual_visited_cells: list[set[tuple[int, int, int]]] = [
            set() for _ in range(self.n_uavs)
        ]

    @staticmethod
    def _step_value(times: np.ndarray, values: np.ndarray, stamp: float) -> float:
        if not times.size:
            return 0.0
        index = int(np.searchsorted(times, stamp, side="right") - 1)
        return float(values[max(0, index)])

    def _reference_position(self, stamp: float) -> np.ndarray | None:
        if not self.reference_times.size:
            return None
        stamp = float(np.clip(stamp, self.reference_times[0], self.reference_times[-1]))
        right = int(np.searchsorted(self.reference_times, stamp, side="right"))
        if right == 0:
            return self.reference_positions[0]
        if right >= self.reference_times.size:
            return self.reference_positions[-1]
        left = right - 1
        span = self.reference_times[right] - self.reference_times[left]
        fraction = 0.0 if span <= 0.0 else (stamp - self.reference_times[left]) / span
        return (
            (1.0 - fraction) * self.reference_positions[left]
            + fraction * self.reference_positions[right]
        )

    def _integrate_coverage_interval(self, start: float, end: float) -> None:
        if end <= start:
            return
        internal = self.coverage_times[
            (self.coverage_times > start) & (self.coverage_times < end)
        ]
        points = np.concatenate(([start], internal, [end]))
        for left, right in zip(points[:-1], points[1:]):
            coverage_pc = self._step_value(
                self.coverage_times, self.coverage_values, float(left)
            )
            duration = float(right - left)
            self.coverage_reference_area += duration * coverage_pc
            self.coverage_deficit_area += duration * max(
                0.0, coverage_pc - self.last_actual_coverage
            )

    def _fallback_redundancy(self, positions: np.ndarray) -> float:
        for drone_id, position in enumerate(positions):
            cell = tuple(np.floor(position / self.redundancy_cell_m).astype(int))
            self.actual_visited_cells[drone_id].add(cell)
        denominator = sum(len(items) for items in self.actual_visited_cells)
        union = set().union(*self.actual_visited_cells)
        return 0.0 if denominator == 0 else 1.0 - len(union) / denominator

    @staticmethod
    def _payload_metric(
        communication: dict[str, Any], mission: dict[str, Any], names: tuple[str, ...]
    ) -> float | None:
        for container in (mission, communication):
            for name in names:
                if name not in container:
                    continue
                try:
                    value = float(container[name])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value >= 0.0:
                    return value
        return None

    def _task_clock(
        self, communication: dict[str, Any], mission: dict[str, Any]
    ) -> tuple[int, float]:
        """Return the producer-owned clock used for every reference lookup.

        Shared-memory training requires both fields. Legacy/mock backends may
        derive the same clock from simulation time for compatibility, but no
        model cycle ID or shared-memory version is ever used here.
        """

        raw_step = communication.get("task_step", mission.get("task_step"))
        raw_time = communication.get("task_time_s", mission.get("task_time_s"))
        if self.require_explicit_task_clock and (
            raw_step is None or raw_time is None
        ):
            raise ValueError(
                "shared-memory task metrics require explicit task_step and "
                "task_time_s"
            )
        duration = float(
            communication.get(
                "task_step_duration_s",
                mission.get(
                    "task_step_duration_s",
                    communication.get("rl_decision_interval_s", 0.1),
                ),
            )
        )
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("task_step_duration_s must be finite and positive")
        if raw_time is None:
            raw_time = communication.get(
                "sim_time_s", mission.get("sim_time_s", 0.0)
            )
        task_time_s = float(raw_time)
        if not np.isfinite(task_time_s) or task_time_s < 0.0:
            raise ValueError("task_time_s must be finite and non-negative")
        task_step = (
            int(raw_step)
            if raw_step is not None
            else int(round(task_time_s / duration))
        )
        canonical_time = np.isclose(
            task_time_s,
            task_step * duration,
            rtol=0.0,
            atol=max(1.0e-8, duration * 1.0e-6),
        )
        if task_step < 0 or (
            not self.variable_task_clock and not canonical_time
        ):
            raise ValueError(
                "task_step/task_time_s are inconsistent with the canonical "
                "task-step duration"
            )
        if self.last_task_step is not None:
            assert self.last_task_time_s is not None
            if (
                task_step < self.last_task_step
                or task_time_s < self.last_task_time_s
            ):
                raise ValueError("task clock moved backwards")
            if task_step == self.last_task_step and not np.isclose(
                task_time_s,
                self.last_task_time_s,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError("one task_step was published with two task times")
        return task_step, task_time_s

    def evaluate(
        self, communication: dict[str, Any], mission: dict[str, Any]
    ) -> TaskMetrics:
        task_step, stamp = self._task_clock(communication, mission)
        if task_step == self.last_task_step:
            if self.last_task_metrics is None:
                raise RuntimeError("task metric cache is missing")
            return self.last_task_metrics
        positions = np.asarray(
            communication.get("positions", mission.get("positions")), np.float64
        )
        stamp = float(np.clip(stamp, 0.0, self.fixed_exploration_time_s))
        if positions.shape != (self.n_uavs, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("task metric evaluator requires finite N x 3 positions")

        reference = self._reference_position(stamp)
        trajectory_deviation_m = 0.0
        if reference is not None:
            distances = np.linalg.norm(positions - reference, axis=1)
            trajectory_deviation_m = float(np.mean(distances))
            self.trajectory_distance_sum_m += float(np.sum(distances))
            self.trajectory_position_count += self.n_uavs
        d_traj = _clip01(
            self.trajectory_distance_sum_m
            / max(
                _EPSILON,
                self.trajectory_position_count * self.workspace_diameter_m,
            )
        )

        coverage = _clip01(
            float(mission.get("coverage", communication.get("coverage", 0.0)))
        )
        if self.last_metric_time is not None:
            self._integrate_coverage_interval(self.last_metric_time, stamp)
        self.last_metric_time = stamp
        self.last_actual_coverage = coverage
        coverage_pc = self._step_value(self.coverage_times, self.coverage_values, stamp)
        d_cov = _clip01(
            self.coverage_deficit_area
            / (self.coverage_reference_area + _EPSILON)
        )

        redundancy = self._payload_metric(
            communication,
            mission,
            ("redundant_exploration_ratio", "redundancy"),
        )
        if redundancy is None:
            if self.require_auxiliary_reference_metrics:
                raise ValueError(
                    "strict constraint requires redundant_exploration_ratio in "
                    "the loss-only telemetry"
                )
            redundancy = self._fallback_redundancy(positions)
        redundancy_max = 1.0 - 1.0 / self.n_uavs
        redundancy = min(_clip01(redundancy), redundancy_max)
        redundancy_pc = min(
            _clip01(
                self._step_value(
                    self.redundancy_times, self.redundancy_values, stamp
                )
            ),
            redundancy_max,
        )
        d_red = _clip01(
            max(0.0, redundancy - redundancy_pc)
            / (redundancy_max - redundancy_pc + _EPSILON)
        )

        bs_coverage = self._payload_metric(
            communication,
            mission,
            ("bs_global_map_coverage", "bs_map_coverage"),
        )
        if bs_coverage is None:
            if self.require_auxiliary_reference_metrics:
                raise ValueError(
                    "strict constraint requires bs_global_map_coverage using "
                    "the same planning-box denominator as joint coverage"
                )
            summary = communication.get(
                "bs_map_summary", communication.get("map_summary", {})
            )
            known = float(summary.get("bs_known_chunks", 0.0))
            local = np.asarray(summary.get("local_known_chunks", []), dtype=float)
            denominator = float(np.max(local)) if local.size else 1.0
            bs_coverage = known / max(1.0, denominator)
        bs_coverage = _clip01(bs_coverage)

        # The map term measures how much of the team's currently known map has
        # not reached the BS at the same canonical task time. Both values are
        # planning-box known-voxel fractions, so no perfect-reference
        # normalization is applied.
        d_map = _clip01(max(0.0, coverage - bs_coverage))

        # Preserve the former IoU diagnostics for existing dashboards. They no
        # longer enter D_map or L_task.
        map_iou = self._payload_metric(
            communication,
            mission,
            ("bs_global_map_iou", "map_iou"),
        )
        map_iou = bs_coverage if map_iou is None else _clip01(map_iou)
        map_iou_pc = _clip01(
            self._step_value(self.map_iou_times, self.map_iou_values, stamp)
        )

        metrics = TaskMetrics(
            d_traj=d_traj,
            d_cov=d_cov,
            d_red=d_red,
            d_map=d_map,
            coverage=coverage,
            coverage_pc=coverage_pc,
            redundancy=redundancy,
            redundancy_pc=redundancy_pc,
            map_iou=map_iou,
            map_iou_pc=map_iou_pc,
            joint_coverage=coverage,
            bs_coverage=bs_coverage,
            trajectory_deviation_m=trajectory_deviation_m,
            task_step=task_step,
            task_time_s=stamp,
        )
        self.last_task_step = task_step
        self.last_task_time_s = stamp
        self.last_task_metrics = metrics
        return metrics
