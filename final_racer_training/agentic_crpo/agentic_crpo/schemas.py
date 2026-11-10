"""Validated in-memory contracts shared by the real and mock backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32)
    if output.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} contains NaN or Inf")
    return output


@dataclass(frozen=True)
class GlobalGuidance:
    task_dependency: np.ndarray
    semantic_importance: np.ndarray

    @classmethod
    def neutral(cls, n_uavs: int) -> "GlobalGuidance":
        return cls(
            task_dependency=np.zeros((n_uavs, n_uavs), dtype=np.float32),
            semantic_importance=np.full(n_uavs, 1.0 / n_uavs, dtype=np.float32),
        )

    @classmethod
    def validated(
        cls,
        task_dependency: Any,
        semantic_importance: Any,
        n_uavs: int,
        normalize_omega: bool = True,
    ) -> "GlobalGuidance":
        dependency = _array(
            task_dependency, (n_uavs, n_uavs), "task_dependency"
        ).copy()
        importance = _array(
            semantic_importance, (n_uavs,), "semantic_importance"
        ).copy()
        np.clip(dependency, 0.0, 1.0, out=dependency)
        np.fill_diagonal(dependency, 0.0)
        np.clip(importance, 0.0, 1.0, out=importance)
        if normalize_omega:
            total = float(importance.sum())
            importance = (
                importance / total
                if total > 1.0e-8
                else np.full(n_uavs, 1.0 / n_uavs, dtype=np.float32)
            )
        return cls(dependency, importance.astype(np.float32, copy=False))


@dataclass(frozen=True)
class TaskMetrics:
    d_traj: float = 0.0
    d_cov: float = 0.0
    d_red: float = 0.0
    d_map: float = 0.0
    coverage: float = 0.0
    coverage_pc: float = 0.0
    redundancy: float = 0.0
    redundancy_pc: float = 0.0
    map_iou: float = 0.0
    map_iou_pc: float = 0.0
    joint_coverage: float = 0.0
    bs_coverage: float = 0.0
    trajectory_deviation_m: float = 0.0
    task_step: int = 0
    task_time_s: float = 0.0

    def __post_init__(self) -> None:
        degradations = np.asarray(
            (self.d_traj, self.d_cov, self.d_red, self.d_map), dtype=float
        )
        diagnostics = np.asarray(
            (
                self.coverage,
                self.coverage_pc,
                self.redundancy,
                self.redundancy_pc,
                self.map_iou,
                self.map_iou_pc,
                self.joint_coverage,
                self.bs_coverage,
                self.trajectory_deviation_m,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(degradations)) or np.any(degradations < 0.0):
            raise ValueError("task degradations must be finite and non-negative")
        if np.any(degradations > 1.0 + 1.0e-6):
            raise ValueError("task degradations must be normalized to [0, 1]")
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("task diagnostics must be finite and non-negative")
        if (
            self.task_step < 0
            or not np.isfinite(self.task_time_s)
            or self.task_time_s < 0.0
        ):
            raise ValueError("task clock must be finite and non-negative")


@dataclass
class RacerSnapshot:
    """One scheduler observation before an action is applied.

    Node ``N`` is the BS in ``channel_snr_db``.  Pair matrices always use
    information owner on rows and UAV receiver on columns.
    """

    n_uavs: int
    slot: int
    sim_time_s: float
    positions: np.ndarray
    velocities: np.ndarray
    yaws: np.ndarray
    fsm_states: list[str]
    channel_snr_db: np.ndarray
    pair_aoi_s: np.ndarray
    bs_aoi_s: np.ndarray
    pair_missing_bytes: np.ndarray
    bs_missing_bytes: np.ndarray
    uplink_queue_bytes: np.ndarray
    relay_queue_bytes: np.ndarray
    # Slow LLM-only fields. These describe information that has reached the
    # BS. The legacy fields below remain for file/mock backend compatibility
    # and are never consumed by LargeStateBuilder.
    bs_map_summary: dict[str, Any] = field(default_factory=dict)
    uav_bs_missing_bytes: np.ndarray | None = None
    bs_uav_missing_bytes: np.ndarray | None = None
    bs_map_coverage_delta: float = 0.0
    map_summary: dict[str, Any] = field(default_factory=dict)
    trajectory_summary: list[dict[str, Any]] = field(default_factory=list)
    region_summary: list[dict[str, Any]] = field(default_factory=list)
    information_version_gap: np.ndarray | None = None
    coverage: float = 0.0
    coverage_delta: float = 0.0
    bs_global_map_coverage: float = 0.0
    task_metrics: TaskMetrics = field(default_factory=TaskMetrics)
    terminated: bool = False
    truncated: bool = False
    communication_slot_index: int | None = None
    rl_decision_index: int | None = None
    action_held_slots: int = 0
    communication_slot_duration_s: float = 0.02
    rl_decision_interval_s: float = 0.1
    state_sequence: int = 0
    step_id: int = 0
    guidance_id: int = 0
    guidance_task_dependency: np.ndarray | None = None
    guidance_semantic_importance: np.ndarray | None = None

    def validate(self) -> "RacerSnapshot":
        n = self.n_uavs
        if n < 1 or self.slot < 0 or not np.isfinite(self.sim_time_s):
            raise ValueError("invalid snapshot identity")
        if self.communication_slot_index is None:
            self.communication_slot_index = self.slot
        if self.rl_decision_index is None:
            self.rl_decision_index = self.slot
        if (
            self.communication_slot_index < 0
            or self.rl_decision_index < 0
            or self.action_held_slots < 0
            or self.state_sequence < 0
            or self.step_id < 0
            or self.guidance_id < 0
            or self.communication_slot_duration_s <= 0.0
            or self.rl_decision_interval_s <= 0.0
        ):
            raise ValueError("invalid synchronous timing metadata")
        self.positions = _array(self.positions, (n, 3), "positions")
        self.velocities = _array(self.velocities, (n, 3), "velocities")
        self.yaws = _array(self.yaws, (n,), "yaws")
        self.channel_snr_db = _array(
            self.channel_snr_db, (n + 1, n + 1), "channel_snr_db"
        )
        self.pair_aoi_s = _array(self.pair_aoi_s, (n, n), "pair_aoi_s")
        self.bs_aoi_s = _array(self.bs_aoi_s, (n,), "bs_aoi_s")
        self.pair_missing_bytes = _array(
            self.pair_missing_bytes, (n, n), "pair_missing_bytes"
        )
        self.bs_missing_bytes = _array(
            self.bs_missing_bytes, (n,), "bs_missing_bytes"
        )
        self.uplink_queue_bytes = _array(
            self.uplink_queue_bytes, (n,), "uplink_queue_bytes"
        )
        self.relay_queue_bytes = _array(
            self.relay_queue_bytes, (n, n), "relay_queue_bytes"
        )
        if self.uav_bs_missing_bytes is None:
            self.uav_bs_missing_bytes = self.bs_missing_bytes.copy()
        self.uav_bs_missing_bytes = _array(
            self.uav_bs_missing_bytes, (n,), "uav_bs_missing_bytes"
        )
        if self.bs_uav_missing_bytes is None:
            self.bs_uav_missing_bytes = np.zeros(n, np.float32)
        self.bs_uav_missing_bytes = _array(
            self.bs_uav_missing_bytes, (n,), "bs_uav_missing_bytes"
        )
        if self.information_version_gap is None:
            self.information_version_gap = np.zeros((n, n + 1), np.float32)
        self.information_version_gap = _array(
            self.information_version_gap,
            (n, n + 1),
            "information_version_gap",
        )
        if self.guidance_task_dependency is not None:
            self.guidance_task_dependency = _array(
                self.guidance_task_dependency, (n, n),
                "guidance_task_dependency",
            )
        if self.guidance_semantic_importance is not None:
            self.guidance_semantic_importance = _array(
                self.guidance_semantic_importance, (n,),
                "guidance_semantic_importance",
            )
        if len(self.fsm_states) != n:
            raise ValueError(f"fsm_states must contain {n} entries")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if not 0.0 <= self.bs_global_map_coverage <= 1.0:
            raise ValueError("BS global-map coverage must be in [0, 1]")
        if not np.isfinite(self.bs_map_coverage_delta):
            raise ValueError("BS map coverage delta must be finite")
        return self


@dataclass(frozen=True)
class BackendStep:
    snapshot: RacerSnapshot
    bs_ul_resource: float
    bs_dl_resource: float
    direct_u2u_resource: float = 0.0
    executed_action_matrix: np.ndarray | None = None
    action_version: int = 0
    policy_version: int = 0
    action_age: float = 0.0
    sim_timestamp: float = 0.0
    action_source_guidance_id: int = 0
    transition_step_id: int = 0
    transition_sim_time_s: float = 0.0
    action_source_step_id: int = 0
    action_source_sim_time_s: float = 0.0

    @property
    def bs_resource(self) -> float:
        return float(self.bs_ul_resource + self.bs_dl_resource)
