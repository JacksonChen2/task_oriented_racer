"""Backend contract, deterministic mock, and filesystem ROS/Isaac bridge."""

from __future__ import annotations

import json
import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import BackendStep, RacerSnapshot, TaskMetrics
from .fast_state import decode_fast_state
from .reference_metrics import ReferenceTaskMetricEvaluator
from .shared_ipc import (
    SharedJsonBlock,
    SharedMemoryError,
    SharedMemoryLayout,
    open_blocks,
    open_rings,
    shutdown_requested,
)


class MissionEndedError(TimeoutError):
    """The final synchronized state was consumed and no next interval exists."""


class RacerBackend(ABC):
    n_uavs: int
    task_metrics_are_async = False
    decoupled_transition_collection = False

    @abstractmethod
    def reset(self, seed: int | None = None) -> RacerSnapshot:
        raise NotImplementedError

    @abstractmethod
    def step(self, action_matrix: np.ndarray) -> BackendStep:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def synchronization_status(self) -> dict[str, Any]:
        return {"enabled": False}

    def set_action_context(
        self, *, guidance_id: int = 0, policy_version: int = 0
    ) -> None:
        del guidance_id, policy_version

    def refresh_latest_snapshot(self) -> RacerSnapshot | None:
        """Refresh an observation after an out-of-band optimizer pause."""

        return None

    def publish_action(
        self,
        action_matrix: np.ndarray,
        source_context: dict[str, Any] | None = None,
    ) -> int:
        del action_matrix, source_context
        raise RuntimeError("backend does not support decoupled action publication")

    def collect_transition(self) -> BackendStep:
        raise RuntimeError("backend does not support decoupled transition collection")

    def resolve_task_metrics(
        self, step_ids: list[int]
    ) -> dict[int, TaskMetrics]:
        del step_ids
        raise RuntimeError("backend does not provide asynchronous task metrics")


class MockRacerBackend(RacerBackend):
    """Fast deterministic dynamics for CRPO and objective-isolation tests."""

    def __init__(
        self,
        n_uavs: int,
        horizon: int = 128,
        seed: int = 42,
        direct_u2u_resource: float = 11.0,
    ) -> None:
        self.n_uavs = n_uavs
        self.horizon = horizon
        self.default_seed = seed
        self.fixed_direct_resource = direct_u2u_resource
        self.rng = np.random.default_rng(seed)
        self.slot = 0
        self.decision_index = 0
        self.positions = np.zeros((n_uavs, 3), np.float32)
        self.velocities = np.zeros_like(self.positions)
        self.yaws = np.zeros(n_uavs, np.float32)
        self.channel = np.zeros((n_uavs + 1, n_uavs + 1), np.float32)
        self.local_bytes = np.zeros(n_uavs, np.float32)
        self.bs_bytes = np.zeros(n_uavs, np.float32)
        self.receiver_bytes = np.zeros((n_uavs, n_uavs), np.float32)
        self.pair_aoi = np.zeros((n_uavs, n_uavs), np.float32)
        self.bs_aoi = np.zeros(n_uavs, np.float32)
        self.coverage = 0.0

    def reset(self, seed: int | None = None) -> RacerSnapshot:
        self.rng = np.random.default_rng(self.default_seed if seed is None else seed)
        self.slot = 0
        self.decision_index = 0
        self.positions = self.rng.uniform(
            low=(-22.0, 3.0, 1.0), high=(2.0, 27.0, 3.0), size=(self.n_uavs, 3)
        ).astype(np.float32)
        self.velocities.fill(0.0)
        self.yaws = self.rng.uniform(-math.pi, math.pi, self.n_uavs).astype(
            np.float32
        )
        self.local_bytes.fill(120_000.0)
        self.bs_bytes.fill(0.0)
        self.receiver_bytes.fill(0.0)
        self.pair_aoi.fill(0.0)
        self.bs_aoi.fill(0.0)
        self.coverage = 0.02
        self._update_channel()
        return self._snapshot(coverage_delta=0.0)

    def _update_channel(self) -> None:
        n = self.n_uavs
        bs = np.asarray((-10.03, 14.89, 7.55), np.float32)
        self.channel.fill(-120.0)
        for sender in range(n):
            for receiver in range(n):
                if sender != receiver:
                    distance = np.linalg.norm(
                        self.positions[sender] - self.positions[receiver]
                    )
                    self.channel[sender, receiver] = 24.0 - 1.5 * distance
            distance_bs = np.linalg.norm(self.positions[sender] - bs)
            self.channel[sender, n] = 30.0 - 1.25 * distance_bs
            self.channel[n, sender] = 35.0 - 1.15 * distance_bs

    def _missing(self) -> tuple[np.ndarray, np.ndarray]:
        pair = np.maximum(self.local_bytes[:, None] - self.receiver_bytes, 0.0)
        np.fill_diagonal(pair, 0.0)
        bs = np.maximum(self.local_bytes - self.bs_bytes, 0.0)
        return pair.astype(np.float32), bs.astype(np.float32)

    def _task_metrics(self) -> TaskMetrics:
        pair, bs = self._missing()
        scale = max(1.0, float(np.sum(self.local_bytes) * self.n_uavs))
        bs_coverage = float(
            np.clip(
                1.0 - np.sum(bs) / max(1.0, np.sum(self.local_bytes)),
                0.0,
                1.0,
            )
        )
        d_map = float(max(0.0, self.coverage - bs_coverage))
        d_red = float(np.sum(pair) / scale)
        progress = self.decision_index / max(1, self.horizon)
        coverage_pc = min(1.0, 0.02 + progress)
        d_cov = float(max(0.0, coverage_pc - self.coverage))
        target = np.stack(
            (
                np.linspace(-20.0, 0.0, self.n_uavs),
                np.linspace(5.0, 25.0, self.n_uavs),
                np.full(self.n_uavs, 2.0),
            ),
            axis=1,
        )
        d_traj = float(np.mean(np.linalg.norm(self.positions - target, axis=1)) / 50.0)
        return TaskMetrics(
            d_traj=d_traj,
            d_cov=d_cov,
            d_red=d_red,
            d_map=d_map,
            coverage=float(self.coverage),
            coverage_pc=float(coverage_pc),
            redundancy=float(d_red),
            redundancy_pc=0.0,
            map_iou=float(1.0 - d_map),
            map_iou_pc=1.0,
            joint_coverage=float(self.coverage),
            bs_coverage=bs_coverage,
            trajectory_deviation_m=float(d_traj * 50.0),
            task_step=int(self.decision_index),
            task_time_s=float(0.1 * self.decision_index),
        )

    def _snapshot(self, coverage_delta: float) -> RacerSnapshot:
        pair_missing, bs_missing = self._missing()
        relay_queues = np.maximum(self.bs_bytes[:, None] - self.receiver_bytes, 0.0)
        np.fill_diagonal(relay_queues, 0.0)
        bs_uav_missing = relay_queues.sum(axis=0).astype(np.float32)
        bs_coverage = float(
            np.clip(
                1.0 - np.sum(bs_missing) / max(1.0, np.sum(self.local_bytes)),
                0.0,
                1.0,
            )
        )
        version_gap = np.concatenate(
            (pair_missing / 1200.0, (bs_missing / 1200.0)[:, None]), axis=1
        )
        return RacerSnapshot(
            n_uavs=self.n_uavs,
            slot=self.slot,
            sim_time_s=0.02 * self.slot,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            yaws=self.yaws.copy(),
            fsm_states=[
                "EXPLORE" if self.decision_index < self.horizon else "FINISH"
            ]
            * self.n_uavs,
            channel_snr_db=self.channel.copy(),
            pair_aoi_s=self.pair_aoi.copy(),
            bs_aoi_s=self.bs_aoi.copy(),
            pair_missing_bytes=pair_missing,
            bs_missing_bytes=bs_missing,
            uplink_queue_bytes=bs_missing.copy(),
            relay_queue_bytes=relay_queues.astype(np.float32),
            bs_map_summary={
                "representation": "mock_bs_inventory",
                "bs_map_coverage": bs_coverage,
                "bs_unknown_ratio": 1.0 - bs_coverage,
            },
            uav_bs_missing_bytes=bs_missing,
            bs_uav_missing_bytes=bs_uav_missing,
            bs_map_coverage_delta=0.0,
            map_summary={
                "representation": "coarse_top_down_statistics",
                "known_ratio": self.coverage,
                "unknown_ratio": 1.0 - self.coverage,
            },
            information_version_gap=version_gap.astype(np.float32),
            coverage=float(self.coverage),
            coverage_delta=float(coverage_delta),
            bs_global_map_coverage=bs_coverage,
            task_metrics=self._task_metrics(),
            terminated=self.decision_index >= self.horizon,
            communication_slot_index=self.slot,
            rl_decision_index=self.decision_index,
            action_held_slots=(0 if self.decision_index == 0 else 5),
            communication_slot_duration_s=0.02,
            rl_decision_interval_s=0.1,
            state_sequence=self.decision_index + 1,
            step_id=self.decision_index,
        ).validate()

    def _step_communication_slot(
        self, action: np.ndarray
    ) -> tuple[float, float]:
        n = self.n_uavs
        self.slot += 1
        self.pair_aoi += 0.02
        self.bs_aoi += 0.02
        ul_resource = 0.0
        dl_resource = 0.0
        quantum = 12_000.0
        for owner in range(n):
            if action[n, owner] and self.channel[owner, n] > -5.0:
                amount = min(quantum, self.local_bytes[owner] - self.bs_bytes[owner])
                if amount > 0.0:
                    self.bs_bytes[owner] += amount
                    self.bs_aoi[owner] = 0.0
                    ul_resource += 66.0 * math.ceil(amount / 1200.0)
        for owner in range(n):
            for receiver in range(n):
                if owner == receiver or not action[owner, receiver]:
                    continue
                if self.channel[n, receiver] <= -5.0:
                    continue
                available = self.bs_bytes[owner] - self.receiver_bytes[owner, receiver]
                amount = min(quantum, available)
                if amount > 0.0:
                    self.receiver_bytes[owner, receiver] += amount
                    self.pair_aoi[owner, receiver] = 0.0
                    dl_resource += 66.0 * math.ceil(amount / 1200.0)

        # Original direct U2U proceeds independently of the RL action.
        for owner in range(n):
            for receiver in range(n):
                if owner != receiver and self.channel[owner, receiver] > 8.0:
                    available = self.local_bytes[owner] - self.receiver_bytes[owner, receiver]
                    if available > 0.0:
                        self.receiver_bytes[owner, receiver] += min(2400.0, available)
                        self.pair_aoi[owner, receiver] = 0.0

        if self.slot % 4 == 0:
            self.local_bytes += self.rng.integers(1200, 4801, self.n_uavs)
        self.coverage = min(
            1.0, self.coverage + 0.0005 + 1.0e-9 * float(np.sum(self.bs_bytes))
        )
        phase = 0.04 * self.slot + np.arange(n)
        self.velocities[:, 0] = 0.25 * np.cos(phase)
        self.velocities[:, 1] = 0.25 * np.sin(phase)
        self.positions += 0.02 * self.velocities
        self.yaws = np.arctan2(self.velocities[:, 1], self.velocities[:, 0])
        self._update_channel()
        return ul_resource, dl_resource

    def step(self, action_matrix: np.ndarray) -> BackendStep:
        n = self.n_uavs
        action = np.asarray(action_matrix, dtype=np.int8)
        if action.shape != (n + 1, n) or np.any(np.diag(action[:n])):
            raise ValueError("invalid CRPO action matrix")
        old_coverage = self.coverage
        transition_step_id = self.decision_index
        ul_resource = 0.0
        dl_resource = 0.0
        for _ in range(5):
            slot_ul, slot_dl = self._step_communication_slot(action)
            ul_resource += slot_ul
            dl_resource += slot_dl
        self.decision_index += 1
        return BackendStep(
            snapshot=self._snapshot(self.coverage - old_coverage),
            bs_ul_resource=ul_resource,
            bs_dl_resource=dl_resource,
            direct_u2u_resource=5.0 * self.fixed_direct_resource,
            transition_step_id=transition_step_id,
            transition_sim_time_s=0.1 * transition_step_id,
        )


class FileBridgeBackend(RacerBackend):
    """Exchange actions/state with the optional C++ scheduler file bridge."""

    def __init__(
        self,
        n_uavs: int,
        action_path: str,
        telemetry_path: str,
        mission_state_path: str | None = None,
        perfect_reference_result: str | None = None,
        auxiliary_reference_result: str | None = None,
        workspace_min: tuple[float, float, float] = (-22.0, 3.0, 1.0),
        workspace_max: tuple[float, float, float] = (2.0, 27.0, 3.0),
        redundancy_cell_m: float = 1.0,
        fixed_exploration_time_s: float = 300.0,
        require_auxiliary_reference_metrics: bool = True,
        require_mission_state: bool = True,
        require_perfect_reference: bool = True,
        max_relay_recipients_per_uav: int | None = None,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.002,
        synchronous_online: bool = False,
        sync_acknowledgement_path: str | None = None,
        sync_release_path: str | None = None,
        communication_slot_s: float = 0.02,
        decision_interval_s: float = 0.1,
        slots_per_decision: int = 5,
        require_explicit_task_clock: bool = False,
    ) -> None:
        self.n_uavs = n_uavs
        self.action_path = Path(action_path)
        self.telemetry_path = Path(telemetry_path)
        self.mission_state_path = (
            None if mission_state_path is None else Path(mission_state_path)
        )
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.require_mission_state = require_mission_state
        self.require_perfect_reference = require_perfect_reference
        self.synchronous_online = bool(synchronous_online)
        self.sync_acknowledgement_path = (
            Path(sync_acknowledgement_path)
            if sync_acknowledgement_path is not None
            else None
        )
        self.sync_release_path = (
            Path(sync_release_path) if sync_release_path is not None else None
        )
        self.communication_slot_s = float(communication_slot_s)
        self.decision_interval_s = float(decision_interval_s)
        self.slots_per_decision = int(slots_per_decision)
        if self.synchronous_online:
            if (
                self.sync_acknowledgement_path is None
                or self.sync_release_path is None
            ):
                raise ValueError(
                    "synchronous file bridge requires acknowledgement and release paths"
                )
            if (
                self.communication_slot_s <= 0.0
                or self.decision_interval_s <= 0.0
                or self.slots_per_decision != 5
                or not math.isclose(
                    self.decision_interval_s,
                    self.communication_slot_s * self.slots_per_decision,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    "synchronous bridge requires T_RL=5*T_comm"
                )
        if max_relay_recipients_per_uav is not None and not (
            0 <= max_relay_recipients_per_uav <= max(0, n_uavs - 1)
        ):
            raise ValueError("invalid file-bridge relay fanout limit")
        self.max_relay_recipients_per_uav = (
            max_relay_recipients_per_uav
        )
        self.metric_evaluator = ReferenceTaskMetricEvaluator(
            perfect_reference_result,
            n_uavs,
            workspace_min,
            workspace_max,
            redundancy_cell_m=redundancy_cell_m,
            fixed_exploration_time_s=fixed_exploration_time_s,
            require_auxiliary_reference_metrics=require_auxiliary_reference_metrics,
            auxiliary_reference_result=auxiliary_reference_result,
            require_explicit_task_clock=require_explicit_task_clock,
        )
        self.action_epoch = 0
        self.last_sequence = -1
        self.last_task_step = -1
        self.last_decision_index = -1
        self.last_communication_slot_index = -1
        self.last_sim_time_s = 0.0
        self.last_snapshot_terminal = False
        self.last_resources = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"bridge payload is not an object: {path}")
        return value

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def synchronization_status(self) -> dict[str, Any]:
        if not self.synchronous_online:
            return {"enabled": False}
        assert self.sync_acknowledgement_path is not None
        status: dict[str, Any] = {"enabled": True}
        try:
            status.update(self._read_json(self.sync_acknowledgement_path))
        except (OSError, ValueError, json.JSONDecodeError):
            status["status"] = "unavailable"
        try:
            payload = self._read_json(self.telemetry_path)
            status["telemetry_sim_time_s"] = float(payload["sim_time_s"])
            status["telemetry_sequence"] = int(payload["sequence"])
            status["telemetry_communication_slot_index"] = int(
                payload["communication_slot_index"]
            )
            status["telemetry_rl_decision_index"] = int(
                payload["rl_decision_index"]
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        return status

    def _wait_synchronous_payload(
        self, expected_decision_index: int | None
    ) -> dict[str, Any]:
        assert self.sync_acknowledgement_path is not None
        deadline = time.monotonic() + self.timeout_s
        last_error: Exception | str | None = None
        while time.monotonic() < deadline:
            try:
                payload = self._read_json(self.telemetry_path)
                if not bool(payload.get("synchronous_online", False)):
                    raise ValueError(
                        "communication state is not in synchronous-online mode"
                    )
                decision_index = int(payload["rl_decision_index"])
                if (
                    expected_decision_index is not None
                    and decision_index > expected_decision_index
                ):
                    raise RuntimeError(
                        "synchronous communication state skipped a decision: "
                        f"expected={expected_decision_index} observed={decision_index}"
                    )
                acknowledgement = self._read_json(
                    self.sync_acknowledgement_path
                )
                matches_expected = (
                    expected_decision_index is None
                    or decision_index == expected_decision_index
                )
                if (
                    matches_expected
                    and int(acknowledgement.get("rl_decision_index", -1))
                    == decision_index
                    and int(acknowledgement.get("state_sequence", -1))
                    == int(payload["sequence"])
                    and acknowledgement.get("status") in {"paused", "terminal"}
                ):
                    state_time = float(payload["sim_time_s"])
                    ack_time = float(acknowledgement["sim_time_s"])
                    if not math.isclose(
                        state_time, ack_time, rel_tol=0.0, abs_tol=1.0e-6
                    ):
                        raise RuntimeError(
                            "Isaac acknowledgement and communication state use "
                            "different simulation times"
                        )
                    return payload
            except RuntimeError:
                raise
            except (
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            "no acknowledged synchronous RL boundary state from "
            f"{self.telemetry_path}; expected_decision="
            f"{expected_decision_index} last_error={last_error}"
        )

    def _wait_payload(self, after_sequence: int | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                payload = self._read_json(self.telemetry_path)
                sequence = int(payload["sequence"])
                if after_sequence is None or sequence > after_sequence:
                    return payload
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                last_error = error
            if after_sequence is not None and self.mission_state_path is not None:
                mission_ended = False
                try:
                    mission = self._read_json(self.mission_state_path)
                    mission_ended = bool(
                        mission.get("terminated", False)
                        or mission.get("truncated", False)
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    last_error = error
                # TimeoutError is an OSError subclass.  Raise outside the file
                # parsing try/except so the terminal boundary is not mistaken
                # for a transient read failure and delayed until timeout_s.
                if mission_ended:
                    raise MissionEndedError(
                        "Isaac mission ended before another RL telemetry "
                        "sequence was published"
                    )
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"no fresh RL telemetry from {self.telemetry_path}; last_error={last_error}"
        )

    def _snapshot_from_payloads(
        self,
        payload: dict[str, Any],
        mission: dict[str, Any],
        *,
        prefer_mission_physical: bool = False,
    ) -> RacerSnapshot:
        n = self.n_uavs
        metrics_payload = mission.get("task_metrics", payload.get("task_metrics"))
        if metrics_payload is not None:
            joint_coverage = float(
                metrics_payload.get(
                    "joint_coverage",
                    metrics_payload.get(
                        "coverage",
                        mission.get("coverage", payload.get("coverage", 0.0)),
                    ),
                )
            )
            bs_coverage = float(
                metrics_payload.get(
                    "bs_coverage",
                    payload.get("bs_global_map_coverage", 0.0),
                )
            )
            metrics = TaskMetrics(
                d_traj=float(metrics_payload["D_traj"]),
                d_cov=float(metrics_payload["D_cov"]),
                d_red=float(metrics_payload["D_red"]),
                d_map=float(max(0.0, joint_coverage - bs_coverage)),
                coverage=joint_coverage,
                coverage_pc=float(metrics_payload.get("coverage_pc", 0.0)),
                redundancy=float(metrics_payload.get("redundancy", 0.0)),
                redundancy_pc=float(metrics_payload.get("redundancy_pc", 0.0)),
                map_iou=float(metrics_payload.get("map_iou", 0.0)),
                map_iou_pc=float(metrics_payload.get("map_iou_pc", 0.0)),
                joint_coverage=joint_coverage,
                bs_coverage=bs_coverage,
                trajectory_deviation_m=float(
                    metrics_payload.get("trajectory_deviation_m", 0.0)
                ),
                task_step=int(metrics_payload.get("task_step", 0)),
                task_time_s=float(metrics_payload.get("task_time_s", 0.0)),
            )
        else:
            metrics = self.metric_evaluator.evaluate(payload, mission)
        # New slow communication payloads contain BS-observable telemetry. Use
        # it for the LLM-facing snapshot even when mission physical state is
        # otherwise preferred for the fast actor/reward path.
        has_bs_observation = "bs_map_summary" in payload
        physical_first = (
            (payload, mission)
            if has_bs_observation
            else ((mission, payload) if prefer_mission_physical else (payload, mission))
        )
        positions = physical_first[0].get(
            "positions", physical_first[1].get("positions")
        )
        velocities = physical_first[0].get(
            "velocities", physical_first[1].get("velocities")
        )
        yaws = physical_first[0].get("yaws", physical_first[1].get("yaws"))
        snapshot = RacerSnapshot(
            n_uavs=n,
            slot=int(payload.get("communication_slot_index", payload["sequence"])),
            sim_time_s=float(payload.get("sim_time_s", mission.get("sim_time_s", 0.0))),
            positions=positions,
            velocities=velocities,
            yaws=yaws,
            fsm_states=list(
                payload.get("fsm_states", mission.get("fsm_states", ["UNKNOWN"] * n))
                if has_bs_observation
                else mission.get("fsm_states", payload.get("fsm_states", ["UNKNOWN"] * n))
            ),
            channel_snr_db=payload["channel_snr_db"],
            pair_aoi_s=payload["pair_aoi_s"],
            bs_aoi_s=payload["bs_aoi_s"],
            pair_missing_bytes=payload["pair_missing_bytes"],
            bs_missing_bytes=payload.get(
                "uav_bs_missing_bytes", payload.get("bs_missing_bytes")
            ),
            uplink_queue_bytes=payload["uplink_queue_bytes"],
            relay_queue_bytes=payload["relay_queue_bytes"],
            bs_map_summary=payload.get("bs_map_summary", {}),
            uav_bs_missing_bytes=payload.get(
                "uav_bs_missing_bytes", payload.get("bs_missing_bytes")
            ),
            bs_uav_missing_bytes=payload.get(
                "bs_uav_missing_bytes", np.zeros(n, dtype=np.float32)
            ),
            bs_map_coverage_delta=float(
                payload.get("bs_map_coverage_delta", 0.0)
            ),
            map_summary=mission.get("map_summary", payload.get("map_summary", {})),
            trajectory_summary=(
                payload.get("trajectory_summary", [])
                if has_bs_observation
                else mission.get("trajectory_summary", payload.get("trajectory_summary", []))
            ),
            region_summary=(
                payload.get("region_summary", [])
                if has_bs_observation
                else mission.get("region_summary", payload.get("region_summary", []))
            ),
            information_version_gap=payload.get("information_version_gap"),
            coverage=float(mission.get("coverage", payload.get("coverage", 0.0))),
            coverage_delta=float(mission.get("coverage_delta", 0.0)),
            bs_global_map_coverage=max(
                0.0, float(payload.get("bs_global_map_coverage", 0.0))
            ),
            task_metrics=metrics,
            terminated=bool(mission.get("terminated", False)),
            truncated=bool(mission.get("truncated", False)),
            communication_slot_index=int(
                payload.get("communication_slot_index", payload["sequence"])
            ),
            rl_decision_index=int(
                payload.get("rl_decision_index", payload["sequence"])
            ),
            action_held_slots=int(payload.get("action_held_slots", 1)),
            communication_slot_duration_s=float(
                payload.get("communication_slot_duration_s", 0.02)
            ),
            rl_decision_interval_s=float(
                payload.get("rl_decision_interval_s", 0.02)
            ),
            state_sequence=int(payload["sequence"]),
        )
        return snapshot.validate()

    def _snapshot(self, payload: dict[str, Any]) -> RacerSnapshot:
        mission: dict[str, Any] = {}
        if self.mission_state_path is not None and self.mission_state_path.exists():
            mission = self._read_json(self.mission_state_path)
        if self.synchronous_online:
            mission_time = float(mission.get("sim_time_s", math.nan))
            payload_time = float(payload["sim_time_s"])
            if not math.isclose(
                mission_time, payload_time, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise RuntimeError(
                    "mission and communication states are not from the same "
                    f"RL boundary: mission={mission_time} communication={payload_time}"
                )
        return self._snapshot_from_payloads(payload, mission)

    def reset(self, seed: int | None = None) -> RacerSnapshot:
        del seed
        self.metric_evaluator.reset()
        if self.require_mission_state and (
            self.mission_state_path is None or not self.mission_state_path.exists()
        ):
            raise FileNotFoundError(
                "real CRPO training requires a fresh Isaac mission_state_path; "
                "the perfect reference/ground-truth metrics are loss-only and "
                "must not be silently replaced with zeros"
            )
        if self.require_perfect_reference:
            self.metric_evaluator.validate_reference()
        payload = (
            self._wait_synchronous_payload(None)
            if self.synchronous_online
            else self._wait_payload()
        )
        self.last_sequence = int(payload["sequence"])
        self.last_task_step = int(
            payload.get("task_step", payload.get("rl_decision_index", -1))
        )
        self.last_decision_index = int(
            payload.get("rl_decision_index", self.last_sequence)
        )
        self.last_communication_slot_index = int(
            payload.get("communication_slot_index", self.last_sequence)
        )
        self.last_sim_time_s = float(payload.get("sim_time_s", 0.0))
        # Continue above any epoch already consumed by a still-running proxy
        # (or a stale action file it read during startup). This makes resumed
        # training sessions immediately acceptable to the monotonic C++ reader.
        self.action_epoch = max(
            self.action_epoch, int(payload.get("action_epoch", 0))
        )
        self.last_resources = np.asarray(
            (
                payload.get("bs_uplink_prb_slots", 0.0),
                payload.get("bs_downlink_prb_slots", 0.0),
                payload.get("direct_u2u_prb_slots", 0.0),
            ),
            dtype=np.float64,
        )
        snapshot = self._snapshot(payload)
        self.last_snapshot_terminal = bool(
            snapshot.terminated or snapshot.truncated
        )
        return snapshot

    def _write_action(
        self, action_matrix: np.ndarray, decision_index: int | None = None
    ) -> int:
        n = self.n_uavs
        matrix = np.asarray(action_matrix, dtype=np.int8)
        if matrix.shape != (n + 1, n):
            raise ValueError("invalid file-bridge action matrix")
        if self.max_relay_recipients_per_uav is not None:
            fanout = np.sum(matrix[:n], axis=1)
            if np.any(fanout > self.max_relay_recipients_per_uav):
                raise ValueError(
                    "file bridge refused action above the configured relay "
                    f"fanout limit {self.max_relay_recipients_per_uav}: "
                    f"counts={fanout.tolist()}"
                )
        # The C++ side deliberately uses a dependency-free whitespace parser.
        flat: list[str] = []
        for owner in range(n):
            for receiver in range(n):
                if owner != receiver:
                    flat.append(str(int(matrix[owner, receiver])))
        flat.extend(str(int(item)) for item in matrix[n])
        self.action_epoch += 1
        if self.synchronous_online:
            if decision_index is None or decision_index < 0:
                raise ValueError("synchronous action requires a decision index")
            header = f"{self.action_epoch} {n} {decision_index} "
        else:
            header = f"{self.action_epoch} {n} "
        # Bind the action to the exact communication/channel state consumed
        # by the policy.  The trailer keeps the legacy leading layout intact
        # for simple tools while allowing the Proxy to reject stale or
        # mismatched channel snapshots instead of consulting current links.
        content = (
            header
            + " ".join(flat)
            + f" {self.last_sequence} {self.last_task_step}\n"
        )
        self.action_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.action_path.with_suffix(self.action_path.suffix + ".tmp")
        temporary.write_text(content, encoding="ascii")
        os.replace(temporary, self.action_path)
        return self.action_epoch

    def _release_synchronous_boundary(
        self, decision_index: int, action_epoch: int
    ) -> None:
        assert self.sync_release_path is not None
        payload = {
            "rl_decision_index": int(decision_index),
            "action_epoch": int(action_epoch),
            "trainer_pid": os.getpid(),
            "simulation_time_s": self.last_sim_time_s,
            "communication_slot_index": self.last_communication_slot_index,
            "released_at_unix_s": time.time(),
        }
        self._atomic_json(self.sync_release_path, payload)
        print(
            "RACER_RL_SYNC_ACTION "
            + json.dumps(payload, separators=(",", ":"), allow_nan=False),
            flush=True,
        )

    def step(self, action_matrix: np.ndarray) -> BackendStep:
        if self.synchronous_online and self.last_snapshot_terminal:
            raise MissionEndedError(
                "final synchronized RL state was already consumed"
            )
        previous_decision = self.last_decision_index
        previous_slot = self.last_communication_slot_index
        previous_sim_time = self.last_sim_time_s
        action_epoch = self._write_action(
            action_matrix,
            decision_index=(previous_decision if self.synchronous_online else None),
        )
        if self.synchronous_online:
            self._release_synchronous_boundary(previous_decision, action_epoch)
            payload = self._wait_synchronous_payload(previous_decision + 1)
        else:
            payload = self._wait_payload(after_sequence=self.last_sequence)
        self.last_sequence = int(payload["sequence"])
        self.last_task_step = int(
            payload.get("task_step", payload.get("rl_decision_index", -1))
        )
        self.last_decision_index = int(
            payload.get("rl_decision_index", self.last_sequence)
        )
        self.last_communication_slot_index = int(
            payload.get("communication_slot_index", self.last_sequence)
        )
        self.last_sim_time_s = float(payload.get("sim_time_s", previous_sim_time))
        if self.synchronous_online:
            slot_delta = self.last_communication_slot_index - previous_slot
            time_delta = self.last_sim_time_s - previous_sim_time
            if slot_delta != self.slots_per_decision:
                raise RuntimeError(
                    "synchronous transition did not contain exactly five "
                    f"communication slots: delta={slot_delta}"
                )
            if not math.isclose(
                time_delta,
                self.decision_interval_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise RuntimeError(
                    "synchronous transition simulation-time delta is not "
                    f"{self.decision_interval_s}: delta={time_delta}"
                )
            if int(payload.get("action_held_slots", -1)) != self.slots_per_decision:
                raise RuntimeError(
                    "communication proxy did not hold the CRPO action for "
                    "exactly five slots"
                )
        cumulative = np.asarray(
            (
                payload.get("bs_uplink_prb_slots", 0.0),
                payload.get("bs_downlink_prb_slots", 0.0),
                payload.get("direct_u2u_prb_slots", 0.0),
            ),
            dtype=np.float64,
        )
        delta = np.maximum(cumulative - self.last_resources, 0.0)
        if self.synchronous_online:
            interval_delta = np.asarray(
                (
                    payload["interval_bs_uplink_prb_slots"],
                    payload["interval_bs_downlink_prb_slots"],
                    payload["interval_direct_u2u_prb_slots"],
                ),
                dtype=np.float64,
            )
            if not np.allclose(delta, interval_delta, rtol=0.0, atol=1.0e-9):
                raise RuntimeError(
                    "100 ms interval resource counters do not match cumulative deltas"
                )
        self.last_resources = cumulative
        snapshot = self._snapshot(payload)
        self.last_snapshot_terminal = bool(
            snapshot.terminated or snapshot.truncated
        )
        return BackendStep(snapshot, *delta.tolist())


class SharedMemoryBackend(FileBridgeBackend):
    """Asynchronous action mailbox and ordered transition-ring consumer.

    Action publication is fire-and-forget.  The communication proxy owns the
    communication-slot clock and writes every configured decision boundary
    into an independent ring, which this backend consumes in order even while
    PPO is updating.
    """

    task_metrics_are_async = True
    decoupled_transition_collection = True

    def __init__(
        self,
        n_uavs: int,
        shared_memory_root: str,
        *,
        perfect_reference_result: str | None = None,
        auxiliary_reference_result: str | None = None,
        workspace_min: tuple[float, float, float] = (-22.0, 3.0, 1.0),
        workspace_max: tuple[float, float, float] = (2.0, 27.0, 3.0),
        redundancy_cell_m: float = 1.0,
        fixed_exploration_time_s: float = 300.0,
        require_auxiliary_reference_metrics: bool = True,
        require_perfect_reference: bool = True,
        max_relay_recipients_per_uav: int | None = None,
        communication_slot_s: float = 0.02,
        decision_interval_s: float = 0.1,
    ) -> None:
        slots_per_decision = int(round(decision_interval_s / communication_slot_s))
        if (
            communication_slot_s <= 0.0
            or decision_interval_s <= 0.0
            or slots_per_decision < 1
            or not math.isclose(
                decision_interval_s,
                slots_per_decision * communication_slot_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "decision_interval_s must be a positive integer multiple of "
                "communication_slot_s"
            )
        # Reuse the validated state decoder and reference metric evaluator, not
        # the file transport.  These paths are never read or written here.
        super().__init__(
            n_uavs,
            action_path="/dev/null",
            telemetry_path="/dev/null",
            mission_state_path=None,
            perfect_reference_result=perfect_reference_result,
            auxiliary_reference_result=auxiliary_reference_result,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            redundancy_cell_m=redundancy_cell_m,
            fixed_exploration_time_s=fixed_exploration_time_s,
            require_auxiliary_reference_metrics=(
                require_auxiliary_reference_metrics
            ),
            require_mission_state=False,
            require_perfect_reference=require_perfect_reference,
            max_relay_recipients_per_uav=max_relay_recipients_per_uav,
            synchronous_online=False,
            require_explicit_task_clock=True,
            communication_slot_s=communication_slot_s,
            decision_interval_s=decision_interval_s,
            slots_per_decision=slots_per_decision,
        )
        self.layout = SharedMemoryLayout.from_value(shared_memory_root)
        self.blocks = open_blocks(
            self.layout,
            (
                "action",
                "shutdown",
                "status_rl",
            ),
        )
        self.transition_ring = open_rings(
            self.layout, ("transition_ring",)
        )["transition_ring"]
        self.task_metric_ring = open_rings(
            self.layout, ("task_metric_ring",)
        )["task_metric_ring"]
        self.last_physical_version = 0
        self.last_communication_version = 0
        self.last_transition_sequence = 0
        self.last_task_metric_sequence = 0
        self.last_task_metric_step_id = -1
        self._resolved_task_metrics: dict[int, TaskMetrics] = {}
        self.state_wait_count = 0
        self.state_wait_total_ms = 0.0
        self.state_wait_max_ms = 0.0
        self.last_physical_sim_step = 0
        self.source_guidance_id = 0
        self.policy_version = 0
        self.skipped_physical_versions = 0
        self.skipped_communication_versions = 0
        self.rl_cycle_id = 0
        self.blocks["status_rl"].publish_json(
            {"ready": True, "process": "rl", "pid": os.getpid()}
        )

    def _stopping(self) -> bool:
        return shutdown_requested(self.blocks["shutdown"])

    def _wait_boundary(self) -> Any:
        started = time.perf_counter()
        record = self.transition_ring.wait_for_newer(
            self.last_transition_sequence, stop=self._stopping
        )
        if record is None:
            raise MissionEndedError(
                "shared-memory episode shut down before another boundary"
            )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        self.state_wait_count += 1
        self.state_wait_total_ms += elapsed_ms
        self.state_wait_max_ms = max(self.state_wait_max_ms, elapsed_ms)
        print(
            "RACER_RL_STATE_WAIT "
            + json.dumps(
                {
                    "step_id": int(record.sim_step),
                    "sim_time": float(record.sim_time_s),
                    "wait_next_state_ms": elapsed_ms,
                    "wait_next_state_mean_ms": (
                        self.state_wait_total_ms / self.state_wait_count
                    ),
                    "wait_next_state_max_ms": self.state_wait_max_ms,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return record

    def _decode_boundary(self, record: Any, *, rebase_resources: bool) -> tuple[
        RacerSnapshot, dict[str, Any]
    ]:
        try:
            fast = decode_fast_state(record.payload, self.n_uavs)
        except ValueError as error:
            raise SharedMemoryError(str(error)) from error
        values = fast.values
        step_id = int(values["step_id"])
        sim_time_s = float(values["sim_time_s"])
        if int(record.sim_step) != step_id or not math.isclose(
            float(record.sim_time_s), sim_time_s, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise SharedMemoryError("fast-state ring metadata is not step aligned")
        if not math.isclose(
            sim_time_s,
            step_id * self.decision_interval_s,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise SharedMemoryError("fast-state step_id/sim_time are inconsistent")
        if (
            int(values["rl_decision_index"]) != step_id
            or int(values["communication_slot_index"])
            != step_id * self.slots_per_decision
        ):
            raise SharedMemoryError(
                "fast-state decision/slot indices are not on the canonical clock"
            )
        physical_version = int(values["physical_version"])
        communication_version = int(values["sequence"])
        self.skipped_physical_versions += max(
            0, physical_version - self.last_physical_version - 1
        )
        self.skipped_communication_versions += max(
            0, communication_version - self.last_communication_version - 1
        )
        self.last_transition_sequence = int(record.sequence)
        self.last_physical_version = physical_version
        self.last_physical_sim_step = step_id
        self.last_communication_version = communication_version
        self.last_sequence = int(values["sequence"])
        self.last_decision_index = int(values["rl_decision_index"])
        self.last_communication_slot_index = int(
            values["communication_slot_index"]
        )
        self.last_sim_time_s = sim_time_s
        delta = np.asarray(
            (
                values["interval_bs_uplink_prb_slots"],
                values["interval_bs_downlink_prb_slots"],
                values["interval_direct_u2u_prb_slots"],
            ),
            dtype=np.float64,
        )
        if rebase_resources:
            self.last_resources = np.zeros(3, dtype=np.float64)
        task_clock = TaskMetrics(task_step=step_id, task_time_s=sim_time_s)
        snapshot = RacerSnapshot(
            n_uavs=self.n_uavs,
            slot=self.last_communication_slot_index,
            sim_time_s=sim_time_s,
            positions=fast.positions,
            velocities=np.zeros((self.n_uavs, 3), dtype=np.float32),
            yaws=np.zeros(self.n_uavs, dtype=np.float32),
            fsm_states=[
                "FINISH" if values["terminated"] else "EXPLORE"
            ] * self.n_uavs,
            channel_snr_db=fast.channel_snr_db,
            pair_aoi_s=fast.pair_aoi_s,
            bs_aoi_s=fast.bs_aoi_s,
            pair_missing_bytes=fast.pair_missing_bytes,
            bs_missing_bytes=fast.bs_missing_bytes,
            uplink_queue_bytes=fast.uplink_queue_bytes,
            relay_queue_bytes=fast.relay_queue_bytes,
            coverage=float(values["coverage"]),
            coverage_delta=float(values["coverage_delta"]),
            task_metrics=task_clock,
            terminated=bool(values["terminated"]),
            truncated=bool(values["truncated"]),
            communication_slot_index=self.last_communication_slot_index,
            rl_decision_index=self.last_decision_index,
            action_held_slots=int(values["action_held_slots"]),
            communication_slot_duration_s=float(
                values["communication_slot_duration_s"]
            ),
            rl_decision_interval_s=float(values["rl_decision_interval_s"]),
            state_sequence=self.last_sequence,
            step_id=step_id,
            guidance_id=int(values["guidance_id"]),
            guidance_task_dependency=fast.guidance_task_dependency,
            guidance_semantic_importance=fast.guidance_semantic_importance,
        ).validate()
        if (
            snapshot.sim_time_s + 1.0e-9
            >= self.metric_evaluator.fixed_exploration_time_s
            and not snapshot.terminated
        ):
            snapshot.truncated = True
        self.last_snapshot_terminal = bool(
            snapshot.terminated or snapshot.truncated
        )
        communication = {
            **values,
            "executed_relay_action": fast.relay_action,
            "executed_upload_action": fast.upload_action,
            "interval_start_sim_time_s": max(
                0.0, sim_time_s - self.decision_interval_s
            ),
            "interval_end_sim_time_s": sim_time_s,
        }
        return snapshot, communication

    def reset(self, seed: int | None = None) -> RacerSnapshot:
        del seed
        self.metric_evaluator.reset()
        if self.require_perfect_reference:
            self.metric_evaluator.validate_reference()
        record = self._wait_boundary()
        snapshot, _ = self._decode_boundary(record, rebase_resources=True)
        self.action_epoch = self.blocks["action"].version
        return snapshot

    def set_action_context(
        self, *, guidance_id: int = 0, policy_version: int = 0
    ) -> None:
        self.source_guidance_id = max(0, int(guidance_id))
        self.policy_version = max(0, int(policy_version))

    def _validate_action(self, action_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(action_matrix, dtype=np.int8)
        if matrix.shape != (self.n_uavs + 1, self.n_uavs):
            raise ValueError("invalid shared-memory action matrix")
        if np.any((matrix != 0) & (matrix != 1)):
            raise ValueError("shared-memory action must be binary")
        if np.any(np.diag(matrix[: self.n_uavs])):
            raise ValueError("shared-memory action contains self relay")
        if self.max_relay_recipients_per_uav is not None:
            fanout = np.sum(matrix[: self.n_uavs], axis=1)
            if np.any(fanout > self.max_relay_recipients_per_uav):
                raise ValueError("shared-memory action exceeds relay fanout")
        return matrix

    def _publish_action(
        self,
        action_matrix: np.ndarray,
        source_context: dict[str, Any] | None = None,
    ) -> int:
        matrix = self._validate_action(action_matrix)
        source = source_context or {}
        source_physical_version = int(
            source.get("consumed_physical_version", self.last_physical_version)
        )
        consumed_communication_version = int(
            source.get(
                "consumed_communication_version",
                self.last_communication_version,
            )
        )
        source_communication_version = int(
            source.get("channel_version", consumed_communication_version)
        )
        if source_communication_version != consumed_communication_version:
            raise ValueError(
                "action channel_version does not match the consumed Fast "
                "State communication version"
            )
        source_sim_step = int(
            source.get("task_step", self.last_physical_sim_step)
        )
        source_sim_time_s = float(
            source.get("sim_time_s", self.last_sim_time_s)
        )
        source_step_id = int(
            source.get("rl_decision_index", self.last_decision_index)
        )
        source_slot = int(
            source.get(
                "communication_slot_index",
                self.last_communication_slot_index,
            )
        )
        flat = [
            str(int(matrix[owner, receiver]))
            for owner in range(self.n_uavs)
            for receiver in range(self.n_uavs)
            if owner != receiver
        ]
        flat.extend(str(int(value)) for value in matrix[self.n_uavs])
        next_id = self.blocks["action"].version + 1
        generated_wall_time_ns = time.time_ns()
        content = " ".join(
            (
                str(next_id),
                str(self.n_uavs),
                str(source_physical_version),
                str(source_communication_version),
                str(self.source_guidance_id),
                str(source_sim_step),
                str(self.policy_version),
                str(generated_wall_time_ns),
                format(source_sim_time_s, ".17g"),
                str(max(0, source_step_id)),
                *flat,
            )
        ) + "\n"
        committed = self.blocks["action"].publish_bytes(
            content.encode("ascii"),
            sim_step=max(0, source_slot),
            sim_time_s=source_sim_time_s,
            version=next_id,
        )
        self.action_epoch = committed
        return committed

    def publish_action(
        self,
        action_matrix: np.ndarray,
        source_context: dict[str, Any] | None = None,
    ) -> int:
        """Publish a decision without waiting for ACK or a future boundary."""

        if self.last_snapshot_terminal:
            raise MissionEndedError("final shared-memory state was consumed")
        source = source_context or {}
        source_physical = int(
            source.get("consumed_physical_version", self.last_physical_version)
        )
        channel_version = int(
            source.get(
                "channel_version",
                source.get(
                    "consumed_communication_version",
                    self.last_communication_version,
                ),
            )
        )
        source_decision = int(
            source.get("rl_decision_index", self.last_decision_index)
        )
        self.rl_cycle_id += 1
        cycle_started_wall_time_ns = time.time_ns()
        action_id = self._publish_action(action_matrix, source_context)
        action_published_wall_time_ns = time.time_ns()
        self._latest_action_publication = {
            "action_id": action_id,
            "rl_cycle_id": self.rl_cycle_id,
            "source_physical_version": source_physical,
            "source_communication_version": channel_version,
            "channel_version": channel_version,
            "source_guidance_id": self.source_guidance_id,
            "source_decision": source_decision,
            "published_policy_version": self.policy_version,
            "cycle_started_wall_time_ns": cycle_started_wall_time_ns,
            "action_published_wall_time_ns": action_published_wall_time_ns,
        }
        return action_id

    def collect_transition(self) -> BackendStep:
        """Consume one ordered boundary independently of action publication."""

        source_slot = self.last_communication_slot_index
        source_decision = self.last_decision_index
        record = self._wait_boundary()
        snapshot, communication = self._decode_boundary(
            record, rebase_resources=True
        )
        if not bool(communication.get("has_completed_transition", False)):
            raise RuntimeError(
                "received a non-transition boundary after environment reset"
            )
        interval_s = float(communication["interval_end_sim_time_s"]) - float(
            communication["interval_start_sim_time_s"]
        )
        if not math.isclose(
            interval_s, self.decision_interval_s, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise RuntimeError(
                f"transition interval is {interval_s}, expected "
                f"{self.decision_interval_s} simulated seconds"
            )
        if (
            self.last_communication_slot_index - source_slot
            != self.slots_per_decision
            or self.last_decision_index - source_decision != 1
            or int(communication.get("action_held_slots", -1))
            not in (1, self.slots_per_decision)
        ):
            raise RuntimeError(
                "transition ring lost fixed-clock alignment: expected one "
                f"decision and either one-shot or {self.slots_per_decision}-slot "
                "action execution"
            )
        delta = np.asarray(
            (
                communication.get("interval_bs_uplink_prb_slots", 0.0),
                communication.get("interval_bs_downlink_prb_slots", 0.0),
                communication.get("interval_direct_u2u_prb_slots", 0.0),
            ),
            dtype=np.float64,
        )
        relay = np.asarray(
            communication.get("executed_relay_action"), dtype=np.int8
        )
        upload = np.asarray(
            communication.get("executed_upload_action"), dtype=np.int8
        )
        if relay.shape != (self.n_uavs, self.n_uavs) or upload.shape != (
            self.n_uavs,
        ):
            raise SharedMemoryError(
                "transition is missing the executed active_action"
            )
        executed_action = np.zeros(
            (self.n_uavs + 1, self.n_uavs), dtype=np.int8
        )
        executed_action[: self.n_uavs] = relay
        executed_action[self.n_uavs] = upload
        action_version = int(communication.get("action_version", 0))
        executed_policy_version = int(communication.get("policy_version", 0))
        action_age = float(communication.get("action_age", 0.0))
        sim_timestamp = float(
            communication.get("sim_timestamp", snapshot.sim_time_s)
        )
        publication = getattr(self, "_latest_action_publication", {})
        print(
            "RACER_RL_ACTION_CYCLE "
            + json.dumps(
                {
                    "action_id": publication.get("action_id", 0),
                    "rl_cycle_id": publication.get(
                        "rl_cycle_id", self.rl_cycle_id
                    ),
                    "source_physical_version": publication.get(
                        "source_physical_version", 0
                    ),
                    "source_communication_version": publication.get(
                        "source_communication_version", 0
                    ),
                    "source_guidance_id": publication.get(
                        "source_guidance_id", 0
                    ),
                    "published_policy_version": publication.get(
                        "published_policy_version", 0
                    ),
                    "executed_action_version": action_version,
                    "executed_policy_version": executed_policy_version,
                    "action_age": action_age,
                    "sim_timestamp": sim_timestamp,
                    "cycle_started_wall_time_ns": publication.get(
                        "cycle_started_wall_time_ns", 0
                    ),
                    "action_published_wall_time_ns": publication.get(
                        "action_published_wall_time_ns", 0
                    ),
                    "result_physical_version": self.last_physical_version,
                    "result_communication_version": (
                        self.last_communication_version
                    ),
                    "result_sim_time_s": snapshot.sim_time_s,
                    "skipped_physical_versions_total": (
                        self.skipped_physical_versions
                    ),
                    "skipped_communication_versions_total": (
                        self.skipped_communication_versions
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return BackendStep(
            snapshot,
            *delta.tolist(),
            executed_action_matrix=executed_action,
            action_version=action_version,
            policy_version=executed_policy_version,
            action_age=action_age,
            sim_timestamp=sim_timestamp,
            action_source_guidance_id=int(
                communication.get("action_source_guidance_id", 0)
            ),
            transition_step_id=source_decision,
            transition_sim_time_s=(
                source_decision * self.decision_interval_s
            ),
            action_source_step_id=int(
                communication.get("action_source_step_id", 0)
            ),
            action_source_sim_time_s=float(
                communication.get("action_source_sim_time_s", 0.0)
            ),
        )

    def step(self, action_matrix: np.ndarray) -> BackendStep:
        self.publish_action(action_matrix)
        return self.collect_transition()

    def refresh_latest_snapshot(self) -> RacerSnapshot | None:
        """Never skip ahead of the ordered transition-ring cursor."""

        return None

    def resolve_task_metrics(
        self, step_ids: list[int]
    ) -> dict[int, TaskMetrics]:
        """Consume async metric records and return exact step-id matches.

        This method is called once a rollout has stopped collecting. It may
        wait for the metric worker, but it can never delay the proxy scheduler
        or actor inference that produced the rollout.
        """

        requested = [int(value) for value in step_ids]
        if any(value < 0 for value in requested):
            raise ValueError("task metric step_id must be non-negative")
        missing = set(requested).difference(self._resolved_task_metrics)
        while missing:
            record = self.task_metric_ring.wait_for_newer(
                self.last_task_metric_sequence, stop=self._stopping
            )
            if record is None:
                raise MissionEndedError(
                    "episode ended before rollout task costs were ready"
                )
            payload = record.json()
            step_id = int(payload.get("step_id", -1))
            sim_time = float(payload.get("sim_time", -1.0))
            if (
                step_id != int(record.sim_step)
                or not math.isclose(
                    sim_time, float(record.sim_time_s),
                    rel_tol=0.0, abs_tol=1.0e-9,
                )
                or not math.isclose(
                    sim_time,
                    step_id * self.decision_interval_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                )
            ):
                raise SharedMemoryError(
                    "task metric ring record has inconsistent step_id/sim_time"
                )
            self.last_task_metric_sequence = int(record.sequence)
            if step_id != self.last_task_metric_step_id + 1:
                raise SharedMemoryError(
                    "task metric ring skipped or duplicated a canonical step: "
                    f"previous={self.last_task_metric_step_id}, current={step_id}"
                )
            self.last_task_metric_step_id = step_id
            # ReferenceTaskMetricEvaluator is intentionally kept ordered: its
            # trajectory and coverage losses are cumulative task definitions.
            metrics = self.metric_evaluator.evaluate(payload, payload)
            self._resolved_task_metrics[step_id] = metrics
            missing.discard(step_id)
        return {value: self._resolved_task_metrics[value] for value in requested}

    def synchronization_status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "status": "async_zoh",
            "sim_time_s": self.last_sim_time_s,
            "communication_slot_index": self.last_communication_slot_index,
            "rl_decision_index": self.last_decision_index,
            "task_step": self.last_decision_index,
            "task_time_s": self.last_sim_time_s,
            # RL is the sole action-block writer, so this is the immutable ID
            # that the next inference result will receive when it is committed.
            "next_action_version": self.action_epoch + 1,
            "physical_version": self.last_physical_version,
            "communication_version": self.last_communication_version,
            "channel_version": self.last_communication_version,
            "consumed_physical_version": self.last_physical_version,
            "consumed_communication_version": (
                self.last_communication_version
            ),
            "transition_ring_write_sequence": self.transition_ring.version,
            "transition_ring_consumed_sequence": (
                self.last_transition_sequence
            ),
            "transition_ring_backlog": max(
                0,
                self.transition_ring.version - self.last_transition_sequence,
            ),
            "task_metric_ring_write_sequence": self.task_metric_ring.version,
            "task_metric_ring_consumed_sequence": (
                self.last_task_metric_sequence
            ),
            "skipped_physical_versions": self.skipped_physical_versions,
            "skipped_communication_versions": (
                self.skipped_communication_versions
            ),
        }

    def close(self) -> None:
        try:
            self.blocks["status_rl"].publish_json(
                {
                    "ready": False,
                    "stopping": True,
                    "process": "rl",
                    "pid": os.getpid(),
                },
                sim_step=max(0, self.last_communication_slot_index),
                sim_time_s=self.last_sim_time_s,
            )
        finally:
            for block in self.blocks.values():
                block.close()
            self.transition_ring.close()
            self.task_metric_ring.close()
