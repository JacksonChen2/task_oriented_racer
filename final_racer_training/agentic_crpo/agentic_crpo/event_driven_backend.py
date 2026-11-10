"""Opt-in shared-memory backend for one-shot, event-driven BS scheduling.

The fixed 100 ms / five-slot :class:`SharedMemoryBackend` remains unchanged.
This backend consumes the same compact ABI with event-specific semantics:
``step_id`` counts completed actions, ``rl_decision_interval_s`` is the actual
variable transition duration, and every completed action must report exactly
one held communication slot.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .backend import MissionEndedError, SharedMemoryBackend
from .fast_state import decode_fast_state
from .schemas import BackendStep, RacerSnapshot, TaskMetrics
from .shared_ipc import SharedMemoryError


class EventDrivenOneShotSharedMemoryBackend(SharedMemoryBackend):
    """Consume action-triggered transitions with strict provenance checks."""

    scheduling_mode = "event_driven_one_shot"
    # One supervisor process owns exactly one finite Isaac mission.  Gym's
    # automatic terminal reset must therefore retain the completed-state
    # timeline until the final partial rollout has backfilled its async task
    # metrics; there is no second episode in this backend instance.
    single_mission_terminal_reset_is_noop = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # These are nominal PHY values only. Each decoded state carries the
        # measured event-to-event duration separately.
        self.communication_slot_s = 0.02
        self.decision_interval_s = self.communication_slot_s
        self.slots_per_decision = 1
        self.metric_evaluator.variable_task_clock = True
        self._event_previous_state_time_s: float | None = None
        self._event_previous_slot: int | None = None
        self._event_state_time_by_step: dict[int, float] = {}

    def _decode_boundary(
        self, record: Any, *, rebase_resources: bool
    ) -> tuple[RacerSnapshot, dict[str, Any]]:
        try:
            fast = decode_fast_state(record.payload, self.n_uavs)
        except ValueError as error:
            raise SharedMemoryError(str(error)) from error
        values = fast.values
        step_id = int(values["step_id"])
        decision_index = int(values["rl_decision_index"])
        communication_slot = int(values["communication_slot_index"])
        sim_time_s = float(values["sim_time_s"])
        communication_slot_s = float(values["communication_slot_duration_s"])
        delta_t_s = float(values["rl_decision_interval_s"])
        completed = bool(values["has_completed_transition"])

        if int(record.sim_step) != step_id or not math.isclose(
            float(record.sim_time_s), sim_time_s, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise SharedMemoryError(
                "event-driven fast-state ring metadata is not aligned"
            )
        if decision_index != step_id:
            raise SharedMemoryError(
                "event-driven decision index must count completed actions"
            )
        if (
            not math.isfinite(communication_slot_s)
            or communication_slot_s <= 0.0
            or not math.isfinite(delta_t_s)
            or delta_t_s <= 0.0
        ):
            raise SharedMemoryError(
                "event-driven state has an invalid slot or delta_t"
            )
        if not math.isclose(
            sim_time_s,
            communication_slot * communication_slot_s,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise SharedMemoryError(
                "event-driven state is not aligned to a physical slot"
            )

        previous_time = self._event_previous_state_time_s
        previous_slot = self._event_previous_slot
        if previous_time is None:
            if completed or step_id != 0:
                raise SharedMemoryError(
                    "first event-driven state must be the initial state"
                )
            interval_start_sim_time_s = sim_time_s
        else:
            if not completed:
                raise SharedMemoryError(
                    "only reset may emit a non-transition event state"
                )
            if step_id != self.last_decision_index + 1:
                raise SharedMemoryError(
                    "event-driven completed-action index is not contiguous"
                )
            if communication_slot <= int(previous_slot):
                raise SharedMemoryError(
                    "event-driven result did not advance a communication slot"
                )
            measured_delta_t_s = sim_time_s - previous_time
            if not math.isclose(
                measured_delta_t_s,
                delta_t_s,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            ):
                raise SharedMemoryError(
                    "event-driven delta_t does not match adjacent states"
                )
            if int(values["action_held_slots"]) != 1:
                raise SharedMemoryError(
                    "event-driven action must be scheduled exactly once"
                )
            interval_start_sim_time_s = previous_time

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
        self.last_sequence = communication_version
        self.last_decision_index = decision_index
        self.last_communication_slot_index = communication_slot
        self.last_sim_time_s = sim_time_s
        self.communication_slot_s = communication_slot_s
        self._event_previous_state_time_s = sim_time_s
        self._event_previous_slot = communication_slot
        self._event_state_time_by_step[step_id] = sim_time_s

        if rebase_resources:
            self.last_resources = np.zeros(3, dtype=np.float64)
        task_clock = TaskMetrics(task_step=step_id, task_time_s=sim_time_s)
        snapshot = RacerSnapshot(
            n_uavs=self.n_uavs,
            slot=communication_slot,
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
            communication_slot_index=communication_slot,
            rl_decision_index=decision_index,
            action_held_slots=int(values["action_held_slots"]),
            communication_slot_duration_s=communication_slot_s,
            rl_decision_interval_s=delta_t_s,
            state_sequence=communication_version,
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
            "interval_start_sim_time_s": interval_start_sim_time_s,
            "interval_end_sim_time_s": sim_time_s,
            "delta_t_s": (
                0.0 if previous_time is None else sim_time_s - previous_time
            ),
            "scheduling_mode": self.scheduling_mode,
        }
        return snapshot, communication

    def reset(self, seed: int | None = None) -> RacerSnapshot:
        self._event_previous_state_time_s = None
        self._event_previous_slot = None
        self._event_state_time_by_step.clear()
        snapshot = super().reset(seed)
        if snapshot.step_id != 0 or snapshot.action_held_slots != 0:
            raise SharedMemoryError(
                "event-driven reset did not receive an idle initial state"
            )
        return snapshot

    def collect_transition(self) -> BackendStep:
        source_slot = self.last_communication_slot_index
        source_decision = self.last_decision_index
        source_sim_time_s = self.last_sim_time_s
        record = self._wait_boundary()
        snapshot, communication = self._decode_boundary(
            record, rebase_resources=True
        )
        if not bool(communication["has_completed_transition"]):
            raise RuntimeError(
                "event-driven collector received an incomplete action event"
            )
        delta_t_s = float(communication["delta_t_s"])
        if delta_t_s < self.communication_slot_s - 1.0e-9:
            raise RuntimeError(
                "event-driven transition is shorter than one physical slot"
            )
        if (
            snapshot.rl_decision_index != source_decision + 1
            or snapshot.communication_slot_index <= source_slot
            or snapshot.action_held_slots != 1
        ):
            raise RuntimeError(
                "event-driven transition violated one-shot ordering"
            )

        relay = np.asarray(
            communication["executed_relay_action"], dtype=np.int8
        )
        upload = np.asarray(
            communication["executed_upload_action"], dtype=np.int8
        )
        if relay.shape != (self.n_uavs, self.n_uavs) or upload.shape != (
            self.n_uavs,
        ):
            raise SharedMemoryError(
                "event-driven transition is missing its executed action"
            )
        executed_action = np.zeros(
            (self.n_uavs + 1, self.n_uavs), dtype=np.int8
        )
        executed_action[: self.n_uavs] = relay
        executed_action[self.n_uavs] = upload
        action_version = int(communication["action_version"])
        publication = getattr(self, "_latest_action_publication", {})
        published_action = int(publication.get("action_id", 0))
        if published_action and action_version != published_action:
            raise SharedMemoryError(
                "event-driven result does not match the outstanding action"
            )

        action_age = float(communication["action_age"])
        result_resources = np.asarray(
            (
                communication["interval_bs_uplink_prb_slots"],
                communication["interval_bs_downlink_prb_slots"],
                communication["interval_direct_u2u_prb_slots"],
            ),
            dtype=np.float64,
        )
        print(
            "RACER_EVENT_RL_ACTION_CYCLE "
            + json.dumps(
                {
                    "action_id": published_action,
                    "rl_cycle_id": publication.get(
                        "rl_cycle_id", self.rl_cycle_id
                    ),
                    "source_step_id": source_decision,
                    "result_step_id": snapshot.rl_decision_index,
                    "source_sim_time_s": source_sim_time_s,
                    "result_sim_time_s": snapshot.sim_time_s,
                    "delta_t_s": delta_t_s,
                    "source_communication_slot": source_slot,
                    "result_communication_slot": (
                        snapshot.communication_slot_index
                    ),
                    "action_version": action_version,
                    "policy_version": int(communication["policy_version"]),
                    "action_age_s": action_age,
                    "action_held_slots": snapshot.action_held_slots,
                    "one_shot": True,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return BackendStep(
            snapshot,
            *result_resources.tolist(),
            executed_action_matrix=executed_action,
            action_version=action_version,
            policy_version=int(communication["policy_version"]),
            action_age=action_age,
            sim_timestamp=snapshot.sim_time_s,
            action_source_guidance_id=int(
                communication["action_source_guidance_id"]
            ),
            transition_step_id=source_decision,
            transition_sim_time_s=source_sim_time_s,
            action_source_step_id=int(
                communication["action_source_step_id"]
            ),
            action_source_sim_time_s=float(
                communication["action_source_sim_time_s"]
            ),
        )

    def resolve_task_metrics(
        self, step_ids: list[int]
    ) -> dict[int, TaskMetrics]:
        """Resolve metrics against event timestamps, never ``step * 0.1``."""

        requested = [int(value) for value in step_ids]
        if any(value < 0 for value in requested):
            raise ValueError("task metric step_id must be non-negative")
        unknown = set(requested).difference(self._event_state_time_by_step)
        if unknown:
            raise SharedMemoryError(
                "event-driven task metrics were requested before their "
                f"states were consumed: {sorted(unknown)}"
            )
        missing = set(requested).difference(self._resolved_task_metrics)
        while missing:
            record = self.task_metric_ring.wait_for_newer(
                self.last_task_metric_sequence, stop=self._stopping
            )
            if record is None:
                raise MissionEndedError(
                    "episode ended before event-driven rollout task costs "
                    "were ready"
                )
            payload = record.json()
            step_id = int(payload.get("step_id", -1))
            sim_time_s = float(payload.get("sim_time", -1.0))
            expected_time_s = self._event_state_time_by_step.get(step_id)
            if (
                step_id != int(record.sim_step)
                or expected_time_s is None
                or not math.isclose(
                    sim_time_s,
                    float(record.sim_time_s),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    sim_time_s,
                    expected_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                )
            ):
                raise SharedMemoryError(
                    "event-driven task metric is not aligned with its "
                    "consumed state timestamp"
                )
            self.last_task_metric_sequence = int(record.sequence)
            if step_id != self.last_task_metric_step_id + 1:
                raise SharedMemoryError(
                    "event-driven task metric ring skipped or duplicated a "
                    "state: "
                    f"previous={self.last_task_metric_step_id}, "
                    f"current={step_id}"
                )
            self.last_task_metric_step_id = step_id
            metrics = self.metric_evaluator.evaluate(payload, payload)
            self._resolved_task_metrics[step_id] = metrics
            missing.discard(step_id)
        return {
            value: self._resolved_task_metrics[value] for value in requested
        }

    def synchronization_status(self) -> dict[str, Any]:
        status = super().synchronization_status()
        status.update(
            {
                "status": self.scheduling_mode,
                "one_shot": True,
                "variable_delta_t": True,
                "communication_slot_duration_s": self.communication_slot_s,
            }
        )
        return status
