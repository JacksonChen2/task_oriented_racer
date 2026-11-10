"""Simulation-time boundary barrier for synchronous online CRPO/PPO."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Callable


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class SynchronousRlBoundaryGate:
    """Freeze Isaac at every fixed RL boundary until its action is ready.

    The communication proxy writes the boundary state.  Isaac acknowledges
    that both physics and /clock are frozen, then waits for a trainer release
    record whose action epoch and decision index match the action file.  No
    wall-clock duration is used to determine a boundary.
    """

    def __init__(
        self,
        state_path: Path | None,
        action_path: Path | None,
        release_path: Path | None,
        acknowledgement_path: Path | None,
        *,
        communication_slot_s: float = 0.02,
        decision_interval_s: float = 0.1,
        slots_per_decision: int = 5,
        maximum_wait_s: float = 1200.0,
        poll_interval_s: float = 0.002,
        boundary_tolerance_s: float = 1.0e-6,
    ) -> None:
        paths = (state_path, action_path, release_path, acknowledgement_path)
        if any(path is None for path in paths) and any(
            path is not None for path in paths
        ):
            raise ValueError("all synchronous-RL bridge paths are required")
        if communication_slot_s <= 0.0 or decision_interval_s <= 0.0:
            raise ValueError("synchronous-RL simulation intervals must be positive")
        if slots_per_decision < 1:
            raise ValueError("slots_per_decision must be positive")
        if maximum_wait_s <= 0.0 or poll_interval_s <= 0.0:
            raise ValueError("synchronous-RL wall-time fail-safes must be positive")
        if boundary_tolerance_s <= 0.0:
            raise ValueError("boundary_tolerance_s must be positive")
        expected = communication_slot_s * slots_per_decision
        if not math.isclose(
            decision_interval_s, expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "decision interval must equal communication slot times "
                "slots_per_decision"
            )
        self.state_path = state_path
        self.action_path = action_path
        self.release_path = release_path
        self.acknowledgement_path = acknowledgement_path
        self.communication_slot_s = float(communication_slot_s)
        self.decision_interval_s = float(decision_interval_s)
        self.slots_per_decision = int(slots_per_decision)
        self.maximum_wait_s = float(maximum_wait_s)
        self.poll_interval_s = float(poll_interval_s)
        self.boundary_tolerance_s = float(boundary_tolerance_s)
        self.next_decision_index = 0
        self.boundary_count = 0
        self.terminal_boundary_count = 0
        self.total_frozen_wall_s = 0.0
        self.maximum_frozen_wall_s = 0.0
        self.last_boundary: dict[str, object] | None = None
        if self.enabled:
            assert self.release_path is not None
            assert self.acknowledgement_path is not None
            self.release_path.unlink(missing_ok=True)
            self.acknowledgement_path.unlink(missing_ok=True)

    @property
    def enabled(self) -> bool:
        return self.state_path is not None

    def boundary_due(self, sim_time_s: float) -> bool:
        if not self.enabled:
            return False
        expected = self.next_decision_index * self.decision_interval_s
        if sim_time_s < expected - self.boundary_tolerance_s:
            return False
        if sim_time_s > expected + self.boundary_tolerance_s:
            raise RuntimeError(
                "Isaac crossed a synchronous RL boundary without stopping: "
                f"expected={expected:.9f} actual={sim_time_s:.9f}"
            )
        return True

    def _wait_for_state(
        self,
        decision_index: int,
        sim_time_s: float,
        is_running: Callable[[], bool],
        state_keepalive: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        assert self.state_path is not None
        deadline = time.monotonic() + self.maximum_wait_s
        next_keepalive = time.monotonic() + 0.05
        last_state: dict[str, object] | None = None
        while is_running() and time.monotonic() < deadline:
            state = _read_json(self.state_path)
            if state is not None:
                last_state = state
                observed_decision = int(state.get("rl_decision_index", -1))
                if observed_decision > decision_index:
                    raise RuntimeError(
                        "communication state skipped an RL boundary: "
                        f"expected={decision_index} observed={observed_decision}"
                    )
                if observed_decision == decision_index:
                    observed_time = float(state.get("sim_time_s", math.nan))
                    observed_slot = int(
                        state.get("communication_slot_index", -1)
                    )
                    expected_slot = decision_index * self.slots_per_decision
                    if not math.isclose(
                        observed_time,
                        sim_time_s,
                        rel_tol=0.0,
                        abs_tol=self.boundary_tolerance_s,
                    ):
                        raise RuntimeError(
                            "communication/Isaac simulation-time mismatch at "
                            f"decision {decision_index}: communication="
                            f"{observed_time:.9f} isaac={sim_time_s:.9f}"
                        )
                    if observed_slot != expected_slot:
                        raise RuntimeError(
                            "communication slot mismatch at RL boundary: "
                            f"expected={expected_slot} observed={observed_slot}"
                        )
                    return state
            now = time.monotonic()
            if state_keepalive is not None and now >= next_keepalive:
                # /clock is not transient-local. Re-publish the exact frozen
                # timestamp until the proxy confirms s_t, covering DDS races
                # without advancing physics or simulation time.
                state_keepalive()
                next_keepalive = now + 0.05
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            "communication proxy did not publish the synchronous RL boundary "
            f"state decision={decision_index}; last_state={last_state}"
        )

    def _action_header(self) -> tuple[int, int] | None:
        assert self.action_path is not None
        try:
            fields = self.action_path.read_text(encoding="ascii").split()
            return int(fields[0]), int(fields[2])
        except (OSError, ValueError, IndexError):
            return None

    def _wait_for_release(
        self, decision_index: int, is_running: Callable[[], bool]
    ) -> int:
        assert self.release_path is not None
        deadline = time.monotonic() + self.maximum_wait_s
        last_release: dict[str, object] | None = None
        while is_running() and time.monotonic() < deadline:
            release = _read_json(self.release_path)
            if release is not None:
                last_release = release
                if int(release.get("rl_decision_index", -1)) == decision_index:
                    action_epoch = int(release.get("action_epoch", -1))
                    if self._action_header() == (action_epoch, decision_index):
                        return action_epoch
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            "trainer did not release the synchronous RL boundary "
            f"decision={decision_index}; last_release={last_release}"
        )

    def _acknowledge(
        self,
        *,
        decision_index: int,
        communication_slot_index: int,
        state_sequence: int,
        sim_time_s: float,
        status: str,
        **extra: object,
    ) -> None:
        assert self.acknowledgement_path is not None
        payload: dict[str, object] = {
            "status": status,
            "rl_decision_index": decision_index,
            "communication_slot_index": communication_slot_index,
            "state_sequence": state_sequence,
            "sim_time_s": float(sim_time_s),
            "isaac_pid": os.getpid(),
            "updated_at_unix_s": time.time(),
        }
        payload.update(extra)
        _atomic_json(self.acknowledgement_path, payload)
        self.last_boundary = payload

    def wait_at_boundary(
        self,
        sim_time_s: float,
        is_running: Callable[[], bool],
        *,
        terminal: bool = False,
        state_keepalive: Callable[[], None] | None = None,
    ) -> bool:
        if not self.boundary_due(sim_time_s):
            return False
        decision_index = self.next_decision_index
        state = self._wait_for_state(
            decision_index,
            sim_time_s,
            is_running,
            state_keepalive=state_keepalive,
        )
        communication_slot_index = int(state["communication_slot_index"])
        state_sequence = int(state["sequence"])
        started = time.monotonic()
        status = "terminal" if terminal else "paused"
        self._acknowledge(
            decision_index=decision_index,
            communication_slot_index=communication_slot_index,
            state_sequence=state_sequence,
            sim_time_s=sim_time_s,
            status=status,
            action_held_slots=int(state.get("action_held_slots", 0)),
        )
        print(
            "RACER_RL_SYNC_BOUNDARY_PAUSED "
            + json.dumps(self.last_boundary, separators=(",", ":")),
            flush=True,
        )

        action_epoch: int | None = None
        if terminal:
            self.terminal_boundary_count += 1
        else:
            action_epoch = self._wait_for_release(decision_index, is_running)
        elapsed = time.monotonic() - started
        self.boundary_count += 1
        self.total_frozen_wall_s += elapsed
        self.maximum_frozen_wall_s = max(self.maximum_frozen_wall_s, elapsed)
        if not terminal:
            self._acknowledge(
                decision_index=decision_index,
                communication_slot_index=communication_slot_index,
                state_sequence=state_sequence,
                sim_time_s=sim_time_s,
                status="resumed",
                action_epoch=action_epoch,
                frozen_wall_s=elapsed,
                simulation_time_unchanged=True,
            )
            print(
                "RACER_RL_SYNC_BOUNDARY_RESUMED "
                + json.dumps(self.last_boundary, separators=(",", ":")),
                flush=True,
            )
        self.next_decision_index += 1
        return True

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "communication_slot_s": self.communication_slot_s,
            "decision_interval_s": self.decision_interval_s,
            "slots_per_decision": self.slots_per_decision,
            "boundary_count": self.boundary_count,
            "terminal_boundary_count": self.terminal_boundary_count,
            "total_frozen_wall_s": self.total_frozen_wall_s,
            "maximum_frozen_wall_s": self.maximum_frozen_wall_s,
            "next_decision_index": self.next_decision_index,
            "last_boundary": self.last_boundary,
        }
