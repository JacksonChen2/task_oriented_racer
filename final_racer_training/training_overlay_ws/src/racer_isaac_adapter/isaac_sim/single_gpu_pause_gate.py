"""Isaac-side file barrier for deterministic single-GPU Qwen time sharing."""

from __future__ import annotations

import json
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


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SingleGpuPauseGate:
    """Stop physics/ROS stepping while an external Qwen process owns the GPU."""

    def __init__(
        self,
        request_path: Path | None,
        acknowledgement_path: Path | None,
        *,
        maximum_pause_s: float = 600.0,
        poll_interval_s: float = 0.01,
    ) -> None:
        if (request_path is None) != (acknowledgement_path is None):
            raise ValueError("both single-GPU pause paths must be configured")
        if maximum_pause_s <= 0.0 or poll_interval_s <= 0.0:
            raise ValueError("single-GPU pause intervals must be positive")
        self.request_path = request_path
        self.acknowledgement_path = acknowledgement_path
        self.maximum_pause_s = float(maximum_pause_s)
        self.poll_interval_s = float(poll_interval_s)
        self.pause_count = 0
        self.total_pause_wall_s = 0.0
        self.maximum_pause_wall_s = 0.0
        self.expired_requests = 0
        if self.enabled:
            assert self.acknowledgement_path is not None
            self.acknowledgement_path.unlink(missing_ok=True)

    @property
    def enabled(self) -> bool:
        return self.request_path is not None

    def _acknowledge(
        self, token: str, status: str, sim_time_s: float, **extra: object
    ) -> None:
        assert self.acknowledgement_path is not None
        payload: dict[str, object] = {
            "token": token,
            "status": status,
            "sim_time_s": float(sim_time_s),
            "isaac_pid": os.getpid(),
            "updated_at_unix_s": time.time(),
        }
        payload.update(extra)
        _atomic_json(self.acknowledgement_path, payload)

    def wait_if_requested(
        self, sim_time_s: float, is_running: Callable[[], bool]
    ) -> bool:
        if not self.enabled:
            return False
        assert self.request_path is not None
        request = _read_json(self.request_path)
        if request is None:
            return False
        token = str(request.get("token", ""))
        owner_pid = int(request.get("owner_pid", -1))
        created_at = float(request.get("created_at_unix_s", 0.0))
        requested_timeout = float(
            request.get("timeout_s", self.maximum_pause_s)
        )
        allowed_pause = min(
            self.maximum_pause_s, max(0.0, requested_timeout)
        )
        if not token or allowed_pause <= 0.0:
            self._acknowledge(token, "error", sim_time_s)
            return False
        if not _process_alive(owner_pid):
            self._acknowledge(token, "owner_gone", sim_time_s)
            return False

        started = time.monotonic()
        self._acknowledge(token, "paused", sim_time_s)
        status = "resumed"
        while is_running():
            current = _read_json(self.request_path)
            if current is None or current.get("token") != token:
                break
            if not _process_alive(owner_pid):
                status = "owner_gone"
                break
            if time.time() - created_at >= allowed_pause:
                status = "expired"
                self.expired_requests += 1
                break
            # Deliberately do not call World.step(), SimulationApp.update(),
            # or rclpy.spin_once() here. Physics, sensors, /clock and RACER
            # callbacks remain frozen while Qwen owns the one GPU.
            time.sleep(self.poll_interval_s)

        elapsed = time.monotonic() - started
        self.pause_count += 1
        self.total_pause_wall_s += elapsed
        self.maximum_pause_wall_s = max(self.maximum_pause_wall_s, elapsed)
        self._acknowledge(
            token, status, sim_time_s, pause_wall_s=elapsed
        )
        return True

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "pause_count": self.pause_count,
            "total_pause_wall_s": self.total_pause_wall_s,
            "maximum_pause_wall_s": self.maximum_pause_wall_s,
            "expired_requests": self.expired_requests,
            "request_path": (
                str(self.request_path) if self.request_path is not None else None
            ),
            "acknowledgement_path": (
                str(self.acknowledgement_path)
                if self.acknowledgement_path is not None
                else None
            ),
        }

