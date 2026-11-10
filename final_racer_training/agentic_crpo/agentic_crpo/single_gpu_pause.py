"""File-based pause barrier for one-GPU Isaac/Qwen time sharing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Iterator


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


@dataclass(frozen=True)
class SingleGpuPauseConfig:
    request_path: Path
    acknowledgement_path: Path
    timeout_s: float = 600.0
    poll_interval_s: float = 0.01

    def validate(self) -> "SingleGpuPauseConfig":
        if self.timeout_s <= 0.0 or self.poll_interval_s <= 0.0:
            raise ValueError("single-GPU pause timeouts must be positive")
        if self.request_path.resolve() == self.acknowledgement_path.resolve():
            raise ValueError("pause request and acknowledgement paths must differ")
        return self


class SingleGpuPauseController:
    """Pause Isaac before a local Qwen inference and resume it afterwards."""

    def __init__(self, config: SingleGpuPauseConfig) -> None:
        self.config = config.validate()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self.pause_count = 0
        self.pause_wait_s = 0.0

    def cancel_pending(self) -> None:
        """Cancel a request that can no longer be served by a finished Isaac run."""

        self._cancelled.set()
        request = _read_json(self.config.request_path)
        if request is not None and int(request.get("owner_pid", -1)) == os.getpid():
            self.config.request_path.unlink(missing_ok=True)

    def _raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            request = _read_json(self.config.request_path)
            if (
                request is not None
                and int(request.get("owner_pid", -1)) == os.getpid()
            ):
                self.config.request_path.unlink(missing_ok=True)
            raise RuntimeError("single-GPU pause request was cancelled")

    def _clear_stale_request(self) -> None:
        request = _read_json(self.config.request_path)
        if request is None:
            self.config.request_path.unlink(missing_ok=True)
            return
        owner_pid = int(request.get("owner_pid", -1))
        if _process_alive(owner_pid):
            raise RuntimeError(
                "another live process owns the single-GPU pause request: "
                f"pid={owner_pid} path={self.config.request_path}"
            )
        self.config.request_path.unlink(missing_ok=True)

    def acquire(self) -> str:
        with self._lock:
            self._raise_if_cancelled()
            self._clear_stale_request()
            token = f"{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
            request = {
                "token": token,
                "owner_pid": os.getpid(),
                "created_at_unix_s": time.time(),
                "timeout_s": self.config.timeout_s,
            }
            _atomic_json(self.config.request_path, request)
            started = time.monotonic()
            deadline = started + self.config.timeout_s
            while time.monotonic() < deadline:
                self._raise_if_cancelled()
                acknowledgement = _read_json(
                    self.config.acknowledgement_path
                )
                if (
                    acknowledgement is not None
                    and acknowledgement.get("token") == token
                ):
                    status = acknowledgement.get("status")
                    if status == "paused":
                        self.pause_count += 1
                        self.pause_wait_s += time.monotonic() - started
                        return token
                    if status in {"expired", "owner_gone", "error"}:
                        self.config.request_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            "Isaac rejected the single-GPU pause request: "
                            f"{acknowledgement}"
                        )
                time.sleep(self.config.poll_interval_s)
            self.config.request_path.unlink(missing_ok=True)
            raise TimeoutError(
                "Isaac did not acknowledge the single-GPU pause request "
                f"within {self.config.timeout_s:.1f} s"
            )

    def release(self, token: str) -> None:
        request = _read_json(self.config.request_path)
        if request is not None and request.get("token") == token:
            self.config.request_path.unlink(missing_ok=True)
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            if self._cancelled.is_set():
                return
            acknowledgement = _read_json(self.config.acknowledgement_path)
            if acknowledgement is None:
                return
            if acknowledgement.get("token") != token:
                return
            if acknowledgement.get("status") in {
                "resumed",
                "expired",
                "owner_gone",
            }:
                return
            time.sleep(self.config.poll_interval_s)
        raise TimeoutError(
            "Isaac did not resume after the single-GPU Qwen inference "
            f"within {self.config.timeout_s:.1f} s"
        )

    @contextmanager
    def paused(self) -> Iterator[None]:
        token = self.acquire()
        try:
            yield
        finally:
            self.release(token)
