"""Cross-interpreter, cross-language shared-memory transport.

The files live on Linux tmpfs (normally ``/dev/shm``) and are memory mapped by
independent processes. Ordinary state/mailbox blocks have one writer and two
payload slots. A writer fills the inactive slot and commits it by changing the
active slot and monotonically increasing ``version`` while holding a very
short ``flock``. Readers copy one complete active slot under a shared lock and
then decode it in private memory. The transition stream uses a separate
single-writer fixed-slot ring so optimizer latency cannot collapse boundaries.

Futex words are notification only; versions/sequences are authoritative.
Ordinary blocks remain latest-state-wins, while the ring preserves every
retained record in sequence order.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import fcntl
import json
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Any, Callable, Iterable
import zlib


MAGIC = b"FRSHM01\0"
ABI_VERSION = 1
HEADER_SIZE = 128
DEFAULT_CAPACITY = 2 * 1024 * 1024

RING_MAGIC = b"FRRNG01\0"
RING_ABI_VERSION = 1
RING_HEADER_SIZE = 128
RING_SLOT_HEADER_SIZE = 32
DEFAULT_RING_SLOTS = 256

# magic, abi, capacity, version, active, len[2], crc[2], futex,
# sim_step[2], sim_time[2], writer_pid, flags
_HEADER = struct.Struct("<8sIIQIIIIIIQQddII")
_FUTEX_OFFSET = 44
_RING_HEADER = struct.Struct("<8sIIIIQII")
_RING_SLOT_HEADER = struct.Struct("<QQdII")
_RING_FUTEX_OFFSET = 32
_SYS_FUTEX = 202  # x86_64 Linux
_FUTEX_WAIT = 0
_FUTEX_WAKE = 1
_INT_MAX = (1 << 31) - 1
_LIBC = ctypes.CDLL(None, use_errno=True)


class SharedMemoryError(RuntimeError):
    """The shared-memory ABI or a committed payload is invalid."""


class SharedRingOverflowError(SharedMemoryError):
    """A ring consumer fell far enough behind that records were overwritten."""


@dataclass(frozen=True)
class SharedSnapshot:
    version: int
    sim_step: int
    sim_time_s: float
    writer_pid: int
    payload: bytes

    def json(self) -> dict[str, Any]:
        value = json.loads(self.payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise SharedMemoryError("shared JSON payload is not an object")
        return value


@dataclass(frozen=True)
class SharedRingSnapshot:
    sequence: int
    sim_step: int
    sim_time_s: float
    payload: bytes

    def json(self) -> dict[str, Any]:
        value = json.loads(self.payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise SharedMemoryError("shared ring JSON payload is not an object")
        return value


@dataclass(frozen=True)
class SharedMemoryLayout:
    root: Path

    JSON_BLOCKS = (
        "physical",
        "physical_fast",
        "communication",
        "state_event",
        "guidance",
        "guidance_fast",
        "action",
        "action_ack",
        "shutdown",
        "status_isaac",
        "status_racer",
        "status_llm",
        "status_rl",
    )
    RING_BLOCKS = ("transition_ring", "task_metric_ring")
    BLOCKS = JSON_BLOCKS + RING_BLOCKS

    @classmethod
    def from_value(cls, value: str | os.PathLike[str]) -> "SharedMemoryLayout":
        root = Path(value).expanduser().resolve()
        if root.parent != Path("/dev/shm"):
            raise ValueError("shared-memory root must be a direct child of /dev/shm")
        return cls(root)

    def path(self, name: str) -> Path:
        if name not in self.BLOCKS:
            raise KeyError(f"unknown shared block: {name}")
        return self.root / f"{name}.shm"


class _Timespec(ctypes.Structure):
    _fields_ = (("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long))


class SharedJsonBlock:
    """A versioned double-buffer block containing UTF-8 JSON or opaque bytes."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
        size = os.fstat(self._fd).st_size
        if size < HEADER_SIZE + 2:
            os.close(self._fd)
            raise SharedMemoryError(f"shared block is too small: {self.path}")
        self._map = mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE)
        self._futex_ref = ctypes.c_uint32.from_buffer(
            self._map, _FUTEX_OFFSET
        )
        header = self._unpack_header()
        if header[0] != MAGIC or header[1] != ABI_VERSION:
            self.close()
            raise SharedMemoryError(f"shared block ABI mismatch: {self.path}")
        self.capacity = int(header[2])
        if size != HEADER_SIZE + 2 * self.capacity:
            self.close()
            raise SharedMemoryError(f"shared block size mismatch: {self.path}")

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        capacity: int = DEFAULT_CAPACITY,
    ) -> "SharedJsonBlock":
        target = Path(path)
        if capacity < 1024:
            raise ValueError("shared block capacity must be at least 1024 bytes")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(
            target,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.ftruncate(fd, HEADER_SIZE + 2 * capacity)
            mapping = mmap.mmap(fd, HEADER_SIZE + 2 * capacity)
            try:
                values = (
                    MAGIC,
                    ABI_VERSION,
                    capacity,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0,
                    0,
                )
                mapping[: _HEADER.size] = _HEADER.pack(*values)
                mapping[HEADER_SIZE:] = b"\0" * (2 * capacity)
                mapping.flush()
            finally:
                mapping.close()
        finally:
            os.close(fd)
        return cls(target)

    def _unpack_header(self) -> tuple[Any, ...]:
        return _HEADER.unpack_from(self._map, 0)

    @property
    def version(self) -> int:
        fcntl.flock(self._fd, fcntl.LOCK_SH)
        try:
            return int(self._unpack_header()[3])
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    @property
    def notification_sequence(self) -> int:
        return int(self._futex_ref.value)

    def publish_bytes(
        self,
        payload: bytes,
        *,
        sim_step: int = 0,
        sim_time_s: float = 0.0,
        version: int | None = None,
        flags: int = 0,
    ) -> int:
        data = bytes(payload)
        if len(data) > self.capacity:
            raise SharedMemoryError(
                f"payload {len(data)} exceeds {self.capacity} bytes in {self.path}"
            )
        if sim_step < 0:
            raise ValueError("sim_step must be non-negative")
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            fields = list(self._unpack_header())
            current_version = int(fields[3])
            next_version = current_version + 1 if version is None else int(version)
            if next_version <= current_version:
                raise SharedMemoryError(
                    f"non-monotonic publish {next_version} <= {current_version}"
                )
            inactive = 1 - int(fields[4])
            start = HEADER_SIZE + inactive * self.capacity
            self._map[start : start + len(data)] = data
            lengths = [int(fields[5]), int(fields[6])]
            checksums = [int(fields[7]), int(fields[8])]
            steps = [int(fields[10]), int(fields[11])]
            times = [float(fields[12]), float(fields[13])]
            lengths[inactive] = len(data)
            checksums[inactive] = zlib.crc32(data) & 0xFFFFFFFF
            steps[inactive] = int(sim_step)
            times[inactive] = float(sim_time_s)
            futex_word = (int(fields[9]) + 1) & 0xFFFFFFFF
            packed = _HEADER.pack(
                MAGIC,
                ABI_VERSION,
                self.capacity,
                next_version,
                inactive,
                lengths[0],
                lengths[1],
                checksums[0],
                checksums[1],
                futex_word,
                steps[0],
                steps[1],
                times[0],
                times[1],
                os.getpid(),
                int(flags),
            )
            self._map[: _HEADER.size] = packed
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._futex_wake()
        return next_version

    def publish_json(
        self,
        payload: dict[str, Any],
        *,
        sim_step: int = 0,
        sim_time_s: float = 0.0,
        version: int | None = None,
        flags: int = 0,
    ) -> int:
        data = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return self.publish_bytes(
            data,
            sim_step=sim_step,
            sim_time_s=sim_time_s,
            version=version,
            flags=flags,
        )

    def _snapshot_unlocked(self) -> SharedSnapshot | None:
        fields = self._unpack_header()
        version = int(fields[3])
        if version == 0:
            return None
        active = int(fields[4])
        length = int(fields[5 + active])
        checksum = int(fields[7 + active])
        if active not in (0, 1) or not 0 <= length <= self.capacity:
            raise SharedMemoryError(f"invalid active slot in {self.path}")
        start = HEADER_SIZE + active * self.capacity
        payload = bytes(self._map[start : start + length])
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            raise SharedMemoryError(f"payload checksum mismatch in {self.path}")
        return SharedSnapshot(
            version=version,
            sim_step=int(fields[10 + active]),
            sim_time_s=float(fields[12 + active]),
            writer_pid=int(fields[14]),
            payload=payload,
        )

    def snapshot(self) -> SharedSnapshot | None:
        fcntl.flock(self._fd, fcntl.LOCK_SH)
        try:
            return self._snapshot_unlocked()
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def snapshot_json(self) -> tuple[SharedSnapshot, dict[str, Any]] | None:
        snapshot = self.snapshot()
        return None if snapshot is None else (snapshot, snapshot.json())

    def _futex_wait(self, expected: int, timeout_s: float | None) -> None:
        timeout_pointer: Any = 0
        timeout = None
        if timeout_s is not None:
            seconds = max(float(timeout_s), 0.0)
            timeout = _Timespec(int(seconds), int((seconds % 1.0) * 1e9))
            timeout_pointer = ctypes.byref(timeout)
        result = _LIBC.syscall(
            _SYS_FUTEX,
            ctypes.byref(self._futex_ref),
            _FUTEX_WAIT,
            ctypes.c_uint32(expected),
            timeout_pointer,
            0,
            0,
        )
        if result == -1 and ctypes.get_errno() not in {
            errno.EAGAIN,
            errno.EINTR,
            errno.ETIMEDOUT,
        }:
            raise OSError(ctypes.get_errno(), "futex wait failed")

    def _futex_wake(self) -> None:
        result = _LIBC.syscall(
            _SYS_FUTEX,
            ctypes.byref(self._futex_ref),
            _FUTEX_WAKE,
            _INT_MAX,
            0,
            0,
            0,
        )
        if result == -1:
            raise OSError(ctypes.get_errno(), "futex wake failed")

    def wait_for_newer(
        self,
        after_version: int,
        *,
        stop: Callable[[], bool] | None = None,
        guard_timeout_s: float | None = None,
    ) -> SharedSnapshot | None:
        """Wait for a newer commit; supervisor ``poke`` handles shutdown."""

        while stop is None or not stop():
            snapshot = self.snapshot()
            if snapshot is not None and snapshot.version > after_version:
                return snapshot
            observed_futex = self.notification_sequence
            # Close the check-to-wait race. FUTEX_WAIT returns EAGAIN if a
            # producer changed this value before the syscall sleeps.
            snapshot = self.snapshot()
            if snapshot is not None and snapshot.version > after_version:
                return snapshot
            self._futex_wait(observed_futex, guard_timeout_s)
        return None

    def wait_for_notification(
        self,
        after_sequence: int,
        *,
        stop: Callable[[], bool] | None = None,
    ) -> int | None:
        """Wait for a futex notification without requiring a payload commit."""

        while stop is None or not stop():
            observed = self.notification_sequence
            if observed != after_sequence:
                return observed
            self._futex_wait(observed, None)
        return None

    def poke(self) -> None:
        """Wake waiters without publishing a new version (shutdown only)."""

        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            fields = list(self._unpack_header())
            fields[9] = (int(fields[9]) + 1) & 0xFFFFFFFF
            self._map[: _HEADER.size] = _HEADER.pack(*fields)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._futex_wake()

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            del self._futex_ref
            self._map.close()
            self._map = None
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "SharedJsonBlock":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SharedJsonRing:
    """Single-writer fixed-capacity stream of JSON boundary records.

    Producers never wait for consumers. Readers use their own sequence cursor,
    so retained records are consumed in order instead of being collapsed into
    the newest state as they are in :class:`SharedJsonBlock`.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
        size = os.fstat(self._fd).st_size
        if size < RING_HEADER_SIZE + RING_SLOT_HEADER_SIZE + 1:
            os.close(self._fd)
            raise SharedMemoryError(f"shared ring is too small: {self.path}")
        self._map = mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE)
        fields = self._unpack_header()
        if (
            fields[0] != RING_MAGIC
            or fields[1] != RING_ABI_VERSION
            or fields[4] != RING_SLOT_HEADER_SIZE
        ):
            self.close()
            raise SharedMemoryError(f"shared ring ABI mismatch: {self.path}")
        self.slot_capacity = int(fields[2])
        self.slot_count = int(fields[3])
        self.slot_stride = RING_SLOT_HEADER_SIZE + self.slot_capacity
        if size != RING_HEADER_SIZE + self.slot_count * self.slot_stride:
            self.close()
            raise SharedMemoryError(f"shared ring size mismatch: {self.path}")
        self._futex_ref = ctypes.c_uint32.from_buffer(
            self._map, _RING_FUTEX_OFFSET
        )

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        slot_capacity: int,
        slot_count: int = DEFAULT_RING_SLOTS,
    ) -> "SharedJsonRing":
        target = Path(path)
        if slot_capacity < 1024:
            raise ValueError("shared ring slot capacity must be at least 1024")
        if slot_count < 2:
            raise ValueError("shared ring must have at least two slots")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(
            target,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        size = RING_HEADER_SIZE + slot_count * (
            RING_SLOT_HEADER_SIZE + slot_capacity
        )
        try:
            os.ftruncate(fd, size)
            mapping = mmap.mmap(fd, size)
            try:
                mapping[: _RING_HEADER.size] = _RING_HEADER.pack(
                    RING_MAGIC,
                    RING_ABI_VERSION,
                    int(slot_capacity),
                    int(slot_count),
                    RING_SLOT_HEADER_SIZE,
                    0,
                    0,
                    0,
                )
                mapping.flush()
            finally:
                mapping.close()
        finally:
            os.close(fd)
        return cls(target)

    def _unpack_header(self) -> tuple[Any, ...]:
        return _RING_HEADER.unpack_from(self._map, 0)

    @property
    def version(self) -> int:
        fcntl.flock(self._fd, fcntl.LOCK_SH)
        try:
            return int(self._unpack_header()[5])
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    @property
    def notification_sequence(self) -> int:
        return int(self._futex_ref.value)

    def publish_bytes(
        self,
        payload: bytes,
        *,
        sim_step: int = 0,
        sim_time_s: float = 0.0,
    ) -> int:
        data = bytes(payload)
        if len(data) > self.slot_capacity:
            raise SharedMemoryError(
                f"payload {len(data)} exceeds ring slot capacity "
                f"{self.slot_capacity} in {self.path}"
            )
        if sim_step < 0:
            raise ValueError("sim_step must be non-negative")
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            fields = list(self._unpack_header())
            sequence = int(fields[5]) + 1
            index = (sequence - 1) % self.slot_count
            start = RING_HEADER_SIZE + index * self.slot_stride
            payload_start = start + RING_SLOT_HEADER_SIZE
            self._map[payload_start : payload_start + len(data)] = data
            self._map[start : start + RING_SLOT_HEADER_SIZE] = (
                _RING_SLOT_HEADER.pack(
                    sequence,
                    int(sim_step),
                    float(sim_time_s),
                    len(data),
                    zlib.crc32(data) & 0xFFFFFFFF,
                )
            )
            fields[5] = sequence
            fields[6] = (int(fields[6]) + 1) & 0xFFFFFFFF
            fields[7] = os.getpid()
            self._map[: _RING_HEADER.size] = _RING_HEADER.pack(*fields)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._futex_wake()
        return sequence

    def publish_json(
        self,
        payload: dict[str, Any],
        *,
        sim_step: int = 0,
        sim_time_s: float = 0.0,
    ) -> int:
        return self.publish_bytes(
            json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
            sim_step=sim_step,
            sim_time_s=sim_time_s,
        )

    def read_after(
        self, after_sequence: int, *, max_items: int | None = None
    ) -> list[SharedRingSnapshot]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        fcntl.flock(self._fd, fcntl.LOCK_SH)
        try:
            write_sequence = int(self._unpack_header()[5])
            if write_sequence <= after_sequence:
                return []
            oldest = max(1, write_sequence - self.slot_count + 1)
            if after_sequence < oldest - 1:
                raise SharedRingOverflowError(
                    f"shared ring overflow in {self.path}: consumer after "
                    f"{after_sequence}, oldest retained {oldest}, newest "
                    f"{write_sequence}"
                )
            first = max(after_sequence + 1, oldest)
            last = write_sequence
            if max_items is not None:
                if max_items < 1:
                    raise ValueError("max_items must be positive")
                last = min(last, first + max_items - 1)
            output: list[SharedRingSnapshot] = []
            for sequence in range(first, last + 1):
                index = (sequence - 1) % self.slot_count
                start = RING_HEADER_SIZE + index * self.slot_stride
                stored, sim_step, sim_time_s, length, checksum = (
                    _RING_SLOT_HEADER.unpack_from(self._map, start)
                )
                if stored != sequence or length > self.slot_capacity:
                    raise SharedMemoryError(
                        f"invalid shared ring slot in {self.path}"
                    )
                payload_start = start + RING_SLOT_HEADER_SIZE
                payload = bytes(
                    self._map[payload_start : payload_start + length]
                )
                if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
                    raise SharedMemoryError(
                        f"shared ring checksum mismatch in {self.path}"
                    )
                output.append(
                    SharedRingSnapshot(
                        sequence=int(stored),
                        sim_step=int(sim_step),
                        sim_time_s=float(sim_time_s),
                        payload=payload,
                    )
                )
            return output
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def wait_for_newer(
        self,
        after_sequence: int,
        *,
        stop: Callable[[], bool] | None = None,
    ) -> SharedRingSnapshot | None:
        while True:
            records = self.read_after(after_sequence, max_items=1)
            if records:
                return records[0]
            if stop is not None and stop():
                return None
            observed_futex = self.notification_sequence
            records = self.read_after(after_sequence, max_items=1)
            if records:
                return records[0]
            if stop is not None and stop():
                return None
            self._futex_wait(observed_futex)

    def _futex_wait(self, expected: int) -> None:
        result = _LIBC.syscall(
            _SYS_FUTEX,
            ctypes.byref(self._futex_ref),
            _FUTEX_WAIT,
            ctypes.c_uint32(expected),
            0,
            0,
            0,
        )
        if result == -1 and ctypes.get_errno() not in {
            errno.EAGAIN,
            errno.EINTR,
        }:
            raise OSError(ctypes.get_errno(), "ring futex wait failed")

    def _futex_wake(self) -> None:
        result = _LIBC.syscall(
            _SYS_FUTEX,
            ctypes.byref(self._futex_ref),
            _FUTEX_WAKE,
            _INT_MAX,
            0,
            0,
            0,
        )
        if result == -1:
            raise OSError(ctypes.get_errno(), "ring futex wake failed")

    def poke(self) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            fields = list(self._unpack_header())
            fields[6] = (int(fields[6]) + 1) & 0xFFFFFFFF
            self._map[: _RING_HEADER.size] = _RING_HEADER.pack(*fields)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._futex_wake()

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            if hasattr(self, "_futex_ref"):
                del self._futex_ref
            self._map.close()
            self._map = None
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "SharedJsonRing":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_layout(
    layout: SharedMemoryLayout,
    *,
    capacities: dict[str, int] | None = None,
    ring_slots: dict[str, int] | None = None,
) -> dict[str, SharedJsonBlock | SharedJsonRing]:
    configured = capacities or {}
    configured_ring_slots = ring_slots or {}
    layout.root.mkdir(mode=0o700, parents=False, exist_ok=False)
    output: dict[str, SharedJsonBlock | SharedJsonRing] = {}
    try:
        for name in layout.BLOCKS:
            if name in layout.RING_BLOCKS:
                output[name] = SharedJsonRing.create(
                    layout.path(name),
                    slot_capacity=int(configured.get(name, DEFAULT_CAPACITY)),
                    slot_count=int(
                        configured_ring_slots.get(name, DEFAULT_RING_SLOTS)
                    ),
                )
            else:
                output[name] = SharedJsonBlock.create(
                    layout.path(name),
                    capacity=int(configured.get(name, DEFAULT_CAPACITY)),
                )
    except Exception:
        for block in output.values():
            block.close()
        for path in layout.root.iterdir():
            path.unlink(missing_ok=True)
        layout.root.rmdir()
        raise
    return output


def open_blocks(
    layout: SharedMemoryLayout, names: Iterable[str]
) -> dict[str, SharedJsonBlock]:
    if any(name in layout.RING_BLOCKS for name in names):
        raise ValueError("open_blocks cannot open ring blocks")
    return {name: SharedJsonBlock(layout.path(name)) for name in names}


def open_rings(
    layout: SharedMemoryLayout, names: Iterable[str]
) -> dict[str, SharedJsonRing]:
    if any(name not in layout.RING_BLOCKS for name in names):
        raise ValueError("open_rings accepts only ring blocks")
    return {name: SharedJsonRing(layout.path(name)) for name in names}


def snapshot_consistent(
    blocks: dict[str, SharedJsonBlock],
    *,
    attempts: int = 100,
) -> dict[str, tuple[SharedSnapshot, dict[str, Any]]]:
    """Atomically copy multiple blocks under deterministic shared locks."""

    if not blocks:
        raise ValueError("at least one shared block is required")
    del attempts  # retained for source compatibility with early prototypes
    ordered = sorted(blocks.items(), key=lambda item: str(item[1].path))
    for _, block in ordered:
        fcntl.flock(block._fd, fcntl.LOCK_SH)
    try:
        values: dict[str, tuple[SharedSnapshot, dict[str, Any]]] = {}
        for name, block in blocks.items():
            snapshot = block._snapshot_unlocked()
            if snapshot is None:
                raise SharedMemoryError(
                    "one or more shared state blocks are not ready"
                )
            values[name] = (snapshot, snapshot.json())
        return values
    finally:
        for _, block in reversed(ordered):
            fcntl.flock(block._fd, fcntl.LOCK_UN)


def shutdown_requested(block: SharedJsonBlock) -> bool:
    value = block.snapshot_json()
    return bool(value is not None and value[1].get("shutdown", False))


def wait_until_ready(
    block: SharedJsonBlock,
    *,
    stop: Callable[[], bool],
) -> SharedSnapshot | None:
    current = block.snapshot()
    if current is not None:
        return current
    return block.wait_for_newer(0, stop=stop)


def wall_time_ns() -> int:
    return time.time_ns()
