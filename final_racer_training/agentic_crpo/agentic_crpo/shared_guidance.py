"""Read-only guidance facade used by the independent RL process."""

from __future__ import annotations

from typing import Any

from .schemas import GlobalGuidance
from .shared_ipc import (
    SharedMemoryLayout,
    open_blocks,
    shutdown_requested,
)


class SharedGuidanceManager:
    """Expose the latest completed LLM result without starting a model/thread."""

    externally_driven = True

    def __init__(self, shared_memory_root: str, n_uavs: int) -> None:
        self.n_uavs = int(n_uavs)
        layout = SharedMemoryLayout.from_value(shared_memory_root)
        self.blocks = open_blocks(layout, ("guidance", "shutdown"))
        self._guidance = GlobalGuidance.neutral(self.n_uavs)
        self.epoch = 0
        self.failures = 0
        self.latency_s = 0.0
        self.source_physical_version = 0
        self.source_communication_version = 0
        self.source_sim_step = 0
        self.source_step_id = 0

    def _refresh(self) -> None:
        value = self.blocks["guidance"].snapshot_json()
        if value is None or value[0].version <= self.epoch:
            return
        snapshot, payload = value
        guidance_id = int(payload.get("guidance_id", snapshot.version))
        if guidance_id != snapshot.version:
            raise ValueError(
                "guidance_id does not match its committed shared version"
            )
        self._guidance = GlobalGuidance.validated(
            payload["task_dependency"],
            payload["semantic_importance"],
            self.n_uavs,
        )
        self.epoch = guidance_id
        self.failures = int(payload.get("parse_failures", self.failures))
        self.latency_s = float(payload.get("inference_latency_s", 0.0))
        self.source_physical_version = int(
            payload["source_physical_version"]
        )
        self.source_communication_version = int(
            payload["source_communication_version"]
        )
        self.source_sim_step = int(payload["source_sim_step"])
        self.source_step_id = int(
            payload.get("source_step_id", self.source_sim_step)
        )

    @property
    def current(self) -> GlobalGuidance:
        self._refresh()
        return GlobalGuidance(
            self._guidance.task_dependency.copy(),
            self._guidance.semantic_importance.copy(),
        )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def single_gpu_pause_count(self) -> int:
        return 0

    @property
    def single_gpu_pause_ack_wait_s(self) -> float:
        return 0.0

    def request_update(self, state: dict[str, Any]) -> bool:
        del state
        return False

    def wait_until_ready(self) -> None:
        if self.blocks["guidance"].snapshot() is not None:
            self._refresh()
            return
        value = self.blocks["guidance"].wait_for_newer(
            0,
            stop=lambda: shutdown_requested(self.blocks["shutdown"]),
        )
        if value is None:
            raise TimeoutError("episode ended before initial Qwen guidance")
        self._refresh()

    def close(self) -> None:
        for block in self.blocks.values():
            block.close()
