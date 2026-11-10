"""Supervisor-compatible idle worker for evaluations that do not use Qwen."""

from __future__ import annotations

import argparse
import os

from .shared_ipc import SharedMemoryLayout, open_blocks, shutdown_requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-memory-root", required=True)
    args = parser.parse_args()
    blocks = open_blocks(
        SharedMemoryLayout.from_value(args.shared_memory_root),
        ("shutdown", "status_llm"),
    )
    try:
        blocks["status_llm"].publish_json(
            {"ready": True, "idle": True, "process": "llm", "pid": os.getpid()}
        )
        notification = blocks["shutdown"].notification_sequence
        while not shutdown_requested(blocks["shutdown"]):
            updated = blocks["shutdown"].wait_for_notification(
                notification,
                stop=lambda: shutdown_requested(blocks["shutdown"]),
            )
            if updated is not None:
                notification = updated
    finally:
        for block in blocks.values():
            block.close()


if __name__ == "__main__":
    main()
