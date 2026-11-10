"""Assemble split-process logs into the familiar final_racer result shape."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def _last_prefixed_json(text: str, marker: str) -> dict[str, Any]:
    matches = re.findall(re.escape(marker) + r"\s+(\{[^\n]+\})", text)
    if not matches:
        return {}
    value = json.loads(matches[-1])
    return value if isinstance(value, dict) else {}


def assemble(output_dir: Path, environment: dict[str, str]) -> Path | None:
    isaac_path = output_dir / "process_isaac.log"
    racer_path = output_dir / "process_racer.log"
    if not isaac_path.is_file() or not racer_path.is_file():
        return None
    isaac = isaac_path.read_text(errors="replace")
    racer = racer_path.read_text(errors="replace")
    metrics = _last_prefixed_json(isaac, "RACER_3D_ISAAC_RESULT")
    if not metrics:
        return None
    statistics = _last_prefixed_json(racer, "RACER_SIONNA_STATS")
    channel_profile = _last_prefixed_json(
        racer, "RACER_SIONNA_CHANNEL_PROFILE"
    )
    topology = environment.get("RACER_NETWORK_TOPOLOGY", "bs_round_robin")
    result = {
        "algorithm": environment.get(
            "RACER_ALGORITHM_LABEL", "final_racer_four_process_crpo"
        ),
        "algorithm_variant": "four_process_shared_memory",
        "random_seed": int(environment.get("RACER_RANDOM_SEED", "42")),
        "drone_count": int(
            environment.get("RACER_FIDELITY_DRONE_COUNT", "10")
        ),
        "communication": {
            "mode": environment.get("RACER_COMMUNICATION_MODE", "sionna"),
            "network_topology": topology,
            "statistics": statistics,
            "channel_profile": channel_profile,
        },
        "metrics": metrics,
        "process_architecture": {
            "top_level_processes": ["isaac", "racer", "llm", "rl"],
            "transport": "versioned_double_buffer_shared_memory",
            "model_scheduling": "event_driven_latest_state_wins",
        },
    }
    target = output_dir / f"warehouse_full_{topology}_result.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
