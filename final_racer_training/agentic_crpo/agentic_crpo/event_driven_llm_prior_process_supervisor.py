"""Four-process supervisor for the event-driven one-shot experiment."""

from __future__ import annotations

import os
from typing import Any

from . import process_supervisor as baseline


BASELINE_TRAIN_MODULE = "agentic_crpo.train_crpo"
EVENT_TRAIN_MODULE = "agentic_crpo.train_event_driven_llm_prior_crpo"


def _replace_rl_train_module(command: list[str]) -> list[str]:
    replaced = list(command)
    matches = [
        index
        for index, value in enumerate(replaced)
        if value == BASELINE_TRAIN_MODULE
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "baseline supervisor RL command no longer contains exactly one "
            f"{BASELINE_TRAIN_MODULE!r} entry"
        )
    replaced[matches[0]] = EVENT_TRAIN_MODULE
    return replaced


def run(args: Any) -> int:
    original_spawn = baseline._spawn
    original_runtime_health = baseline.RuntimeHealth

    def runtime_health_with_scene_startup_margin(started: float) -> Any:
        startup_timeout = float(
            os.environ.get("RACER_EVENT_STARTUP_TIMEOUT_S", "900")
        )
        if startup_timeout <= 0.0:
            raise ValueError("RACER_EVENT_STARTUP_TIMEOUT_S must be positive")
        return original_runtime_health(started, startup_timeout=startup_timeout)

    def spawn_event_variant(
        name: str,
        command: list[str],
        output_dir: Any,
        environment: dict[str, str],
    ) -> Any:
        if name == "rl":
            command[:] = _replace_rl_train_module(command)
        environment["RACER_RL_BS_EVENT_DRIVEN_ONE_SHOT"] = "true"
        return original_spawn(name, command, output_dir, environment)

    baseline._spawn = spawn_event_variant
    baseline.RuntimeHealth = runtime_health_with_scene_startup_margin
    try:
        return baseline.run(args)
    finally:
        baseline._spawn = original_spawn
        baseline.RuntimeHealth = original_runtime_health


def main() -> None:
    raise SystemExit(run(baseline.parse_args()))


if __name__ == "__main__":
    main()
