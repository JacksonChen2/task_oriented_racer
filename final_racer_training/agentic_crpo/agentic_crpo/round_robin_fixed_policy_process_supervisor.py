"""Four-process supervisor for the fixed-clock row round-robin policy."""

from __future__ import annotations

import os
import sys
from typing import Any

from . import process_supervisor as baseline


TRAIN_MODULE = "agentic_crpo.train_crpo"
EVAL_MODULE = "agentic_crpo.fixed_policy_eval"


def run(args: Any) -> int:
    original_spawn = baseline._spawn
    original_runtime_health = baseline.RuntimeHealth
    decision_period_ms = float(
        os.environ.get("RACER_RR_DECISION_PERIOD_MS", "100.0")
    )
    communication_slot_ms = 20.0
    slots_per_decision = decision_period_ms / communication_slot_ms
    if (
        decision_period_ms <= 0.0
        or abs(slots_per_decision - round(slots_per_decision)) > 1.0e-9
    ):
        raise ValueError(
            "RACER_RR_DECISION_PERIOD_MS must be a positive multiple of "
            "the 20 ms communication slot"
        )

    def runtime_health_with_scene_startup_margin(started: float) -> Any:
        # A loaded 10-UAV warehouse scene can need more than the baseline
        # three-minute startup window before its first physical publication.
        # Keep this local to the fixed-policy experiment and configurable.
        startup_timeout = float(os.environ.get("RACER_RR_STARTUP_TIMEOUT_S", "900"))
        return original_runtime_health(started, startup_timeout=startup_timeout)

    def spawn_fixed(
        name: str,
        command: list[str],
        output_dir: Any,
        environment: dict[str, str],
    ) -> Any:
        if name == "rl":
            matches = [index for index, value in enumerate(command) if value == TRAIN_MODULE]
            if len(matches) != 1:
                raise RuntimeError("baseline RL module replacement is ambiguous")
            command[matches[0]] = EVAL_MODULE
            command.extend(("--policy", "round_robin_row", "--action-seed", "42"))
        elif name == "llm":
            command[:] = [
                sys.executable,
                "-m",
                "agentic_crpo.idle_llm_worker",
                "--shared-memory-root",
                environment["RACER_SHARED_MEMORY_ROOT"],
            ]
        # Keep fixed-clock shared-memory scheduling. The runtime publishes one
        # transition boundary per configured decision period.
        environment["RACER_RL_BS_EVENT_DRIVEN_ONE_SHOT"] = "false"
        environment["RACER_RL_BS_DECISION_PERIOD_MS"] = format(
            decision_period_ms, ".17g"
        )
        return original_spawn(name, command, output_dir, environment)

    baseline._spawn = spawn_fixed
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
