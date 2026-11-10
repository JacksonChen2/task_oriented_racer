"""Four-process supervisor adapter for the fixed-policy sanity evaluations."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from . import process_supervisor as baseline


TRAIN_MODULE = "agentic_crpo.train_crpo"
EVAL_MODULE = "agentic_crpo.fixed_policy_eval"
POLICIES = {"episode_01": "zero", "episode_02": "one", "episode_03": "bernoulli_05"}


def run(args: Any) -> int:
    episode = Path(args.output_dir).name
    if episode not in POLICIES:
        raise ValueError(f"fixed-policy campaign has no policy for {episode}")
    policy = POLICIES[episode]
    original_spawn = baseline._spawn

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
            command.extend(("--policy", policy, "--action-seed", "42"))
        elif name == "llm":
            command[:] = [
                sys.executable,
                "-m",
                "agentic_crpo.idle_llm_worker",
                "--shared-memory-root",
                environment["RACER_SHARED_MEMORY_ROOT"],
            ]
        environment["RACER_RL_BS_EVENT_DRIVEN_ONE_SHOT"] = "true"
        return original_spawn(name, command, output_dir, environment)

    baseline._spawn = spawn_fixed
    try:
        return baseline.run(args)
    finally:
        baseline._spawn = original_spawn


def main() -> None:
    raise SystemExit(run(baseline.parse_args()))


if __name__ == "__main__":
    main()
