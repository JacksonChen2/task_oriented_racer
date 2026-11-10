"""Four-process supervisor entry for the separate LLM-prior CRPO variant.

The lifecycle implementation is reused from the baseline supervisor.  Only
the RL child module is replaced; invoking the baseline supervisor still starts
the unchanged baseline trainer.
"""

from __future__ import annotations

from typing import Any

from . import process_supervisor as baseline


BASELINE_TRAIN_MODULE = "agentic_crpo.train_crpo"
LLM_PRIOR_TRAIN_MODULE = "agentic_crpo.train_llm_prior_crpo"


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
    replaced[matches[0]] = LLM_PRIOR_TRAIN_MODULE
    return replaced


def run(args: Any) -> int:
    original_spawn = baseline._spawn

    def spawn_llm_prior(
        name: str,
        command: list[str],
        output_dir: Any,
        environment: dict[str, str],
    ) -> Any:
        if name == "rl":
            # Mutate the run-local command list as well as the actual spawn
            # input so process_manifest.json records the truthful new module.
            command[:] = _replace_rl_train_module(command)
        return original_spawn(
            name, command, output_dir, environment
        )

    # baseline.run resolves _spawn from its own module globals. Keep the
    # substitution scoped to this call and restore it even on interruption.
    baseline._spawn = spawn_llm_prior
    try:
        return baseline.run(args)
    finally:
        baseline._spawn = original_spawn


def main() -> None:
    raise SystemExit(run(baseline.parse_args()))


if __name__ == "__main__":
    main()
