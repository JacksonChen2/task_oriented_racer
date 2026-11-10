"""Training entry for the opt-in event-driven one-shot BS experiment."""

from __future__ import annotations

from . import train_llm_prior_crpo as fixed_trainer
from .event_driven_crpo import EventDrivenOneShotLLMPriorCRPOPPO


def main() -> None:
    configured_algorithm = fixed_trainer.LLMPriorCRPOPPO
    fixed_trainer.LLMPriorCRPOPPO = EventDrivenOneShotLLMPriorCRPOPPO
    try:
        fixed_trainer.main()
    finally:
        fixed_trainer.LLMPriorCRPOPPO = configured_algorithm


if __name__ == "__main__":
    main()
