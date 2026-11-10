"""Opt-in episode-level no-link suppression, followed by a hard mask."""

from __future__ import annotations

import math
from typing import Any


def validate_no_link_options(actor: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration and return a normalized, default-off schedule."""
    hard_mask = actor.get("mask_no_link_actions", False)
    if not isinstance(hard_mask, bool):
        raise ValueError("actor.mask_no_link_actions must be a boolean")
    # Runtime penalties come only from the explicitly enabled schedule.
    if actor.get("no_link_logit_penalty", 0.0) != 0.0:
        raise ValueError("Use no_link_logit_schedule; configured no_link_logit_penalty must be zero")
    raw = actor.get("no_link_logit_schedule", {})
    if not isinstance(raw, dict):
        raise ValueError("actor.no_link_logit_schedule must be a mapping")
    if set(raw) - {"enabled", "total_episodes", "max_penalty"}:
        raise ValueError("Unknown no_link_logit_schedule option")
    schedule = {"enabled": False, "total_episodes": 50, "max_penalty": 12.0, **raw}
    if not isinstance(schedule["enabled"], bool):
        raise ValueError("no_link_logit_schedule.enabled must be a boolean")
    total = schedule["total_episodes"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 3:
        raise ValueError("no_link_logit_schedule.total_episodes must be an integer >= 3")
    maximum = schedule["max_penalty"]
    if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum <= 0.0):
        raise ValueError("no_link_logit_schedule.max_penalty must be finite and positive")
    schedule["max_penalty"] = float(maximum)
    if hard_mask and schedule["enabled"]:
        raise ValueError("Choose constant hard masking OR the schedule: set mask_no_link_actions=false for scheduling")
    return schedule


def no_link_mask_for_episode(
    actor: dict[str, Any], episode_number: int | None
) -> tuple[bool, float]:
    """Return effective (hard_mask, logit_penalty), without changing config.

    With scheduling enabled, episode 1 has no suppression, episode N-1
    reaches max_penalty, and episode N onward uses the hard mask. The
    campaign's requested run length does not change this schedule.
    """
    schedule = validate_no_link_options(actor)
    if not schedule["enabled"]:
        return actor.get("mask_no_link_actions", False), 0.0
    if (isinstance(episode_number, bool) or not isinstance(episode_number, int)
            or episode_number < 1):
        raise ValueError("Enabled no-link schedule requires a positive integer episode number")
    total = schedule["total_episodes"]
    if episode_number >= total:
        return True, 0.0
    return False, schedule["max_penalty"] * (episode_number - 1) / (total - 2)
