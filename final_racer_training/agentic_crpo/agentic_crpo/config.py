"""Strict-enough YAML loading while preserving experiment metadata."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import yaml

from .no_link_schedule import validate_no_link_options


def set_action_mask_mode(
    config: dict[str, Any], enabled: bool, *, schedule_enabled: bool | None = None
) -> None:
    """Set the explicit mask option and keep portable experiment labels accurate."""
    if not isinstance(enabled, bool):
        raise ValueError("actor.mask_no_link_actions must be a boolean")
    actor = copy.deepcopy(config.get("actor", {}))
    actor["mask_no_link_actions"] = enabled
    if schedule_enabled is not None:
        actor.setdefault("no_link_logit_schedule", {})["enabled"] = schedule_enabled
    actor["no_link_logit_schedule"] = validate_no_link_options(actor)
    config["actor"] = actor
    variants = {
        "unmasked": "portable_phaoi_z_source_owned_bs_ofdma_u2u_csma_unmasked_v1",
        "masked": "portable_phaoi_z_source_owned_bs_ofdma_u2u_csma_masked_v1",
        "scheduled": "portable_phaoi_z_source_owned_bs_ofdma_u2u_csma_scheduled_v1",
    }
    mode = "scheduled" if actor["no_link_logit_schedule"]["enabled"] else ("masked" if enabled else "unmasked")
    algorithm = config.get("algorithm", {})
    if algorithm.get("variant") in variants.values():
        algorithm["variant"] = variants[mode]


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("CRPO config root must be a mapping")
    config = copy.deepcopy(value)
    config["_config_path"] = str(source)
    local_reference = os.environ.get("RACER_QWEN8_PERFECT_RESULT")
    if local_reference:
        config.setdefault("environment", {}).setdefault("backend", {})[
            "perfect_reference_result"
        ] = local_reference
    for section, key in ((config.get("qwen", {}), "model_path"),
                         (config.get("environment", {}).get("backend", {}), "perfect_reference_result"),
                         (config.get("training", {}), "output_dir")):
        if section.get(key):
            value = os.path.expandvars(str(section[key]))
            if "$" in value:
                raise ValueError(f"Unresolved environment variable in {key}: {value}")
            target = Path(value).expanduser()
            section[key] = str((source.parent / target).resolve() if not target.is_absolute() else target)
    actor = config.setdefault("actor", {})
    if not isinstance(actor, dict):
        raise ValueError("actor config must be a mapping")
    set_action_mask_mode(config, actor.get("mask_no_link_actions", False))
    config["actor"]["no_link_logit_penalty"] = 0.0
    required = (
        "seed",
        "n_uavs",
        "qwen",
        "environment",
        "task_loss",
        "constraint",
        "crpo",
        "training",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    if int(config["n_uavs"]) < 1:
        raise ValueError("n_uavs must be positive")
    reward_type = str(
        config.get("reward", {}).get("type", "negative_bs_resource")
    )
    if reward_type not in {
        "negative_bs_resource",
        "negative_total_prb",
        "coverage",
        "coverage_delta",
    }:
        raise ValueError(f"unsupported reward.type: {reward_type}")
    no_link_prb_per_action = float(
        config.get("reward", {}).get("no_link_prb_per_action", 0.0)
    )
    if (
        not math.isfinite(no_link_prb_per_action)
        or no_link_prb_per_action < 0.0
    ):
        raise ValueError(
            "reward.no_link_prb_per_action must be finite and non-negative"
        )
    constraint = config["constraint"]
    constraint_type = str(constraint.get("type", "task_loss"))
    if constraint_type not in {"task_loss", "relay_fanout"}:
        raise ValueError(f"unsupported constraint.type: {constraint_type}")
    if constraint_type == "relay_fanout":
        limit = int(constraint.get("max_relay_recipients_per_uav", -1))
        if not 0 <= limit <= int(config["n_uavs"]) - 1:
            raise ValueError(
                "constraint.max_relay_recipients_per_uav must be between "
                "0 and n_uavs - 1"
            )
    episode_duration_s = float(constraint.get("episode_duration_s", 300.0))
    if not math.isfinite(episode_duration_s) or episode_duration_s <= 0.0:
        raise ValueError(
            "constraint.episode_duration_s must be finite and positive"
        )
    estimator = str(config["crpo"].get("constraint_estimator", ""))
    if estimator == "time_weighted_mean" and abs(
        float(config["crpo"].get("gamma_cost", 1.0)) - 1.0
    ) > 1.0e-12:
        raise ValueError(
            "crpo.gamma_cost must be 1 for the time_weighted_mean constraint"
        )
    return config


def require_reference_input(config: dict[str, Any]) -> Path:
    """Resolve the user-supplied input before launching real simulation."""
    value = config.get("environment", {}).get("backend", {}).get("perfect_reference_result")
    if not value:
        raise FileNotFoundError(
            "Set RACER_QWEN8_PERFECT_RESULT in env.local.sh to your local reference JSON; "
            "reference data is not included in this repository"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local reference input does not exist: {path}")
    return path


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
