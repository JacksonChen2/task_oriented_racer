"""Resolve a manual or deterministic-baseline calibrated task constraint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConstraintCalibration:
    gamma_task: float
    eta: float
    calibration_mode: str
    rho: float
    source: str
    l_full: float | None = None
    l_dist: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _final_losses(path: str | Path) -> list[float]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"baseline result must be a JSON object: {source}")
    if "mean_final_task_loss" in payload:
        return [float(payload["mean_final_task_loss"])]
    records = payload.get("episodes", payload.get("episode_statistics", []))
    values = []
    for item in records if isinstance(records, list) else []:
        if not isinstance(item, dict):
            continue
        value = item.get("L_task_final", item.get("final_task_loss"))
        if value is not None:
            values.append(float(value))
    if not values:
        raise ValueError(f"no final task losses in baseline result: {source}")
    return values


def _mean_from_files(paths: Any, label: str) -> float:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"constraint.{label}_results must list baseline JSON files")
    values = [value for path in paths for value in _final_losses(path)]
    result = float(np.mean(values))
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} mean task loss must be in [0, 1], got {result}")
    return result


def resolve_constraint(config: dict[str, Any]) -> ConstraintCalibration:
    section = config.get("constraint", {})
    eta = float(section.get("eta", config.get("crpo", {}).get("eta", 0.01)))
    rho = float(section.get("rho", 0.25))
    if eta < 0.0 or not 0.0 <= rho <= 1.0:
        raise ValueError("constraint eta must be non-negative and rho must be in [0, 1]")

    manual = section.get("gamma_task")
    if manual is not None:
        gamma = float(manual)
        source = "manual"
        calibration = ConstraintCalibration(
            gamma_task=gamma,
            eta=eta,
            calibration_mode=str(section.get("calibration_mode", "baseline_gap")),
            rho=rho,
            source=source,
        )
    else:
        mode = str(section.get("calibration_mode", "baseline_gap"))
        if mode != "baseline_gap":
            raise ValueError(f"unsupported constraint calibration_mode: {mode}")
        l_full = _mean_from_files(section.get("full_bs_results"), "full_bs")
        l_dist = _mean_from_files(
            section.get("distributed_only_results"), "distributed_only"
        )
        if l_full > l_dist:
            raise ValueError(
                "baseline calibration expects L_full <= L_dist; check paired runs"
            )
        gamma = l_full + rho * (l_dist - l_full)
        calibration = ConstraintCalibration(
            gamma_task=gamma,
            eta=eta,
            calibration_mode=mode,
            rho=rho,
            source="baseline_gap",
            l_full=l_full,
            l_dist=l_dist,
        )
    if not np.isfinite(calibration.gamma_task) or not 0.0 <= calibration.gamma_task <= 1.0:
        raise ValueError("Gamma_task must be finite and in [0, 1]")
    return calibration


def print_calibration(value: ConstraintCalibration) -> None:
    print("Constraint calibration:")
    print(f"  source={value.source}")
    if value.l_full is not None:
        print(f"  L_full={value.l_full:.6f}")
        print(f"  L_dist={value.l_dist:.6f}")
    print(f"  rho={value.rho:.6f}")
    print(f"  Gamma_task={value.gamma_task:.6f}")
    print(f"  eta={value.eta:.6f}")
