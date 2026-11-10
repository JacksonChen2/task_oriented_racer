"""Slow Qwen state summaries and the fixed-dimensional actor observation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import GlobalGuidance, RacerSnapshot


NO_LINK_SNR_DB = -120.0
MIN_AVAILABLE_CHANNEL_VALUE = 1.0e-6


@dataclass(frozen=True)
class StateNormalization:
    position_min: tuple[float, float, float] = (-27.0, 0.6, 0.4)
    position_max: tuple[float, float, float] = (6.0, 30.6, 8.4)
    snr_min_db: float = -20.0
    snr_max_db: float = 40.0
    max_aoi_s: float = 30.0
    max_missing_bytes: float = 4_000_000.0
    max_queue_bytes: float = 4_000_000.0


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    return np.asarray(
        [matrix[i, j] for i in range(n) for j in range(n) if i != j],
        dtype=np.float32,
    )


def _unit_log(values: np.ndarray, maximum: float) -> np.ndarray:
    maximum = max(float(maximum), 1.0)
    return np.clip(np.log1p(np.maximum(values, 0.0)) / np.log1p(maximum), 0, 1)


def _normalize_channel_state(
    channel_snr_db: np.ndarray,
    normalization: StateNormalization,
) -> np.ndarray:
    """Normalize SNR while reserving exact zero for the no-link sentinel.

    The communication bridge writes ``-120 dB`` when no usable LinkQuality is
    present.  An available but very weak link may also fall below
    ``snr_min_db``; retaining a small positive floor keeps it distinguishable
    from no-link without changing the observation width or SNR scaling.
    """

    normalized = np.clip(
        (channel_snr_db - normalization.snr_min_db)
        / max(normalization.snr_max_db - normalization.snr_min_db, 1.0e-6),
        0.0,
        1.0,
    )
    available = channel_snr_db > NO_LINK_SNR_DB
    return np.where(
        available,
        np.maximum(normalized, MIN_AVAILABLE_CHANNEL_VALUE),
        0.0,
    ).astype(np.float32, copy=False)


def _compact_number(value: float, decimals: int = 3) -> str:
    """Format a finite number without token-wasting trailing zeroes."""

    rounded = round(float(value), decimals)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def _compact_directed_u2u_snr(mean_snr: np.ndarray, n_uavs: int) -> str:
    """Encode all directed U2U links with one-based UAV IDs."""

    return " ".join(
        f"{sender + 1}-{receiver + 1}:"
        f"{_compact_number(mean_snr[sender, receiver])}"
        for sender in range(n_uavs)
        for receiver in range(n_uavs)
        if sender != receiver
    )


def _compact_bs_snr(values: np.ndarray, n_uavs: int) -> str:
    """Encode a directional BS link vector using one-based UAV IDs."""

    return " ".join(
        f"{uav + 1}:{_compact_number(values[uav])}" for uav in range(n_uavs)
    )


class SmallStateBuilder:
    """Build a finite float32 actor observation.

    Layout ``phaoi_z`` uses exactly ``[P,H,AoI,z]`` (350 values for N=10).
    The default Qwen-guided variants use ``[P,H,AoI,L,Q,z]``. A small-model-only
    variant can set ``include_global_guidance=False`` to build
    ``[P,H,AoI,L,Q]`` instead; this removes Qwen's ``z`` output from the
    network input rather than retaining a constant neutral placeholder.
    """

    def __init__(
        self,
        n_uavs: int,
        normalization: StateNormalization,
        *,
        observation_layout: str = "full",
        include_global_guidance: bool = True,
        include_exploration_phase: bool = False,
    ) -> None:
        if observation_layout not in {"full", "phaoi_z"}:
            raise ValueError(f"unsupported observation layout: {observation_layout}")
        if observation_layout == "phaoi_z" and (
            not include_global_guidance or include_exploration_phase
        ):
            raise ValueError("phaoi_z requires guidance and excludes exploration phase")
        self.observation_layout = observation_layout
        self.n_uavs = n_uavs
        self.norm = normalization
        self.include_global_guidance = bool(include_global_guidance)
        self.include_exploration_phase = bool(include_exploration_phase)
        self.slices: dict[str, slice] = {}
        cursor = 0
        dimensions = {
            "positions": 3 * n_uavs,
            "channels": n_uavs * n_uavs + n_uavs,
            "aoi": n_uavs * n_uavs,
            "missing": n_uavs * n_uavs,
            "queues": n_uavs * n_uavs,
        }
        if self.observation_layout == "phaoi_z":
            del dimensions["missing"], dimensions["queues"]
        if self.include_global_guidance:
            dimensions["guidance"] = n_uavs * n_uavs + n_uavs
        if self.include_exploration_phase:
            dimensions["exploration_phase"] = 1
        for name, width in dimensions.items():
            self.slices[name] = slice(cursor, cursor + width)
            cursor += width
        self.observation_dim = cursor
        expected = 4 * n_uavs * n_uavs + 4 * n_uavs
        if self.observation_layout == "phaoi_z":
            expected -= 2 * n_uavs * n_uavs
        if self.include_global_guidance:
            expected += n_uavs * n_uavs + n_uavs
        if self.include_exploration_phase:
            expected += 1
        assert cursor == expected

    def build(
        self,
        snapshot: RacerSnapshot,
        guidance: GlobalGuidance,
    ) -> np.ndarray:
        snapshot.validate()
        n = self.n_uavs
        position_min = np.asarray(self.norm.position_min, np.float32)
        position_max = np.asarray(self.norm.position_max, np.float32)
        positions = np.clip(
            (snapshot.positions - position_min)
            / np.maximum(position_max - position_min, 1.0e-6),
            0.0,
            1.0,
        ).reshape(-1)

        snr = _normalize_channel_state(snapshot.channel_snr_db, self.norm)
        direct = _off_diagonal(snr[:n, :n])
        uplink = snr[:n, n]
        downlink = snr[n, :n]

        pair_aoi = _off_diagonal(snapshot.pair_aoi_s)
        aoi = np.concatenate((pair_aoi, snapshot.bs_aoi_s))
        aoi = np.clip(aoi / max(self.norm.max_aoi_s, 1.0e-6), 0.0, 1.0)

        pair_missing = _off_diagonal(snapshot.pair_missing_bytes)
        missing = _unit_log(
            np.concatenate((pair_missing, snapshot.bs_missing_bytes)),
            self.norm.max_missing_bytes,
        )
        relay_queues = _off_diagonal(snapshot.relay_queue_bytes)
        queues = _unit_log(
            np.concatenate((snapshot.uplink_queue_bytes, relay_queues)),
            self.norm.max_queue_bytes,
        )
        components = [
            positions,
            direct,
            uplink,
            downlink,
            aoi,
        ]
        if self.observation_layout == "full":
            components.extend((missing, queues))
        if self.include_global_guidance:
            components.append(
                np.concatenate(
                    (
                        guidance.task_dependency.reshape(-1),
                        guidance.semantic_importance,
                    )
                )
            )
        if self.include_exploration_phase:
            sim_time_s = float(snapshot.sim_time_s)
            if not np.isfinite(sim_time_s):
                raise ValueError("exploration time contains NaN or Inf")
            components.append(
                np.asarray([np.clip(sim_time_s / 300.0, 0.0, 1.0)], np.float32)
            )
        output = np.concatenate(components).astype(np.float32, copy=False)
        if output.shape != (self.observation_dim,):
            raise RuntimeError(
                f"internal observation shape {output.shape} != "
                f"({self.observation_dim},)"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("small-model observation contains NaN or Inf")
        return output


class LargeStateBuilder:
    """Build a compact LLM payload from BS-observable state only.

    Detailed channel matrices and pairwise AoI remain in ``RacerSnapshot`` for
    the small policy. Only receiver-oriented summaries cross the LLM boundary.
    """

    def __init__(self, n_uavs: int, channel_history: int = 64) -> None:
        if channel_history < 1:
            raise ValueError("channel_history must be positive")
        self.n_uavs = n_uavs
        self.channels: deque[np.ndarray] = deque(maxlen=channel_history)

    def reset(self) -> None:
        self.channels.clear()

    def observe(self, snapshot: RacerSnapshot) -> None:
        snapshot.validate()
        self.channels.append(snapshot.channel_snr_db.copy())

    def build(self, snapshot: RacerSnapshot) -> dict[str, Any]:
        self.observe(snapshot)
        history = np.stack(tuple(self.channels), axis=0)
        finite = np.isfinite(history)
        available = finite & (history > NO_LINK_SNR_DB)

        def link_quality(sender: int, receiver: int) -> float | None:
            values = history[:, sender, receiver]
            valid = available[:, sender, receiver]
            if not np.any(valid):
                return None
            return round(float(np.mean(values[valid])), 3)

        reliability = available.mean(axis=0)
        coarse_communication = []
        for uav in range(self.n_uavs):
            peer_quality = [
                link_quality(uav, peer)
                for peer in range(self.n_uavs)
                if peer != uav
            ]
            reachable_quality = [
                value for value in peer_quality if value is not None
            ]
            coarse_communication.append(
                {
                    "uav_id": uav + 1,
                    "ul_bs_link_quality_db": link_quality(uav, self.n_uavs),
                    "dl_bs_link_quality_db": link_quality(self.n_uavs, uav),
                    "ul_bs_reliability": round(
                        float(reliability[uav, self.n_uavs]), 4
                    ),
                    "dl_bs_reliability": round(
                        float(reliability[self.n_uavs, uav]), 4
                    ),
                    "ul_bs_outage_ratio": round(
                        1.0 - float(reliability[uav, self.n_uavs]), 4
                    ),
                    "dl_bs_outage_ratio": round(
                        1.0 - float(reliability[self.n_uavs, uav]), 4
                    ),
                    "reachable_u2u_neighbor_count": len(reachable_quality),
                    "mean_u2u_link_quality_db": (
                        round(float(np.mean(reachable_quality)), 3)
                        if reachable_quality
                        else None
                    ),
                    "best_u2u_link_quality_db": (
                        round(float(np.max(reachable_quality)), 3)
                        if reachable_quality
                        else None
                    ),
                }
            )

        peer_aoi_summary = []
        for receiver in range(self.n_uavs):
            peer_aoi = np.delete(snapshot.pair_aoi_s[:, receiver], receiver)
            peer_aoi_summary.append(
                {
                    "uav_id": receiver + 1,
                    "max_peer_aoi_s": (
                        round(float(np.max(peer_aoi)), 4)
                        if peer_aoi.size
                        else 0.0
                    ),
                    "mean_peer_aoi_s": (
                        round(float(np.mean(peer_aoi)), 4)
                        if peer_aoi.size
                        else 0.0
                    ),
                    "stale_peer_count": int(np.count_nonzero(peer_aoi > 5.0)),
                }
            )
        trajectories = snapshot.trajectory_summary or [
            {
                "goal_position": snapshot.positions[i].tolist(),
                "trajectory_length": 0.0,
                "expected_execution_time": 0.0,
            }
            for i in range(self.n_uavs)
        ]
        # Missing BS region data means unknown. Never synthesize regions from
        # UAV positions or simulator union coverage.
        regions = snapshot.region_summary or []
        return {
            "bs_map_summary": snapshot.bs_map_summary,
            "bs_exploration_progress": [
                round(float(snapshot.bs_global_map_coverage), 6),
                round(float(snapshot.bs_map_coverage_delta), 6),
            ],
            "uav_bs_missing_bytes": snapshot.uav_bs_missing_bytes.tolist(),
            "bs_uav_missing_bytes": snapshot.bs_uav_missing_bytes.tolist(),
            "racer_states": list(snapshot.fsm_states),
            "positions": snapshot.positions.round(4).tolist(),
            "trajectory_summary": trajectories,
            "region_summary": regions,
            "bs_aoi_s": snapshot.bs_aoi_s.round(4).tolist(),
            "peer_aoi_summary": peer_aoi_summary,
            "coarse_communication_summary": coarse_communication,
        }
