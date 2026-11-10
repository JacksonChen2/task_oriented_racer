"""Fixed-size binary ABIs for the 10 Hz actor state data plane.

The payload length depends only on the configured UAV count, never on map,
trajectory, packet-history, or queue length.  C++ writes the fast-state ABI;
Python writes the tiny physical/guidance companion ABIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import numpy as np

FAST_STATE_MAGIC = b"FRFAST1\0"
FAST_STATE_ABI = 1
FAST_STATE_HEADER = struct.Struct(
    "<8sII"      # magic, abi, n
    "QQQQ"       # step_id, sequence, communication slot, decision
    "ddd"        # sim time, communication slot, decision interval
    "QQQQQQQ"    # action/policy/version provenance, source step_id
    "dd"         # action source sim time, action age
    "QQ"         # held slots, physical version
    "ddddd"      # coverage/delta and interval UL/DL/U2U resources
    "BBBB"       # terminated, truncated, completed transition, reserved
    "Q"          # embedded guidance id
)

PHYSICAL_FAST_MAGIC = b"FRPHY01\0"
PHYSICAL_FAST_ABI = 1
PHYSICAL_FAST_HEADER = struct.Struct("<8sIIQdddBB6x")

GUIDANCE_FAST_MAGIC = b"FRGDN01\0"
GUIDANCE_FAST_ABI = 1
GUIDANCE_FAST_HEADER = struct.Struct("<8sIIQ")


@dataclass(frozen=True)
class DecodedFastState:
    values: dict[str, Any]
    positions: np.ndarray
    channel_snr_db: np.ndarray
    pair_aoi_s: np.ndarray
    bs_aoi_s: np.ndarray
    pair_missing_bytes: np.ndarray
    bs_missing_bytes: np.ndarray
    uplink_queue_bytes: np.ndarray
    relay_queue_bytes: np.ndarray
    relay_action: np.ndarray
    upload_action: np.ndarray
    guidance_task_dependency: np.ndarray
    guidance_semantic_importance: np.ndarray


def encode_physical_fast(
    *,
    n_uavs: int,
    sim_step: int,
    sim_time_s: float,
    coverage: float,
    coverage_delta: float,
    terminated: bool,
    truncated: bool,
) -> bytes:
    return PHYSICAL_FAST_HEADER.pack(
        PHYSICAL_FAST_MAGIC,
        PHYSICAL_FAST_ABI,
        int(n_uavs),
        int(sim_step),
        float(sim_time_s),
        float(coverage),
        float(coverage_delta),
        bool(terminated),
        bool(truncated),
    )


def encode_guidance_fast(
    *,
    guidance_id: int,
    task_dependency: np.ndarray,
    semantic_importance: np.ndarray,
) -> bytes:
    task = np.asarray(task_dependency, dtype="<f4")
    importance = np.asarray(semantic_importance, dtype="<f4")
    if task.ndim != 2 or task.shape[0] != task.shape[1]:
        raise ValueError("task_dependency must be square")
    n = int(task.shape[0])
    if importance.shape != (n,):
        raise ValueError("semantic_importance must have shape (N,)")
    return b"".join(
        (
            GUIDANCE_FAST_HEADER.pack(
                GUIDANCE_FAST_MAGIC, GUIDANCE_FAST_ABI, n, int(guidance_id)
            ),
            task.tobytes(order="C"),
            importance.tobytes(order="C"),
        )
    )


def decode_fast_state(payload: bytes, expected_n_uavs: int) -> DecodedFastState:
    if len(payload) < FAST_STATE_HEADER.size:
        raise ValueError("fast-state payload is shorter than its header")
    fields = FAST_STATE_HEADER.unpack_from(payload)
    if fields[0] != FAST_STATE_MAGIC or fields[1] != FAST_STATE_ABI:
        raise ValueError("fast-state ABI mismatch")
    n = int(fields[2])
    if n != int(expected_n_uavs):
        raise ValueError(f"fast-state UAV count {n} != {expected_n_uavs}")
    names = (
        "magic", "abi", "n_uavs", "step_id", "sequence",
        "communication_slot_index", "rl_decision_index", "sim_time_s",
        "communication_slot_duration_s", "rl_decision_interval_s",
        "action_version", "policy_version", "action_generated_wall_time_ns",
        "action_source_guidance_id", "action_source_physical_version",
        "action_source_communication_version", "action_source_step_id",
        "action_source_sim_time_s", "action_age", "action_held_slots",
        "physical_version", "coverage", "coverage_delta",
        "interval_bs_uplink_prb_slots", "interval_bs_downlink_prb_slots",
        "interval_direct_u2u_prb_slots", "terminated", "truncated",
        "has_completed_transition", "reserved", "guidance_id",
    )
    values = dict(zip(names, fields))
    offset = FAST_STATE_HEADER.size

    def array(dtype: str, count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        item = np.dtype(dtype)
        end = offset + item.itemsize * count
        if end > len(payload):
            raise ValueError("truncated fast-state array payload")
        output = np.frombuffer(payload, dtype=item, count=count, offset=offset)
        offset = end
        return output.reshape(shape).copy()

    decoded = DecodedFastState(
        values=values,
        positions=array("<f4", 3 * n, (n, 3)),
        channel_snr_db=array("<f4", (n + 1) ** 2, (n + 1, n + 1)),
        pair_aoi_s=array("<f4", n * n, (n, n)),
        bs_aoi_s=array("<f4", n, (n,)),
        pair_missing_bytes=array("<u8", n * n, (n, n)).astype(np.float32),
        bs_missing_bytes=array("<u8", n, (n,)).astype(np.float32),
        uplink_queue_bytes=array("<u8", n, (n,)).astype(np.float32),
        relay_queue_bytes=array("<u8", n * n, (n, n)).astype(np.float32),
        relay_action=array("u1", n * n, (n, n)).astype(np.int8),
        upload_action=array("u1", n, (n,)).astype(np.int8),
        guidance_task_dependency=array("<f4", n * n, (n, n)),
        guidance_semantic_importance=array("<f4", n, (n,)),
    )
    if offset != len(payload):
        raise ValueError("fast-state payload has trailing bytes")
    return decoded
