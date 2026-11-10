"""Independent, single-worker, latest-state-wins Qwen process."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from .config import load_config
from .fast_state import encode_guidance_fast
from .qwen_global_agent import QwenGlobalAgent, qwen_config_from_mapping
from .schemas import RacerSnapshot, TaskMetrics
from .shared_ipc import (
    SharedMemoryError,
    SharedMemoryLayout,
    open_blocks,
    shutdown_requested,
    snapshot_consistent,
    wall_time_ns,
)
from .state_builder import LargeStateBuilder


def _snapshot(
    physical: dict[str, Any], communication: dict[str, Any], n: int
) -> RacerSnapshot:
    """Build an LLM snapshot exclusively from the BS-observable slow state.

    ``physical`` is consulted only for episode lifecycle flags. Simulator
    positions, local-map summaries, union coverage and union coverage deltas
    must never cross this boundary.
    """

    sequence = int(communication["sequence"])
    zeros_3 = np.zeros((n, 3), dtype=np.float32)
    zeros_1 = np.zeros(n, dtype=np.float32)
    zeros_nn = np.zeros((n, n), dtype=np.float32)
    zeros_channel = np.full((n + 1, n + 1), -120.0, dtype=np.float32)
    np.fill_diagonal(zeros_channel, 0.0)
    return RacerSnapshot(
        n_uavs=n,
        slot=int(communication.get("communication_slot_index", sequence)),
        sim_time_s=float(communication.get("sim_time_s", 0.0)),
        positions=communication.get("positions", zeros_3),
        velocities=communication.get("velocities", zeros_3),
        yaws=communication.get("yaws", zeros_1),
        fsm_states=list(communication.get("fsm_states", ["UNKNOWN"] * n)),
        channel_snr_db=communication.get("channel_snr_db", zeros_channel),
        pair_aoi_s=communication.get("pair_aoi_s", zeros_nn),
        bs_aoi_s=communication.get("bs_aoi_s", zeros_1),
        pair_missing_bytes=communication.get("pair_missing_bytes", zeros_nn),
        bs_missing_bytes=communication.get(
            "uav_bs_missing_bytes",
            communication.get("bs_missing_bytes", zeros_1),
        ),
        uplink_queue_bytes=communication.get("uplink_queue_bytes", zeros_1),
        relay_queue_bytes=communication.get("relay_queue_bytes", zeros_nn),
        bs_map_summary=communication.get("bs_map_summary", {}),
        uav_bs_missing_bytes=communication.get(
            "uav_bs_missing_bytes",
            communication.get("bs_missing_bytes", zeros_1),
        ),
        bs_uav_missing_bytes=communication.get("bs_uav_missing_bytes", zeros_1),
        bs_map_coverage_delta=float(
            communication.get("bs_map_coverage_delta", 0.0)
        ),
        trajectory_summary=communication.get("trajectory_summary", []),
        region_summary=communication.get("region_summary", []),
        coverage=0.0,
        coverage_delta=0.0,
        bs_global_map_coverage=max(
            0.0, float(communication.get("bs_global_map_coverage", 0.0))
        ),
        task_metrics=TaskMetrics(),
        terminated=bool(physical.get("terminated", False)),
        truncated=bool(physical.get("truncated", False)),
        communication_slot_index=int(
            communication.get("communication_slot_index", sequence)
        ),
        rl_decision_index=int(
            communication.get("rl_decision_index", sequence)
        ),
        action_held_slots=int(communication.get("action_held_slots", 0)),
        communication_slot_duration_s=float(
            communication.get("communication_slot_duration_s", 0.02)
        ),
        rl_decision_interval_s=float(
            communication.get("rl_decision_interval_s", 0.1)
        ),
        state_sequence=sequence,
    ).validate()


def run(config_path: str, shared_memory_root: str) -> int:
    config = load_config(config_path)
    n = int(config["n_uavs"])
    qwen = config["qwen"]
    if not bool(qwen.get("enabled", True)):
        raise ValueError("LLM worker cannot start with qwen.enabled=false")
    constraint = config.get("constraint", {})
    fanout_limit = (
        int(constraint["max_relay_recipients_per_uav"])
        if constraint.get("type") == "relay_fanout"
        else None
    )
    layout = SharedMemoryLayout.from_value(shared_memory_root)
    blocks = open_blocks(
        layout,
        (
            "physical",
            "communication",
            "state_event",
            "guidance",
            "guidance_fast",
            "shutdown",
            "status_llm",
        ),
    )
    state_blocks = {
        "physical": blocks["physical"],
        "communication": blocks["communication"],
    }
    agent = QwenGlobalAgent(
        qwen_config_from_mapping(qwen, seed=int(config["seed"])),
        n,
        max_relay_recipients_per_uav=fanout_limit,
    )
    builder = LargeStateBuilder(
        n, int(config["environment"].get("channel_history", 50))
    )
    last_physical = 0
    last_communication = 0
    state_notification = blocks["state_event"].notification_sequence
    failures = 0
    llm_cycle_id = 0

    def stopping() -> bool:
        return shutdown_requested(blocks["shutdown"])

    blocks["status_llm"].publish_json(
        {"ready": False, "loading": True, "process": "llm", "pid": os.getpid()}
    )
    try:
        agent.load()
        blocks["status_llm"].publish_json(
            {"ready": True, "process": "llm", "pid": os.getpid()}
        )
        while not stopping():
            try:
                values = snapshot_consistent(state_blocks)
            except SharedMemoryError:
                updated = blocks["state_event"].wait_for_notification(
                    state_notification, stop=stopping
                )
                if updated is not None:
                    state_notification = updated
                continue
            physical_meta, physical = values["physical"]
            communication_meta, communication = values["communication"]
            state_notification = (
                blocks["state_event"].notification_sequence
            )
            if (
                physical_meta.version == last_physical
                and communication_meta.version == last_communication
            ):
                updated = blocks["state_event"].wait_for_notification(
                    state_notification, stop=stopping
                )
                if updated is not None:
                    state_notification = updated
                continue
            skipped_physical = max(
                0, physical_meta.version - last_physical - 1
            )
            skipped_communication = max(
                0, communication_meta.version - last_communication - 1
            )
            source_physical = physical_meta.version
            source_communication = communication_meta.version
            source_sim_step = physical_meta.sim_step
            source_step_id = int(
                communication.get("step_id", communication.get("task_step", 0))
            )
            source_sim_time = max(
                physical_meta.sim_time_s, communication_meta.sim_time_s
            )
            llm_cycle_id += 1
            started_ns = wall_time_ns()
            print(
                "RACER_LLM_CYCLE_BEGIN "
                + json.dumps(
                    {
                        "wall_time_ns": started_ns,
                        "llm_cycle_id": llm_cycle_id,
                        "source_physical_version": source_physical,
                        "source_communication_version": source_communication,
                        "source_racer_version": int(
                            communication.get("racer_version", communication["sequence"])
                        ),
                        "source_map_version": int(
                            communication.get("map_version", communication["sequence"])
                        ),
                        "source_sim_step": source_sim_step,
                        "source_step_id": source_step_id,
                        "source_sim_time_s": source_sim_time,
                        "skipped_physical_versions": skipped_physical,
                        "skipped_communication_versions": skipped_communication,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            # Mark this source consumed even on parse failure. A failure does
            # not create a retry queue; the next cycle takes the newest state.
            last_physical = source_physical
            last_communication = source_communication
            try:
                guidance = agent.infer(
                    builder.build(_snapshot(physical, communication, n))
                )
            except Exception as error:
                failures += 1
                print(
                    "RACER_LLM_CYCLE_ERROR "
                    + json.dumps(
                        {
                            "source_physical_version": source_physical,
                            "llm_cycle_id": llm_cycle_id,
                            "source_communication_version": source_communication,
                            "error": repr(error),
                            "failures": failures,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                continue
            guidance_id = blocks["guidance"].version + 1
            ended_ns = wall_time_ns()
            payload = {
                "guidance_id": guidance_id,
                "source_physical_version": source_physical,
                "source_communication_version": source_communication,
                "source_racer_version": int(
                    communication.get("racer_version", communication["sequence"])
                ),
                "source_map_version": int(
                    communication.get("map_version", communication["sequence"])
                ),
                "source_sim_step": source_sim_step,
                "source_step_id": source_step_id,
                "source_sim_time_s": source_sim_time,
                "task_dependency": guidance.task_dependency.tolist(),
                "semantic_importance": guidance.semantic_importance.tolist(),
                "inference_started_wall_time_ns": started_ns,
                "inference_completed_wall_time_ns": ended_ns,
                "inference_latency_s": agent.inference_latency_s,
                "parse_failures": failures + agent.parse_failures,
            }
            blocks["guidance"].publish_json(
                payload,
                sim_step=source_sim_step,
                sim_time_s=source_sim_time,
                version=guidance_id,
            )
            # The proxy embeds this fixed binary snapshot in subsequent Fast
            # RL State records, eliminating a second JSON read in inference.
            blocks["guidance_fast"].publish_bytes(
                encode_guidance_fast(
                    guidance_id=guidance_id,
                    task_dependency=guidance.task_dependency,
                    semantic_importance=guidance.semantic_importance,
                ),
                sim_step=source_sim_step,
                sim_time_s=source_sim_time,
                version=guidance_id,
            )
            print(
                "RACER_LLM_CYCLE_END "
                + json.dumps(
                    {
                        "guidance_id": guidance_id,
                        "llm_cycle_id": llm_cycle_id,
                        "source_physical_version": source_physical,
                        "source_communication_version": source_communication,
                        "source_racer_version": int(
                            communication.get("racer_version", communication["sequence"])
                        ),
                        "source_map_version": int(
                            communication.get("map_version", communication["sequence"])
                        ),
                        "source_sim_step": source_sim_step,
                        "source_step_id": source_step_id,
                        "started_wall_time_ns": started_ns,
                        "completed_wall_time_ns": ended_ns,
                        "latency_s": agent.inference_latency_s,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return 0
    finally:
        try:
            blocks["status_llm"].publish_json(
                {
                    "ready": False,
                    "stopping": True,
                    "process": "llm",
                    "pid": os.getpid(),
                }
            )
        finally:
            agent.close()
            for block in blocks.values():
                block.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shared-memory-root", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.shared_memory_root))


if __name__ == "__main__":
    main()
