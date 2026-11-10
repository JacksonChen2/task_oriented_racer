"""Create matching mock/real environments from one experiment config."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .backend import (
    FileBridgeBackend,
    MockRacerBackend,
    RacerBackend,
    SharedMemoryBackend,
)
from .event_driven_backend import EventDrivenOneShotSharedMemoryBackend
from .gym_env import RacerCRPOEnv
from .qwen_global_agent import (
    GuidanceManager,
    QwenGlobalAgent,
    qwen_config_from_mapping,
)
from .relay_fanout import RelayFanoutConstraint
from .single_gpu_pause import SingleGpuPauseConfig, SingleGpuPauseController
from .shared_guidance import SharedGuidanceManager
from .state_builder import StateNormalization
from .task_loss import TaskLossWeights


def make_backend(config: dict[str, Any], force_mock: bool = False) -> RacerBackend:
    backend = config["environment"]["backend"]
    kind = "mock" if force_mock else str(backend.get("kind", "mock"))
    n = int(config["n_uavs"])
    if kind == "mock":
        return MockRacerBackend(
            n,
            horizon=int(backend.get("horizon", 128)),
            seed=int(config["seed"]),
            direct_u2u_resource=float(
                backend.get("mock_direct_u2u_resource", 11.0)
            ),
        )
    if kind == "file_bridge":
        normalization = config["environment"]["normalization"]
        constraint = config.get("constraint", {})
        synchronous = backend.get("synchronous_online", {})
        fanout_limit = (
            int(constraint["max_relay_recipients_per_uav"])
            if constraint.get("type") == "relay_fanout"
            and bool(constraint.get("hard_enforcement", True))
            else None
        )
        return FileBridgeBackend(
            n,
            action_path=backend["action_path"],
            telemetry_path=backend["telemetry_path"],
            mission_state_path=backend.get("mission_state_path"),
            perfect_reference_result=backend.get("perfect_reference_result"),
            auxiliary_reference_result=backend.get(
                "auxiliary_reference_result"
            ),
            workspace_min=tuple(normalization["position_min"]),
            workspace_max=tuple(normalization["position_max"]),
            redundancy_cell_m=float(backend.get("redundancy_cell_m", 1.0)),
            fixed_exploration_time_s=float(
                backend.get("fixed_exploration_time_s", 300.0)
            ),
            require_auxiliary_reference_metrics=bool(
                backend.get("require_auxiliary_reference_metrics", True)
            ),
            require_mission_state=bool(
                backend.get("require_mission_state", True)
            ),
            require_perfect_reference=bool(
                backend.get("require_perfect_reference", True)
            ),
            max_relay_recipients_per_uav=fanout_limit,
            timeout_s=float(backend.get("timeout_s", 5.0)),
            poll_interval_s=float(backend.get("poll_interval_s", 0.002)),
            synchronous_online=bool(synchronous.get("enabled", False)),
            sync_acknowledgement_path=synchronous.get(
                "acknowledgement_path"
            ),
            sync_release_path=synchronous.get("release_path"),
            communication_slot_s=float(
                synchronous.get("communication_slot_s", 0.02)
            ),
            decision_interval_s=float(
                synchronous.get("decision_interval_s", 0.1)
            ),
            slots_per_decision=int(
                synchronous.get("slots_per_decision", 5)
            ),
        )
    if kind == "shared_memory":
        normalization = config["environment"]["normalization"]
        constraint = config.get("constraint", {})
        fanout_limit = (
            int(constraint["max_relay_recipients_per_uav"])
            if constraint.get("type") == "relay_fanout"
            and bool(constraint.get("hard_enforcement", True))
            else None
        )
        root = str(
            backend.get("shared_memory_root")
            or os.environ.get("RACER_SHARED_MEMORY_ROOT", "")
        )
        if not root:
            raise ValueError(
                "shared_memory backend requires shared_memory_root or "
                "RACER_SHARED_MEMORY_ROOT"
            )
        return SharedMemoryBackend(
            n,
            root,
            perfect_reference_result=backend.get(
                "perfect_reference_result"
            ),
            auxiliary_reference_result=backend.get(
                "auxiliary_reference_result"
            ),
            workspace_min=tuple(normalization["position_min"]),
            workspace_max=tuple(normalization["position_max"]),
            redundancy_cell_m=float(
                backend.get("redundancy_cell_m", 1.0)
            ),
            fixed_exploration_time_s=float(
                backend.get("fixed_exploration_time_s", 300.0)
            ),
            require_auxiliary_reference_metrics=bool(
                backend.get("require_auxiliary_reference_metrics", True)
            ),
            require_perfect_reference=bool(
                backend.get("require_perfect_reference", True)
            ),
            max_relay_recipients_per_uav=fanout_limit,
            communication_slot_s=(
                float(os.environ.get("RACER_RL_BS_COMMUNICATION_SLOT_MS", "20.0"))
                / 1000.0
            ),
            decision_interval_s=(
                float(os.environ.get("RACER_RL_BS_DECISION_PERIOD_MS", "100.0"))
                / 1000.0
            ),
        )
    if kind == "shared_memory_event_driven_one_shot":
        normalization = config["environment"]["normalization"]
        constraint = config.get("constraint", {})
        fanout_limit = (
            int(constraint["max_relay_recipients_per_uav"])
            if constraint.get("type") == "relay_fanout"
            and bool(constraint.get("hard_enforcement", True))
            else None
        )
        root = str(
            backend.get("shared_memory_root")
            or os.environ.get("RACER_SHARED_MEMORY_ROOT", "")
        )
        if not root:
            raise ValueError(
                "event-driven shared_memory backend requires "
                "shared_memory_root or RACER_SHARED_MEMORY_ROOT"
            )
        return EventDrivenOneShotSharedMemoryBackend(
            n,
            root,
            perfect_reference_result=backend.get(
                "perfect_reference_result"
            ),
            auxiliary_reference_result=backend.get(
                "auxiliary_reference_result"
            ),
            workspace_min=tuple(normalization["position_min"]),
            workspace_max=tuple(normalization["position_max"]),
            redundancy_cell_m=float(
                backend.get("redundancy_cell_m", 1.0)
            ),
            fixed_exploration_time_s=float(
                backend.get("fixed_exploration_time_s", 300.0)
            ),
            require_auxiliary_reference_metrics=bool(
                backend.get("require_auxiliary_reference_metrics", True)
            ),
            require_perfect_reference=bool(
                backend.get("require_perfect_reference", True)
            ),
            max_relay_recipients_per_uav=fanout_limit,
        )
    raise ValueError(f"unknown backend kind: {kind}")


def make_env(
    config: dict[str, Any],
    *,
    force_mock: bool = False,
    disable_qwen: bool = False,
) -> RacerCRPOEnv:
    n = int(config["n_uavs"])
    qwen_config = config["qwen"]
    constraint_config = config.get("constraint", {})
    constraint_mode = str(constraint_config.get("type", "task_loss"))
    fanout_limit = (
        int(constraint_config["max_relay_recipients_per_uav"])
        if constraint_mode == "relay_fanout"
        else None
    )
    qwen_enabled = bool(qwen_config.get("enabled", True)) and not disable_qwen
    agent = None
    pause_controller = None
    backend_kind = str(
        config["environment"]["backend"].get("kind", "mock")
    )
    shared_memory_kinds = {
        "shared_memory",
        "shared_memory_event_driven_one_shot",
    }
    if (
        qwen_enabled
        and backend_kind in shared_memory_kinds
        and not force_mock
    ):
        root = str(
            config["environment"]["backend"].get("shared_memory_root")
            or os.environ.get("RACER_SHARED_MEMORY_ROOT", "")
        )
        if not root:
            raise ValueError("shared guidance requires shared-memory root")
        manager = SharedGuidanceManager(root, n)
    elif qwen_enabled:
        agent = QwenGlobalAgent(
            qwen_config_from_mapping(qwen_config, seed=int(config["seed"])),
            n,
            max_relay_recipients_per_uav=fanout_limit,
        )
        pause_config = qwen_config.get("single_gpu_pause", {})
        if bool(pause_config.get("enabled", False)):
            pause_controller = SingleGpuPauseController(
                SingleGpuPauseConfig(
                    request_path=Path(pause_config["request_path"]),
                    acknowledgement_path=Path(
                        pause_config["acknowledgement_path"]
                    ),
                    timeout_s=float(pause_config.get("timeout_s", 600.0)),
                    poll_interval_s=float(
                        pause_config.get("poll_interval_s", 0.01)
                    ),
                )
            )
    if not (
        qwen_enabled
        and backend_kind in shared_memory_kinds
        and not force_mock
    ):
        manager = GuidanceManager(
            agent,
            n,
            asynchronous=bool(qwen_config.get("async_llm", True)),
            pause_controller=pause_controller,
        )
    environment = config["environment"]
    normalization = environment["normalization"]
    task = config["task_loss"]
    reward = config.get("reward", {})
    resource_normalizer = environment.get(
        "resource_normalizer", environment.get("bs_resource_normalizer")
    )
    if resource_normalizer is None:
        raise ValueError(
            "environment.resource_normalizer is required "
            "(legacy bs_resource_normalizer is also accepted)"
        )
    fanout = (
        RelayFanoutConstraint(n, fanout_limit)
        if fanout_limit is not None
        else None
    )
    return RacerCRPOEnv(
        make_backend(config, force_mock),
        manager,
        high_level_interval=int(environment["high_level_interval"]),
        channel_history=int(environment["channel_history"]),
        task_weights=TaskLossWeights(
            float(task["alpha_traj"]),
            float(task["alpha_cov"]),
            float(task["alpha_red"]),
            float(task["alpha_map"]),
        ),
        normalization=StateNormalization(
            position_min=tuple(normalization["position_min"]),
            position_max=tuple(normalization["position_max"]),
            snr_min_db=float(normalization["snr_min_db"]),
            snr_max_db=float(normalization["snr_max_db"]),
            max_aoi_s=float(normalization["max_aoi_s"]),
            max_missing_bytes=float(normalization["max_missing_bytes"]),
            max_queue_bytes=float(normalization["max_queue_bytes"]),
        ),
        bs_resource_normalizer=float(resource_normalizer),
        observation_layout=str(environment.get("observation_layout", "full")),
        include_global_guidance=bool(
            environment.get("include_global_guidance", True)
        ),
        include_exploration_phase=bool(
            config.get("actor", {}).get("include_exploration_phase", False)
        ),
        reward_mode=str(reward.get("type", "negative_bs_resource")),
        reward_scale=float(reward.get("scale", 1.0)),
        no_link_prb_per_action=float(
            reward.get("no_link_prb_per_action", 0.0)
        ),
        constraint_mode=constraint_mode,
        constraint_episode_duration_s=float(
            constraint_config.get("episode_duration_s", 300.0)
        ),
        relay_fanout=fanout,
        enforce_relay_fanout=bool(
            constraint_config.get("hard_enforcement", True)
        ),
        fanout_priority_weights=constraint_config.get(
            "projection_priority_weights"
        ),
    )
