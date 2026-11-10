import math
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _node_ros_parameters(config_path, node_name):
    """Load one node's defaults so launch arguments can override them.

    Passing the node-specific YAML file and a later anonymous parameter map to
    ``Node`` is not sufficient: ROS parameter-file specificity can make the
    node-specific defaults win over the anonymous launch map.  Merge them in
    Python first and pass one map to make the launch arguments authoritative.
    """

    with Path(config_path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    node = document.get(node_name, {})
    parameters = node.get("ros__parameters", {})
    if not isinstance(parameters, dict):
        raise RuntimeError(
            f"missing ros__parameters for {node_name} in {config_path}"
        )
    return dict(parameters)


def _shared_remappings(zero_id):
    tx = f"/racer_sionna/tx/drone_{zero_id}"
    rx = f"/racer_sionna/rx/drone_{zero_id}"
    cache = f"/racer_sionna/cache/drone_{zero_id}"
    pairs = (
        ("/swarm_expl/drone_state_send", tx + "/drone_state"),
        ("/swarm_expl/drone_state_recv", rx + "/drone_state"),
        ("/swarm_expl/pair_opt_send", tx + "/pair_opt"),
        ("/swarm_expl/pair_opt_recv", rx + "/pair_opt"),
        ("/swarm_expl/pair_opt_res_send", tx + "/pair_opt_res"),
        ("/swarm_expl/pair_opt_res_recv", rx + "/pair_opt_res"),
        ("/planning/swarm_traj_send", tx + "/trajectory"),
        ("/planning/swarm_traj_recv", rx + "/trajectory"),
        ("/multi_map_manager/chunk_stamps_send", tx + "/chunk_stamps"),
        ("/multi_map_manager/chunk_stamps_recv", rx + "/chunk_stamps"),
        ("/multi_map_manager/chunk_data_send", tx + "/chunk_data"),
        ("/multi_map_manager/chunk_data_recv", rx + "/chunk_data"),
        ("/multi_map_manager/chunk_payload_cache", cache + "/chunk_data"),
        ("/swarm_expl/global_assignment_send", tx + "/global_assignment"),
        ("/swarm_expl/global_assignment_recv", rx + "/global_assignment"),
        (
            "/hybrid_communication_racer/drone_state_send",
            tx + "/hybrid_drone_state",
        ),
        (
            "/hybrid_communication_racer/drone_state_recv",
            rx + "/hybrid_drone_state",
        ),
        (
            "/hybrid_communication_racer/global_assignment_send",
            tx + "/hybrid_global_assignment",
        ),
        (
            "/hybrid_communication_racer/global_assignment_recv",
            rx + "/hybrid_global_assignment",
        ),
    )
    return list(pairs)


def _launch_nodes(context):
    drone_count = int(LaunchConfiguration("drone_count").perform(context))
    trigger_minimum_cloud_frames = int(
        LaunchConfiguration("trigger_minimum_cloud_frames").perform(context)
    )
    trigger_delay_s = float(LaunchConfiguration("trigger_delay_s").perform(context))
    coverage_update_rate_hz = float(
        LaunchConfiguration("coverage_update_rate_hz").perform(context)
    )
    if trigger_minimum_cloud_frames < 0:
        raise RuntimeError("trigger_minimum_cloud_frames must be non-negative")
    if trigger_delay_s < 0.0:
        raise RuntimeError("trigger_delay_s must be non-negative")
    if not math.isfinite(coverage_update_rate_hz) or coverage_update_rate_hz <= 0.0:
        raise RuntimeError("coverage_update_rate_hz must be finite and positive")
    coverage_update_period_s = 1.0 / coverage_update_rate_hz
    scenario = LaunchConfiguration("scenario").perform(context)
    if scenario not in (
        "warehouse_simple", "warehouse_loaded", "warehouse_loaded_center",
        "warehouse_loaded_full", "warehouse_full",
    ):
        raise RuntimeError(f"unsupported warehouse scenario: {scenario}")
    communication_mode = LaunchConfiguration("communication_mode").perform(context)
    network_topology = LaunchConfiguration("network_topology").perform(context)
    if network_topology not in (
        "distributed", "nearest_neighbors", "distance_radius", "ap_assisted",
        "bs_round_robin",
    ):
        raise RuntimeError(
            "network_topology must be distributed, nearest_neighbors, "
            "distance_radius, ap_assisted, or bs_round_robin"
        )
    nearest_neighbor_count = int(
        LaunchConfiguration("nearest_neighbor_count").perform(context)
    )
    lossless_nearest_neighbor_count = int(
        LaunchConfiguration("lossless_nearest_neighbor_count").perform(context)
    )
    lossless_communication_range_m = float(
        LaunchConfiguration("lossless_communication_range_m").perform(context)
    )
    lossless_control_only = LaunchConfiguration(
        "lossless_control_only"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    directed_message_unicast = LaunchConfiguration(
        "directed_message_unicast"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    bs_channel_access_mode = LaunchConfiguration("bs_channel_access_mode").perform(context)
    bs_downlink_source_owned_chunks_only = LaunchConfiguration(
        "bs_downlink_source_owned_chunks_only"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    uav_channel_access_mode = LaunchConfiguration(
        "uav_channel_access_mode"
    ).perform(context)
    uav_csma_cw_min = int(
        LaunchConfiguration("uav_csma_cw_min").perform(context)
    )
    uav_csma_cw_max = int(
        LaunchConfiguration("uav_csma_cw_max").perform(context)
    )
    uav_csma_difs_slots = int(
        LaunchConfiguration("uav_csma_difs_slots").perform(context)
    )
    chunk_data_pre_enqueue_dedup = LaunchConfiguration(
        "chunk_data_pre_enqueue_dedup"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    chunk_data_max_pending_per_link = int(
        LaunchConfiguration("chunk_data_max_pending_per_link").perform(context)
    )
    communication_range_m = float(
        LaunchConfiguration("communication_range_m").perform(context)
    )
    ideal_coalesce_window_ms = float(
        LaunchConfiguration("ideal_coalesce_window_ms").perform(context)
    )
    preserve_ideal_direct_with_bs = LaunchConfiguration(
        "preserve_ideal_direct_with_bs"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    initial_assignment_perfect_delivery = LaunchConfiguration(
        "initial_assignment_perfect_delivery"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    initial_assignment_perfect_via_bs = LaunchConfiguration(
        "initial_assignment_perfect_via_bs"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    initial_partition_enabled = LaunchConfiguration(
        "initial_partition_enabled"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    require_sionna = LaunchConfiguration("require_sionna").perform(context).lower() in (
        "1", "true", "yes", "on"
    )
    lkh_dir = Path(LaunchConfiguration("lkh_dir").perform(context)).resolve()
    lkh_dir.mkdir(parents=True, exist_ok=True)
    params_file = str(
        Path(get_package_share_directory("racer_isaac_adapter"))
        / "config"
        / "original_warehouse_simple.yaml"
    )
    scenario_parameters = {}
    ap_position = [0.0, 3.0, 8.10]
    if scenario in (
        "warehouse_loaded", "warehouse_loaded_center", "warehouse_loaded_full"
    ):
        # Coordinate/storage adaptation only; all upstream RACER planning
        # modules continue to consume the same parameters and code paths.
        scenario_parameters = {
            "sdf_map.map_size_x": 56.2,
            "sdf_map.map_size_y": 54.0,
            "sdf_map.map_size_z": 9.0,
            "sdf_map.virtual_ceil_height": 8.4,
            "sdf_map.box_min_x": -26.8,
            "sdf_map.box_min_y": 7.2,
            "sdf_map.box_min_z": 0.4,
            "sdf_map.box_max_x": 5.8,
            "sdf_map.box_max_y": 26.2,
            "sdf_map.box_max_z": 8.4,
        }
        # Sionna radio-map optimum across the complete factory footprint and
        # the loaded RACER rack zone.  The enclosure hangs from z=9.0 m and
        # retains the same 0.45 m mounting-to-phase-centre offset.
        ap_position = [-13.5, 16.0, 8.55]
    if scenario == "warehouse_loaded_full":
        # Expand both the storage volume and coverage box to the complete
        # enclosed factory.  The original rack-zone profile remains unchanged.
        scenario_parameters = {
            "sdf_map.map_size_x": 56.2,
            "sdf_map.map_size_y": 64.0,
            "sdf_map.map_size_z": 9.0,
            "sdf_map.virtual_ceil_height": 8.4,
            "sdf_map.box_min_x": -27.0,
            "sdf_map.box_min_y": -23.0,
            "sdf_map.box_min_z": 0.4,
            "sdf_map.box_max_x": 6.0,
            "sdf_map.box_max_y": 30.0,
            "sdf_map.box_max_z": 8.4,
        }
    if scenario == "warehouse_full":
        scenario_parameters = {
            "sdf_map.map_size_x": 56.2,
            "sdf_map.map_size_y": 54.0,
            "sdf_map.map_size_z": 9.0,
            "sdf_map.virtual_ceil_height": 8.4,
            "sdf_map.box_min_x": -27.0,
            "sdf_map.box_min_y": 0.6,
            "sdf_map.box_min_z": 0.4,
            "sdf_map.box_max_x": 6.0,
            "sdf_map.box_max_y": 30.6,
            "sdf_map.box_max_z": 8.4,
        }
        ap_position = [-10.02891489217081, 14.888611215255622, 7.55]

    ap_overrides = [
        LaunchConfiguration(name).perform(context)
        for name in ("ap_position_x", "ap_position_y", "ap_position_z")
    ]
    if any(ap_overrides):
        if not all(ap_overrides):
            raise RuntimeError("all three ap_position overrides must be supplied")
        ap_position = [float(value) for value in ap_overrides]
    ap_tx_power_dbm = float(
        LaunchConfiguration("ap_tx_power_dbm").perform(context)
    )
    uav_tx_power_dbm = float(
        LaunchConfiguration("uav_tx_power_dbm").perform(context)
    )
    carrier_frequency_hz = float(
        LaunchConfiguration("carrier_frequency_hz").perform(context)
    )
    bandwidth_hz = float(
        LaunchConfiguration("bandwidth_hz").perform(context)
    )
    resource_blocks = int(
        LaunchConfiguration("resource_blocks").perform(context)
    )
    uav_broadcast_bandwidth_hz = float(
        LaunchConfiguration("uav_broadcast_bandwidth_hz").perform(context)
    )
    uav_broadcast_resource_blocks = int(
        LaunchConfiguration("uav_broadcast_resource_blocks").perform(context)
    )
    bs_bandwidth_hz = float(
        LaunchConfiguration("bs_bandwidth_hz").perform(context)
    )
    bs_resource_blocks = int(
        LaunchConfiguration("bs_resource_blocks").perform(context)
    )
    occupied_bandwidth_hz = 12.0 * 120.0e3 * resource_blocks
    if (
        not math.isfinite(bandwidth_hz)
        or bandwidth_hz <= 0.0
        or resource_blocks <= 0
        or occupied_bandwidth_hz > bandwidth_hz
    ):
        raise RuntimeError("invalid bandwidth_hz/resource_blocks allocation")
    bs_max_retries = int(
        LaunchConfiguration("bs_max_retries").perform(context)
    )
    max_retries = int(LaunchConfiguration("max_retries").perform(context))
    fixed_mcs_index = int(
        LaunchConfiguration("fixed_mcs_index").perform(context)
    )
    rl_bs_scheduler_enabled = LaunchConfiguration(
        "rl_bs_scheduler_enabled"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    force_bs_perfect_delivery = LaunchConfiguration(
        "force_bs_perfect_delivery"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    bs_periodic_upload_request_enabled = LaunchConfiguration(
        "bs_periodic_upload_request_enabled"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    bs_periodic_upload_request_period_ms = float(
        LaunchConfiguration("bs_periodic_upload_request_period_ms").perform(context)
    )
    bs_periodic_upload_chunks_per_request = int(
        LaunchConfiguration("bs_periodic_upload_chunks_per_request").perform(context)
    )
    bs_max_inflight_chunks_per_uav = int(
        LaunchConfiguration("bs_max_inflight_chunks_per_uav").perform(context)
    )
    bs_payload_precache_enabled = LaunchConfiguration(
        "bs_payload_precache_enabled"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    bs_payload_precache_batch_size = int(
        LaunchConfiguration("bs_payload_precache_batch_size").perform(context)
    )
    bs_max_payload_precache_inflight = int(
        LaunchConfiguration("bs_max_payload_precache_inflight").perform(context)
    )
    bs_payload_precache_timeout_ms = float(
        LaunchConfiguration("bs_payload_precache_timeout_ms").perform(context)
    )
    bs_max_uplink_chunks_per_rl_slot = int(
        LaunchConfiguration("bs_max_uplink_chunks_per_rl_slot").perform(context)
    )
    bs_max_downlink_chunks_per_rl_slot = int(
        LaunchConfiguration("bs_max_downlink_chunks_per_rl_slot").perform(context)
    )
    bs_downlink_newest_missing_chunk_first = LaunchConfiguration(
        "bs_downlink_newest_missing_chunk_first"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    bs_control_ttl_s = float(
        LaunchConfiguration("bs_control_ttl_s").perform(context)
    )
    if (
        not math.isfinite(bs_periodic_upload_request_period_ms)
        or bs_periodic_upload_request_period_ms <= 0.0
        or bs_periodic_upload_chunks_per_request <= 0
        or bs_max_inflight_chunks_per_uav < 0
        or (
            bs_max_inflight_chunks_per_uav > 0
            and bs_periodic_upload_chunks_per_request
            > bs_max_inflight_chunks_per_uav
        )
        or bs_payload_precache_batch_size <= 0
        or bs_max_payload_precache_inflight <= 0
        or bs_payload_precache_batch_size > bs_max_payload_precache_inflight
        or not math.isfinite(bs_payload_precache_timeout_ms)
        or bs_payload_precache_timeout_ms <= 0.0
        or bs_max_uplink_chunks_per_rl_slot < 0
        or bs_max_downlink_chunks_per_rl_slot < 0
        or not math.isfinite(bs_control_ttl_s)
        or bs_control_ttl_s <= 0.0
    ):
        raise RuntimeError(
            "BS upload timing must be positive; chunk limits may be zero for "
            "automatic sizing; periodic chunk budget must not exceed a "
            "positive per-UAV in-flight limit"
        )
    bs_all_to_all_relay_enabled = LaunchConfiguration(
        "bs_all_to_all_relay_enabled"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    pair_control_reservation_enabled = LaunchConfiguration(
        "pair_control_reservation_enabled"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    pair_control_reservation_s = float(
        LaunchConfiguration("pair_control_reservation_s").perform(context)
    )
    if not math.isfinite(pair_control_reservation_s) or pair_control_reservation_s <= 0.0:
        raise RuntimeError("pair_control_reservation_s must be finite and positive")
    bs_event_global_min_interval_s = float(
        LaunchConfiguration("bs_event_global_min_interval_s").perform(context)
    )
    bs_event_global_retry_interval_s = float(
        LaunchConfiguration("bs_event_global_retry_interval_s").perform(context)
    )
    bs_event_global_state_freshness_s = float(
        LaunchConfiguration("bs_event_global_state_freshness_s").perform(context)
    )
    bs_event_global_balance_weight_m = float(
        LaunchConfiguration("bs_event_global_balance_weight_m").perform(context)
    )
    bs_event_global_load_gap = int(
        LaunchConfiguration("bs_event_global_load_gap").perform(context)
    )
    if (
        any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                bs_event_global_min_interval_s,
                bs_event_global_retry_interval_s,
                bs_event_global_state_freshness_s,
                bs_event_global_balance_weight_m,
            )
        )
        or bs_event_global_load_gap <= 0
    ):
        raise RuntimeError("invalid BS event-global assignment parameters")
    rl_bs_synchronous_mode = LaunchConfiguration(
        "rl_bs_synchronous_mode"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    rl_bs_event_driven_one_shot = LaunchConfiguration(
        "rl_bs_event_driven_one_shot"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    rl_bs_action_path = LaunchConfiguration("rl_bs_action_path").perform(context)
    rl_bs_state_path = LaunchConfiguration("rl_bs_state_path").perform(context)
    rl_shared_memory_root = LaunchConfiguration(
        "rl_shared_memory_root"
    ).perform(context)
    ground_truth_occupied_voxels_path = LaunchConfiguration(
        "ground_truth_occupied_voxels_path"
    ).perform(context)
    observed_occupied_voxels_path = LaunchConfiguration(
        "observed_occupied_voxels_path"
    ).perform(context)
    require_ground_truth_map = LaunchConfiguration(
        "require_ground_truth_map"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    task_metric_observer_mode = LaunchConfiguration(
        "task_metric_observer_mode"
    ).perform(context)
    if task_metric_observer_mode not in ("inline", "async", "off"):
        raise RuntimeError(
            "task_metric_observer_mode must be inline, async, or off"
        )
    rl_bs_decision_period_ms = float(
        LaunchConfiguration("rl_bs_decision_period_ms").perform(context)
    )
    rl_bs_repeat_action_each_slot = LaunchConfiguration(
        "rl_bs_repeat_action_each_slot"
    ).perform(context).lower() in ("1", "true", "yes", "on")
    rl_bs_communication_slot_ms = float(
        LaunchConfiguration("rl_bs_communication_slot_ms").perform(context)
    )
    rl_llm_state_period_ms = float(
        LaunchConfiguration("rl_llm_state_period_ms").perform(context)
    )
    random_seed = int(LaunchConfiguration("random_seed").perform(context))
    algorithm_variant = LaunchConfiguration("algorithm_variant").perform(context)
    if algorithm_variant not in ("original", "oracle"):
        raise RuntimeError("algorithm_variant must be original or oracle")
    exploration_assignment_mode = LaunchConfiguration(
        "exploration_assignment_mode"
    ).perform(context)
    if exploration_assignment_mode not in (
        "original", "bs_event_global", "global_cooperative"
    ):
        raise RuntimeError(
            "exploration_assignment_mode must be original, bs_event_global, "
            "or global_cooperative"
        )
    if exploration_assignment_mode == "bs_event_global":
        if (
            network_topology != "bs_round_robin"
            or not rl_bs_scheduler_enabled
            or not initial_assignment_perfect_delivery
            or not initial_assignment_perfect_via_bs
        ):
            raise RuntimeError(
                "bs_event_global requires bs_round_robin, the RL BS scheduler, "
                "and perfect initial assignment via BS"
            )
        if pair_control_reservation_enabled:
            raise RuntimeError(
                "bs_event_global disables BS pair-control reservation"
            )
    lkh_executable = str(
        Path(get_package_prefix("racer_original_core"))
        / "lib"
        / "racer_original_core"
        / "racer_original_lkh_cli"
    )
    nodes = []
    for zero_id in range(drone_count):
        drone_id = zero_id + 1
        prefix = f"/drone_{zero_id}"
        agent_parameters = [
            params_file,
            scenario_parameters,
            {
                "exploration.drone_id": drone_id,
                "exploration.drone_num": drone_count,
                "exploration.vis_drone_id": 1,
                "exploration.tsp_dir": str(lkh_dir),
                "exploration.mtsp_dir": str(lkh_dir),
                "exploration.random_seed": random_seed,
                "topo_prm.random_seed": random_seed,
                # This ROS simulation-time timer is set to 10 Hz by the
                # synchronous Qwen/CRPO training launcher.
                "map_ros.coverage_diagnostic_period": coverage_update_period_s,
                "traj_server.drone_id": drone_id,
                "traj_server.drone_num": drone_count,
                "multi_map_manager.chunk_size": 50 if algorithm_variant == "oracle" else 200,
                # Push each sealed local ChunkData into the Proxy repository.
                # The -1 local marker is consumed before radio scheduling;
                # actual BS delivery still requires an RL-selected PHY upload.
                "multi_map_manager.push_chunk_payload_cache": (
                    algorithm_variant != "oracle"
                ),
                "fsm.repeat_send_num": 1 if algorithm_variant == "oracle" else 10,
                "oracle_shared_map.stamp_period": 0.05,
                "oracle_shared_map.chunk_period": 0.02,
                "oracle_shared_map.stamp_throttle": 0.05,
                "oracle_global.assignment_interval": 0.5,
                "oracle_global.balance_weight_m": 1.0,
                "exploration_assignment_mode": exploration_assignment_mode,
                "fsm.initial_partition_enabled": initial_partition_enabled,
                "fsm.bs_event_global_min_interval": (
                    bs_event_global_min_interval_s
                ),
                "fsm.bs_event_global_retry_interval": (
                    bs_event_global_retry_interval_s
                ),
                "fsm.bs_event_global_state_freshness": (
                    bs_event_global_state_freshness_s
                ),
                "fsm.bs_event_global_balance_weight": (
                    bs_event_global_balance_weight_m
                ),
                "fsm.bs_event_global_load_gap": bs_event_global_load_gap,
                "global_cooperative.assignment_interval": 0.5,
                "global_cooperative.state_freshness": 0.75,
                "global_cooperative.nominal_velocity": 1.5,
                "global_cooperative.gain_scale": 0.001,
                "global_cooperative.utility_epsilon": 0.1,
                "global_cooperative.lambda_overlap": 2.0,
                "global_cooperative.lambda_age": 0.05,
                "global_cooperative.overlap_distance": 6.0,
                "global_cooperative.cluster_distance": 3.0,
                "global_cooperative.distance_min": 8.0,
                "global_cooperative.distance_max": 45.0,
                "global_cooperative.distance_gamma": 0.7,
                "global_cooperative.distance_fallback_when_idle": True,
                "global_cooperative.finish_confirmation_cycles": 6,
                "global_cooperative.local_failures_before_report": 2,
                "global_cooperative.unreachable_distinct_uavs": 2,
                "global_cooperative.unreachable_global_failures": 4,
                "global_cooperative.unreachable_retry_s": 20.0,
            },
        ]
        agent_remaps = [
            ("/odom_world", prefix + "/odom"),
            ("/racer/sensor_points", prefix + "/points"),
            ("/racer/tracking_lost", prefix + "/tracking_lost"),
            ("/move_base_simple/goal", "/racer/start"),
            ("/planning/replan", prefix + "/planning/replan"),
            ("/planning/new", prefix + "/planning/new"),
            ("/planning/bspline", prefix + "/planning/bspline"),
            ("/planning_vis/trajectory", prefix + "/planning_vis/trajectory"),
            ("/planning_vis/topo_path", prefix + "/planning_vis/topo_path"),
            ("/planning_vis/prediction", prefix + "/planning_vis/prediction"),
            ("/planning_vis/visib_constraint", prefix + "/planning_vis/visib_constraint"),
            ("/planning_vis/frontier", prefix + "/planning_vis/frontier"),
            ("/planning_vis/yaw", prefix + "/planning_vis/yaw"),
            ("/planning_vis/viewpoints", prefix + "/planning_vis/viewpoints"),
            ("/swarm_expl/grid_tour_send", prefix + "/planning_vis/grid_tour"),
            ("/swarm_expl/hgrid_send", prefix + "/planning_vis/hgrid"),
            (f"/multi_map_manager/marker_{drone_id}", prefix + "/planning_vis/map_chunks"),
        ] + _shared_remappings(zero_id)
        nodes.append(
            Node(
                package="racer_original_core",
                executable=("racer_oracle_exploration_node"
                            if algorithm_variant == "oracle"
                            else "racer_original_exploration_node"),
                name=(f"racer_oracle_exploration_{drone_id}"
                      if algorithm_variant == "oracle"
                      else f"racer_original_exploration_{drone_id}"),
                output="screen",
                parameters=agent_parameters,
                remappings=agent_remaps,
            )
        )
        nodes.append(
            Node(
                package="racer_original_core",
                executable="racer_original_traj_server",
                name=f"racer_original_traj_server_{drone_id}",
                output="screen",
                parameters=agent_parameters,
                remappings=[
                    ("/odom_world", prefix + "/odom"),
                    ("/planning/bspline", prefix + "/planning/bspline"),
                    ("/planning/replan", prefix + "/planning/replan"),
                    ("/planning/new", prefix + "/planning/new"),
                    ("/position_cmd", prefix + "/position_cmd"),
                    ("/planning/position_cmd_vis", prefix + "/planning_vis/position_cmd"),
                    ("/planning/travel_traj", prefix + "/planning_vis/travel_traj"),
                    ("/loop_fusion/pg_T_vio", prefix + "/unused/pg_T_vio"),
                ],
            )
        )
        nodes.append(
            Node(
                package="racer_isaac_adapter",
                executable="position_command_adapter",
                name=f"racer_isaac_control_adapter_{drone_id}",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "drone_id": drone_id,
                        "position_gain": 2.5,
                        "maximum_speed": 2.0,
                        "command_timeout": 0.25,
                        "tracking_error_threshold": 1.2,
                        "tracking_error_hold": 1.0,
                        "tracking_low_speed_threshold": 0.25,
                        "tracking_severe_error_threshold": 3.0,
                        "recovery_cooldown": 2.0,
                        "recovery_brake_time": 0.4,
                    }
                ],
            )
        )
        for problem_id, label in ((1, "tsp"), (2, "acvrp")):
            nodes.append(
                Node(
                    package="racer_original_core",
                    executable="racer_original_lkh_server",
                    name=f"racer_original_{label}_{drone_id}",
                    output="screen",
                    parameters=[
                        {
                            "use_sim_time": True,
                            "exploration.drone_id": drone_id,
                            "exploration.problem_id": problem_id,
                            "exploration.mtsp_dir": str(lkh_dir),
                            "exploration.lkh_executable": lkh_executable,
                        }
                    ],
                )
            )
    nodes.append(
        Node(
            package="racer_isaac_adapter",
            executable="exploration_trigger",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "delay": trigger_delay_s,
                    "repeats": 10,
                    "drone_count": drone_count,
                    "minimum_cloud_frames": trigger_minimum_cloud_frames,
                }
            ],
        )
    )
    communication_share = Path(get_package_share_directory("racer_sionna_comm"))
    communication_config = str(
        communication_share / "config" / "warehouse_simple_communication.yaml"
    )
    nodes.append(
        Node(
            package="racer_sionna_comm",
            executable="racer_sionna_communication_proxy",
            name="racer_sionna_communication_proxy",
            output="screen",
            parameters=[
                params_file,
                scenario_parameters,
                {
                    **_node_ros_parameters(
                        communication_config,
                        "racer_sionna_communication_proxy",
                    ),
                    "use_sim_time": True,
                    "drone_count": drone_count,
                    "mode": communication_mode,
                    "network_topology": network_topology,
                    "nearest_neighbor_count": nearest_neighbor_count,
                    "lossless_nearest_neighbor_count": lossless_nearest_neighbor_count,
                    "lossless_communication_range_m": lossless_communication_range_m,
                    "lossless_control_only": lossless_control_only,
                    "directed_message_unicast": directed_message_unicast,
                    "bs_channel_access_mode": bs_channel_access_mode,
                    "bs_downlink_source_owned_chunks_only": bs_downlink_source_owned_chunks_only,
                    "uav_channel_access_mode": uav_channel_access_mode,
                    "uav_csma_cw_min": uav_csma_cw_min,
                    "uav_csma_cw_max": uav_csma_cw_max,
                    "uav_csma_difs_slots": uav_csma_difs_slots,
                    "chunk_data_pre_enqueue_dedup": chunk_data_pre_enqueue_dedup,
                    "chunk_data_max_pending_per_link": chunk_data_max_pending_per_link,
                    "communication_range_m": communication_range_m,
                    "ideal_coalesce_window_ms": ideal_coalesce_window_ms,
                    "preserve_ideal_direct_with_bs": (
                        preserve_ideal_direct_with_bs
                    ),
                    "initial_assignment_perfect_delivery": (
                        initial_assignment_perfect_delivery
                    ),
                    "initial_assignment_perfect_via_bs": (
                        initial_assignment_perfect_via_bs
                    ),
                    "ap_tx_power_dbm": ap_tx_power_dbm,
                    "tx_power_dbm": uav_tx_power_dbm,
                    "carrier_frequency_hz": carrier_frequency_hz,
                    "bandwidth_hz": bandwidth_hz,
                    "resource_blocks": resource_blocks,
                    "uav_broadcast_bandwidth_hz": uav_broadcast_bandwidth_hz,
                    "uav_broadcast_resource_blocks": uav_broadcast_resource_blocks,
                    "bs_bandwidth_hz": bs_bandwidth_hz,
                    "bs_resource_blocks": bs_resource_blocks,
                    "max_retries": max_retries,
                    "bs_max_retries": bs_max_retries,
                    "fixed_mcs_index": fixed_mcs_index,
                    "rl_bs_scheduler_enabled": rl_bs_scheduler_enabled,
                    "force_bs_perfect_delivery": force_bs_perfect_delivery,
                    "bs_periodic_upload_request_enabled": (
                        bs_periodic_upload_request_enabled
                    ),
                    "bs_periodic_upload_request_period_ms": (
                        bs_periodic_upload_request_period_ms
                    ),
                    "bs_periodic_upload_chunks_per_request": (
                        bs_periodic_upload_chunks_per_request
                    ),
                    "bs_max_inflight_chunks_per_uav": (
                        bs_max_inflight_chunks_per_uav
                    ),
                    "bs_payload_precache_enabled": bs_payload_precache_enabled,
                    "bs_payload_precache_batch_size": (
                        bs_payload_precache_batch_size
                    ),
                    "bs_max_payload_precache_inflight": (
                        bs_max_payload_precache_inflight
                    ),
                    "bs_payload_precache_timeout_ms": (
                        bs_payload_precache_timeout_ms
                    ),
                    "bs_max_uplink_chunks_per_rl_slot": (
                        bs_max_uplink_chunks_per_rl_slot
                    ),
                    "bs_max_downlink_chunks_per_rl_slot": (
                        bs_max_downlink_chunks_per_rl_slot
                    ),
                    "bs_downlink_newest_missing_chunk_first": (
                        bs_downlink_newest_missing_chunk_first
                    ),
                    "bs_control_ttl_s": bs_control_ttl_s,
                    "bs_all_to_all_relay_enabled": (
                        bs_all_to_all_relay_enabled
                    ),
                    "pair_control_reservation_enabled": (
                        pair_control_reservation_enabled
                    ),
                    "pair_control_reservation_s": pair_control_reservation_s,
                    "rl_bs_synchronous_mode": rl_bs_synchronous_mode,
                    "rl_bs_event_driven_one_shot": (
                        rl_bs_event_driven_one_shot
                    ),
                    "rl_bs_action_path": rl_bs_action_path,
                    "rl_bs_state_path": rl_bs_state_path,
                    "rl_shared_memory_root": rl_shared_memory_root,
                    "rl_bs_communication_slot_ms": (
                        rl_bs_communication_slot_ms
                    ),
                    "ground_truth_occupied_voxels_path": (
                        ground_truth_occupied_voxels_path
                    ),
                    "observed_occupied_voxels_path": (
                        observed_occupied_voxels_path
                    ),
                    "require_ground_truth_map": require_ground_truth_map,
                    "task_metric_observer_mode": task_metric_observer_mode,
                    "rl_bs_decision_period_ms": rl_bs_decision_period_ms,
                    "rl_bs_repeat_action_each_slot": (
                        rl_bs_repeat_action_each_slot
                    ),
                    "rl_llm_state_period_ms": rl_llm_state_period_ms,
                    "random_seed": random_seed,
                },
            ],
        )
    )
    if communication_mode != "ideal":
        nodes.append(
            Node(
                package="racer_sionna_comm",
                executable="sionna_channel_node.py",
                name="racer_sionna_channel",
                output="screen",
                parameters=[
                    {
                        **_node_ros_parameters(
                            communication_config,
                            "racer_sionna_channel",
                        ),
                        "use_sim_time": True,
                        "drone_count": drone_count,
                        "network_topology": network_topology,
                        "ap_position": ap_position,
                        "ap_tx_power_dbm": ap_tx_power_dbm,
                        "tx_power_dbm": uav_tx_power_dbm,
                        "carrier_frequency_hz": carrier_frequency_hz,
                        "bandwidth_hz": bandwidth_hz,
                        "random_seed": random_seed,
                        "scene_xml": LaunchConfiguration("sionna_scene_xml").perform(context),
                        "radio_map_cache": LaunchConfiguration("radio_map_cache").perform(context),
                        "require_sionna": require_sionna,
                    },
                ],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("drone_count", default_value="5"),
            DeclareLaunchArgument(
                "trigger_minimum_cloud_frames", default_value="25"
            ),
            DeclareLaunchArgument("trigger_delay_s", default_value="5.0"),
            DeclareLaunchArgument(
                "coverage_update_rate_hz", default_value="10.0"
            ),
            DeclareLaunchArgument("scenario", default_value="warehouse_simple"),
            DeclareLaunchArgument("communication_mode", default_value="sionna"),
            DeclareLaunchArgument("algorithm_variant", default_value="original"),
            DeclareLaunchArgument(
                "exploration_assignment_mode", default_value="original"
            ),
            DeclareLaunchArgument("network_topology", default_value="distributed"),
            DeclareLaunchArgument("nearest_neighbor_count", default_value="0"),
            DeclareLaunchArgument(
                "lossless_nearest_neighbor_count", default_value="0"
            ),
            DeclareLaunchArgument(
                "lossless_communication_range_m", default_value="0.0"
            ),
            DeclareLaunchArgument("lossless_control_only", default_value="false"),
            DeclareLaunchArgument("directed_message_unicast", default_value="false"),
            DeclareLaunchArgument("bs_channel_access_mode", default_value="scheduled",
                                  choices=["scheduled", "csma", "ofdma"]),
            DeclareLaunchArgument("bs_downlink_source_owned_chunks_only", default_value="false"),
            DeclareLaunchArgument(
                "uav_channel_access_mode",
                default_value=EnvironmentVariable(
                    "RACER_UAV_CHANNEL_ACCESS_MODE", default_value="csma"
                ),
            ),
            DeclareLaunchArgument(
                "uav_csma_cw_min",
                default_value=EnvironmentVariable(
                    "RACER_UAV_CSMA_CW_MIN", default_value="15"
                ),
            ),
            DeclareLaunchArgument(
                "uav_csma_cw_max",
                default_value=EnvironmentVariable(
                    "RACER_UAV_CSMA_CW_MAX", default_value="1023"
                ),
            ),
            DeclareLaunchArgument(
                "uav_csma_difs_slots",
                default_value=EnvironmentVariable(
                    "RACER_UAV_CSMA_DIFS_SLOTS", default_value="2"
                ),
            ),
            DeclareLaunchArgument(
                "chunk_data_pre_enqueue_dedup", default_value="false"
            ),
            DeclareLaunchArgument(
                "chunk_data_max_pending_per_link", default_value="0"
            ),
            DeclareLaunchArgument("communication_range_m", default_value="4.0"),
            DeclareLaunchArgument("ideal_coalesce_window_ms", default_value="20.0"),
            DeclareLaunchArgument(
                "preserve_ideal_direct_with_bs", default_value="false"
            ),
            DeclareLaunchArgument(
                "initial_assignment_perfect_delivery", default_value="false"
            ),
            DeclareLaunchArgument(
                "initial_assignment_perfect_via_bs", default_value="false"
            ),
            DeclareLaunchArgument(
                "initial_partition_enabled", default_value="false"
            ),
            DeclareLaunchArgument("require_sionna", default_value="true"),
            DeclareLaunchArgument("sionna_scene_xml", default_value=""),
            DeclareLaunchArgument("radio_map_cache", default_value=""),
            DeclareLaunchArgument("ap_position_x", default_value=""),
            DeclareLaunchArgument("ap_position_y", default_value=""),
            DeclareLaunchArgument("ap_position_z", default_value=""),
            DeclareLaunchArgument("ap_tx_power_dbm", default_value="40.0"),
            DeclareLaunchArgument("uav_tx_power_dbm", default_value="23.0"),
            DeclareLaunchArgument(
                "carrier_frequency_hz", default_value="28000000000.0"
            ),
            DeclareLaunchArgument("bandwidth_hz", default_value="100000000.0"),
            DeclareLaunchArgument("resource_blocks", default_value="66"),
            DeclareLaunchArgument(
                "uav_broadcast_bandwidth_hz", default_value="50000000.0"
            ),
            DeclareLaunchArgument(
                "uav_broadcast_resource_blocks", default_value="33"
            ),
            DeclareLaunchArgument("bs_bandwidth_hz", default_value="50000000.0"),
            DeclareLaunchArgument("bs_resource_blocks", default_value="33"),
            DeclareLaunchArgument("max_retries", default_value="3"),
            DeclareLaunchArgument("bs_max_retries", default_value="3"),
            DeclareLaunchArgument("fixed_mcs_index", default_value="-1"),
            DeclareLaunchArgument(
                "rl_bs_scheduler_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "force_bs_perfect_delivery", default_value="false"
            ),
            DeclareLaunchArgument(
                "bs_periodic_upload_request_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "bs_periodic_upload_request_period_ms", default_value="200.0"
            ),
            DeclareLaunchArgument(
                "bs_periodic_upload_chunks_per_request", default_value="8"
            ),
            DeclareLaunchArgument(
                "bs_max_inflight_chunks_per_uav", default_value="0"
            ),
            DeclareLaunchArgument(
                "bs_payload_precache_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "bs_payload_precache_batch_size", default_value="8"
            ),
            DeclareLaunchArgument(
                "bs_max_payload_precache_inflight", default_value="16"
            ),
            DeclareLaunchArgument(
                "bs_payload_precache_timeout_ms", default_value="5000.0"
            ),
            DeclareLaunchArgument(
                "bs_max_uplink_chunks_per_rl_slot", default_value="0"
            ),
            DeclareLaunchArgument(
                "bs_max_downlink_chunks_per_rl_slot", default_value="0"
            ),
            DeclareLaunchArgument(
                "bs_downlink_newest_missing_chunk_first",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "bs_control_ttl_s", default_value="2.0"
            ),
            DeclareLaunchArgument(
                "bs_all_to_all_relay_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "pair_control_reservation_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "pair_control_reservation_s", default_value="1.8"
            ),
            DeclareLaunchArgument(
                "bs_event_global_min_interval_s", default_value="10.0"
            ),
            DeclareLaunchArgument(
                "bs_event_global_retry_interval_s", default_value="0.5"
            ),
            DeclareLaunchArgument(
                "bs_event_global_state_freshness_s", default_value="2.0"
            ),
            DeclareLaunchArgument(
                "bs_event_global_balance_weight_m", default_value="2.0"
            ),
            DeclareLaunchArgument(
                "bs_event_global_load_gap", default_value="3"
            ),
            DeclareLaunchArgument(
                "rl_bs_synchronous_mode", default_value="false"
            ),
            DeclareLaunchArgument(
                "rl_bs_event_driven_one_shot", default_value="false"
            ),
            DeclareLaunchArgument(
                "rl_bs_action_path",
                default_value="/tmp/racer_agentic_crpo/action.txt",
            ),
            DeclareLaunchArgument(
                "rl_bs_state_path",
                default_value=(
                    "/tmp/racer_agentic_crpo/communication_state.json"
                ),
            ),
            DeclareLaunchArgument("rl_shared_memory_root", default_value=""),
            DeclareLaunchArgument(
                "ground_truth_occupied_voxels_path", default_value=""
            ),
            DeclareLaunchArgument(
                "observed_occupied_voxels_path", default_value=""
            ),
            DeclareLaunchArgument(
                "require_ground_truth_map", default_value="false"
            ),
            DeclareLaunchArgument(
                "task_metric_observer_mode", default_value="inline"
            ),
            DeclareLaunchArgument(
                "rl_bs_decision_period_ms", default_value="20.0"
            ),
            DeclareLaunchArgument(
                "rl_bs_repeat_action_each_slot", default_value="true"
            ),
            DeclareLaunchArgument(
                "rl_bs_communication_slot_ms", default_value="20.0"
            ),
            DeclareLaunchArgument(
                "rl_llm_state_period_ms", default_value="5000.0"
            ),
            DeclareLaunchArgument("random_seed", default_value="42"),
            DeclareLaunchArgument(
                "lkh_dir", default_value="/tmp/racer_original_fidelity_lkh"
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
