from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _shared_remappings(zero_id):
    tx = f"/racer_sionna/tx/drone_{zero_id}"
    rx = f"/racer_sionna/rx/drone_{zero_id}"
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
        ("/racer_recovery/status", tx + "/recovery_status"),
        ("/racer_recovery/command", rx + "/recovery_command"),
    )
    return list(pairs)


def _launch_nodes(context):
    drone_count = int(LaunchConfiguration("drone_count").perform(context))
    scenario = LaunchConfiguration("scenario").perform(context)
    if scenario not in (
        "warehouse_simple", "warehouse_loaded", "warehouse_loaded_center",
        "warehouse_loaded_full",
    ):
        raise RuntimeError(f"unsupported warehouse scenario: {scenario}")
    communication_mode = LaunchConfiguration("communication_mode").perform(context)
    network_topology = LaunchConfiguration("network_topology").perform(context)
    if network_topology not in (
        "distributed", "ap_assisted", "bs_round_robin"
    ):
        raise RuntimeError(
            "network_topology must be distributed, ap_assisted, or "
            "bs_round_robin"
        )
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
    recovery_params_file = str(
        Path(get_package_share_directory("racer_recovery_core"))
        / "config"
        / "recovery.yaml"
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
        # Centre of the loaded RACER flight volume, mounted with the same
        # enclosure/phase-centre offset as the Warehouse Simple BS.
        ap_position = [-13.5, 16.0, 8.55]
    if scenario == "warehouse_loaded_full":
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
    lkh_executable = str(
        Path(get_package_prefix("racer_recovery_core"))
        / "lib"
        / "racer_recovery_core"
        / "racer_recovery_lkh_cli"
    )
    nodes = []
    for zero_id in range(drone_count):
        drone_id = zero_id + 1
        prefix = f"/drone_{zero_id}"
        agent_parameters = [
            params_file,
            recovery_params_file,
            scenario_parameters,
            {
                "exploration.drone_id": drone_id,
                "exploration.drone_num": drone_count,
                "exploration.vis_drone_id": 1,
                "exploration.tsp_dir": str(lkh_dir),
                "exploration.mtsp_dir": str(lkh_dir),
                "traj_server.drone_id": drone_id,
                "traj_server.drone_num": drone_count,
                "recovery.ap_enabled": network_topology in (
                    "ap_assisted", "bs_round_robin"
                ),
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
                package="racer_recovery_core",
                executable="racer_recovery_exploration_node",
                name=f"racer_recovery_exploration_{drone_id}",
                output="screen",
                parameters=agent_parameters,
                remappings=agent_remaps,
            )
        )
        nodes.append(
            Node(
                package="racer_recovery_core",
                executable="racer_recovery_traj_server",
                name=f"racer_recovery_traj_server_{drone_id}",
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
                    package="racer_recovery_core",
                    executable="racer_recovery_lkh_server",
                    name=f"racer_recovery_{label}_{drone_id}",
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
                    "delay": 5.0,
                    "repeats": 10,
                    "drone_count": drone_count,
                    "minimum_cloud_frames": 25,
                }
            ],
        )
    )
    if network_topology in ("ap_assisted", "bs_round_robin"):
        nodes.append(
            Node(
                package="racer_recovery_core",
                executable="racer_ap_recovery_coordinator",
                name="racer_ap_recovery_coordinator",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "drone_count": drone_count,
                        "status_timeout_s": 3.0,
                        "command_cooldown_s": 2.0,
                        "grid_block_duration_s": 30.0,
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
                communication_config,
                {
                    "use_sim_time": True,
                    "drone_count": drone_count,
                    "mode": communication_mode,
                    "network_topology": network_topology,
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
                    communication_config,
                    {
                        "use_sim_time": True,
                        "drone_count": drone_count,
                        "network_topology": network_topology,
                        "ap_position": ap_position,
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
            DeclareLaunchArgument("scenario", default_value="warehouse_simple"),
            DeclareLaunchArgument("communication_mode", default_value="sionna"),
            DeclareLaunchArgument("network_topology", default_value="distributed"),
            DeclareLaunchArgument("require_sionna", default_value="true"),
            DeclareLaunchArgument("sionna_scene_xml", default_value=""),
            DeclareLaunchArgument("radio_map_cache", default_value=""),
            DeclareLaunchArgument(
                "lkh_dir", default_value="/tmp/racer_recovery_lkh"
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
