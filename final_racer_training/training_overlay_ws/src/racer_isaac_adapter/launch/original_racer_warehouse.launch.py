from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _shared_remappings():
    pairs = (
        ("/swarm_expl/drone_state_send", "/racer_fidelity/swarm/drone_state"),
        ("/swarm_expl/drone_state_recv", "/racer_fidelity/swarm/drone_state"),
        ("/swarm_expl/pair_opt_send", "/racer_fidelity/swarm/pair_opt"),
        ("/swarm_expl/pair_opt_recv", "/racer_fidelity/swarm/pair_opt"),
        ("/swarm_expl/pair_opt_res_send", "/racer_fidelity/swarm/pair_opt_res"),
        ("/swarm_expl/pair_opt_res_recv", "/racer_fidelity/swarm/pair_opt_res"),
        ("/planning/swarm_traj_send", "/racer_fidelity/swarm/trajectory"),
        ("/planning/swarm_traj_recv", "/racer_fidelity/swarm/trajectory"),
        ("/multi_map_manager/chunk_stamps_send", "/racer_fidelity/map/chunk_stamps"),
        ("/multi_map_manager/chunk_stamps_recv", "/racer_fidelity/map/chunk_stamps"),
        ("/multi_map_manager/chunk_data_send", "/racer_fidelity/map/chunk_data"),
        ("/multi_map_manager/chunk_data_recv", "/racer_fidelity/map/chunk_data"),
    )
    return list(pairs)


def _launch_nodes(context):
    drone_count = int(LaunchConfiguration("drone_count").perform(context))
    lkh_dir = Path(LaunchConfiguration("lkh_dir").perform(context)).resolve()
    lkh_dir.mkdir(parents=True, exist_ok=True)
    params_file = str(
        Path(get_package_share_directory("racer_isaac_adapter"))
        / "config"
        / "original_warehouse_simple.yaml"
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
            {
                "exploration.drone_id": drone_id,
                "exploration.drone_num": drone_count,
                "exploration.vis_drone_id": 1,
                "exploration.tsp_dir": str(lkh_dir),
                "exploration.mtsp_dir": str(lkh_dir),
                "traj_server.drone_id": drone_id,
                "traj_server.drone_num": drone_count,
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
        ] + _shared_remappings()
        nodes.append(
            Node(
                package="racer_original_core",
                executable="racer_original_exploration_node",
                name=f"racer_original_exploration_{drone_id}",
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
                    "delay": 5.0,
                    "repeats": 10,
                    "drone_count": drone_count,
                    "minimum_cloud_frames": 25,
                }
            ],
        )
    )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("drone_count", default_value="5"),
            DeclareLaunchArgument(
                "lkh_dir", default_value="/tmp/racer_original_fidelity_lkh"
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
