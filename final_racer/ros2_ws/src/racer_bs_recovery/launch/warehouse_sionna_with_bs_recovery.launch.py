"""Launch RACER Chen Sionna with BS recovery and a scoped velocity remap."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def _launch(context):
    drone_count = int(LaunchConfiguration("drone_count").perform(context))
    original_launch = str(
        Path(get_package_share_directory("racer_sionna_comm"))
        / "launch"
        / "original_racer_warehouse_sionna.launch.py"
    )
    forwarded_names = (
        "drone_count",
        "scenario",
        "communication_mode",
        "network_topology",
        "require_sionna",
        "sionna_scene_xml",
        "radio_map_cache",
        "lkh_dir",
    )
    forwarded = {
        name: LaunchConfiguration(name) for name in forwarded_names
    }
    # SetRemap is scoped to the included original launch. It redirects only
    # the adapter output; no racer_original_core topic or algorithm changes.
    original_group = GroupAction(
        actions=[
            *[
                SetRemap(
                    src=f"/drone_{index}/cmd_vel_3d",
                    dst=f"/drone_{index}/cmd_vel_3d/racer",
                )
                for index in range(drone_count)
            ],
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(original_launch),
                launch_arguments=forwarded.items(),
            ),
        ]
    )
    recovery_share = Path(get_package_share_directory("racer_bs_recovery"))
    config = str(recovery_share / "config" / "default.yaml")
    recovery_nodes = [
        Node(
            package="racer_bs_recovery",
            executable="bs_recovery_supervisor",
            name="racer_bs_recovery_supervisor",
            output="screen",
            parameters=[config, {"drone_count": drone_count}],
        )
    ]
    for index in range(drone_count):
        recovery_nodes.append(
            Node(
                package="racer_bs_recovery",
                executable="cmd_vel_mux",
                name=f"racer_bs_cmd_vel_mux_{index + 1}",
                output="screen",
                parameters=[{"use_sim_time": True, "drone_id": index + 1}],
            )
        )
    return [original_group, *recovery_nodes]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("drone_count", default_value="5"),
            DeclareLaunchArgument("scenario", default_value="warehouse_loaded"),
            DeclareLaunchArgument("communication_mode", default_value="sionna"),
            DeclareLaunchArgument("network_topology", default_value="bs_round_robin"),
            DeclareLaunchArgument("require_sionna", default_value="true"),
            DeclareLaunchArgument("sionna_scene_xml", default_value=""),
            DeclareLaunchArgument("radio_map_cache", default_value=""),
            DeclareLaunchArgument(
                "lkh_dir", default_value="/tmp/racer_original_fidelity_lkh"
            ),
            OpaqueFunction(function=_launch),
        ]
    )
