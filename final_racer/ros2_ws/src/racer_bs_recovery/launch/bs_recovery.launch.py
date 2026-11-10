"""Launch only the external BS recovery nodes."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    drone_count = int(LaunchConfiguration("drone_count").perform(context))
    config = str(
        Path(get_package_share_directory("racer_bs_recovery"))
        / "config"
        / "default.yaml"
    )
    result = [
        Node(
            package="racer_bs_recovery",
            executable="bs_recovery_supervisor",
            name="racer_bs_recovery_supervisor",
            output="screen",
            parameters=[config, {"drone_count": drone_count}],
        )
    ]
    for zero_id in range(drone_count):
        result.append(
            Node(
                package="racer_bs_recovery",
                executable="cmd_vel_mux",
                name=f"racer_bs_cmd_vel_mux_{zero_id + 1}",
                output="screen",
                parameters=[
                    {"use_sim_time": True, "drone_id": zero_id + 1}
                ],
            )
        )
    return result


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("drone_count", default_value="5"),
            OpaqueFunction(function=_nodes),
        ]
    )
