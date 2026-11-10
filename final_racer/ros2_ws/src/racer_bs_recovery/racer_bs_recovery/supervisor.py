"""Central BS supervisor that temporarily moves a path-stuck RACER Chen UAV."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import re
from typing import Deque, Dict, List, Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from rcl_interfaces.msg import Log
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Empty
from visualization_msgs.msg import Marker

from .grid_planner import PlannedRecovery, SparseVoxelMap


NORMAL = "normal"
ACTIVE = "active"
SETTLING = "settling"
COOLDOWN = "cooldown"


@dataclass
class Agent:
    position: Optional[np.ndarray] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    errors: Deque[float] = field(default_factory=deque)
    failure_anchor: Optional[np.ndarray] = None
    state: str = NORMAL
    state_since: float = 0.0
    cooldown_until: float = 0.0
    recovery_origin: Optional[np.ndarray] = None
    plan: Optional[PlannedRecovery] = None
    waypoint_index: int = 1


class BsRecoverySupervisor(Node):
    """Observe RACER externally and own velocity only during bounded recovery."""

    def __init__(self) -> None:
        super().__init__("racer_bs_recovery_supervisor")
        defaults = {
            "drone_count": 5,
            "map_resolution": 0.30,
            "map_origin": [-28.1, -27.0, 0.0],
            "clearance_m": 0.45,
            "failure_window_s": 3.0,
            "failure_count": 20,
            "minimum_progress_m": 0.25,
            "maximum_recovery_distance_m": 3.0,
            "staging_search_radius_m": 1.8,
            "maximum_recovery_time_s": 12.0,
            "maximum_speed_mps": 0.60,
            "position_gain": 1.2,
            "waypoint_tolerance_m": 0.35,
            "settle_time_s": 0.40,
            "cooldown_s": 8.0,
            "cloud_min_period_s": 0.50,
            "cloud_point_stride": 32,
            "frontier_timeout_s": 5.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.drone_count = int(self._param("drone_count"))
        self.failure_window = float(self._param("failure_window_s"))
        self.failure_count = int(self._param("failure_count"))
        self.minimum_progress = float(self._param("minimum_progress_m"))
        self.maximum_motion = float(self._param("maximum_recovery_distance_m"))
        self.staging_radius = float(self._param("staging_search_radius_m"))
        self.maximum_time = float(self._param("maximum_recovery_time_s"))
        self.maximum_speed = float(self._param("maximum_speed_mps"))
        self.position_gain = float(self._param("position_gain"))
        self.waypoint_tolerance = float(self._param("waypoint_tolerance_m"))
        self.settle_time = float(self._param("settle_time_s"))
        self.cooldown = float(self._param("cooldown_s"))
        self.cloud_min_period = float(self._param("cloud_min_period_s"))
        self.cloud_stride = int(self._param("cloud_point_stride"))
        self.frontier_timeout = float(self._param("frontier_timeout_s"))
        if self.drone_count < 1 or self.failure_count < 1 or self.cloud_stride < 1:
            raise ValueError("invalid BS recovery parameters")

        self.grid = SparseVoxelMap(
            float(self._param("map_resolution")),
            self._param("map_origin"),
            float(self._param("clearance_m")),
        )
        self.agents = [Agent() for _ in range(self.drone_count)]
        self.frontiers: Dict[Tuple[int, str, int], Tuple[float, np.ndarray]] = {}
        self.last_cloud_at = [-math.inf] * self.drone_count
        self.command_publishers = []
        self.override_publishers = []
        self.replan_publishers = []

        override_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for index in range(self.drone_count):
            prefix = f"/drone_{index}"
            self.create_subscription(
                Odometry,
                prefix + "/odom",
                lambda msg, i=index: self._on_odom(i, msg),
                20,
            )
            self.create_subscription(
                PointCloud2,
                prefix + "/points",
                lambda msg, i=index: self._on_cloud(i, msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Marker,
                prefix + "/planning_vis/frontier",
                lambda msg, i=index: self._on_frontier(i, msg),
                100,
            )
            self.command_publishers.append(
                self.create_publisher(Twist, prefix + "/cmd_vel_3d/bs", 20)
            )
            self.override_publishers.append(
                self.create_publisher(
                    Bool, prefix + "/bs_recovery/override", override_qos
                )
            )
            # This is RACER's existing measured-odometry recovery interface.
            self.replan_publishers.append(
                self.create_publisher(Empty, prefix + "/tracking_lost", 10)
            )
        self.create_subscription(Log, "/rosout", self._on_log, 100)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"BS recovery ready for {self.drone_count} UAVs; "
            "RACER source remains untouched"
        )

    def _param(self, name):
        return self.get_parameter(name).value

    def _time(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_odom(self, index: int, message: Odometry) -> None:
        p = message.pose.pose.position
        v = message.twist.twist.linear
        self.agents[index].position = np.asarray((p.x, p.y, p.z), dtype=float)
        self.agents[index].velocity = np.asarray((v.x, v.y, v.z), dtype=float)

    def _on_log(self, message: Log) -> None:
        if "No path to next viewpoint" not in message.msg:
            return
        match = re.search(r"(?:exploration_|_)(\d+)$", message.name)
        if match is None:
            return
        index = int(match.group(1)) - 1
        if not 0 <= index < self.drone_count:
            return
        agent = self.agents[index]
        now = self._time()
        agent.errors.append(now)
        if agent.failure_anchor is None and agent.position is not None:
            agent.failure_anchor = agent.position.copy()

    def _on_cloud(self, index: int, message: PointCloud2) -> None:
        now = self._time()
        if now - self.last_cloud_at[index] < self.cloud_min_period:
            return
        position = self.agents[index].position
        if position is None:
            return
        self.last_cloud_at[index] = now
        try:
            rows = point_cloud2.read_points(
                message,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=True,
            )
            for sample_index, row in enumerate(rows):
                if sample_index % self.cloud_stride:
                    continue
                endpoint = (float(row[0]), float(row[1]), float(row[2]))
                self.grid.integrate_ray(position, endpoint, float(row[3]) > 0.5)
        except (AssertionError, IndexError, TypeError, ValueError) as error:
            self.get_logger().warning(
                f"cannot decode drone {index + 1} point cloud: {error}",
                throttle_duration_sec=5.0,
            )

    def _on_frontier(self, index: int, message: Marker) -> None:
        key = (index, message.ns, int(message.id))
        if message.action in (Marker.DELETE, Marker.DELETEALL):
            if message.action == Marker.DELETEALL:
                self.frontiers = {
                    k: value for k, value in self.frontiers.items() if k[0] != index
                }
            else:
                self.frontiers.pop(key, None)
            return
        points = np.asarray(
            [(point.x, point.y, point.z) for point in message.points], dtype=float
        ).reshape((-1, 3))
        if len(points):
            self.frontiers[key] = (self._time(), points[::max(1, len(points) // 500)])
        else:
            self.frontiers.pop(key, None)

    def _fresh_frontiers(self, now: float) -> List[np.ndarray]:
        expired = [
            key for key, (stamp, _) in self.frontiers.items()
            if now - stamp > self.frontier_timeout
        ]
        for key in expired:
            self.frontiers.pop(key, None)
        return [points for _, points in self.frontiers.values()]

    def _set_override(self, index: int, enabled: bool) -> None:
        message = Bool()
        message.data = enabled
        self.override_publishers[index].publish(message)

    def _begin_recovery(self, index: int, now: float) -> None:
        agent = self.agents[index]
        plan = self.grid.plan_to_frontiers(
            agent.position,
            self._fresh_frontiers(now),
            self.staging_radius,
            self.maximum_motion,
        )
        if plan is None:
            agent.errors.clear()
            agent.failure_anchor = None
            agent.cooldown_until = now + 1.0
            self.get_logger().warning(
                f"BS recovery has no known-free path for UAV {index + 1}; holding RACER authority"
            )
            return
        agent.state = ACTIVE
        agent.state_since = now
        agent.recovery_origin = agent.position.copy()
        agent.plan = plan
        agent.waypoint_index = 1
        self._set_override(index, True)
        self.get_logger().warning(
            f"BS_RECOVERY_START drone={index + 1} waypoints={len(plan.path)} "
            f"frontier={np.round(plan.frontier, 2).tolist()}"
        )

    def _begin_settle(self, index: int, now: float, reason: str) -> None:
        agent = self.agents[index]
        agent.state = SETTLING
        agent.state_since = now
        self.command_publishers[index].publish(Twist())
        self.get_logger().warning(
            f"BS_RECOVERY_RELEASE drone={index + 1} reason={reason}"
        )

    def _finish_release(self, index: int, now: float) -> None:
        agent = self.agents[index]
        self.command_publishers[index].publish(Twist())
        self._set_override(index, False)
        self.replan_publishers[index].publish(Empty())
        agent.state = COOLDOWN
        agent.cooldown_until = now + self.cooldown
        agent.errors.clear()
        agent.failure_anchor = None
        agent.plan = None
        agent.recovery_origin = None

    def _active_command(self, index: int, now: float) -> None:
        agent = self.agents[index]
        if agent.position is None or agent.plan is None:
            self._begin_settle(index, now, "state_lost")
            return
        path = agent.plan.path
        while agent.waypoint_index < len(path):
            target = np.asarray(path[agent.waypoint_index])
            if np.linalg.norm(target - agent.position) > self.waypoint_tolerance:
                break
            agent.waypoint_index += 1
        if agent.waypoint_index >= len(path):
            self._begin_settle(index, now, "staging_point_reached")
            return
        if now - agent.state_since > self.maximum_time:
            self._begin_settle(index, now, "timeout")
            return
        target = np.asarray(path[agent.waypoint_index])
        velocity = self.position_gain * (target - agent.position)
        speed = float(np.linalg.norm(velocity))
        if speed > self.maximum_speed:
            velocity *= self.maximum_speed / speed
        command = Twist()
        command.linear.x, command.linear.y, command.linear.z = map(float, velocity)
        if abs(velocity[0]) + abs(velocity[1]) > 1.0e-6:
            command.angular.z = math.atan2(velocity[1], velocity[0])
        self._set_override(index, True)
        self.command_publishers[index].publish(command)

    def _tick(self) -> None:
        now = self._time()
        for index, agent in enumerate(self.agents):
            while agent.errors and now - agent.errors[0] > self.failure_window:
                agent.errors.popleft()
            if agent.state == NORMAL:
                self._set_override(index, False)
                if now < agent.cooldown_until or agent.position is None:
                    continue
                progress = math.inf
                if agent.failure_anchor is not None:
                    progress = float(np.linalg.norm(agent.position - agent.failure_anchor))
                if len(agent.errors) >= self.failure_count and progress < self.minimum_progress:
                    self._begin_recovery(index, now)
            elif agent.state == ACTIVE:
                self._active_command(index, now)
            elif agent.state == SETTLING:
                self._set_override(index, True)
                self.command_publishers[index].publish(Twist())
                if now - agent.state_since >= self.settle_time:
                    self._finish_release(index, now)
            elif agent.state == COOLDOWN:
                self._set_override(index, False)
                if now >= agent.cooldown_until:
                    agent.state = NORMAL


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BsRecoverySupervisor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
