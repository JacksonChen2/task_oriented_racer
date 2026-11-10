"""Fail-safe RACER Chen/BS velocity mux with explicit BS arming."""

from geometry_msgs.msg import Twist
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool


class CommandVelocityMux(Node):
    """Select exactly one fresh command source for a UAV."""

    def __init__(self) -> None:
        super().__init__("racer_bs_cmd_vel_mux")
        self.declare_parameter("drone_id", 1)
        self.declare_parameter("input_timeout_s", 0.25)
        drone_id = int(self.get_parameter("drone_id").value) - 1
        self.timeout = float(self.get_parameter("input_timeout_s").value)
        prefix = f"/drone_{drone_id}"
        self.racer = Twist()
        self.bs = Twist()
        self.racer_at = None
        self.bs_at = None
        self.override = False
        self.create_subscription(
            Twist, prefix + "/cmd_vel_3d/racer", self._on_racer, 20
        )
        self.create_subscription(Twist, prefix + "/cmd_vel_3d/bs", self._on_bs, 20)
        self.create_subscription(
            Bool, prefix + "/bs_recovery/override", self._on_override, 10
        )
        self.publisher = self.create_publisher(Twist, prefix + "/cmd_vel_3d", 20)
        self.create_timer(0.01, self._publish)

    def _on_racer(self, message: Twist) -> None:
        self.racer, self.racer_at = message, self.get_clock().now()

    def _on_bs(self, message: Twist) -> None:
        self.bs, self.bs_at = message, self.get_clock().now()

    def _on_override(self, message: Bool) -> None:
        self.override = bool(message.data)

    def _fresh(self, stamp) -> bool:
        return stamp is not None and (
            self.get_clock().now() - stamp
        ).nanoseconds * 1.0e-9 <= self.timeout

    def _publish(self) -> None:
        if self.override:
            output = self.bs if self._fresh(self.bs_at) else Twist()
        else:
            output = self.racer if self._fresh(self.racer_at) else Twist()
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandVelocityMux()
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
