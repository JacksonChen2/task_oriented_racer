#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <racer_fidelity_msgs/msg/position_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

class PositionCommandAdapter final : public rclcpp::Node {
 public:
  PositionCommandAdapter() : Node("position_command_adapter") {
    drone_id_ = declare_parameter<int>("drone_id", 1);
    position_gain_ = declare_parameter<double>("position_gain", 2.5);
    maximum_speed_ = declare_parameter<double>("maximum_speed", 2.0);
    command_timeout_ = declare_parameter<double>("command_timeout", 0.25);
    tracking_error_threshold_ =
        declare_parameter<double>("tracking_error_threshold", 1.2);
    tracking_error_hold_ = declare_parameter<double>("tracking_error_hold", 1.0);
    tracking_low_speed_threshold_ =
        declare_parameter<double>("tracking_low_speed_threshold", 0.25);
    tracking_severe_error_threshold_ =
        declare_parameter<double>("tracking_severe_error_threshold", 3.0);
    recovery_cooldown_ = declare_parameter<double>("recovery_cooldown", 2.0);
    recovery_brake_time_ = declare_parameter<double>("recovery_brake_time", 0.4);
    const std::string prefix = "/drone_" + std::to_string(drone_id_ - 1);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        prefix + "/odom", rclcpp::QoS(20).reliable(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
          odom_ = *message;
          have_odom_ = true;
        });
    command_sub_ = create_subscription<racer_fidelity_msgs::msg::PositionCommand>(
        prefix + "/position_cmd", rclcpp::QoS(20).reliable(),
        [this](racer_fidelity_msgs::msg::PositionCommand::ConstSharedPtr message) {
          command_ = *message;
          command_stamp_ = now();
          have_command_ = true;
        });
    output_pub_ = create_publisher<geometry_msgs::msg::Twist>(
        prefix + "/cmd_vel_3d", rclcpp::QoS(20).reliable());
    tracking_lost_pub_ = create_publisher<std_msgs::msg::Empty>(
        prefix + "/tracking_lost", rclcpp::QoS(10).reliable());
    timer_ = create_wall_timer(
        std::chrono::milliseconds(10), [this]() { publishCommand(); });
  }

 private:
  void publishCommand() {
    geometry_msgs::msg::Twist output;
    const auto current_time = now();
    if (!have_odom_ || !have_command_ ||
        (current_time - command_stamp_).seconds() > command_timeout_) {
      tracking_error_since_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      output_pub_->publish(output);
      return;
    }
    const double dx = command_.position.x - odom_.pose.pose.position.x;
    const double dy = command_.position.y - odom_.pose.pose.position.y;
    const double dz = command_.position.z - odom_.pose.pose.position.z;
    const double tracking_error = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double feedforward_speed = std::sqrt(
        command_.velocity.x * command_.velocity.x +
        command_.velocity.y * command_.velocity.y +
        command_.velocity.z * command_.velocity.z);
    const bool tracking_failure_candidate =
        tracking_error > tracking_error_threshold_ &&
        (feedforward_speed < tracking_low_speed_threshold_ ||
            tracking_error > tracking_severe_error_threshold_);
    if (tracking_error <= 0.5 * tracking_error_threshold_) {
      tracking_error_since_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      recovery_armed_ = true;
    } else if (tracking_failure_candidate && recovery_armed_) {
      if (tracking_error_since_.nanoseconds() == 0) {
        tracking_error_since_ = current_time;
      } else if ((current_time - tracking_error_since_).seconds() >=
                     tracking_error_hold_ &&
                 (last_recovery_.nanoseconds() == 0 ||
                     (current_time - last_recovery_).seconds() >= recovery_cooldown_)) {
        tracking_lost_pub_->publish(std_msgs::msg::Empty());
        last_recovery_ = current_time;
        tracking_error_since_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        recovery_armed_ = false;
        recovery_brake_until_ = current_time + rclcpp::Duration::from_seconds(
                                                   recovery_brake_time_);
        RCLCPP_WARN(get_logger(),
            "Execution tracking lost (error %.3f m, trajectory speed %.3f "
            "m/s); braking and requesting replan from measured odometry",
            tracking_error, feedforward_speed);
      }
    } else if (!tracking_failure_candidate) {
      tracking_error_since_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    }

    if (recovery_brake_until_.nanoseconds() != 0 &&
        current_time < recovery_brake_until_) {
      output.angular.z = command_.yaw;
      output_pub_->publish(output);
      return;
    }
    double values[3] = {
        command_.velocity.x + position_gain_ * dx,
        command_.velocity.y + position_gain_ * dy,
        command_.velocity.z + position_gain_ * dz};
    const double norm = std::sqrt(
        values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
    const double scale = norm > maximum_speed_ ? maximum_speed_ / norm : 1.0;
    output.linear.x = values[0] * scale;
    output.linear.y = values[1] * scale;
    output.linear.z = values[2] * scale;
    // The Isaac plant interprets angular.z as an absolute world-frame yaw
    // target. The value comes directly from the original yaw B-spline.
    output.angular.z = command_.yaw;
    output_pub_->publish(output);
  }

  int drone_id_{1};
  double position_gain_{2.5};
  double maximum_speed_{2.0};
  double command_timeout_{0.25};
  double tracking_error_threshold_{1.2};
  double tracking_error_hold_{1.0};
  double tracking_low_speed_threshold_{0.25};
  double tracking_severe_error_threshold_{3.0};
  double recovery_cooldown_{2.0};
  double recovery_brake_time_{0.4};
  bool have_odom_{false};
  bool have_command_{false};
  bool recovery_armed_{true};
  rclcpp::Time command_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tracking_error_since_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_recovery_{0, 0, RCL_ROS_TIME};
  rclcpp::Time recovery_brake_until_{0, 0, RCL_ROS_TIME};
  nav_msgs::msg::Odometry odom_;
  racer_fidelity_msgs::msg::PositionCommand command_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<racer_fidelity_msgs::msg::PositionCommand>::SharedPtr command_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr output_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr tracking_lost_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PositionCommandAdapter>());
  rclcpp::shutdown();
  return 0;
}
