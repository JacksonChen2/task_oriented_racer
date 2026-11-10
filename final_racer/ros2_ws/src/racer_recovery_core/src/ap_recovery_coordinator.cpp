#include <racer_recovery_core/msg/recovery_command.hpp>
#include <racer_recovery_core/msg/recovery_status.hpp>

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

namespace racer_recovery_core {

using RecoveryCommand = racer_recovery_core::msg::RecoveryCommand;
using RecoveryStatus = racer_recovery_core::msg::RecoveryStatus;

class ApRecoveryCoordinator final : public rclcpp::Node {
 public:
  ApRecoveryCoordinator() : Node("racer_ap_recovery_coordinator") {
    drone_count_ = declare_parameter<int>("drone_count", 5);
    status_timeout_s_ = declare_parameter<double>("status_timeout_s", 3.0);
    command_cooldown_s_ = declare_parameter<double>("command_cooldown_s", 2.0);
    grid_block_duration_s_ =
        declare_parameter<double>("grid_block_duration_s", 30.0);
    if (drone_count_ < 1 || status_timeout_s_ <= 0.0 ||
        command_cooldown_s_ < 0.0 || grid_block_duration_s_ <= 0.0) {
      throw std::runtime_error("invalid AP recovery coordinator parameters");
    }

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(100)).reliable();
    status_subscription_ = create_subscription<RecoveryStatus>(
        "/racer_ap_recovery/status_uplink", qos,
        [this](RecoveryStatus::ConstSharedPtr status) { onStatus(*status); });
    command_publisher_ = create_publisher<RecoveryCommand>(
        "/racer_ap_recovery/command_downlink", qos);

    RCLCPP_INFO(get_logger(),
                "AP recovery coordinator ready: drones=%d timeout=%.2fs "
                "grid_block=%.2fs",
                drone_count_, status_timeout_s_, grid_block_duration_s_);
  }

 private:
  struct AgentSnapshot {
    RecoveryStatus status;
    rclcpp::Time received_at{0, 0, RCL_ROS_TIME};
  };

  static double distance(const RecoveryStatus &left,
                         const RecoveryStatus &right) {
    const double dx = left.position.x - right.position.x;
    const double dy = left.position.y - right.position.y;
    const double dz = left.position.z - right.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  bool fresh(const AgentSnapshot &snapshot, const rclcpp::Time &now) const {
    return (now - snapshot.received_at).seconds() <= status_timeout_s_;
  }

  int selectPartner(const RecoveryStatus &failed,
                    const rclcpp::Time &now) const {
    int selected = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto &[drone_id, snapshot] : agents_) {
      if (drone_id == failed.drone_id || !fresh(snapshot, now) ||
          snapshot.status.phase == RecoveryStatus::PHASE_WAITING_FOR_AP ||
          snapshot.status.phase == RecoveryStatus::PHASE_HOLD ||
          snapshot.status.grid_ids.empty()) {
        continue;
      }
      const double candidate_distance = distance(failed, snapshot.status);
      if (candidate_distance < best_distance ||
          (candidate_distance == best_distance && drone_id < selected)) {
        best_distance = candidate_distance;
        selected = drone_id;
      }
    }
    return selected;
  }

  void onStatus(const RecoveryStatus &status) {
    if (status.drone_id < 1 || status.drone_id > drone_count_) {
      RCLCPP_WARN(get_logger(), "ignore status from invalid drone id %d",
                  status.drone_id);
      return;
    }
    const rclcpp::Time current_time = now();
    agents_.insert_or_assign(status.drone_id,
                             AgentSnapshot{status, current_time});
    if (!status.request_repartition) return;

    const auto command_key =
        (static_cast<std::uint64_t>(status.drone_id) << 32U) |
        static_cast<std::uint64_t>(status.episode_id);
    const auto previous = last_command_at_.find(command_key);
    if (previous != last_command_at_.end() &&
        (current_time - previous->second).seconds() < command_cooldown_s_) {
      return;
    }

    RecoveryCommand command;
    command.stamp = current_time;
    command.assignment_epoch = ++assignment_epoch_;
    command.episode_id = status.episode_id;
    command.drone_id = status.drone_id;
    command.partner_id = selectPartner(status, current_time);
    command.blocked_grid_id = status.failed_grid_id;
    command.blocked_until_s =
        current_time.seconds() + grid_block_duration_s_;
    command.action = command.partner_id > 0
                         ? RecoveryCommand::ACTION_REALLOCATE
                         : RecoveryCommand::ACTION_RETRY_LOCAL;
    command_publisher_->publish(command);
    last_command_at_.insert_or_assign(command_key, current_time);

    RCLCPP_WARN(get_logger(),
                "RACER_RECOVERY_AP episode=%u drone=%d action=%u partner=%d "
                "blocked_grid=%d epoch=%u",
                command.episode_id, command.drone_id, command.action,
                command.partner_id, command.blocked_grid_id,
                command.assignment_epoch);
  }

  int drone_count_{};
  double status_timeout_s_{};
  double command_cooldown_s_{};
  double grid_block_duration_s_{};
  std::uint32_t assignment_epoch_{};
  std::unordered_map<int, AgentSnapshot> agents_;
  std::unordered_map<std::uint64_t, rclcpp::Time> last_command_at_;
  rclcpp::Subscription<RecoveryStatus>::SharedPtr status_subscription_;
  rclcpp::Publisher<RecoveryCommand>::SharedPtr command_publisher_;
};

}  // namespace racer_recovery_core

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<racer_recovery_core::ApRecoveryCoordinator>());
  rclcpp::shutdown();
  return 0;
}
