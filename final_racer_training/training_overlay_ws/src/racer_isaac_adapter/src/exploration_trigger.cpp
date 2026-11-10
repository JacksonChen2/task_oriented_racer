#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <memory>
#include <string>
#include <vector>

class ExplorationTrigger final : public rclcpp::Node {
 public:
  ExplorationTrigger() : Node("racer_original_exploration_trigger") {
    delay_ = declare_parameter<double>("delay", 5.0);
    repeats_ = declare_parameter<int>("repeats", 10);
    drone_count_ = declare_parameter<int>("drone_count", 5);
    minimum_cloud_frames_ = declare_parameter<int>("minimum_cloud_frames", 25);
    publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/racer/start", rclcpp::QoS(10).reliable());
    odom_ready_.assign(drone_count_, false);
    cloud_frames_.assign(drone_count_, 0);
    const auto qos = rclcpp::QoS(10).reliable();
    const auto cloud_qos = rclcpp::QoS(5).best_effort();
    for (int index = 0; index < drone_count_; ++index) {
      const auto prefix = std::string("/drone_") + std::to_string(index);
      odom_subscriptions_.push_back(create_subscription<nav_msgs::msg::Odometry>(
          prefix + "/odom", qos,
          [this, index](nav_msgs::msg::Odometry::ConstSharedPtr) {
            odom_ready_[index] = true;
          }));
      cloud_subscriptions_.push_back(create_subscription<sensor_msgs::msg::PointCloud2>(
          prefix + "/points", cloud_qos,
          [this, index](sensor_msgs::msg::PointCloud2::ConstSharedPtr) {
            if (cloud_frames_[index] < minimum_cloud_frames_) ++cloud_frames_[index];
          }));
    }
    timer_ = rclcpp::create_timer(
        this, get_clock(), rclcpp::Duration::from_seconds(0.5), [this]() {
          if (sent_ >= repeats_) return;
          int ready_count = 0;
          for (int index = 0; index < drone_count_; ++index) {
            if (odom_ready_[index] && cloud_frames_[index] >= minimum_cloud_frames_)
              ++ready_count;
          }
          if (ready_count != last_ready_count_) {
            RCLCPP_INFO(get_logger(),
                "sensor readiness %d/%d (minimum cloud frames per agent: %d)",
                ready_count, drone_count_, minimum_cloud_frames_);
            last_ready_count_ = ready_count;
          }
          if (now().seconds() < delay_ || ready_count < drone_count_ ||
              publisher_->get_subscription_count() < static_cast<size_t>(drone_count_))
            return;
          geometry_msgs::msg::PoseStamped message;
          message.header.stamp = now();
          message.header.frame_id = "world";
          message.pose.orientation.w = 1.0;
          publisher_->publish(message);
          ++sent_;
          RCLCPP_INFO(get_logger(), "published original RACER start trigger %d/%d",
                      sent_, repeats_);
          if (sent_ >= repeats_) {
            // Readiness topics are a launch barrier only; stop consuming their
            // data after the original trigger burst has been delivered.
            odom_subscriptions_.clear();
            cloud_subscriptions_.clear();
          }
        });
  }

 private:
  double delay_{5.0};
  int repeats_{10};
  int drone_count_{5};
  int minimum_cloud_frames_{25};
  int sent_{0};
  int last_ready_count_{-1};
  std::vector<bool> odom_ready_;
  std::vector<int> cloud_frames_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>
      odom_subscriptions_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr>
      cloud_subscriptions_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ExplorationTrigger>());
  rclcpp::shutdown();
  return 0;
}
