#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/msg/string.hpp>

#include <Eigen/Eigen>
#include <memory>
#include <string>
#include <vector>

namespace fast_planner {
class SDFMap;

// ROS 2 / Isaac sensor boundary for the original SDFMap.  Occupancy fusion,
// inflation and ESDF computation remain in the unmodified upstream SDFMap.
class MapROS {
 public:
  MapROS();
  ~MapROS();
  void setMap(SDFMap *map);
  void init();
  void prepareReturnCorridor();

 private:
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &message);
  void odometryCallback(const nav_msgs::OdometryConstPtr &message);
  void updateESDFCallback(const ros::TimerEvent &event);
  void visCallback(const ros::TimerEvent &event);
  void coverageCallback(const ros::TimerEvent &event);
  void recordTraversedPosition(const Eigen::Vector3d &position);
  void clearCorridorSegment(const Eigen::Vector3d &start,
      const Eigen::Vector3d &finish, Eigen::Vector3d &update_min,
      Eigen::Vector3d &update_max);
  void finishCorridorUpdate(const Eigen::Vector3d &update_min,
      const Eigen::Vector3d &update_max, bool synchronous);

  SDFMap *map_{nullptr};
  ros::NodeHandle node_;
  ros::Subscriber cloud_sub_, odom_sub_;
  ros::Publisher coverage_pub_;
  ros::Timer esdf_timer_, vis_timer_, coverage_timer_;
  Eigen::Vector3d camera_pos_{Eigen::Vector3d::Zero()};
  bool have_odom_{false};
  bool return_corridor_enabled_{false};
  bool local_updated_{false};
  bool esdf_need_update_{false};
  std::string frame_id_{"map"};
  double traversed_clearance_radius_{0.0};
  double coverage_diagnostic_period_{2.0};
  pcl::PointCloud<pcl::PointXYZ> point_cloud_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      traversed_positions_;

  friend SDFMap;
};
}  // namespace fast_planner
