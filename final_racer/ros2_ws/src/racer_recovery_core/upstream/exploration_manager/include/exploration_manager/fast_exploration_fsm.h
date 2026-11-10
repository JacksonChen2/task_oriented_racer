#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <exploration_manager/DroneState.h>
#include <exploration_manager/PairOpt.h>
#include <exploration_manager/PairOptResponse.h>
#include <bspline/Bspline.h>
#include <racer_recovery_core/msg/recovery_command.hpp>
#include <racer_recovery_core/msg/recovery_status.hpp>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <thread>

using Eigen::Vector3d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace fast_planner {
class FastPlannerManager;
class FastExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum EXPL_STATE { INIT, WAIT_TRIGGER, PLAN_TRAJ, PUB_TRAJ, EXEC_TRAJ, FINISH, IDLE };

class FastExplorationFSM {

public:
  FastExplorationFSM(/* args */) {
  }
  ~FastExplorationFSM() {
  }

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* helper functions */
  int callExplorationPlanner();
  void transitState(EXPL_STATE new_state, string pos_call);
  void visualize(int content);
  void clearVisMarker();
  int getId();
  void findUnallocated(const vector<int>& actives, vector<int>& missed);
  void resetFailureWindow();
  void handlePlanFailure();
  void startLocalReselect();
  bool startLocalEscape();
  void requestApRepartition();
  void completeRecovery(const string& reason);
  void enforceRecoveryLease();
  bool selectEscapeTarget(Eigen::Vector3d& target) const;
  void publishRecoveryStatus(bool request_repartition = false);
  void recoveryStatusTimerCallback(const ros::TimerEvent& e);
  void recoveryCommandCallback(
      const racer_recovery_core::msg::RecoveryCommand::ConstSharedPtr& msg);

  /* ROS functions */
  void FSMCallback(const ros::TimerEvent& e);
  void safetyCallback(const ros::TimerEvent& e);
  void trackingLostCallback(const std_msgs::EmptyConstPtr& msg);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);

  // Swarm
  void droneStateTimerCallback(const ros::TimerEvent& e);
  void droneStateMsgCallback(const exploration_manager::DroneStateConstPtr& msg);
  void optTimerCallback(const ros::TimerEvent& e);
  void optMsgCallback(const exploration_manager::PairOptConstPtr& msg);
  void optResMsgCallback(const exploration_manager::PairOptResponseConstPtr& msg);
  void swarmTrajCallback(const bspline::BsplineConstPtr& msg);
  void swarmTrajTimerCallback(const ros::TimerEvent& e);

  /* planning utils */
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FastExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  EXPL_STATE state_;

  /* ROS utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, safety_timer_, vis_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, tracking_lost_sub_;
  ros::Subscriber recovery_command_sub_;
  ros::Publisher replan_pub_, new_pub_, bspline_pub_, recovery_status_pub_;
  ros::Timer recovery_status_timer_;

  // Swarm state
  ros::Publisher drone_state_pub_, opt_pub_, opt_res_pub_, swarm_traj_pub_, grid_tour_pub_,
      hgrid_pub_;
  ros::Subscriber drone_state_sub_, opt_sub_, opt_res_sub_, swarm_traj_sub_;
  ros::Timer drone_state_timer_, opt_timer_, swarm_traj_timer_;

  // Recovery state is deliberately kept outside ExplorationData so the
  // original frontier and trajectory data structures remain unchanged.
  bool failure_window_active_{false};
  bool recovery_reselect_active_{false};
  bool recovery_escape_active_{false};
  bool waiting_for_ap_{false};
  bool waiting_for_pair_response_{false};
  int recovery_stage_{0};
  int recovery_attempts_{0};
  int consecutive_plan_failures_{0};
  int failed_grid_id_{-1};
  int forced_partner_id_{-1};
  int forced_blocked_grid_id_{-1};
  double forced_blocked_until_s_{0.0};
  int leased_grid_id_{-1};
  int leased_partner_id_{-1};
  double leased_until_s_{0.0};
  std::uint32_t recovery_episode_id_{0};
  std::uint32_t last_assignment_epoch_{0};
  ros::Time failure_started_at_;
  ros::Time next_plan_retry_at_;
  ros::Time ap_request_started_at_;
  Eigen::Vector3d failure_origin_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d recovery_origin_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d recovery_escape_target_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d last_failed_goal_{Eigen::Vector3d::Zero()};
  std::deque<Eigen::Vector3d> safe_position_history_;
};

}  // namespace fast_planner

#endif
