#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <cstdint>
#include <vector>
#include <bspline/Bspline.h>

using Eigen::Vector3d;
using std::vector;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;

  // Swarm collision avoidance
  bool avoid_collision_, go_back_;
  ros::Time fsm_init_time_;
  ros::Time last_check_frontier_time_;

  Eigen::Vector3d start_pos_;
};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second

  // Swarm
  double attempt_interval_;   // Min interval of opt attempt
  double pair_opt_interval_;  // Min interval of successful pair opt
  int repeat_send_num_;
  double pair_transaction_timeout_;
  double pair_retry_interval_;
  double assignment_commitment_duration_;
  double pair_min_improvement_;
  double pair_switch_penalty_;
  double swarm_state_freshness_;
  int work_steal_failure_threshold_;
  double work_steal_idle_delay_;
  double idle_map_wakeup_interval_;
};

struct DroneState {
  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  double yaw_;
  double stamp_;                // Stamp of pos,vel,yaw
  double recent_attempt_time_;  // Stamp of latest opt attempt with any drone

  vector<int> grid_ids_;         // Id of grid tour
  double recent_interact_time_;  // Stamp of latest opt with this drone
  uint64_t assignment_epoch_;    // Monotonic owner/version state for grid_ids_
  double commitment_until_;      // Avoid switching a healthy task too frequently
  bool requesting_work_;         // Empty/idle/failed UAV asks peers for work
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_;
  vector<double> yaws_;
  vector<Vector3d> frontier_tour_;
  vector<vector<Vector3d>> other_tours_;

  vector<int> refined_ids_;
  vector<vector<Vector3d>> n_points_;
  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  Vector3d next_goal_;
  vector<Vector3d> path_next_goal_, kino_path_;
  Vector3d next_pos_;
  double next_yaw_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;

  // Swarm, other drones' state
  vector<DroneState> swarm_state_;
  vector<double> pair_opt_stamps_, pair_opt_res_stamps_;
  vector<int> ego_ids_, other_ids_;
  double pair_opt_stamp_;
  bool reallocated_, wait_response_;

  // Coverage planning
  vector<Vector3d> grid_tour_, grid_tour2_;
  // int prev_first_id_;
  vector<int> last_grid_ids_;

  int plan_num_;
};

struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;   // resource dir of tsp solver
  string mtsp_dir_;  // resource dir of tsp solver
  double relax_time_;
  int init_plan_num_;

  // Swarm
  int drone_num_;
  int drone_id_;
  int random_seed_;
};

}  // namespace fast_planner

#endif
