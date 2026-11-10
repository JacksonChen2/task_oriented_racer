
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <exploration_manager/HGrid.h>
#include <exploration_manager/GridTour.h>

#include <plan_env/edt_environment.h>
#include <plan_env/map_ros.h>
#include <plan_env/sdf_map.h>
#include <plan_env/multi_map_manager.h>
#include <active_perception/perception_utils.h>
#include <active_perception/hgrid.h>
#include <hybrid_communication_racer/assignment.hpp>
// #include <active_perception/uniform_grid.h>
// #include <lkh_tsp_solver/lkh_interface.h>
// #include <lkh_mtsp_solver/lkh3_interface.h>

#include <fstream>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <sstream>

using Eigen::Vector4d;

namespace fast_planner {
namespace {
string formatIdList(const vector<int>& ids) {
  std::ostringstream stream;
  stream << "[";
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) stream << ",";
    stream << ids[index];
  }
  stream << "]";
  return stream.str();
}
}  // namespace

void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("fsm/attempt_interval", fp_->attempt_interval_, 0.2);
  nh.param("fsm/pair_opt_interval", fp_->pair_opt_interval_, 1.0);
  nh.param("fsm/repeat_send_num", fp_->repeat_send_num_, 10);
  nh.param("fsm/pair_transaction_timeout", fp_->pair_transaction_timeout_, 1.5);
  nh.param("fsm/pair_retry_interval", fp_->pair_retry_interval_, 0.15);
  nh.param("fsm/assignment_commitment_duration", fp_->assignment_commitment_duration_, 3.0);
  nh.param("fsm/pair_min_improvement", fp_->pair_min_improvement_, 1.0);
  nh.param("fsm/pair_switch_penalty", fp_->pair_switch_penalty_, 1.5);
  nh.param("fsm/swarm_state_freshness", fp_->swarm_state_freshness_, 0.5);
  nh.param("fsm/work_steal_failure_threshold", fp_->work_steal_failure_threshold_, 5);
  nh.param("fsm/work_steal_idle_delay", fp_->work_steal_idle_delay_, 0.5);
  nh.param("fsm/idle_map_wakeup_interval", fp_->idle_map_wakeup_interval_, 0.5);
  nh.param("fsm/initial_partition_enabled", initial_partition_enabled_, false);
  nh.param("fsm/initial_partition_interval", initial_partition_interval_s_, 0.5);
  nh.param("fsm/initial_partition_balance_weight", initial_partition_balance_weight_m_, 2.0);
  nh.param("fsm/bs_event_global_retry_interval", bs_event_global_retry_interval_s_, 0.5);
  nh.param("fsm/bs_event_global_state_freshness", bs_event_global_state_freshness_s_, 2.0);
  nh.param("fsm/global_recovery_pair_fail_threshold",
      global_recovery_pair_fail_threshold_, 4);
  nh.param("fsm/global_recovery_cooldown_cycles",
      global_recovery_cooldown_cycles_, 5);
  nh.param("fsm/global_recovery_coalesce_cycles",
      global_recovery_coalesce_cycles_, 2);
  nh.param("fsm/global_recovery_max_tasks_per_idle",
      global_recovery_max_tasks_per_idle_, 1);
  nh.param("fsm/global_finish_confirmation_cycles",
      global_finish_confirmation_cycles_, 3);
  task_coordination_gate_.configure(global_recovery_pair_fail_threshold_,
      global_recovery_cooldown_cycles_);
  nh.param("fsm/local_component_neighbor_freshness",
      local_component_neighbor_freshness_s_, 0.5);
  nh.param("fsm/local_component_stability_duration",
      local_component_stability_duration_s_, 1.0);
  nh.param("fsm/local_component_assignment_interval",
      local_component_assignment_interval_s_, 0.5);
  nh.param("exploration_assignment_mode", exploration_assignment_mode_, string("original"));
  nh.param("global_cooperative.assignment_interval", cooperative_assignment_interval_s_, 0.5);
  nh.param("global_cooperative.state_freshness", cooperative_state_freshness_s_, 0.75);
  nh.param("global_cooperative.nominal_velocity", nominal_velocity_, 1.5);
  nh.param("global_cooperative.gain_scale", utility_gain_scale_, 0.001);
  nh.param("global_cooperative.utility_epsilon", utility_epsilon_, 0.1);
  nh.param("global_cooperative.lambda_overlap", lambda_overlap_, 2.0);
  nh.param("global_cooperative.lambda_age", lambda_age_, 0.05);
  nh.param("global_cooperative.overlap_distance", overlap_distance_, 6.0);
  nh.param("global_cooperative.cluster_distance", cluster_distance_, 3.0);
  nh.param("global_cooperative.distance_min", distance_min_, 8.0);
  nh.param("global_cooperative.distance_max", distance_max_, 45.0);
  nh.param("global_cooperative.distance_gamma", distance_gamma_, 0.7);
  nh.param("global_cooperative.distance_fallback_when_idle",
      distance_fallback_when_idle_, true);
  nh.param("global_cooperative.finish_confirmation_cycles",
      finish_confirmation_cycles_, 6);
  nh.param("global_cooperative.local_failures_before_report",
      local_failures_before_report_, 2);
  nh.param("global_cooperative.unreachable_distinct_uavs",
      unreachable_distinct_uavs_, 2);
  nh.param("global_cooperative.unreachable_global_failures",
      unreachable_global_failures_, 4);
  nh.param("global_cooperative.unreachable_retry_s", unreachable_retry_s_, 20.0);
  if (exploration_assignment_mode_ != "original" &&
      exploration_assignment_mode_ != "local_component" &&
      exploration_assignment_mode_ != "bs_event_global" &&
      exploration_assignment_mode_ != "global_cooperative") {
    ROS_ERROR("Unknown exploration_assignment_mode '%s'; falling back to original.",
        exploration_assignment_mode_.c_str());
    exploration_assignment_mode_ = "original";
  }

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  expl_manager_->setUnassignedFrontierFallback(bsEventGlobalMode());
  visualization_.reset(new PlanningVisualization(nh));

  planner_manager_ = expl_manager_->planner_manager_;
  const int drone_num = expl_manager_->ep_->drone_num_;
  last_reported_assignment_epochs_.assign(drone_num, 0);
  hybrid_states_.resize(drone_num);
  last_failure_sequences_.assign(drone_num, 0);
  coverage_contributions_.assign(drone_num, 0.0);
  local_component_members_ = { getId() };
  local_component_coordinator_id_ = getId();
  local_component_last_state_received_s_.assign(drone_num, -1.0);
  local_component_peer_heard_ids_.resize(drone_num);
  local_component_peer_member_ids_.resize(drone_num);
  local_component_peer_ack_ids_.resize(drone_num);
  local_component_peer_coordinator_ids_.assign(drone_num, -1);
  local_component_peer_assignment_epochs_.assign(drone_num, 0);
  local_component_positions_.assign(drone_num, Eigen::Vector3d::Zero());
  local_component_position_valid_.assign(drone_num, 0);
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH", "IDL"
                                                                                              "E" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  fd_->avoid_collision_ = false;
  fd_->go_back_ = false;

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.5), &FastExplorationFSM::frontierCallback, this);

  trigger_sub_ =
      nh.subscribe("/move_base_simple/goal", 1, &FastExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);
  // ROS2/Isaac execution boundary only: the physical vehicle can be stopped by
  // the independent safety layer while the original ideal tracker advances.
  // Recover the original collision-replan semantics from measured odometry;
  // do not select a new frontier, task, viewpoint, or goal here.
  tracking_lost_sub_ =
      nh.subscribe("/racer/tracking_lost", 1, &FastExplorationFSM::trackingLostCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);

  // Swarm, timer, pub and sub
  drone_state_timer_ =
      nh.createTimer(ros::Duration(0.04), &FastExplorationFSM::droneStateTimerCallback, this);
  drone_state_pub_ =
      nh.advertise<exploration_manager::DroneState>("/swarm_expl/drone_state_send", 10);
  drone_state_sub_ = nh.subscribe(
      "/swarm_expl/drone_state_recv", 10, &FastExplorationFSM::droneStateMsgCallback, this);

  opt_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::optTimerCallback, this);
  opt_pub_ = nh.advertise<exploration_manager::PairOpt>("/swarm_expl/pair_opt_send", 10);
  opt_sub_ = nh.subscribe("/swarm_expl/pair_opt_recv", 100, &FastExplorationFSM::optMsgCallback,
      this, ros::TransportHints().tcpNoDelay());

  opt_res_pub_ =
      nh.advertise<exploration_manager::PairOptResponse>("/swarm_expl/pair_opt_res_send", 10);
  opt_res_sub_ = nh.subscribe("/swarm_expl/pair_opt_res_recv", 100,
      &FastExplorationFSM::optResMsgCallback, this, ros::TransportHints().tcpNoDelay());

  global_assignment_pub_ = nh.advertise<exploration_manager::GlobalGridAssignment>(
      "/swarm_expl/global_assignment_send", 10, true);
  global_assignment_sub_ = nh.subscribe(
      "/swarm_expl/global_assignment_recv", 10,
      &FastExplorationFSM::globalAssignmentMsgCallback, this,
      ros::TransportHints().tcpNoDelay());

  if (globalCooperativeMode()) {
    // These absolute topics intentionally bypass the radio-loss proxy: this
    // algorithm is defined only for perfect communication/global information.
    hybrid_state_pub_ = nh.advertise<hybrid_communication_racer::msg::HybridDroneState>(
        "/hybrid_communication_racer/drone_state", 50);
    hybrid_state_sub_ = nh.subscribe(
        "/hybrid_communication_racer/drone_state", 100,
        &FastExplorationFSM::hybridStateMsgCallback, this,
        ros::TransportHints().tcpNoDelay());
    hybrid_assignment_pub_ =
        nh.advertise<hybrid_communication_racer::msg::HybridGlobalAssignment>(
            "/hybrid_communication_racer/global_assignment", 10, true);
    hybrid_assignment_sub_ = nh.subscribe(
        "/hybrid_communication_racer/global_assignment", 20,
        &FastExplorationFSM::hybridAssignmentMsgCallback, this,
        ros::TransportHints().tcpNoDelay());
  }

#ifdef RACER_ORACLE_VARIANT
  nh.param("oracle_global/assignment_interval", global_assignment_interval_s_, 0.5);
  nh.param("oracle_global/balance_weight_m", global_balance_weight_m_, 1.0);
  ROS_WARN("RACER_ORACLE_GLOBAL_COORDINATION enabled coordinator=1 interval=%.3fs balance=%.3fm",
      global_assignment_interval_s_, global_balance_weight_m_);
#else
  ROS_WARN("RACER_PAIRWISE_ROBUST enabled epoch transactions, initial partition, commitment, "
           "work stealing and known-free connectivity filtering");
#endif

  const char* blocking_start = std::getenv("RACER_BLOCKING_INITIAL_ASSIGNMENT");
  blocking_initial_assignment_ = exploration_assignment_mode_ == "original" &&
      blocking_start && string(blocking_start) == "true";
  if (blocking_initial_assignment_) {
    initial_partition_enabled_ = true;
    ROS_WARN("RACER_BLOCKING_INITIAL_ENABLED uav=%d coordinator=1 timeout=none", getId());
  }

  if (globalCooperativeMode()) {
    initial_partition_enabled_ = false;
    ROS_WARN("RACER_HYBRID_ENABLED mode=global_cooperative coordinator=1 perfect_comm_required "
             "interval=%.2f R=[%.1f,%.1f] gamma=%.2f cluster=%.1f overlap=%.1f",
        cooperative_assignment_interval_s_, distance_min_, distance_max_,
        distance_gamma_, cluster_distance_, overlap_distance_);
  } else if (localComponentMode()) {
    initial_partition_enabled_ = false;
    ROS_WARN("RACER_LOCAL_COMPONENT_ENABLED normal_sionna_only neighbor_freshness=%.2f "
             "stability=%.2f assignment_interval=%.2f",
        local_component_neighbor_freshness_s_, local_component_stability_duration_s_,
        local_component_assignment_interval_s_);
  } else if (bsEventGlobalMode()) {
    ROS_WARN("RACER_BS_EVENT_GLOBAL_ENABLED pairwise_normal_mode=enabled "
             "initial_partition=%s recovery_after_failures=%d cooldown_cycles=%d "
             "coalesce_cycles=%d retry_interval=%.2fs state_freshness=%.2fs",
        initial_partition_enabled_ ? "enabled" : "disabled",
        global_recovery_pair_fail_threshold_, global_recovery_cooldown_cycles_,
        global_recovery_coalesce_cycles_, bs_event_global_retry_interval_s_,
        bs_event_global_state_freshness_s_);
  } else {
    ROS_WARN("RACER_PAIRWISE_GLOBAL_RECOVERY mode=original initial_partition=%s "
             "recovery_after_failures=%d cooldown_cycles=%d coalesce_cycles=%d",
        initial_partition_enabled_ ? "enabled" : "disabled",
        global_recovery_pair_fail_threshold_, global_recovery_cooldown_cycles_,
        global_recovery_coalesce_cycles_);
  }

  if (pairwiseGlobalRecoveryMode())
    ROS_WARN("[TASK MODE] uav=%d NORMAL reason=initialized", getId());

  swarm_traj_pub_ = nh.advertise<bspline::Bspline>("/planning/swarm_traj_send", 100);
  swarm_traj_sub_ =
      nh.subscribe("/planning/swarm_traj_recv", 100, &FastExplorationFSM::swarmTrajCallback, this);
  swarm_traj_timer_ =
      nh.createTimer(ros::Duration(0.1), &FastExplorationFSM::swarmTrajTimerCallback, this);

  hgrid_pub_ = nh.advertise<exploration_manager::HGrid>("/swarm_expl/hgrid_send", 10);
  grid_tour_pub_ = nh.advertise<exploration_manager::GridTour>("/swarm_expl/grid_tour_send", 10);
}

int FastExplorationFSM::getId() {
  return expl_manager_->ep_->drone_id_;
}

bool FastExplorationFSM::globalCooperativeMode() const {
  return exploration_assignment_mode_ == "global_cooperative";
}

bool FastExplorationFSM::localComponentMode() const {
  return exploration_assignment_mode_ == "local_component";
}

bool FastExplorationFSM::bsEventGlobalMode() const {
  return exploration_assignment_mode_ == "bs_event_global";
}

bool FastExplorationFSM::pairwiseGlobalRecoveryMode() const {
#ifdef RACER_ORACLE_VARIANT
  return false;
#else
  return !globalCooperativeMode() && !localComponentMode();
#endif
}

void FastExplorationFSM::logTaskMode(
    const char* reason, TaskCoordinationMode previous_mode) {
  if (previous_mode == task_coordination_gate_.mode()) return;
  ROS_WARN("[TASK MODE] uav=%d %s -> %s reason=%s", getId(),
      taskCoordinationModeName(previous_mode),
      taskCoordinationModeName(task_coordination_gate_.mode()), reason);
}

void FastExplorationFSM::enterPairSearch(const char* reason) {
  if (!pairwiseGlobalRecoveryMode()) return;
  const auto previous_mode = task_coordination_gate_.mode();
  task_coordination_gate_.localTaskEmpty();
  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  // Empty ownership itself already makes pair optimization urgent. Keep this
  // flag false until the failure gate explicitly asks the coordinator for
  // global recovery.
  ego.requesting_work_ =
      task_coordination_gate_.mode() == TaskCoordinationMode::GLOBAL_RECOVERY;
  if (idle_since_s_ < 0.0) idle_since_s_ = ros::Time::now().toSec();
  fd_->last_check_frontier_time_ = ros::Time::now();
  fd_->static_state_ = true;
  fd_->go_back_ = false;
  if (state_ != IDLE) transitState(IDLE, "pairSearch");
  logTaskMode(reason, previous_mode);
}

void FastExplorationFSM::recordPairSearchFailure(
    int candidate_id, const char* result) {
  if (!pairwiseGlobalRecoveryMode()) return;
  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  if (!ego.grid_ids_.empty()) return;
  const auto previous_mode = task_coordination_gate_.mode();
  const bool request_global = task_coordination_gate_.pairAttemptFailed();
  ROS_WARN("[PAIR SEARCH] uav=%d attempt=%d candidate=%d result=%s failure_count=%d",
      getId(), task_coordination_gate_.pairAttemptCount(), candidate_id, result,
      task_coordination_gate_.pairFailureCount());
  if (request_global && !outgoing_transaction_.active &&
      !incoming_transaction_.active) {
    ego.requesting_work_ = true;
    ROS_WARN("[GLOBAL RECOVERY REQUEST] uav=%d pair_failure_count=%d "
             "local_task_count=%zu",
        getId(), task_coordination_gate_.pairFailureCount(), ego.grid_ids_.size());
  }
  logTaskMode(result, previous_mode);
}

void FastExplorationFSM::markPairSearchStarted(
    int candidate_id, const char* result) {
  if (!pairwiseGlobalRecoveryMode()) return;
  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  if (!ego.grid_ids_.empty()) return;
  const auto previous_mode = task_coordination_gate_.mode();
  task_coordination_gate_.pairAttemptStarted();
  ROS_WARN("[PAIR SEARCH] uav=%d attempt=%d candidate=%d result=%s failure_count=%d",
      getId(), task_coordination_gate_.pairAttemptCount(), candidate_id, result,
      task_coordination_gate_.pairFailureCount());
  logTaskMode(result, previous_mode);
}

void FastExplorationFSM::markTaskNormal(const char* reason) {
  if (!pairwiseGlobalRecoveryMode()) return;
  const auto previous_mode = task_coordination_gate_.mode();
  task_coordination_gate_.localTaskAvailable();
  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  ego.requesting_work_ = false;
  idle_since_s_ = -1.0;
  logTaskMode(reason, previous_mode);
}

void FastExplorationFSM::enterHelpMode(const char* reason, bool clear_assignment) {
  if (!globalCooperativeMode()) return;
  const bool new_help_episode = state_ != IDLE || idle_since_s_ < 0.0;
  if (clear_assignment) releaseAssignmentForWork(reason);
  primary_grid_id_ = -1;
  primary_target_.setZero();
  target_utility_ = 0.0;
  target_information_gain_ = 0.0;
  target_travel_time_ = 0.0;
  target_overlap_penalty_ = 0.0;
  current_target_takeover_ = false;
  if (new_help_episode) ++help_mode_count_;
  idle_since_s_ = ros::Time::now().toSec();
  fd_->last_check_frontier_time_ = ros::Time::now();
  fd_->static_state_ = true;
  fd_->go_back_ = false;
  replan_pub_.publish(std_msgs::Empty());
  if (state_ != IDLE) transitState(IDLE, "hybridHelp");
  ROS_WARN("RACER_HYBRID_HELP uav=%d reason=%s count=%lu", getId(), reason,
      static_cast<unsigned long>(help_mode_count_));
}

void FastExplorationFSM::reportUnreachableTask(const char* reason) {
  if (!globalCooperativeMode() || primary_grid_id_ < 0) {
    enterHelpMode(reason);
    return;
  }
  failed_grid_id_ = primary_grid_id_;
  ++failure_sequence_;
  ROS_WARN("RACER_HYBRID_UNREACHABLE_REPORT uav=%d grid=%d sequence=%lu failures=%d",
      getId(), failed_grid_id_, static_cast<unsigned long>(failure_sequence_),
      consecutive_plan_failures_);
  enterHelpMode(reason);
}

bool FastExplorationFSM::validateGridIds(const vector<int>& ids, const char* source) const {
  for (const int id : ids) {
    if (!expl_manager_->hgrid_->isValidGridId(id)) {
      ROS_ERROR("Reject %s with invalid grid id %d (valid range [0,%d)).", source, id,
          expl_manager_->hgrid_->getGridCount());
      return false;
    }
  }
  return true;
}

void FastExplorationFSM::publishPairTransaction(uint8_t phase) {
  if (!outgoing_transaction_.active) return;

  exploration_manager::PairOpt msg;
  msg.from_drone_id = getId();
  msg.to_drone_id = outgoing_transaction_.peer_id;
  msg.phase = phase;
  msg.transaction_id = outgoing_transaction_.transaction_id;
  msg.from_assignment_epoch = outgoing_transaction_.from_epoch;
  msg.to_assignment_epoch = outgoing_transaction_.to_epoch;
  msg.new_assignment_epoch = outgoing_transaction_.new_epoch;
  msg.commitment_until = outgoing_transaction_.commitment_until;
  msg.stamp = ros::Time::now().toSec();
  msg.ego_ids = outgoing_transaction_.ego_ids;
  msg.other_ids = outgoing_transaction_.other_ids;
  for (int i = 0; i < std::max(1, fp_->repeat_send_num_); ++i) opt_pub_.publish(msg);
  outgoing_transaction_.last_publish_at = msg.stamp;
}

void FastExplorationFSM::clearOutgoingTransaction(const char* reason, bool pair_failed) {
  const int failed_peer = outgoing_transaction_.peer_id;
  if (outgoing_transaction_.active)
    ROS_WARN("Pair transaction %lu with UAV %d cleared: %s",
        static_cast<unsigned long>(outgoing_transaction_.transaction_id),
        outgoing_transaction_.peer_id, reason);
  outgoing_transaction_ = PairTransaction();
  expl_manager_->ed_->wait_response_ = false;
  if (pair_failed)
    recordPairSearchFailure(failed_peer, reason);
}

void FastExplorationFSM::servicePairTransactions(double now_s) {
  if (incoming_transaction_.active && now_s > incoming_transaction_.expires_at) {
    ROS_WARN("Prepared pair transaction %lu expired without commit.",
        static_cast<unsigned long>(incoming_transaction_.transaction_id));
    incoming_transaction_ = PairTransaction();
  }

  if (!outgoing_transaction_.active) return;
  // Before PREPARED it is safe to abandon a proposal. Once COMMIT has been sent, the responder
  // may already have applied it; aborting locally would create split ownership. Keep retrying the
  // idempotent COMMIT until either its ACK or the responder's DroneState confirms the epoch.
  if (!outgoing_transaction_.commit_sent && now_s > outgoing_transaction_.expires_at) {
    clearOutgoingTransaction("timeout", true);
    return;
  }
  if (outgoing_transaction_.last_publish_at < 0.0 ||
      now_s - outgoing_transaction_.last_publish_at >= fp_->pair_retry_interval_) {
    publishPairTransaction(outgoing_transaction_.commit_sent
            ? exploration_manager::PairOpt::PHASE_COMMIT
            : exploration_manager::PairOpt::PHASE_PROPOSE);
  }
}

void FastExplorationFSM::applyCommittedPairAssignment(int peer_id, const vector<int>& ego_ids,
    const vector<int>& peer_ids, uint64_t new_epoch, double commitment_until) {
  if (peer_id <= 0 || peer_id > static_cast<int>(expl_manager_->ed_->swarm_state_.size()) ||
      !validateGridIds(ego_ids, "committed ego assignment") ||
      !validateGridIds(peer_ids, "committed peer assignment"))
    return;

  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  auto& peer = expl_manager_->ed_->swarm_state_[peer_id - 1];
  ego.grid_ids_ = ego_ids;
  peer.grid_ids_ = peer_ids;
  ego.assignment_epoch_ = new_epoch;
  peer.assignment_epoch_ = new_epoch;
  ego.commitment_until_ = commitment_until;
  peer.commitment_until_ = commitment_until;
  ego.requesting_work_ = ego.grid_ids_.empty();
  peer.requesting_work_ = peer.grid_ids_.empty();
  ego.recent_interact_time_ = ros::Time::now().toSec();
  peer.recent_interact_time_ = ego.recent_interact_time_;
  expl_manager_->ed_->reallocated_ = true;

  if (!ego.grid_ids_.empty()) markTaskNormal("pair committed");
  if ((state_ == IDLE || state_ == FINISH) && !ego.grid_ids_.empty()) {
    fd_->go_back_ = false;
    consecutive_plan_failures_ = 0;
    transitState(PLAN_TRAJ, "pairCommit");
    ROS_WARN("UAV %d resumed exploration after committed work transfer.", getId());
  }
}

void FastExplorationFSM::releaseAssignmentForWork(const char* reason) {
  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  if (outgoing_transaction_.active || incoming_transaction_.active) {
    // A prepared transaction can still be safely abandoned by its initiator, but an uncertain
    // COMMIT must finish before any unilateral assignment change.
    if (outgoing_transaction_.active && !outgoing_transaction_.commit_sent) {
      publishPairTransaction(exploration_manager::PairOpt::PHASE_ABORT);
      clearOutgoingTransaction("local work release before commit");
    } else {
      ego.requesting_work_ = true;
      ROS_WARN_THROTTLE(1.0,
          "UAV %d defers work release until pair transaction %lu is resolved.", getId(),
          static_cast<unsigned long>(outgoing_transaction_.active
                  ? outgoing_transaction_.transaction_id
                  : incoming_transaction_.transaction_id));
      return;
    }
  }
  if (!ego.grid_ids_.empty()) {
    ROS_WARN("UAV %d releases %zu grids for work stealing after %s.",
        getId(), ego.grid_ids_.size(), reason);
    ego.grid_ids_.clear();
    ++ego.assignment_epoch_;
  }
  ego.commitment_until_ = 0.0;
  ego.requesting_work_ = true;
  ego.recent_attempt_time_ = 0.0;
  expl_manager_->ed_->reallocated_ = true;
}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(
      1.0, "[FSM]: Drone " << getId() << " state: " << fd_->state_str_[int(state_)]);

  switch (state_) {
    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom");
        return;
      }
      if ((ros::Time::now() - fd_->fsm_init_time_).toSec() < 2.0) {
        ROS_WARN_THROTTLE(1.0, "wait for init");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      break;
    }

    case FINISH: {
      if (globalCooperativeMode() && !cooperative_global_finish_) {
        enterHelpMode("global pool not finished");
        break;
      }
      if (pairwiseGlobalRecoveryMode() &&
          task_coordination_gate_.mode() != TaskCoordinationMode::FINISH) {
        enterPairSearch("local finish rejected before global confirmation");
        break;
      }
      ROS_INFO_THROTTLE(1.0, "finish exploration.");
      break;
    }

    case IDLE: {
      if (waitingForInitialAssignment()) break;
      if (globalCooperativeMode()) {
        expl_manager_->ed_->swarm_state_[getId() - 1].requesting_work_ = true;
        if (cooperative_global_finish_) transitState(FINISH, "hybridGlobalFinish");
        break;
      }
      // In local-component mode no UAV may independently claim a grid while
      // its component-scoped startup assignment is still in flight.
      if (localComponentMode() && !local_component_assignment_applied_) break;
      // A viewpoint that was disconnected while the local map was small can
      // become reachable after peer map chunks arrive.  Rebuild the frontier
      // structure at a bounded rate and wake the normal RACER planner without
      // fabricating free voxels or changing the assigned task.
      const double now_s = ros::Time::now().toSec();
      if (last_idle_map_wakeup_check_s_ < 0.0 ||
          now_s - last_idle_map_wakeup_check_s_ >= fp_->idle_map_wakeup_interval_) {
        last_idle_map_wakeup_check_s_ = now_s;
        if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
          fd_->go_back_ = false;
          consecutive_plan_failures_ = 0;
          ROS_WARN("UAV %d wakes from IDLE after shared-map frontier expansion.", getId());
          transitState(PLAN_TRAJ, "sharedMapWakeup");
          break;
        }
      }
      double check_interval = (ros::Time::now() - fd_->last_check_frontier_time_).toSec();
      if (check_interval > 100.0) {
        // if (!expl_manager_->updateFrontierStruct(fd_->odom_pos_)) {
        ROS_WARN("Go back to (0,0,1)");
        // if (getId() == 1) {
        //   expl_manager_->ed_->next_pos_ = Eigen::Vector3d(-3, 1.9, 1);
        // } else
        // expl_manager_->ed_->next_pos_ = Eigen::Vector3d(0, 0.0, 1);
        // Eigen::Vector3d dir = (fd_->start_pos_ - fd_->odom_pos_);
        // expl_manager_->ed_->next_yaw_ = atan2(dir[1], dir[0]);

        expl_manager_->ed_->next_pos_ = fd_->start_pos_;
        expl_manager_->ed_->next_yaw_ = 0.0;

        fd_->go_back_ = true;
        transitState(PLAN_TRAJ, "FSM");
        // } else {
        //   fd_->last_check_frontier_time_ = ros::Time::now();
        // }
      }
      break;
    }

    case PLAN_TRAJ: {
      if (fd_->static_state_) {
        // Plan from static state (hover)
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();
        fd_->start_yaw_ << fd_->odom_yaw_, 0, 0;
      } else {
        // Replan from non-static state, starting from 'replan_time' seconds later
        LocalTrajData* info = &planner_manager_->local_data_;
        double t_r = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;
        fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
        fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
        fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
        fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
      }
      // Inform traj_server the replanning
      replan_pub_.publish(std_msgs::Empty());
      int res = callExplorationPlanner();
      if (res == SUCCEED) {
        consecutive_plan_failures_ = 0;
        expl_manager_->ed_->swarm_state_[getId() - 1].requesting_work_ = false;
        markTaskNormal("planner has a valid task");
        transitState(PUB_TRAJ, "FSM");
      } else if (res == FAIL) {  // Keep trying to replan
        fd_->static_state_ = true;
        ++consecutive_plan_failures_;
        ROS_WARN("Plan fail");
        if (globalCooperativeMode() &&
            consecutive_plan_failures_ >= local_failures_before_report_) {
          reportUnreachableTask("consecutive plan failures");
        } else if (!globalCooperativeMode() &&
            consecutive_plan_failures_ >= fp_->work_steal_failure_threshold_) {
            releaseAssignmentForWork("consecutive plan failures");
            if (pairwiseGlobalRecoveryMode())
              enterPairSearch("planner failures exhausted local task");
            else {
              fd_->last_check_frontier_time_ = ros::Time::now();
              idle_since_s_ = ros::Time::now().toSec();
              transitState(IDLE, "workSteal");
            }
        }
      } else if (res == NO_GRID) {
        fd_->static_state_ = true;
        fd_->last_check_frontier_time_ = ros::Time::now();
        idle_since_s_ = ros::Time::now().toSec();
        ROS_WARN("No grid");
        if (globalCooperativeMode())
          enterHelpMode("No grid");
        else if (pairwiseGlobalRecoveryMode())
          enterPairSearch("no assigned grid");
        else {
          expl_manager_->ed_->swarm_state_[getId() - 1].requesting_work_ = true;
          transitState(IDLE, "FSM");
        }
        visualize(1);
        // clearVisMarker();
      }
      break;
    }

    case PUB_TRAJ: {
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        bspline_pub_.publish(fd_->newest_traj_);
        fd_->static_state_ = false;

        // fd_->newest_traj_.drone_id = planner_manager_->swarm_traj_data_.drone_id_;
        fd_->newest_traj_.drone_id = expl_manager_->ep_->drone_id_;
        swarm_traj_pub_.publish(fd_->newest_traj_);

        thread vis_thread(&FastExplorationFSM::visualize, this, 2);
        vis_thread.detach();
        transitState(EXEC_TRAJ, "FSM");
      }
      break;
    }

    case EXEC_TRAJ: {
      auto tn = ros::Time::now();
      // Check whether replan is needed
      LocalTrajData* info = &planner_manager_->local_data_;
      double t_cur = (tn - info->start_time_).toSec();

      if (!fd_->go_back_) {
        bool need_replan = false;
        if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) {
          ROS_WARN("Replan: cluster covered=====================================");
          need_replan = true;
        } else if (info->duration_ - t_cur < fp_->replan_thresh1_) {
          // Replan if traj is almost fully executed
          ROS_WARN("Replan: traj fully executed=================================");
          need_replan = true;
        } else if (t_cur > fp_->replan_thresh3_) {
          // Replan after some time
          ROS_WARN("Replan: periodic call=======================================");
          need_replan = true;
        }

        if (need_replan) {
          if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
            // Update frontier and plan new motion
            thread vis_thread(&FastExplorationFSM::visualize, this, 1);
            vis_thread.detach();
            transitState(PLAN_TRAJ, "FSM");
          } else {
            // No frontier detected, finish exploration
            fd_->last_check_frontier_time_ = ros::Time::now();
            idle_since_s_ = ros::Time::now().toSec();
            if (globalCooperativeMode())
              enterHelpMode("no local frontier");
            else if (pairwiseGlobalRecoveryMode())
              enterPairSearch("no local frontier");
            else {
              expl_manager_->ed_->swarm_state_[getId() - 1].requesting_work_ = true;
              transitState(IDLE, "FSM");
            }
            ROS_WARN("Idle since no frontier is detected");
            fd_->static_state_ = true;
            replan_pub_.publish(std_msgs::Empty());
            // clearVisMarker();
            visualize(1);
          }
        }
      } else {
        // Check if reach goal
        auto pos = info->position_traj_.evaluateDeBoorT(t_cur);
        if ((pos - expl_manager_->ed_->next_pos_).norm() < 1.0) {
          replan_pub_.publish(std_msgs::Empty());
          clearVisMarker();
          if (globalCooperativeMode())
            enterHelpMode("return completed before global finish");
          else if (pairwiseGlobalRecoveryMode())
            enterPairSearch("return completed before global finish");
          else
            transitState(FINISH, "FSM");
          return;
        }
        if (t_cur > fp_->replan_thresh3_ || info->duration_ - t_cur < fp_->replan_thresh1_) {
          // Replan for going back
          replan_pub_.publish(std_msgs::Empty());
          transitState(PLAN_TRAJ, "FSM");
          thread vis_thread(&FastExplorationFSM::visualize, this, 1);
          vis_thread.detach();
        }
      }

      break;
    }
  }
}

int FastExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);

  // The ROS2 sensor boundary rate-limits background ESDF updates to 2 Hz.
  // Preserve original planner semantics by synchronously refreshing a dirty
  // transform immediately before either full planning or collision replanning.
  MapROS::updateESDFForPlanning(expl_manager_->sdf_map_.get());

  int res;
  if (fd_->avoid_collision_ || fd_->go_back_) {  // Only replan trajectory
    // ROS2/Isaac sensor boundary only: just before the unchanged return
    // planner runs, expose the collision-monitored corridor actually
    // traversed by the physical body.  Keeping this disabled during normal
    // exploration prevents measured tube surfaces from becoming frontiers.
    if (fd_->go_back_) expl_manager_->sdf_map_->prepareReturnCorridor();
    res = expl_manager_->planTrajToView(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
        fd_->start_yaw_, expl_manager_->ed_->next_pos_, expl_manager_->ed_->next_yaw_);
    fd_->avoid_collision_ = false;
  } else {  // Do full planning normally
    res = expl_manager_->planExploreMotion(
        fd_->start_pt_, fd_->start_vel_, fd_->start_acc_, fd_->start_yaw_);
  }

  if (res == SUCCEED) {
    auto info = &planner_manager_->local_data_;
    info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;

    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;
    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
    fd_->newest_traj_ = bspline;
  }
  return res;
}

void FastExplorationFSM::visualize(int content) {
  // content 1: frontier; 2 paths & trajs
  auto info = &planner_manager_->local_data_;
  auto plan_data = &planner_manager_->plan_data_;
  auto ed_ptr = expl_manager_->ed_;

  auto getColorVal = [&](const int& id, const int& num, const int& drone_id) {
    double a = (drone_id - 1) / double(num + 1);
    double b = 1 / double(num + 1);
    return a + b * double(id) / ed_ptr->frontiers_.size();
  };

  if (content == 1) {
    // Draw frontier
    static int last_ftr_num = 0;
    static int last_dftr_num = 0;
    for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed_ptr->frontiers_[i], 0.1,
          visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.4), "frontier", i, 4);

      // getColorVal(i, expl_manager_->ep_->drone_num_, expl_manager_->ep_->drone_id_)
      // double(i) / ed_ptr->frontiers_.size()

      // visualization_->drawBox(ed_ptr->frontier_boxes_[i].first,
      // ed_ptr->frontier_boxes_[i].second,
      //     Vector4d(0.5, 0, 1, 0.3), "frontier_boxes", i, 4);
    }
    for (int i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }
    last_ftr_num = ed_ptr->frontiers_.size();

    // for (int i = 0; i < ed_ptr->dead_frontiers_.size(); ++i)
    //   visualization_->drawCubes(
    //       ed_ptr->dead_frontiers_[i], 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);
    // for (int i = ed_ptr->dead_frontiers_.size(); i < last_dftr_num; ++i)
    //   visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);
    // last_dftr_num = ed_ptr->dead_frontiers_.size();

    // // Draw updated box
    // Vector3d bmin, bmax;
    // planner_manager_->edt_environment_->sdf_map_->getUpdatedBox(bmin, bmax, false);
    // visualization_->drawBox(
    //     (bmin + bmax) / 2.0, bmax - bmin, Vector4d(0, 1, 0, 0.3), "updated_box", 0, 4);

    // vector<Eigen::Vector3d> bmins, bmaxs;
    // planner_manager_->edt_environment_->sdf_map_->mm_->getChunkBoxes(bmins, bmaxs, false);
    // for (int i = 0; i < bmins.size(); ++i) {
    //   visualization_->drawBox((bmins[i] + bmaxs[i]) / 2.0, bmaxs[i] - bmins[i],
    //       Vector4d(0, 1, 1, 0.3), "updated_box", i + 1, 4);
    // }

  } else if (content == 2) {

    // Hierarchical grid and global tour --------------------------------
    // vector<Eigen::Vector3d> pts1, pts2;
    // expl_manager_->uniform_grid_->getPath(pts1, pts2);
    // visualization_->drawLines(pts1, pts2, 0.05, Eigen::Vector4d(1, 0.3, 0, 1), "partition", 0,
    // 6);

    if (expl_manager_->ep_->drone_id_ == 1) {
      vector<Eigen::Vector3d> pts1, pts2;
      expl_manager_->hgrid_->getGridMarker(pts1, pts2);
      visualization_->drawLines(pts1, pts2, 0.05, Eigen::Vector4d(1, 0, 1, 0.5), "partition", 1, 6);

      vector<Eigen::Vector3d> pts;
      vector<string> texts;
      expl_manager_->hgrid_->getGridMarker2(pts, texts);
      static int last_text_num = 0;
      for (int i = 0; i < pts.size(); ++i) {
        visualization_->drawText(pts[i], texts[i], 1, Eigen::Vector4d(0, 0, 0, 1), "text", i, 6);
      }
      for (int i = pts.size(); i < last_text_num; ++i) {
        visualization_->drawText(
            Eigen::Vector3d(0, 0, 0), string(""), 1, Eigen::Vector4d(0, 0, 0, 1), "text", i, 6);
      }
      last_text_num = pts.size();

      // // Pub hgrid to ground node
      // exploration_manager::HGrid hgrid;
      // hgrid.stamp = ros::Time::now().toSec();
      // for (int i = 0; i < pts1.size(); ++i) {
      //   geometry_msgs::Point pt1, pt2;
      //   pt1.x = pts1[i][0];
      //   pt1.y = pts1[i][1];
      //   pt1.z = pts1[i][2];
      //   hgrid.points1.push_back(pt1);
      //   pt2.x = pts2[i][0];
      //   pt2.y = pts2[i][1];
      //   pt2.z = pts2[i][2];
      //   hgrid.points2.push_back(pt2);
      // }
      // hgrid_pub_.publish(hgrid);
    }

    auto grid_tour = expl_manager_->ed_->grid_tour_;
    // auto grid_tour = expl_manager_->ed_->grid_tour2_;
    // for (auto& pt : grid_tour) pt = pt + trans;

    visualization_->drawLines(grid_tour, 0.05,
        PlanningVisualization::getColor(
            (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_)),
        "grid_tour", 0, 6);

    // Publish grid tour to ground node
    exploration_manager::GridTour tour;
    for (int i = 0; i < grid_tour.size(); ++i) {
      geometry_msgs::Point point;
      point.x = grid_tour[i][0];
      point.y = grid_tour[i][1];
      point.z = grid_tour[i][2];
      tour.points.push_back(point);
    }
    tour.drone_id = expl_manager_->ep_->drone_id_;
    tour.stamp = ros::Time::now().toSec();
    grid_tour_pub_.publish(tour);

    // visualization_->drawSpheres(
    //     expl_manager_->ed_->grid_tour_, 0.3, Eigen::Vector4d(0, 1, 0, 1), "grid_tour", 1, 6);
    // visualization_->drawLines(
    //     expl_manager_->ed_->grid_tour2_, 0.05, Eigen::Vector4d(0, 1, 0, 0.5), "grid_tour", 2, 6);

    // Top viewpoints and frontier tour-------------------------------------

    // visualization_->drawSpheres(ed_ptr->points_, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->points_, ed_ptr->views_, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->points_, ed_ptr->averages_, 0.03, Vector4d(1, 0, 0, 1), "point-average", 0, 6);

    // auto frontier = ed_ptr->frontier_tour_;
    // for (auto& pt : frontier) pt = pt + trans;
    // visualization_->drawLines(frontier, 0.07,
    //     PlanningVisualization::getColor(
    //         (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_), 0.6),
    //     "frontier_tour", 0, 6);

    // for (int i = 0; i < ed_ptr->other_tours_.size(); ++i) {
    //   visualization_->drawLines(
    //       ed_ptr->other_tours_[i], 0.07, Eigen::Vector4d(0, 0, 1, 1), "other_tours", i, 6);
    // }

    // Locally refined viewpoints and refined tour-------------------------------

    // visualization_->drawSpheres(
    //     ed_ptr->refined_points_, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->refined_points_, ed_ptr->refined_views_, 0.05, Vector4d(0.5, 0, 1, 1),
    //     "refined_view", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->refined_tour_, 0.07,
    //     PlanningVisualization::getColor(
    //         (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_), 0.6),
    //     "refined_tour", 0, 6);

    // visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.04, Vector4d(0,
    // 0, 0, 1),
    //                           "refined_view", 0, 6);
    // visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->unrefined_points_, 0.05,
    // Vector4d(1, 1, 0, 1),
    //                           "refine_pair", 0, 6);
    // for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
    //   visualization_->drawSpheres(ed_ptr->n_points_[i], 0.1,
    //                               visualization_->getColor(double(ed_ptr->refined_ids_[i]) /
    //                               ed_ptr->frontiers_.size()),
    //                               "n_points", i, 6);
    // for (int i = ed_ptr->n_points_.size(); i < 15; ++i)
    //   visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);

    // Trajectory-------------------------------------------

    // visualization_->drawSpheres(
    //     { ed_ptr->next_goal_ /* + trans */ }, 0.3, Vector4d(0, 0, 1, 1), "next_goal", 0, 6);

    // vector<Eigen::Vector3d> next_yaw_vis;
    // next_yaw_vis.push_back(ed_ptr->next_goal_ /* + trans */);
    // next_yaw_vis.push_back(
    //     ed_ptr->next_goal_ /* + trans */ +
    //     2.0 * Eigen::Vector3d(cos(ed_ptr->next_yaw_), sin(ed_ptr->next_yaw_), 0));
    // visualization_->drawLines(next_yaw_vis, 0.1, Eigen::Vector4d(0, 0, 1, 1), "next_goal", 1, 6);
    // visualization_->drawSpheres(
    //     { ed_ptr->next_pos_ /* + trans */ }, 0.3, Vector4d(0, 1, 0, 1), "next_pos", 0, 6);

    // Eigen::MatrixXd ctrl_pt = info->position_traj_.getControlPoint();
    // for (int i = 0; i < ctrl_pt.rows(); ++i) {
    //   for (int j = 0; j < 3; ++j) ctrl_pt(i, j) = ctrl_pt(i, j) + trans[j];
    // }
    // NonUniformBspline position_traj(ctrl_pt, 3, info->position_traj_.getKnotSpan());

    visualization_->drawBspline(info->position_traj_, 0.1,
        PlanningVisualization::getColor(
            (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_)),
        false, 0.15, Vector4d(1, 1, 0, 1));

    // visualization_->drawLines(
    //     expl_manager_->ed_->path_next_goal_, 0.1, Eigen::Vector4d(0, 1, 0, 1), "astar", 0, 6);
    // visualization_->drawSpheres(
    //     expl_manager_->ed_->kino_path_, 0.1, Eigen::Vector4d(0, 0, 1, 1), "kino", 0, 6);
    // visualization_->drawSpheres(plan_data->kino_path_, 0.1, Vector4d(1, 0, 1, 1), "kino_path", 0,
    // 0); visualization_->drawLines(ed_ptr->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1),
    // "next_goal", 1, 6);

    // // Draw trajs of other drones
    // vector<NonUniformBspline> trajs;
    // planner_manager_->swarm_traj_data_.getValidTrajs(trajs);
    // for (int k = 0; k < trajs.size(); ++k) {
    //   visualization_->drawBspline(trajs[k], 0.1, Eigen::Vector4d(1, 1, 0, 1), false, 0.15,
    //       Eigen::Vector4d(0, 0, 1, 1), k + 1);
    // }
  }
}

void FastExplorationFSM::clearVisMarker() {
  for (int i = 0; i < 10; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    // visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "dead_frontier", i, 4);
    // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
    // "frontier_boxes", i, 4);
  }
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "frontier_tour", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "grid_tour", 0, 6);
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  // visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  // visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 1, 1), "B-Spline", 0, 0);

  // visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "current_pose", 0, 6);
}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  if (state_ == WAIT_TRIGGER) {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;

    auto getColorVal = [&](const int& id, const int& num, const int& drone_id) {
      double a = (drone_id - 1) / double(num + 1);
      double b = 1 / double(num + 1);
      return a + b * double(id) / ed->frontiers_.size();
    };

    // ft->searchFrontiers();
    // ft->computeFrontiersToVisit();
    // ft->updateFrontierCostMatrix();

    // ft->getFrontiers(ed->frontiers_);
    // ft->getFrontierBoxes(ed->frontier_boxes_);

    expl_manager_->updateFrontierStruct(fd_->odom_pos_);

    cout << "odom: " << fd_->odom_pos_.transpose() << endl;
    vector<int> tmp_id1;
    vector<vector<int>> tmp_id2;
    bool status = expl_manager_->findGlobalTourOfGrid(
        { fd_->odom_pos_ }, { fd_->odom_vel_ }, tmp_id1, tmp_id2, true);

    // Draw frontier and bounding box
    for (int i = 0; i < ed->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed->frontiers_[i], 0.1,
          visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4), "frontier", i, 4);
      // getColorVal(i, expl_manager_->ep_->drone_num_, expl_manager_->ep_->drone_id_)
      // double(i) / ed->frontiers_.size()
      // visualization_->drawBox(ed->frontier_boxes_[i].first, ed->frontier_boxes_[i].second,
      // Vector4d(0.5, 0, 1, 0.3),
      //                         "frontier_boxes", i, 4);
    }
    for (int i = ed->frontiers_.size(); i < 50; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }
    if (status)
      visualize(2);
    else
      visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "grid_tour", 0, 6);

    // Draw grid tour
  }
}

void FastExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg) {

  // // Debug traj planner
  // Eigen::Vector3d pos;
  // pos << msg->pose.position.x, msg->pose.position.y, 1;
  // expl_manager_->ed_->next_pos_ = pos;

  // Eigen::Vector3d dir = pos - fd_->odom_pos_;
  // expl_manager_->ed_->next_yaw_ = atan2(dir[1], dir[0]);
  // fd_->go_back_ = true;
  // transitState(PLAN_TRAJ, "triggerCallback");
  // return;

  if (state_ != WAIT_TRIGGER) return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  fd_->start_pos_ = fd_->odom_pos_;
  ROS_WARN_STREAM("Start expl pos: " << fd_->start_pos_.transpose());

  if (waitingForInitialAssignment()) {
    fd_->static_state_ = true;
    fd_->go_back_ = false;
    idle_since_s_ = ros::Time::now().toSec();
    transitState(IDLE, "awaitUav1InitialAssignment");
    ROS_WARN("RACER_BLOCKING_INITIAL_WAIT uav=%d time_s=%.3f", getId(), idle_since_s_);
  } else if (globalCooperativeMode()) {
    // All UAVs wait for the first joint assignment; none independently claims
    // the locally best grid during startup.
    expl_manager_->updateFrontierStruct(fd_->odom_pos_);
    enterHelpMode("await initial global assignment");
  } else if (localComponentMode()) {
    // Discovery has already been running through normal DroneState traffic.
    // Freeze only when a valid component assignment is applied.
    expl_manager_->updateFrontierStruct(fd_->odom_pos_);
    auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
    ego.requesting_work_ = true;
    idle_since_s_ = ros::Time::now().toSec();
    transitState(IDLE, "awaitLocalComponentAssignment");
    ROS_WARN("RACER_LOCAL_COMPONENT_WAIT uav=%d coordinator=%d members=%s",
        getId(), local_component_coordinator_id_,
        formatIdList(local_component_members_).c_str());
  } else if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
    markTaskNormal("startup local task available");
    transitState(PLAN_TRAJ, "triggerCallback");
  } else {
    enterPairSearch("startup local task empty");
  }
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    double dist;
    bool safe = planner_manager_->checkTrajCollision(dist);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      fd_->avoid_collision_ = true;
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::trackingLostCallback(const std_msgs::EmptyConstPtr& msg) {
  (void)msg;
  if (state_ != EXPL_STATE::PLAN_TRAJ && state_ != EXPL_STATE::PUB_TRAJ &&
      state_ != EXPL_STATE::EXEC_TRAJ)
    return;

  // Keep the already selected next_pos_/next_yaw_. This only invalidates the
  // unexecuted trajectory and restores a physically valid initial condition.
  replan_pub_.publish(std_msgs::Empty());
  fd_->static_state_ = true;
  fd_->avoid_collision_ = true;
  transitState(PLAN_TRAJ, "trackingLostCallback");
  ROS_WARN("Execution tracking lost: replan current goal from measured odometry");
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  if (!fd_->have_odom_) {
    fd_->have_odom_ = true;
    fd_->fsm_init_time_ = ros::Time::now();
  }
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  if (waitingForInitialAssignment() &&
      (new_state == PLAN_TRAJ || new_state == PUB_TRAJ ||
       new_state == EXEC_TRAJ || new_state == FINISH)) {
    ROS_WARN_THROTTLE(2.0, "RACER_BLOCKING_INITIAL_GUARD uav=%d source=%s",
        getId(), pos_call.c_str());
    return;
  }
  int pre_s = int(state_);
  state_ = new_state;
  ROS_INFO_STREAM("[" + pos_call + "]: Drone "
                  << getId()
                  << " from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]);
}

void FastExplorationFSM::updateLocalComponentMembership(double now_s) {
  if (!localComponentMode()) return;

  const int self = getId();
  local_component_positions_[self - 1] = fd_->odom_pos_;
  local_component_position_valid_[self - 1] = fd_->have_odom_ ? 1 : 0;
  if (local_component_last_membership_change_s_ < 0.0)
    local_component_last_membership_change_s_ = now_s;

  if (!local_component_membership_frozen_) {
    std::unordered_set<int> discovered(
        local_component_members_.begin(), local_component_members_.end());
    discovered.insert(self);
    for (std::size_t peer = 0; peer < local_component_last_state_received_s_.size(); ++peer) {
      const int peer_id = static_cast<int>(peer) + 1;
      if (peer_id == self || local_component_last_state_received_s_[peer] < 0.0 ||
          now_s - local_component_last_state_received_s_[peer] >
              local_component_neighbor_freshness_s_)
        continue;
      const auto& heard = local_component_peer_heard_ids_[peer];
      if (std::find(heard.begin(), heard.end(), self) == heard.end()) continue;

      // This is a proven bidirectional edge.  Merge the peer's converged view
      // to compute the transitive local connected component without any
      // out-of-band global link oracle.
      discovered.insert(peer_id);
      for (const int member : local_component_peer_member_ids_[peer])
        if (member > 0 &&
            member <= static_cast<int>(local_component_positions_.size()) &&
            local_component_position_valid_[member - 1])
          discovered.insert(member);
    }

    vector<int> next_members(discovered.begin(), discovered.end());
    std::sort(next_members.begin(), next_members.end());
    if (next_members != local_component_members_) {
      local_component_members_ = std::move(next_members);
      local_component_coordinator_id_ = local_component_members_.front();
      local_component_last_membership_change_s_ = now_s;
      ROS_WARN("RACER_LOCAL_COMPONENT_DISCOVERY uav=%d coordinator=%d members=%s",
          self, local_component_coordinator_id_,
          formatIdList(local_component_members_).c_str());
    }
  }

  if (!local_component_assignment_applied_) return;
  local_component_ack_ids_.insert(self);
  for (std::size_t peer = 0; peer < local_component_last_state_received_s_.size(); ++peer) {
    if (local_component_last_state_received_s_[peer] < 0.0 ||
        now_s - local_component_last_state_received_s_[peer] >
            local_component_neighbor_freshness_s_ ||
        local_component_peer_coordinator_ids_[peer] != local_component_coordinator_id_ ||
        local_component_peer_assignment_epochs_[peer] != local_component_assignment_epoch_ ||
        local_component_peer_member_ids_[peer] != local_component_members_)
      continue;
    for (const int ack : local_component_peer_ack_ids_[peer])
      if (std::binary_search(local_component_members_.begin(),
              local_component_members_.end(), ack))
        local_component_ack_ids_.insert(ack);
  }

  bool all_acked = true;
  for (const int member : local_component_members_)
    all_acked = all_acked && local_component_ack_ids_.count(member) != 0;
  if (all_acked && !local_component_bootstrap_complete_) {
    local_component_bootstrap_complete_ = true;
    ROS_WARN("RACER_LOCAL_COMPONENT_COMMITTED uav=%d coordinator=%d epoch=%u "
             "members=%s acks=%zu/%zu",
        self, local_component_coordinator_id_, local_component_assignment_epoch_,
        formatIdList(local_component_members_).c_str(),
        local_component_ack_ids_.size(), local_component_members_.size());
    if (state_ == IDLE &&
        expl_manager_->ed_->swarm_state_[self - 1].grid_ids_.empty())
      transitState(FINISH, "localComponentNoLocalTask");
  }
}

void FastExplorationFSM::droneStateTimerCallback(const ros::TimerEvent& e) {
  // Broadcast own state periodically
  exploration_manager::DroneState msg;
  msg.drone_id = getId();

  auto& state = expl_manager_->ed_->swarm_state_[msg.drone_id - 1];

  if (fd_->static_state_) {
    state.pos_ = fd_->odom_pos_;
    state.vel_ = fd_->odom_vel_;
    state.yaw_ = fd_->odom_yaw_;
  } else {
    LocalTrajData* info = &planner_manager_->local_data_;
    double t_r = (ros::Time::now() - info->start_time_).toSec();
    state.pos_ = info->position_traj_.evaluateDeBoorT(t_r);
    state.vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
    state.yaw_ = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
  }
  state.stamp_ = ros::Time::now().toSec();
  updateLocalComponentMembership(state.stamp_);
  msg.pos = { float(state.pos_[0]), float(state.pos_[1]), float(state.pos_[2]) };
  msg.vel = { float(state.vel_[0]), float(state.vel_[1]), float(state.vel_[2]) };
  msg.yaw = state.yaw_;
  for (auto id : state.grid_ids_) msg.grid_ids.push_back(id);
  msg.assignment_epoch = state.assignment_epoch_;
  msg.commitment_until = state.commitment_until_;
  msg.requesting_work = state.requesting_work_;
  msg.recent_attempt_time = state.recent_attempt_time_;
  msg.stamp = state.stamp_;

  if (localComponentMode()) {
    for (std::size_t peer = 0; peer < local_component_last_state_received_s_.size(); ++peer) {
      if (static_cast<int>(peer) + 1 != getId() &&
          local_component_last_state_received_s_[peer] >= 0.0 &&
          state.stamp_ - local_component_last_state_received_s_[peer] <=
              local_component_neighbor_freshness_s_)
        msg.heard_neighbor_ids.push_back(static_cast<int32_t>(peer) + 1);
    }
    msg.local_component_coordinator_id = local_component_coordinator_id_;
    msg.local_component_member_ids = local_component_members_;
    for (const int member : local_component_members_) {
      const auto& position = local_component_positions_[member - 1];
      msg.local_component_member_positions.push_back(float(position[0]));
      msg.local_component_member_positions.push_back(float(position[1]));
      msg.local_component_member_positions.push_back(float(position[2]));
    }
    msg.local_component_assignment_epoch = local_component_assignment_epoch_;
    msg.local_component_ack_ids.assign(
        local_component_ack_ids_.begin(), local_component_ack_ids_.end());
    std::sort(msg.local_component_ack_ids.begin(), msg.local_component_ack_ids.end());
  }

  drone_state_pub_.publish(msg);
  if (globalCooperativeMode()) hybridStateTimerCallback();
}

void FastExplorationFSM::droneStateMsgCallback(const exploration_manager::DroneStateConstPtr& msg) {
  // Update other drones' states
  if (msg->drone_id == getId()) return;
  if (msg->drone_id <= 0 ||
      msg->drone_id > static_cast<int>(expl_manager_->ed_->swarm_state_.size()) ||
      msg->pos.size() < 3 || msg->vel.size() < 3) {
    ROS_ERROR("Reject malformed drone state from id %d.", msg->drone_id);
    return;
  }

  // Simulate swarm communication loss
  Eigen::Vector3d msg_pos(msg->pos[0], msg->pos[1], msg->pos[2]);
  // if ((msg_pos - fd_->odom_pos_).norm() > 6.0) return;

  auto& drone_state = expl_manager_->ed_->swarm_state_[msg->drone_id - 1];
  if (drone_state.stamp_ + 1e-4 >= msg->stamp) return;  // Avoid unordered msg

  drone_state.pos_ = Eigen::Vector3d(msg->pos[0], msg->pos[1], msg->pos[2]);
  drone_state.vel_ = Eigen::Vector3d(msg->vel[0], msg->vel[1], msg->vel[2]);
  drone_state.yaw_ = msg->yaw;
  vector<int> received_ids;
  std::unordered_set<int> received_set;
  for (const auto id : msg->grid_ids)
    if (received_set.insert(id).second) received_ids.push_back(id);
  if (msg->assignment_epoch >= drone_state.assignment_epoch_ &&
      validateGridIds(received_ids, "drone state")) {
    drone_state.grid_ids_ = received_ids;
    drone_state.assignment_epoch_ = msg->assignment_epoch;
    drone_state.commitment_until_ = msg->commitment_until;
    drone_state.requesting_work_ = msg->requesting_work;
  }
  last_reported_assignment_epochs_[msg->drone_id - 1] =
      std::max(last_reported_assignment_epochs_[msg->drone_id - 1], msg->assignment_epoch);
  drone_state.stamp_ = msg->stamp;
  drone_state.recent_attempt_time_ = msg->recent_attempt_time;

  if (localComponentMode()) {
    const std::size_t peer = static_cast<std::size_t>(msg->drone_id - 1);
    local_component_last_state_received_s_[peer] = ros::Time::now().toSec();

    auto sanitize_ids = [&](const auto& source) {
      vector<int> result;
      std::unordered_set<int> seen;
      for (const auto raw : source) {
        const int id = static_cast<int>(raw);
        if (id <= 0 || id > static_cast<int>(local_component_positions_.size()) ||
            !seen.insert(id).second)
          continue;
        result.push_back(id);
      }
      std::sort(result.begin(), result.end());
      return result;
    };
    local_component_peer_heard_ids_[peer] =
        sanitize_ids(msg->heard_neighbor_ids);

    vector<int> advertised_members =
        sanitize_ids(msg->local_component_member_ids);
    const bool valid_positions = msg->local_component_member_positions.size() ==
        3 * advertised_members.size() &&
        advertised_members.size() == msg->local_component_member_ids.size() &&
        std::equal(advertised_members.begin(), advertised_members.end(),
            msg->local_component_member_ids.begin());
    if (valid_positions) {
      bool finite_positions = true;
      for (std::size_t index = 0; index < advertised_members.size(); ++index) {
        Eigen::Vector3d position(
            msg->local_component_member_positions[3 * index],
            msg->local_component_member_positions[3 * index + 1],
            msg->local_component_member_positions[3 * index + 2]);
        if (!position.allFinite()) {
          finite_positions = false;
          break;
        }
        local_component_positions_[advertised_members[index] - 1] = position;
        local_component_position_valid_[advertised_members[index] - 1] = 1;
      }
      if (finite_positions)
        local_component_peer_member_ids_[peer] = std::move(advertised_members);
    }
    // The sender's own position is always authoritative even before its
    // component-gossip vector has converged.
    local_component_positions_[peer] = drone_state.pos_;
    local_component_position_valid_[peer] = 1;
    local_component_peer_coordinator_ids_[peer] =
        msg->local_component_coordinator_id;
    local_component_peer_assignment_epochs_[peer] =
        msg->local_component_assignment_epoch;
    vector<int> ack_ids;
    std::unordered_set<int> ack_seen;
    for (const auto raw : msg->local_component_ack_ids) {
      const int id = static_cast<int>(raw);
      if (id > 0 && id <= static_cast<int>(local_component_positions_.size()) &&
          ack_seen.insert(id).second)
        ack_ids.push_back(id);
    }
    std::sort(ack_ids.begin(), ack_ids.end());
    local_component_peer_ack_ids_[peer] = std::move(ack_ids);
    updateLocalComponentMembership(ros::Time::now().toSec());
  }

  // A COMMITTED response can be lost even though the peer has already committed. Its periodic
  // state broadcast is an independent, versioned confirmation that safely completes our side.
  if (outgoing_transaction_.active && outgoing_transaction_.commit_sent &&
      outgoing_transaction_.peer_id == msg->drone_id &&
      msg->assignment_epoch == outgoing_transaction_.new_epoch &&
      received_ids == outgoing_transaction_.other_ids) {
    applyCommittedPairAssignment(outgoing_transaction_.peer_id,
        outgoing_transaction_.ego_ids, outgoing_transaction_.other_ids,
        outgoing_transaction_.new_epoch, outgoing_transaction_.commitment_until);
    last_committed_transaction_id_ = outgoing_transaction_.transaction_id;
    last_committed_transaction_peer_ = outgoing_transaction_.peer_id;
    clearOutgoingTransaction("committed via state confirmation");
  }

  // std::cout << "Drone " << getId() << " get drone " << int(msg->drone_id) << "'s state" <<
  // std::endl; std::cout << drone_state.pos_.transpose() << std::endl;
}

void FastExplorationFSM::hybridStateTimerCallback() {
  using StateMsg = hybrid_communication_racer::msg::HybridDroneState;
  StateMsg msg;
  msg.drone_id = getId();
  msg.stamp = ros::Time::now().toSec();
  msg.position = {float(fd_->odom_pos_[0]), float(fd_->odom_pos_[1]), float(fd_->odom_pos_[2])};
  if (state_ == FINISH)
    msg.task_status = StateMsg::TASK_FINISHED;
  else if (state_ == IDLE)
    msg.task_status = StateMsg::TASK_HELP;
  else if (state_ == INIT || state_ == WAIT_TRIGGER)
    msg.task_status = StateMsg::TASK_WAITING;
  else
    msg.task_status = StateMsg::TASK_ACTIVE;

  expl_manager_->hgrid_->getActiveGrids(msg.active_grid_ids);
  auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
  msg.assigned_grid_ids = ego.grid_ids_;
  msg.has_target = primary_grid_id_ >= 0 && !ego.grid_ids_.empty();
  msg.primary_grid_id = primary_grid_id_;
  msg.current_target = {float(primary_target_[0]), float(primary_target_[1]),
      float(primary_target_[2])};
  msg.target_utility = target_utility_;
  msg.estimated_information_gain = target_information_gain_;
  msg.estimated_travel_time = target_travel_time_;
  msg.overlap_penalty = target_overlap_penalty_;
  msg.adaptive_distance_limit = adaptive_distance_limit_;
  msg.requesting_reassignment = ego.requesting_work_ || state_ == IDLE;
  msg.failed_grid_id = failed_grid_id_;
  msg.failure_sequence = failure_sequence_;
  msg.reassignment_count = reassignment_count_;
  msg.help_mode_count = help_mode_count_;
  msg.takeover_count = takeover_count_;
  msg.idle_time = idle_since_s_ < 0.0 ? 0.0 : std::max(0.0, msg.stamp - idle_since_s_);
  msg.coverage_contribution = coverage_contributions_[getId() - 1];

  hybrid_states_[getId() - 1] = msg;
  hybrid_state_pub_.publish(msg);
  ROS_INFO_THROTTLE(1.0,
      "RACER_HYBRID_STATS uav=%d status=%u target=%d utility=%.5f gain=%.1f "
      "travel=%.3f overlap=%.3f R=%.2f reassign=%lu help=%lu takeover=%lu "
      "idle=%.3f coverage_contribution=%.1f",
      getId(), msg.task_status, primary_grid_id_, target_utility_,
      target_information_gain_, target_travel_time_, target_overlap_penalty_,
      adaptive_distance_limit_, static_cast<unsigned long>(reassignment_count_),
      static_cast<unsigned long>(help_mode_count_),
      static_cast<unsigned long>(takeover_count_), msg.idle_time,
      msg.coverage_contribution);
}

void FastExplorationFSM::hybridStateMsgCallback(
    const hybrid_communication_racer::msg::HybridDroneState::ConstSharedPtr& msg) {
  if (!globalCooperativeMode() || !msg || msg->drone_id <= 0 ||
      msg->drone_id > static_cast<int>(hybrid_states_.size()) || msg->position.size() < 3 ||
      (msg->has_target && msg->current_target.size() < 3))
    return;
  if (!validateGridIds(msg->active_grid_ids, "hybrid active pool") ||
      !validateGridIds(msg->assigned_grid_ids, "hybrid assigned grids"))
    return;
  auto& stored = hybrid_states_[msg->drone_id - 1];
  if (stored.stamp + 1e-4 >= msg->stamp) return;
  stored = *msg;
}

void FastExplorationFSM::optTimerCallback(const ros::TimerEvent& e) {
  if (state_ == INIT) return;

  auto tn = ros::Time::now().toSec();
  if (globalCooperativeMode()) {
    globalCooperativeAssignmentTimerCallback();
    return;
  }

  // Always finish an already-started transaction before any startup/recovery
  // gate can defer creation of new pair negotiations.
  servicePairTransactions(tn);

#ifdef RACER_ORACLE_VARIANT
  globalAssignmentTimerCallback();
  return;
#else
  if (localComponentMode()) {
    localComponentAssignmentTimerCallback(tn);
    // Planning begins as soon as this UAV applies its component assignment,
    // but pairwise redistribution starts only after this component (not the
    // whole fleet) has acknowledged the epoch.
    if (!local_component_bootstrap_complete_) return;
  } else if (initial_partition_enabled_) {
    initialAssignmentTimerCallback(tn);
    // This branch exists only for explicitly requested legacy experiments.
    // The default starts pairwise negotiation without a global prerequisite.
    if (last_applied_global_assignment_epoch_ == 0) return;
  }
#endif

  if (pairwiseGlobalRecoveryMode())
    bsEventGlobalAssignmentTimerCallback(tn);

  if (outgoing_transaction_.active || incoming_transaction_.active) return;

  // Select nearby drone not interacting with recently
  auto& states = expl_manager_->ed_->swarm_state_;
  auto& state1 = states[getId() - 1];
  const bool urgent = state1.grid_ids_.empty() || state1.requesting_work_;

  if (state1.grid_ids_.empty() &&
      task_coordination_gate_.mode() == TaskCoordinationMode::NORMAL)
    enterPairSearch("empty ownership observed by pair allocator");
  if (pairwiseGlobalRecoveryMode() &&
      task_coordination_gate_.mode() == TaskCoordinationMode::GLOBAL_RECOVERY)
    return;

  // Avoid frequent attempt
  if (tn - state1.recent_attempt_time_ < fp_->attempt_interval_) return;
  if (!urgent && tn < state1.commitment_until_) return;
  state1.recent_attempt_time_ = tn;

  int select_id = -1;
  int fresh_peer_count = 0;
  double max_interval = -1.0;
  for (int i = 0; i < states.size(); ++i) {
    if (i + 1 == getId()) continue;
    // Preserve RACER's lower-id initiator rule for normal balancing. An idle UAV may contact any
    // peer so high-id agents can also steal work; epoch locking prevents conflicting commits.
    if (!urgent && i + 1 < getId()) continue;
    // Check if have communication recently
    // or the drone just experience another opt
    // or the drone is interacted with recently /* !urgent &&  */
    // or the candidate drone dominates enough grids
    if (tn - states[i].stamp_ > fp_->swarm_state_freshness_) continue;
    ++fresh_peer_count;
    if (tn - states[i].recent_attempt_time_ < fp_->attempt_interval_) continue;
    if (tn - states[i].recent_interact_time_ < fp_->pair_opt_interval_) continue;
    if (states[i].grid_ids_.size() + state1.grid_ids_.size() == 0) continue;
    if (!urgent && tn < states[i].commitment_until_) continue;
    if (urgent && states[i].grid_ids_.empty()) continue;

    double interval = urgent ? static_cast<double>(states[i].grid_ids_.size())
                             : tn - states[i].recent_interact_time_;
    if (interval <= max_interval) continue;
    select_id = i + 1;
    max_interval = interval;
  }
  if (select_id == -1) {
    if (state1.grid_ids_.empty() && fresh_peer_count > 0) {
      markPairSearchStarted(-1, "fresh_peer_scan");
      recordPairSearchFailure(-1, "no_transferable_candidate");
    } else if (state1.grid_ids_.empty()) {
      ROS_WARN_THROTTLE(1.0,
          "[PAIR SEARCH] uav=%d attempt=%d candidate=-1 result=no_fresh_peer "
          "failure_count=%d",
          getId(), task_coordination_gate_.pairAttemptCount(),
          task_coordination_gate_.pairFailureCount());
    }
    return;
  }

  // The attempt timestamp was set before candidate search, so both missing
  // candidates and rejected reallocations count at most once per allocation
  // cycle rather than at the 20 Hz timer frequency.
  if (state1.grid_ids_.empty())
    markPairSearchStarted(select_id, "candidate_selected");

  std::cout << "\nSelect: " << select_id << std::endl;
  ROS_WARN("Pair opt %d & %d", getId(), select_id);

  // Do pairwise optimization with selected drone, allocate the union of their domiance grids
  unordered_map<int, char> opt_ids_map;
  auto& state2 = states[select_id - 1];
  for (auto id : state1.grid_ids_) opt_ids_map[id] = 1;
  for (auto id : state2.grid_ids_) opt_ids_map[id] = 1;
  vector<int> opt_ids;
  for (auto pair : opt_ids_map) opt_ids.push_back(pair.first);

  std::cout << "Pair Opt id: ";
  for (auto id : opt_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Find missed grids to reallocated them
  vector<int> actives, missed;
  expl_manager_->hgrid_->getActiveGrids(actives);
  findUnallocated(actives, missed);
  std::cout << "Missed: ";
  for (auto id : missed) std::cout << id << ", ";
  std::cout << "" << std::endl;
  opt_ids.insert(opt_ids.end(), missed.begin(), missed.end());
  std::sort(opt_ids.begin(), opt_ids.end());
  opt_ids.erase(std::unique(opt_ids.begin(), opt_ids.end()), opt_ids.end());
  if (!validateGridIds(opt_ids, "pair optimization input")) {
    recordPairSearchFailure(select_id, "invalid_pair_input");
    return;
  }

  // Do partition of the grid
  vector<Eigen::Vector3d> positions = { state1.pos_, state2.pos_ };
  vector<Eigen::Vector3d> velocities = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0) };
  vector<int> first_ids1, second_ids1, first_ids2, second_ids2;
  if (state_ != WAIT_TRIGGER) {
    expl_manager_->hgrid_->getConsistentGrid(
        state1.grid_ids_, state1.grid_ids_, first_ids1, second_ids1);
    expl_manager_->hgrid_->getConsistentGrid(
        state2.grid_ids_, state2.grid_ids_, first_ids2, second_ids2);
  }

  auto t1 = ros::Time::now();

  vector<int> ego_ids, other_ids;
  expl_manager_->allocateGrids(positions, velocities, { first_ids1, first_ids2 },
      { second_ids1, second_ids2 }, opt_ids, ego_ids, other_ids);

  // ACVRP may legally leave an empty route. During work stealing, give the idle UAV the closest
  // transferable grid while leaving at least one grid with its active peer.
  if (urgent && ego_ids.empty() && other_ids.size() > 1) {
    auto best = other_ids.begin();
    double best_dist = std::numeric_limits<double>::infinity();
    for (auto it = other_ids.begin(); it != other_ids.end(); ++it) {
      const double dist = (expl_manager_->hgrid_->getCenter(*it) - state1.pos_).norm();
      if (dist < best_dist) {
        best_dist = dist;
        best = it;
      }
    }
    ego_ids.push_back(*best);
    other_ids.erase(best);
  }

  double alloc_time = (ros::Time::now() - t1).toSec();

  std::cout << "Ego1  : ";
  for (auto id : state1.grid_ids_) std::cout << id << ", ";
  std::cout << "\nOther1: ";
  for (auto id : state2.grid_ids_) std::cout << id << ", ";
  std::cout << "\nEgo2  : ";
  for (auto id : ego_ids) std::cout << id << ", ";
  std::cout << "\nOther2: ";
  for (auto id : other_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Check results
  double prev_app1 = expl_manager_->computeGridPathCost(state1.pos_, state1.grid_ids_, first_ids1,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  double prev_app2 = expl_manager_->computeGridPathCost(state2.pos_, state2.grid_ids_, first_ids2,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  std::cout << "prev cost: " << prev_app1 << ", " << prev_app2 << ", " << prev_app1 + prev_app2
            << std::endl;
  double cur_app1 = expl_manager_->computeGridPathCost(state1.pos_, ego_ids, first_ids1,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  double cur_app2 = expl_manager_->computeGridPathCost(state2.pos_, other_ids, first_ids2,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  std::cout << "cur cost : " << cur_app1 << ", " << cur_app2 << ", " << cur_app1 + cur_app2
            << std::endl;
  const double previous_cost = prev_app1 + prev_app2;
  const double switching_cost = urgent ? 0.0 : 2.0 * fp_->pair_switch_penalty_;
  const double proposed_cost = cur_app1 + cur_app2 + switching_cost;
  const double improvement = previous_cost - proposed_cost;
  if ((!urgent && improvement < fp_->pair_min_improvement_) ||
      (urgent && ego_ids.empty())) {
    ROS_WARN_THROTTLE(1.0,
        "Pair reallocation skipped after cooldown: improvement %.3f m, urgent=%d, ego=%zu.",
        improvement, int(urgent), ego_ids.size());
    if (state1.grid_ids_.empty())
      recordPairSearchFailure(select_id, "no_transferable_task");
    return;
  }

  if (!state1.grid_ids_.empty() && !ego_ids.empty() &&
      !expl_manager_->hgrid_->isConsistent(state1.grid_ids_[0], ego_ids[0])) {
    ROS_ERROR("Path 1 inconsistent");
  }
  if (!state2.grid_ids_.empty() && !other_ids.empty() &&
      !expl_manager_->hgrid_->isConsistent(state2.grid_ids_[0], other_ids[0])) {
    ROS_ERROR("Path 2 inconsistent");
  }

  if (!validateGridIds(ego_ids, "proposed ego assignment") ||
      !validateGridIds(other_ids, "proposed peer assignment")) {
    if (state1.grid_ids_.empty())
      recordPairSearchFailure(select_id, "invalid_pair_result");
    return;
  }

  outgoing_transaction_ = PairTransaction();
  outgoing_transaction_.active = true;
  outgoing_transaction_.peer_id = select_id;
  outgoing_transaction_.transaction_id =
      (static_cast<uint64_t>(getId()) << 56U) | next_transaction_sequence_++;
  outgoing_transaction_.from_epoch = state1.assignment_epoch_;
  outgoing_transaction_.to_epoch = state2.assignment_epoch_;
  outgoing_transaction_.new_epoch =
      std::max(state1.assignment_epoch_, state2.assignment_epoch_) + 1;
  outgoing_transaction_.commitment_until = tn + fp_->assignment_commitment_duration_;
  outgoing_transaction_.expires_at = tn + fp_->pair_transaction_timeout_;
  outgoing_transaction_.ego_ids = std::move(ego_ids);
  outgoing_transaction_.other_ids = std::move(other_ids);
  expl_manager_->ed_->wait_response_ = true;
  publishPairTransaction(exploration_manager::PairOpt::PHASE_PROPOSE);

  if (state1.grid_ids_.empty())
    ROS_WARN("[PAIR SEARCH] uav=%d attempt=%d candidate=%d result=propose "
             "failure_count=%d",
        getId(), task_coordination_gate_.pairAttemptCount(), select_id,
        task_coordination_gate_.pairFailureCount());

  ROS_WARN("UAV %d prepared pair transaction %lu with UAV %d (gain %.3f m, alloc %.3f s).",
      getId(), static_cast<unsigned long>(outgoing_transaction_.transaction_id), select_id,
      improvement, alloc_time);
}

void FastExplorationFSM::initialAssignmentTimerCallback(double now_s) {
  if (!initial_partition_enabled_ || initial_partition_sent_ || getId() != 1 ||
      state_ == WAIT_TRIGGER)
    return;
  if (last_initial_partition_publish_s_ >= 0.0 &&
      now_s - last_initial_partition_publish_s_ < initial_partition_interval_s_)
    return;

  auto& states = expl_manager_->ed_->swarm_state_;
  if (states.empty()) return;
  if (initial_assignment_msg_.epoch != 0) {
    bool all_applied = true;
    for (const auto epoch : last_reported_assignment_epochs_)
      all_applied = all_applied && epoch >= initial_assignment_msg_.epoch;
    if (all_applied) {
      initial_partition_sent_ = true;
      ROS_WARN("Initial multi-UAV partition committed by all %zu UAVs.", states.size());
      return;
    }
    global_assignment_pub_.publish(initial_assignment_msg_);
    last_initial_partition_publish_s_ = now_s;
    return;
  }

  for (const auto& state : states) {
    if (state.stamp_ <= 0.0 || now_s - state.stamp_ > fp_->swarm_state_freshness_) return;
  }

  vector<int> active_grids;
  expl_manager_->updateGlobalGridPool(fd_->odom_pos_, active_grids);
  std::sort(active_grids.begin(), active_grids.end());
  active_grids.erase(std::unique(active_grids.begin(), active_grids.end()), active_grids.end());
  if (active_grids.empty() || !validateGridIds(active_grids, "initial partition")) return;

  vector<vector<int>> assignments(states.size());
  vector<Eigen::Vector3d> route_ends;
  route_ends.reserve(states.size());
  for (const auto& state : states) route_ends.push_back(state.pos_);

  // Give every UAV one nearby grid first, then greedily extend spatially coherent routes.
  vector<int> remaining = active_grids;
  for (std::size_t drone = 0; drone < states.size() && !remaining.empty(); ++drone) {
    auto best = remaining.begin();
    double best_dist = std::numeric_limits<double>::infinity();
    for (auto it = remaining.begin(); it != remaining.end(); ++it) {
      const double dist = (route_ends[drone] - expl_manager_->hgrid_->getCenter(*it)).norm();
      if (dist < best_dist) {
        best_dist = dist;
        best = it;
      }
    }
    assignments[drone].push_back(*best);
    route_ends[drone] = expl_manager_->hgrid_->getCenter(*best);
    remaining.erase(best);
  }
  for (const int grid_id : remaining) {
    const Eigen::Vector3d center = expl_manager_->hgrid_->getCenter(grid_id);
    int best_drone = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int drone = 0; drone < static_cast<int>(states.size()); ++drone) {
      const double cost = (route_ends[drone] - center).norm() +
          initial_partition_balance_weight_m_ * assignments[drone].size();
      if (cost < best_cost) {
        best_cost = cost;
        best_drone = drone;
      }
    }
    assignments[best_drone].push_back(grid_id);
    route_ends[best_drone] = center;
  }

  initial_assignment_msg_ = exploration_manager::GlobalGridAssignment();
  initial_assignment_msg_.coordinator_id = 1;
  initial_assignment_msg_.stamp = now_s;
  initial_assignment_msg_.epoch = ++global_assignment_epoch_;
  initial_assignment_msg_.purpose =
      exploration_manager::GlobalGridAssignment::PURPOSE_BOOTSTRAP;
  initial_assignment_msg_.global_finish = false;
  initial_assignment_msg_.offsets.push_back(0);
  for (const auto& assignment : assignments) {
    initial_assignment_msg_.grid_ids.insert(initial_assignment_msg_.grid_ids.end(),
        assignment.begin(), assignment.end());
    initial_assignment_msg_.offsets.push_back(
        static_cast<int32_t>(initial_assignment_msg_.grid_ids.size()));
  }
  applyGlobalAssignment(initial_assignment_msg_);
  global_assignment_pub_.publish(initial_assignment_msg_);
  last_initial_partition_publish_s_ = now_s;
  ROS_WARN("Initial spatial partition epoch=%u assigned %zu grids to %zu UAVs.",
      initial_assignment_msg_.epoch, active_grids.size(), states.size());
}

void FastExplorationFSM::bsEventGlobalAssignmentTimerCallback(double now_s) {
  if (!pairwiseGlobalRecoveryMode() || getId() != 1 || state_ == WAIT_TRIGGER) return;

  auto& states = expl_manager_->ed_->swarm_state_;
  if (states.empty()) return;

  // Only one recovery epoch may be in flight. Requests arriving during this
  // interval remain represented by DroneState and are merged into the next
  // recovery window instead of creating overlapping global assignments.
  if (dynamic_assignment_msg_.epoch != 0) {
    bool all_applied = true;
    bool superseded = false;
    for (const int uav_id : global_recovery_expected_ack_ids_) {
      const auto reported_epoch = last_reported_assignment_epochs_[uav_id - 1];
      all_applied = all_applied && reported_epoch >= dynamic_assignment_msg_.epoch;
      superseded = superseded || reported_epoch > dynamic_assignment_msg_.epoch;
    }
    if (superseded) {
      ROS_WARN("Global recovery epoch=%u was superseded by a newer pair/global epoch; "
               "recomputing from fresh ownership.",
          dynamic_assignment_msg_.epoch);
      dynamic_assignment_msg_ = exploration_manager::GlobalGridAssignment();
      global_recovery_expected_ack_ids_.clear();
      global_recovery_window_remaining_cycles_ =
          std::max(1, global_recovery_coalesce_cycles_);
      return;
    }
    if (all_applied) {
      ROS_WARN("RACER_GLOBAL_RECOVERY_COMMITTED epoch=%u acknowledgements=%zu/%zu",
          dynamic_assignment_msg_.epoch, global_recovery_expected_ack_ids_.size(),
          global_recovery_expected_ack_ids_.size());
      for (const auto uav_id : dynamic_assignment_msg_.recovery_uav_ids)
        pending_global_recovery_uavs_.erase(uav_id);
      dynamic_assignment_msg_ = exploration_manager::GlobalGridAssignment();
      global_recovery_expected_ack_ids_.clear();
    } else if (last_bs_event_global_publish_s_ < 0.0 ||
        now_s - last_bs_event_global_publish_s_ >= bs_event_global_retry_interval_s_) {
      global_assignment_pub_.publish(dynamic_assignment_msg_);
      last_bs_event_global_publish_s_ = now_s;
    }
    return;
  }

  if (last_global_recovery_cycle_s_ >= 0.0 &&
      now_s - last_global_recovery_cycle_s_ < fp_->attempt_interval_)
    return;
  last_global_recovery_cycle_s_ = now_s;

  for (const auto& state : states) {
    if (state.stamp_ <= 0.0 ||
        now_s - state.stamp_ > bs_event_global_state_freshness_s_) {
      ROS_WARN_THROTTLE(2.0,
          "RACER_GLOBAL_RECOVERY_WAIT_STATE requires fresh state from every UAV");
      return;
    }
  }

  for (std::size_t drone = 0; drone < states.size(); ++drone) {
    if (states[drone].requesting_work_ && states[drone].grid_ids_.empty())
      pending_global_recovery_uavs_.insert(static_cast<int>(drone) + 1);
  }
  for (auto iterator = pending_global_recovery_uavs_.begin();
       iterator != pending_global_recovery_uavs_.end();) {
    if (!states[*iterator - 1].grid_ids_.empty())
      iterator = pending_global_recovery_uavs_.erase(iterator);
    else
      ++iterator;
  }
  if (pending_global_recovery_uavs_.empty()) {
    global_recovery_window_remaining_cycles_ = 0;
    global_finish_empty_cycles_ = 0;
    return;
  }

  if (global_recovery_window_remaining_cycles_ <= 0)
    global_recovery_window_remaining_cycles_ =
        std::max(1, global_recovery_coalesce_cycles_);
  --global_recovery_window_remaining_cycles_;
  if (global_recovery_window_remaining_cycles_ > 0) return;

  vector<int> active_grids;
  expl_manager_->updateGlobalGridPool(fd_->odom_pos_, active_grids);
  if (!validateGridIds(active_grids, "global recovery pool")) return;
  std::sort(active_grids.begin(), active_grids.end());
  active_grids.erase(std::unique(active_grids.begin(), active_grids.end()),
      active_grids.end());

  vector<vector<int>> current_assignments;
  current_assignments.reserve(states.size());
  int idle_uav_count = 0;
  int active_uav_count = 0;
  for (const auto& state : states) {
    current_assignments.push_back(state.grid_ids_);
    if (state.grid_ids_.empty())
      ++idle_uav_count;
    else
      ++active_uav_count;
  }
  vector<int> requester_indices;
  vector<int> requester_ids(
      pending_global_recovery_uavs_.begin(), pending_global_recovery_uavs_.end());
  std::sort(requester_ids.begin(), requester_ids.end());
  for (const int uav_id : requester_ids) requester_indices.push_back(uav_id - 1);

  const auto plan = buildMinimalRecoveryPlan(current_assignments, active_grids,
      requester_indices, global_recovery_max_tasks_per_idle_,
      [&](int idle_index, int grid_id) {
        return expl_manager_->estimateGridTravelCost(states[idle_index].pos_, grid_id);
      });
  if (!plan.valid) {
    ROS_ERROR("Reject global recovery calculation because current ownership contains duplicates.");
    global_recovery_window_remaining_cycles_ = 1;
    return;
  }

  bool global_finish = false;
  const bool globally_empty_snapshot =
      active_grids.empty() && active_uav_count == 0;
  if (globally_empty_snapshot) {
    ++global_finish_empty_cycles_;
    global_finish = global_finish_empty_cycles_ >=
        std::max(1, global_finish_confirmation_cycles_);
  } else {
    global_finish_empty_cycles_ = 0;
  }
  ROS_WARN("[GLOBAL FINISH CHECK] remaining_global_tasks=%zu active_uavs=%d "
           "idle_uavs=%d finish=%s confirmation=%d/%d",
      plan.remaining_global_tasks, active_uav_count, idle_uav_count,
      global_finish ? "true" : "false", global_finish_empty_cycles_,
      std::max(1, global_finish_confirmation_cycles_));
  if (!global_finish && active_grids.empty()) {
    // Keep requesters in GLOBAL_RECOVERY/IDLE while the coordinator confirms
    // that the empty frontier pool is stable over several allocation cycles.
    global_recovery_window_remaining_cycles_ = 1;
    return;
  }

  vector<vector<int>> assignments = plan.assignments;
  if (global_finish) {
    assignments.assign(states.size(), {});
    requester_ids.clear();
    for (std::size_t drone = 0; drone < states.size(); ++drone)
      requester_ids.push_back(static_cast<int>(drone) + 1);
  }

  uint64_t maximum_epoch = global_assignment_epoch_;
  for (std::size_t drone = 0; drone < states.size(); ++drone) {
    maximum_epoch = std::max(maximum_epoch, states[drone].assignment_epoch_);
    maximum_epoch = std::max(maximum_epoch, last_reported_assignment_epochs_[drone]);
  }
  if (maximum_epoch >= std::numeric_limits<uint32_t>::max()) {
    ROS_ERROR("Global recovery epoch space exhausted.");
    return;
  }

  dynamic_assignment_msg_ = exploration_manager::GlobalGridAssignment();
  dynamic_assignment_msg_.coordinator_id = 1;
  dynamic_assignment_msg_.stamp = now_s;
  dynamic_assignment_msg_.epoch = static_cast<uint32_t>(maximum_epoch + 1);
  global_assignment_epoch_ = dynamic_assignment_msg_.epoch;
  dynamic_assignment_msg_.purpose =
      exploration_manager::GlobalGridAssignment::PURPOSE_RECOVERY;
  dynamic_assignment_msg_.global_finish = global_finish;
  dynamic_assignment_msg_.recovery_uav_ids = requester_ids;
  dynamic_assignment_msg_.offsets.reserve(assignments.size() + 1);
  dynamic_assignment_msg_.offsets.push_back(0);
  for (const auto& assignment : assignments) {
    dynamic_assignment_msg_.grid_ids.insert(dynamic_assignment_msg_.grid_ids.end(),
        assignment.begin(), assignment.end());
    dynamic_assignment_msg_.offsets.push_back(
        static_cast<int32_t>(dynamic_assignment_msg_.grid_ids.size()));
  }

  global_recovery_expected_ack_ids_.clear();
  for (const int requester_id : requester_ids)
    global_recovery_expected_ack_ids_.insert(requester_id);
  for (std::size_t drone = 0; drone < assignments.size(); ++drone) {
    if (assignments[drone] != current_assignments[drone])
      global_recovery_expected_ack_ids_.insert(static_cast<int>(drone) + 1);
    dynamic_assignment_msg_.base_assignment_epochs.push_back(
        states[drone].assignment_epoch_);
  }
  dynamic_assignment_msg_.affected_uav_ids.assign(
      global_recovery_expected_ack_ids_.begin(),
      global_recovery_expected_ack_ids_.end());
  std::sort(dynamic_assignment_msg_.affected_uav_ids.begin(),
      dynamic_assignment_msg_.affected_uav_ids.end());

  if (!applyGlobalAssignment(dynamic_assignment_msg_)) {
    ROS_WARN("Coordinator deferred global recovery epoch=%u due to a conflicting local pair.",
        dynamic_assignment_msg_.epoch);
    dynamic_assignment_msg_ = exploration_manager::GlobalGridAssignment();
    global_recovery_expected_ack_ids_.clear();
    global_recovery_window_remaining_cycles_ = 1;
    return;
  }
  global_assignment_pub_.publish(dynamic_assignment_msg_);
  last_bs_event_global_publish_s_ = now_s;
  for (const int requester_id : requester_ids) {
    vector<int> assigned;
    int donor_id = -1;
    for (const auto& transfer : plan.transfers) {
      if (transfer.idle_uav_index + 1 != requester_id) continue;
      assigned.push_back(transfer.task_id);
      if (transfer.donor_uav_index >= 0) donor_id = transfer.donor_uav_index + 1;
    }
    ROS_WARN("[GLOBAL RECOVERY RESULT] idle_uav=%d donor_uav=%d assigned_tasks=%s "
             "global_version=%u finish=%s",
        requester_id, donor_id, formatIdList(assigned).c_str(),
        dynamic_assignment_msg_.epoch, global_finish ? "true" : "false");
  }
}

void FastExplorationFSM::localComponentAssignmentTimerCallback(double now_s) {
  if (!localComponentMode()) return;
  updateLocalComponentMembership(now_s);
  if (!fd_->trigger_ || state_ == INIT || state_ == WAIT_TRIGGER) return;

  if (local_component_assignment_applied_) {
    if (!local_component_bootstrap_complete_ &&
        (local_component_last_assignment_publish_s_ < 0.0 ||
            now_s - local_component_last_assignment_publish_s_ >=
                local_component_assignment_interval_s_)) {
      // Every applied member relays the coordinator's identical message, so
      // assignment delivery also works across a multi-hop component.
      global_assignment_pub_.publish(local_component_assignment_msg_);
      local_component_last_assignment_publish_s_ = now_s;
    }
    return;
  }

  if (local_component_last_membership_change_s_ < 0.0 ||
      now_s - local_component_last_membership_change_s_ <
          local_component_stability_duration_s_ ||
      getId() != local_component_coordinator_id_)
    return;
  for (const int member : local_component_members_)
    if (!local_component_position_valid_[member - 1]) return;

  vector<int> active_grids;
  expl_manager_->hgrid_->getActiveGrids(active_grids);
  std::sort(active_grids.begin(), active_grids.end());
  active_grids.erase(std::unique(active_grids.begin(), active_grids.end()),
      active_grids.end());
  if (!validateGridIds(active_grids, "local component initial partition")) return;

  vector<vector<int>> assignments(local_component_members_.size());
  vector<Eigen::Vector3d> route_ends;
  route_ends.reserve(local_component_members_.size());
  for (const int member : local_component_members_)
    route_ends.push_back(local_component_positions_[member - 1]);

  // Use only this component's members and the coordinator's local HGrid.
  // Seed one nearby task per member, then grow coherent balanced routes.
  vector<int> remaining = active_grids;
  for (std::size_t index = 0; index < assignments.size() && !remaining.empty(); ++index) {
    auto best = remaining.begin();
    double best_distance = std::numeric_limits<double>::infinity();
    for (auto it = remaining.begin(); it != remaining.end(); ++it) {
      const double distance =
          (route_ends[index] - expl_manager_->hgrid_->getCenter(*it)).norm();
      if (distance < best_distance) {
        best_distance = distance;
        best = it;
      }
    }
    assignments[index].push_back(*best);
    route_ends[index] = expl_manager_->hgrid_->getCenter(*best);
    remaining.erase(best);
  }
  for (const int grid_id : remaining) {
    const Eigen::Vector3d center = expl_manager_->hgrid_->getCenter(grid_id);
    std::size_t best_member = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < assignments.size(); ++index) {
      const double cost = (route_ends[index] - center).norm() +
          initial_partition_balance_weight_m_ * assignments[index].size();
      if (cost < best_cost) {
        best_cost = cost;
        best_member = index;
      }
    }
    assignments[best_member].push_back(grid_id);
    route_ends[best_member] = center;
  }

  uint64_t previous_epoch = 0;
  auto& states = expl_manager_->ed_->swarm_state_;
  for (const int member : local_component_members_)
    previous_epoch = std::max(previous_epoch, states[member - 1].assignment_epoch_);
  if (previous_epoch >= std::numeric_limits<uint32_t>::max()) {
    ROS_ERROR("Cannot create local-component assignment after epoch %lu.",
        static_cast<unsigned long>(previous_epoch));
    return;
  }

  exploration_manager::GlobalGridAssignment assignment;
  assignment.coordinator_id = local_component_coordinator_id_;
  assignment.stamp = now_s;
  assignment.epoch = static_cast<uint32_t>(previous_epoch + 1);
  assignment.member_ids = local_component_members_;
  assignment.offsets.push_back(0);
  for (const auto& member_assignment : assignments) {
    assignment.grid_ids.insert(assignment.grid_ids.end(), member_assignment.begin(),
        member_assignment.end());
    assignment.offsets.push_back(
        static_cast<int32_t>(assignment.grid_ids.size()));
  }
  if (!applyLocalComponentAssignment(assignment)) return;
  global_assignment_pub_.publish(assignment);
  local_component_last_assignment_publish_s_ = now_s;
  ROS_WARN("RACER_LOCAL_COMPONENT_ASSIGN coordinator=%d epoch=%u members=%s grids=%zu",
      assignment.coordinator_id, assignment.epoch,
      formatIdList(local_component_members_).c_str(), assignment.grid_ids.size());
}

void FastExplorationFSM::globalCooperativeAssignmentTimerCallback() {
  using AssignmentMsg = hybrid_communication_racer::msg::HybridGlobalAssignment;
  using StateMsg = hybrid_communication_racer::msg::HybridDroneState;
  if (getId() != 1 || !fd_->trigger_ || state_ == WAIT_TRIGGER || state_ == INIT) return;
  const double now_s = ros::Time::now().toSec();
  if (last_cooperative_assignment_s_ >= 0.0 &&
      now_s - last_cooperative_assignment_s_ < cooperative_assignment_interval_s_)
    return;
  if (hybrid_states_.empty()) return;
  for (const auto& state : hybrid_states_)
    if (state.stamp <= 0.0 || now_s - state.stamp > cooperative_state_freshness_s_) return;

  // Versioned, per-UAV failures: one UAV's failure only blocks that UAV.
  // Multiple distinct reports temporarily suppress the task globally.
  for (std::size_t drone = 0; drone < hybrid_states_.size(); ++drone) {
    const auto& state = hybrid_states_[drone];
    if (state.failed_grid_id < 0 ||
        state.failure_sequence <= last_failure_sequences_[drone])
      continue;
    last_failure_sequences_[drone] = state.failure_sequence;
    auto& failure = failure_records_[state.failed_grid_id];
    ++failure.total_failures;
    ++failure.failures_by_uav[static_cast<int>(drone) + 1];
    failure.last_failure_by_uav[static_cast<int>(drone) + 1] = now_s;
    if (static_cast<int>(failure.failures_by_uav.size()) >= unreachable_distinct_uavs_ ||
        failure.total_failures >= unreachable_global_failures_) {
      failure.confirmed_until = now_s + unreachable_retry_s_;
      ROS_WARN("RACER_HYBRID_GLOBAL_UNREACHABLE grid=%d distinct_uavs=%zu failures=%d "
               "retry_after=%.3f",
          state.failed_grid_id, failure.failures_by_uav.size(), failure.total_failures,
          failure.confirmed_until);
    }
  }
  for (auto& entry : failure_records_)
    if (entry.second.confirmed_until > 0.0 && entry.second.confirmed_until <= now_s)
      entry.second = FailureRecord();

  // Merge and de-duplicate every UAV's reported grid pool with UAV 1's
  // freshly rebuilt pool from the fused perfect-communication map.
  vector<int> coordinator_pool;
  expl_manager_->updateGlobalGridPool(fd_->odom_pos_, coordinator_pool);
  std::unordered_set<int> pool_set(coordinator_pool.begin(), coordinator_pool.end());
  for (const auto& state : hybrid_states_)
    pool_set.insert(state.active_grid_ids.begin(), state.active_grid_ids.end());
  vector<int> global_pool(pool_set.begin(), pool_set.end());
  std::sort(global_pool.begin(), global_pool.end());
  global_pool.erase(std::remove_if(global_pool.begin(), global_pool.end(), [&](int grid_id) {
    return !expl_manager_->hgrid_->isValidGridId(grid_id) ||
        expl_manager_->hgrid_->getUnknownCellsNum(grid_id) <= 0;
  }), global_pool.end());

  // Estimate marginal per-UAV coverage without touching SDFMap: attribute the
  // decrease in a grid's unknown count to its previous assignment owner.
  std::unordered_map<int, int> current_unknown_by_grid;
  for (const auto& previous : previous_grid_unknown_) {
    if (!expl_manager_->hgrid_->isValidGridId(previous.first)) continue;
    const int current = std::max(0, expl_manager_->hgrid_->getUnknownCellsNum(previous.first));
    current_unknown_by_grid[previous.first] = current;
    const int delta = std::max(0, previous.second - current);
    const auto owner = previous_grid_owner_.find(previous.first);
    if (delta > 0 && owner != previous_grid_owner_.end() && owner->second > 0 &&
        owner->second <= static_cast<int>(coverage_contributions_.size()))
      coverage_contributions_[owner->second - 1] += delta;
  }
  double current_unknown = 0.0;
  for (const int grid_id : global_pool) {
    const int unknown = std::max(0, expl_manager_->hgrid_->getUnknownCellsNum(grid_id));
    current_unknown_by_grid[grid_id] = unknown;
    current_unknown += unknown;
  }
  previous_grid_unknown_ = std::move(current_unknown_by_grid);
  initial_unknown_voxels_ = std::max(initial_unknown_voxels_, current_unknown);
  const double instantaneous_coverage = initial_unknown_voxels_ <= 0.0 ? 1.0 :
      std::clamp(1.0 - current_unknown / initial_unknown_voxels_, 0.0, 1.0);
  // New frontier regions can temporarily enlarge the active unknown pool.
  // Keep its explored-fraction proxy monotonic so R(C) never contracts later
  // in the mission merely because the shared map exposed another region.
  global_coverage_ratio_ = std::max(global_coverage_ratio_, instantaneous_coverage);
  const double coverage_ratio = global_coverage_ratio_;
  adaptive_distance_limit_ = hybrid_communication_racer::adaptiveDistanceLimit(
      coverage_ratio, distance_min_, distance_max_, distance_gamma_);

  vector<int> eligible_pool;
  for (const int grid_id : global_pool) {
    const auto failure = failure_records_.find(grid_id);
    if (failure == failure_records_.end() || failure->second.confirmed_until <= now_s)
      eligible_pool.push_back(grid_id);
  }
  if (eligible_pool.empty())
    ++empty_global_pool_cycles_;
  else
    empty_global_pool_cycles_ = 0;
  const bool global_finish =
      empty_global_pool_cycles_ >= std::max(1, finish_confirmation_cycles_);

  vector<hybrid_communication_racer::GridTask> raw_tasks;
  for (const int grid_id : eligible_pool) {
    const auto seen = task_first_seen_s_.emplace(grid_id, now_s);
    const auto center = expl_manager_->hgrid_->getCenter(grid_id);
    raw_tasks.push_back({grid_id, {center[0], center[1], center[2]},
        static_cast<double>(std::max(0, expl_manager_->hgrid_->getUnknownCellsNum(grid_id))),
        std::max(0.0, now_s - seen.first->second), {grid_id}});
  }
  const auto clustered_tasks =
      hybrid_communication_racer::clusterTasks(raw_tasks, cluster_distance_);

  const std::size_t drone_count = hybrid_states_.size();
  vector<vector<int>> assignments(drone_count);
  vector<int> primary_ids(drone_count, -1);
  vector<Eigen::Vector3d> primary_targets(drone_count, Eigen::Vector3d::Zero());
  vector<double> utilities(drone_count, 0.0), gains(drone_count, 0.0),
      travel_times(drone_count, 0.0), overlaps(drone_count, 0.0);
  vector<uint8_t> takeover_flags(drone_count, 0);
  vector<bool> used_tasks(clustered_tasks.size(), false);
  vector<hybrid_communication_racer::Point3> occupied_targets;
  vector<hybrid_communication_racer::Agent> agents;
  vector<std::size_t> agent_to_drone;

  // Preserve a healthy committed cluster; HELP/idle/failed agents enter the
  // joint matching below. Startup deliberately reassigns every agent.
  for (std::size_t drone = 0; drone < drone_count; ++drone) {
    const auto& state = hybrid_states_[drone];
    std::size_t task_index = clustered_tasks.size();
    if (last_applied_cooperative_epoch_ > 0 && state.has_target &&
        !state.requesting_reassignment && state.task_status == StateMsg::TASK_ACTIVE) {
      for (std::size_t index = 0; index < clustered_tasks.size(); ++index)
        if (!used_tasks[index] && std::find(clustered_tasks[index].member_ids.begin(),
            clustered_tasks[index].member_ids.end(), state.primary_grid_id) !=
            clustered_tasks[index].member_ids.end()) {
          task_index = index;
          break;
        }
    }
    if (task_index < clustered_tasks.size()) {
      const auto& task = clustered_tasks[task_index];
      used_tasks[task_index] = true;
      // Keep the committed member list stable while its primary task remains
      // healthy. Frontier refreshes can slightly change cluster membership on
      // every epoch; propagating those harmless changes as reallocations would
      // repeatedly interrupt the unchanged local RACER planner.
      for (const int grid_id : state.assigned_grid_ids)
        if (std::find(task.member_ids.begin(), task.member_ids.end(), grid_id) !=
            task.member_ids.end())
          assignments[drone].push_back(grid_id);
      if (assignments[drone].empty() ||
          std::find(assignments[drone].begin(), assignments[drone].end(),
              state.primary_grid_id) == assignments[drone].end())
        assignments[drone] = task.member_ids;
      primary_ids[drone] = state.primary_grid_id;
      primary_targets[drone] = Eigen::Vector3d(state.current_target[0],
          state.current_target[1], state.current_target[2]);
      utilities[drone] = state.target_utility;
      gains[drone] = state.estimated_information_gain;
      travel_times[drone] = state.estimated_travel_time;
      overlaps[drone] = state.overlap_penalty;
      occupied_targets.push_back({primary_targets[drone][0], primary_targets[drone][1],
          primary_targets[drone][2]});
    } else {
      agents.push_back({static_cast<int>(drone) + 1,
          {state.position[0], state.position[1], state.position[2]}});
      agent_to_drone.push_back(drone);
    }
  }

  vector<hybrid_communication_racer::GridTask> tasks;
  for (std::size_t index = 0; index < clustered_tasks.size(); ++index)
    if (!used_tasks[index]) tasks.push_back(clustered_tasks[index]);
  vector<hybrid_communication_racer::CandidateMetric> metrics;
  metrics.reserve(agents.size() * tasks.size());
  for (const auto& agent : agents) {
    for (const auto& task : tasks) {
      bool failed_recently = false;
      for (const int member : task.member_ids) {
        const auto failure = failure_records_.find(member);
        if (failure == failure_records_.end()) continue;
        const auto failed = failure->second.last_failure_by_uav.find(agent.id);
        if (failed != failure->second.last_failure_by_uav.end() &&
            now_s - failed->second < unreachable_retry_s_) failed_recently = true;
      }
      double path_cost = std::numeric_limits<double>::infinity();
      if (!failed_recently)
        for (const int member : task.member_ids)
          path_cost = std::min(path_cost, expl_manager_->estimateGridTravelCost(
              Eigen::Vector3d(agent.position.x, agent.position.y, agent.position.z), member));
      metrics.push_back({path_cost / std::max(1.0e-3, nominal_velocity_),
          hybrid_communication_racer::distance(agent.position, task.center)});
    }
  }

  hybrid_communication_racer::MatcherConfig config;
  config.gain_scale = utility_gain_scale_;
  config.epsilon = utility_epsilon_;
  config.lambda_overlap = lambda_overlap_;
  config.lambda_age = lambda_age_;
  config.overlap_distance = overlap_distance_;
  config.maximum_distance = adaptive_distance_limit_;
  struct SelectedMatch {
    std::size_t agent;
    std::size_t task;
    hybrid_communication_racer::Match metric;
  };
  vector<SelectedMatch> selected;
  const auto near_matches = hybrid_communication_racer::greedyMaximumWeightMatching(
      agents, tasks, metrics, occupied_targets, config);
  vector<bool> matched_agents(agents.size(), false), matched_tasks(tasks.size(), false);
  for (const auto& match : near_matches) {
    matched_agents[match.agent_index] = true;
    matched_tasks[match.task_index] = true;
    selected.push_back({match.agent_index, match.task_index, match});
    occupied_targets.push_back(tasks[match.task_index].center);
  }

  // R is a priority limit. If it would leave an otherwise useful UAV idle,
  // retry only the unmatched rows/columns up to R_max.
  if (distance_fallback_when_idle_ && adaptive_distance_limit_ < distance_max_) {
    vector<hybrid_communication_racer::Agent> remaining_agents;
    vector<hybrid_communication_racer::GridTask> remaining_tasks;
    vector<std::size_t> remaining_agent_map, remaining_task_map;
    for (std::size_t i = 0; i < agents.size(); ++i)
      if (!matched_agents[i]) { remaining_agent_map.push_back(i); remaining_agents.push_back(agents[i]); }
    for (std::size_t i = 0; i < tasks.size(); ++i)
      if (!matched_tasks[i]) { remaining_task_map.push_back(i); remaining_tasks.push_back(tasks[i]); }
    vector<hybrid_communication_racer::CandidateMetric> remaining_metrics;
    for (const std::size_t agent : remaining_agent_map)
      for (const std::size_t task : remaining_task_map)
        remaining_metrics.push_back(metrics[agent * tasks.size() + task]);
    config.maximum_distance = distance_max_;
    const auto far_matches = hybrid_communication_racer::greedyMaximumWeightMatching(
        remaining_agents, remaining_tasks, remaining_metrics, occupied_targets, config);
    if (!far_matches.empty())
      ROS_WARN("RACER_HYBRID_DISTANCE_FALLBACK R=%.3f R_max=%.3f matched=%zu",
          adaptive_distance_limit_, distance_max_, far_matches.size());
    for (const auto& match : far_matches)
      selected.push_back({remaining_agent_map[match.agent_index],
          remaining_task_map[match.task_index], match});
  }

  for (const auto& selected_match : selected) {
    const std::size_t drone = agent_to_drone[selected_match.agent];
    const auto& task = tasks[selected_match.task];
    const auto& match = selected_match.metric;
    assignments[drone] = task.member_ids;
    primary_ids[drone] = task.id;
    primary_targets[drone] = Eigen::Vector3d(task.center.x, task.center.y, task.center.z);
    utilities[drone] = match.utility;
    gains[drone] = match.gain;
    travel_times[drone] = match.travel_time;
    overlaps[drone] = match.overlap_penalty;
    for (const int member : task.member_ids) {
      const auto failure = failure_records_.find(member);
      if (failure == failure_records_.end()) continue;
      for (const auto& failed : failure->second.failures_by_uav)
        if (failed.first != static_cast<int>(drone) + 1) takeover_flags[drone] = 1;
    }
    ROS_WARN("RACER_HYBRID_TARGET uav=%zu grid=%d target=[%.3f,%.3f,%.3f] "
             "utility=%.6f gain=%.1f travel_time=%.3f overlap=%.3f R=%.3f takeover=%u",
        drone + 1, task.id, task.center.x, task.center.y, task.center.z, match.utility,
        match.gain, match.travel_time, match.overlap_penalty,
        adaptive_distance_limit_, takeover_flags[drone]);
  }

  AssignmentMsg msg;
  msg.coordinator_id = 1;
  msg.stamp = now_s;
  msg.epoch = ++cooperative_assignment_epoch_;
  msg.coverage_ratio = coverage_ratio;
  msg.adaptive_distance_limit = adaptive_distance_limit_;
  msg.global_finish = global_finish;
  msg.global_pool_grid_ids = global_pool;
  msg.cluster_offsets.push_back(0);
  for (const auto& task : clustered_tasks) {
    msg.cluster_representative_ids.push_back(task.id);
    msg.cluster_grid_ids.insert(
        msg.cluster_grid_ids.end(), task.member_ids.begin(), task.member_ids.end());
    msg.cluster_offsets.push_back(static_cast<int32_t>(msg.cluster_grid_ids.size()));
  }
  msg.assignment_offsets.push_back(0);
  for (std::size_t drone = 0; drone < drone_count; ++drone) {
    msg.assigned_grid_ids.insert(msg.assigned_grid_ids.end(),
        assignments[drone].begin(), assignments[drone].end());
    msg.assignment_offsets.push_back(static_cast<int32_t>(msg.assigned_grid_ids.size()));
    msg.primary_grid_ids.push_back(primary_ids[drone]);
    msg.primary_targets.insert(msg.primary_targets.end(),
        {float(primary_targets[drone][0]), float(primary_targets[drone][1]),
            float(primary_targets[drone][2])});
    msg.utilities.push_back(utilities[drone]);
    msg.estimated_information_gains.push_back(gains[drone]);
    msg.estimated_travel_times.push_back(travel_times[drone]);
    msg.overlap_penalties.push_back(overlaps[drone]);
    msg.takeover_flags.push_back(takeover_flags[drone] != 0);
  }
  msg.coverage_contributions = coverage_contributions_;

  previous_grid_owner_.clear();
  for (std::size_t drone = 0; drone < assignments.size(); ++drone)
    for (const int grid_id : assignments[drone])
      previous_grid_owner_[grid_id] = static_cast<int>(drone) + 1;
  applyHybridAssignment(msg);
  hybrid_assignment_pub_.publish(msg);
  last_cooperative_assignment_s_ = now_s;
  ROS_WARN("RACER_HYBRID_POOL epoch=%lu pool=%zu eligible=%zu clusters=%zu locked=%zu "
           "matched=%zu coverage=%.6f R=%.3f empty_cycles=%d global_finish=%d",
      static_cast<unsigned long>(msg.epoch), global_pool.size(), eligible_pool.size(),
      clustered_tasks.size(), occupied_targets.size() - selected.size(), selected.size(),
      coverage_ratio, adaptive_distance_limit_, empty_global_pool_cycles_, global_finish);
}

#ifdef RACER_ORACLE_VARIANT
void FastExplorationFSM::globalAssignmentTimerCallback() {
  if (getId() != 1 || state_ == WAIT_TRIGGER) return;

  const double now_s = ros::Time::now().toSec();
  if (last_global_assignment_time_s_ >= 0.0 &&
      now_s - last_global_assignment_time_s_ < global_assignment_interval_s_) {
    return;
  }

  auto& states = expl_manager_->ed_->swarm_state_;
  if (states.empty()) return;
  for (const auto& state : states) {
    if (state.stamp_ <= 0.0 || now_s - state.stamp_ > 0.2) return;
  }

  vector<int> active_grids;
  expl_manager_->hgrid_->getActiveGrids(active_grids);
  if (active_grids.empty()) return;
  std::sort(active_grids.begin(), active_grids.end());
  active_grids.erase(std::unique(active_grids.begin(), active_grids.end()),
      active_grids.end());

  vector<vector<int>> assignments(states.size());
  vector<Eigen::Vector3d> route_ends;
  route_ends.reserve(states.size());
  for (const auto& state : states) route_ends.push_back(state.pos_);

  // Greedy global allocation over the shared HGrid.  The route-end update
  // rewards spatially coherent assignments, while the load term prevents a
  // single nearby UAV from claiming the whole map.
  for (const int grid_id : active_grids) {
    const Eigen::Vector3d center = expl_manager_->hgrid_->getCenter(grid_id);
    int best_drone = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int drone = 0; drone < static_cast<int>(states.size()); ++drone) {
      const double cost = (route_ends[drone] - center).norm() +
          global_balance_weight_m_ * assignments[drone].size();
      if (cost < best_cost) {
        best_cost = cost;
        best_drone = drone;
      }
    }
    assignments[best_drone].push_back(grid_id);
    route_ends[best_drone] = center;
  }

  exploration_manager::GlobalGridAssignment msg;
  msg.coordinator_id = 1;
  msg.stamp = now_s;
  msg.epoch = ++global_assignment_epoch_;
  msg.offsets.reserve(assignments.size() + 1);
  msg.offsets.push_back(0);
  for (const auto& assignment : assignments) {
    msg.grid_ids.insert(msg.grid_ids.end(), assignment.begin(), assignment.end());
    msg.offsets.push_back(static_cast<int32_t>(msg.grid_ids.size()));
  }

  applyGlobalAssignment(msg);
  global_assignment_pub_.publish(msg);
  last_global_assignment_time_s_ = now_s;
  ROS_WARN_THROTTLE(2.0,
      "RACER_ORACLE_GLOBAL_ASSIGNMENT epoch=%u active=%zu drones=%zu",
      msg.epoch, active_grids.size(), assignments.size());
}
#endif

void FastExplorationFSM::globalAssignmentMsgCallback(
    const exploration_manager::GlobalGridAssignmentConstPtr& msg) {
  if (globalCooperativeMode() || !msg) return;
  if (localComponentMode()) {
    if (msg->member_ids.empty()) return;
    applyLocalComponentAssignment(*msg);
    return;
  }
  if (msg->coordinator_id != 1 || !msg->member_ids.empty()) return;
  applyGlobalAssignment(*msg);
}

bool FastExplorationFSM::applyGlobalAssignment(
    const exploration_manager::GlobalGridAssignment& msg) {
  auto& states = expl_manager_->ed_->swarm_state_;
  const bool recovery =
      msg.purpose == exploration_manager::GlobalGridAssignment::PURPOSE_RECOVERY;
  if (waitingForInitialAssignment() && recovery) return false;
  bool bootstrap_allowed = initial_partition_enabled_;
#ifdef RACER_ORACLE_VARIANT
  bootstrap_allowed = true;
#endif
  if ((!recovery && !bootstrap_allowed) || !msg.member_ids.empty() ||
      msg.epoch <= last_applied_global_assignment_epoch_ ||
      msg.offsets.size() != states.size() + 1 || msg.offsets.front() != 0 ||
      msg.offsets.back() != static_cast<int32_t>(msg.grid_ids.size())) {
    if (!recovery && !bootstrap_allowed)
      ROS_WARN_THROTTLE(2.0,
          "Ignore GlobalGridAssignment bootstrap because initial partition is disabled.");
    return false;
  }
  if (recovery && !pairwiseGlobalRecoveryMode()) return false;

  vector<int> recovery_uav_ids(msg.recovery_uav_ids.begin(), msg.recovery_uav_ids.end());
  std::sort(recovery_uav_ids.begin(), recovery_uav_ids.end());
  vector<int> affected_uav_ids(msg.affected_uav_ids.begin(), msg.affected_uav_ids.end());
  std::sort(affected_uav_ids.begin(), affected_uav_ids.end());
  if (recovery && (recovery_uav_ids.empty() ||
      std::adjacent_find(recovery_uav_ids.begin(), recovery_uav_ids.end()) !=
          recovery_uav_ids.end() || recovery_uav_ids.front() <= 0 ||
      recovery_uav_ids.back() > static_cast<int>(states.size()) ||
      affected_uav_ids.empty() ||
      std::adjacent_find(affected_uav_ids.begin(), affected_uav_ids.end()) !=
          affected_uav_ids.end() || affected_uav_ids.front() <= 0 ||
      affected_uav_ids.back() > static_cast<int>(states.size()) ||
      msg.base_assignment_epochs.size() != states.size())) {
    ROS_ERROR("Reject malformed global recovery epoch %u requester scope.", msg.epoch);
    return false;
  }
  if (recovery) {
    for (const int requester : recovery_uav_ids) {
      if (!std::binary_search(affected_uav_ids.begin(), affected_uav_ids.end(), requester)) {
        ROS_ERROR("Reject global recovery epoch %u: requester UAV %d is not affected.",
            msg.epoch, requester);
        return false;
      }
    }
  }
  vector<int> all_grid_ids(msg.grid_ids.begin(), msg.grid_ids.end());
  if (!validateGridIds(all_grid_ids, "global assignment")) return false;
  auto unique_grid_ids = all_grid_ids;
  std::sort(unique_grid_ids.begin(), unique_grid_ids.end());
  if (std::adjacent_find(unique_grid_ids.begin(), unique_grid_ids.end()) !=
      unique_grid_ids.end()) {
    ROS_ERROR("Reject global assignment epoch %u with duplicate grid ownership.", msg.epoch);
    return false;
  }
  for (std::size_t drone = 0; drone < states.size(); ++drone) {
    const int32_t begin = msg.offsets[drone];
    const int32_t end = msg.offsets[drone + 1];
    if (begin < 0 || end < begin ||
        end > static_cast<int32_t>(msg.grid_ids.size())) return false;
  }

  vector<vector<int>> next_assignments(states.size());
  vector<int> affected_indices;
  std::unordered_set<int> affected_set;
  for (std::size_t drone = 0; drone < states.size(); ++drone) {
    const auto begin = msg.grid_ids.begin() + msg.offsets[drone];
    const auto end = msg.grid_ids.begin() + msg.offsets[drone + 1];
    next_assignments[drone] = vector<int>(begin, end);
    if (!recovery || std::binary_search(affected_uav_ids.begin(),
                         affected_uav_ids.end(), static_cast<int>(drone) + 1)) {
      affected_indices.push_back(static_cast<int>(drone));
      affected_set.insert(static_cast<int>(drone));
    }
  }

  if (recovery) {
    // Apply the recovery as a compare-and-swap against the coordinator's
    // ownership snapshot. The local UAV's own epoch must match exactly; a
    // newer peer epoch already observed locally also vetoes the recovery.
    for (const int drone : affected_indices) {
      const uint64_t current_epoch = states[drone].assignment_epoch_;
      const uint64_t base_epoch = msg.base_assignment_epochs[drone];
      const bool already_applied = current_epoch == msg.epoch &&
          states[drone].grid_ids_ == next_assignments[drone];
      if (!already_applied &&
          ((drone + 1 == getId() && current_epoch != base_epoch) ||
              current_epoch > base_epoch)) {
        ROS_WARN("Reject stale global recovery epoch=%u at UAV %d: affected UAV %d "
                 "base_epoch=%lu current_epoch=%lu.",
            msg.epoch, getId(), drone + 1,
            static_cast<unsigned long>(base_epoch),
            static_cast<unsigned long>(current_epoch));
        return false;
      }
    }

    const auto transaction_conflicts = [&](const PairTransaction& transaction) {
      return transaction.active &&
          (affected_set.find(getId() - 1) != affected_set.end() ||
              affected_set.find(transaction.peer_id - 1) != affected_set.end());
    };
    if (transaction_conflicts(outgoing_transaction_)) {
      if (outgoing_transaction_.commit_sent) {
        ROS_WARN("Defer global recovery epoch=%u until uncertain pair COMMIT %lu resolves.",
            msg.epoch, static_cast<unsigned long>(outgoing_transaction_.transaction_id));
        return false;
      }
      publishPairTransaction(exploration_manager::PairOpt::PHASE_ABORT);
      clearOutgoingTransaction("conflicting global recovery");
    }
    if (transaction_conflicts(incoming_transaction_)) {
      ROS_WARN("Cancel prepared pair transaction %lu due to conflicting global recovery %u.",
          static_cast<unsigned long>(incoming_transaction_.transaction_id), msg.epoch);
      incoming_transaction_ = PairTransaction();
    }
  }

  bool ego_changed = false;
  const double commitment_until = msg.stamp + fp_->assignment_commitment_duration_;
  for (const int drone : affected_indices) {
    if (drone + 1 == getId() && next_assignments[drone] != states[drone].grid_ids_) {
      ego_changed = true;
    }
    states[drone].grid_ids_ = std::move(next_assignments[drone]);
    states[drone].recent_interact_time_ = msg.stamp;
    states[drone].assignment_epoch_ = msg.epoch;
    states[drone].commitment_until_ = commitment_until;
    states[drone].requesting_work_ = false;
  }
  if (!recovery) {
    outgoing_transaction_ = PairTransaction();
    incoming_transaction_ = PairTransaction();
  }
  if (!recovery && blocking_initial_assignment_ &&
      !states[getId() - 1].grid_ids_.empty()) {
    blocking_initial_assignment_received_ = true;
    ROS_WARN("RACER_BLOCKING_INITIAL_RECEIVED uav=%d time_s=%.3f epoch=%u grids=%zu",
        getId(), ros::Time::now().toSec(), msg.epoch, states[getId() - 1].grid_ids_.size());
  }
  last_applied_global_assignment_epoch_ = msg.epoch;
  global_assignment_epoch_ = std::max(global_assignment_epoch_, msg.epoch);
  const bool ego_affected = affected_set.find(getId() - 1) != affected_set.end();
  if (ego_affected) last_reported_assignment_epochs_[getId() - 1] = msg.epoch;
  if (ego_changed) {
    expl_manager_->ed_->reallocated_ = true;
  }

  if (recovery && msg.global_finish) {
    const auto previous_mode = task_coordination_gate_.mode();
    task_coordination_gate_.globalFinished();
    logTaskMode("global completion confirmed", previous_mode);
    fd_->static_state_ = true;
    fd_->go_back_ = false;
    replan_pub_.publish(std_msgs::Empty());
    if (state_ != FINISH) transitState(FINISH, "globalFinishConfirmed");
    return true;
  }

  if (recovery &&
      std::binary_search(recovery_uav_ids.begin(), recovery_uav_ids.end(), getId())) {
    const auto previous_mode = task_coordination_gate_.mode();
    if (!states[getId() - 1].grid_ids_.empty()) {
      task_coordination_gate_.globalRecovered();
      idle_since_s_ = -1.0;
      fd_->go_back_ = false;
      consecutive_plan_failures_ = 0;
      logTaskMode("global recovery assigned work", previous_mode);
      if (state_ == IDLE || state_ == FINISH)
        transitState(PLAN_TRAJ, "globalRecoveryAssignment");
    } else {
      task_coordination_gate_.globalWait();
      logTaskMode("global recovery found no transferable task", previous_mode);
      if (state_ != IDLE) transitState(IDLE, "globalRecoveryWait");
    }
  } else if (!recovery && ego_changed &&
      (state_ == IDLE || state_ == FINISH) &&
      !states[getId() - 1].grid_ids_.empty()) {
    fd_->go_back_ = false;
    consecutive_plan_failures_ = 0;
    transitState(PLAN_TRAJ, "globalAssignmentMsgCallback");
  }
  return true;
}

bool FastExplorationFSM::applyLocalComponentAssignment(
    const exploration_manager::GlobalGridAssignment& msg) {
  if (!localComponentMode() || msg.member_ids.empty()) return false;

  vector<int> members(msg.member_ids.begin(), msg.member_ids.end());
  const int drone_num = static_cast<int>(expl_manager_->ed_->swarm_state_.size());
  if (!std::is_sorted(members.begin(), members.end()) ||
      std::adjacent_find(members.begin(), members.end()) != members.end() ||
      members.front() <= 0 || members.back() > drone_num ||
      msg.coordinator_id != members.front() ||
      !std::binary_search(members.begin(), members.end(), getId()) ||
      members != local_component_members_ || msg.offsets.size() != members.size() + 1 ||
      msg.offsets.empty() || msg.offsets.front() != 0 ||
      msg.offsets.back() != static_cast<int32_t>(msg.grid_ids.size())) {
    ROS_WARN_THROTTLE(1.0,
        "Reject local-component assignment at UAV %d: scope does not match discovered component.",
        getId());
    return false;
  }
  if (local_component_assignment_applied_) {
    // Idempotent repeats are expected while ACKs traverse the component.
    if (msg.epoch == local_component_assignment_epoch_ &&
        msg.coordinator_id == local_component_coordinator_id_ &&
        msg.member_ids == local_component_assignment_msg_.member_ids &&
        msg.offsets == local_component_assignment_msg_.offsets &&
        msg.grid_ids == local_component_assignment_msg_.grid_ids) {
      local_component_assignment_msg_ = msg;
      return true;
    }
    return false;
  }
  if (msg.epoch == 0) return false;

  vector<int> all_grid_ids(msg.grid_ids.begin(), msg.grid_ids.end());
  if (!validateGridIds(all_grid_ids, "local component assignment")) return false;
  auto unique_grid_ids = all_grid_ids;
  std::sort(unique_grid_ids.begin(), unique_grid_ids.end());
  if (std::adjacent_find(unique_grid_ids.begin(), unique_grid_ids.end()) !=
      unique_grid_ids.end()) {
    ROS_ERROR("Reject local-component assignment epoch %u with duplicate grid ownership.",
        msg.epoch);
    return false;
  }
  for (std::size_t index = 0; index < members.size(); ++index) {
    const int32_t begin = msg.offsets[index];
    const int32_t end = msg.offsets[index + 1];
    if (begin < 0 || end < begin ||
        end > static_cast<int32_t>(msg.grid_ids.size()))
      return false;
  }

  auto& states = expl_manager_->ed_->swarm_state_;
  bool ego_changed = false;
  const double commitment_until = msg.stamp + fp_->assignment_commitment_duration_;
  for (std::size_t index = 0; index < members.size(); ++index) {
    const int member = members[index];
    const auto begin = msg.grid_ids.begin() + msg.offsets[index];
    const auto end = msg.grid_ids.begin() + msg.offsets[index + 1];
    vector<int> next(begin, end);
    if (member == getId() && next != states[member - 1].grid_ids_)
      ego_changed = true;
    states[member - 1].grid_ids_ = std::move(next);
    states[member - 1].recent_interact_time_ = msg.stamp;
    states[member - 1].assignment_epoch_ = msg.epoch;
    states[member - 1].commitment_until_ = commitment_until;
    states[member - 1].requesting_work_ = states[member - 1].grid_ids_.empty();
  }

  outgoing_transaction_ = PairTransaction();
  incoming_transaction_ = PairTransaction();
  local_component_membership_frozen_ = true;
  local_component_assignment_applied_ = true;
  local_component_coordinator_id_ = msg.coordinator_id;
  local_component_assignment_epoch_ = msg.epoch;
  local_component_assignment_msg_ = msg;
  local_component_ack_ids_.clear();
  local_component_ack_ids_.insert(getId());
  last_reported_assignment_epochs_[getId() - 1] = msg.epoch;
  if (ego_changed) expl_manager_->ed_->reallocated_ = true;

  auto& ego = states[getId() - 1];
  if (!ego.grid_ids_.empty() && (state_ == IDLE || state_ == FINISH)) {
    fd_->go_back_ = false;
    consecutive_plan_failures_ = 0;
    transitState(PLAN_TRAJ, "localComponentAssignment");
  }
  ROS_WARN("RACER_LOCAL_COMPONENT_APPLIED uav=%d coordinator=%d epoch=%u grids=%zu",
      getId(), msg.coordinator_id, msg.epoch, ego.grid_ids_.size());
  return true;
}

void FastExplorationFSM::hybridAssignmentMsgCallback(
    const hybrid_communication_racer::msg::HybridGlobalAssignment::ConstSharedPtr& msg) {
  if (!globalCooperativeMode() || !msg || msg->coordinator_id != 1) return;
  applyHybridAssignment(*msg);
}

void FastExplorationFSM::applyHybridAssignment(
    const hybrid_communication_racer::msg::HybridGlobalAssignment& msg) {
  auto& states = expl_manager_->ed_->swarm_state_;
  const std::size_t drone_count = states.size();
  if (msg.epoch <= last_applied_cooperative_epoch_ ||
      msg.assignment_offsets.size() != drone_count + 1 ||
      msg.assignment_offsets.front() != 0 ||
      msg.assignment_offsets.back() != static_cast<int32_t>(msg.assigned_grid_ids.size()) ||
      msg.primary_grid_ids.size() != drone_count ||
      msg.primary_targets.size() != 3 * drone_count || msg.utilities.size() != drone_count ||
      msg.estimated_information_gains.size() != drone_count ||
      msg.estimated_travel_times.size() != drone_count ||
      msg.overlap_penalties.size() != drone_count || msg.takeover_flags.size() != drone_count ||
      msg.coverage_contributions.size() != drone_count) {
    ROS_ERROR("Reject malformed hybrid assignment epoch %lu.",
        static_cast<unsigned long>(msg.epoch));
    return;
  }
  vector<int> all_grid_ids(msg.assigned_grid_ids.begin(), msg.assigned_grid_ids.end());
  if (!validateGridIds(all_grid_ids, "hybrid assignment")) return;
  auto unique_grid_ids = all_grid_ids;
  std::sort(unique_grid_ids.begin(), unique_grid_ids.end());
  if (std::adjacent_find(unique_grid_ids.begin(), unique_grid_ids.end()) !=
      unique_grid_ids.end()) {
    ROS_ERROR("Reject hybrid assignment epoch %lu with duplicate grid ownership.",
        static_cast<unsigned long>(msg.epoch));
    return;
  }
  for (std::size_t drone = 0; drone < drone_count; ++drone) {
    const int32_t begin = msg.assignment_offsets[drone];
    const int32_t end = msg.assignment_offsets[drone + 1];
    if (begin < 0 || end < begin ||
        end > static_cast<int32_t>(msg.assigned_grid_ids.size())) return;
    if (msg.primary_grid_ids[drone] >= 0 &&
        std::find(msg.assigned_grid_ids.begin() + begin,
            msg.assigned_grid_ids.begin() + end, msg.primary_grid_ids[drone]) ==
            msg.assigned_grid_ids.begin() + end) {
      ROS_ERROR("Reject hybrid assignment: UAV %zu primary grid is outside its cluster.",
          drone + 1);
      return;
    }
  }

  const std::size_t ego_index = static_cast<std::size_t>(getId() - 1);
  bool ego_changed = false;
  for (std::size_t drone = 0; drone < drone_count; ++drone) {
    const auto begin = msg.assigned_grid_ids.begin() + msg.assignment_offsets[drone];
    const auto end = msg.assigned_grid_ids.begin() + msg.assignment_offsets[drone + 1];
    vector<int> next(begin, end);
    // A changing set of live member grids inside the same cluster is normal as
    // frontiers are covered. Only a primary-cluster change is a target
    // reassignment that should interrupt/restart the local RACER planner.
    if (drone == ego_index && msg.primary_grid_ids[drone] != primary_grid_id_)
      ego_changed = true;
    states[drone].grid_ids_ = std::move(next);
    states[drone].assignment_epoch_ = msg.epoch;
    states[drone].commitment_until_ = msg.stamp + fp_->assignment_commitment_duration_;
    states[drone].recent_interact_time_ = msg.stamp;
    states[drone].requesting_work_ = states[drone].grid_ids_.empty() && !msg.global_finish;
  }
  outgoing_transaction_ = PairTransaction();
  incoming_transaction_ = PairTransaction();
  last_applied_cooperative_epoch_ = msg.epoch;
  cooperative_assignment_epoch_ = std::max(cooperative_assignment_epoch_, msg.epoch);
  cooperative_global_finish_ = msg.global_finish;
  adaptive_distance_limit_ = msg.adaptive_distance_limit;
  coverage_contributions_ = msg.coverage_contributions;
  primary_grid_id_ = msg.primary_grid_ids[ego_index];
  primary_target_ = Eigen::Vector3d(msg.primary_targets[3 * ego_index],
      msg.primary_targets[3 * ego_index + 1], msg.primary_targets[3 * ego_index + 2]);
  target_utility_ = msg.utilities[ego_index];
  target_information_gain_ = msg.estimated_information_gains[ego_index];
  target_travel_time_ = msg.estimated_travel_times[ego_index];
  target_overlap_penalty_ = msg.overlap_penalties[ego_index];
  current_target_takeover_ = msg.takeover_flags[ego_index];

  if (ego_changed && primary_grid_id_ >= 0) {
    ++reassignment_count_;
    if (current_target_takeover_) ++takeover_count_;
    ROS_WARN("RACER_HYBRID_REASSIGN uav=%d epoch=%lu target=%d utility=%.6f "
             "gain=%.1f travel_time=%.3f overlap=%.3f R=%.3f count=%lu takeover=%d",
        getId(), static_cast<unsigned long>(msg.epoch), primary_grid_id_, target_utility_,
        target_information_gain_, target_travel_time_, target_overlap_penalty_,
        adaptive_distance_limit_, static_cast<unsigned long>(reassignment_count_),
        current_target_takeover_);
  }
  expl_manager_->ed_->reallocated_ = ego_changed;
  if (cooperative_global_finish_) {
    fd_->static_state_ = true;
    replan_pub_.publish(std_msgs::Empty());
    if (state_ != FINISH) transitState(FINISH, "hybridGlobalFinish");
  } else if (primary_grid_id_ >= 0) {
    idle_since_s_ = -1.0;
    fd_->go_back_ = false;
    consecutive_plan_failures_ = 0;
    if (state_ == IDLE || state_ == FINISH) transitState(PLAN_TRAJ, "hybridAssignment");
  } else {
    enterHelpMode("global pool has no unoccupied task", false);
  }
}

void FastExplorationFSM::findUnallocated(const vector<int>& actives, vector<int>& missed) {
  // Create map of all active
  unordered_map<int, char> active_map;
  for (auto ativ : actives) {
    active_map[ativ] = 1;
  }

  // Remove allocated ones
  for (auto state : expl_manager_->ed_->swarm_state_) {
    for (auto id : state.grid_ids_) {
      if (active_map.find(id) != active_map.end()) {
        active_map.erase(id);
      } else {
        // ROS_ERROR("Inactive grid %d is allocated.", id);
      }
    }
  }

  missed.clear();
  for (auto p : active_map) {
    missed.push_back(p.first);
  }
}

void FastExplorationFSM::optMsgCallback(const exploration_manager::PairOptConstPtr& msg) {
  if (waitingForInitialAssignment()) return;
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;
  if (msg->from_drone_id <= 0 ||
      msg->from_drone_id > static_cast<int>(expl_manager_->ed_->swarm_state_.size()))
    return;

  auto send_response = [&](int32_t status) {
    exploration_manager::PairOptResponse response;
    response.from_drone_id = getId();
    response.to_drone_id = msg->from_drone_id;
    response.status = status;
    response.stamp = ros::Time::now().toSec();
    response.transaction_id = msg->transaction_id;
    response.responder_assignment_epoch =
        expl_manager_->ed_->swarm_state_[getId() - 1].assignment_epoch_;
    for (int i = 0; i < std::max(1, fp_->repeat_send_num_); ++i)
      opt_res_pub_.publish(response);
  };

  if (msg->phase == exploration_manager::PairOpt::PHASE_ABORT) {
    if (incoming_transaction_.active &&
        incoming_transaction_.transaction_id == msg->transaction_id)
      incoming_transaction_ = PairTransaction();
    send_response(exploration_manager::PairOptResponse::STATUS_REJECTED);
    return;
  }

  vector<int> proposer_ids;
  vector<int> responder_ids;
  std::unordered_set<int> proposer_set;
  std::unordered_set<int> responder_set;
  bool duplicate_id = false;
  for (const auto id : msg->ego_ids)
    if (proposer_set.insert(id).second)
      proposer_ids.push_back(id);
    else
      duplicate_id = true;
  for (const auto id : msg->other_ids)
    if (responder_set.insert(id).second)
      responder_ids.push_back(id);
    else
      duplicate_id = true;
  if (duplicate_id || !validateGridIds(proposer_ids, "pair proposal") ||
      !validateGridIds(responder_ids, "pair proposal")) {
    send_response(exploration_manager::PairOptResponse::STATUS_REJECTED);
    return;
  }
  std::size_t overlap_count = 0;
  for (const int id : proposer_set)
    if (responder_set.find(id) != responder_set.end()) ++overlap_count;
  if (overlap_count != 0) {
    ROS_ERROR("Reject pair transaction %lu: %zu grids have two owners.",
        static_cast<unsigned long>(msg->transaction_id), overlap_count);
    send_response(exploration_manager::PairOptResponse::STATUS_REJECTED);
    return;
  }

  auto& proposer = expl_manager_->ed_->swarm_state_[msg->from_drone_id - 1];
  auto& responder = expl_manager_->ed_->swarm_state_[getId() - 1];

  if (msg->phase == exploration_manager::PairOpt::PHASE_PROPOSE) {
    if (last_committed_transaction_id_ == msg->transaction_id &&
        last_committed_transaction_peer_ == msg->from_drone_id) {
      send_response(exploration_manager::PairOptResponse::STATUS_COMMITTED);
      return;
    }
    if (incoming_transaction_.active) {
      send_response(incoming_transaction_.transaction_id == msg->transaction_id
              ? exploration_manager::PairOptResponse::STATUS_PREPARED
              : exploration_manager::PairOptResponse::STATUS_BUSY);
      return;
    }
    const double now_s = ros::Time::now().toSec();
    const bool work_steal = proposer.requesting_work_ || responder.requesting_work_ ||
        proposer.grid_ids_.empty() || responder.grid_ids_.empty();
    if (msg->new_assignment_epoch !=
            std::max(msg->from_assignment_epoch, msg->to_assignment_epoch) + 1 ||
        (!work_steal &&
            (now_s < proposer.commitment_until_ || now_s < responder.commitment_until_)) ||
        outgoing_transaction_.active || proposer.assignment_epoch_ != msg->from_assignment_epoch ||
        responder.assignment_epoch_ != msg->to_assignment_epoch) {
      send_response(exploration_manager::PairOptResponse::STATUS_BUSY);
      return;
    }

    // The transaction must preserve the complete pair-owned union, plus only grids that this UAV
    // also sees as globally unallocated. This prevents a stale initiator from creating a third
    // UAV's duplicate ownership while still allowing RACER's missed-grid recovery.
    vector<int> active_grids, missed_grids;
    expl_manager_->hgrid_->getActiveGrids(active_grids);
    findUnallocated(active_grids, missed_grids);
    std::unordered_set<int> expected_union(
        proposer.grid_ids_.begin(), proposer.grid_ids_.end());
    expected_union.insert(responder.grid_ids_.begin(), responder.grid_ids_.end());
    expected_union.insert(missed_grids.begin(), missed_grids.end());
    std::unordered_set<int> proposed_union = proposer_set;
    proposed_union.insert(responder_set.begin(), responder_set.end());
    if (proposed_union != expected_union) {
      ROS_WARN("Reject pair transaction %lu: proposed grid union is stale or incomplete.",
          static_cast<unsigned long>(msg->transaction_id));
      send_response(exploration_manager::PairOptResponse::STATUS_BUSY);
      return;
    }

    incoming_transaction_ = PairTransaction();
    incoming_transaction_.active = true;
    incoming_transaction_.peer_id = msg->from_drone_id;
    incoming_transaction_.transaction_id = msg->transaction_id;
    incoming_transaction_.from_epoch = msg->from_assignment_epoch;
    incoming_transaction_.to_epoch = msg->to_assignment_epoch;
    incoming_transaction_.new_epoch = msg->new_assignment_epoch;
    incoming_transaction_.commitment_until = msg->commitment_until;
    incoming_transaction_.expires_at =
        ros::Time::now().toSec() + fp_->pair_transaction_timeout_;
    incoming_transaction_.ego_ids = std::move(proposer_ids);
    incoming_transaction_.other_ids = std::move(responder_ids);
    responder.recent_attempt_time_ = ros::Time::now().toSec();
    send_response(exploration_manager::PairOptResponse::STATUS_PREPARED);
    return;
  }

  if (msg->phase != exploration_manager::PairOpt::PHASE_COMMIT) {
    send_response(exploration_manager::PairOptResponse::STATUS_REJECTED);
    return;
  }
  if (last_committed_transaction_id_ == msg->transaction_id &&
      last_committed_transaction_peer_ == msg->from_drone_id) {
    send_response(exploration_manager::PairOptResponse::STATUS_COMMITTED);
    return;
  }
  if (!incoming_transaction_.active ||
      incoming_transaction_.transaction_id != msg->transaction_id ||
      proposer.assignment_epoch_ != incoming_transaction_.from_epoch ||
      responder.assignment_epoch_ != incoming_transaction_.to_epoch) {
    send_response(exploration_manager::PairOptResponse::STATUS_REJECTED);
    return;
  }

  applyCommittedPairAssignment(msg->from_drone_id, incoming_transaction_.other_ids,
      incoming_transaction_.ego_ids, incoming_transaction_.new_epoch,
      incoming_transaction_.commitment_until);
  last_committed_transaction_id_ = incoming_transaction_.transaction_id;
  last_committed_transaction_peer_ = incoming_transaction_.peer_id;
  incoming_transaction_ = PairTransaction();
  send_response(exploration_manager::PairOptResponse::STATUS_COMMITTED);
}

void FastExplorationFSM::optResMsgCallback(
    const exploration_manager::PairOptResponseConstPtr& msg) {
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;
  if (!outgoing_transaction_.active ||
      msg->from_drone_id != outgoing_transaction_.peer_id ||
      msg->transaction_id != outgoing_transaction_.transaction_id)
    return;

  if (msg->status == exploration_manager::PairOptResponse::STATUS_PREPARED) {
    if (outgoing_transaction_.commit_sent) return;
    const auto& ego = expl_manager_->ed_->swarm_state_[getId() - 1];
    if (ego.assignment_epoch_ != outgoing_transaction_.from_epoch ||
        msg->responder_assignment_epoch != outgoing_transaction_.to_epoch) {
      publishPairTransaction(exploration_manager::PairOpt::PHASE_ABORT);
      clearOutgoingTransaction("responder epoch changed before commit", true);
      return;
    }
    outgoing_transaction_.commit_sent = true;
    outgoing_transaction_.expires_at =
        ros::Time::now().toSec() + fp_->pair_transaction_timeout_;
    publishPairTransaction(exploration_manager::PairOpt::PHASE_COMMIT);
    return;
  }

  if (msg->status == exploration_manager::PairOptResponse::STATUS_COMMITTED) {
    applyCommittedPairAssignment(outgoing_transaction_.peer_id, outgoing_transaction_.ego_ids,
        outgoing_transaction_.other_ids, outgoing_transaction_.new_epoch,
        outgoing_transaction_.commitment_until);
    last_committed_transaction_id_ = outgoing_transaction_.transaction_id;
    last_committed_transaction_peer_ = outgoing_transaction_.peer_id;
    clearOutgoingTransaction("committed");
    return;
  }

  // Once COMMIT is in flight, a delayed BUSY/REJECTED response to the earlier PROPOSE cannot be
  // treated as an abort. The peer may already have committed; keep reconciling idempotently.
  if (outgoing_transaction_.commit_sent) return;

  clearOutgoingTransaction(msg->status == exploration_manager::PairOptResponse::STATUS_BUSY
          ? "peer busy or epoch mismatch"
          : "proposal rejected", true);
}

void FastExplorationFSM::swarmTrajCallback(const bspline::BsplineConstPtr& msg) {
  // Get newest trajs from other drones, for inter-drone collision avoidance
  auto& sdat = planner_manager_->swarm_traj_data_;

  // Ignore self trajectory
  if (msg->drone_id == sdat.drone_id_) return;

  // Ignore outdated trajectory
  if (sdat.receive_flags_[msg->drone_id - 1] == true &&
      ros::Time(msg->start_time).toSec() <=
        sdat.swarm_trajs_[msg->drone_id - 1].start_time_ + 1e-3)
    return;

  // Convert the msg to B-spline
  Eigen::MatrixXd pos_pts(msg->pos_pts.size(), 3);
  Eigen::VectorXd knots(msg->knots.size());
  for (int i = 0; i < msg->knots.size(); ++i) knots(i) = msg->knots[i];

  for (int i = 0; i < msg->pos_pts.size(); ++i) {
    pos_pts(i, 0) = msg->pos_pts[i].x;
    pos_pts(i, 1) = msg->pos_pts[i].y;
    pos_pts(i, 2) = msg->pos_pts[i].z;
  }

  // // Transform of drone's basecoor, optional step (skip if use swarm_pilot)
  // Eigen::Vector4d tf;
  // planner_manager_->edt_environment_->sdf_map_->getBaseCoor(msg->drone_id, tf);
  // double yaw = tf[3];
  // Eigen::Matrix3d rot;
  // rot << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  // Eigen::Vector3d trans = tf.head<3>();
  // for (int i = 0; i < pos_pts.rows(); ++i) {
  //   Eigen::Vector3d tmp = pos_pts.row(i);
  //   tmp = rot * tmp + trans;
  //   pos_pts.row(i) = tmp;
  // }

  sdat.swarm_trajs_[msg->drone_id - 1].setUniformBspline(pos_pts, msg->order, 0.1);
  sdat.swarm_trajs_[msg->drone_id - 1].setKnot(knots);
  sdat.swarm_trajs_[msg->drone_id - 1].start_time_ = ros::Time(msg->start_time).toSec();
  sdat.receive_flags_[msg->drone_id - 1] = true;

  if (state_ == EXEC_TRAJ) {
    // Check collision with received trajectory
    if (!planner_manager_->checkSwarmCollision(msg->drone_id)) {
      ROS_ERROR("Drone %d collide with drone %d.", sdat.drone_id_, msg->drone_id);
      fd_->avoid_collision_ = true;
      transitState(PLAN_TRAJ, "swarmTrajCallback");
    }
  }
}

void FastExplorationFSM::swarmTrajTimerCallback(const ros::TimerEvent& e) {
  // Broadcast newest traj of this drone to others
  if (state_ == EXEC_TRAJ) {
    swarm_traj_pub_.publish(fd_->newest_traj_);

  } else if (state_ == WAIT_TRIGGER) {
    // Publish a virtual traj at current pose, to avoid collision
    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = ros::Time::now();
    bspline.traj_id = planner_manager_->local_data_.traj_id_;

    Eigen::MatrixXd pos_pts(4, 3);
    for (int i = 0; i < 4; ++i) pos_pts.row(i) = fd_->odom_pos_.transpose();

    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }

    NonUniformBspline tmp(pos_pts, planner_manager_->pp_.bspline_degree_, 1.0);
    Eigen::VectorXd knots = tmp.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    bspline.drone_id = expl_manager_->ep_->drone_id_;
    swarm_traj_pub_.publish(bspline);
  }
}

}  // namespace fast_planner
