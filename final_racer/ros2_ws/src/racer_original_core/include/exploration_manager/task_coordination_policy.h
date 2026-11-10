#ifndef _TASK_COORDINATION_POLICY_H_
#define _TASK_COORDINATION_POLICY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fast_planner {

enum class TaskCoordinationMode {
  NORMAL,
  PAIR_SEARCH,
  GLOBAL_RECOVERY,
  FINISH,
};

inline const char* taskCoordinationModeName(TaskCoordinationMode mode) {
  switch (mode) {
    case TaskCoordinationMode::NORMAL:
      return "NORMAL";
    case TaskCoordinationMode::PAIR_SEARCH:
      return "PAIR_SEARCH";
    case TaskCoordinationMode::GLOBAL_RECOVERY:
      return "GLOBAL_RECOVERY";
    case TaskCoordinationMode::FINISH:
      return "FINISH";
  }
  return "UNKNOWN";
}

// Cycle-based gate between normal pairwise allocation and the exceptional
// global recovery path. A local task shortage never implies global finish.
class TaskCoordinationGate {
public:
  void configure(int pair_failure_threshold, int recovery_cooldown_cycles) {
    pair_failure_threshold_ = std::max(1, pair_failure_threshold);
    recovery_cooldown_cycles_ = std::max(0, recovery_cooldown_cycles);
  }

  TaskCoordinationMode mode() const { return mode_; }
  int pairFailureCount() const { return consecutive_pair_failures_; }
  int pairAttemptCount() const { return pair_attempt_count_; }
  int cooldownRemaining() const { return cooldown_remaining_; }

  void localTaskAvailable() {
    mode_ = TaskCoordinationMode::NORMAL;
    consecutive_pair_failures_ = 0;
    pair_attempt_count_ = 0;
  }

  void localTaskEmpty() {
    if (mode_ == TaskCoordinationMode::NORMAL)
      mode_ = TaskCoordinationMode::PAIR_SEARCH;
  }

  // Returns true exactly on the cycle that should issue a recovery request.
  bool pairAttemptFailed() {
    if (mode_ == TaskCoordinationMode::FINISH ||
        mode_ == TaskCoordinationMode::GLOBAL_RECOVERY)
      return false;
    mode_ = TaskCoordinationMode::PAIR_SEARCH;
    ++consecutive_pair_failures_;
    if (cooldown_remaining_ > 0) --cooldown_remaining_;
    if (pair_attempt_count_ > 0 && cooldown_remaining_ == 0 &&
        consecutive_pair_failures_ >= pair_failure_threshold_) {
      mode_ = TaskCoordinationMode::GLOBAL_RECOVERY;
      return true;
    }
    return false;
  }

  void pairAttemptStarted() {
    if (mode_ != TaskCoordinationMode::FINISH &&
        mode_ != TaskCoordinationMode::GLOBAL_RECOVERY) {
      mode_ = TaskCoordinationMode::PAIR_SEARCH;
      ++pair_attempt_count_;
    }
  }

  void pairSucceeded() { localTaskAvailable(); }

  void globalRecovered() {
    localTaskAvailable();
    cooldown_remaining_ = recovery_cooldown_cycles_;
  }

  void globalWait() {
    mode_ = TaskCoordinationMode::PAIR_SEARCH;
    consecutive_pair_failures_ = 0;
    pair_attempt_count_ = 0;
    cooldown_remaining_ = recovery_cooldown_cycles_;
  }

  void globalFinished() {
    mode_ = TaskCoordinationMode::FINISH;
    consecutive_pair_failures_ = 0;
    pair_attempt_count_ = 0;
  }

private:
  TaskCoordinationMode mode_{ TaskCoordinationMode::NORMAL };
  int pair_failure_threshold_{ 4 };
  int recovery_cooldown_cycles_{ 5 };
  int consecutive_pair_failures_{ 0 };
  int pair_attempt_count_{ 0 };
  int cooldown_remaining_{ 0 };
};

struct RecoveryTransfer {
  int idle_uav_index{ -1 };
  int donor_uav_index{ -1 };  // -1 means the task was globally unassigned.
  int task_id{ -1 };
};

struct MinimalRecoveryPlan {
  bool valid{ true };
  std::vector<std::vector<int>> assignments;
  std::vector<RecoveryTransfer> transfers;
  std::size_t remaining_global_tasks{ 0 };
  int active_uavs{ 0 };
};

// Preserve every current ownership except the explicit one-task transfers
// needed to recover idle UAVs. Unassigned active tasks are consumed before a
// task is donated by the most loaded UAV. The score callback lets the runtime
// choose the task nearest to each idle UAV without coupling this policy to
// Eigen, ROS, HGrid, or the planner.
inline MinimalRecoveryPlan buildMinimalRecoveryPlan(
    const std::vector<std::vector<int>>& current_assignments,
    const std::vector<int>& active_tasks,
    const std::vector<int>& idle_uav_indices,
    int max_tasks_per_idle,
    const std::function<double(int, int)>& score) {
  MinimalRecoveryPlan plan;
  plan.assignments = current_assignments;
  plan.remaining_global_tasks = active_tasks.size();
  max_tasks_per_idle = std::max(1, max_tasks_per_idle);

  std::unordered_set<int> active_set(active_tasks.begin(), active_tasks.end());
  std::unordered_set<int> owned;
  std::vector<int> active_load(current_assignments.size(), 0);
  for (std::size_t uav = 0; uav < current_assignments.size(); ++uav) {
    for (const int task : current_assignments[uav]) {
      if (!owned.insert(task).second) {
        plan.valid = false;
        return plan;
      }
      if (active_set.find(task) != active_set.end()) ++active_load[uav];
    }
    if (active_load[uav] > 0) ++plan.active_uavs;
  }

  std::vector<int> unassigned;
  for (const int task : active_tasks)
    if (owned.find(task) == owned.end()) unassigned.push_back(task);

  const auto select_nearest = [&](int idle_index, const std::vector<int>& candidates) {
    auto best = candidates.begin();
    double best_score = std::numeric_limits<double>::infinity();
    for (auto iterator = candidates.begin(); iterator != candidates.end(); ++iterator) {
      const double candidate_score = score ? score(idle_index, *iterator) : 0.0;
      if (candidate_score < best_score ||
          (candidate_score == best_score && *iterator < *best)) {
        best = iterator;
        best_score = candidate_score;
      }
    }
    return best;
  };

  for (const int idle_index : idle_uav_indices) {
    if (idle_index < 0 || idle_index >= static_cast<int>(plan.assignments.size())) {
      plan.valid = false;
      return plan;
    }
    if (!plan.assignments[idle_index].empty()) continue;

    for (int count = 0; count < max_tasks_per_idle; ++count) {
      if (!unassigned.empty()) {
        const auto best = select_nearest(idle_index, unassigned);
        const int task = *best;
        unassigned.erase(best);
        plan.assignments[idle_index].push_back(task);
        plan.transfers.push_back({ idle_index, -1, task });
        ++active_load[idle_index];
        continue;
      }

      int donor_index = -1;
      int donor_load = 1;
      for (std::size_t candidate = 0; candidate < plan.assignments.size(); ++candidate) {
        if (static_cast<int>(candidate) == idle_index) continue;
        if (active_load[candidate] > donor_load) {
          donor_load = active_load[candidate];
          donor_index = static_cast<int>(candidate);
        }
      }
      if (donor_index < 0) break;

      std::vector<int> donor_candidates;
      for (const int task : plan.assignments[donor_index])
        if (active_set.find(task) != active_set.end()) donor_candidates.push_back(task);
      if (donor_candidates.size() <= 1) break;
      const int task = *select_nearest(idle_index, donor_candidates);
      auto& donor_tasks = plan.assignments[donor_index];
      donor_tasks.erase(std::find(donor_tasks.begin(), donor_tasks.end(), task));
      plan.assignments[idle_index].push_back(task);
      plan.transfers.push_back({ idle_index, donor_index, task });
      --active_load[donor_index];
      ++active_load[idle_index];
    }
  }
  return plan;
}

inline bool recoveryEpochCanApply(
    uint64_t recovery_epoch,
    uint64_t last_applied_global_epoch,
    const std::vector<uint64_t>& current_assignment_epochs,
    const std::vector<uint64_t>& base_assignment_epochs,
    const std::vector<int>& affected_uav_indices) {
  if (recovery_epoch == 0 || recovery_epoch <= last_applied_global_epoch ||
      current_assignment_epochs.size() != base_assignment_epochs.size())
    return false;
  for (const int index : affected_uav_indices) {
    if (index < 0 || index >= static_cast<int>(current_assignment_epochs.size()) ||
        recovery_epoch <= current_assignment_epochs[index] ||
        current_assignment_epochs[index] != base_assignment_epochs[index])
      return false;
  }
  return true;
}

}  // namespace fast_planner

#endif
