#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace hybrid_communication_racer {

struct Point3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct GridTask {
  int id{-1};
  Point3 center;
  double gain{0.0};
  double age{0.0};
  std::vector<int> member_ids;
};

struct Agent {
  int id{-1};
  Point3 position;
};

struct CandidateMetric {
  // Row-major [agent][task]. Non-finite values mark forbidden pairs.
  double travel_time{std::numeric_limits<double>::infinity()};
  double distance{std::numeric_limits<double>::infinity()};
};

struct MatcherConfig {
  double gain_scale{0.001};
  double epsilon{0.1};
  double lambda_overlap{2.0};
  double lambda_age{0.05};
  double overlap_distance{6.0};
  double maximum_distance{20.0};
};

struct Match {
  std::size_t agent_index{0};
  std::size_t task_index{0};
  double utility{-std::numeric_limits<double>::infinity()};
  double gain{0.0};
  double travel_time{0.0};
  double overlap_penalty{0.0};
  double distance{0.0};
};

double distance(const Point3& first, const Point3& second);

double adaptiveDistanceLimit(
    double coverage, double minimum, double maximum, double gamma);

// HGrid already provides spatial task cells. This light second-stage clustering
// groups only centers inside one configured radius of the seed and therefore
// avoids transitive chaining across a long warehouse aisle.
std::vector<GridTask> clusterTasks(
    const std::vector<GridTask>& tasks, double cluster_distance);

// Deterministic maximum-utility greedy matching. Every agent and task appears
// at most once. Spatial penalty is recomputed after each accepted pair so later
// matches spread away from locked and newly selected targets.
std::vector<Match> greedyMaximumWeightMatching(
    const std::vector<Agent>& agents, const std::vector<GridTask>& tasks,
    const std::vector<CandidateMetric>& metrics,
    const std::vector<Point3>& locked_targets, const MatcherConfig& config);

}  // namespace hybrid_communication_racer
