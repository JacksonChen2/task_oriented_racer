#include <hybrid_communication_racer/assignment.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace hybrid_communication_racer {
namespace {

double overlapPenalty(
    const Point3& candidate, const std::vector<Point3>& occupied, double radius) {
  if (!(radius > 0.0)) return 0.0;
  double penalty = 0.0;
  for (const auto& target : occupied) {
    const double separation = distance(candidate, target);
    if (separation < radius) penalty += 1.0 - separation / radius;
  }
  return penalty;
}

}  // namespace

double distance(const Point3& first, const Point3& second) {
  const double dx = first.x - second.x;
  const double dy = first.y - second.y;
  const double dz = first.z - second.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double adaptiveDistanceLimit(
    double coverage, double minimum, double maximum, double gamma) {
  coverage = std::clamp(coverage, 0.0, 1.0);
  if (maximum < minimum) std::swap(minimum, maximum);
  if (!(gamma > 0.0) || !std::isfinite(gamma)) gamma = 1.0;
  return minimum + (maximum - minimum) * std::pow(coverage, gamma);
}

std::vector<GridTask> clusterTasks(
    const std::vector<GridTask>& tasks, double cluster_distance) {
  if (!(cluster_distance > 0.0)) return tasks;

  std::vector<std::size_t> order(tasks.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    return tasks[left].id < tasks[right].id;
  });
  std::vector<bool> consumed(tasks.size(), false);
  std::vector<GridTask> clusters;

  for (const std::size_t seed_index : order) {
    if (consumed[seed_index]) continue;
    consumed[seed_index] = true;
    const auto& seed = tasks[seed_index];
    GridTask cluster = seed;
    cluster.member_ids.clear();
    double center_weight = std::max(1.0, seed.gain);
    cluster.center.x *= center_weight;
    cluster.center.y *= center_weight;
    cluster.center.z *= center_weight;
    const std::vector<int> seed_members =
        seed.member_ids.empty() ? std::vector<int>{seed.id} : seed.member_ids;
    cluster.member_ids.insert(
        cluster.member_ids.end(), seed_members.begin(), seed_members.end());

    for (const std::size_t candidate_index : order) {
      if (consumed[candidate_index] ||
          distance(seed.center, tasks[candidate_index].center) > cluster_distance)
        continue;
      consumed[candidate_index] = true;
      const auto& candidate = tasks[candidate_index];
      const double weight = std::max(1.0, candidate.gain);
      cluster.center.x += candidate.center.x * weight;
      cluster.center.y += candidate.center.y * weight;
      cluster.center.z += candidate.center.z * weight;
      center_weight += weight;
      cluster.gain += candidate.gain;
      cluster.age = std::max(cluster.age, candidate.age);
      const std::vector<int> candidate_members = candidate.member_ids.empty()
          ? std::vector<int>{candidate.id} : candidate.member_ids;
      cluster.member_ids.insert(cluster.member_ids.end(),
          candidate_members.begin(), candidate_members.end());
    }

    cluster.center.x /= center_weight;
    cluster.center.y /= center_weight;
    cluster.center.z /= center_weight;
    std::sort(cluster.member_ids.begin(), cluster.member_ids.end());
    cluster.member_ids.erase(
        std::unique(cluster.member_ids.begin(), cluster.member_ids.end()),
        cluster.member_ids.end());
    cluster.id = cluster.member_ids.front();
    clusters.push_back(std::move(cluster));
  }
  return clusters;
}

std::vector<Match> greedyMaximumWeightMatching(
    const std::vector<Agent>& agents, const std::vector<GridTask>& tasks,
    const std::vector<CandidateMetric>& metrics,
    const std::vector<Point3>& locked_targets, const MatcherConfig& config) {
  if (metrics.size() != agents.size() * tasks.size())
    throw std::invalid_argument("candidate metric matrix has the wrong size");

  std::vector<bool> used_agents(agents.size(), false);
  std::vector<bool> used_tasks(tasks.size(), false);
  std::vector<Point3> occupied_targets = locked_targets;
  std::vector<Match> result;

  while (result.size() < std::min(agents.size(), tasks.size())) {
    Match best;
    bool found = false;
    for (std::size_t agent = 0; agent < agents.size(); ++agent) {
      if (used_agents[agent]) continue;
      for (std::size_t task = 0; task < tasks.size(); ++task) {
        if (used_tasks[task]) continue;
        const auto& metric = metrics[agent * tasks.size() + task];
        if (!std::isfinite(metric.travel_time) || !std::isfinite(metric.distance) ||
            metric.travel_time < 0.0 || metric.distance > config.maximum_distance)
          continue;
        const double overlap = overlapPenalty(
            tasks[task].center, occupied_targets, config.overlap_distance);
        const double utility = config.gain_scale * tasks[task].gain /
                (metric.travel_time + std::max(1e-9, config.epsilon)) -
            config.lambda_overlap * overlap + config.lambda_age * tasks[task].age;
        const bool better = !found || utility > best.utility + 1e-12 ||
            (std::abs(utility - best.utility) <= 1e-12 &&
                (agents[agent].id < agents[best.agent_index].id ||
                    (agents[agent].id == agents[best.agent_index].id &&
                        tasks[task].id < tasks[best.task_index].id)));
        if (!better) continue;
        found = true;
        best = Match{agent, task, utility, tasks[task].gain,
            metric.travel_time, overlap, metric.distance};
      }
    }
    if (!found) break;
    used_agents[best.agent_index] = true;
    used_tasks[best.task_index] = true;
    occupied_targets.push_back(tasks[best.task_index].center);
    result.push_back(best);
  }
  return result;
}

}  // namespace hybrid_communication_racer
