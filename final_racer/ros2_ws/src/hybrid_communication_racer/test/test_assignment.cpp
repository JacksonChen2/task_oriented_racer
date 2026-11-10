#include <gtest/gtest.h>

#include <hybrid_communication_racer/assignment.hpp>

#include <cmath>
#include <set>

namespace hcr = hybrid_communication_racer;

TEST(HybridAssignment, AdaptiveDistanceUsesCoverageFormula) {
  EXPECT_DOUBLE_EQ(hcr::adaptiveDistanceLimit(0.0, 8.0, 48.0, 0.5), 8.0);
  EXPECT_NEAR(hcr::adaptiveDistanceLimit(0.25, 8.0, 48.0, 0.5), 28.0, 1e-9);
  EXPECT_DOUBLE_EQ(hcr::adaptiveDistanceLimit(1.0, 8.0, 48.0, 0.5), 48.0);
}

TEST(HybridAssignment, ClustersNearbyTasksWithoutTransitiveChaining) {
  std::vector<hcr::GridTask> tasks{
      {1, {0, 0, 0}, 10, 1, {1}},
      {2, {2, 0, 0}, 20, 2, {2}},
      {3, {4, 0, 0}, 30, 3, {3}},
  };
  const auto clusters = hcr::clusterTasks(tasks, 2.1);
  ASSERT_EQ(clusters.size(), 2u);
  EXPECT_EQ(clusters[0].member_ids, (std::vector<int>{1, 2}));
  EXPECT_EQ(clusters[1].member_ids, (std::vector<int>{3}));
  EXPECT_DOUBLE_EQ(clusters[0].gain, 30.0);
}

TEST(HybridAssignment, JointMatchingHasUniqueAgentsAndTasks) {
  std::vector<hcr::Agent> agents{{1, {0, 0, 0}}, {2, {10, 0, 0}}};
  std::vector<hcr::GridTask> tasks{
      {10, {1, 0, 0}, 1000, 0, {10}},
      {20, {9, 0, 0}, 1000, 0, {20}},
  };
  std::vector<hcr::CandidateMetric> metrics{{1, 1}, {9, 9}, {9, 9}, {1, 1}};
  hcr::MatcherConfig config;
  config.maximum_distance = 20;
  const auto matches = hcr::greedyMaximumWeightMatching(agents, tasks, metrics, {}, config);
  ASSERT_EQ(matches.size(), 2u);
  std::set<std::size_t> agent_ids, task_ids;
  for (const auto& match : matches) {
    agent_ids.insert(match.agent_index);
    task_ids.insert(match.task_index);
  }
  EXPECT_EQ(agent_ids.size(), 2u);
  EXPECT_EQ(task_ids.size(), 2u);
}

TEST(HybridAssignment, SpatialPenaltySpreadsTargets) {
  std::vector<hcr::Agent> agents{{1, {0, 0, 0}}, {2, {0, 0, 0}}};
  std::vector<hcr::GridTask> tasks{
      {10, {5, 0, 0}, 1000, 0, {10}},
      {11, {5.2, 0, 0}, 990, 0, {11}},
      {20, {-5, 0, 0}, 800, 0, {20}},
  };
  std::vector<hcr::CandidateMetric> metrics(agents.size() * tasks.size(), {5, 5});
  hcr::MatcherConfig config;
  config.gain_scale = 0.01;
  config.lambda_overlap = 10.0;
  config.overlap_distance = 3.0;
  config.maximum_distance = 20.0;
  const auto matches = hcr::greedyMaximumWeightMatching(agents, tasks, metrics, {}, config);
  ASSERT_EQ(matches.size(), 2u);
  std::set<int> selected;
  for (const auto& match : matches) selected.insert(tasks[match.task_index].id);
  EXPECT_TRUE(selected.count(10));
  EXPECT_TRUE(selected.count(20));
  EXPECT_FALSE(selected.count(11));
}

TEST(HybridAssignment, AdaptiveLimitForbidsFarTask) {
  std::vector<hcr::Agent> agents{{1, {0, 0, 0}}};
  std::vector<hcr::GridTask> tasks{
      {10, {20, 0, 0}, 10000, 0, {10}},
      {20, {5, 0, 0}, 100, 0, {20}},
  };
  std::vector<hcr::CandidateMetric> metrics{{10, 20}, {5, 5}};
  hcr::MatcherConfig config;
  config.maximum_distance = 8.0;
  const auto matches = hcr::greedyMaximumWeightMatching(agents, tasks, metrics, {}, config);
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(tasks[matches[0].task_index].id, 20);
}
