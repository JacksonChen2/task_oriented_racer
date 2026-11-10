#include <racer_sionna_comm/shared_ofdma_scheduler.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <set>
#include <vector>

namespace {

int allocatedPrbs(
    const std::vector<racer_sionna_comm::PrbAllocation> &allocations) {
  return std::accumulate(
      allocations.begin(), allocations.end(), 0,
      [](int total, const auto &allocation) {
        return total + allocation.prb_count;
      });
}

}  // namespace

TEST(SharedOfdmaScheduler, ThreeActiveSendersReceiveEqualDisjointShares) {
  racer_sionna_comm::SharedOfdmaScheduler scheduler(66);
  const auto allocations = scheduler.allocate({2, 0, 1});
  ASSERT_EQ(allocations.size(), 3U);
  EXPECT_EQ(allocatedPrbs(allocations), 66);
  for (std::size_t index = 0; index < allocations.size(); ++index) {
    EXPECT_EQ(allocations[index].sender, static_cast<int>(index));
    EXPECT_EQ(allocations[index].first_prb, 22 * static_cast<int>(index));
    EXPECT_EQ(allocations[index].prb_count, 22);
  }
}

TEST(SharedOfdmaScheduler, TenSendersShareSixAndSevenPrbsFairly) {
  racer_sionna_comm::SharedOfdmaScheduler scheduler(66);
  std::vector<int> senders(10);
  std::iota(senders.begin(), senders.end(), 0);
  const auto first = scheduler.allocate(senders);
  const auto second = scheduler.allocate(senders);
  EXPECT_EQ(allocatedPrbs(first), 66);
  EXPECT_EQ(allocatedPrbs(second), 66);
  EXPECT_EQ(first.size(), 10U);
  EXPECT_EQ(second.size(), 10U);
  EXPECT_EQ(first.front().sender, 0);
  EXPECT_EQ(second.front().sender, 6);
  for (const auto &allocation : first) {
    EXPECT_TRUE(allocation.prb_count == 6 || allocation.prb_count == 7);
  }
}

TEST(SharedOfdmaScheduler, AtMostOneSenderPerPrb) {
  racer_sionna_comm::SharedOfdmaScheduler scheduler(66);
  std::vector<int> senders(70);
  std::iota(senders.begin(), senders.end(), 0);
  const auto allocations = scheduler.allocate(senders);
  ASSERT_EQ(allocations.size(), 66U);
  EXPECT_EQ(allocatedPrbs(allocations), 66);
  std::set<int> occupied;
  for (const auto &allocation : allocations) {
    for (int prb = allocation.first_prb;
         prb < allocation.first_prb + allocation.prb_count; ++prb) {
      EXPECT_TRUE(occupied.insert(prb).second);
    }
  }
}

TEST(SharedOfdmaScheduler, RejectsInvalidPoolAndHandlesNoTraffic) {
  EXPECT_THROW(racer_sionna_comm::SharedOfdmaScheduler(0),
               std::invalid_argument);
  racer_sionna_comm::SharedOfdmaScheduler scheduler(66);
  EXPECT_TRUE(scheduler.allocate({}).empty());
}
