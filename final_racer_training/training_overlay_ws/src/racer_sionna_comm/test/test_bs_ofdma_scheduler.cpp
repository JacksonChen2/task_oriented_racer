#include <racer_sionna_comm/bs_ofdma_scheduler.hpp>
#include <racer_sionna_comm/link_model.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <set>

using racer_sionna_comm::BsOfdmaScheduler;

TEST(BsOfdmaScheduler, ConcurrentPeersHaveDisjointPrbsWithinOneDirection) {
  BsOfdmaScheduler scheduler(33);
  const auto slot = scheduler.allocate({2, 0, 1}, {0, 1});
  ASSERT_TRUE(slot.uplink);
  ASSERT_EQ(slot.allocations.size(), 3U);
  std::set<int> occupied;
  for (const auto &grant : slot.allocations) {
    EXPECT_EQ(grant.prb_count, 11);
    for (int prb = grant.first_prb; prb < grant.first_prb + grant.prb_count; ++prb) {
      EXPECT_TRUE(occupied.insert(prb).second);
    }
  }
  EXPECT_EQ(occupied.size(), 33U);
  const auto next = scheduler.allocate({0, 1, 2}, {0, 1});
  EXPECT_FALSE(next.uplink);
  EXPECT_EQ(next.allocations.size(), 2U);
}

TEST(BsOfdmaScheduler, IdleDirectionDoesNotWasteSlots) {
  BsOfdmaScheduler scheduler(33);
  EXPECT_TRUE(scheduler.allocate({}, {}).allocations.empty());
  for (int i = 0; i < 3; ++i) {
    const auto slot = scheduler.allocate({}, {7});
    ASSERT_FALSE(slot.uplink);
    ASSERT_EQ(slot.allocations.size(), 1U);
    EXPECT_EQ(slot.allocations.front().prb_count, 33);
  }
  EXPECT_TRUE(scheduler.allocate({2}, {7}).uplink);
}

TEST(BsOfdmaScheduler, MorePeersThanPrbsReceiveRotatingService) {
  BsOfdmaScheduler scheduler(2);
  std::set<int> served;
  for (int i = 0; i < 4; ++i) {
    const auto slot = scheduler.allocate({0, 1, 2, 3}, {});
    ASSERT_EQ(slot.allocations.size(), 2U);
    for (const auto &grant : slot.allocations) {
      EXPECT_EQ(grant.prb_count, 1);
      served.insert(grant.sender);
    }
  }
  EXPECT_EQ(served.size(), 4U);
}

TEST(BsOfdmaScheduler, DividingBandwidthDoesNotMultiplyAggregateCapacity) {
  racer_sionna_comm::LinkModelConfig config;
  config.bandwidth_hz = 50e6;
  config.resource_blocks = 33;
  racer_sionna_comm::LinkModel model(config);
  BsOfdmaScheduler scheduler(33);
  double total_bits = 0;
  for (const auto &grant : scheduler.allocate({0, 1, 2}, {}).allocations) {
    total_bits += model.bitsPerSlot(21.0, grant.prb_count);
  }
  EXPECT_NEAR(total_bits, model.bitsPerSlot(21.0, 33), 1e-9);
  const double remaining = 8.0 * 100000;
  EXPECT_GT(remaining, model.bitsPerSlot(21.0, 11));
}
