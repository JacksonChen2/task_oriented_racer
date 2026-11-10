#include <racer_sionna_comm/event_driven_one_shot.hpp>

#include <gtest/gtest.h>

namespace {

using racer_sionna_comm::EventDrivenOneShotGate;
using racer_sionna_comm::OneShotActionTicket;
using racer_sionna_comm::OneShotEventKind;

TEST(EventDrivenOneShotGateTest, ConsumesEachActionExactlyOnce) {
  EventDrivenOneShotGate gate(0.02);
  const auto initial = gate.initialize(5U, 0.10);
  EXPECT_EQ(initial.kind, OneShotEventKind::kInitialState);
  EXPECT_EQ(gate.completedTransitions(), 0U);

  const OneShotActionTicket action{1U, 0U, 8U, 5U, 0.10};
  ASSERT_TRUE(gate.offer(action));
  EXPECT_FALSE(gate.offer(action));

  const auto applied = gate.advance(7U, 0.14);
  EXPECT_EQ(applied.kind, OneShotEventKind::kApplyAction);
  EXPECT_EQ(applied.action_version, 1U);
  EXPECT_DOUBLE_EQ(applied.action_age_s, 0.04);
  EXPECT_TRUE(gate.inFlight());

  EXPECT_EQ(gate.advance(7U, 0.14).kind, OneShotEventKind::kNone);
  EXPECT_EQ(gate.advance(8U, 0.16).kind, OneShotEventKind::kNone);
  const auto completed = gate.advance(9U, 0.18, true);
  EXPECT_EQ(completed.kind, OneShotEventKind::kTransitionComplete);
  EXPECT_EQ(completed.action_version, 1U);
  EXPECT_EQ(completed.transition_step_id, 0U);
  EXPECT_DOUBLE_EQ(completed.delta_t_s, 0.08);
  EXPECT_EQ(gate.completedTransitions(), 1U);
  EXPECT_FALSE(gate.inFlight());

  // The already consumed version cannot be replayed on a newer state.
  EXPECT_FALSE(gate.offer(action));
}

TEST(EventDrivenOneShotGateTest, RejectsActionsFromAnyOtherStateVersion) {
  EventDrivenOneShotGate gate(0.02);
  gate.initialize(0U, 0.0);
  EXPECT_FALSE(gate.offer({1U, 1U, 1U, 0U, 0.0}));
  EXPECT_TRUE(gate.offer({2U, 0U, 1U, 0U, 0.0}));
  EXPECT_EQ(gate.advance(1U, 0.02).kind,
            OneShotEventKind::kApplyAction);
  EXPECT_FALSE(gate.offer({3U, 0U, 1U, 1U, 0.02}));
  EXPECT_EQ(gate.advance(2U, 0.04, true).kind,
            OneShotEventKind::kTransitionComplete);
  EXPECT_FALSE(gate.offer({4U, 0U, 1U, 2U, 0.04}));
  EXPECT_TRUE(gate.offer({5U, 1U, 2U, 2U, 0.04}));
}

TEST(EventDrivenOneShotGateTest, AllowsIdleSimulationTimeBetweenActions) {
  EventDrivenOneShotGate gate(0.02);
  gate.initialize(0U, 0.0);
  EXPECT_EQ(gate.advance(20U, 0.40).kind, OneShotEventKind::kNone);
  ASSERT_TRUE(gate.offer({7U, 0U, 1U, 0U, 0.0}));
  const auto applied = gate.advance(25U, 0.50);
  EXPECT_EQ(applied.kind, OneShotEventKind::kApplyAction);
  EXPECT_DOUBLE_EQ(applied.action_age_s, 0.50);
  const auto completed = gate.advance(26U, 0.52, true);
  EXPECT_EQ(completed.kind, OneShotEventKind::kTransitionComplete);
  EXPECT_DOUBLE_EQ(completed.delta_t_s, 0.52);
}

TEST(EventDrivenOneShotGateTest, RejectsNonMonotonicClock) {
  EventDrivenOneShotGate gate(0.02);
  gate.initialize(4U, 0.08);
  EXPECT_THROW(gate.advance(3U, 0.06), std::logic_error);
}

}  // namespace
