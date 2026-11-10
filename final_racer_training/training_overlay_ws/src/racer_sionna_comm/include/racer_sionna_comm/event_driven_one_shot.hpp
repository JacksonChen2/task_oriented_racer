#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace racer_sionna_comm {

// A transport-agnostic lifecycle for the experimental BS scheduler.  The
// existing fixed-clock scheduler does not use this class.
enum class OneShotEventKind {
  kNone,
  kInitialState,
  kApplyAction,
  kTransitionComplete,
};

struct OneShotActionTicket {
  std::uint64_t version{};
  std::uint64_t source_step_id{};
  std::uint64_t source_communication_version{};
  std::uint64_t source_slot{};
  double source_sim_time_s{};
};

struct OneShotEvent {
  OneShotEventKind kind{OneShotEventKind::kNone};
  std::uint64_t action_version{};
  std::uint64_t transition_step_id{};
  std::uint64_t slot{};
  double sim_time_s{};
  double interval_start_sim_time_s{};
  double delta_t_s{};
  double action_age_s{};
};

class EventDrivenOneShotGate {
 public:
  explicit EventDrivenOneShotGate(double communication_slot_s)
      : communication_slot_s_(communication_slot_s) {
    if (!std::isfinite(communication_slot_s_) ||
        communication_slot_s_ <= 0.0) {
      throw std::invalid_argument(
          "communication_slot_s must be finite and positive");
    }
  }

  OneShotEvent initialize(std::uint64_t observed_slot, double sim_time_s) {
    validateTime(sim_time_s);
    if (initialized_) {
      throw std::logic_error("one-shot gate is already initialized");
    }
    initialized_ = true;
    last_observed_slot_ = observed_slot;
    last_observed_sim_time_s_ = sim_time_s;
    last_state_sim_time_s_ = sim_time_s;
    return {OneShotEventKind::kInitialState, 0U, 0U, observed_slot,
            sim_time_s, sim_time_s, 0.0, 0.0};
  }

  // Only an action produced from the most recently emitted state is valid.
  // This prevents an old state/action pair from being silently rebound to a
  // newer event-driven state. Duplicate versions are consumed only once.
  bool offer(const OneShotActionTicket &ticket) {
    if (!initialized_ || ticket.version == 0U ||
        ticket.version <= last_seen_version_) {
      return false;
    }
    // A newer mailbox value is consumed even when its provenance is invalid;
    // retrying the same stale action forever would violate one-shot semantics.
    last_seen_version_ = ticket.version;
    if (in_flight_ || pending_ ||
        ticket.source_step_id != completed_transitions_) {
      return false;
    }
    validateTime(ticket.source_sim_time_s);
    if (ticket.source_sim_time_s + tolerance_ < last_state_sim_time_s_) {
      return false;
    }
    pending_ticket_ = ticket;
    pending_ = true;
    return true;
  }

  OneShotEvent advance(std::uint64_t observed_slot, double sim_time_s,
                       bool action_settled = false) {
    if (!initialized_) {
      throw std::logic_error("one-shot gate must be initialized first");
    }
    validateTime(sim_time_s);
    if (observed_slot < last_observed_slot_ ||
        sim_time_s + tolerance_ < last_observed_sim_time_s_) {
      throw std::logic_error("event-driven communication clock moved backwards");
    }
    last_observed_slot_ = observed_slot;
    last_observed_sim_time_s_ = sim_time_s;

    // A result becomes observable only after the single physical slot in
    // which the action was scheduled has closed.
    if (in_flight_ && observed_slot > applied_slot_ && action_settled) {
      const double delta_t_s = sim_time_s - last_state_sim_time_s_;
      const auto version = in_flight_ticket_.version;
      ++completed_transitions_;
      in_flight_ = false;
      last_state_sim_time_s_ = sim_time_s;
      return {OneShotEventKind::kTransitionComplete, version,
              completed_transitions_ - 1U, observed_slot, sim_time_s,
              sim_time_s - delta_t_s, delta_t_s,
              applied_sim_time_s_ - in_flight_ticket_.source_sim_time_s};
    }

    // Never inject an action into the slot whose state produced it. Even when
    // inference finishes before /clock advances, the earliest legal apply
    // point is the following physical slot boundary.
    if (pending_ && !in_flight_ &&
        observed_slot > pending_ticket_.source_slot) {
      in_flight_ticket_ = pending_ticket_;
      pending_ = false;
      in_flight_ = true;
      applied_slot_ = observed_slot;
      applied_sim_time_s_ = sim_time_s;
      return {OneShotEventKind::kApplyAction,
              in_flight_ticket_.version, completed_transitions_,
              observed_slot, sim_time_s, last_state_sim_time_s_,
              sim_time_s - last_state_sim_time_s_,
              sim_time_s - in_flight_ticket_.source_sim_time_s};
    }
    return {};
  }

  bool initialized() const noexcept { return initialized_; }
  bool pending() const noexcept { return pending_; }
  bool inFlight() const noexcept { return in_flight_; }
  std::uint64_t completedTransitions() const noexcept {
    return completed_transitions_;
  }
  std::uint64_t lastSeenVersion() const noexcept {
    return last_seen_version_;
  }
  std::uint64_t inFlightActionVersion() const noexcept {
    return in_flight_ ? in_flight_ticket_.version : 0U;
  }
  double communicationSlotSeconds() const noexcept {
    return communication_slot_s_;
  }

 private:
  static constexpr double tolerance_ = 1.0e-9;

  static void validateTime(double value) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument(
          "event-driven simulation time must be finite and non-negative");
    }
  }

  double communication_slot_s_{};
  bool initialized_{false};
  bool pending_{false};
  bool in_flight_{false};
  std::uint64_t completed_transitions_{};
  std::uint64_t last_seen_version_{};
  std::uint64_t last_observed_slot_{};
  std::uint64_t applied_slot_{};
  double last_observed_sim_time_s_{};
  double last_state_sim_time_s_{};
  double applied_sim_time_s_{};
  OneShotActionTicket pending_ticket_{};
  OneShotActionTicket in_flight_ticket_{};
};

}  // namespace racer_sionna_comm
