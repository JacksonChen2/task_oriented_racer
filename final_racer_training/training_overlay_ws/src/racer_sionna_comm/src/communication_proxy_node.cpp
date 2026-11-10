#include <racer_sionna_comm/link_model.hpp>
#include <racer_sionna_comm/bs_ofdma_scheduler.hpp>
#include <racer_sionna_comm/event_driven_one_shot.hpp>
#include <racer_sionna_comm/shared_json_block.hpp>
#include <racer_sionna_comm/shared_ofdma_scheduler.hpp>

#include <racer_sionna_interfaces/msg/comm_statistics.hpp>
#include <racer_sionna_interfaces/msg/link_quality.hpp>
#include <racer_sionna_interfaces/msg/link_quality_array.hpp>

#include <racer_fidelity_msgs/msg/chunk_data.hpp>
#include <racer_fidelity_msgs/msg/chunk_stamps.hpp>
#include <racer_fidelity_msgs/msg/bspline.hpp>
#include <racer_fidelity_msgs/msg/drone_state.hpp>
#include <racer_fidelity_msgs/msg/global_grid_assignment.hpp>
#include <racer_fidelity_msgs/msg/pair_opt.hpp>
#include <racer_fidelity_msgs/msg/pair_opt_response.hpp>
#include <racer_recovery_core/msg/recovery_command.hpp>
#include <racer_recovery_core/msg/recovery_status.hpp>

#include <rclcpp/generic_publisher.hpp>
#include <rclcpp/generic_subscription.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp/serialization.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racer_sionna_comm {
namespace {

using LinkQuality = racer_sionna_interfaces::msg::LinkQuality;
using LinkQualityArray = racer_sionna_interfaces::msg::LinkQualityArray;

struct TopicPolicy {
  std::string key;
  std::string type;
  int priority{};
  double ttl_s{};
  bool reliable{};
};

const std::vector<TopicPolicy> kTopicPolicies{
    {"drone_state", "racer_fidelity_msgs/msg/DroneState", 6, 0.30, false},
    {"pair_opt", "racer_fidelity_msgs/msg/PairOpt", 7, 2.0, true},
    {"pair_opt_res", "racer_fidelity_msgs/msg/PairOptResponse", 7, 2.0, true},
    {"trajectory", "racer_fidelity_msgs/msg/Bspline", 8, 1.0, true},
    {"recovery_status", "racer_recovery_core/msg/RecoveryStatus", 9, 2.0, true},
    {"recovery_command", "racer_recovery_core/msg/RecoveryCommand", 10, 5.0, true},
    {"chunk_stamps", "racer_fidelity_msgs/msg/ChunkStamps", 4, 3.0, true},
    {"chunk_data", "racer_fidelity_msgs/msg/ChunkData", 2, 10.0, true},
    {"global_assignment", "racer_fidelity_msgs/msg/GlobalGridAssignment", 10, 2.0, true},
    {"hybrid_drone_state", "hybrid_communication_racer/msg/HybridDroneState", 9, 0.75, true},
    {"hybrid_global_assignment", "hybrid_communication_racer/msg/HybridGlobalAssignment", 10, 2.0, true},
};

// Admit map chunks against only part of each logical BS slot.  The remaining
// airtime is available to control/peer traffic and to variance around the
// retry expectation used by the adaptive budget below.
constexpr double kBsMapAdmissionFraction = 0.80;
constexpr std::size_t kRepresentativeMapChunkBytes = 1068U;

struct LinkKey {
  int sender{};
  int receiver{};

  bool operator==(const LinkKey &other) const noexcept {
    return sender == other.sender && receiver == other.receiver;
  }
};

struct LinkKeyHash {
  std::size_t operator()(const LinkKey &key) const noexcept {
    return (static_cast<std::size_t>(static_cast<std::uint32_t>(key.sender))
            << 32U) ^
           static_cast<std::size_t>(static_cast<std::uint32_t>(key.receiver));
  }
};

enum class RouteStage {
  kDirect,
  kApUplink,
  kApDownlink,
  kBsControl,
  kBsUplink,
  kBsDownlink,
};

struct ChunkKey {
  int owner{};
  std::uint32_t index{};

  bool operator==(const ChunkKey &other) const noexcept {
    return owner == other.owner && index == other.index;
  }

  bool operator<(const ChunkKey &other) const noexcept {
    return owner < other.owner ||
           (owner == other.owner && index < other.index);
  }
};

struct ChunkKeyHash {
  std::size_t operator()(const ChunkKey &key) const noexcept {
    return (static_cast<std::size_t>(static_cast<std::uint32_t>(key.owner))
            << 32U) ^ static_cast<std::size_t>(key.index);
  }
};

double timeSeconds(const builtin_interfaces::msg::Time &value) {
  return static_cast<double>(value.sec) + 1.0e-9 * value.nanosec;
}

std::string txTopic(int sender, const TopicPolicy &policy) {
  return "/racer_sionna/tx/drone_" + std::to_string(sender) + "/" +
         policy.key;
}

std::string rxTopic(int receiver, const TopicPolicy &policy) {
  return "/racer_sionna/rx/drone_" + std::to_string(receiver) + "/" +
         policy.key;
}

std::string payloadCacheTopic(int sender) {
  return "/racer_sionna/cache/drone_" + std::to_string(sender) +
         "/chunk_data";
}

}  // namespace

class CommunicationProxy final : public rclcpp::Node {
 public:
  CommunicationProxy()
      : Node("racer_sionna_communication_proxy"),
        mode_(declare_parameter<std::string>("mode", "sionna_hybrid")),
        drone_count_(declare_parameter<int>("drone_count", 5)),
        topology_(declare_parameter<std::string>("network_topology",
                                                 "distributed")),
        nearest_neighbor_count_(
            declare_parameter<int>("nearest_neighbor_count", 0)),
        lossless_nearest_neighbor_count_(
            declare_parameter<int>("lossless_nearest_neighbor_count", 0)),
        lossless_communication_range_m_(declare_parameter<double>(
            "lossless_communication_range_m", 0.0)),
        lossless_control_only_(
            declare_parameter<bool>("lossless_control_only", false)),
        directed_message_unicast_(
            declare_parameter<bool>("directed_message_unicast", false)),
        uav_udp_directed_unicast_(declare_parameter<bool>(
            "uav_udp_directed_unicast", true)),
        chunk_data_pre_enqueue_dedup_(declare_parameter<bool>(
            "chunk_data_pre_enqueue_dedup", false)),
        chunk_data_max_pending_per_link_(declare_parameter<int>(
            "chunk_data_max_pending_per_link", 0)),
        communication_range_m_(
            declare_parameter<double>("communication_range_m", 4.0)),
        ideal_coalesce_window_s_(
            1.0e-3 * declare_parameter<double>(
                           "ideal_coalesce_window_ms", 20.0)),
        preserve_ideal_direct_with_bs_(declare_parameter<bool>(
            "preserve_ideal_direct_with_bs", false)),
        initial_assignment_perfect_delivery_(declare_parameter<bool>(
            "initial_assignment_perfect_delivery", false)),
        initial_assignment_perfect_via_bs_(declare_parameter<bool>(
            "initial_assignment_perfect_via_bs", false)),
        active_link_hold_s_(
            declare_parameter<double>("active_link_hold_s", 1.0)),
        active_link_publish_period_s_(declare_parameter<double>(
            "active_link_publish_period_s", 0.02)),
        carrier_frequency_hz_(
            declare_parameter<double>("carrier_frequency_hz", 28.0e9)),
        bs_tx_power_dbm_(declare_parameter<double>("ap_tx_power_dbm", 40.0)),
        uav_tx_power_dbm_(declare_parameter<double>("tx_power_dbm", 23.0)),
        bs_array_rows_(declare_parameter<int>("ap_array_rows", 8)),
        bs_array_cols_(declare_parameter<int>("ap_array_cols", 8)),
        uav_array_rows_(declare_parameter<int>("uav_array_rows", 4)),
        uav_array_cols_(declare_parameter<int>("uav_array_cols", 4)),
        base_latency_s_(
            1.0e-3 * declare_parameter<double>("base_latency_ms", 20.0)),
        jitter_s_(1.0e-3 * declare_parameter<double>("jitter_ms", 10.0)),
        queue_capacity_bytes_(static_cast<std::size_t>(
            declare_parameter<int>("queue_capacity_bytes", 262144))),
        udp_header_bytes_(
            declare_parameter<int>("uav_udp_header_bytes", 8)),
        uav_channel_access_mode_(declare_parameter<std::string>(
            "uav_channel_access_mode", "csma")),
        bs_channel_access_mode_(declare_parameter<std::string>(
            "bs_channel_access_mode", "scheduled")),
        bs_downlink_source_owned_chunks_only_(declare_parameter<bool>(
            "bs_downlink_source_owned_chunks_only", false)),
        csma_cw_min_(declare_parameter<int>("uav_csma_cw_min", 15)),
        csma_cw_max_(declare_parameter<int>("uav_csma_cw_max", 1023)),
        csma_difs_slots_(declare_parameter<int>("uav_csma_difs_slots", 2)),
        ofdma_diagnostic_logging_(declare_parameter<bool>(
            "uav_ofdma_diagnostic_logging", true)),
        ofdma_log_packet_stride_(
            declare_parameter<int>("uav_ofdma_log_packet_stride", 1000)),
        max_retries_(declare_parameter<int>("max_retries", 3)),
        bs_max_retries_(declare_parameter<int>("bs_max_retries", 3)),
        retry_backoff_s_(
            1.0e-3 * declare_parameter<double>("retry_backoff_ms", 8.0)),
        bs_min_turn_s_(1.0e-3 *
                       declare_parameter<double>("bs_min_turn_ms", 10.0)),
        bs_control_bytes_(static_cast<std::size_t>(
            declare_parameter<int>("bs_control_bytes", 32))),
        bs_max_downlink_chunks_per_turn_(declare_parameter<int>(
            "bs_max_downlink_chunks_per_turn", 0)),
        bs_max_uplink_chunks_per_turn_(declare_parameter<int>(
            "bs_max_uplink_chunks_per_turn", 0)),
        bs_max_uplink_chunks_per_rl_slot_(declare_parameter<int>(
            "bs_max_uplink_chunks_per_rl_slot", 0)),
        bs_max_downlink_chunks_per_rl_slot_(declare_parameter<int>(
            "bs_max_downlink_chunks_per_rl_slot", 0)),
        bs_downlink_newest_missing_chunk_first_(declare_parameter<bool>(
            "bs_downlink_newest_missing_chunk_first", false)),
        rl_bs_scheduler_enabled_(declare_parameter<bool>(
            "rl_bs_scheduler_enabled", false)),
        force_bs_perfect_delivery_(declare_parameter<bool>(
            "force_bs_perfect_delivery", false)),
        bs_periodic_upload_request_enabled_(declare_parameter<bool>(
            "bs_periodic_upload_request_enabled", false)),
        bs_periodic_upload_request_period_s_(1.0e-3 * declare_parameter<double>(
            "bs_periodic_upload_request_period_ms", 200.0)),
        bs_periodic_upload_chunks_per_request_(declare_parameter<int>(
            "bs_periodic_upload_chunks_per_request", 8)),
        bs_max_inflight_chunks_per_uav_(declare_parameter<int>(
            "bs_max_inflight_chunks_per_uav", 0)),
        bs_payload_precache_enabled_(declare_parameter<bool>(
            "bs_payload_precache_enabled", false)),
        bs_payload_precache_batch_size_(declare_parameter<int>(
            "bs_payload_precache_batch_size", 8)),
        bs_max_payload_precache_inflight_(declare_parameter<int>(
            "bs_max_payload_precache_inflight", 16)),
        bs_payload_precache_timeout_s_(1.0e-3 * declare_parameter<double>(
            "bs_payload_precache_timeout_ms", 5000.0)),
        bs_control_ttl_s_(declare_parameter<double>(
            "bs_control_ttl_s", 2.0)),
        bs_all_to_all_relay_enabled_(declare_parameter<bool>(
            "bs_all_to_all_relay_enabled", false)),
        pair_control_reservation_enabled_(declare_parameter<bool>(
            "pair_control_reservation_enabled", false)),
        pair_control_reservation_s_(declare_parameter<double>(
            "pair_control_reservation_s", 1.8)),
        rl_bs_synchronous_mode_(declare_parameter<bool>(
            "rl_bs_synchronous_mode", false)),
        rl_bs_event_driven_one_shot_(declare_parameter<bool>(
            "rl_bs_event_driven_one_shot", false)),
        rl_bs_action_path_(declare_parameter<std::string>(
            "rl_bs_action_path", "/tmp/racer_agentic_crpo/action.txt")),
        rl_bs_state_path_(declare_parameter<std::string>(
            "rl_bs_state_path",
            "/tmp/racer_agentic_crpo/communication_state.json")),
        rl_shared_memory_root_(declare_parameter<std::string>(
            "rl_shared_memory_root", "")),
        rl_bs_communication_slot_s_(1.0e-3 * declare_parameter<double>(
            "rl_bs_communication_slot_ms", 20.0)),
        rl_bs_decision_period_s_(1.0e-3 * declare_parameter<double>(
            "rl_bs_decision_period_ms", 20.0)),
        rl_bs_repeat_action_each_slot_(declare_parameter<bool>(
            "rl_bs_repeat_action_each_slot", true)),
        rl_llm_state_period_s_(1.0e-3 * declare_parameter<double>(
            "rl_llm_state_period_ms", 5000.0)),
        ground_truth_occupied_voxels_path_(declare_parameter<std::string>(
            "ground_truth_occupied_voxels_path", "")),
        observed_occupied_voxels_path_(declare_parameter<std::string>(
            "observed_occupied_voxels_path", "")),
        require_ground_truth_map_(declare_parameter<bool>(
            "require_ground_truth_map", false)),
        task_metric_observer_mode_(declare_parameter<std::string>(
            "task_metric_observer_mode", "inline")),
        bs_map_resolution_(declare_parameter<double>(
            "sdf_map.resolution", 0.1)),
        bs_map_size_{
            declare_parameter<double>("sdf_map.map_size_x", 22.0),
            declare_parameter<double>("sdf_map.map_size_y", 36.0),
            declare_parameter<double>("sdf_map.map_size_z", 9.0)},
        bs_map_ground_height_(declare_parameter<double>(
            "sdf_map.ground_height", 0.0)),
        bs_map_box_min_{
            declare_parameter<double>("sdf_map.box_min_x", -10.0),
            declare_parameter<double>("sdf_map.box_min_y", -11.9),
            declare_parameter<double>("sdf_map.box_min_z", 0.4)},
        bs_map_box_max_{
            declare_parameter<double>("sdf_map.box_max_x", 9.0),
            declare_parameter<double>("sdf_map.box_max_y", 17.6),
            declare_parameter<double>("sdf_map.box_max_z", 8.6)},
        model_(LinkModelConfig{
            declare_parameter<double>("bandwidth_hz", 100.0e6),
            declare_parameter<double>("subcarrier_spacing_hz", 120.0e3),
            static_cast<int>(declare_parameter<int>("resource_blocks", 66)),
            declare_parameter<double>("data_re_efficiency", 0.82),
            declare_parameter<double>("target_initial_tbler", 0.10),
            declare_parameter<double>("tbler_slope_db", 1.35),
            static_cast<std::size_t>(
                declare_parameter<int>("transport_block_bytes", 1200)),
            static_cast<int>(
                declare_parameter<int>("fixed_mcs_index", -1))}),
        uav_broadcast_model_(LinkModelConfig{
            declare_parameter<double>("uav_broadcast_bandwidth_hz", 50.0e6),
            model_.config().subcarrier_spacing_hz,
            static_cast<int>(declare_parameter<int>(
                "uav_broadcast_resource_blocks", 33)),
            model_.config().data_re_efficiency,
            model_.config().target_initial_tbler,
            model_.config().tbler_slope_db,
            model_.config().transport_block_bytes,
            static_cast<int>(declare_parameter<int>(
                "uav_broadcast_fixed_mcs_index", 14))}),
        bs_model_(LinkModelConfig{
            declare_parameter<double>("bs_bandwidth_hz", 50.0e6),
            model_.config().subcarrier_spacing_hz,
            static_cast<int>(
                declare_parameter<int>("bs_resource_blocks", 33)),
            model_.config().data_re_efficiency,
            model_.config().target_initial_tbler,
            model_.config().tbler_slope_db,
            model_.config().transport_block_bytes,
            -1}),
        bs_ofdma_scheduler_(bs_model_.config().resource_blocks),
        ofdma_scheduler_(model_.config().resource_blocks),
        uav_broadcast_ofdma_scheduler_(
            uav_broadcast_model_.config().resource_blocks) {
    random_seed_ = declare_parameter<int>("random_seed", 42);
    if (mode_ != "ideal" && mode_ != "sionna" &&
        mode_ != "sionna_hybrid") {
      throw std::runtime_error(
          "mode must be ideal, sionna, or sionna_hybrid");
    }
    if (uav_channel_access_mode_ != "ofdma" &&
        uav_channel_access_mode_ != "csma") {
      throw std::runtime_error(
          "uav_channel_access_mode must be ofdma or csma");
    }
    uav_csma_enabled_ = uav_channel_access_mode_ == "csma";
    if (bs_channel_access_mode_ != "scheduled" &&
        bs_channel_access_mode_ != "csma" && bs_channel_access_mode_ != "ofdma") {
      throw std::invalid_argument("bs_channel_access_mode must be scheduled, csma, or ofdma");
    }
    bs_ofdma_diagnostic_logging_ = declare_parameter<bool>(
        "bs_ofdma_diagnostic_logging", false);
    if (bs_downlink_source_owned_chunks_only_ && bs_downlink_newest_missing_chunk_first_) {
      throw std::invalid_argument("source-owned distance ranking excludes newest-first override");
    }
    if (task_metric_observer_mode_ != "inline" &&
        task_metric_observer_mode_ != "async" &&
        task_metric_observer_mode_ != "off") {
      throw std::runtime_error(
          "task_metric_observer_mode must be inline, async, or off");
    }
    configureBsCoverageGeometry();
    if (topology_ != "distributed" && topology_ != "nearest_neighbors" &&
        topology_ != "distance_radius" && topology_ != "ap_assisted" &&
        topology_ != "bs_round_robin") {
      throw std::runtime_error(
          "network_topology must be distributed, nearest_neighbors, "
          "distance_radius, ap_assisted, or bs_round_robin");
    }
    if (bs_max_uplink_chunks_per_rl_slot_ == 0 ||
        bs_max_downlink_chunks_per_rl_slot_ == 0) {
      // This is only a hard ceiling.  Each slot's actual admission budget is
      // calculated later from the action-bound Channel State, selected MCS,
      // NR serialization quantization and expected retransmissions.  Use the
      // best-MCS serialization time here so an automatic ceiling does not
      // suppress useful capacity when a narrower configured band has a
      // strong channel.
      const double per_direction_airtime_s =
          0.5 * kBsMapAdmissionFraction * rl_bs_communication_slot_s_;
      const double best_mcs_chunk_airtime_s = bs_model_.serializationDelay(
          40.0, kRepresentativeMapChunkBytes);
      const int unrounded_per_direction = static_cast<int>(std::floor(
          per_direction_airtime_s / best_mcs_chunk_airtime_s + 1.0e-9));
      const int rounded_per_direction = std::max(
          1, unrounded_per_direction >= 8
                 ? 8 * (unrounded_per_direction / 8)
                 : unrounded_per_direction);
      if (bs_max_uplink_chunks_per_rl_slot_ == 0) {
        bs_max_uplink_chunks_per_rl_slot_ = rounded_per_direction;
      }
      if (bs_max_downlink_chunks_per_rl_slot_ == 0) {
        bs_max_downlink_chunks_per_rl_slot_ = rounded_per_direction;
      }
    }
    if (drone_count_ < 1 || queue_capacity_bytes_ == 0U ||
        udp_header_bytes_ <= 0 || ofdma_log_packet_stride_ <= 0 ||
        csma_cw_min_ < 0 || csma_cw_max_ < csma_cw_min_ ||
        csma_difs_slots_ < 0 ||
        max_retries_ < 0 || bs_max_retries_ < 0 ||
        chunk_data_max_pending_per_link_ < 0 ||
        ideal_coalesce_window_s_ <= 0.0 || active_link_hold_s_ <= 0.0 ||
        active_link_publish_period_s_ <= 0.0 || base_latency_s_ < 0.0 ||
        jitter_s_ < 0.0 ||
        retry_backoff_s_ < 0.0 || bs_min_turn_s_ < 0.0 ||
        bs_control_bytes_ == 0U || bs_max_downlink_chunks_per_turn_ < 0 ||
        bs_max_uplink_chunks_per_turn_ < 0 ||
        bs_max_uplink_chunks_per_rl_slot_ < 1 ||
        bs_max_downlink_chunks_per_rl_slot_ < 1 ||
        carrier_frequency_hz_ <= 0.0 ||
        bs_periodic_upload_chunks_per_request_ < 1 ||
        bs_max_inflight_chunks_per_uav_ < 0 ||
        bs_payload_precache_batch_size_ < 1 ||
        bs_max_payload_precache_inflight_ < 1 ||
        bs_payload_precache_batch_size_ >
            bs_max_payload_precache_inflight_ ||
        bs_payload_precache_timeout_s_ <= 0.0 ||
        bs_control_ttl_s_ <= 0.0 ||
        rl_bs_communication_slot_s_ <= 0.0 ||
        rl_bs_decision_period_s_ <= 0.0 ||
        rl_llm_state_period_s_ <= 0.0 ||
        bs_array_rows_ < 1 || bs_array_cols_ < 1 || uav_array_rows_ < 1 ||
        uav_array_cols_ < 1) {
      throw std::runtime_error("invalid communication proxy parameters");
    }
    nearest_neighbors_enabled_ = topology_ == "nearest_neighbors";
    distance_radius_enabled_ = topology_ == "distance_radius";
    if (nearest_neighbors_enabled_ &&
        (nearest_neighbor_count_ < 1 ||
         nearest_neighbor_count_ >= drone_count_)) {
      throw std::runtime_error(
          "nearest_neighbor_count must be in [1, drone_count - 1]");
    }
    if (distance_radius_enabled_ && communication_range_m_ <= 0.0) {
      throw std::runtime_error("communication_range_m must be positive");
    }
    ap_enabled_ = topology_ == "ap_assisted" ||
                  topology_ == "bs_round_robin";
    if (initial_assignment_perfect_via_bs_ &&
        !initial_assignment_perfect_delivery_) {
      throw std::runtime_error(
          "initial_assignment_perfect_via_bs requires "
          "initial_assignment_perfect_delivery=true");
    }
    if (initial_assignment_perfect_delivery_) {
      if (mode_ == "ideal") {
        throw std::runtime_error(
            "initial_assignment_perfect_delivery requires a non-ideal mode");
      }
      if (initial_assignment_perfect_via_bs_) {
        if (!ap_enabled_) {
          throw std::runtime_error(
              "initial_assignment_perfect_via_bs requires a BS/AP-assisted "
              "topology");
        }
      } else if (topology_ != "distributed" || ap_enabled_) {
        throw std::runtime_error(
            "direct initial_assignment_perfect_delivery requires a "
            "distributed topology without a BS/AP");
      }
    }
    if (ap_enabled_) {
      const double assigned_bandwidth =
          uav_broadcast_model_.config().bandwidth_hz +
          bs_model_.config().bandwidth_hz;
      const int assigned_prbs =
          uav_broadcast_model_.config().resource_blocks +
          bs_model_.config().resource_blocks;
      if (std::abs(assigned_bandwidth - model_.config().bandwidth_hz) >
              1.0e-6 * model_.config().bandwidth_hz ||
          assigned_prbs != model_.config().resource_blocks ||
          uav_broadcast_model_.config().fixed_mcs_index < 0) {
        throw std::runtime_error(
            "BS-assisted PHY requires UAV and BS subbands to exactly "
            "partition bandwidth_hz/resource_blocks and UAV broadcast to "
            "use a fixed MCS");
      }
    }
    shared_uav_radio_enabled_ =
        mode_ != "ideal" && (topology_ == "distributed" || ap_enabled_);
    if (lossless_nearest_neighbor_count_ < 0 ||
        lossless_nearest_neighbor_count_ >= drone_count_) {
      throw std::runtime_error(
          "lossless_nearest_neighbor_count must be in [0, drone_count - 1]");
    }
    if ((lossless_nearest_neighbor_count_ > 0 ||
         lossless_communication_range_m_ > 0.0) &&
        (mode_ == "ideal" || topology_ != "distributed" || ap_enabled_)) {
      throw std::runtime_error(
          "lossless UAV-link overrides require a non-ideal "
          "distributed UAV-to-UAV topology");
    }
    if (!std::isfinite(lossless_communication_range_m_) ||
        lossless_communication_range_m_ < 0.0) {
      throw std::runtime_error(
          "lossless_communication_range_m must be finite and non-negative");
    }
    // AP-assisted modes normally model an actual two-hop radio path.  A
    // training-only compatibility mode keeps the original perfect direct
    // RACER exchange intact while the RL scheduler controls the auxiliary BS
    // upload/relay plane.  This prevents scheduler exploration from changing
    // the exploration algorithm's communication semantics.
    ideal_direct_enabled_ =
        mode_ == "ideal" && (!ap_enabled_ || preserve_ideal_direct_with_bs_);
    bs_round_robin_enabled_ = topology_ == "bs_round_robin";
    if (rl_bs_scheduler_enabled_ && !bs_round_robin_enabled_) {
      throw std::runtime_error(
          "rl_bs_scheduler_enabled requires network_topology=bs_round_robin");
    }
    if (force_bs_perfect_delivery_ && !ap_enabled_) {
      throw std::runtime_error(
          "force_bs_perfect_delivery requires an AP/BS-assisted topology");
    }
    if (bs_periodic_upload_request_enabled_ && !ap_enabled_) {
      throw std::runtime_error(
          "bs_periodic_upload_request_enabled requires an AP/BS-assisted "
          "topology");
    }
    if (bs_periodic_upload_request_period_s_ <= 0.0) {
      throw std::runtime_error(
          "bs_periodic_upload_request_period_ms must be positive");
    }
    if (bs_max_inflight_chunks_per_uav_ > 0 &&
        bs_periodic_upload_chunks_per_request_ >
            bs_max_inflight_chunks_per_uav_) {
      throw std::runtime_error(
          "bs_periodic_upload_chunks_per_request must not exceed "
          "bs_max_inflight_chunks_per_uav");
    }
    if (bs_all_to_all_relay_enabled_ &&
        (!bs_round_robin_enabled_ || !rl_bs_scheduler_enabled_)) {
      throw std::runtime_error(
          "bs_all_to_all_relay_enabled requires bs_round_robin and the RL "
          "BS scheduler");
    }
    if (!std::isfinite(pair_control_reservation_s_) ||
        pair_control_reservation_s_ <= 0.0) {
      throw std::runtime_error(
          "pair_control_reservation_s must be finite and positive");
    }
    if (pair_control_reservation_enabled_ &&
        (!bs_round_robin_enabled_ || !rl_bs_scheduler_enabled_)) {
      throw std::runtime_error(
          "pair_control_reservation_enabled requires the RL BS scheduler");
    }
    if (rl_bs_synchronous_mode_ && !rl_bs_scheduler_enabled_) {
      throw std::runtime_error(
          "rl_bs_synchronous_mode requires rl_bs_scheduler_enabled");
    }
    if (rl_bs_event_driven_one_shot_ &&
        (!rl_bs_scheduler_enabled_ || rl_bs_synchronous_mode_ ||
         rl_shared_memory_root_.empty() ||
         bs_periodic_upload_request_enabled_ ||
         pair_control_reservation_enabled_)) {
      throw std::runtime_error(
          "rl_bs_event_driven_one_shot requires the shared-memory RL BS "
          "scheduler, cannot use synchronous fixed-clock mode, and requires "
          "periodic/reserved BS traffic to be disabled");
    }
    if (!rl_shared_memory_root_.empty() && rl_bs_synchronous_mode_) {
      throw std::runtime_error(
          "event-driven shared-memory scheduling cannot use the legacy "
          "synchronous boundary gate");
    }
    if (!rl_bs_event_driven_one_shot_ &&
        (rl_bs_synchronous_mode_ || !rl_shared_memory_root_.empty())) {
      const double ratio =
          rl_bs_decision_period_s_ / rl_bs_communication_slot_s_;
      rl_slots_per_decision_ = static_cast<std::uint64_t>(std::llround(ratio));
      if (rl_slots_per_decision_ < 1U ||
          std::abs(
              rl_bs_decision_period_s_ -
              static_cast<double>(rl_slots_per_decision_) *
                  rl_bs_communication_slot_s_) > 1.0e-12) {
        throw std::runtime_error(
            "RL scheduling requires decision_period to be a positive integer "
            "multiple of communication_slot");
      }
    }
    if (rl_bs_event_driven_one_shot_) {
      // The experimental controller consumes a fresh action in exactly one
      // physical communication slot. The fixed-clock path above holds each
      // action for the configured integer number of communication slots.
      rl_slots_per_decision_ = 1U;
      event_driven_gate_ = std::make_unique<EventDrivenOneShotGate>(
          rl_bs_communication_slot_s_);
    }
    ap_node_id_ = drone_count_;
    radio_node_count_ = drone_count_ + static_cast<int>(ap_enabled_);
    uav_udp_queues_.resize(static_cast<std::size_t>(drone_count_));
    csma_backoff_slots_.assign(static_cast<std::size_t>(drone_count_), -1);
    csma_contention_windows_.assign(
        static_cast<std::size_t>(drone_count_), csma_cw_min_);
    uav_sender_rngs_.reserve(static_cast<std::size_t>(drone_count_));
    for (int sender = 0; sender < drone_count_; ++sender) {
      std::seed_seq seed{random_seed_, sender, 0x554450, 0x4f46444d};
      uav_sender_rngs_.emplace_back(seed);
    }

    // Three independent executor lanes: radio scheduling/action application,
    // packet/ROS I/O, and binary Fast RL State publication. Heavy task metric
    // work never enters the executor; it is owned by task_observer_thread_.
    scheduler_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    io_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    payload_cache_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    fast_state_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    telemetry_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions io_subscription_options;
    io_subscription_options.callback_group = io_callback_group_;
    rclcpp::SubscriptionOptions payload_cache_subscription_options;
    payload_cache_subscription_options.callback_group =
        payload_cache_callback_group_;

    const auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1000)).reliable();
    publishers_.resize(kTopicPolicies.size());
    for (std::size_t topic_index = 0; topic_index < kTopicPolicies.size();
         ++topic_index) {
      const auto &policy = kTopicPolicies[topic_index];
      publishers_[topic_index].reserve(
          static_cast<std::size_t>(drone_count_));
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        publishers_[topic_index].push_back(create_generic_publisher(
            rxTopic(receiver, policy), policy.type, data_qos));
      }
      for (int sender = 0; sender < drone_count_; ++sender) {
        subscriptions_.push_back(create_generic_subscription(
            txTopic(sender, policy), policy.type, data_qos,
            [this, sender, topic_index](
                std::shared_ptr<rclcpp::SerializedMessage> message) {
              std::lock_guard<std::mutex> lock(communication_mutex_);
              onTransmit(sender, topic_index, std::move(message));
            }, io_subscription_options));
      }
    }
    const auto payload_cache_qos =
        rclcpp::QoS(rclcpp::KeepLast(50000)).reliable();
    payload_cache_subscriptions_.reserve(
        static_cast<std::size_t>(drone_count_));
    for (int sender = 0; sender < drone_count_; ++sender) {
      payload_cache_subscriptions_.push_back(create_generic_subscription(
          payloadCacheTopic(sender), "racer_fidelity_msgs/msg/ChunkData",
          payload_cache_qos,
          [this, sender](
              std::shared_ptr<rclcpp::SerializedMessage> message) {
            std::lock_guard<std::mutex> lock(communication_mutex_);
            onPushedChunkPayload(sender, std::move(message));
          }, payload_cache_subscription_options));
    }

    // Pre-create every physical queue. AP forwarding can then enqueue from a
    // scheduler callback without rehashing and invalidating the active queue.
    const auto directed_link_count = static_cast<std::size_t>(
        radio_node_count_ * std::max(0, radio_node_count_ - 1));
    queues_.reserve(directed_link_count);
    link_rngs_.reserve(directed_link_count);
    for (int sender = 0; sender < radio_node_count_; ++sender) {
      for (int receiver = 0; receiver < radio_node_count_; ++receiver) {
        if (sender == receiver) continue;
        const LinkKey key{sender, receiver};
        // Distributed Sionna UAV traffic uses per-sender UDP queues below.
        // Keep the legacy per-link queues only for untouched non-distributed
        // and UAV-BS paths.
        const bool direct_uav_link =
            sender < drone_count_ && receiver < drone_count_;
        if (!(shared_uav_radio_enabled_ && direct_uav_link)) {
          queues_.emplace(key, LinkQueue{});
        }
        std::seed_seq seed{
            random_seed_, sender, receiver, 0x52414345, 0x53494f4e};
        link_rngs_.emplace(key, std::mt19937(seed));
      }
    }
    ap_latest_messages_.resize(static_cast<std::size_t>(drone_count_));
    for (auto &topics : ap_latest_messages_) {
      topics.resize(kTopicPolicies.size());
    }
    uav_latest_peer_messages_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<CachedPeerMessage>(kTopicPolicies.size()));
    bs_latest_peer_messages_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<CachedPeerMessage>(kTopicPolicies.size()));
    bs_peer_uplink_scheduled_by_topic_.assign(kTopicPolicies.size(), 0U);
    bs_peer_uplink_delivered_by_topic_.assign(kTopicPolicies.size(), 0U);
    bs_peer_downlink_scheduled_by_topic_.assign(kTopicPolicies.size(), 0U);
    bs_peer_downlink_delivered_by_topic_.assign(kTopicPolicies.size(), 0U);
    uav_chunks_.resize(static_cast<std::size_t>(drone_count_));
    ordered_bs_missing_chunks_.resize(
        static_cast<std::size_t>(drone_count_));
    ordered_uav_missing_chunks_.resize(
        static_cast<std::size_t>(drone_count_));
    downlink_priority_tables_.resize(static_cast<std::size_t>(drone_count_));
    downlink_priority_positions_.resize(static_cast<std::size_t>(drone_count_));
    downlink_priority_position_valid_.assign(
        static_cast<std::size_t>(drone_count_), false);
    downlink_priority_versions_.assign(static_cast<std::size_t>(drone_count_),
                                       std::numeric_limits<std::uint64_t>::max());
    pending_bs_chunk_requests_.resize(
        static_cast<std::size_t>(drone_count_));
    bs_periodic_upload_request_pending_.assign(
        static_cast<std::size_t>(drone_count_), false);
    bs_uplink_budget_action_id_.assign(
        static_cast<std::size_t>(drone_count_), 0U);
    bs_uplink_chunks_selected_for_action_.assign(
        static_cast<std::size_t>(drone_count_), 0U);
    cached_pair_missing_bytes_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<std::uint64_t>(static_cast<std::size_t>(drone_count_), 0U));
    cached_pair_missing_chunks_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<std::uint64_t>(static_cast<std::size_t>(drone_count_), 0U));
    cached_bs_missing_bytes_.assign(static_cast<std::size_t>(drone_count_), 0U);
    cached_bs_missing_chunks_.assign(static_cast<std::size_t>(drone_count_), 0U);
    cached_bs_uav_missing_bytes_.assign(
        static_cast<std::size_t>(drone_count_), 0U);
    cached_bs_uav_missing_chunks_.assign(
        static_cast<std::size_t>(drone_count_), 0U);
    cached_uplink_queue_bytes_.assign(
        static_cast<std::size_t>(drone_count_), 0U);
    cached_relay_queue_bytes_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<std::uint64_t>(static_cast<std::size_t>(drone_count_), 0U));
    uav_observed_voxels_.resize(static_cast<std::size_t>(drone_count_));
    uav_positions_.resize(static_cast<std::size_t>(drone_count_));
    uav_velocities_.resize(static_cast<std::size_t>(drone_count_));
    uav_yaws_.assign(static_cast<std::size_t>(drone_count_), 0.0);
    uav_position_valid_.assign(static_cast<std::size_t>(drone_count_), false);
    initial_assignment_epoch_acks_.assign(
        static_cast<std::size_t>(drone_count_), false);
    last_uav_info_received_s_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<double>(static_cast<std::size_t>(drone_count_), 0.0));
    last_bs_info_received_s_.assign(static_cast<std::size_t>(drone_count_),
                                    0.0);
    rl_relay_action_.assign(
        static_cast<std::size_t>(drone_count_),
        std::vector<bool>(static_cast<std::size_t>(drone_count_), false));
    rl_upload_action_.assign(static_cast<std::size_t>(drone_count_), false);
    rl_latest_relay_action_ = rl_relay_action_;
    rl_latest_upload_action_ = rl_upload_action_;
    trajectory_summaries_.resize(static_cast<std::size_t>(drone_count_));
    if (!rl_shared_memory_root_.empty()) {
      shared_physical_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/physical.shm");
      shared_physical_fast_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/physical_fast.shm");
      shared_communication_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/communication.shm");
      shared_transition_ring_ = std::make_unique<SharedJsonRing>(
          rl_shared_memory_root_ + "/transition_ring.shm");
      shared_task_metric_ring_ = std::make_unique<SharedJsonRing>(
          rl_shared_memory_root_ + "/task_metric_ring.shm");
      shared_guidance_fast_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/guidance_fast.shm");
      shared_state_event_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/state_event.shm");
      shared_action_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/action.shm");
      shared_action_ack_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/action_ack.shm");
      shared_status_ = std::make_unique<SharedJsonBlock>(
          rl_shared_memory_root_ + "/status_racer.shm");
    }
    loadGroundTruthOccupiedVoxels();
    odometry_subscriptions_.reserve(static_cast<std::size_t>(drone_count_));
    for (int drone = 0; drone < drone_count_; ++drone) {
      odometry_subscriptions_.push_back(
          create_subscription<nav_msgs::msg::Odometry>(
              "/drone_" + std::to_string(drone) + "/odom",
              rclcpp::SensorDataQoS(),
              [this, drone](nav_msgs::msg::Odometry::ConstSharedPtr message) {
                std::lock_guard<std::mutex> lock(communication_mutex_);
                const auto &position = message->pose.pose.position;
                uav_positions_[static_cast<std::size_t>(drone)] =
                    {position.x, position.y, position.z};
                const auto &velocity = message->twist.twist.linear;
                uav_velocities_[static_cast<std::size_t>(drone)] =
                    {velocity.x, velocity.y, velocity.z};
                const auto &orientation = message->pose.pose.orientation;
                uav_yaws_[static_cast<std::size_t>(drone)] = std::atan2(
                    2.0 * (orientation.w * orientation.z +
                           orientation.x * orientation.y),
                    1.0 - 2.0 * (orientation.y * orientation.y +
                                 orientation.z * orientation.z));
                uav_position_valid_[static_cast<std::size_t>(drone)] = true;
              }, io_subscription_options));
    }

    if (ap_enabled_) {
      const auto status_index = topicIndex("recovery_status");
      const auto command_index = topicIndex("recovery_command");
      ap_recovery_status_publisher_ = create_generic_publisher(
          "/racer_ap_recovery/status_uplink",
          kTopicPolicies.at(status_index).type, data_qos);
      ap_recovery_command_subscription_ = create_generic_subscription(
          "/racer_ap_recovery/command_downlink",
          kTopicPolicies.at(command_index).type, data_qos,
          [this, command_index](
              std::shared_ptr<rclcpp::SerializedMessage> message) {
            std::lock_guard<std::mutex> lock(communication_mutex_);
            onApRecoveryCommand(command_index, std::move(message));
          }, io_subscription_options);
    }

    link_subscription_ = create_subscription<LinkQualityArray>(
        "/racer_sionna/link_quality", rclcpp::QoS(20).reliable(),
        [this](LinkQualityArray::ConstSharedPtr message) {
          std::lock_guard<std::mutex> lock(communication_mutex_);
          onLinkQuality(*message);
        }, io_subscription_options);
    statistics_publisher_ =
        create_publisher<racer_sionna_interfaces::msg::CommStatistics>(
            "/racer_sionna/comm_statistics", rclcpp::QoS(10).reliable());
    active_link_publisher_ = create_publisher<LinkQualityArray>(
        "/racer_sionna/active_links", rclcpp::QoS(20).reliable());
    scheduler_timer_ = create_wall_timer(
        std::chrono::milliseconds(2),
        [this]() {
          std::lock_guard<std::mutex> lock(communication_mutex_);
          schedulerTick();
        },
        scheduler_callback_group_);
    fast_state_timer_ = create_wall_timer(
        std::chrono::milliseconds(1), [this]() { publishFastRlState(); },
        fast_state_callback_group_);
    active_link_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(active_link_publish_period_s_)),
        [this]() {
          std::lock_guard<std::mutex> lock(communication_mutex_);
          publishActiveLinks();
        },
        telemetry_callback_group_);
    statistics_timer_ = create_wall_timer(
        std::chrono::seconds(1),
        [this]() {
          std::lock_guard<std::mutex> lock(communication_mutex_);
          publishStatistics();
        },
        telemetry_callback_group_);
    if (!rl_shared_memory_root_.empty() &&
        task_metric_observer_mode_ != "async") {
      RCLCPP_WARN(
          get_logger(),
          "shared-memory training forces task_metric_observer_mode=async so "
          "map metrics cannot enter the scheduler callback");
      task_metric_observer_mode_ = "async";
    }
    if (task_metric_observer_mode_ == "async") {
      task_observer_thread_ = std::thread([this]() { taskObserverLoop(); });
    }

    if (shared_status_) {
      std::ostringstream status;
      status << "{\"ready\":true,\"process\":\"racer\",\"pid\":"
             << ::getpid() << '}';
      shared_status_->publish(status.str(), 0U, 0.0);
    }

    RCLCPP_INFO(
        get_logger(),
        "source-faithful communication proxy ready: mode=%s topology=%s "
        "drones=%d nearest_neighbors=%d lossless_nearest_neighbors=%d "
        "lossless_communication_range_m=%.3f communication_range_m=%.3f "
        "radio_nodes=%d bs_node_id=%d seed=%d phy=NR-LDPC "
        "waveform=CP-OFDM shared_uav_radio=%d uav_channel_access=%s "
        "uav_transport=UDP "
        "initial_assignment_perfect_delivery=%d "
        "initial_assignment_perfect_via_bs=%d "
        "rl_bs_scheduler=%d force_bs_perfect_delivery=%d "
        "bs_periodic_upload_request=%d bs_periodic_upload_period_ms=%.3f "
        "bs_payload_precache=%d precache_batch=%d precache_inflight=%d "
        "bs_adaptive_chunk_budget=1 bs_ul_slot_cap=%d bs_dl_slot_cap=%d "
        "bs_all_to_all_relay=%d "
        "pair_control_reservation=%d pair_control_reservation_s=%.3f "
        "rl_sync=%d rl_event_one_shot=%d "
        "Tcomm_ms=%.3f TRL_ms=%.3f "
        "total_bandwidth_mhz=%.1f "
        "uav_bandwidth_mhz=%.1f uav_mcs=%d bs_bandwidth_mhz=%.1f "
        "bs_mcs=adaptive task_metric_observer=%s",
        mode_.c_str(), topology_.c_str(), drone_count_, nearest_neighbor_count_,
        lossless_nearest_neighbor_count_, lossless_communication_range_m_,
        communication_range_m_, radio_node_count_,
        ap_enabled_ ? ap_node_id_ : -1, random_seed_,
        shared_uav_radio_enabled_ ? 1 : 0,
        uav_channel_access_mode_.c_str(),
        initial_assignment_perfect_delivery_ ? 1 : 0,
        initial_assignment_perfect_via_bs_ ? 1 : 0,
        rl_bs_scheduler_enabled_ ? 1 : 0,
        force_bs_perfect_delivery_ ? 1 : 0,
        bs_periodic_upload_request_enabled_ ? 1 : 0,
        1.0e3 * bs_periodic_upload_request_period_s_,
        bs_payload_precache_enabled_ ? 1 : 0,
        bs_payload_precache_batch_size_, bs_max_payload_precache_inflight_,
        bs_max_uplink_chunks_per_rl_slot_,
        bs_max_downlink_chunks_per_rl_slot_,
        bs_all_to_all_relay_enabled_ ? 1 : 0,
        pair_control_reservation_enabled_ ? 1 : 0,
        pair_control_reservation_s_,
        rl_bs_synchronous_mode_ ? 1 : 0,
        rl_bs_event_driven_one_shot_ ? 1 : 0,
        1.0e3 * rl_bs_communication_slot_s_,
        1.0e3 * rl_bs_decision_period_s_,
        1.0e-6 * model_.config().bandwidth_hz,
        1.0e-6 * directRadioModel().config().bandwidth_hz,
        directRadioModel().config().fixed_mcs_index,
        1.0e-6 * bs_model_.config().bandwidth_hz,
        task_metric_observer_mode_.c_str());
  }

  ~CommunicationProxy() override {
    if (shared_status_) {
      try {
        std::ostringstream status;
        status << "{\"ready\":false,\"process\":\"racer\",\"pid\":"
               << ::getpid() << ",\"stopping\":true}";
        shared_status_->publish(status.str(), rl_state_sequence_,
                                rl_last_decision_s_);
      } catch (const std::exception &) {
      }
    }
    stopTaskObserver();
    exportObservedOccupiedVoxels();
  }

 private:
  struct DeliveryFlow {
    int origin_sender{};
    std::size_t topic_index{};
    double born_at{};
    std::size_t bytes{};
    std::vector<bool> delivered;
    std::vector<bool> intended;
    bool ap_received{false};
  };

  struct CachedPeerMessage {
    std::shared_ptr<rclcpp::SerializedMessage> message;
    std::shared_ptr<DeliveryFlow> flow;
    std::size_t bytes{};
    double born_at{};
    std::uint64_t version{};
  };

  // Immutable PHY view captured at the same Fast State boundary that the RL
  // policy consumed.  The Proxy keeps this separate from links_, which Sionna
  // is free to update while the resulting action is waiting to be applied.
  struct ActionChannelSnapshot {
    std::uint64_t channel_version{};
    std::uint64_t step_id{};
    double sim_time_s{};
    std::vector<float> effective_snr_db;
    std::vector<std::uint8_t> available;
  };

  struct AdaptiveChunkQuota {
    bool available{false};
    double effective_snr_db{-std::numeric_limits<double>::infinity()};
    std::string_view mcs{};
    std::size_t chunks{};
  };

  // An uncached RACER chunk is requested from the selected UAV locally, but
  // its eventual radio packet must retain the action and decision-time PHY
  // snapshot that caused the request.
  struct PendingBsChunkRequest {
    double requested_at{};
    double born_at{};
    int action_sender{-1};
    std::uint64_t action_id{};
    std::uint64_t action_step_id{};
    std::shared_ptr<const ActionChannelSnapshot> action_channel_snapshot;
  };

  struct InflightPayloadPrecache {
    int sender{-1};
    double requested_at{};
  };

  struct PendingPacket {
    std::shared_ptr<rclcpp::SerializedMessage> message;
    std::shared_ptr<DeliveryFlow> flow;
    std::size_t topic_index{};
    RouteStage route{RouteStage::kDirect};
    int final_receiver{-1};
    double born_at{};
    double enqueued_at{};
    double delivery_at{};
    std::size_t bytes{};
    int attempts{};
    bool transmitting{false};
    bool bs_csma_collision{false};
    // Fragment one attempt across actual BS PHY-slot grants. Propagation and
    // reliable retries still use the normal completion/TTL/action-snapshot path.
    bool bs_ofdma_started{false};
    double bs_ofdma_remaining_bits{};
    bool has_chunk_key{false};
    bool reserved_pair_control{false};
    bool periodic_bs_upload_request{false};
    // Physical endpoints for this hop.  For map traffic these are distinct
    // from chunk_key.owner, which permanently identifies the UAV that first
    // created the chunk.
    int current_sender{-1};
    int current_receiver{-1};
    // UAV row whose B[i,j] action scheduled a BS downlink.  This is routing
    // attribution only and never changes the chunk's origin owner/version.
    int action_sender{-1};
    // A packet can remain queued after Sionna publishes a newer LinkQuality.
    // Keep the action provenance and immutable channel view with the packet
    // so start, completion and every retry use the decision-time channel.
    std::uint64_t action_id{};
    std::uint64_t action_step_id{};
    std::uint64_t channel_version{};
    std::shared_ptr<const ActionChannelSnapshot> action_channel_snapshot;
    ChunkKey chunk_key{};
    std::uint64_t peer_message_version{};
  };

  enum class PacketStartResult {
    kStarted,
    kDeferred,
    kNoLink,
  };

  enum class PairControlStage {
    kAwaitPrepared,
    kAwaitCommit,
    kAwaitCommitted,
  };

  struct PairControlReservation {
    std::uint64_t transaction_id{};
    int proposer{-1};
    int responder{-1};
    double started_at{};
    double expires_at{};
    PairControlStage stage{PairControlStage::kAwaitPrepared};
  };

  struct PairControlMessage {
    bool valid{false};
    bool response{false};
    std::uint64_t transaction_id{};
    int sender{-1};
    int receiver{-1};
    std::uint8_t phase{};
    std::int32_t status{};
  };

  struct LinkQueue {
    std::deque<PendingPacket> packets;
    std::size_t bytes{};
  };

  struct UdpReceiverState {
    int receiver{};
    bool force_lossless{false};
    bool half_duplex_blocked{false};
    bool channel_unavailable{false};
    double received_bits{};
    double snr_linear_bit_sum{};
    double last_snr_db{-std::numeric_limits<double>::infinity()};
    int last_allocated_prbs{};
  };

  struct UdpDatagram {
    std::shared_ptr<rclcpp::SerializedMessage> message;
    std::shared_ptr<DeliveryFlow> flow;
    std::size_t topic_index{};
    double born_at{};
    double enqueued_at{};
    double delivery_at{};
    std::size_t bytes{};
    double total_bits{};
    double remaining_bits{};
    std::uint64_t packet_id{};
    std::uint64_t first_slot{};
    std::uint64_t last_slot{};
    std::uint64_t allocated_prb_slots{};
    int last_allocated_prbs{};
    bool started{false};
    bool has_chunk_key{false};
    ChunkKey chunk_key{};
    std::vector<UdpReceiverState> receivers;
  };

  struct UavUdpQueue {
    std::deque<UdpDatagram> datagrams;
    std::size_t bytes{};
  };

  struct CachedChunk {
    std::shared_ptr<rclcpp::SerializedMessage> message;
    std::size_t bytes{};
    std::array<double, 3> voxel_center{};
    double mean_voxel_norm_squared{};
    bool has_voxel_geometry{false};
  };

  struct TrajectorySummary {
    std::array<double, 3> goal{};
    double length{};
    double expected_execution_time{};
    bool valid{false};
  };

  struct TaskQualitySample {
    double time_s{};
    double redundancy{};
    double map_iou{-1.0};
    double map_coverage{-1.0};
    std::uint64_t task_step{};
    double source_sim_time_s{};
    std::uint64_t bs_uplink_prb_slots{};
    std::uint64_t bs_downlink_prb_slots{};
    std::uint64_t direct_u2u_prb_slots{};
  };

  struct BsGlobalMapMetrics {
    double iou{-1.0};
    double coverage{};
    std::size_t known{};
    std::size_t total{};
    std::size_t occupied{};
    std::size_t intersection{};
  };

  struct BsMapPromptSummary {
    BsGlobalMapMetrics metrics;
    std::size_t frontier_count{};
    std::array<double, 3> frontier_centroid{};
  };

  struct BsObservableTelemetry {
    std::vector<std::array<double, 3>> positions;
    std::vector<std::array<double, 3>> velocities;
    std::vector<double> yaws;
    std::vector<std::string> position_sources;
    std::vector<std::string> racer_states;
    std::vector<TrajectorySummary> trajectories;
  };

  struct FastBoundarySnapshot {
    std::uint64_t step_id{};
    std::uint64_t sequence{};
    std::uint64_t communication_slot{};
    std::uint64_t decision_index{};
    double sim_time_s{};
    double interval_start_sim_time_s{};
    double interval_duration_s{};
    std::uint64_t action_version{};
    std::uint64_t policy_version{};
    std::uint64_t action_generated_wall_time_ns{};
    std::uint64_t action_source_guidance_id{};
    std::uint64_t action_source_physical_version{};
    std::uint64_t action_source_communication_version{};
    std::uint64_t action_source_step_id{};
    double action_source_sim_time_s{};
    double action_age{};
    std::uint64_t action_held_slots{};
    std::uint64_t physical_version{};
    double coverage{};
    double coverage_delta{};
    bool terminated{};
    bool truncated{};
    bool completed_transition{};
    double interval_uplink_prb_slots{};
    double interval_downlink_prb_slots{};
    double interval_direct_prb_slots{};
    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<float, 3>> velocities;
    std::vector<float> yaws;
    std::vector<float> channel;
    std::vector<float> pair_aoi;
    std::vector<float> bs_aoi;
    std::vector<std::uint64_t> pair_missing_bytes;
    std::vector<std::uint64_t> bs_missing_bytes;
    std::vector<std::uint64_t> bs_uav_missing_bytes;
    std::vector<std::uint64_t> uplink_queue_bytes;
    std::vector<std::uint64_t> relay_queue_bytes;
    std::uint64_t bs_known_chunks{};
    std::vector<std::uint8_t> relay_action;
    std::vector<std::uint8_t> upload_action;
    std::uint64_t guidance_id{};
    std::vector<float> guidance_task_dependency;
    std::vector<float> guidance_semantic_importance;
    double build_ms{};
    double build_mean_ms{};
    double build_max_ms{};
    double missing_update_mean_ms{};
    double queue_update_mean_ms{};
  };

  enum class TaskObservationKind {
    kUavChunk,
    kBsChunk,
    kTrajectory,
    kBoundary,
  };

  struct TaskObservation {
    TaskObservationKind kind{TaskObservationKind::kTrajectory};
    int sender{-1};
    std::shared_ptr<rclcpp::SerializedMessage> serialized;
    std::shared_ptr<racer_fidelity_msgs::msg::ChunkData> chunk;
    std::shared_ptr<FastBoundarySnapshot> boundary;
  };

  template <typename Message>
  bool deserialize(
      const std::shared_ptr<rclcpp::SerializedMessage> &serialized,
      Message &message) const {
    try {
      rclcpp::Serialization<Message> serializer;
      serializer.deserialize_message(serialized.get(), &message);
      return true;
    } catch (const std::exception &exception) {
      RCLCPP_WARN(get_logger(), "failed to inspect serialized map message: %s",
                  exception.what());
      return false;
    }
  }

  void loadGroundTruthOccupiedVoxels() {
    if (ground_truth_occupied_voxels_path_.empty()) {
      if (require_ground_truth_map_) {
        throw std::runtime_error(
            "require_ground_truth_map=true needs "
            "ground_truth_occupied_voxels_path");
      }
      return;
    }
    std::ifstream input(ground_truth_occupied_voxels_path_);
    if (!input) {
      throw std::runtime_error(
          "cannot open ground-truth occupied voxel address file: " +
          ground_truth_occupied_voxels_path_);
    }
    std::uint64_t address{};
    while (input >> address) {
      if (address > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "ground-truth voxel address exceeds uint32 range");
      }
      ground_truth_occupied_voxels_.insert(
          static_cast<std::uint32_t>(address));
    }
    if (ground_truth_occupied_voxels_.empty()) {
      throw std::runtime_error(
          "ground-truth occupied voxel address file is empty");
    }
  }

  void configureBsCoverageGeometry() {
    if (!std::isfinite(bs_map_resolution_) || bs_map_resolution_ <= 0.0 ||
        !std::isfinite(bs_map_ground_height_)) {
      throw std::runtime_error(
          "BS map coverage requires a finite positive resolution and "
          "finite ground height");
    }
    const double inverse_resolution = 1.0 / bs_map_resolution_;
    bs_map_origin_ = {
        -0.5 * bs_map_size_[0], -0.5 * bs_map_size_[1],
        bs_map_ground_height_};
    std::uint64_t full_map_voxels = 1U;
    std::uint64_t planning_box_voxels = 1U;
    for (std::size_t axis = 0; axis < 3U; ++axis) {
      if (!std::isfinite(bs_map_size_[axis]) || bs_map_size_[axis] <= 0.0 ||
          !std::isfinite(bs_map_box_min_[axis]) ||
          !std::isfinite(bs_map_box_max_[axis]) ||
          bs_map_box_max_[axis] <= bs_map_box_min_[axis]) {
        throw std::runtime_error("invalid BS map coverage geometry");
      }
      bs_map_voxel_num_[axis] = static_cast<std::int64_t>(
          std::ceil(bs_map_size_[axis] * inverse_resolution));
      bs_map_box_begin_[axis] = static_cast<std::int64_t>(std::floor(
          (bs_map_box_min_[axis] - bs_map_origin_[axis]) * inverse_resolution));
      bs_map_box_end_[axis] = static_cast<std::int64_t>(std::floor(
          (bs_map_box_max_[axis] - bs_map_origin_[axis]) * inverse_resolution));
      if (bs_map_voxel_num_[axis] <= 0 ||
          bs_map_box_end_[axis] <= bs_map_box_begin_[axis]) {
        throw std::runtime_error("invalid BS map coverage voxel geometry");
      }
      full_map_voxels *=
          static_cast<std::uint64_t>(bs_map_voxel_num_[axis]);
      planning_box_voxels *= static_cast<std::uint64_t>(
          bs_map_box_end_[axis] - bs_map_box_begin_[axis]);
    }
    if (full_map_voxels == 0U ||
        full_map_voxels >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) + 1U ||
        planning_box_voxels == 0U ||
        planning_box_voxels >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("BS map coverage voxel count is out of range");
    }
    bs_full_map_voxels_ = full_map_voxels;
    bs_planning_box_voxels_ =
        static_cast<std::size_t>(planning_box_voxels);
    bs_known_voxel_bits_.assign(
        static_cast<std::size_t>((full_map_voxels + 63U) / 64U), 0U);
  }

  bool bsAddressInPlanningBox(std::uint32_t address) const noexcept {
    if (static_cast<std::uint64_t>(address) >= bs_full_map_voxels_) {
      return false;
    }
    std::uint64_t remainder = address;
    const std::uint64_t yz =
        static_cast<std::uint64_t>(bs_map_voxel_num_[1]) *
        static_cast<std::uint64_t>(bs_map_voxel_num_[2]);
    const std::int64_t x = static_cast<std::int64_t>(remainder / yz);
    remainder %= yz;
    const std::int64_t y = static_cast<std::int64_t>(
        remainder / static_cast<std::uint64_t>(bs_map_voxel_num_[2]));
    const std::int64_t z = static_cast<std::int64_t>(
        remainder % static_cast<std::uint64_t>(bs_map_voxel_num_[2]));
    return x >= bs_map_box_begin_[0] && x < bs_map_box_end_[0] &&
           y >= bs_map_box_begin_[1] && y < bs_map_box_end_[1] &&
           z >= bs_map_box_begin_[2] && z < bs_map_box_end_[2];
  }

  void markBsKnownVoxel(std::uint32_t address) {
    if (!bsAddressInPlanningBox(address)) return;
    const std::size_t word = static_cast<std::size_t>(address / 64U);
    const std::uint64_t mask = std::uint64_t{1U} << (address % 64U);
    if ((bs_known_voxel_bits_[word] & mask) != 0U) return;
    bs_known_voxel_bits_[word] |= mask;
    ++bs_known_planning_voxels_;
  }

  void applyTaskChunkObservation(
      const racer_fidelity_msgs::msg::ChunkData &chunk,
      bool update_uav_observation, bool update_bs_map) {
    std::lock_guard<std::mutex> lock(task_metric_mutex_);
    if (update_uav_observation && chunk.chunk_drone_id >= 1 &&
        chunk.chunk_drone_id <= drone_count_) {
      auto &observed = uav_observed_voxels_[
          static_cast<std::size_t>(chunk.chunk_drone_id - 1)];
      for (const auto address : chunk.voxel_adrs) {
        if (!observed.insert(address).second) continue;
        ++redundant_observation_sum_;
        ++uav_observation_refcounts_[address];
      }
    }
    if (update_bs_map) {
      const auto count =
          std::min(chunk.voxel_adrs.size(), chunk.voxel_occ.size());
      for (std::size_t index = 0; index < count; ++index) {
        const auto address = chunk.voxel_adrs[index];
        // A transmitted map chunk explicitly classifies every address as
        // either FREE (0) or OCCUPIED (1). Both are known SDF voxels. Keep a
        // compact de-duplicated mask so BS coverage uses the same numerator
        // as the per-UAV planning-box coverage diagnostic.
        markBsKnownVoxel(address);
        if (chunk.voxel_occ[index] == 1U) {
          if (bs_occupied_voxels_.insert(address).second &&
              ground_truth_occupied_voxels_.count(address) != 0U) {
            ++bs_gt_intersection_voxels_;
          }
        } else if (bs_occupied_voxels_.erase(address) != 0U &&
                   ground_truth_occupied_voxels_.count(address) != 0U) {
          --bs_gt_intersection_voxels_;
        }
        refreshBsFrontierAroundWithoutLock(address);
      }
    }
  }

  TrajectorySummary summarizeTrajectory(
      const std::shared_ptr<rclcpp::SerializedMessage> &message) const {
    racer_fidelity_msgs::msg::Bspline trajectory;
    TrajectorySummary summary;
    if (!message || !deserialize(message, trajectory) ||
        trajectory.pos_pts.empty()) {
      return summary;
    }
    const auto &goal = trajectory.pos_pts.back();
    summary.goal = {goal.x, goal.y, goal.z};
    for (std::size_t index = 1; index < trajectory.pos_pts.size(); ++index) {
      const auto &left = trajectory.pos_pts[index - 1U];
      const auto &right = trajectory.pos_pts[index];
      summary.length += std::hypot(
          std::hypot(right.x - left.x, right.y - left.y), right.z - left.z);
    }
    if (trajectory.knots.size() >= 2U) {
      summary.expected_execution_time = std::max(
          0.0, trajectory.knots.back() - trajectory.knots.front());
    }
    summary.valid = true;
    return summary;
  }

  void applyTrajectoryObservation(
      int sender,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) {
    const auto summary = summarizeTrajectory(message);
    if (!summary.valid) return;
    std::lock_guard<std::mutex> lock(task_metric_mutex_);
    trajectory_summaries_[static_cast<std::size_t>(sender)] = summary;
  }

  void enqueueTaskObservation(TaskObservation observation) {
    {
      std::lock_guard<std::mutex> lock(task_observer_queue_mutex_);
      task_observer_queue_.push_back(std::move(observation));
      task_observer_peak_queue_depth_ = std::max(
          task_observer_peak_queue_depth_, task_observer_queue_.size());
      ++task_observer_events_enqueued_;
    }
    task_observer_condition_.notify_one();
  }

  void taskObserverLoop() {
    while (true) {
      TaskObservation observation;
      {
        std::unique_lock<std::mutex> lock(task_observer_queue_mutex_);
        task_observer_condition_.wait(lock, [this]() {
          return task_observer_stopping_ || !task_observer_queue_.empty();
        });
        if (task_observer_queue_.empty()) {
          if (task_observer_stopping_) break;
          continue;
        }
        observation = std::move(task_observer_queue_.front());
        task_observer_queue_.pop_front();
      }
      if (observation.kind == TaskObservationKind::kUavChunk &&
          observation.chunk) {
        applyTaskChunkObservation(
            *observation.chunk, true,
            mode_ == "ideal" && !rl_bs_scheduler_enabled_);
      } else if (observation.kind == TaskObservationKind::kBsChunk &&
                 observation.serialized) {
        racer_fidelity_msgs::msg::ChunkData chunk;
        if (deserialize(observation.serialized, chunk)) {
          applyTaskChunkObservation(chunk, false, true);
        }
      } else if (observation.kind == TaskObservationKind::kTrajectory &&
                 observation.serialized && observation.sender >= 0 &&
                 observation.sender < drone_count_) {
        applyTrajectoryObservation(observation.sender,
                                   observation.serialized);
      } else if (observation.kind == TaskObservationKind::kBoundary &&
                 observation.boundary) {
        // FIFO ordering guarantees that all chunk/trajectory events submitted
        // before this exact simulation boundary are reflected in metric k.
        computeAndPublishTaskMetric(*observation.boundary);
      }
      ++task_observer_events_processed_;
    }
  }

  void stopTaskObserver() {
    if (!task_observer_thread_.joinable()) return;
    {
      std::lock_guard<std::mutex> lock(task_observer_queue_mutex_);
      task_observer_stopping_ = true;
    }
    task_observer_condition_.notify_one();
    task_observer_thread_.join();
  }

  std::size_t taskObserverQueueDepth() const {
    std::lock_guard<std::mutex> lock(task_observer_queue_mutex_);
    return task_observer_queue_.size();
  }

  std::vector<TrajectorySummary> trajectorySummariesSnapshot() const {
    std::lock_guard<std::mutex> lock(task_metric_mutex_);
    return trajectory_summaries_;
  }

  BsObservableTelemetry bsObservableTelemetrySnapshot(
      const FastBoundarySnapshot &state) {
    BsObservableTelemetry output;
    const auto n = static_cast<std::size_t>(drone_count_);
    output.positions.resize(n);
    output.velocities.resize(n);
    output.yaws.assign(n, 0.0);
    output.position_sources.assign(n, "unavailable");
    output.racer_states.assign(n, "UNKNOWN");
    output.trajectories.resize(n);

    // /odom is an independent reliable control/telemetry channel in this
    // simulator. Use it only as a labelled fallback; successfully uploaded
    // DroneState/Bspline messages take precedence.
    std::lock_guard<std::mutex> lock(communication_mutex_);
    const auto state_topic = topicIndex("drone_state");
    const auto trajectory_topic = topicIndex("trajectory");
    for (int drone = 0; drone < drone_count_; ++drone) {
      const auto index = static_cast<std::size_t>(drone);
      if (index < state.positions.size() && uav_position_valid_[index]) {
        output.positions[index] = {
            state.positions[index][0], state.positions[index][1],
            state.positions[index][2]};
        output.velocities[index] = {
            state.velocities[index][0], state.velocities[index][1],
            state.velocities[index][2]};
        output.yaws[index] = state.yaws[index];
        output.position_sources[index] = "reliable_control_telemetry";
      }

      const auto &cached_state = bs_latest_peer_messages_[index][state_topic];
      racer_fidelity_msgs::msg::DroneState message;
      if (cached_state.message && deserialize(cached_state.message, message)) {
        if (message.pos.size() >= 3U) {
          output.positions[index] = {
              message.pos[0], message.pos[1], message.pos[2]};
          output.position_sources[index] = "bs_received_drone_state";
        }
        if (message.vel.size() >= 3U) {
          output.velocities[index] = {
              message.vel[0], message.vel[1], message.vel[2]};
        }
        output.yaws[index] = message.yaw;
        output.racer_states[index] =
            message.requesting_work
                ? "REQUESTING_WORK"
                : (message.grid_ids.empty() ? "IDLE" : "ASSIGNED");
      }

      const auto &cached_trajectory =
          bs_latest_peer_messages_[index][trajectory_topic];
      output.trajectories[index] =
          summarizeTrajectory(cached_trajectory.message);
    }
    return output;
  }

  double redundantExplorationRatio() const {
    std::lock_guard<std::mutex> lock(task_metric_mutex_);
    if (redundant_observation_sum_ == 0U) return 0.0;
    return std::clamp(
        1.0 - static_cast<double>(uav_observation_refcounts_.size()) /
                  static_cast<double>(redundant_observation_sum_),
        0.0, 1.0);
  }

  BsGlobalMapMetrics bsGlobalMapMetrics() const {
    std::lock_guard<std::mutex> lock(task_metric_mutex_);
    BsGlobalMapMetrics output;
    output.known = bs_known_planning_voxels_;
    output.total = bs_planning_box_voxels_;
    output.coverage = output.total == 0U
                          ? 0.0
                          : static_cast<double>(output.known) /
                                static_cast<double>(output.total);
    output.occupied = bs_occupied_voxels_.size();
    if (ground_truth_occupied_voxels_.empty()) return output;
    output.intersection = bs_gt_intersection_voxels_;
    const std::size_t union_size =
        bs_occupied_voxels_.size() + ground_truth_occupied_voxels_.size() -
        bs_gt_intersection_voxels_;
    output.iou = union_size == 0U
                     ? 1.0
                     : static_cast<double>(bs_gt_intersection_voxels_) /
                           static_cast<double>(union_size);
    return output;
  }

  std::array<std::int64_t, 3> bsAddressToIndex(
      std::uint32_t address) const noexcept {
    std::uint64_t remainder = address;
    const std::uint64_t yz =
        static_cast<std::uint64_t>(bs_map_voxel_num_[1]) *
        static_cast<std::uint64_t>(bs_map_voxel_num_[2]);
    const auto x = static_cast<std::int64_t>(remainder / yz);
    remainder %= yz;
    const auto y = static_cast<std::int64_t>(
        remainder / static_cast<std::uint64_t>(bs_map_voxel_num_[2]));
    const auto z = static_cast<std::int64_t>(
        remainder % static_cast<std::uint64_t>(bs_map_voxel_num_[2]));
    return {x, y, z};
  }

  CachedChunk cacheChunk(
      std::shared_ptr<rclcpp::SerializedMessage> message,
      std::size_t bytes,
      const racer_fidelity_msgs::msg::ChunkData &chunk) const {
    CachedChunk cached{std::move(message), bytes};
    double norm_sum = 0.0;
    std::size_t count = 0U;
    for (const auto address : chunk.voxel_adrs) {
      if (static_cast<std::uint64_t>(address) >= bs_full_map_voxels_) continue;
      const auto index = bsAddressToIndex(address);
      double norm_squared = 0.0;
      for (std::size_t axis = 0; axis < 3U; ++axis) {
        const double coordinate = bs_map_origin_[axis] +
            (static_cast<double>(index[axis]) + 0.5) * bs_map_resolution_;
        cached.voxel_center[axis] += coordinate;
        norm_squared += coordinate * coordinate;
      }
      norm_sum += norm_squared;
      ++count;
    }
    if (count > 0U) {
      for (double &coordinate : cached.voxel_center) {
        coordinate /= static_cast<double>(count);
      }
      cached.mean_voxel_norm_squared = norm_sum / static_cast<double>(count);
      cached.has_voxel_geometry = true;
    }
    return cached;
  }

  std::uint32_t bsIndexToAddress(
      const std::array<std::int64_t, 3> &index) const noexcept {
    const auto address =
        (static_cast<std::uint64_t>(index[0]) *
             static_cast<std::uint64_t>(bs_map_voxel_num_[1]) +
         static_cast<std::uint64_t>(index[1])) *
            static_cast<std::uint64_t>(bs_map_voxel_num_[2]) +
        static_cast<std::uint64_t>(index[2]);
    return static_cast<std::uint32_t>(address);
  }

  bool bsKnownWithoutLock(std::uint32_t address) const noexcept {
    const std::size_t word = static_cast<std::size_t>(address / 64U);
    const std::uint64_t mask = std::uint64_t{1U} << (address % 64U);
    return word < bs_known_voxel_bits_.size() &&
           (bs_known_voxel_bits_[word] & mask) != 0U;
  }

  bool bsIndexInPlanningBox(
      const std::array<std::int64_t, 3> &index) const noexcept {
    return index[0] >= bs_map_box_begin_[0] &&
           index[0] < bs_map_box_end_[0] &&
           index[1] >= bs_map_box_begin_[1] &&
           index[1] < bs_map_box_end_[1] &&
           index[2] >= bs_map_box_begin_[2] &&
           index[2] < bs_map_box_end_[2];
  }

  bool bsFrontierWithoutLock(std::uint32_t address) const noexcept {
    if (!bsAddressInPlanningBox(address) ||
        !bsKnownWithoutLock(address) ||
        bs_occupied_voxels_.count(address) != 0U) {
      return false;
    }
    const auto index = bsAddressToIndex(address);
    constexpr std::array<std::array<std::int64_t, 3>, 6> offsets{{
        {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
        {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}},
    }};
    for (const auto &offset : offsets) {
      const std::array<std::int64_t, 3> neighbor{
          index[0] + offset[0], index[1] + offset[1],
          index[2] + offset[2]};
      if (bsIndexInPlanningBox(neighbor) &&
          !bsKnownWithoutLock(bsIndexToAddress(neighbor))) {
        return true;
      }
    }
    return false;
  }

  void refreshBsFrontierAroundWithoutLock(std::uint32_t address) {
    const auto index = bsAddressToIndex(address);
    constexpr std::array<std::array<std::int64_t, 3>, 7> offsets{{
        {{0, 0, 0}}, {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
        {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}},
    }};
    for (const auto &offset : offsets) {
      const std::array<std::int64_t, 3> candidate{
          index[0] + offset[0], index[1] + offset[1],
          index[2] + offset[2]};
      if (!bsIndexInPlanningBox(candidate)) continue;
      const auto candidate_address = bsIndexToAddress(candidate);
      if (bsFrontierWithoutLock(candidate_address)) {
        bs_frontier_voxels_.insert(candidate_address);
      } else {
        bs_frontier_voxels_.erase(candidate_address);
      }
    }
  }

  BsMapPromptSummary bsMapPromptSummary(
      const BsGlobalMapMetrics &metrics) const {
    BsMapPromptSummary output;
    output.metrics = metrics;
    std::array<double, 3> centroid_sum{};
    std::lock_guard<std::mutex> lock(task_metric_mutex_);
    for (const auto address : bs_frontier_voxels_) {
      const auto index = bsAddressToIndex(address);
      ++output.frontier_count;
      for (std::size_t axis = 0; axis < 3U; ++axis) {
        centroid_sum[axis] += bs_map_origin_[axis] +
            (static_cast<double>(index[axis]) + 0.5) * bs_map_resolution_;
      }
    }
    if (output.frontier_count > 0U) {
      for (std::size_t axis = 0; axis < 3U; ++axis) {
        output.frontier_centroid[axis] = centroid_sum[axis] /
            static_cast<double>(output.frontier_count);
      }
    } else {
      for (std::size_t axis = 0; axis < 3U; ++axis) {
        output.frontier_centroid[axis] =
            0.5 * (bs_map_box_min_[axis] + bs_map_box_max_[axis]);
      }
    }
    return output;
  }

  void publishLlmState(const FastBoundarySnapshot &state,
                       const BsGlobalMapMetrics &map_metrics) {
    if (!shared_communication_) return;
    if (rl_bs_event_driven_one_shot_) {
      if (rl_have_last_event_llm_state_time_ &&
          state.sim_time_s - rl_last_event_llm_state_time_s_ + 1.0e-9 <
              rl_llm_state_period_s_) {
        return;
      }
      rl_have_last_event_llm_state_time_ = true;
      rl_last_event_llm_state_time_s_ = state.sim_time_s;
    } else {
      const auto llm_step_interval = static_cast<std::uint64_t>(std::max(
          1.0,
          std::round(rl_llm_state_period_s_ / rl_bs_decision_period_s_)));
      if (state.step_id % llm_step_interval != 0U) return;
    }
    const auto telemetry = bsObservableTelemetrySnapshot(state);
    const auto map_summary = bsMapPromptSummary(map_metrics);
    const double bs_coverage_delta = have_last_llm_bs_coverage_
        ? map_metrics.coverage - last_llm_bs_coverage_
        : 0.0;
    have_last_llm_bs_coverage_ = true;
    last_llm_bs_coverage_ = map_metrics.coverage;
    std::ostringstream json;
    json << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "{\"sequence\":" << state.sequence
         << ",\"step_id\":" << state.step_id
         << ",\"sim_time\":" << state.sim_time_s
         << ",\"sim_time_s\":" << state.sim_time_s
         << ",\"task_step\":" << state.step_id
         << ",\"task_time_s\":" << state.sim_time_s
         << ",\"task_step_duration_s\":" << state.interval_duration_s
         << ",\"communication_slot_index\":" << state.communication_slot
         << ",\"rl_decision_index\":" << state.decision_index
         << ",\"communication_slot_duration_s\":"
         << rl_bs_communication_slot_s_
         << ",\"rl_decision_interval_s\":" << state.interval_duration_s
         << ",\"positions\":[";
    for (std::size_t drone = 0; drone < telemetry.positions.size(); ++drone) {
      if (drone) json << ',';
      const auto &point = telemetry.positions[drone];
      json << '[' << point[0] << ',' << point[1] << ',' << point[2] << ']';
    }
    json << "],\"velocities\":[";
    for (std::size_t drone = 0; drone < telemetry.velocities.size(); ++drone) {
      if (drone) json << ',';
      const auto &value = telemetry.velocities[drone];
      json << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
    }
    json << "],\"yaws\":[";
    for (std::size_t drone = 0; drone < telemetry.yaws.size(); ++drone) {
      if (drone) json << ',';
      json << telemetry.yaws[drone];
    }
    json << "],\"fsm_states\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      json << '\"' << telemetry.racer_states[static_cast<std::size_t>(drone)]
           << '\"';
    }
    json << "],\"position_sources\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      json << '\"' << telemetry.position_sources[static_cast<std::size_t>(drone)]
           << '\"';
    }
    json << "],\"channel_snr_db\":[";
    for (int sender = 0; sender < radio_node_count_; ++sender) {
      if (sender) json << ',';
      json << '[';
      for (int receiver = 0; receiver < radio_node_count_; ++receiver) {
        if (receiver) json << ',';
        json << state.channel[static_cast<std::size_t>(
            sender * radio_node_count_ + receiver)];
      }
      json << ']';
    }
    json << "],\"pair_aoi_s\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        json << state.pair_aoi[static_cast<std::size_t>(
            owner * drone_count_ + receiver)];
      }
      json << ']';
    }
    json << "],\"bs_aoi_s\":[";
    for (std::size_t owner = 0; owner < state.bs_aoi.size(); ++owner) {
      if (owner) json << ',';
      json << state.bs_aoi[owner];
    }
    json << "],\"pair_missing_bytes\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        json << state.pair_missing_bytes[static_cast<std::size_t>(
            owner * drone_count_ + receiver)];
      }
      json << ']';
    }
    json << "],\"uav_bs_missing_bytes\":[";
    for (std::size_t owner = 0; owner < state.bs_missing_bytes.size(); ++owner) {
      if (owner) json << ',';
      json << state.bs_missing_bytes[owner];
    }
    json << "],\"bs_uav_missing_bytes\":[";
    for (std::size_t receiver = 0;
         receiver < state.bs_uav_missing_bytes.size(); ++receiver) {
      if (receiver) json << ',';
      json << state.bs_uav_missing_bytes[receiver];
    }
    json << "],\"uplink_queue_bytes\":[";
    for (std::size_t owner = 0; owner < state.uplink_queue_bytes.size(); ++owner) {
      if (owner) json << ',';
      json << state.uplink_queue_bytes[owner];
    }
    json << "],\"relay_queue_bytes\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        json << state.relay_queue_bytes[static_cast<std::size_t>(
            owner * drone_count_ + receiver)];
      }
      json << ']';
    }
    json << "],\"trajectory_summary\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      const auto &summary = telemetry.trajectories[static_cast<std::size_t>(drone)];
      const auto &goal = summary.valid
                             ? summary.goal
                             : std::array<double, 3>{
                                   telemetry.positions[static_cast<std::size_t>(drone)][0],
                                   telemetry.positions[static_cast<std::size_t>(drone)][1],
                                   telemetry.positions[static_cast<std::size_t>(drone)][2]};
      json << "{\"goal_position\":[" << goal[0] << ',' << goal[1] << ','
           << goal[2] << "],\"trajectory_length\":" << summary.length
           << ",\"expected_execution_time\":"
           << summary.expected_execution_time
           << ",\"available\":" << (summary.valid ? "true" : "false")
           << '}';
    }
    const double unknown_ratio = std::clamp(
        1.0 - map_metrics.coverage, 0.0, 1.0);
    json << "],\"region_summary\":[{\"region_id\":0,"
            "\"region_centroid\":["
         << map_summary.frontier_centroid[0] << ','
         << map_summary.frontier_centroid[1] << ','
         << map_summary.frontier_centroid[2]
         << "],\"normalized_region_size\":" << unknown_ratio
         << ",\"unknown_ratio\":" << unknown_ratio
         << ",\"frontier_count\":" << map_summary.frontier_count
         << "}],\"bs_map_summary\":{"
            "\"representation\":\"bs_received_known_voxels\","
            "\"bs_map_coverage\":"
         << map_metrics.coverage
         << ",\"bs_known_voxels\":" << map_metrics.known
         << ",\"bs_total_voxels\":" << map_metrics.total
         << ",\"bs_unknown_ratio\":" << unknown_ratio
         << ",\"bs_known_chunks\":" << state.bs_known_chunks
         << ",\"bs_frontier_count\":" << map_summary.frontier_count
         << "},\"bs_map_coverage_delta\":" << bs_coverage_delta
         << ",\"bs_global_map_coverage\":" << map_metrics.coverage
         << ",\"terminated\":" << (state.terminated ? "true" : "false")
         << ",\"truncated\":" << (state.truncated ? "true" : "false")
         << '}';
    const auto version = shared_communication_->publish(
        json.str(), state.communication_slot, state.sim_time_s);
    if (shared_state_event_) shared_state_event_->poke();
    RCLCPP_INFO(get_logger(),
                "RACER_LLM_STATE step_id=%llu sim_time=%.9f version=%llu",
                static_cast<unsigned long long>(state.step_id),
                state.sim_time_s, static_cast<unsigned long long>(version));
  }

  void computeAndPublishTaskMetric(const FastBoundarySnapshot &state) {
    const auto started = std::chrono::steady_clock::now();
    const double redundancy = redundantExplorationRatio();
    const auto map_metrics = bsGlobalMapMetrics();
    std::ostringstream json;
    json << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "{\"step_id\":" << state.step_id
         << ",\"state_step_id\":" << state.step_id
         << ",\"transition_step_id\":"
         << (state.completed_transition && state.step_id > 0U
                 ? static_cast<long long>(state.step_id - 1U)
                 : -1LL)
         << ",\"sim_time\":" << state.sim_time_s
         << ",\"sim_time_s\":" << state.sim_time_s
         << ",\"task_step\":" << state.step_id
         << ",\"task_time_s\":" << state.sim_time_s
         << ",\"task_step_duration_s\":" << state.interval_duration_s
         << ",\"coverage\":" << state.coverage
         << ",\"coverage_delta\":" << state.coverage_delta
         << ",\"redundant_exploration_ratio\":" << redundancy
         << ",\"bs_global_map_iou\":" << map_metrics.iou
         << ",\"bs_global_map_coverage\":" << map_metrics.coverage
         << ",\"positions\":[";
    for (std::size_t drone = 0; drone < state.positions.size(); ++drone) {
      if (drone) json << ',';
      const auto &point = state.positions[drone];
      json << '[' << point[0] << ',' << point[1] << ',' << point[2] << ']';
    }
    json << "]}";
    std::uint64_t metric_sequence{};
    if (shared_task_metric_ring_) {
      metric_sequence = shared_task_metric_ring_->publish(
          json.str(), state.step_id, state.sim_time_s);
    }
    const double elapsed_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    {
      std::lock_guard<std::mutex> lock(task_metric_snapshot_mutex_);
      latest_task_quality_ = {state.sim_time_s, redundancy, map_metrics.iou,
                              map_metrics.coverage, state.step_id,
                              state.sim_time_s};
      latest_bs_map_metrics_ = map_metrics;
      task_metric_compute_total_ms_ += elapsed_ms;
      ++task_metric_compute_count_;
      task_metric_compute_max_ms_ =
          std::max(task_metric_compute_max_ms_, elapsed_ms);
    }
    publishLlmState(state, map_metrics);
    if (state.step_id % 10U == 0U) exportObservedOccupiedVoxels();
    RCLCPP_INFO(
        get_logger(),
        "RACER_ASYNC_TASK_METRIC step_id=%llu sim_time=%.9f "
        "metric_sequence=%llu task_metric_compute_ms=%.6f "
        "task_metric_compute_mean_ms=%.6f task_metric_compute_max_ms=%.6f",
        static_cast<unsigned long long>(state.step_id), state.sim_time_s,
        static_cast<unsigned long long>(metric_sequence), elapsed_ms,
        task_metric_compute_total_ms_ /
            static_cast<double>(task_metric_compute_count_),
        task_metric_compute_max_ms_);
  }

  void exportObservedOccupiedVoxels() const {
    if (observed_occupied_voxels_path_.empty()) return;
    const std::filesystem::path output_path(observed_occupied_voxels_path_);
    if (!output_path.parent_path().empty()) {
      std::error_code error;
      std::filesystem::create_directories(output_path.parent_path(), error);
    }
    const auto temporary = output_path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) return;
    std::vector<std::uint32_t> addresses;
    {
      std::lock_guard<std::mutex> lock(task_metric_mutex_);
      addresses.assign(bs_occupied_voxels_.begin(),
                       bs_occupied_voxels_.end());
    }
    std::sort(addresses.begin(), addresses.end());
    for (const auto address : addresses) output << address << '\n';
    output.close();
    if (output) std::rename(temporary.c_str(), output_path.string().c_str());
  }

  static void subtractSaturating(std::uint64_t &value, std::uint64_t amount) {
    value -= std::min(value, amount);
  }

  // Fast RL State cache chain -------------------------------------------------
  // Chunk-set mutations are the only source of missing-byte changes.  Keeping
  // the exact byte/count delta here makes the 10 Hz state snapshot independent
  // of the number of accumulated map chunks.
  bool insertKnownUavChunk(int receiver, const ChunkKey &key) {
    const auto started = std::chrono::steady_clock::now();
    if (receiver < 0 || receiver >= drone_count_) return false;
    auto &known = uav_chunks_[static_cast<std::size_t>(receiver)];
    if (!known.insert(key).second) return false;
    const auto receiver_index = static_cast<std::size_t>(receiver);
    if (bs_chunks_.count(key)) {
      ordered_uav_missing_chunks_[receiver_index].erase(key);
    } else {
      ordered_bs_missing_chunks_[receiver_index].insert(key);
    }
    ++downlink_inventory_version_;
    const auto repository = chunk_repository_.find(key);
    if (repository != chunk_repository_.end()) {
      const auto bytes = static_cast<std::uint64_t>(repository->second.bytes);

      // C_sender -> receiver is defined by possession, not provenance.  A UAV
      // that learned this chunk from another UAV can immediately advertise and
      // forward it just like original RACER's ChunkStamp anti-entropy loop.
      for (int target = 0; target < drone_count_; ++target) {
        if (target == receiver ||
            uav_chunks_[static_cast<std::size_t>(target)].count(key)) {
          continue;
        }
        cached_pair_missing_bytes_[receiver_index]
                                  [static_cast<std::size_t>(target)] += bytes;
        ++cached_pair_missing_chunks_[receiver_index]
                                     [static_cast<std::size_t>(target)];
      }
      if (!bs_chunks_.count(key)) {
        cached_bs_missing_bytes_[receiver_index] += bytes;
        ++cached_bs_missing_chunks_[receiver_index];
      } else {
        subtractSaturating(cached_bs_uav_missing_bytes_[receiver_index], bytes);
        subtractSaturating(cached_bs_uav_missing_chunks_[receiver_index], 1U);
      }

      // This receiver no longer lacks the chunk from any UAV that currently
      // has it, regardless of the immutable origin owner in ChunkKey.
      for (int sender = 0; sender < drone_count_; ++sender) {
        if (sender == receiver ||
            !uav_chunks_[static_cast<std::size_t>(sender)].count(key)) {
          continue;
        }
        subtractSaturating(
            cached_pair_missing_bytes_[static_cast<std::size_t>(sender)]
                                      [receiver_index],
            bytes);
        subtractSaturating(
            cached_pair_missing_chunks_[static_cast<std::size_t>(sender)]
                                       [receiver_index],
            1U);
      }
    }
    missing_bytes_update_total_ms_ += 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    ++missing_bytes_update_count_;
    return true;
  }

  bool insertKnownBsChunk(const ChunkKey &key) {
    const auto started = std::chrono::steady_clock::now();
    if (!bs_chunks_.insert(key).second) return false;
    bs_chunk_arrival_sequences_[key] = ++bs_chunk_arrival_sequence_;
    for (int drone = 0; drone < drone_count_; ++drone) {
      const auto drone_index = static_cast<std::size_t>(drone);
      if (uav_chunks_[drone_index].count(key)) {
        ordered_bs_missing_chunks_[drone_index].erase(key);
      } else {
        ordered_uav_missing_chunks_[drone_index].insert(key);
      }
    }
    ++downlink_inventory_version_;
    const auto repository = chunk_repository_.find(key);
    if (repository != chunk_repository_.end()) {
      const auto bytes = static_cast<std::uint64_t>(repository->second.bytes);
      for (int sender = 0; sender < drone_count_; ++sender) {
        if (!uav_chunks_[static_cast<std::size_t>(sender)].count(key)) continue;
        subtractSaturating(
            cached_bs_missing_bytes_[static_cast<std::size_t>(sender)], bytes);
        subtractSaturating(
            cached_bs_missing_chunks_[static_cast<std::size_t>(sender)], 1U);
      }
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (uav_chunks_[static_cast<std::size_t>(receiver)].count(key)) continue;
        cached_bs_uav_missing_bytes_[static_cast<std::size_t>(receiver)] +=
            bytes;
        ++cached_bs_uav_missing_chunks_[static_cast<std::size_t>(receiver)];
      }
    }
    missing_bytes_update_total_ms_ += 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    ++missing_bytes_update_count_;
    return true;
  }

  void cacheNewChunkDefinition(const ChunkKey &key, std::size_t bytes) {
    const auto started = std::chrono::steady_clock::now();
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (!uav_chunks_[static_cast<std::size_t>(sender)].count(key)) continue;
      const auto sender_index = static_cast<std::size_t>(sender);
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver == sender ||
            uav_chunks_[static_cast<std::size_t>(receiver)].count(key)) {
          continue;
        }
        cached_pair_missing_bytes_[sender_index]
                                  [static_cast<std::size_t>(receiver)] += bytes;
        ++cached_pair_missing_chunks_[sender_index]
                                     [static_cast<std::size_t>(receiver)];
      }
      if (!bs_chunks_.count(key)) {
        cached_bs_missing_bytes_[sender_index] += bytes;
        ++cached_bs_missing_chunks_[sender_index];
      }
    }
    if (bs_chunks_.count(key)) {
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (uav_chunks_[static_cast<std::size_t>(receiver)].count(key)) continue;
        cached_bs_uav_missing_bytes_[static_cast<std::size_t>(receiver)] +=
            bytes;
        ++cached_bs_uav_missing_chunks_[static_cast<std::size_t>(receiver)];
      }
    }
    missing_bytes_update_total_ms_ += 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    ++missing_bytes_update_count_;
  }

  void updateRouteQueueCache(const PendingPacket &packet, bool adding) {
    const auto started = std::chrono::steady_clock::now();
    if (!packet.has_chunk_key) return;
    auto update = [adding, &packet](std::uint64_t &value) {
      if (adding) {
        value += static_cast<std::uint64_t>(packet.bytes);
      } else {
        subtractSaturating(value, static_cast<std::uint64_t>(packet.bytes));
      }
    };
    if (packet.route == RouteStage::kBsUplink) {
      if (packet.current_sender < 0 ||
          packet.current_sender >= drone_count_) return;
      update(cached_uplink_queue_bytes_[static_cast<std::size_t>(
          packet.current_sender)]);
    } else if (packet.route == RouteStage::kBsDownlink &&
               packet.final_receiver >= 0 &&
               packet.final_receiver < drone_count_) {
      const int action_sender =
          packet.action_sender >= 0 && packet.action_sender < drone_count_
              ? packet.action_sender
              : packet.chunk_key.owner - 1;
      if (action_sender < 0 || action_sender >= drone_count_) return;
      update(cached_relay_queue_bytes_[static_cast<std::size_t>(action_sender)]
                                      [static_cast<std::size_t>(
                                          packet.final_receiver)]);
    } else {
      return;
    }
    queue_cache_update_total_ms_ += 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    ++queue_cache_update_count_;
  }

  void observeChunkStamps(
      int sender,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) {
    racer_fidelity_msgs::msg::ChunkStamps stamps;
    if (!deserialize(message, stamps) || sender < 0 ||
        sender >= drone_count_) {
      return;
    }
    const auto owner_count = std::min(
        stamps.idx_lists.size(), static_cast<std::size_t>(drone_count_));
    for (std::size_t owner = 0; owner < owner_count; ++owner) {
      const auto &ranges = stamps.idx_lists[owner].ids;
      for (std::size_t offset = 0; offset + 1U < ranges.size(); offset += 2U) {
        const std::uint32_t first = ranges[offset];
        const std::uint32_t last = ranges[offset + 1U];
        if (first == 0U || last < first || last - first > 1000000U) continue;
        for (std::uint32_t index = first; index <= last; ++index) {
          const ChunkKey key{static_cast<int>(owner) + 1, index};
          insertKnownUavChunk(sender, key);
          // The Proxy observes a UAV's own inventory announcement before it
          // enters the simulated radio.  Materialize each immutable payload
          // once from its original owner so later RL slots only perform radio
          // admission; holder copies never create duplicate cache work.
          if (bs_payload_precache_enabled_ && key.owner == sender + 1) {
            queuePayloadPrecache(key);
          }
          if (index == std::numeric_limits<std::uint32_t>::max()) break;
        }
      }
    }
    servicePayloadPrecache(now().seconds());
  }

  void observeUavTransmit(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      ChunkKey *chunk_key) {
    if (kTopicPolicies[topic_index].key == "chunk_stamps") {
      observeChunkStamps(sender, message);
      return;
    }
    if (kTopicPolicies[topic_index].key != "chunk_data") {
      return;
    }
    racer_fidelity_msgs::msg::ChunkData chunk;
    if (!deserialize(message, chunk) || chunk.chunk_drone_id < 1 ||
        chunk.chunk_drone_id > drone_count_ || chunk.idx == 0U) return;
    *chunk_key = {chunk.chunk_drone_id, chunk.idx};
    const std::size_t bytes = 64U + message->size();
    const bool new_definition = chunk_repository_.find(*chunk_key) ==
                                chunk_repository_.end();
    chunk_repository_.insert_or_assign(*chunk_key,
                                       cacheChunk(message, bytes, chunk));
    ++downlink_inventory_version_;
    if (new_definition) {
      // Chunk stamps can announce a key before its serialized payload arrives.
      // Seed deficits for every UAV already known to possess the chunk.  The
      // current sender is inserted below if its stamp was not observed first.
      cacheNewChunkDefinition(*chunk_key, bytes);
    }
    insertKnownUavChunk(sender, *chunk_key);
    if (task_metric_observer_mode_ == "inline") {
      // An ideal run represents the perfect global-map aggregator used only
      // to create the offline reference/GT diagnostics. Real BS runs update
      // this map exclusively after successful UAV-to-BS delivery below.
      applyTaskChunkObservation(
          chunk, true, mode_ == "ideal" && !rl_bs_scheduler_enabled_);
    } else if (task_metric_observer_mode_ == "async") {
      auto task_chunk =
          std::make_shared<racer_fidelity_msgs::msg::ChunkData>(
              std::move(chunk));
      enqueueTaskObservation(
          {TaskObservationKind::kUavChunk, sender, nullptr,
           std::move(task_chunk)});
    }
  }

  void observeTrajectory(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) {
    if (sender < 0 || sender >= drone_count_ ||
        kTopicPolicies.at(topic_index).key != "trajectory") {
      return;
    }
    if (task_metric_observer_mode_ == "inline") {
      applyTrajectoryObservation(sender, message);
    } else if (task_metric_observer_mode_ == "async") {
      enqueueTaskObservation(
          {TaskObservationKind::kTrajectory, sender, message, nullptr});
    }
  }

  void onLinkQuality(const LinkQualityArray &message) {
    for (const auto &link : message.links) {
      if (link.sender_id < 0 || link.sender_id >= radio_node_count_ ||
          link.receiver_id < 0 ||
          link.receiver_id >= radio_node_count_ ||
          link.sender_id == link.receiver_id) {
        continue;
      }
      links_[{link.sender_id, link.receiver_id}] = link;
      ++link_model_counts_[link.model];
    }
  }

  void onTransmit(int sender, std::size_t topic_index,
                  std::shared_ptr<rclcpp::SerializedMessage> message) {
    if (sender < 0 || sender >= drone_count_ ||
        topic_index >= kTopicPolicies.size()) {
      return;
    }
    const double stamp = now().seconds();
    const std::size_t bytes = shared_uav_radio_enabled_
                                  ? udp_header_bytes_ + message->size()
                                  : 64U + message->size();
    const auto &topic_key = kTopicPolicies.at(topic_index).key;
    ChunkKey chunk_key{};
    observeUavTransmit(sender, topic_index, message, &chunk_key);
    observeTrajectory(sender, topic_index, message);
    const bool has_chunk_key = chunk_key.owner > 0 && chunk_key.index > 0U;

    // MultiMapManager generated this ChunkData in response to a local BS
    // request.  It has now entered the Proxy through the normal UAV TX topic;
    // bind it to the originating RL action and enqueue the ordinary BS uplink
    // instead of broadcasting the internal to_drone_id==0 marker to peers.
    if (topic_key == "chunk_data" && has_chunk_key &&
        handleRequestedBsChunkResponse(sender, chunk_key, message)) {
      return;
    }
    if (topic_key == "chunk_data" && has_chunk_key) {
      // A normal RACER exchange may populate the global immutable payload
      // cache before the owner's local precache request completes.
      satisfyPayloadPrecache(chunk_key, false);
      servicePayloadPrecache(stamp);
    }

    if (maybeDeliverPerfectInitialAssignment(
            sender, topic_index, message, stamp)) {
      return;
    }

    if (ideal_direct_enabled_) {
      // Preserve source callback order: perfect mode forwards this complete
      // serialized logical message immediately, without queue or PER work.
      deliverIdealDirect(sender, topic_index, std::move(message), stamp,
                         has_chunk_key ? &chunk_key : nullptr);
      return;
    }

    auto flow = std::make_shared<DeliveryFlow>();
    flow->origin_sender = sender;
    flow->topic_index = topic_index;
    flow->born_at = stamp;
    flow->bytes = bytes;
    flow->delivered.assign(static_cast<std::size_t>(drone_count_), false);
    flow->delivered[static_cast<std::size_t>(sender)] = true;

    if (topic_key == "recovery_status") {
      ++logical_attempted_packets_;
      logical_attempted_bytes_ += bytes;
      if (ap_enabled_) {
        enqueue(sender, ap_node_id_, topic_index, message, flow,
                RouteStage::kApUplink, -1, stamp);
      }
      return;
    }
    if (topic_key == "recovery_command") return;

    const auto receivers = messageReceivers(sender, topic_index, message);
    const auto lossless_receivers =
        losslessEligible(topic_index)
            ? losslessNearestReceivers(sender, receivers)
            : std::vector<int>{};
    std::vector<bool> lossless_receiver_mask(
        static_cast<std::size_t>(drone_count_), false);
    for (const int receiver : lossless_receivers) {
      lossless_receiver_mask[static_cast<std::size_t>(receiver)] = true;
    }
    const auto intended_receivers =
        static_cast<std::uint64_t>(receivers.size());
    flow->intended.assign(static_cast<std::size_t>(drone_count_), false);
    for (const int receiver : receivers) {
      flow->intended[static_cast<std::size_t>(receiver)] = true;
    }
    if (bs_round_robin_enabled_ && topic_key != "chunk_data" &&
        bsPeerRelayEligible(topic_index)) {
      auto &cached =
          uav_latest_peer_messages_[static_cast<std::size_t>(sender)]
                                   [topic_index];
      cached.message = message;
      cached.flow = flow;
      cached.bytes = 64U + message->size();
      cached.born_at = stamp;
      cached.version = ++uav_peer_message_sequence_;
    }
    logical_attempted_packets_ += intended_receivers;
    logical_attempted_bytes_ += intended_receivers * bytes;
    if (shared_uav_radio_enabled_) {
      enqueueUavUdpDatagram(
          sender, topic_index, message, flow, receivers, lossless_receivers,
          stamp, has_chunk_key ? &chunk_key : nullptr);
      return;
    }
    for (const int receiver : receivers) {
      if (lossless_receiver_mask[static_cast<std::size_t>(receiver)]) {
        deliverLosslessNearest(
            sender, receiver, topic_index, message, flow, stamp,
            has_chunk_key ? &chunk_key : nullptr);
        continue;
      }
      enqueue(sender, receiver, topic_index, message, flow,
              RouteStage::kDirect, receiver, stamp,
              has_chunk_key ? &chunk_key : nullptr);
    }
    if (ap_enabled_ && !bs_round_robin_enabled_) {
      enqueue(sender, ap_node_id_, topic_index, message, flow,
              RouteStage::kApUplink, -1, stamp,
              has_chunk_key ? &chunk_key : nullptr);
    }
  }

  void deliverPerfectInitialAssignment(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      double born_at) {
    const double stamp = now().seconds();
    const std::size_t bytes = shared_uav_radio_enabled_
                                  ? udp_header_bytes_ + message->size()
                                  : 64U + message->size();
    auto flow = std::make_shared<DeliveryFlow>();
    flow->origin_sender = sender;
    flow->topic_index = topic_index;
    flow->born_at = born_at;
    flow->bytes = bytes;
    flow->delivered.assign(static_cast<std::size_t>(drone_count_), false);
    flow->delivered[static_cast<std::size_t>(sender)] = true;

    const auto receiver_count =
        static_cast<std::uint64_t>(std::max(0, drone_count_ - 1));
    logical_attempted_packets_ += receiver_count;
    logical_attempted_bytes_ += receiver_count * bytes;
    for (int receiver = 0; receiver < drone_count_; ++receiver) {
      if (receiver == sender) continue;
      ++attempted_packets_;
      attempted_bytes_ += bytes;
      ++direct_attempted_packets_;
      PendingPacket packet;
      packet.message = message;
      packet.flow = flow;
      packet.topic_index = topic_index;
      packet.route = RouteStage::kDirect;
      packet.final_receiver = receiver;
      packet.born_at = born_at;
      packet.enqueued_at = stamp;
      packet.delivery_at = stamp;
      packet.bytes = bytes;
      completeSuccessfulPacket({sender, receiver}, packet, stamp);
      ++initial_assignment_perfect_forwarded_packets_;
    }
  }

  void deliverPerfectInitialAssignmentViaBs(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      double born_at) {
    const double stamp = now().seconds();
    const std::size_t bytes = 64U + message->size();
    auto flow = std::make_shared<DeliveryFlow>();
    flow->origin_sender = sender;
    flow->topic_index = topic_index;
    flow->born_at = born_at;
    flow->bytes = bytes;
    flow->delivered.assign(static_cast<std::size_t>(drone_count_), false);
    flow->delivered[static_cast<std::size_t>(sender)] = true;
    flow->intended.assign(static_cast<std::size_t>(drone_count_), false);
    for (int receiver = 0; receiver < drone_count_; ++receiver) {
      if (receiver != sender) {
        flow->intended[static_cast<std::size_t>(receiver)] = true;
      }
    }

    std::uint64_t peer_message_version{};
    if (bsPeerRelayEligible(topic_index)) {
      auto &cached =
          uav_latest_peer_messages_[static_cast<std::size_t>(sender)]
                                   [topic_index];
      cached.message = message;
      cached.flow = flow;
      cached.bytes = bytes;
      cached.born_at = born_at;
      cached.version = ++uav_peer_message_sequence_;
      peer_message_version = cached.version;
    }

    const auto receiver_count =
        static_cast<std::uint64_t>(std::max(0, drone_count_ - 1));
    logical_attempted_packets_ += receiver_count;
    logical_attempted_bytes_ += receiver_count * bytes;

    // The bootstrap control plane is an explicit, reliable two-hop BS path.
    // It deliberately bypasses the learned scheduler and Sionna only until
    // every UAV acknowledges the selected initial assignment epoch.
    ++attempted_packets_;
    attempted_bytes_ += bytes;
    ++bs_uplink_attempted_packets_;
    PendingPacket uplink;
    uplink.message = message;
    uplink.flow = flow;
    uplink.topic_index = topic_index;
    uplink.route = RouteStage::kBsUplink;
    uplink.final_receiver = -1;
    uplink.born_at = born_at;
    uplink.enqueued_at = stamp;
    uplink.delivery_at = stamp;
    uplink.bytes = bytes;
    uplink.peer_message_version = peer_message_version;
    completeSuccessfulPacket({sender, ap_node_id_}, uplink, stamp);
    ++initial_assignment_perfect_bs_uplinks_;

    for (int receiver = 0; receiver < drone_count_; ++receiver) {
      if (receiver == sender) continue;
      ++attempted_packets_;
      attempted_bytes_ += bytes;
      ++bs_downlink_attempted_packets_;
      PendingPacket downlink;
      downlink.message = message;
      downlink.flow = flow;
      downlink.topic_index = topic_index;
      downlink.route = RouteStage::kBsDownlink;
      downlink.final_receiver = receiver;
      downlink.born_at = born_at;
      downlink.enqueued_at = stamp;
      downlink.delivery_at = stamp;
      downlink.bytes = bytes;
      completeSuccessfulPacket(
          {ap_node_id_, receiver}, downlink, stamp);
      // The bootstrap downlink is delivered immediately rather than queued as
      // a normal versioned relay, so account for its logical delivery here.
      ++logical_delivered_packets_;
      logical_delivered_bytes_ += bytes;
      cumulative_end_to_end_delay_s_ += stamp - born_at;
      ++initial_assignment_perfect_forwarded_packets_;
      ++initial_assignment_perfect_bs_downlinks_;
    }
  }

  bool maybeDeliverPerfectInitialAssignment(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      double born_at) {
    if (!initial_assignment_perfect_delivery_ ||
        initial_assignment_perfect_complete_) {
      return false;
    }
    const auto &topic_key = kTopicPolicies.at(topic_index).key;
    if (topic_key == "drone_state") {
      racer_fidelity_msgs::msg::DroneState state;
      if (!deserialize(message, state)) return false;
      ++initial_assignment_perfect_drone_state_messages_;
      if (initial_assignment_perfect_via_bs_) {
        deliverPerfectInitialAssignmentViaBs(
            sender, topic_index, message, born_at);
      } else {
        deliverPerfectInitialAssignment(sender, topic_index, message, born_at);
      }
      if (initial_assignment_perfect_epoch_ > 0U &&
          state.assignment_epoch >= initial_assignment_perfect_epoch_) {
        initial_assignment_epoch_acks_[static_cast<std::size_t>(sender)] =
            true;
        const auto ack_count = static_cast<int>(std::count(
            initial_assignment_epoch_acks_.begin(),
            initial_assignment_epoch_acks_.end(), true));
        if (ack_count == drone_count_) {
          initial_assignment_perfect_complete_ = true;
          initial_assignment_perfect_completed_at_s_ = now().seconds();
          RCLCPP_INFO(
              get_logger(),
              "Initial assignment perfect-delivery window closed: "
              "epoch=%lu acknowledgements=%d/%d; subsequent communication "
              "uses Sionna.",
              static_cast<unsigned long>(initial_assignment_perfect_epoch_),
              ack_count, drone_count_);
        }
      }
      return true;
    }
    if (topic_key == "global_assignment") {
      racer_fidelity_msgs::msg::GlobalGridAssignment assignment;
      if (!deserialize(message, assignment) || assignment.epoch == 0U) {
        return false;
      }
      if (initial_assignment_perfect_epoch_ == 0U) {
        initial_assignment_perfect_epoch_ = assignment.epoch;
        std::fill(initial_assignment_epoch_acks_.begin(),
                  initial_assignment_epoch_acks_.end(), false);
        RCLCPP_INFO(
            get_logger(),
            "Initial assignment perfect-delivery window selected epoch=%lu.",
            static_cast<unsigned long>(initial_assignment_perfect_epoch_));
      }
      if (assignment.epoch != initial_assignment_perfect_epoch_) {
        return false;
      }
      ++initial_assignment_perfect_global_assignment_messages_;
      if (initial_assignment_perfect_via_bs_) {
        deliverPerfectInitialAssignmentViaBs(
            sender, topic_index, message, born_at);
      } else {
        deliverPerfectInitialAssignment(sender, topic_index, message, born_at);
      }
      return true;
    }
    return false;
  }

  void deliverLosslessNearest(
      int sender, int receiver, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      const std::shared_ptr<DeliveryFlow> &flow, double born_at,
      const ChunkKey *chunk_key = nullptr) {
    const double stamp = now().seconds();
    const std::size_t bytes = 64U + (message ? message->size() : 0U);
    ++attempted_packets_;
    attempted_bytes_ += bytes;
    ++direct_attempted_packets_;
    PendingPacket packet;
    packet.message = message;
    packet.flow = flow;
    packet.topic_index = topic_index;
    packet.route = RouteStage::kDirect;
    packet.final_receiver = receiver;
    packet.born_at = born_at;
    packet.enqueued_at = stamp;
    packet.delivery_at = stamp;
    packet.bytes = bytes;
    if (chunk_key != nullptr) {
      packet.has_chunk_key = true;
      packet.chunk_key = *chunk_key;
    }
    completeSuccessfulPacket({sender, receiver}, packet, stamp);
    ++lossless_nearest_forwarded_packets_;
    if (lossless_communication_range_m_ > 0.0 &&
        uav_position_valid_[static_cast<std::size_t>(sender)] &&
        uav_position_valid_[static_cast<std::size_t>(receiver)]) {
      const auto &source = uav_positions_[static_cast<std::size_t>(sender)];
      const auto &target = uav_positions_[static_cast<std::size_t>(receiver)];
      const double dx = source[0] - target[0];
      const double dy = source[1] - target[1];
      const double dz = source[2] - target[2];
      if (dx * dx + dy * dy + dz * dz <=
          lossless_communication_range_m_ *
                  lossless_communication_range_m_ +
              1.0e-12) {
        ++lossless_range_forwarded_packets_;
      }
    }
  }

  void deliverIdealDirect(
      int sender, std::size_t topic_index,
      std::shared_ptr<rclcpp::SerializedMessage> message, double born_at,
      const ChunkKey *chunk_key = nullptr) {
    const auto &topic_key = kTopicPolicies.at(topic_index).key;
    // Preserve the existing recovery routing contract: status is consumed by
    // the AP only, and commands originate at the AP.
    if (topic_key == "recovery_status" || topic_key == "recovery_command") {
      return;
    }
    const auto profile_started = std::chrono::steady_clock::now();
    const double stamp = now().seconds();
    const std::size_t bytes = 64U + (message ? message->size() : 0U);
    const auto receivers = messageReceivers(sender, topic_index, message);
    ++ideal_logical_messages_;
    logical_attempted_packets_ += receivers.size();
    logical_attempted_bytes_ += receivers.size() * bytes;

    for (const int receiver : receivers) {
      ++attempted_packets_;
      attempted_bytes_ += bytes;
      ++direct_attempted_packets_;
      ++delivered_packets_;
      delivered_bytes_ += bytes;
      ++direct_delivered_packets_;
      ideal_statistical_transport_blocks_ +=
          model_.transportBlockCount(bytes);
      ideal_statistical_bytes_ += bytes;
      const auto receiver_index = static_cast<std::size_t>(receiver);
      if (chunk_key != nullptr &&
          uav_chunks_[receiver_index].find(*chunk_key) !=
              uav_chunks_[receiver_index].end()) {
        ++duplicates_suppressed_;
        ++ideal_direct_forwarded_packets_;
        continue;
      }
      publishers_[topic_index][receiver_index]->publish(*message);
      if (chunk_key != nullptr) insertKnownUavChunk(receiver, *chunk_key);
      ++logical_delivered_packets_;
      logical_delivered_bytes_ += bytes;
      cumulative_end_to_end_delay_s_ += stamp - born_at;
      ++direct_delivery_wins_;
      ++ideal_direct_forwarded_packets_;
    }
    const double elapsed_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - profile_started).count();
    ++perfect_forward_profile_calls_;
    perfect_forward_profile_receivers_ += receivers.size();
    perfect_forward_profile_total_ms_ += elapsed_ms;
    perfect_forward_profile_max_ms_ =
        std::max(perfect_forward_profile_max_ms_, elapsed_ms);
  }

  void markActiveLink(int sender, int receiver) {
    if (mode_ == "ideal" || sender < 0 || receiver < 0 ||
        sender >= radio_node_count_ || receiver >= radio_node_count_ ||
        sender == receiver) {
      return;
    }
    const LinkKey key{sender, receiver};
    if (active_links_.find(key) == active_links_.end()) {
      active_links_dirty_ = true;
    }
    // Link liveness is simulation state.  Wall-clock ageing would otherwise
    // expire links while Isaac is intentionally frozen for a PPO update.
    active_links_[key] = now().seconds();
  }

  void publishActiveLinks() {
    if (mode_ == "ideal" || !active_link_publisher_) return;
    if (rl_bs_synchronous_mode_ && rl_sync_boundary_frozen_) return;
    const double simulation_now = now().seconds();
    // The BS scheduler needs channel observability before it can select its
    // first upload or relay action.  Keep both directions of every UAV-BS
    // link in the same active set as data-driven UAV-UAV links; otherwise a
    // missing BS channel is masked by the policy, no BS packet is enqueued,
    // and enqueue() can never activate the link (a bootstrap deadlock).
    // Sionna canonicalizes reciprocal geometry, so these 2*N directed
    // requests add only N UAV-BS ray-tracing geometries.
    if (ap_enabled_) {
      for (int drone = 0; drone < drone_count_; ++drone) {
        for (const LinkKey key :
             {LinkKey{drone, ap_node_id_}, LinkKey{ap_node_id_, drone}}) {
          const auto [iterator, inserted] =
              active_links_.try_emplace(key, simulation_now);
          if (inserted) {
            active_links_dirty_ = true;
          } else {
            iterator->second = simulation_now;
          }
        }
      }
    }
    // A queued packet keeps its link active even if no new logical message
    // arrived during the hold interval.
    for (const auto &[key, queue] : queues_) {
      if (queue.packets.empty()) continue;
      if (active_links_.find(key) == active_links_.end()) {
        active_links_dirty_ = true;
      }
      active_links_[key] = simulation_now;
    }
    std::vector<LinkKey> requested_links;
    requested_links.reserve(active_links_.size());
    for (auto iterator = active_links_.begin();
         iterator != active_links_.end();) {
      const double age_s = std::max(0.0, simulation_now - iterator->second);
      if (age_s > active_link_hold_s_) {
        iterator = active_links_.erase(iterator);
        active_links_dirty_ = true;
        continue;
      }
      requested_links.push_back(iterator->first);
      ++iterator;
    }
    if (requested_links.empty()) return;
    // The 20 ms timer gives a newly active link low request latency. Once the
    // set is stable, a half-TTL keepalive is sufficient and avoids publishing
    // thousands of identical 90-link arrays during slow-wall-clock runs.
    const double keepalive_s = std::min(0.5, 0.5 * active_link_hold_s_);
    if (!active_links_dirty_ && have_active_link_publish_sim_stamp_ &&
        simulation_now - last_active_link_publish_sim_stamp_s_ < keepalive_s) {
      return;
    }
    LinkQualityArray request;
    request.stamp = now();
    request.links.reserve(requested_links.size());
    for (const auto &key : requested_links) {
      LinkQuality link;
      link.stamp = request.stamp;
      link.sender_id = key.sender;
      link.receiver_id = key.receiver;
      request.links.push_back(std::move(link));
    }
    active_link_publisher_->publish(request);
    active_links_dirty_ = false;
    have_active_link_publish_sim_stamp_ = true;
    last_active_link_publish_sim_stamp_s_ = simulation_now;
    ++active_link_publications_;
    active_link_samples_published_ += request.links.size();
  }

  std::vector<int> intendedReceivers(int sender) {
    std::vector<int> receivers;
    receivers.reserve(static_cast<std::size_t>(std::max(0, drone_count_ - 1)));
    for (int receiver = 0; receiver < drone_count_; ++receiver) {
      if (receiver != sender) receivers.push_back(receiver);
    }
    if (!nearest_neighbors_enabled_ && !distance_radius_enabled_) {
      return receivers;
    }
    if (!uav_position_valid_[static_cast<std::size_t>(sender)]) {
      position_unavailable_receivers_ += receivers.size();
      return {};
    }
    const auto &source = uav_positions_[static_cast<std::size_t>(sender)];
    receivers.erase(
        std::remove_if(receivers.begin(), receivers.end(),
                       [this](int receiver) {
                         return !uav_position_valid_[
                             static_cast<std::size_t>(receiver)];
                       }),
        receivers.end());
    if (distance_radius_enabled_) {
      const double range_squared = communication_range_m_ * communication_range_m_;
      const auto before_filter = receivers.size();
      receivers.erase(
          std::remove_if(receivers.begin(), receivers.end(),
                         [this, &source, range_squared](int receiver) {
                           const auto &target = uav_positions_[
                               static_cast<std::size_t>(receiver)];
                           const double dx = source[0] - target[0];
                           const double dy = source[1] - target[1];
                           const double dz = source[2] - target[2];
                           return dx * dx + dy * dy + dz * dz >
                                  range_squared;
                         }),
          receivers.end());
      range_filtered_receivers_ += before_filter - receivers.size();
      return receivers;
    }
    std::sort(receivers.begin(), receivers.end(),
              [this, &source](int left, int right) {
                const auto squared_distance = [&source](const auto &point) {
                  const double dx = source[0] - point[0];
                  const double dy = source[1] - point[1];
                  const double dz = source[2] - point[2];
                  return dx * dx + dy * dy + dz * dz;
                };
                const double left_distance = squared_distance(
                    uav_positions_[static_cast<std::size_t>(left)]);
                const double right_distance = squared_distance(
                    uav_positions_[static_cast<std::size_t>(right)]);
                if (left_distance != right_distance) {
                  return left_distance < right_distance;
                }
                return left < right;
              });
    if (receivers.size() > static_cast<std::size_t>(nearest_neighbor_count_)) {
      nearest_filtered_receivers_ +=
          receivers.size() - static_cast<std::size_t>(nearest_neighbor_count_);
      receivers.resize(static_cast<std::size_t>(nearest_neighbor_count_));
    }
    return receivers;
  }

  int directedReceiver(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) const {
    const bool unicast_enabled = shared_uav_radio_enabled_
                                     ? uav_udp_directed_unicast_
                                     : directed_message_unicast_;
    if (!unicast_enabled) return -1;
    const auto &key = kTopicPolicies.at(topic_index).key;
    int to_drone_id = 0;
    if (key == "pair_opt") {
      racer_fidelity_msgs::msg::PairOpt directed;
      if (!deserialize(message, directed)) return -2;
      to_drone_id = directed.to_drone_id;
    } else if (key == "pair_opt_res") {
      racer_fidelity_msgs::msg::PairOptResponse directed;
      if (!deserialize(message, directed)) return -2;
      to_drone_id = directed.to_drone_id;
    } else if (key == "chunk_data") {
      racer_fidelity_msgs::msg::ChunkData directed;
      if (!deserialize(message, directed)) return -2;
      // The separate oracle algorithm uses zero for an intentional fan-out.
      if (directed.to_drone_id == 0) return -1;
      to_drone_id = directed.to_drone_id;
    } else {
      return -1;
    }
    const int receiver = to_drone_id - 1;
    if (receiver < 0 || receiver >= drone_count_ || receiver == sender) {
      return -2;
    }
    return receiver;
  }

  std::vector<int> messageReceivers(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) {
    const int receiver = directedReceiver(sender, topic_index, message);
    if (receiver == -2) {
      ++invalid_directed_messages_dropped_;
      return {};
    }
    if (receiver >= 0) {
      ++directed_unicast_messages_;
      return {receiver};
    }
    return intendedReceivers(sender);
  }

  bool shouldLogUdpDatagram(std::uint64_t packet_id) const noexcept {
    return ofdma_diagnostic_logging_ &&
           packet_id % static_cast<std::uint64_t>(ofdma_log_packet_stride_) ==
               0U;
  }

  static std::size_t remainingBytes(const UdpDatagram &datagram) {
    return static_cast<std::size_t>(
        std::ceil(std::max(0.0, datagram.remaining_bits) / 8.0));
  }

  bool enqueueUavUdpDatagram(
      int sender, std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      const std::shared_ptr<DeliveryFlow> &flow,
      const std::vector<int> &receivers,
      const std::vector<int> &lossless_receivers, double born_at,
      const ChunkKey *chunk_key) {
    if (sender < 0 || sender >= drone_count_ || receivers.empty()) return false;
    const std::size_t bytes = udp_header_bytes_ + (message ? message->size() : 0U);
    auto &queue = uav_udp_queues_[static_cast<std::size_t>(sender)];
    const bool chunk_data =
        chunk_key != nullptr &&
        kTopicPolicies.at(topic_index).key == "chunk_data";
    const std::unordered_set<int> lossless_set(
        lossless_receivers.begin(), lossless_receivers.end());
    std::vector<UdpReceiverState> receiver_states;
    receiver_states.reserve(receivers.size());
    for (const int receiver : receivers) {
      if (receiver < 0 || receiver >= drone_count_ || receiver == sender) continue;
      if (chunk_data && chunk_data_pre_enqueue_dedup_ &&
          uav_chunks_[static_cast<std::size_t>(receiver)].find(*chunk_key) !=
              uav_chunks_[static_cast<std::size_t>(receiver)].end()) {
        ++chunk_enqueue_duplicates_suppressed_;
        continue;
      }
      if (chunk_data && chunk_data_max_pending_per_link_ > 0) {
        const auto pending = std::count_if(
            queue.datagrams.begin(), queue.datagrams.end(),
            [receiver](const UdpDatagram &datagram) {
              return datagram.has_chunk_key &&
                     std::any_of(
                         datagram.receivers.begin(), datagram.receivers.end(),
                         [receiver](const UdpReceiverState &state) {
                           return state.receiver == receiver;
                         });
            });
        if (pending >= chunk_data_max_pending_per_link_) {
          ++chunk_enqueue_budget_drops_;
          continue;
        }
      }
      receiver_states.push_back(
          {receiver, lossless_set.find(receiver) != lossless_set.end()});
    }
    if (receiver_states.empty()) return false;

    const auto receiver_count =
        static_cast<std::uint64_t>(receiver_states.size());
    attempted_packets_ += receiver_count;
    attempted_bytes_ += receiver_count * bytes;
    direct_attempted_packets_ += receiver_count;
    uav_udp_receiver_attempts_ += receiver_count;
    ++uav_udp_datagrams_enqueued_;
    uav_udp_physical_bytes_enqueued_ += bytes;
    if (receiver_states.size() == 1U) {
      ++uav_udp_unicast_datagrams_;
    } else {
      ++uav_udp_multicast_datagrams_;
    }

    const auto accountQueueDrop = [this](const UdpDatagram &datagram) {
      dropped_queue_ += datagram.receivers.size();
      ++uav_udp_datagrams_dropped_queue_;
    };
    if (bytes > queue_capacity_bytes_) {
      dropped_queue_ += receiver_count;
      ++uav_udp_datagrams_dropped_queue_;
      return false;
    }
    const int incoming_priority = packetPriority(RouteStage::kDirect, topic_index);
    while (queue.bytes + bytes > queue_capacity_bytes_) {
      auto candidate = queue.datagrams.end();
      for (auto iterator = queue.datagrams.end();
           iterator != queue.datagrams.begin();) {
        --iterator;
        if (!iterator->started &&
            packetPriority(RouteStage::kDirect, iterator->topic_index) <
                incoming_priority) {
          candidate = iterator;
          break;
        }
      }
      if (candidate == queue.datagrams.end()) {
        dropped_queue_ += receiver_count;
        ++uav_udp_datagrams_dropped_queue_;
        return false;
      }
      queue.bytes -= std::min(queue.bytes, candidate->bytes);
      accountQueueDrop(*candidate);
      queue.datagrams.erase(candidate);
    }

    UdpDatagram datagram;
    datagram.message = message;
    datagram.flow = flow;
    datagram.topic_index = topic_index;
    datagram.born_at = born_at;
    datagram.enqueued_at = now().seconds();
    datagram.bytes = bytes;
    datagram.total_bits = 8.0 * static_cast<double>(bytes);
    datagram.remaining_bits = datagram.total_bits;
    datagram.packet_id = ++uav_udp_next_packet_id_;
    datagram.receivers = std::move(receiver_states);
    if (chunk_key != nullptr) {
      datagram.has_chunk_key = true;
      datagram.chunk_key = *chunk_key;
    }
    for (const auto &receiver : datagram.receivers) {
      markActiveLink(sender, receiver.receiver);
    }

    auto position = queue.datagrams.begin();
    if (position != queue.datagrams.end() && position->started) ++position;
    while (position != queue.datagrams.end() &&
           packetPriority(RouteStage::kDirect, position->topic_index) >=
               incoming_priority) {
      ++position;
    }
    queue.datagrams.insert(position, std::move(datagram));
    queue.bytes += bytes;
    return true;
  }

  const LinkModel &directRadioModel() const {
    return ap_enabled_ ? uav_broadcast_model_ : model_;
  }

  SharedOfdmaScheduler &directOfdmaScheduler() {
    return ap_enabled_ ? uav_broadcast_ofdma_scheduler_ : ofdma_scheduler_;
  }

  bool usesBsRadio(RouteStage route) const {
    return route == RouteStage::kApUplink ||
           route == RouteStage::kApDownlink ||
           route == RouteStage::kBsControl ||
           route == RouteStage::kBsUplink ||
           route == RouteStage::kBsDownlink;
  }

  bool perfectBsRoute(RouteStage route) const {
    return force_bs_perfect_delivery_ && usesBsRadio(route);
  }

  const LinkModel &radioModel(RouteStage route) const {
    return usesBsRadio(route) ? bs_model_ : directRadioModel();
  }

  double radioSnrDb(const LinkQuality *link, const LinkModel &radio) const {
    if (!usable(link)) return -std::numeric_limits<double>::infinity();
    if (mode_ == "ideal") return snr(link);
    // LinkQuality.snr_db is referenced to the configured total 100 MHz.
    // Each assisted radio occupies only its own 50 MHz subband, so recompute
    // the receiver noise bandwidth without inventing additional TX power.
    return snr(link) +
           10.0 * std::log10(model_.config().bandwidth_hz /
                             radio.config().bandwidth_hz);
  }

  double allocatedSnrDb(const LinkQuality *link, int allocated_prbs) const {
    if (!usable(link)) return -std::numeric_limits<double>::infinity();
    // Sionna reports SNR using the configured full channel bandwidth. A UAV's
    // single configured TX-power budget is concentrated over its assigned
    // PRBs, never duplicated per receiver.
    const auto &radio = directRadioModel();
    return radioSnrDb(link, radio) + 10.0 * std::log10(
        static_cast<double>(radio.config().resource_blocks) /
        static_cast<double>(allocated_prbs));
  }

  double rateSelectionSnr(const UdpDatagram &datagram, int sender,
                          int allocated_prbs, double stamp) const {
    if (directRadioModel().config().fixed_mcs_index >= 0) return 0.0;
    double worst = std::numeric_limits<double>::infinity();
    for (const auto &receiver : datagram.receivers) {
      if (receiver.force_lossless) continue;
      const auto *link = linkFor({sender, receiver.receiver}, stamp);
      if (usable(link)) {
        worst = std::min(worst, allocatedSnrDb(link, allocated_prbs));
      }
    }
    return std::isfinite(worst) ? worst : -100.0;
  }

  bool udpExpired(const UdpDatagram &datagram, double stamp) const {
    const double ttl = kTopicPolicies[datagram.topic_index].ttl_s;
    return ttl > 0.0 && stamp - datagram.born_at > ttl;
  }

  void removeUavUdpFront(UavUdpQueue &queue) {
    if (queue.datagrams.empty()) return;
    queue.bytes -= std::min(queue.bytes, queue.datagrams.front().bytes);
    queue.datagrams.pop_front();
  }

  void expireUavUdpQueues(double stamp) {
    for (int sender = 0; sender < drone_count_; ++sender) {
      auto &queue = uav_udp_queues_[static_cast<std::size_t>(sender)];
      while (!queue.datagrams.empty() && udpExpired(queue.datagrams.front(), stamp)) {
        const auto &datagram = queue.datagrams.front();
        dropped_ttl_ += datagram.receivers.size();
        ++uav_udp_datagrams_dropped_ttl_;
        if (shouldLogUdpDatagram(datagram.packet_id)) {
          RCLCPP_INFO(
              get_logger(),
              "RACER_UAV_OFDMA_DROP slot=%lu sender=%d udp_type=%s "
              "packet_id=%lu packet_size=%zu allocated_prbs=0 "
              "remaining_bytes=%zu receiver=-1 snr_db=nan per=nan "
              "result=ttl_expired",
              static_cast<unsigned long>(uav_ofdma_slot_index_), sender,
              kTopicPolicies[datagram.topic_index].key.c_str(),
              static_cast<unsigned long>(datagram.packet_id), datagram.bytes,
              remainingBytes(datagram));
        }
        removeUavUdpFront(queue);
      }
    }
  }

  void finishUavPhysicalTransmission(int sender, UavUdpQueue &queue,
                                     double slot_end) {
    UdpDatagram completed = std::move(queue.datagrams.front());
    removeUavUdpFront(queue);
    const double random_jitter =
        jitter_s_ <= 0.0
            ? 0.0
            : std::uniform_real_distribution<double>(-jitter_s_, jitter_s_)(
                  uav_sender_rngs_[static_cast<std::size_t>(sender)]);
    completed.delivery_at =
        slot_end + std::max(0.0, base_latency_s_ + random_jitter);
    ++uav_physical_transmissions_completed_;
    uav_udp_physical_bytes_transmitted_ += completed.bytes;
    completed_uav_datagrams_.push_back(std::move(completed));
  }

  bool transmitUavDatagramSlot(
      const PrbAllocation &allocation, double slot_start,
      const std::vector<bool> &transmitting) {
    auto &queue =
        uav_udp_queues_[static_cast<std::size_t>(allocation.sender)];
    if (queue.datagrams.empty()) return false;
    auto &datagram = queue.datagrams.front();
    for (const auto &receiver : datagram.receivers) {
      markActiveLink(allocation.sender, receiver.receiver);
    }
    const double rate_snr = rateSelectionSnr(
        datagram, allocation.sender, allocation.prb_count, slot_start);
    const auto &radio = directRadioModel();
    const double available_bits = radio.bitsPerSlot(
        rate_snr, allocation.prb_count);
    const double transmitted_bits =
        std::min(datagram.remaining_bits, available_bits);
    if (!datagram.started) {
      datagram.started = true;
      datagram.first_slot = uav_ofdma_slot_index_;
      ++uav_physical_transmissions_started_;
      ++uav_tx_power_applications_;
      const std::string selected_mcs(radio.selectMcs(rate_snr).name);
      ++mcs_counts_[selected_mcs];
      ++uav_mcs_counts_[selected_mcs];
    }
    datagram.last_slot = uav_ofdma_slot_index_;
    datagram.last_allocated_prbs = allocation.prb_count;
    datagram.allocated_prb_slots +=
        static_cast<std::uint64_t>(allocation.prb_count);
    ++uav_ofdma_prb_allocation_events_;
    uav_ofdma_allocated_prb_slots_ += allocation.prb_count;
    direct_u2u_prb_slots_ += allocation.prb_count;

    for (auto &receiver : datagram.receivers) {
      receiver.last_allocated_prbs = allocation.prb_count;
      if (transmitting[static_cast<std::size_t>(receiver.receiver)]) {
        receiver.half_duplex_blocked = true;
        continue;
      }
      if (receiver.force_lossless) {
        receiver.received_bits += transmitted_bits;
        continue;
      }
      const auto *link =
          linkFor({allocation.sender, receiver.receiver}, slot_start);
      if (!usable(link)) {
        receiver.channel_unavailable = true;
        continue;
      }
      const double receiver_snr =
          allocatedSnrDb(link, allocation.prb_count);
      receiver.last_snr_db = receiver_snr;
      receiver.received_bits += transmitted_bits;
      receiver.snr_linear_bit_sum +=
          std::pow(10.0, receiver_snr / 10.0) * transmitted_bits;
    }
    datagram.remaining_bits =
        std::max(0.0, datagram.remaining_bits - transmitted_bits);
    if (shouldLogUdpDatagram(datagram.packet_id)) {
      RCLCPP_INFO(
          get_logger(),
          "%s slot=%lu sender=%d udp_type=%s packet_id=%lu "
          "packet_size=%zu prb_start=%d allocated_prbs=%d "
          "remaining_bytes=%zu receivers=%zu tx_power_dbm=%.3f",
          uav_csma_enabled_ ? "RACER_UAV_CSMA_SLOT"
                            : "RACER_UAV_OFDMA_SLOT",
          static_cast<unsigned long>(uav_ofdma_slot_index_),
          allocation.sender,
          kTopicPolicies[datagram.topic_index].key.c_str(),
          static_cast<unsigned long>(datagram.packet_id), datagram.bytes,
          allocation.first_prb, allocation.prb_count,
          remainingBytes(datagram), datagram.receivers.size(),
          uav_tx_power_dbm_);
    }
    if (datagram.remaining_bits > 1.0e-9) return false;
    finishUavPhysicalTransmission(
        allocation.sender, queue, slot_start + radio.slotDuration());
    return true;
  }

  void processUavOfdmaSlot(double slot_start) {
    expireUavUdpQueues(slot_start);
    std::vector<int> active_senders;
    active_senders.reserve(static_cast<std::size_t>(drone_count_));
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (!uav_udp_queues_[static_cast<std::size_t>(sender)].datagrams.empty()) {
        active_senders.push_back(sender);
      }
    }
    ++uav_ofdma_slots_processed_;
    if (active_senders.empty()) return;

    const auto allocations = directOfdmaScheduler().allocate(active_senders);
    if (allocations.empty()) return;
    ++uav_ofdma_active_slots_;
    uav_ofdma_peak_active_senders_ = std::max<std::uint64_t>(
        uav_ofdma_peak_active_senders_, allocations.size());
    std::vector<bool> transmitting(
        static_cast<std::size_t>(drone_count_), false);
    int slot_prbs = 0;
    for (const auto &allocation : allocations) {
      transmitting[static_cast<std::size_t>(allocation.sender)] = true;
      slot_prbs += allocation.prb_count;
    }
    uav_ofdma_max_allocated_prbs_per_slot_ = std::max(
        uav_ofdma_max_allocated_prbs_per_slot_, slot_prbs);

    for (const auto &allocation : allocations) {
      transmitUavDatagramSlot(allocation, slot_start, transmitting);
    }
  }

  void resetCsmaSender(int sender, bool collision) {
    const auto index = static_cast<std::size_t>(sender);
    if (collision) {
      const long long doubled =
          2LL * (static_cast<long long>(csma_contention_windows_[index]) + 1LL) -
          1LL;
      csma_contention_windows_[index] = static_cast<int>(
          std::min<long long>(csma_cw_max_, doubled));
    } else {
      csma_contention_windows_[index] = csma_cw_min_;
    }
    csma_backoff_slots_[index] = -1;
  }

  std::uint64_t dropCsmaCollisionDatagram(int sender, double slot_start) {
    auto &queue = uav_udp_queues_[static_cast<std::size_t>(sender)];
    if (queue.datagrams.empty()) return 1U;
    auto &datagram = queue.datagrams.front();
    for (const auto &receiver : datagram.receivers) {
      markActiveLink(sender, receiver.receiver);
    }
    const auto &radio = directRadioModel();
    const int prbs = radio.config().resource_blocks;
    const double rate_snr =
        rateSelectionSnr(datagram, sender, prbs, slot_start);
    const double bits_per_slot = std::max(1.0, radio.bitsPerSlot(rate_snr, prbs));
    const auto occupied_slots = static_cast<std::uint64_t>(std::max(
        1.0, std::ceil(datagram.remaining_bits / bits_per_slot)));
    datagram.started = true;
    datagram.first_slot = uav_ofdma_slot_index_;
    datagram.last_slot = uav_ofdma_slot_index_ + occupied_slots - 1U;
    datagram.last_allocated_prbs = prbs;
    datagram.allocated_prb_slots =
        occupied_slots * static_cast<std::uint64_t>(prbs);
    ++uav_physical_transmissions_started_;
    ++uav_physical_transmissions_completed_;
    ++uav_tx_power_applications_;
    uav_udp_physical_bytes_transmitted_ += datagram.bytes;
    const std::string selected_mcs(radio.selectMcs(rate_snr).name);
    ++mcs_counts_[selected_mcs];
    ++uav_mcs_counts_[selected_mcs];
    dropped_per_ += datagram.receivers.size();
    uav_udp_receiver_collision_failures_ += datagram.receivers.size();
    ++uav_csma_collided_datagrams_;
    if (shouldLogUdpDatagram(datagram.packet_id)) {
      RCLCPP_INFO(
          get_logger(),
          "RACER_UAV_CSMA_COLLISION slot=%lu sender=%d udp_type=%s "
          "packet_id=%lu packet_size=%zu contenders_collision=1 "
          "occupied_slots=%lu receivers=%zu result=collision",
          static_cast<unsigned long>(uav_ofdma_slot_index_), sender,
          kTopicPolicies[datagram.topic_index].key.c_str(),
          static_cast<unsigned long>(datagram.packet_id), datagram.bytes,
          static_cast<unsigned long>(occupied_slots),
          datagram.receivers.size());
    }
    removeUavUdpFront(queue);
    return occupied_slots;
  }

  void accountCsmaOccupiedSlot(std::size_t transmitting_senders) {
    const int prbs = directRadioModel().config().resource_blocks;
    ++uav_ofdma_active_slots_;
    ++uav_csma_busy_slots_;
    uav_ofdma_peak_active_senders_ = std::max<std::uint64_t>(
        uav_ofdma_peak_active_senders_, transmitting_senders);
    uav_ofdma_max_allocated_prbs_per_slot_ = std::max(
        uav_ofdma_max_allocated_prbs_per_slot_, prbs);
  }

  void processUavCsmaSlot(double slot_start) {
    expireUavUdpQueues(slot_start);
    ++uav_ofdma_slots_processed_;

    if (csma_collision_busy_slots_remaining_ > 0U) {
      accountCsmaOccupiedSlot(csma_last_collision_senders_);
      const int prbs = directRadioModel().config().resource_blocks;
      uav_ofdma_allocated_prb_slots_ += prbs;
      direct_u2u_prb_slots_ += prbs;
      --csma_collision_busy_slots_remaining_;
      return;
    }

    if (csma_transmitting_sender_ >= 0) {
      auto &queue = uav_udp_queues_[
          static_cast<std::size_t>(csma_transmitting_sender_)];
      if (queue.datagrams.empty() || !queue.datagrams.front().started) {
        resetCsmaSender(csma_transmitting_sender_, false);
        csma_transmitting_sender_ = -1;
        csma_idle_guard_slots_ = 0;
      }
    }

    if (csma_transmitting_sender_ >= 0) {
      const int sender = csma_transmitting_sender_;
      accountCsmaOccupiedSlot(1U);
      std::vector<bool> transmitting(
          static_cast<std::size_t>(drone_count_), false);
      transmitting[static_cast<std::size_t>(sender)] = true;
      const bool completed = transmitUavDatagramSlot(
          {sender, 0, directRadioModel().config().resource_blocks},
          slot_start, transmitting);
      if (completed) {
        resetCsmaSender(sender, false);
        csma_transmitting_sender_ = -1;
        csma_idle_guard_slots_ = 0;
      }
      return;
    }

    std::vector<int> active_senders;
    active_senders.reserve(static_cast<std::size_t>(drone_count_));
    for (int sender = 0; sender < drone_count_; ++sender) {
      const auto index = static_cast<std::size_t>(sender);
      if (uav_udp_queues_[index].datagrams.empty()) {
        resetCsmaSender(sender, false);
        continue;
      }
      active_senders.push_back(sender);
      if (csma_backoff_slots_[index] < 0) {
        csma_backoff_slots_[index] =
            std::uniform_int_distribution<int>(
                0, csma_contention_windows_[index])(
                uav_sender_rngs_[index]);
      }
    }
    if (active_senders.empty()) {
      csma_idle_guard_slots_ = 0;
      return;
    }

    if (csma_idle_guard_slots_ < csma_difs_slots_) {
      ++csma_idle_guard_slots_;
      ++uav_csma_idle_slots_;
      return;
    }

    std::vector<int> contenders;
    for (const int sender : active_senders) {
      if (csma_backoff_slots_[static_cast<std::size_t>(sender)] == 0) {
        contenders.push_back(sender);
      }
    }
    if (contenders.empty()) {
      for (const int sender : active_senders) {
        auto &backoff = csma_backoff_slots_[static_cast<std::size_t>(sender)];
        if (backoff > 0) --backoff;
      }
      ++uav_csma_idle_slots_;
      ++uav_csma_backoff_slots_;
      return;
    }

    csma_idle_guard_slots_ = 0;
    if (contenders.size() == 1U) {
      const int sender = contenders.front();
      csma_transmitting_sender_ = sender;
      ++uav_csma_successful_contentions_;
      accountCsmaOccupiedSlot(1U);
      std::vector<bool> transmitting(
          static_cast<std::size_t>(drone_count_), false);
      transmitting[static_cast<std::size_t>(sender)] = true;
      const bool completed = transmitUavDatagramSlot(
          {sender, 0, directRadioModel().config().resource_blocks},
          slot_start, transmitting);
      if (completed) {
        resetCsmaSender(sender, false);
        csma_transmitting_sender_ = -1;
      }
      return;
    }

    ++uav_csma_collision_events_;
    csma_last_collision_senders_ = contenders.size();
    accountCsmaOccupiedSlot(contenders.size());
    const int prbs = directRadioModel().config().resource_blocks;
    uav_ofdma_allocated_prb_slots_ += prbs;
    direct_u2u_prb_slots_ += prbs;
    std::uint64_t occupied_slots = 1U;
    for (const int sender : contenders) {
      ++uav_ofdma_prb_allocation_events_;
      occupied_slots = std::max(
          occupied_slots, dropCsmaCollisionDatagram(sender, slot_start));
      resetCsmaSender(sender, true);
    }
    csma_collision_busy_slots_remaining_ = occupied_slots - 1U;
  }

  void deliverCompletedUavDatagrams(double stamp) {
    for (auto iterator = completed_uav_datagrams_.begin();
         iterator != completed_uav_datagrams_.end();) {
      if (iterator->delivery_at > stamp + 1.0e-12) {
        ++iterator;
        continue;
      }
      UdpDatagram datagram = std::move(*iterator);
      iterator = completed_uav_datagrams_.erase(iterator);
      bool any_success = false;
      for (auto &receiver : datagram.receivers) {
        bool success = false;
        double receiver_snr = receiver.last_snr_db;
        double per = 1.0;
        const char *result = "per_failure";
        if (receiver.half_duplex_blocked) {
          result = "half_duplex_blocked";
          ++uav_udp_receiver_half_duplex_failures_;
        } else if (receiver.force_lossless) {
          success = true;
          per = 0.0;
          receiver_snr = std::numeric_limits<double>::infinity();
          result = "success_lossless_override";
          ++lossless_nearest_forwarded_packets_;
        } else if (receiver.channel_unavailable ||
                   receiver.received_bits + 1.0e-6 < datagram.total_bits) {
          result = "no_link";
          ++dropped_no_link_;
          ++uav_udp_receiver_no_link_failures_;
        } else {
          const double mean_linear_snr =
              receiver.snr_linear_bit_sum /
              std::max(1.0, receiver.received_bits);
          receiver_snr = 10.0 * std::log10(std::max(1.0e-30, mean_linear_snr));
          const auto &radio = directRadioModel();
          per = radio.packetErrorRate(receiver_snr, datagram.bytes);
          cumulative_initial_tbler_ +=
              radio.transportBlockErrorRate(receiver_snr);
          ++initial_tbler_samples_;
          success =
              std::uniform_real_distribution<double>(0.0, 1.0)(
                  linkRng({datagram.flow->origin_sender, receiver.receiver})) >=
              per;
          result = success ? "success" : "per_failure";
          if (!success) {
            ++dropped_per_;
            ++uav_udp_receiver_per_failures_;
          }
        }

        if (success) {
          any_success = true;
          ++delivered_packets_;
          delivered_bytes_ += datagram.bytes;
          cumulative_delay_s_ += stamp - datagram.enqueued_at;
          ++direct_delivered_packets_;
          ++uav_udp_receiver_successes_;
          PendingPacket delivered;
          delivered.message = datagram.message;
          delivered.flow = datagram.flow;
          delivered.topic_index = datagram.topic_index;
          delivered.route = RouteStage::kDirect;
          delivered.final_receiver = receiver.receiver;
          delivered.born_at = datagram.born_at;
          delivered.enqueued_at = datagram.enqueued_at;
          delivered.delivery_at = datagram.delivery_at;
          delivered.bytes = datagram.bytes;
          delivered.current_sender = datagram.flow->origin_sender;
          delivered.current_receiver = receiver.receiver;
          delivered.has_chunk_key = datagram.has_chunk_key;
          delivered.chunk_key = datagram.chunk_key;
          deliverToUav(delivered, receiver.receiver, stamp);
        }
        if (shouldLogUdpDatagram(datagram.packet_id)) {
          RCLCPP_INFO(
              get_logger(),
              "%s slot=%lu sender=%d udp_type=%s "
              "packet_id=%lu packet_size=%zu allocated_prbs=%d "
              "remaining_bytes=0 receiver=%d snr_db=%.3f per=%.6f "
              "result=%s",
              uav_csma_enabled_ ? "RACER_UAV_CSMA_RX"
                                : "RACER_UAV_OFDMA_RX",
              static_cast<unsigned long>(datagram.last_slot),
              datagram.flow->origin_sender,
              kTopicPolicies[datagram.topic_index].key.c_str(),
              static_cast<unsigned long>(datagram.packet_id), datagram.bytes,
              receiver.last_allocated_prbs, receiver.receiver, receiver_snr,
              per, result);
        }
      }
      if (any_success) ++uav_physical_transmissions_with_success_;
    }
  }

  void advanceUavOfdma(double stamp) {
    if (!uav_ofdma_timeline_initialized_) {
      uav_ofdma_timeline_initialized_ = true;
      uav_ofdma_next_slot_start_ = stamp;
    }
    const double slot = directRadioModel().slotDuration();
    while (uav_ofdma_next_slot_start_ + slot <= stamp + 1.0e-12) {
      if (uav_csma_enabled_) {
        processUavCsmaSlot(uav_ofdma_next_slot_start_);
      } else {
        processUavOfdmaSlot(uav_ofdma_next_slot_start_);
      }
      uav_ofdma_next_slot_start_ += slot;
      ++uav_ofdma_slot_index_;
    }
    deliverCompletedUavDatagrams(stamp);
  }

  bool losslessEligible(std::size_t topic_index) const {
    if (!lossless_control_only_) return true;
    const auto &key = kTopicPolicies.at(topic_index).key;
    return key == "drone_state" || key == "trajectory" ||
           key == "pair_opt" || key == "pair_opt_res" ||
           key == "global_assignment";
  }

  std::vector<int> losslessNearestReceivers(
      int sender, const std::vector<int> &candidates) {
    if ((lossless_nearest_neighbor_count_ <= 0 &&
         lossless_communication_range_m_ <= 0.0) || candidates.empty()) {
      return {};
    }
    if (!uav_position_valid_[static_cast<std::size_t>(sender)]) {
      ++lossless_nearest_position_unavailable_events_;
      return {};
    }
    const auto &source = uav_positions_[static_cast<std::size_t>(sender)];
    std::vector<int> receivers;
    receivers.reserve(static_cast<std::size_t>(drone_count_ - 1));
    for (int receiver = 0; receiver < drone_count_; ++receiver) {
      if (receiver == sender) continue;
      if (!uav_position_valid_[static_cast<std::size_t>(receiver)]) {
        ++lossless_nearest_position_unavailable_events_;
        return {};
      }
      receivers.push_back(receiver);
    }
    std::sort(receivers.begin(), receivers.end(),
              [this, &source](int left, int right) {
                const auto squared_distance = [&source](const auto &point) {
                  const double dx = source[0] - point[0];
                  const double dy = source[1] - point[1];
                  const double dz = source[2] - point[2];
                  return dx * dx + dy * dy + dz * dz;
                };
                const double left_distance = squared_distance(
                    uav_positions_[static_cast<std::size_t>(left)]);
                const double right_distance = squared_distance(
                    uav_positions_[static_cast<std::size_t>(right)]);
                if (left_distance != right_distance) {
                  return left_distance < right_distance;
                }
                return left < right;
              });
    std::unordered_set<int> selected;
    const auto nearest_limit = std::min(
        receivers.size(),
        static_cast<std::size_t>(lossless_nearest_neighbor_count_));
    for (std::size_t index = 0; index < nearest_limit; ++index) {
      selected.insert(receivers[index]);
    }
    if (lossless_communication_range_m_ > 0.0) {
      const double range_squared = lossless_communication_range_m_ *
                                   lossless_communication_range_m_;
      for (const int receiver : receivers) {
        const auto &target =
            uav_positions_[static_cast<std::size_t>(receiver)];
        const double dx = source[0] - target[0];
        const double dy = source[1] - target[1];
        const double dz = source[2] - target[2];
        if (dx * dx + dy * dy + dz * dz <= range_squared + 1.0e-12) {
          selected.insert(receiver);
        }
      }
    }
    receivers.erase(
        std::remove_if(receivers.begin(), receivers.end(),
                       [&candidates, &selected](int receiver) {
                         if (selected.find(receiver) == selected.end()) {
                           return true;
                         }
                         return std::find(candidates.begin(), candidates.end(),
                                          receiver) == candidates.end();
                       }),
        receivers.end());
    return receivers;
  }

  std::size_t topicIndex(const std::string &key) const {
    for (std::size_t index = 0; index < kTopicPolicies.size(); ++index) {
      if (kTopicPolicies[index].key == key) return index;
    }
    throw std::logic_error("missing communication topic policy: " + key);
  }

  PairControlMessage inspectPairControlMessage(
      std::size_t topic_index,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) const {
    PairControlMessage output;
    if (!message || topic_index >= kTopicPolicies.size()) return output;
    const auto &topic = kTopicPolicies[topic_index].key;
    if (topic == "pair_opt") {
      racer_fidelity_msgs::msg::PairOpt pair;
      if (!deserialize(message, pair)) return output;
      output.response = false;
      output.transaction_id = pair.transaction_id;
      output.sender = pair.from_drone_id - 1;
      output.receiver = pair.to_drone_id - 1;
      output.phase = pair.phase;
    } else if (topic == "pair_opt_res") {
      racer_fidelity_msgs::msg::PairOptResponse response;
      if (!deserialize(message, response)) return output;
      output.response = true;
      output.transaction_id = response.transaction_id;
      output.sender = response.from_drone_id - 1;
      output.receiver = response.to_drone_id - 1;
      output.status = response.status;
    } else {
      return output;
    }
    output.valid = output.transaction_id != 0U && output.sender >= 0 &&
                   output.sender < drone_count_ && output.receiver >= 0 &&
                   output.receiver < drone_count_ &&
                   output.sender != output.receiver;
    return output;
  }

  bool matchesPairControlReservation(
      const PairControlReservation &reservation,
      const PairControlMessage &message) const {
    if (!message.valid ||
        message.transaction_id != reservation.transaction_id) {
      return false;
    }
    if (message.response) {
      return message.sender == reservation.responder &&
             message.receiver == reservation.proposer;
    }
    return message.sender == reservation.proposer &&
           message.receiver == reservation.responder;
  }

  bool isExpectedReservedPairControl(
      const PairControlReservation &reservation,
      const PairControlMessage &message) const {
    if (!matchesPairControlReservation(reservation, message)) return false;
    if (reservation.stage == PairControlStage::kAwaitPrepared) {
      return message.response &&
             (message.status ==
                  racer_fidelity_msgs::msg::PairOptResponse::STATUS_PREPARED ||
              message.status ==
                  racer_fidelity_msgs::msg::PairOptResponse::STATUS_BUSY ||
              message.status ==
                  racer_fidelity_msgs::msg::PairOptResponse::STATUS_REJECTED ||
              message.status ==
                  racer_fidelity_msgs::msg::PairOptResponse::STATUS_COMMITTED);
    }
    if (reservation.stage == PairControlStage::kAwaitCommit) {
      return !message.response &&
             (message.phase ==
                  racer_fidelity_msgs::msg::PairOpt::PHASE_COMMIT ||
              message.phase ==
                  racer_fidelity_msgs::msg::PairOpt::PHASE_ABORT);
    }
    return message.response &&
           message.status ==
               racer_fidelity_msgs::msg::PairOptResponse::STATUS_COMMITTED;
  }

  void observePairControlDelivery(const PendingPacket &packet, int receiver,
                                  double stamp) {
    if (!pair_control_reservation_enabled_ || !packet.message) return;
    const auto message =
        inspectPairControlMessage(packet.topic_index, packet.message);
    if (!message.valid || message.receiver != receiver) return;

    auto found = pair_control_reservations_.find(message.transaction_id);
    if (found == pair_control_reservations_.end()) {
      if (packet.route != RouteStage::kBsDownlink || message.response ||
          message.phase != racer_fidelity_msgs::msg::PairOpt::PHASE_PROPOSE) {
        return;
      }
      PairControlReservation reservation;
      reservation.transaction_id = message.transaction_id;
      reservation.proposer = message.sender;
      reservation.responder = message.receiver;
      reservation.started_at = stamp;
      reservation.expires_at = stamp + pair_control_reservation_s_;
      pair_control_reservations_.emplace(message.transaction_id, reservation);
      ++pair_control_reservations_started_;
      RCLCPP_INFO(
          get_logger(),
          "RACER_PAIR_CONTROL_RESERVATION_START transaction=%llu pair=%d->%d "
          "start=%.6f deadline=%.6f",
          static_cast<unsigned long long>(message.transaction_id),
          message.sender + 1, message.receiver + 1, stamp,
          reservation.expires_at);
      return;
    }

    auto &reservation = found->second;
    if (!matchesPairControlReservation(reservation, message)) return;
    if (!message.response) {
      if (message.phase == racer_fidelity_msgs::msg::PairOpt::PHASE_COMMIT) {
        if (reservation.stage != PairControlStage::kAwaitCommitted) {
          ++pair_control_reserved_commits_delivered_;
        }
        reservation.stage = PairControlStage::kAwaitCommitted;
      } else if (message.phase ==
                 racer_fidelity_msgs::msg::PairOpt::PHASE_ABORT) {
        ++pair_control_reservations_terminated_;
        pair_control_reservations_.erase(found);
      }
      return;
    }

    if (message.status ==
        racer_fidelity_msgs::msg::PairOptResponse::STATUS_PREPARED) {
      if (reservation.stage == PairControlStage::kAwaitPrepared) {
        reservation.stage = PairControlStage::kAwaitCommit;
        ++pair_control_reserved_prepared_delivered_;
      }
      return;
    }
    if (message.status ==
        racer_fidelity_msgs::msg::PairOptResponse::STATUS_COMMITTED) {
      ++pair_control_reservations_completed_;
      RCLCPP_INFO(
          get_logger(),
          "RACER_PAIR_CONTROL_RESERVATION_COMPLETE transaction=%llu "
          "pair=%d->%d elapsed=%.6f",
          static_cast<unsigned long long>(message.transaction_id),
          reservation.proposer + 1, reservation.responder + 1,
          stamp - reservation.started_at);
      pair_control_reservations_.erase(found);
      return;
    }
    if (reservation.stage == PairControlStage::kAwaitPrepared &&
        (message.status ==
             racer_fidelity_msgs::msg::PairOptResponse::STATUS_BUSY ||
         message.status ==
             racer_fidelity_msgs::msg::PairOptResponse::STATUS_REJECTED)) {
      ++pair_control_reservations_terminated_;
      pair_control_reservations_.erase(found);
    }
  }

  void scheduleReservedPairControl(double stamp) {
    if (!pair_control_reservation_enabled_) return;
    for (auto iterator = pair_control_reservations_.begin();
         iterator != pair_control_reservations_.end();) {
      if (stamp > iterator->second.expires_at + 1.0e-9) {
        RCLCPP_WARN(
            get_logger(),
            "RACER_PAIR_CONTROL_RESERVATION_EXPIRED transaction=%llu "
            "pair=%d->%d elapsed=%.6f",
            static_cast<unsigned long long>(iterator->first),
            iterator->second.proposer + 1,
            iterator->second.responder + 1,
            stamp - iterator->second.started_at);
        ++pair_control_reservations_expired_;
        iterator = pair_control_reservations_.erase(iterator);
      } else {
        ++iterator;
      }
    }

    const auto pair_opt_topic = topicIndex("pair_opt");
    const auto pair_response_topic = topicIndex("pair_opt_res");
    for (const auto &[transaction_id, reservation] :
         pair_control_reservations_) {
      (void)transaction_id;
      const int sender = reservation.stage == PairControlStage::kAwaitCommit
                             ? reservation.proposer
                             : reservation.responder;
      const std::size_t topic =
          reservation.stage == PairControlStage::kAwaitCommit
              ? pair_opt_topic
              : pair_response_topic;
      const auto sender_index = static_cast<std::size_t>(sender);
      const auto &source = uav_latest_peer_messages_[sender_index][topic];
      const auto source_message =
          inspectPairControlMessage(topic, source.message);
      if (source.message &&
          isExpectedReservedPairControl(reservation, source_message)) {
        const LinkKey uplink{sender, ap_node_id_};
        const auto &at_bs = bs_latest_peer_messages_[sender_index][topic];
        if (source.version > at_bs.version &&
            !hasPendingPeerMessage(uplink, RouteStage::kBsUplink, source) &&
            enqueue(sender, ap_node_id_, topic, source.message, source.flow,
                    RouteStage::kBsUplink, -1, source.born_at, nullptr,
                    source.bytes, source.version, true)) {
          ++pair_control_reserved_uplinks_scheduled_;
        }
      }

      const auto &at_bs = bs_latest_peer_messages_[sender_index][topic];
      const auto bs_message =
          inspectPairControlMessage(topic, at_bs.message);
      if (!at_bs.message || !at_bs.flow ||
          !isExpectedReservedPairControl(reservation, bs_message)) {
        continue;
      }
      const int receiver = bs_message.receiver;
      const auto receiver_index = static_cast<std::size_t>(receiver);
      if (at_bs.flow->intended.size() !=
              static_cast<std::size_t>(drone_count_) ||
          !at_bs.flow->intended[receiver_index] ||
          at_bs.flow->delivered[receiver_index]) {
        continue;
      }
      const LinkKey downlink{ap_node_id_, receiver};
      if (!hasPendingPeerMessage(
              downlink, RouteStage::kBsDownlink, at_bs) &&
          enqueue(ap_node_id_, receiver, topic, at_bs.message, at_bs.flow,
                  RouteStage::kBsDownlink, receiver, at_bs.born_at, nullptr,
                  at_bs.bytes, at_bs.version, true)) {
        ++pair_control_reserved_downlinks_scheduled_;
      }
    }
  }

  void onApRecoveryCommand(
      std::size_t topic_index,
      std::shared_ptr<rclcpp::SerializedMessage> message) {
    if (!ap_enabled_ || !message) return;
    racer_recovery_core::msg::RecoveryCommand command;
    if (!deserialize(message, command) || command.drone_id < 1 ||
        command.drone_id > drone_count_) {
      RCLCPP_WARN(get_logger(), "drop malformed AP recovery command");
      return;
    }
    const int receiver = command.drone_id - 1;
    const double stamp = now().seconds();
    const std::size_t bytes = 64U + message->size();
    auto flow = std::make_shared<DeliveryFlow>();
    flow->origin_sender = ap_node_id_;
    flow->topic_index = topic_index;
    flow->born_at = stamp;
    flow->bytes = bytes;
    flow->delivered.assign(static_cast<std::size_t>(drone_count_), false);
    ++logical_attempted_packets_;
    logical_attempted_bytes_ += bytes;
    enqueue(ap_node_id_, receiver, topic_index, std::move(message), flow,
            RouteStage::kApDownlink, receiver, stamp);
  }

  int packetPriority(RouteStage route, std::size_t topic_index,
                     bool reserved_pair_control = false) const {
    if (route == RouteStage::kBsControl) return 100;
    if (reserved_pair_control) return 50;
    return kTopicPolicies.at(topic_index).priority;
  }

  bool enqueue(int sender, int receiver, std::size_t topic_index,
               const std::shared_ptr<rclcpp::SerializedMessage> &message,
               const std::shared_ptr<DeliveryFlow> &flow, RouteStage route,
               int final_receiver, double born_at,
               const ChunkKey *chunk_key = nullptr,
               std::size_t override_bytes = 0U,
               std::uint64_t peer_message_version = 0U,
               bool reserved_pair_control = false,
               int action_sender = -1,
               std::uint64_t action_id = 0U,
               std::uint64_t action_step_id = 0U,
               const std::shared_ptr<const ActionChannelSnapshot>
                   &action_channel_snapshot = nullptr,
               bool periodic_bs_upload_request = false) {
    markActiveLink(sender, receiver);
    const double stamp = now().seconds();
    const std::size_t bytes = override_bytes > 0U
                                  ? override_bytes
                                  : 64U + (message ? message->size() : 0U);
    const LinkKey key{sender, receiver};
    // In the one-shot perfect-BS experiment, an RL-selected BS route is a
    // lossless transport service.  Keep its finite serialization time and PRB
    // accounting, but do not turn queue capacity into a second packet-loss
    // mechanism after the action has selected the transfer.
    const bool lossless_bs_transport = perfectBsRoute(route);
    auto found = queues_.find(key);
    if (found == queues_.end()) {
      ++dropped_queue_;
      return false;
    }
    auto &queue = found->second;
    const bool chunk_data =
        chunk_key != nullptr &&
        kTopicPolicies.at(topic_index).key == "chunk_data";
    if (chunk_data) {
      const bool receiver_has_chunk =
          receiver >= 0 && receiver < drone_count_ &&
          uav_chunks_[static_cast<std::size_t>(receiver)].find(*chunk_key) !=
              uav_chunks_[static_cast<std::size_t>(receiver)].end();
      const bool already_pending = std::any_of(
          queue.packets.begin(), queue.packets.end(),
          [chunk_key](const PendingPacket &packet) {
            return packet.has_chunk_key && packet.chunk_key == *chunk_key;
          });
      if (chunk_data_pre_enqueue_dedup_ &&
          (receiver_has_chunk || already_pending)) {
        ++chunk_enqueue_duplicates_suppressed_;
        return false;
      }
      if (chunk_data_max_pending_per_link_ > 0) {
        const auto pending_chunks = std::count_if(
            queue.packets.begin(), queue.packets.end(),
            [](const PendingPacket &packet) {
              return packet.has_chunk_key &&
                     kTopicPolicies.at(packet.topic_index).key == "chunk_data";
            });
        if (pending_chunks >= chunk_data_max_pending_per_link_) {
          ++chunk_enqueue_budget_drops_;
          return false;
        }
      }
    }
    ++attempted_packets_;
    attempted_bytes_ += bytes;
    if (route == RouteStage::kDirect) {
      ++direct_attempted_packets_;
    } else if (route == RouteStage::kApUplink) {
      ++ap_uplink_attempted_packets_;
    } else if (route == RouteStage::kApDownlink) {
      ++ap_downlink_attempted_packets_;
    } else if (route == RouteStage::kBsControl) {
      ++bs_control_attempted_packets_;
    } else if (route == RouteStage::kBsUplink) {
      ++bs_uplink_attempted_packets_;
    } else {
      ++bs_downlink_attempted_packets_;
    }
    if (!lossless_bs_transport && bytes > queue_capacity_bytes_) {
      ++dropped_queue_;
      return false;
    }

    const int incoming_priority =
        packetPriority(route, topic_index, reserved_pair_control);
    while (!lossless_bs_transport &&
           queue.bytes + bytes > queue_capacity_bytes_) {
      auto candidate = queue.packets.end();
      for (auto iterator = queue.packets.end();
           iterator != queue.packets.begin();) {
        --iterator;
        if (!iterator->transmitting &&
            packetPriority(iterator->route, iterator->topic_index,
                           iterator->reserved_pair_control) <
                incoming_priority) {
          candidate = iterator;
          break;
        }
      }
      if (candidate == queue.packets.end()) {
        ++dropped_queue_;
        return false;
      }
      updateRouteQueueCache(*candidate, false);
      queue.bytes -= std::min(queue.bytes, candidate->bytes);
      queue.packets.erase(candidate);
      ++dropped_queue_;
    }

    PendingPacket pending;
    pending.message = message;
    pending.flow = flow;
    pending.topic_index = topic_index;
    pending.route = route;
    pending.final_receiver = final_receiver;
    pending.born_at = born_at;
    pending.enqueued_at = stamp;
    pending.bytes = bytes;
    pending.reserved_pair_control = reserved_pair_control;
    pending.periodic_bs_upload_request = periodic_bs_upload_request;
    pending.current_sender = sender;
    pending.current_receiver = receiver;
    pending.action_sender = action_sender;
    pending.action_id = action_id;
    pending.action_step_id = action_step_id;
    pending.channel_version = action_channel_snapshot
                                  ? action_channel_snapshot->channel_version
                                  : 0U;
    pending.action_channel_snapshot = action_channel_snapshot;
    if (chunk_key != nullptr) {
      pending.has_chunk_key = true;
      pending.chunk_key = *chunk_key;
    }
    pending.peer_message_version = peer_message_version;
    updateRouteQueueCache(pending, true);
    // Transmission priority is chosen globally when BS airtime is assigned.
    // Keeping ownership FIFO here avoids treating already-serialized packets
    // as a head-of-line barrier while their propagation delay is still in
    // flight.
    queue.packets.push_back(std::move(pending));
    queue.bytes += bytes;
    if (action_channel_snapshot) ++rl_action_bound_packets_enqueued_;
    return true;
  }

  std::size_t chunkDataTopicIndex() const {
    for (std::size_t index = 0; index < kTopicPolicies.size(); ++index) {
      if (kTopicPolicies[index].key == "chunk_data") return index;
    }
    throw std::logic_error("chunk_data topic policy is missing");
  }

  bool bsPeerRelayEligible(std::size_t topic_index) const {
    const auto &key = kTopicPolicies.at(topic_index).key;
    if (key == "drone_state" || key == "trajectory" ||
        key == "global_assignment") {
      return true;
    }
    // Retain the old reservation experiment only behind its explicit legacy
    // switch.  The current BS-event training mode keeps this switch false, so
    // PairOpt/PairOptResponse remain exclusively on the direct UAV channel.
    return pair_control_reservation_enabled_ &&
           (key == "pair_opt" || key == "pair_opt_res");
  }

  std::vector<ChunkKey> sortedChunks(
      const std::unordered_set<ChunkKey, ChunkKeyHash> &available,
      const std::unordered_set<ChunkKey, ChunkKeyHash> &excluded,
      bool require_cached_payload = true) const {
    std::vector<ChunkKey> output;
    output.reserve(available.size());
    for (const auto &key : available) {
      if (excluded.find(key) == excluded.end() &&
          (!require_cached_payload ||
           chunk_repository_.find(key) != chunk_repository_.end())) {
        output.push_back(key);
      }
    }
    std::sort(output.begin(), output.end());
    return output;
  }

  const std::vector<ChunkKey> &rankedDownlinkChunks(int receiver) {
    const auto index = static_cast<std::size_t>(receiver);
    const bool positioned = uav_position_valid_[index];
    const auto position = positioned ? uav_positions_[index]
                                     : std::array<double, 3>{};
    if (downlink_priority_versions_[index] == downlink_inventory_version_ &&
        downlink_priority_position_valid_[index] == positioned &&
        (!positioned || downlink_priority_positions_[index] == position)) {
      return downlink_priority_tables_[index];
    }

    // A chunk is indivisible. Its score is the mean value of its voxels,
    // value(v) = -||v - UAV||^2. Expanding the square lets each UAV's table
    // be refreshed from one cached centroid and second moment per chunk.
    struct RankedChunk {
      bool newest{};
      double mean_distance_squared{};
      ChunkKey key{};
    };
    ChunkKey newest_key{};
    std::uint64_t newest_sequence{};
    bool have_newest = false;
    if (bs_downlink_newest_missing_chunk_first_) {
      for (const auto &key : ordered_uav_missing_chunks_[index]) {
        const auto sequence = bs_chunk_arrival_sequences_.find(key);
        if (sequence != bs_chunk_arrival_sequences_.end() &&
            (!have_newest || sequence->second > newest_sequence)) {
          newest_key = key;
          newest_sequence = sequence->second;
          have_newest = true;
        }
      }
    }
    std::vector<RankedChunk> ranked;
    ranked.reserve(ordered_uav_missing_chunks_[index].size());
    for (const auto &key : ordered_uav_missing_chunks_[index]) {
      double mean_distance_squared = std::numeric_limits<double>::infinity();
      const auto found = chunk_repository_.find(key);
      if (positioned && found != chunk_repository_.end() &&
          found->second.has_voxel_geometry) {
        const auto &chunk = found->second;
        mean_distance_squared = chunk.mean_voxel_norm_squared;
        for (std::size_t axis = 0; axis < 3U; ++axis) {
          mean_distance_squared += position[axis] * position[axis] -
              2.0 * position[axis] * chunk.voxel_center[axis];
        }
      }
      ranked.push_back(
          {have_newest && key == newest_key, mean_distance_squared, key});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &left, const auto &right) {
                if (left.newest != right.newest) return left.newest;
                if (left.mean_distance_squared !=
                    right.mean_distance_squared) {
                  return left.mean_distance_squared <
                      right.mean_distance_squared;
                }
                return left.key < right.key;
              });
    auto &table = downlink_priority_tables_[index];
    table.clear();
    table.reserve(ranked.size());
    for (const auto &entry : ranked) table.push_back(entry.key);
    downlink_priority_versions_[index] = downlink_inventory_version_;
    downlink_priority_position_valid_[index] = positioned;
    downlink_priority_positions_[index] = position;
    return table;
  }

  std::shared_ptr<rclcpp::SerializedMessage> retargetChunkMessage(
      const std::shared_ptr<rclcpp::SerializedMessage> &message,
      int current_sender, int current_receiver) const {
    racer_fidelity_msgs::msg::ChunkData chunk;
    if (!message || !deserialize(message, chunk)) return nullptr;
    // ChunkData uses one-based node identifiers.  The BS is represented by
    // ap_node_id_ + 1, while chunk_drone_id and idx remain the immutable
    // origin owner and chunk version respectively.
    chunk.from_drone_id = current_sender + 1;
    chunk.to_drone_id = current_receiver + 1;
    auto output = std::make_shared<rclcpp::SerializedMessage>();
    rclcpp::Serialization<racer_fidelity_msgs::msg::ChunkData> serializer;
    serializer.serialize_message(&chunk, output.get());
    return output;
  }

  bool hasPendingChunk(const LinkKey &link, const ChunkKey &chunk,
                       RouteStage route) const {
    const auto found = queues_.find(link);
    if (found == queues_.end()) return false;
    return std::any_of(
        found->second.packets.begin(), found->second.packets.end(),
        [&chunk, route](const PendingPacket &packet) {
          return packet.route == route && packet.has_chunk_key &&
                 packet.chunk_key == chunk;
        });
  }

  bool hasPendingBsUplink(const ChunkKey &chunk) const {
    for (const auto &[link, queue] : queues_) {
      if (link.receiver != ap_node_id_) continue;
      if (std::any_of(
              queue.packets.begin(), queue.packets.end(),
              [&chunk](const PendingPacket &packet) {
                return packet.route == RouteStage::kBsUplink &&
                       packet.has_chunk_key && packet.chunk_key == chunk;
              })) {
        return true;
      }
    }
    return false;
  }

  bool payloadPrecacheNeeded(const ChunkKey &key) const {
    if (!bs_payload_precache_enabled_ || !bs_round_robin_enabled_ ||
        key.owner < 1 || key.owner > drone_count_ || key.index == 0U ||
        chunk_repository_.count(key) || bs_chunks_.count(key)) {
      return false;
    }
    return uav_chunks_[static_cast<std::size_t>(key.owner - 1)].count(key) !=
           0U;
  }

  void queuePayloadPrecache(const ChunkKey &key) {
    if (!payloadPrecacheNeeded(key) ||
        pending_payload_precache_keys_.count(key) ||
        inflight_payload_precache_.count(key)) {
      return;
    }
    pending_payload_precache_.push_back(key);
    pending_payload_precache_keys_.insert(key);
    payload_precache_pending_peak_ = std::max(
        payload_precache_pending_peak_, pending_payload_precache_keys_.size());
    ++payload_precache_queued_;
  }

  void satisfyPayloadPrecache(const ChunkKey &key, bool requested_response) {
    const bool was_pending = pending_payload_precache_keys_.erase(key) > 0U;
    const bool was_inflight = inflight_payload_precache_.erase(key) > 0U;
    if (!was_pending && !was_inflight) return;
    if (requested_response) {
      ++payload_precache_responses_received_;
      if (was_pending && !was_inflight) {
        ++payload_precache_late_responses_;
      }
    } else {
      ++payload_precache_satisfied_by_peer_data_;
    }
  }

  void compactPayloadPrecacheQueueIfNeeded() {
    if (pending_payload_precache_.size() <=
        2U * pending_payload_precache_keys_.size() + 1024U) {
      return;
    }
    std::deque<ChunkKey> compacted;
    for (const auto &key : pending_payload_precache_) {
      if (pending_payload_precache_keys_.count(key)) {
        compacted.push_back(key);
      }
    }
    pending_payload_precache_.swap(compacted);
  }

  void pruneExpiredPayloadPrecaches(double stamp) {
    std::vector<ChunkKey> retry;
    for (auto iterator = inflight_payload_precache_.begin();
         iterator != inflight_payload_precache_.end();) {
      if (stamp - iterator->second.requested_at <=
          bs_payload_precache_timeout_s_) {
        ++iterator;
        continue;
      }
      retry.push_back(iterator->first);
      iterator = inflight_payload_precache_.erase(iterator);
      ++payload_precache_timeouts_;
    }
    for (const auto &key : retry) {
      if (payloadPrecacheNeeded(key) &&
          pending_payload_precache_keys_.insert(key).second) {
        pending_payload_precache_.push_back(key);
      }
    }
  }

  void servicePayloadPrecache(double stamp) {
    if (!bs_payload_precache_enabled_ || !bs_round_robin_enabled_) return;
    pruneExpiredPayloadPrecaches(stamp);
    compactPayloadPrecacheQueueIfNeeded();
    const auto inflight_limit =
        static_cast<std::size_t>(bs_max_payload_precache_inflight_);
    const auto batch_limit =
        static_cast<std::size_t>(bs_payload_precache_batch_size_);
    while (inflight_payload_precache_.size() < inflight_limit) {
      while (!pending_payload_precache_.empty()) {
        const auto &front = pending_payload_precache_.front();
        if (!pending_payload_precache_keys_.count(front)) {
          pending_payload_precache_.pop_front();
          continue;
        }
        if (!payloadPrecacheNeeded(front)) {
          pending_payload_precache_keys_.erase(front);
          pending_payload_precache_.pop_front();
          continue;
        }
        break;
      }
      if (pending_payload_precache_.empty()) break;

      const int sender = pending_payload_precache_.front().owner - 1;
      std::vector<ChunkKey> batch;
      const auto request_limit = std::min(
          batch_limit, inflight_limit - inflight_payload_precache_.size());
      while (!pending_payload_precache_.empty() &&
             batch.size() < request_limit) {
        const auto key = pending_payload_precache_.front();
        if (!pending_payload_precache_keys_.count(key)) {
          pending_payload_precache_.pop_front();
          continue;
        }
        if (!payloadPrecacheNeeded(key)) {
          pending_payload_precache_keys_.erase(key);
          pending_payload_precache_.pop_front();
          continue;
        }
        // One ChunkStamps request targets one UAV. Preserve FIFO order rather
        // than overtaking another owner's newly sealed chunks.
        if (key.owner - 1 != sender) break;
        pending_payload_precache_.pop_front();
        pending_payload_precache_keys_.erase(key);
        inflight_payload_precache_.emplace(
            key, InflightPayloadPrecache{sender, stamp});
        batch.push_back(key);
      }
      if (batch.empty()) break;
      if (!publishBsChunkRequest(sender, batch, stamp, true)) {
        for (auto iterator = batch.rbegin(); iterator != batch.rend();
             ++iterator) {
          inflight_payload_precache_.erase(*iterator);
          pending_payload_precache_keys_.insert(*iterator);
          pending_payload_precache_.push_front(*iterator);
        }
        break;
      }
      payload_precache_inflight_peak_ = std::max(
          payload_precache_inflight_peak_, inflight_payload_precache_.size());
    }
  }

  bool hasPendingBsChunkRequest(const ChunkKey &chunk) const {
    if (pending_payload_precache_keys_.count(chunk) ||
        inflight_payload_precache_.count(chunk)) {
      return true;
    }
    return std::any_of(
        pending_bs_chunk_requests_.begin(),
        pending_bs_chunk_requests_.end(),
        [&chunk](const auto &requests) {
          return requests.find(chunk) != requests.end();
        });
  }

  std::size_t pendingBsUplinkChunkCount(int sender) const {
    if (sender < 0 || sender >= drone_count_) return 0U;
    const LinkKey link{sender, ap_node_id_};
    const auto found = queues_.find(link);
    if (found == queues_.end()) return 0U;
    return static_cast<std::size_t>(std::count_if(
        found->second.packets.begin(), found->second.packets.end(),
        [](const PendingPacket &packet) {
          return packet.route == RouteStage::kBsUplink &&
                 packet.has_chunk_key;
        }));
  }

  std::size_t bsUploadInflightChunkCount(int sender) const {
    if (sender < 0 || sender >= drone_count_) return 0U;
    return pending_bs_chunk_requests_[static_cast<std::size_t>(sender)].size() +
           pendingBsUplinkChunkCount(sender);
  }

  void pruneExpiredBsChunkRequests(double stamp) {
    // The request is an in-process ROS handoff to MultiMapManager, so a
    // response normally arrives immediately.  Expiry prevents a stopped UAV
    // process from reserving the same owner/index forever; a late response is
    // still cached and a later action can enqueue it without another query.
    constexpr double kRequestTimeoutS = 2.0;
    for (auto &requests : pending_bs_chunk_requests_) {
      for (auto iterator = requests.begin(); iterator != requests.end();) {
        if (stamp - iterator->second.requested_at > kRequestTimeoutS) {
          iterator = requests.erase(iterator);
          ++bs_chunk_payload_request_timeouts_;
        } else {
          ++iterator;
        }
      }
    }
  }

  bool enqueueCachedBsChunkUpload(
      int sender, const ChunkKey &key, double born_at, int action_sender,
      std::uint64_t action_id, std::uint64_t action_step_id,
      const std::shared_ptr<const ActionChannelSnapshot> &action_snapshot) {
    if (sender < 0 || sender >= drone_count_ || bs_chunks_.count(key) ||
        hasPendingBsUplink(key)) {
      return false;
    }
    const auto repository = chunk_repository_.find(key);
    if (repository == chunk_repository_.end() || !repository->second.message) {
      return false;
    }
    const auto topic_index = chunkDataTopicIndex();
    const auto &cached = repository->second;
    auto routed = retargetChunkMessage(cached.message, sender, ap_node_id_);
    if (!routed) return false;
    auto flow = makeCentralFlow(key.owner - 1, topic_index, cached.bytes,
                                born_at);
    if (!enqueue(sender, ap_node_id_, topic_index, routed, flow,
                 RouteStage::kBsUplink, -1, born_at, &key, cached.bytes, 0U,
                 false, action_sender, action_id, action_step_id,
                 action_snapshot)) {
      return false;
    }
    ++bs_incremental_chunks_scheduled_uplink_;
    return true;
  }

  bool publishBsChunkRequest(int sender,
                             const std::vector<ChunkKey> &keys,
                             double stamp,
                             bool payload_precache = false) {
    if (sender < 0 || sender >= drone_count_ || keys.empty()) return false;
    racer_fidelity_msgs::msg::ChunkStamps request;
    // Zero is outside RACER's one-based UAV ID space and denotes a local BS
    // payload request.  Each idx_lists entry contains exact chunks to emit,
    // rather than an inventory complement.
    request.from_drone_id = 0;
    request.time = stamp;
    request.idx_lists.resize(static_cast<std::size_t>(drone_count_));
    for (const auto &key : keys) {
      if (key.owner < 1 || key.owner > drone_count_ || key.index == 0U ||
          key.index > static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::max())) {
        continue;
      }
      auto &ranges = request.idx_lists[static_cast<std::size_t>(key.owner - 1)]
                         .ids;
      const auto index = static_cast<std::int32_t>(key.index);
      if (ranges.size() >= 2U &&
          ranges.back() < std::numeric_limits<std::int32_t>::max() &&
          ranges.back() + 1 == index) {
        ranges.back() = index;
      } else {
        ranges.push_back(index);
        ranges.push_back(index);
      }
    }
    auto serialized = std::make_shared<rclcpp::SerializedMessage>();
    try {
      rclcpp::Serialization<racer_fidelity_msgs::msg::ChunkStamps>
          serializer;
      serializer.serialize_message(&request, serialized.get());
      publishers_[topicIndex("chunk_stamps")]
                 [static_cast<std::size_t>(sender)]->publish(*serialized);
    } catch (const std::exception &exception) {
      RCLCPP_WARN(get_logger(),
                  "failed to request RACER chunks from UAV %d: %s",
                  sender, exception.what());
      return false;
    }
    ++bs_chunk_payload_request_messages_;
    bs_chunk_payload_chunks_requested_ += keys.size();
    if (payload_precache) {
      ++payload_precache_request_messages_;
      payload_precache_chunks_requested_ += keys.size();
    }
    return true;
  }

  bool handleRequestedBsChunkResponse(
      int sender, const ChunkKey &key,
      const std::shared_ptr<rclcpp::SerializedMessage> &message) {
    racer_fidelity_msgs::msg::ChunkData chunk;
    if (!deserialize(message, chunk) || chunk.to_drone_id != 0) return false;
    if (sender < 0 || sender >= drone_count_ ||
        chunk.from_drone_id != sender + 1) {
      ++bs_chunk_payload_unsolicited_responses_;
      return true;
    }
    const auto precache = inflight_payload_precache_.find(key);
    const bool pending_precache =
        key.owner - 1 == sender && pending_payload_precache_keys_.count(key);
    if ((precache != inflight_payload_precache_.end() &&
         precache->second.sender == sender) || pending_precache) {
      satisfyPayloadPrecache(key, true);
      ++bs_chunk_payload_responses_received_;
      servicePayloadPrecache(now().seconds());
      // Precache only materializes immutable payload bytes. A later RL slot
      // must still select a real holder before the chunk enters the radio.
      return true;
    }
    auto &requests =
        pending_bs_chunk_requests_[static_cast<std::size_t>(sender)];
    const auto found = requests.find(key);
    if (found == requests.end()) {
      // Never expose a BS-local response on a UAV receive topic.  Its payload
      // remains useful in chunk_repository_ for a later selected action.
      ++bs_chunk_payload_unsolicited_responses_;
      return true;
    }
    const PendingBsChunkRequest pending = found->second;
    requests.erase(found);
    ++bs_chunk_payload_responses_received_;
    if (bs_chunks_.count(key) || hasPendingBsUplink(key)) {
      ++bs_chunk_payload_request_duplicates_suppressed_;
      return true;
    }
    enqueueCachedBsChunkUpload(
        sender, key, pending.born_at, pending.action_sender,
        pending.action_id, pending.action_step_id,
        pending.action_channel_snapshot);
    return true;
  }

  void onPushedChunkPayload(
      int sender, std::shared_ptr<rclcpp::SerializedMessage> message) {
    racer_fidelity_msgs::msg::ChunkData chunk;
    if (!deserialize(message, chunk) || chunk.to_drone_id != -1 ||
        chunk.idx == 0U || chunk.chunk_drone_id < 1 ||
        chunk.chunk_drone_id > drone_count_) {
      ++payload_direct_pushes_rejected_;
      return;
    }
    const ChunkKey key{chunk.chunk_drone_id, chunk.idx};
    if (sender < 0 || sender >= drone_count_ ||
        chunk.from_drone_id != sender + 1 || key.owner != sender + 1) {
      ++payload_direct_pushes_rejected_;
      return;
    }
    // This callback is independent of the ordinary UAV TX subscriptions. It
    // only fills the immutable local repository and UAV inventory; the BS map
    // changes later, after enqueueCachedBsChunkUpload() passes the payload
    // through the selected action and real PHY/MAC path.
    const std::size_t bytes = 64U + message->size();
    const bool new_definition =
        chunk_repository_.find(key) == chunk_repository_.end();
    chunk_repository_.insert_or_assign(
        key, cacheChunk(std::move(message), bytes, chunk));
    ++downlink_inventory_version_;
    if (new_definition) cacheNewChunkDefinition(key, bytes);
    insertKnownUavChunk(sender, key);
    if (task_metric_observer_mode_ == "inline") {
      applyTaskChunkObservation(chunk, true, false);
    } else if (task_metric_observer_mode_ == "async") {
      enqueueTaskObservation(
          {TaskObservationKind::kUavChunk, sender, nullptr,
           std::make_shared<racer_fidelity_msgs::msg::ChunkData>(
               std::move(chunk))});
    }
    ++payload_direct_pushes_received_;
    satisfyPayloadPrecache(key, false);
    servicePayloadPrecache(now().seconds());
  }

  bool hasPendingPeerMessage(const LinkKey &link, RouteStage route,
                             const CachedPeerMessage &cached) const {
    const auto found = queues_.find(link);
    if (found == queues_.end()) return false;
    return std::any_of(
        found->second.packets.begin(), found->second.packets.end(),
        [&cached, route](const PendingPacket &packet) {
          return packet.route == route &&
                 packet.peer_message_version == cached.version &&
                 packet.message == cached.message;
        });
  }

  bool bsBudgetChannelSnr(int sender, int receiver, double stamp,
                          double &effective_snr_db) const {
    if (sender < 0 || receiver < 0 || sender >= radio_node_count_ ||
        receiver >= radio_node_count_) {
      return false;
    }
    if (force_bs_perfect_delivery_) {
      effective_snr_db = 40.0;
      return true;
    }
    if (rl_action_channel_snapshot_) {
      const auto &snapshot = *rl_action_channel_snapshot_;
      const auto index = static_cast<std::size_t>(sender) *
                             static_cast<std::size_t>(radio_node_count_) +
                         static_cast<std::size_t>(receiver);
      if (index >= snapshot.available.size() ||
          index >= snapshot.effective_snr_db.size() ||
          snapshot.available[index] == 0U) {
        return false;
      }
      effective_snr_db =
          static_cast<double>(snapshot.effective_snr_db[index]);
      return std::isfinite(effective_snr_db);
    }
    const auto *link = linkFor({sender, receiver}, stamp);
    if (!usable(link)) return false;
    effective_snr_db = radioSnrDb(link, bs_model_);
    return std::isfinite(effective_snr_db);
  }

  AdaptiveChunkQuota adaptiveBsChunkQuota(
      int sender, int receiver, double stamp, double admitted_airtime_s) const {
    AdaptiveChunkQuota quota;
    if (!bsBudgetChannelSnr(
            sender, receiver, stamp, quota.effective_snr_db)) {
      return quota;
    }
    quota.available = true;
    quota.mcs = bs_model_.selectMcs(quota.effective_snr_db).name;
    if (admitted_airtime_s <= 0.0) return quota;

    const int budgeted_retries =
        mode_ == "ideal" || force_bs_perfect_delivery_ ? 0 : bs_max_retries_;
    quota.chunks = bs_model_.reliablePacketCapacity(
        quota.effective_snr_db, kRepresentativeMapChunkBytes,
        budgeted_retries, admitted_airtime_s);
    return quota;
  }

  std::size_t scheduleUavBsUpload(
      int sender, double stamp, int action_sender,
      std::uint64_t action_id, std::uint64_t action_step_id,
      const std::shared_ptr<const ActionChannelSnapshot> &action_snapshot,
      std::uint64_t selection_budget_id,
      std::size_t requested_selection_limit = 32U) {
    if (sender < 0 || sender >= drone_count_) return 0U;
    const auto sender_index = static_cast<std::size_t>(sender);
    pruneExpiredBsChunkRequests(stamp);
    if (bs_uplink_budget_action_id_[sender_index] != selection_budget_id) {
      bs_uplink_budget_action_id_[sender_index] = selection_budget_id;
      bs_uplink_chunks_selected_for_action_[sender_index] = 0U;
    }
    // The action-bound PHY quota is the normal admission limit. A positive
    // legacy in-flight value may still tighten it for controlled experiments.
    // With the default zero setting, keep roughly two current physical slots
    // outstanding. This bandwidth-delay window changes with the action-bound
    // SNR/MCS quota and drains from real delivery feedback; it is not a fixed
    // global chunk limit.
    std::size_t action_limit = requested_selection_limit;
    if (bs_max_uplink_chunks_per_turn_ > 0) {
      action_limit = std::min(
          action_limit,
          static_cast<std::size_t>(bs_max_uplink_chunks_per_turn_));
    }
    if (bs_max_inflight_chunks_per_uav_ > 0) {
      const auto inflight = bsUploadInflightChunkCount(sender);
      const auto inflight_limit =
          static_cast<std::size_t>(bs_max_inflight_chunks_per_uav_);
      const auto available_inflight_slots =
          inflight < inflight_limit ? inflight_limit - inflight : 0U;
      action_limit = std::min(action_limit, available_inflight_slots);
      if (requested_selection_limit > 0U && action_limit == 0U) {
        ++bs_upload_selection_backpressure_skips_;
      }
    } else if (requested_selection_limit > 0U) {
      const auto inflight = bsUploadInflightChunkCount(sender);
      constexpr std::size_t kAdaptiveOutstandingSlots = 2U;
      const auto adaptive_window =
          kAdaptiveOutstandingSlots * requested_selection_limit;
      const auto available_inflight_slots =
          inflight < adaptive_window ? adaptive_window - inflight : 0U;
      action_limit = std::min(action_limit, available_inflight_slots);
      if (action_limit == 0U) {
        ++bs_upload_selection_backpressure_skips_;
      }
    }
    auto &selected_for_action =
        bs_uplink_chunks_selected_for_action_[sender_index];
    const auto selected_before = selected_for_action;
    // Original RACER semantics: any chunk the current sender possesses can be
    // forwarded, regardless of which UAV created it.  ChunkKey::owner remains
    // provenance only.  Include stamp-announced chunks whose payload has not
    // yet passed through the Proxy: those are requested from MultiMapManager
    // below instead of being silently omitted.
    const auto &candidates = ordered_bs_missing_chunks_[sender_index];
    const LinkKey link{sender, ap_node_id_};
    std::vector<ChunkKey> payload_requests;
    for (const auto &key : candidates) {
      if (selected_for_action >= action_limit) break;
      // Multiple UAVs may know the same immutable owner/index.  Permit a
      // retry from another holder after failure, but never queue simultaneous
      // uploads or materialization requests for a chunk BS still lacks.
      if (hasPendingBsUplink(key) || hasPendingBsChunkRequest(key)) {
        ++bs_chunk_payload_request_duplicates_suppressed_;
        continue;
      }
      const auto repository = chunk_repository_.find(key);
      if (repository != chunk_repository_.end()) {
        if (enqueueCachedBsChunkUpload(
                sender, key, stamp, action_sender, action_id, action_step_id,
                action_snapshot)) {
          ++selected_for_action;
        }
        continue;
      }
      if (bs_payload_precache_enabled_) continue;
      PendingBsChunkRequest pending;
      pending.requested_at = stamp;
      pending.born_at = stamp;
      pending.action_sender = action_sender;
      pending.action_id = action_id;
      pending.action_step_id = action_step_id;
      pending.action_channel_snapshot = action_snapshot;
      pending_bs_chunk_requests_[sender_index].emplace(key,
                                                       std::move(pending));
      payload_requests.push_back(key);
      ++selected_for_action;
    }
    if (!payload_requests.empty() &&
        !publishBsChunkRequest(sender, payload_requests, stamp)) {
      for (const auto &key : payload_requests) {
        pending_bs_chunk_requests_[sender_index].erase(key);
      }
      selected_for_action -= std::min(selected_for_action,
                                      payload_requests.size());
    }
    const auto &latest = uav_latest_peer_messages_[sender_index];
    const auto &at_bs = bs_latest_peer_messages_[sender_index];
    for (std::size_t peer_topic = 0; peer_topic < kTopicPolicies.size();
         ++peer_topic) {
      if (!bsPeerRelayEligible(peer_topic)) continue;
      const auto &cached = latest[peer_topic];
      if (!cached.message || cached.version <= at_bs[peer_topic].version ||
          hasPendingPeerMessage(link, RouteStage::kBsUplink, cached)) {
        continue;
      }
      if (enqueue(sender, ap_node_id_, peer_topic, cached.message, cached.flow,
                  RouteStage::kBsUplink, -1, cached.born_at, nullptr,
                  cached.bytes, cached.version, false, action_sender, action_id,
                  action_step_id, action_snapshot)) {
        ++bs_peer_uplink_scheduled_;
        ++bs_peer_uplink_scheduled_by_topic_[peer_topic];
      }
    }
    return selected_for_action - selected_before;
  }

  std::size_t scheduleRlUpload(
      int sender, double stamp, std::size_t requested_selection_limit,
      std::uint64_t selection_budget_id) {
    const auto action_snapshot =
        rl_have_action_ ? rl_action_channel_snapshot_ : nullptr;
    const auto action_id = rl_have_action_ ? rl_action_epoch_ : 0U;
    const auto action_step_id =
        rl_have_action_ ? rl_action_source_step_id_ : 0U;
    return scheduleUavBsUpload(
        sender, stamp, sender, action_id, action_step_id, action_snapshot,
        selection_budget_id, requested_selection_limit);
  }

  void schedulePeriodicBsUpload(int sender, double stamp,
                                std::uint64_t request_round) {
    constexpr std::uint64_t kPeriodicBudgetNamespace = 1ULL << 63U;
    (void)scheduleUavBsUpload(
        sender, stamp, sender, 0U, request_round, nullptr,
        kPeriodicBudgetNamespace | request_round,
        static_cast<std::size_t>(bs_periodic_upload_chunks_per_request_));
    ++bs_periodic_upload_triggers_;
  }

  std::size_t scheduleRlRelay(
      int action_sender, int receiver, double stamp,
      std::size_t requested_selection_limit) {
    if (action_sender == receiver) return 0U;
    const auto action_snapshot =
        rl_have_action_ ? rl_action_channel_snapshot_ : nullptr;
    const auto action_id = rl_have_action_ ? rl_action_epoch_ : 0U;
    const auto action_step_id =
        rl_have_action_ ? rl_action_source_step_id_ : 0U;
    const LinkKey link{ap_node_id_, receiver};
    // Optional source-owned mode restricts B[i,j] to immutable owner i.
    // BS may still use its existing cache independently of concurrent upload.
    const auto &candidates = rankedDownlinkChunks(receiver);
    std::size_t limit = requested_selection_limit;
    if (bs_max_downlink_chunks_per_turn_ > 0) {
      limit = std::min(
          limit,
          static_cast<std::size_t>(bs_max_downlink_chunks_per_turn_));
    }
    const auto topic_index = chunkDataTopicIndex();
    std::size_t chunks_scheduled{};
    for (const auto &key : candidates) {
      if (chunks_scheduled >= limit) break;
      if (bs_downlink_source_owned_chunks_only_ && key.owner != action_sender + 1) continue;
      if (hasPendingChunk(link, key, RouteStage::kBsDownlink)) continue;
      const auto repository = chunk_repository_.find(key);
      if (repository == chunk_repository_.end()) continue;
      const auto &cached = repository->second;
      auto routed = retargetChunkMessage(
          cached.message, ap_node_id_, receiver);
      if (!routed) continue;
      auto flow = makeCentralFlow(key.owner - 1, topic_index, cached.bytes,
                                  stamp);
      if (enqueue(ap_node_id_, receiver, topic_index, routed, flow,
                  RouteStage::kBsDownlink, receiver, stamp, &key,
                  cached.bytes, 0U, false, action_sender, action_id,
                  action_step_id, action_snapshot)) {
        ++bs_missing_chunks_scheduled_downlink_;
        ++chunks_scheduled;
      }
    }
    const auto &latest =
        bs_latest_peer_messages_[static_cast<std::size_t>(action_sender)];
    for (std::size_t peer_topic = 0; peer_topic < kTopicPolicies.size();
         ++peer_topic) {
      if (!bsPeerRelayEligible(peer_topic)) continue;
      const auto &cached = latest[peer_topic];
      const auto receiver_index = static_cast<std::size_t>(receiver);
      if (!cached.message || !cached.flow ||
          cached.flow->intended.size() !=
              static_cast<std::size_t>(drone_count_) ||
          !cached.flow->intended[receiver_index] ||
          cached.flow->delivered[receiver_index] ||
          hasPendingPeerMessage(link, RouteStage::kBsDownlink, cached)) {
        continue;
      }
      if (enqueue(ap_node_id_, receiver, peer_topic, cached.message,
                  cached.flow, RouteStage::kBsDownlink, receiver,
                  cached.born_at, nullptr, cached.bytes, cached.version,
                  false, action_sender, action_id, action_step_id,
                  action_snapshot)) {
        ++bs_peer_downlink_scheduled_;
        ++bs_peer_downlink_scheduled_by_topic_[peer_topic];
      }
    }
    return chunks_scheduled;
  }

  bool readRlAction(std::uint64_t expected_decision_index = 0U) {
    std::ifstream file_stream;
    std::istringstream shared_stream;
    std::istream *stream{};
    SharedBlockSnapshot shared_snapshot;
    if (shared_action_) {
      if (!shared_action_->snapshot(shared_snapshot) ||
          shared_snapshot.version <= rl_shared_action_version_) {
        return false;
      }
      shared_stream.str(shared_snapshot.payload);
      stream = &shared_stream;
    } else {
      file_stream.open(rl_bs_action_path_);
      stream = &file_stream;
    }
    std::uint64_t epoch{};
    int action_drone_count{};
    if (!(*stream >> epoch >> action_drone_count) ||
        action_drone_count != drone_count_ ||
        epoch <= (shared_action_ ? rl_latest_action_epoch_
                                : rl_action_epoch_)) {
      return false;
    }
    std::uint64_t action_decision_index{};
    std::uint64_t source_physical_version{};
    std::uint64_t source_communication_version{};
    std::uint64_t source_guidance_id{};
    std::uint64_t source_sim_step{};
    std::uint64_t policy_version{};
    std::uint64_t generated_wall_time_ns{};
    double source_sim_time_s{};
    std::uint64_t source_step_id{};
    if (shared_action_) {
      if (epoch != shared_snapshot.version ||
          !(*stream >> source_physical_version
                   >> source_communication_version
                   >> source_guidance_id
                   >> source_sim_step
                   >> policy_version
                   >> generated_wall_time_ns
                   >> source_sim_time_s
                   >> source_step_id)) {
        return false;
      }
    } else if (rl_bs_synchronous_mode_) {
      if (!(*stream >> action_decision_index) ||
          action_decision_index != expected_decision_index) {
        return false;
      }
    }
    std::vector<std::vector<bool>> relay(
        static_cast<std::size_t>(drone_count_),
        std::vector<bool>(static_cast<std::size_t>(drone_count_), false));
    std::vector<bool> upload(static_cast<std::size_t>(drone_count_), false);
    int bit{};
    for (int sender = 0; sender < drone_count_; ++sender) {
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (sender == receiver) continue;
        if (!(*stream >> bit) || (bit != 0 && bit != 1)) return false;
        relay[static_cast<std::size_t>(sender)]
             [static_cast<std::size_t>(receiver)] = bit != 0;
      }
    }
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (!(*stream >> bit) || (bit != 0 && bit != 1)) return false;
      upload[static_cast<std::size_t>(sender)] = bit != 0;
    }
    std::shared_ptr<const ActionChannelSnapshot> channel_snapshot;
    if (shared_action_) {
      channel_snapshot = actionChannelSnapshot(source_communication_version);
      if (!channel_snapshot) {
        // Consume the malformed/stale mailbox value so it cannot be retried
        // forever, but never execute it against links_ or another snapshot.
        rl_shared_action_version_ = shared_snapshot.version;
        ++rl_rejected_missing_channel_snapshot_actions_;
        RCLCPP_ERROR(
            get_logger(),
            "RACER_ACTION_REJECT action_id=%llu step_id=%llu "
            "channel_version=%llu reason=missing_channel_snapshot",
            static_cast<unsigned long long>(epoch),
            static_cast<unsigned long long>(source_step_id),
            static_cast<unsigned long long>(source_communication_version));
        return false;
      }
      if (channel_snapshot->step_id != source_step_id) {
        rl_shared_action_version_ = shared_snapshot.version;
        ++rl_rejected_mismatched_channel_snapshot_actions_;
        RCLCPP_ERROR(
            get_logger(),
            "RACER_ACTION_REJECT action_id=%llu step_id=%llu "
            "channel_version=%llu snapshot_step_id=%llu "
            "reason=channel_step_mismatch",
            static_cast<unsigned long long>(epoch),
            static_cast<unsigned long long>(source_step_id),
            static_cast<unsigned long long>(source_communication_version),
            static_cast<unsigned long long>(channel_snapshot->step_id));
        return false;
      }
      // Reading a commit only replaces the pending/latest value. The action
      // currently driving the radio is switched exclusively at a 100 ms
      // control boundary.
      rl_shared_action_version_ = shared_snapshot.version;
      rl_latest_action_epoch_ = epoch;
      rl_latest_action_source_physical_version_ = source_physical_version;
      rl_latest_action_source_communication_version_ =
          source_communication_version;
      rl_latest_action_source_guidance_id_ = source_guidance_id;
      rl_latest_action_source_sim_step_ = source_sim_step;
      rl_latest_policy_version_ = policy_version;
      rl_latest_action_generated_wall_time_ns_ = generated_wall_time_ns;
      rl_latest_action_source_sim_time_s_ = source_sim_time_s;
      rl_latest_action_source_step_id_ = source_step_id;
      rl_latest_action_source_slot_ = shared_snapshot.sim_step;
      rl_latest_action_channel_snapshot_ = channel_snapshot;
      rl_have_latest_action_ = true;
      rl_latest_relay_action_ = std::move(relay);
      rl_latest_upload_action_ = std::move(upload);
    } else {
      // File-bridge actions append channel_version and source_step_id after
      // the action bits. Keep old hand-written actions usable by binding the
      // already emitted state for their decision step.
      if (*stream >> source_communication_version >> source_step_id) {
        channel_snapshot = actionChannelSnapshot(source_communication_version);
      } else {
        stream->clear();
        if (rl_bs_synchronous_mode_) {
          source_step_id = action_decision_index;
          channel_snapshot = actionChannelSnapshotForStep(source_step_id);
        } else if (!action_channel_snapshot_versions_.empty()) {
          channel_snapshot = actionChannelSnapshot(
              action_channel_snapshot_versions_.back());
          if (channel_snapshot) source_step_id = channel_snapshot->step_id;
        }
        if (channel_snapshot) {
          source_communication_version = channel_snapshot->channel_version;
          RCLCPP_WARN_ONCE(
              get_logger(),
              "legacy RL action has no channel trailer; inferred the frozen "
              "snapshot from step_id for compatibility");
        }
      }
      if (!channel_snapshot || channel_snapshot->step_id != source_step_id) {
        if (!channel_snapshot) {
          ++rl_rejected_missing_channel_snapshot_actions_;
        } else {
          ++rl_rejected_mismatched_channel_snapshot_actions_;
        }
        RCLCPP_ERROR(
            get_logger(),
            "RACER_ACTION_REJECT action_id=%llu step_id=%llu "
            "channel_version=%llu reason=invalid_file_channel_snapshot",
            static_cast<unsigned long long>(epoch),
            static_cast<unsigned long long>(source_step_id),
            static_cast<unsigned long long>(source_communication_version));
        return false;
      }
      rl_action_epoch_ = epoch;
      rl_action_decision_index_ = action_decision_index;
      rl_action_source_communication_version_ = source_communication_version;
      rl_action_source_step_id_ = source_step_id;
      rl_action_source_sim_time_s_ = channel_snapshot->sim_time_s;
      rl_action_channel_snapshot_ = channel_snapshot;
      rl_have_action_ = true;
      rl_relay_action_ = std::move(relay);
      rl_upload_action_ = std::move(upload);
    }
    return true;
  }

  void scheduleRlActionWithinSlotBudgets(double stamp) {
    std::vector<int> upload_senders;
    std::vector<std::pair<int, int>> relay_pairs;
    for (int sender = 0; sender < drone_count_; ++sender) {
      bool route_selected = bs_all_to_all_relay_enabled_;
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (sender == receiver) continue;
        if (bs_all_to_all_relay_enabled_ ||
            rl_relay_action_[static_cast<std::size_t>(sender)]
                            [static_cast<std::size_t>(receiver)]) {
          route_selected = true;
          relay_pairs.emplace_back(sender, receiver);
        }
      }
      if (route_selected ||
          rl_upload_action_[static_cast<std::size_t>(sender)]) {
        upload_senders.push_back(sender);
      }
    }

    // UL and DL serialize on one BS radio timeline.  Reserve half of the map
    // airtime for each direction and divide it equally among its selected,
    // usable links.  The per-link quota then follows the action-bound Channel
    // State, adaptive MCS, NR slot quantization and expected retry airtime.
    // The configured per-slot values are hard ceilings across all links; the
    // adaptive sum is normally lower on weak channels.
    const auto logical_slot = static_cast<std::uint64_t>(std::max(
        0.0, std::floor(stamp / rl_bs_communication_slot_s_ + 1.0e-9)));
    const std::size_t upload_count = upload_senders.size();
    const std::size_t upload_offset =
        upload_count == 0U ? 0U : logical_slot % upload_count;
    std::rotate(upload_senders.begin(),
                upload_senders.begin() + static_cast<std::ptrdiff_t>(
                                             upload_offset),
                upload_senders.end());
    std::vector<AdaptiveChunkQuota> upload_quotas(upload_count);
    std::size_t usable_uploads{};
    for (std::size_t position = 0; position < upload_count; ++position) {
      double ignored_snr{};
      if (bsBudgetChannelSnr(upload_senders[position], ap_node_id_, stamp,
                             ignored_snr)) {
        ++usable_uploads;
      }
    }
    const double directional_map_airtime_s =
        0.5 * kBsMapAdmissionFraction * rl_bs_communication_slot_s_;
    const double upload_airtime_per_link_s =
        usable_uploads == 0U
            ? 0.0
            : directional_map_airtime_s /
                  static_cast<double>(usable_uploads);
    std::size_t adaptive_uplink_budget{};
    for (std::size_t position = 0; position < upload_count; ++position) {
      auto &quota = upload_quotas[position];
      quota = adaptiveBsChunkQuota(upload_senders[position], ap_node_id_,
                                   stamp, upload_airtime_per_link_s);
      if (!quota.available) {
        ++bs_adaptive_budget_unavailable_links_;
        continue;
      }
      ++bs_adaptive_budget_channel_samples_;
      ++bs_adaptive_budget_mcs_counts_[std::string(quota.mcs)];
      adaptive_uplink_budget += quota.chunks;
    }
    const std::size_t uplink_slot_budget = std::min(
        adaptive_uplink_budget,
        static_cast<std::size_t>(bs_max_uplink_chunks_per_rl_slot_));
    bs_last_adaptive_uplink_chunk_budget_ = uplink_slot_budget;
    bs_adaptive_uplink_chunk_budget_total_ += uplink_slot_budget;
    std::size_t uplink_remaining = uplink_slot_budget;
    std::size_t uplink_selected{};
    constexpr std::uint64_t kRlSlotBudgetNamespace = 1ULL << 62U;
    const auto selection_budget_id = kRlSlotBudgetNamespace | logical_slot;
    for (std::size_t position = 0; position < upload_count; ++position) {
      const auto &quota = upload_quotas[position];
      const std::size_t senders_left = static_cast<std::size_t>(std::count_if(
          upload_quotas.begin() + static_cast<std::ptrdiff_t>(position),
          upload_quotas.end(),
          [](const AdaptiveChunkQuota &candidate) {
            return candidate.chunks > 0U;
          }));
      const std::size_t fair_limit =
          senders_left == 0U
              ? 0U
              : std::min(
                    quota.chunks,
                    (uplink_remaining + senders_left - 1U) / senders_left);
      const auto scheduled = scheduleRlUpload(
          upload_senders[position], stamp, fair_limit, selection_budget_id);
      uplink_remaining -= std::min(uplink_remaining, scheduled);
      uplink_selected += scheduled;
    }
    bs_last_adaptive_uplink_chunks_selected_ = uplink_selected;
    bs_adaptive_uplink_chunks_selected_total_ += uplink_selected;
    if (uplink_slot_budget > 0U && uplink_remaining == 0U) {
      ++bs_uplink_slot_budget_exhaustions_;
    }

    const std::size_t relay_count = relay_pairs.size();
    const std::size_t relay_offset =
        relay_count == 0U ? 0U : logical_slot % relay_count;
    std::rotate(relay_pairs.begin(),
                relay_pairs.begin() + static_cast<std::ptrdiff_t>(
                                         relay_offset),
                relay_pairs.end());
    std::vector<AdaptiveChunkQuota> relay_quotas(relay_count);
    std::size_t usable_relays{};
    for (const auto &[sender, receiver] : relay_pairs) {
      (void)sender;
      double ignored_snr{};
      if (bsBudgetChannelSnr(ap_node_id_, receiver, stamp, ignored_snr)) {
        ++usable_relays;
      }
    }
    const double relay_airtime_per_link_s =
        usable_relays == 0U
            ? 0.0
            : directional_map_airtime_s /
                  static_cast<double>(usable_relays);
    std::size_t adaptive_downlink_budget{};
    for (std::size_t position = 0; position < relay_count; ++position) {
      const auto &[sender, receiver] = relay_pairs[position];
      (void)sender;
      auto &quota = relay_quotas[position];
      quota = adaptiveBsChunkQuota(ap_node_id_, receiver, stamp,
                                   relay_airtime_per_link_s);
      if (!quota.available) {
        ++bs_adaptive_budget_unavailable_links_;
        continue;
      }
      ++bs_adaptive_budget_channel_samples_;
      ++bs_adaptive_budget_mcs_counts_[std::string(quota.mcs)];
      adaptive_downlink_budget += quota.chunks;
    }
    const std::size_t downlink_slot_budget = std::min(
        adaptive_downlink_budget,
        static_cast<std::size_t>(bs_max_downlink_chunks_per_rl_slot_));
    bs_last_adaptive_downlink_chunk_budget_ = downlink_slot_budget;
    bs_adaptive_downlink_chunk_budget_total_ += downlink_slot_budget;
    std::size_t downlink_remaining = downlink_slot_budget;
    std::size_t downlink_selected{};
    for (std::size_t position = 0; position < relay_count; ++position) {
      const auto &quota = relay_quotas[position];
      const std::size_t pairs_left = static_cast<std::size_t>(std::count_if(
          relay_quotas.begin() + static_cast<std::ptrdiff_t>(position),
          relay_quotas.end(),
          [](const AdaptiveChunkQuota &candidate) {
            return candidate.chunks > 0U;
          }));
      const std::size_t fair_limit =
          pairs_left == 0U
              ? 0U
              : std::min(
                    quota.chunks,
                    (downlink_remaining + pairs_left - 1U) / pairs_left);
      const auto &[sender, receiver] = relay_pairs[position];
      const auto scheduled =
          scheduleRlRelay(sender, receiver, stamp, fair_limit);
      downlink_remaining -= std::min(downlink_remaining, scheduled);
      downlink_selected += scheduled;
    }
    bs_last_adaptive_downlink_chunks_selected_ = downlink_selected;
    bs_adaptive_downlink_chunks_selected_total_ += downlink_selected;
    ++bs_adaptive_chunk_budget_slots_;
    if (downlink_slot_budget > 0U && downlink_remaining == 0U) {
      ++bs_downlink_slot_budget_exhaustions_;
    }
  }

  void scheduleActiveRlAction(double stamp) {
    if (!rl_have_action_ && !bs_all_to_all_relay_enabled_) return;
    if (rl_have_action_ && !rl_action_channel_snapshot_) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "refusing to schedule RL action %llu without its channel snapshot",
          static_cast<unsigned long long>(rl_action_epoch_));
      return;
    }
    scheduleRlActionWithinSlotBudgets(stamp);
  }

  void activateLatestRlAction(double boundary_stamp) {
    // The action carries the exact Fast State decision that produced it.  It
    // may become active only at a later 100 ms decision boundary.  Do not use
    // now().seconds() here: when the wall timer is catching up queued logical
    // slots, ROS time is ahead of boundary_stamp and that comparison can
    // reject every new action forever even though it was computed in time for
    // the next logical boundary.
    const bool source_precedes_boundary =
        rl_latest_action_source_step_id_ < rl_decision_index_;
    if (!rl_have_latest_action_ ||
        rl_latest_action_epoch_ <= rl_action_epoch_ ||
        !source_precedes_boundary) {
      return;
    }
    if (!rl_latest_action_channel_snapshot_) {
      RCLCPP_ERROR(get_logger(),
                   "refusing to activate action %llu without its channel "
                   "snapshot",
                   static_cast<unsigned long long>(
                       rl_latest_action_epoch_));
      return;
    }
    const auto apply_started = std::chrono::steady_clock::now();
    rl_action_epoch_ = rl_latest_action_epoch_;
    rl_action_source_physical_version_ =
        rl_latest_action_source_physical_version_;
    rl_action_source_communication_version_ =
        rl_latest_action_source_communication_version_;
    rl_action_source_guidance_id_ = rl_latest_action_source_guidance_id_;
    rl_action_source_sim_step_ = rl_latest_action_source_sim_step_;
    rl_action_policy_version_ = rl_latest_policy_version_;
    rl_action_generated_wall_time_ns_ =
        rl_latest_action_generated_wall_time_ns_;
    rl_action_source_sim_time_s_ = rl_latest_action_source_sim_time_s_;
    rl_action_source_step_id_ = rl_latest_action_source_step_id_;
    rl_action_channel_snapshot_ = rl_latest_action_channel_snapshot_;
    rl_relay_action_ = rl_latest_relay_action_;
    rl_upload_action_ = rl_latest_upload_action_;
    rl_action_decision_index_ = rl_decision_index_;
    rl_action_applied_sim_time_s_ = boundary_stamp;
    rl_have_action_ = true;
    publishSharedActionAck(boundary_stamp);
    const double elapsed_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - apply_started).count();
    action_apply_ack_total_ms_ += elapsed_ms;
    ++action_apply_ack_count_;
    action_apply_ack_max_ms_ = std::max(action_apply_ack_max_ms_, elapsed_ms);
    RCLCPP_INFO(get_logger(),
                "RACER_ACTION_APPLY step_id=%llu action_version=%llu "
                "channel_version=%llu "
                "action_apply_ack_ms=%.6f action_apply_ack_mean_ms=%.6f "
                "action_apply_ack_max_ms=%.6f",
                static_cast<unsigned long long>(rl_decision_index_),
                static_cast<unsigned long long>(rl_action_epoch_),
                static_cast<unsigned long long>(
                    rl_action_source_communication_version_), elapsed_ms,
                action_apply_ack_total_ms_ /
                    static_cast<double>(action_apply_ack_count_),
                action_apply_ack_max_ms_);
  }

  void activateLatestEventDrivenAction(double apply_stamp) {
    // This is deliberately separate from activateLatestRlAction(): the
    // fixed-clock path switches active actions only at 100 ms boundaries,
    // while the experiment consumes a validated version exactly once at the
    // next available physical communication slot.
    rl_action_epoch_ = rl_latest_action_epoch_;
    rl_action_source_physical_version_ =
        rl_latest_action_source_physical_version_;
    rl_action_source_communication_version_ =
        rl_latest_action_source_communication_version_;
    rl_action_source_guidance_id_ = rl_latest_action_source_guidance_id_;
    rl_action_source_sim_step_ = rl_latest_action_source_sim_step_;
    rl_action_policy_version_ = rl_latest_policy_version_;
    rl_action_generated_wall_time_ns_ =
        rl_latest_action_generated_wall_time_ns_;
    rl_action_source_sim_time_s_ = rl_latest_action_source_sim_time_s_;
    rl_action_source_step_id_ = rl_latest_action_source_step_id_;
    rl_action_channel_snapshot_ = rl_latest_action_channel_snapshot_;
    rl_relay_action_ = rl_latest_relay_action_;
    rl_upload_action_ = rl_latest_upload_action_;
    rl_action_decision_index_ = rl_decision_index_;
    rl_action_applied_sim_time_s_ = apply_stamp;
    rl_have_latest_action_ = false;
    rl_have_action_ = true;
    publishSharedActionAck(apply_stamp);
  }

  void clearEventDrivenAction() {
    rl_have_action_ = false;
    rl_action_channel_snapshot_.reset();
    for (auto &row : rl_relay_action_) {
      std::fill(row.begin(), row.end(), false);
    }
    std::fill(rl_upload_action_.begin(), rl_upload_action_.end(), false);
  }

  bool eventDrivenActionSettled(std::uint64_t action_version) const {
    if (action_version == 0U) return false;
    for (const auto &[key, queue] : queues_) {
      (void)key;
      if (std::any_of(
              queue.packets.begin(), queue.packets.end(),
              [action_version](const PendingPacket &packet) {
                return packet.action_id == action_version;
              })) {
        return false;
      }
    }
    for (const auto &requests : pending_bs_chunk_requests_) {
      if (std::any_of(
              requests.begin(), requests.end(),
              [action_version](const auto &entry) {
                return entry.second.action_id == action_version;
              })) {
        return false;
      }
    }
    return true;
  }

  bool eventDrivenPacketMayUseBs(const PendingPacket &packet) const {
    return !rl_bs_event_driven_one_shot_ ||
           !usesBsRadio(packet.route) ||
           (rl_event_executing_action_version_ != 0U &&
            packet.action_id == rl_event_executing_action_version_);
  }

  bool advanceEventDrivenOneShotClock(double stamp) {
    if (!event_driven_gate_) {
      throw std::logic_error("event-driven one-shot gate is unavailable");
    }
    readRlAction();
    if (!std::all_of(
            uav_position_valid_.begin(), uav_position_valid_.end(),
            [](bool valid) { return valid; })) {
      return false;
    }
    // A missing local payload response is part of this action's result. Its
    // existing bounded request timeout must progress even though no later
    // action is allowed to enter while this one is settling.
    pruneExpiredBsChunkRequests(stamp);
    const auto observed_slot = static_cast<std::uint64_t>(std::max(
        0.0, std::floor(stamp / rl_bs_communication_slot_s_ + 1.0e-9)));
    const double slot_stamp =
        static_cast<double>(observed_slot) * rl_bs_communication_slot_s_;

    if (!event_driven_gate_->initialized()) {
      const auto initial = event_driven_gate_->initialize(
          observed_slot, slot_stamp);
      rl_communication_slot_index_ = observed_slot;
      rl_decision_index_ = 0U;
      rl_last_action_held_slots_ = 0U;
      rl_completed_interval_available_ = false;
      rl_pending_boundary_stamp_s_ = initial.sim_time_s;
      rl_event_interval_start_sim_time_s_ = initial.sim_time_s;
      rl_event_delta_t_s_ = rl_bs_communication_slot_s_;
      rl_event_last_state_slot_ = observed_slot;
      return true;
    }

    if (rl_have_latest_action_ &&
        rl_latest_action_epoch_ > event_driven_gate_->lastSeenVersion()) {
      const OneShotActionTicket ticket{
          rl_latest_action_epoch_, rl_latest_action_source_step_id_,
          rl_latest_action_source_communication_version_,
          rl_latest_action_source_slot_, rl_latest_action_source_sim_time_s_};
      const bool exact_source =
          ticket.source_communication_version == rl_state_sequence_ &&
          ticket.source_slot == rl_event_last_state_slot_;
      if (!exact_source || !event_driven_gate_->offer(ticket)) {
        ++rl_event_rejected_action_count_;
        rl_have_latest_action_ = false;
        RCLCPP_ERROR(
            get_logger(),
            "RACER_EVENT_ACTION_REJECT action_version=%llu source_step=%llu "
            "expected_step=%llu source_sim_time=%.9f current_sim_time=%.9f",
            static_cast<unsigned long long>(ticket.version),
            static_cast<unsigned long long>(ticket.source_step_id),
            static_cast<unsigned long long>(
                event_driven_gate_->completedTransitions()),
            ticket.source_sim_time_s, slot_stamp);
        throw std::runtime_error(
            "event-driven action provenance did not match the latest state");
      }
    }

    const auto in_flight_version =
        event_driven_gate_->inFlightActionVersion();
    const bool action_settled =
        in_flight_version != 0U &&
        eventDrivenActionSettled(in_flight_version);
    const auto event = event_driven_gate_->advance(
        observed_slot, slot_stamp, action_settled);
    rl_communication_slot_index_ = observed_slot;
    if (event.kind == OneShotEventKind::kApplyAction) {
      activateLatestEventDrivenAction(event.sim_time_s);
      scheduleActiveRlAction(event.sim_time_s);
      // In-flight packets may finish normally, but no subsequent slot can
      // enqueue work from this action version.
      rl_have_action_ = false;
      rl_completed_interval_available_ = false;
      rl_event_action_age_s_ = event.action_age_s;
      rl_event_executing_action_version_ = event.action_version;
      ++rl_event_applied_action_count_;
      RCLCPP_INFO(
          get_logger(),
          "RACER_EVENT_ACTION_ONESHOT action_version=%llu source_step=%llu "
          "apply_slot=%llu apply_sim_time=%.9f action_age=%.9f",
          static_cast<unsigned long long>(event.action_version),
          static_cast<unsigned long long>(event.transition_step_id),
          static_cast<unsigned long long>(event.slot), event.sim_time_s,
          event.action_age_s);
      return false;
    }
    if (event.kind == OneShotEventKind::kTransitionComplete) {
      rl_decision_index_ = event_driven_gate_->completedTransitions();
      rl_last_action_held_slots_ = 1U;
      rl_completed_interval_available_ = true;
      rl_pending_boundary_stamp_s_ = event.sim_time_s;
      rl_event_interval_start_sim_time_s_ =
          event.interval_start_sim_time_s;
      rl_event_delta_t_s_ = event.delta_t_s;
      rl_event_last_state_slot_ = event.slot;
      rl_event_executing_action_version_ = 0U;
      ++rl_event_completed_transition_count_;
      RCLCPP_INFO(
          get_logger(),
          "RACER_EVENT_TRANSITION_COMPLETE action_version=%llu "
          "transition_step=%llu result_slot=%llu result_sim_time=%.9f "
          "delta_t=%.9f held_slots=1",
          static_cast<unsigned long long>(event.action_version),
          static_cast<unsigned long long>(event.transition_step_id),
          static_cast<unsigned long long>(event.slot), event.sim_time_s,
          event.delta_t_s);
      return true;
    }
    return false;
  }

  void publishSharedActionAck(double stamp) {
    if (!shared_action_ack_) return;
    rl_action_applied_sim_time_s_ = stamp;
    std::ostringstream json;
    json << "{\"applied_action_id\":" << rl_action_epoch_
         << ",\"action_id\":" << rl_action_epoch_
         << ",\"source_physical_version\":"
         << rl_action_source_physical_version_
         << ",\"source_communication_version\":"
         << rl_action_source_communication_version_
         << ",\"channel_version\":"
         << rl_action_source_communication_version_
         << ",\"channel_snapshot_step_id\":"
         << (rl_action_channel_snapshot_
                 ? rl_action_channel_snapshot_->step_id
                 : 0U)
         << ",\"source_guidance_id\":" << rl_action_source_guidance_id_
         << ",\"source_sim_step\":" << rl_action_source_sim_step_
         << ",\"source_step_id\":" << rl_action_source_step_id_
         << ",\"racer_state_version\":" << rl_state_sequence_
         << ",\"apply_sim_time_s\":" << stamp
         << ",\"racer_pid\":" << ::getpid() << '}';
    const auto ack_version = shared_action_ack_->publish(
        json.str(), rl_state_sequence_, stamp);
    RCLCPP_INFO(
        get_logger(),
        "RACER_SHM_ACTION_APPLIED action_id=%llu ack_version=%llu "
        "source_physical=%llu source_communication=%llu channel_version=%llu "
        "source_step_id=%llu source_guidance=%llu sim_time=%.9f",
        static_cast<unsigned long long>(rl_action_epoch_),
        static_cast<unsigned long long>(ack_version),
        static_cast<unsigned long long>(rl_action_source_physical_version_),
        static_cast<unsigned long long>(rl_action_source_communication_version_),
        static_cast<unsigned long long>(rl_action_source_communication_version_),
        static_cast<unsigned long long>(rl_action_source_step_id_),
        static_cast<unsigned long long>(rl_action_source_guidance_id_), stamp);
  }

  bool beginRlDecision(double stamp) {
    if (!rl_bs_scheduler_enabled_) return false;

    if (!shared_communication_) {
      if (rl_have_decision_stamp_ &&
          stamp - rl_last_decision_s_ + 1.0e-12 <
              rl_bs_decision_period_s_) {
        return false;
      }
      rl_have_decision_stamp_ = true;
      rl_last_decision_s_ = stamp;
      readRlAction();
      scheduleActiveRlAction(stamp);
      return true;
    }

    // The action mailbox is sampled independently of the radio clock. A new
    // value only updates latest_action; active_action is not touched here.
    readRlAction();
    if (!std::all_of(
            uav_position_valid_.begin(), uav_position_valid_.end(),
            [](bool valid) { return valid; })) {
      return false;
    }
    const auto observed_slot = static_cast<std::uint64_t>(std::max(
        0.0, std::floor(stamp / rl_bs_communication_slot_s_ + 1.0e-9)));
    if (!rl_async_clock_initialized_) {
      if (observed_slot >= rl_slots_per_decision_) {
        throw std::runtime_error(
            "Fast RL State did not initialize before the first 100 ms "
            "boundary; refusing to invent skipped state snapshots");
      }
      rl_async_clock_initialized_ = true;
      rl_decision_index_ = observed_slot / rl_slots_per_decision_;
      rl_communication_slot_index_ =
          rl_decision_index_ * rl_slots_per_decision_;
      rl_pending_boundary_stamp_s_ =
          static_cast<double>(rl_decision_index_) * rl_bs_decision_period_s_;
      rl_last_action_held_slots_ = 0U;
      rl_completed_interval_available_ = false;
      rl_async_observed_slot_ = observed_slot;
      return true;
    }
    const auto next_boundary_slot =
        (rl_communication_slot_index_ / rl_slots_per_decision_ + 1U) *
        rl_slots_per_decision_;
    if (rl_communication_slot_index_ < next_boundary_slot &&
        observed_slot > next_boundary_slot) {
      // A long-running process can occasionally be descheduled across a
      // boundary. Never abort the communication pipeline and never emit a
      // burst of identical snapshots in this callback. The loop below
      // advances to exactly one boundary and returns; executor callbacks can
      // then apply pending RX/TX/odometry work before a subsequent boundary
      // is recovered. step_id therefore stays contiguous while each record
      // is still built by a separate scheduler callback.
      ++rl_late_boundary_recoveries_;
      RCLCPP_WARN(
          get_logger(),
          "RACER_RL_BOUNDARY_LATE next_boundary_slot=%llu "
          "observed_slot=%llu late_slots=%llu recoveries=%llu",
          static_cast<unsigned long long>(next_boundary_slot),
          static_cast<unsigned long long>(observed_slot),
          static_cast<unsigned long long>(observed_slot - next_boundary_slot),
          static_cast<unsigned long long>(rl_late_boundary_recoveries_));
    }
    while (rl_communication_slot_index_ < observed_slot) {
      const auto next_slot = rl_communication_slot_index_ + 1U;
      if (next_slot % rl_slots_per_decision_ == 0U) {
        rl_communication_slot_index_ = next_slot;
        rl_decision_index_ = next_slot / rl_slots_per_decision_;
        rl_last_action_held_slots_ = rl_current_action_held_slots_;
        rl_current_action_held_slots_ = 0U;
        rl_completed_interval_available_ = true;
        rl_pending_boundary_stamp_s_ =
            static_cast<double>(next_slot) * rl_bs_communication_slot_s_;
        rl_async_observed_slot_ = observed_slot;
        return true;
      }
      rl_communication_slot_index_ = next_slot;
      if (rl_bs_repeat_action_each_slot_) {
        scheduleActiveRlAction(
            static_cast<double>(next_slot) * rl_bs_communication_slot_s_);
        ++rl_current_action_held_slots_;
      }
    }
    return false;
  }

  void completeRlDecisionBoundary() {
    activateLatestRlAction(rl_pending_boundary_stamp_s_);
    scheduleActiveRlAction(rl_pending_boundary_stamp_s_);
    ++rl_current_action_held_slots_;

    // A 2 ms wall callback normally sees at most one 20 ms simulated slot.
    // Catch up extra non-boundary slots here. If another boundary is already
    // due, leave it for the immediately following callback so each boundary
    // still receives its own ring record rather than collapsing two samples.
    while (rl_communication_slot_index_ < rl_async_observed_slot_) {
      const auto next_slot = rl_communication_slot_index_ + 1U;
      if (next_slot % rl_slots_per_decision_ == 0U) {
        break;
      }
      rl_communication_slot_index_ = next_slot;
      if (rl_bs_repeat_action_each_slot_) {
        scheduleActiveRlAction(
            static_cast<double>(next_slot) * rl_bs_communication_slot_s_);
        ++rl_current_action_held_slots_;
      }
    }
  }

  void applyHeldRlAction(double stamp, std::uint64_t decision_index) {
    if (!bs_all_to_all_relay_enabled_ &&
        (!rl_have_action_ || rl_action_decision_index_ != decision_index)) {
      readRlAction(decision_index);
    }
    if (!bs_all_to_all_relay_enabled_ &&
        (!rl_have_action_ || rl_action_decision_index_ != decision_index)) {
      ++rl_missing_action_slots_;
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "missing synchronous RL action at decision=%llu comm_slot=%llu",
          static_cast<unsigned long long>(decision_index),
          static_cast<unsigned long long>(rl_communication_slot_index_ + 1U));
      return;
    }
    if (rl_have_action_ && !rl_action_channel_snapshot_) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "refusing to schedule synchronous RL action %llu without its "
          "channel snapshot",
          static_cast<unsigned long long>(rl_action_epoch_));
      return;
    }
    scheduleRlActionWithinSlotBudgets(stamp);
  }

  bool advanceSynchronousRlClock(double stamp) {
    constexpr double kTolerance = 1.0e-9;
    if (!rl_sync_clock_initialized_) {
      if (!std::all_of(
              uav_position_valid_.begin(), uav_position_valid_.end(),
              [](bool valid) { return valid; })) {
        return false;
      }
      rl_sync_clock_initialized_ = true;
      rl_last_sync_stamp_s_ = stamp;
      rl_sync_boundary_frozen_ = true;
      // The Isaac boundary gate keeps /clock at zero until this initial state
      // has been consumed and action a_0 is atomically committed.
      if (std::abs(stamp) > kTolerance) {
        throw std::runtime_error(
            "synchronous RL communication proxy did not start at sim time 0");
      }
      return true;
    }
    if (stamp + kTolerance < rl_last_sync_stamp_s_) {
      throw std::runtime_error("simulation time moved backwards in RL scheduler");
    }
    if (stamp > rl_last_sync_stamp_s_ + kTolerance) {
      rl_sync_boundary_frozen_ = false;
    }
    rl_last_sync_stamp_s_ = stamp;
    bool state_due = false;
    while (stamp + kTolerance >=
           static_cast<double>(rl_communication_slot_index_ + 1U) *
               rl_bs_communication_slot_s_) {
      if (state_due) {
        // Isaac must freeze at the first crossed 100 ms boundary. Reaching a
        // second interval here would mean states were skipped.
        throw std::runtime_error(
            "synchronous RL crossed a decision boundary before Isaac paused");
      }
      const std::uint64_t interval_decision =
          rl_communication_slot_index_ / rl_slots_per_decision_;
      const bool first_slot_in_decision =
          rl_communication_slot_index_ % rl_slots_per_decision_ == 0U;
      if (rl_bs_repeat_action_each_slot_ || first_slot_in_decision) {
        applyHeldRlAction(stamp, interval_decision);
        ++rl_current_action_held_slots_;
      }
      ++rl_communication_slot_index_;
      if (rl_communication_slot_index_ % rl_slots_per_decision_ == 0U) {
        rl_decision_index_ =
            rl_communication_slot_index_ / rl_slots_per_decision_;
        rl_last_action_held_slots_ = rl_current_action_held_slots_;
        rl_current_action_held_slots_ = 0U;
        rl_sync_boundary_frozen_ = true;
        state_due = true;
      }
    }
    return state_due;
  }

  std::size_t missingBytes(
      int sender,
      const std::unordered_set<ChunkKey, ChunkKeyHash> &known) const {
    std::size_t bytes{};
    const auto &source = uav_chunks_[static_cast<std::size_t>(sender)];
    for (const auto &key : source) {
      if (known.find(key) != known.end()) continue;
      const auto found = chunk_repository_.find(key);
      if (found != chunk_repository_.end()) bytes += found->second.bytes;
    }
    return bytes;
  }

  std::size_t missingBytes(
      const std::unordered_set<ChunkKey, ChunkKeyHash> &available,
      const std::unordered_set<ChunkKey, ChunkKeyHash> &known) const {
    std::size_t bytes{};
    for (const auto &key : available) {
      if (known.find(key) != known.end()) continue;
      const auto found = chunk_repository_.find(key);
      if (found != chunk_repository_.end()) bytes += found->second.bytes;
    }
    return bytes;
  }

  std::size_t missingChunkCount(
      int sender,
      const std::unordered_set<ChunkKey, ChunkKeyHash> &known) const {
    std::size_t chunks{};
    const auto &source = uav_chunks_[static_cast<std::size_t>(sender)];
    for (const auto &key : source) {
      if (known.find(key) != known.end()) continue;
      if (chunk_repository_.find(key) != chunk_repository_.end()) ++chunks;
    }
    return chunks;
  }

  std::size_t queuedRouteBytes(const LinkKey &link, RouteStage route,
                               int action_sender = -1) const {
    const auto found = queues_.find(link);
    if (found == queues_.end()) return 0U;
    std::size_t bytes{};
    for (const auto &packet : found->second.packets) {
      if (packet.route != route) continue;
      if (action_sender >= 0) {
        if (!packet.has_chunk_key) continue;
        const int attributed_sender =
            route == RouteStage::kBsUplink
                ? packet.current_sender
                : (packet.action_sender >= 0
                       ? packet.action_sender
                       : packet.chunk_key.owner - 1);
        if (attributed_sender != action_sender) continue;
      }
      bytes += packet.bytes;
    }
    return bytes;
  }

  void validateIncrementalCaches() const {
    std::size_t mismatches{};
    for (int sender = 0; sender < drone_count_; ++sender) {
      const auto sender_index = static_cast<std::size_t>(sender);
      if (cached_bs_missing_bytes_[sender_index] !=
          missingBytes(sender, bs_chunks_)) {
        ++mismatches;
      }
      if (cached_bs_missing_chunks_[sender_index] !=
          missingChunkCount(sender, bs_chunks_)) {
        ++mismatches;
      }
      if (cached_bs_uav_missing_bytes_[sender_index] != missingBytes(
              bs_chunks_, uav_chunks_[sender_index])) {
        ++mismatches;
      }
      if (cached_uplink_queue_bytes_[sender_index] != queuedRouteBytes(
              {sender, ap_node_id_}, RouteStage::kBsUplink, sender)) {
        ++mismatches;
      }
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        const auto receiver_index = static_cast<std::size_t>(receiver);
        const auto pair_expected =
            sender == receiver
                ? 0U
                : missingBytes(sender, uav_chunks_[receiver_index]);
        if (cached_pair_missing_bytes_[sender_index][receiver_index] !=
            pair_expected) {
          ++mismatches;
        }
        const auto pair_chunks_expected =
            sender == receiver
                ? 0U
                : missingChunkCount(sender, uav_chunks_[receiver_index]);
        if (cached_pair_missing_chunks_[sender_index][receiver_index] !=
            pair_chunks_expected) {
          ++mismatches;
        }
        const auto relay_expected =
            sender == receiver
                ? 0U
                : queuedRouteBytes({ap_node_id_, receiver},
                                   RouteStage::kBsDownlink, sender);
        if (cached_relay_queue_bytes_[sender_index][receiver_index] !=
            relay_expected) {
          ++mismatches;
        }
      }
    }
    RCLCPP_INFO(get_logger(),
                "RACER_INCREMENTAL_CACHE_VALIDATION mismatches=%zu",
                mismatches);
  }

  double telemetrySnr(int sender, int receiver, double stamp) const {
    if (sender == receiver) return 0.0;
    if (force_bs_perfect_delivery_ && ap_enabled_ &&
        (sender == ap_node_id_ || receiver == ap_node_id_)) {
      return 40.0;
    }
    const auto *link = linkFor({sender, receiver}, stamp);
    if (!usable(link)) return -120.0;
    const bool bs_link = ap_enabled_ &&
                         (sender == ap_node_id_ || receiver == ap_node_id_);
    return radioSnrDb(link, bs_link ? bs_model_ : directRadioModel());
  }

  bool telemetryLinkAvailable(int sender, int receiver, double stamp) const {
    if (sender == receiver) return true;
    if (force_bs_perfect_delivery_ && ap_enabled_ &&
        (sender == ap_node_id_ || receiver == ap_node_id_)) {
      return true;
    }
    return usable(linkFor({sender, receiver}, stamp));
  }

  void rememberActionChannelSnapshot(
      const std::shared_ptr<ActionChannelSnapshot> &snapshot) {
    if (!snapshot) return;
    action_channel_snapshots_[snapshot->channel_version] = snapshot;
    action_channel_snapshot_versions_.push_back(snapshot->channel_version);
    constexpr std::size_t kSnapshotCacheLimit = 4096U;
    while (action_channel_snapshot_versions_.size() > kSnapshotCacheLimit) {
      const auto oldest = action_channel_snapshot_versions_.front();
      action_channel_snapshot_versions_.pop_front();
      action_channel_snapshots_.erase(oldest);
    }
  }

  std::shared_ptr<const ActionChannelSnapshot> actionChannelSnapshot(
      std::uint64_t channel_version) const {
    const auto found = action_channel_snapshots_.find(channel_version);
    return found == action_channel_snapshots_.end() ? nullptr : found->second;
  }

  std::shared_ptr<const ActionChannelSnapshot> actionChannelSnapshotForStep(
      std::uint64_t step_id) const {
    for (auto iterator = action_channel_snapshot_versions_.rbegin();
         iterator != action_channel_snapshot_versions_.rend(); ++iterator) {
      const auto snapshot = actionChannelSnapshot(*iterator);
      if (snapshot && snapshot->step_id == step_id) return snapshot;
    }
    return nullptr;
  }

  template <typename Value>
  static bool readBinaryValue(const std::string &payload, std::size_t &offset,
                              Value &value) {
    if (offset + sizeof(Value) > payload.size()) return false;
    std::memcpy(&value, payload.data() + offset, sizeof(Value));
    offset += sizeof(Value);
    return true;
  }

  template <typename Value>
  static void appendBinaryValue(std::string &payload, const Value &value) {
    payload.append(reinterpret_cast<const char *>(&value), sizeof(Value));
  }

  std::shared_ptr<FastBoundarySnapshot> captureFastRlState(double stamp) {
    const auto started = std::chrono::steady_clock::now();
    auto state = std::make_shared<FastBoundarySnapshot>();
    state->step_id = rl_bs_event_driven_one_shot_
        ? rl_decision_index_
        : static_cast<std::uint64_t>(std::llround(
              std::max(0.0, stamp) / rl_bs_decision_period_s_));
    state->sim_time_s = rl_bs_event_driven_one_shot_
        ? stamp
        : static_cast<double>(state->step_id) * rl_bs_decision_period_s_;
    state->sequence = ++rl_state_sequence_;
    state->communication_slot = rl_communication_slot_index_;
    state->decision_index = rl_decision_index_;
    state->interval_start_sim_time_s = rl_bs_event_driven_one_shot_
        ? rl_event_interval_start_sim_time_s_
        : (state->step_id == 0U
               ? 0.0
               : static_cast<double>(state->step_id - 1U) *
                     rl_bs_decision_period_s_);
    state->interval_duration_s = rl_bs_event_driven_one_shot_
        ? rl_event_delta_t_s_
        : rl_bs_decision_period_s_;
    state->action_version = rl_action_epoch_;
    state->policy_version = rl_action_policy_version_;
    state->action_generated_wall_time_ns =
        rl_action_generated_wall_time_ns_;
    state->action_source_guidance_id = rl_action_source_guidance_id_;
    state->action_source_physical_version =
        rl_action_source_physical_version_;
    state->action_source_communication_version =
        rl_action_source_communication_version_;
    state->action_source_step_id = rl_action_source_step_id_;
    state->action_source_sim_time_s = rl_action_source_sim_time_s_;
    state->action_age = rl_bs_event_driven_one_shot_
        ? (rl_completed_interval_available_ ? rl_event_action_age_s_ : 0.0)
        : (rl_completed_interval_available_ && rl_have_action_
               ? std::max(0.0, state->interval_start_sim_time_s -
                                   rl_action_source_sim_time_s_)
               : 0.0);
    state->action_held_slots = rl_last_action_held_slots_;
    state->completed_transition = rl_completed_interval_available_;
    state->interval_uplink_prb_slots = static_cast<double>(
        bs_uplink_prb_slots_ - rl_last_state_bs_uplink_prb_slots_);
    state->interval_downlink_prb_slots = static_cast<double>(
        bs_downlink_prb_slots_ - rl_last_state_bs_downlink_prb_slots_);
    state->interval_direct_prb_slots = static_cast<double>(
        direct_u2u_prb_slots_ - rl_last_state_direct_u2u_prb_slots_);

    const auto n = static_cast<std::size_t>(drone_count_);
    state->positions.reserve(n);
    state->velocities.reserve(n);
    state->yaws.reserve(n);
    state->channel.reserve((n + 1U) * (n + 1U));
    auto channel_snapshot = std::make_shared<ActionChannelSnapshot>();
    channel_snapshot->channel_version = state->sequence;
    channel_snapshot->step_id = state->step_id;
    channel_snapshot->sim_time_s = state->sim_time_s;
    channel_snapshot->effective_snr_db.reserve(
        (n + 1U) * (n + 1U));
    channel_snapshot->available.reserve((n + 1U) * (n + 1U));
    state->pair_aoi.reserve(n * n);
    state->bs_aoi.reserve(n);
    state->pair_missing_bytes.reserve(n * n);
    state->relay_queue_bytes.reserve(n * n);
    state->relay_action.reserve(n * n);
    for (int drone = 0; drone < drone_count_; ++drone) {
      const auto &position = uav_positions_[static_cast<std::size_t>(drone)];
      const auto &velocity = uav_velocities_[static_cast<std::size_t>(drone)];
      state->positions.push_back(
          {static_cast<float>(position[0]), static_cast<float>(position[1]),
           static_cast<float>(position[2])});
      state->velocities.push_back(
          {static_cast<float>(velocity[0]), static_cast<float>(velocity[1]),
           static_cast<float>(velocity[2])});
      state->yaws.push_back(
          static_cast<float>(uav_yaws_[static_cast<std::size_t>(drone)]));
    }
    for (int sender = 0; sender < radio_node_count_; ++sender) {
      for (int receiver = 0; receiver < radio_node_count_; ++receiver) {
        const auto effective_snr =
            static_cast<float>(telemetrySnr(sender, receiver, stamp));
        state->channel.push_back(effective_snr);
        channel_snapshot->effective_snr_db.push_back(effective_snr);
        channel_snapshot->available.push_back(
            telemetryLinkAvailable(sender, receiver, stamp) ? 1U : 0U);
      }
    }
    rememberActionChannelSnapshot(channel_snapshot);
    for (int sender = 0; sender < drone_count_; ++sender) {
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        const auto sender_index = static_cast<std::size_t>(sender);
        const auto receiver_index = static_cast<std::size_t>(receiver);
        state->pair_aoi.push_back(
            sender == receiver
                ? 0.0F
                : static_cast<float>(std::max(
                      0.0, stamp - last_uav_info_received_s_[sender_index]
                                                            [receiver_index])));
        state->pair_missing_bytes.push_back(
            cached_pair_missing_bytes_[sender_index][receiver_index]);
        state->relay_queue_bytes.push_back(
            cached_relay_queue_bytes_[sender_index][receiver_index]);
        state->relay_action.push_back(
            rl_relay_action_[sender_index][receiver_index] ? 1U : 0U);
      }
      const auto sender_index = static_cast<std::size_t>(sender);
      state->bs_aoi.push_back(static_cast<float>(std::max(
          0.0, stamp - last_bs_info_received_s_[sender_index])));
      state->bs_missing_bytes.push_back(cached_bs_missing_bytes_[sender_index]);
      state->bs_uav_missing_bytes.push_back(
          cached_bs_uav_missing_bytes_[sender_index]);
      state->uplink_queue_bytes.push_back(
          cached_uplink_queue_bytes_[sender_index]);
      state->upload_action.push_back(
          rl_upload_action_[sender_index] ? 1U : 0U);
    }
    state->bs_known_chunks = static_cast<std::uint64_t>(bs_chunks_.size());

    // Capture compact simulator scalars at the boundary. This is a bounded
    // binary copy; the large physical JSON remains exclusively on the LLM
    // path.
    SharedBlockSnapshot physical;
    if (shared_physical_fast_ && shared_physical_fast_->snapshot(physical)) {
      std::size_t offset = 0U;
      char magic[8]{};
      std::uint32_t abi{}, physical_n{};
      std::uint64_t sim_step{};
      double sim_time{}, coverage{}, coverage_delta{};
      std::uint8_t terminated{}, truncated{};
      if (physical.payload.size() >= sizeof(magic)) {
        std::memcpy(magic, physical.payload.data(), sizeof(magic));
        offset += sizeof(magic);
      }
      if (std::memcmp(magic, "FRPHY01", 7) == 0 &&
          readBinaryValue(physical.payload, offset, abi) && abi == 1U &&
          readBinaryValue(physical.payload, offset, physical_n) &&
          physical_n == static_cast<std::uint32_t>(drone_count_) &&
          readBinaryValue(physical.payload, offset, sim_step) &&
          readBinaryValue(physical.payload, offset, sim_time) &&
          readBinaryValue(physical.payload, offset, coverage) &&
          readBinaryValue(physical.payload, offset, coverage_delta) &&
          readBinaryValue(physical.payload, offset, terminated) &&
          readBinaryValue(physical.payload, offset, truncated)) {
        state->physical_version = physical.version;
        state->coverage = coverage;
        state->coverage_delta = coverage_delta;
        // The simulator and communication proxy advance independently.  A
        // loaded proxy can be several simulated seconds behind while the
        // latest physical block already contains Isaac's terminal flag.  Do
        // not attach that future flag to an earlier RL transition.  One RL
        // decision interval accommodates the normal 100 Hz/10 Hz boundary
        // skew, including Isaac's final physics step at duration + dt.
        const bool terminal_time_aligned =
            sim_time <= state->sim_time_s +
                (rl_bs_event_driven_one_shot_
                     ? rl_bs_communication_slot_s_
                     : rl_bs_decision_period_s_) +
                1.0e-9;
        state->terminated = terminal_time_aligned && terminated != 0U;
        state->truncated = terminal_time_aligned && truncated != 0U;
      }
    }

    state->guidance_task_dependency.assign(n * n, 0.0F);
    state->guidance_semantic_importance.assign(
        n, 1.0F / static_cast<float>(n));
    SharedBlockSnapshot guidance;
    if (shared_guidance_fast_ && shared_guidance_fast_->snapshot(guidance)) {
      std::size_t offset = 0U;
      char magic[8]{};
      std::uint32_t abi{}, guidance_n{};
      std::uint64_t guidance_id{};
      if (guidance.payload.size() >= sizeof(magic)) {
        std::memcpy(magic, guidance.payload.data(), sizeof(magic));
        offset += sizeof(magic);
      }
      const auto value_count = n * n + n;
      if (std::memcmp(magic, "FRGDN01", 7) == 0 &&
          readBinaryValue(guidance.payload, offset, abi) && abi == 1U &&
          readBinaryValue(guidance.payload, offset, guidance_n) &&
          guidance_n == static_cast<std::uint32_t>(drone_count_) &&
          readBinaryValue(guidance.payload, offset, guidance_id) &&
          offset + value_count * sizeof(float) == guidance.payload.size()) {
        state->guidance_id = guidance_id;
        std::memcpy(state->guidance_task_dependency.data(),
                    guidance.payload.data() + offset, n * n * sizeof(float));
        offset += n * n * sizeof(float);
        std::memcpy(state->guidance_semantic_importance.data(),
                    guidance.payload.data() + offset, n * sizeof(float));
      }
    }
    state->build_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    fast_rl_state_build_total_ms_ += state->build_ms;
    ++fast_rl_state_build_count_;
    fast_rl_state_build_max_ms_ =
        std::max(fast_rl_state_build_max_ms_, state->build_ms);
    state->build_mean_ms = fast_rl_state_build_total_ms_ /
                           static_cast<double>(fast_rl_state_build_count_);
    state->build_max_ms = fast_rl_state_build_max_ms_;
    state->missing_update_mean_ms = missing_bytes_update_total_ms_ /
        std::max<std::uint64_t>(1U, missing_bytes_update_count_);
    state->queue_update_mean_ms = queue_cache_update_total_ms_ /
        std::max<std::uint64_t>(1U, queue_cache_update_count_);
    return state;
  }

  void enqueueFastRlState(const std::shared_ptr<FastBoundarySnapshot> &state) {
    {
      std::lock_guard<std::mutex> lock(fast_state_queue_mutex_);
      fast_state_queue_.push_back(state);
    }
    enqueueTaskObservation(
        {TaskObservationKind::kBoundary, -1, nullptr, nullptr, state});
  }

  std::string encodeFastRlState(const FastBoundarySnapshot &state) const {
    std::string payload;
    const auto n = static_cast<std::size_t>(drone_count_);
    payload.reserve(256U + 4U * (3U * n + (n + 1U) * (n + 1U) +
                                n * n + n + n * n + n) +
                    8U * (2U * n * n + 2U * n) + n * n + n);
    payload.append("FRFAST1", 7U);
    payload.push_back('\0');
    appendBinaryValue(payload, std::uint32_t{1U});
    appendBinaryValue(payload, static_cast<std::uint32_t>(drone_count_));
    appendBinaryValue(payload, state.step_id);
    appendBinaryValue(payload, state.sequence);
    appendBinaryValue(payload, state.communication_slot);
    appendBinaryValue(payload, state.decision_index);
    appendBinaryValue(payload, state.sim_time_s);
    appendBinaryValue(payload, rl_bs_communication_slot_s_);
    appendBinaryValue(payload, state.interval_duration_s);
    appendBinaryValue(payload, state.action_version);
    appendBinaryValue(payload, state.policy_version);
    appendBinaryValue(payload, state.action_generated_wall_time_ns);
    appendBinaryValue(payload, state.action_source_guidance_id);
    appendBinaryValue(payload, state.action_source_physical_version);
    appendBinaryValue(payload, state.action_source_communication_version);
    appendBinaryValue(payload, state.action_source_step_id);
    appendBinaryValue(payload, state.action_source_sim_time_s);
    appendBinaryValue(payload, state.action_age);
    appendBinaryValue(payload, state.action_held_slots);
    appendBinaryValue(payload, state.physical_version);
    appendBinaryValue(payload, state.coverage);
    appendBinaryValue(payload, state.coverage_delta);
    appendBinaryValue(payload, state.interval_uplink_prb_slots);
    appendBinaryValue(payload, state.interval_downlink_prb_slots);
    appendBinaryValue(payload, state.interval_direct_prb_slots);
    payload.push_back(state.terminated ? 1 : 0);
    payload.push_back(state.truncated ? 1 : 0);
    payload.push_back(state.completed_transition ? 1 : 0);
    payload.push_back(0);
    appendBinaryValue(payload, state.guidance_id);
    for (const auto &position : state.positions) {
      for (const auto value : position) appendBinaryValue(payload, value);
    }
    for (const auto value : state.channel) appendBinaryValue(payload, value);
    for (const auto value : state.pair_aoi) appendBinaryValue(payload, value);
    for (const auto value : state.bs_aoi) appendBinaryValue(payload, value);
    for (const auto value : state.pair_missing_bytes)
      appendBinaryValue(payload, value);
    for (const auto value : state.bs_missing_bytes)
      appendBinaryValue(payload, value);
    for (const auto value : state.uplink_queue_bytes)
      appendBinaryValue(payload, value);
    for (const auto value : state.relay_queue_bytes)
      appendBinaryValue(payload, value);
    payload.append(reinterpret_cast<const char *>(state.relay_action.data()),
                   state.relay_action.size());
    payload.append(reinterpret_cast<const char *>(state.upload_action.data()),
                   state.upload_action.size());
    for (const auto value : state.guidance_task_dependency)
      appendBinaryValue(payload, value);
    for (const auto value : state.guidance_semantic_importance)
      appendBinaryValue(payload, value);
    return payload;
  }

  void publishFastRlState() {
    std::shared_ptr<FastBoundarySnapshot> state;
    {
      std::lock_guard<std::mutex> lock(fast_state_queue_mutex_);
      if (fast_state_queue_.empty()) return;
      state = std::move(fast_state_queue_.front());
      fast_state_queue_.pop_front();
    }
    const auto started = std::chrono::steady_clock::now();
    const auto ring_sequence = shared_transition_ring_->publish(
        encodeFastRlState(*state), state->step_id,
        state->sim_time_s);
    const double elapsed_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    fast_rl_state_publish_total_ms_ += elapsed_ms;
    ++fast_rl_state_publish_count_;
    fast_rl_state_publish_max_ms_ =
        std::max(fast_rl_state_publish_max_ms_, elapsed_ms);
    RCLCPP_INFO(
        get_logger(),
        "RACER_FAST_RL_STATE step_id=%llu sim_time=%.9f ring_sequence=%llu "
        "fast_rl_state_build_ms=%.6f fast_rl_state_build_mean_ms=%.6f "
        "fast_rl_state_build_max_ms=%.6f fast_rl_state_publish_ms=%.6f "
        "fast_rl_state_publish_mean_ms=%.6f "
        "fast_rl_state_publish_max_ms=%.6f "
        "missing_bytes_update_ms=%.6f queue_cache_update_ms=%.6f",
        static_cast<unsigned long long>(state->step_id), state->sim_time_s,
        static_cast<unsigned long long>(ring_sequence),
        state->build_ms, state->build_mean_ms, state->build_max_ms,
        elapsed_ms,
        fast_rl_state_publish_total_ms_ /
            static_cast<double>(fast_rl_state_publish_count_),
        fast_rl_state_publish_max_ms_,
        state->missing_update_mean_ms, state->queue_update_mean_ms);
  }

  void writeRlState(double stamp) {
    if (!rl_bs_scheduler_enabled_) return;
    if (shared_transition_ring_) {
      const auto state = captureFastRlState(stamp);
      enqueueFastRlState(state);
      rl_last_state_bs_uplink_prb_slots_ = bs_uplink_prb_slots_;
      rl_last_state_bs_downlink_prb_slots_ = bs_downlink_prb_slots_;
      rl_last_state_direct_u2u_prb_slots_ = direct_u2u_prb_slots_;
      return;
    }
    writeLegacyRlState(stamp);
  }

  // Compatibility-only JSON/file bridge. This path intentionally retains the
  // historical diagnostic fields and validation scans. Shared-memory training
  // returns above and can never execute any whole-map/task-metric traversal.
  void writeLegacyRlState(double stamp) {
    ++rl_state_sequence_;
    const auto task_step = static_cast<std::uint64_t>(std::llround(
        std::max(0.0, stamp) / rl_bs_decision_period_s_));
    const double task_time_s =
        static_cast<double>(task_step) * rl_bs_decision_period_s_;
    auto channel_snapshot = std::make_shared<ActionChannelSnapshot>();
    channel_snapshot->channel_version = rl_state_sequence_;
    channel_snapshot->step_id = task_step;
    channel_snapshot->sim_time_s = task_time_s;
    const auto channel_entries = static_cast<std::size_t>(radio_node_count_) *
                                 static_cast<std::size_t>(radio_node_count_);
    channel_snapshot->effective_snr_db.reserve(channel_entries);
    channel_snapshot->available.reserve(channel_entries);
    for (int sender = 0; sender < radio_node_count_; ++sender) {
      for (int receiver = 0; receiver < radio_node_count_; ++receiver) {
        channel_snapshot->effective_snr_db.push_back(
            static_cast<float>(telemetrySnr(sender, receiver, stamp)));
        channel_snapshot->available.push_back(
            telemetryLinkAvailable(sender, receiver, stamp) ? 1U : 0U);
      }
    }
    rememberActionChannelSnapshot(channel_snapshot);
    const double interval_start_sim_time_s =
        rl_decision_index_ == 0U
            ? 0.0
            : static_cast<double>(rl_decision_index_ - 1U) *
                  rl_bs_decision_period_s_;
    const double executed_action_age_s =
        rl_completed_interval_available_ && rl_have_action_
            ? std::max(0.0, interval_start_sim_time_s -
                                rl_action_source_sim_time_s_)
            : 0.0;
    const auto bs_map_metrics = bsGlobalMapMetrics();
    std::ostringstream json;
    // Preserve the canonical task clock even when a test or non-simulated ROS
    // source supplies an epoch-sized timestamp.  The default stream precision
    // would otherwise round task_time_s enough to break task_step alignment.
    json << std::setprecision(std::numeric_limits<double>::max_digits10);
    json << "{\"sequence\":" << rl_state_sequence_
         << ",\"channel_version\":" << rl_state_sequence_
         << ",\"racer_version\":" << rl_state_sequence_
         << ",\"map_version\":" << rl_state_sequence_
         << ",\"task_step\":" << task_step
         << ",\"task_time_s\":" << task_time_s
         << ",\"task_step_duration_s\":" << rl_bs_decision_period_s_
         << ",\"action_epoch\":" << rl_action_epoch_
         << ",\"sim_time_s\":" << stamp
         << ",\"sim_timestamp\":" << stamp
         << ",\"synchronous_online\":"
         << (rl_bs_synchronous_mode_ ? "true" : "false")
         << ",\"communication_slot_duration_s\":"
         << rl_bs_communication_slot_s_
         << ",\"rl_decision_interval_s\":"
         << rl_bs_decision_period_s_
         << ",\"slots_per_decision\":" << rl_slots_per_decision_
         << ",\"communication_slot_index\":"
         << rl_communication_slot_index_
         << ",\"rl_decision_index\":" << rl_decision_index_
         << ",\"action_decision_index\":"
         << (rl_have_action_
                 ? static_cast<long long>(rl_action_decision_index_)
                 : -1LL)
         << ",\"action_held_slots\":" << rl_last_action_held_slots_
         << ",\"has_completed_transition\":"
         << (rl_completed_interval_available_ ? "true" : "false")
         << ",\"action_version\":" << rl_action_epoch_
         << ",\"policy_version\":" << rl_action_policy_version_
         << ",\"action_age\":" << executed_action_age_s
         << ",\"action_generated_wall_time_ns\":"
         << rl_action_generated_wall_time_ns_
         << ",\"action_source_sim_time_s\":"
         << rl_action_source_sim_time_s_
         << ",\"action_source_physical_version\":"
         << rl_action_source_physical_version_
         << ",\"action_source_communication_version\":"
         << rl_action_source_communication_version_
         << ",\"action_source_guidance_id\":"
         << rl_action_source_guidance_id_
         << ",\"interval_start_sim_time_s\":"
         << interval_start_sim_time_s
         << ",\"interval_end_sim_time_s\":" << stamp
         << ",\"executed_relay_action\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        json << (rl_relay_action_[static_cast<std::size_t>(owner)]
                                  [static_cast<std::size_t>(receiver)]
                     ? 1
                     : 0);
      }
      json << ']';
    }
    json << "],\"executed_upload_action\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << (rl_upload_action_[static_cast<std::size_t>(owner)] ? 1 : 0);
    }
    json << "]"
         << ",\"positions\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      const auto &value = uav_positions_[static_cast<std::size_t>(drone)];
      json << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
    }
    json << "],\"velocities\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      const auto &value = uav_velocities_[static_cast<std::size_t>(drone)];
      json << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
    }
    json << "],\"yaws\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      json << uav_yaws_[static_cast<std::size_t>(drone)];
    }
    json << "],\"fsm_states\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      json << (uav_position_valid_[static_cast<std::size_t>(drone)]
                   ? "\"EXPLORE\""
                   : "\"UNKNOWN\"");
    }
    json << "],\"channel_snr_db\":[";
    for (int sender = 0; sender < radio_node_count_; ++sender) {
      if (sender) json << ',';
      json << '[';
      for (int receiver = 0; receiver < radio_node_count_; ++receiver) {
        if (receiver) json << ',';
        json << telemetrySnr(sender, receiver, stamp);
      }
      json << ']';
    }
    json << "],\"pair_aoi_s\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        const double aoi = owner == receiver
                               ? 0.0
                               : std::max(0.0, stamp - last_uav_info_received_s_
                                                        [static_cast<std::size_t>(owner)]
                                                        [static_cast<std::size_t>(receiver)]);
        json << aoi;
      }
      json << ']';
    }
    json << "],\"bs_aoi_s\":[";
    for (int owner = 0; owner < drone_count_; ++owner) {
      if (owner) json << ',';
      json << std::max(0.0, stamp - last_bs_info_received_s_
                                        [static_cast<std::size_t>(owner)]);
    }
    json << "],\"pair_missing_bytes\":[";
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (sender) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        json << (sender == receiver
                     ? 0U
                     : missingBytes(sender, uav_chunks_[static_cast<std::size_t>(receiver)]));
      }
      json << ']';
    }
    json << "],\"bs_missing_bytes\":[";
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (sender) json << ',';
      json << missingBytes(sender, bs_chunks_);
    }
    json << "],\"uplink_queue_bytes\":[";
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (sender) json << ',';
      json << queuedRouteBytes({sender, ap_node_id_}, RouteStage::kBsUplink,
                               sender);
    }
    json << "],\"relay_queue_bytes\":[";
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (sender) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        json << (sender == receiver
                     ? 0U
                     : queuedRouteBytes({ap_node_id_, receiver},
                                        RouteStage::kBsDownlink, sender));
      }
      json << ']';
    }
    json << "],\"information_version_gap\":[";
    for (int sender = 0; sender < drone_count_; ++sender) {
      if (sender) json << ',';
      json << '[';
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        if (receiver) json << ',';
        std::size_t count{};
        for (const auto &key : uav_chunks_[static_cast<std::size_t>(sender)]) {
          if (uav_chunks_[static_cast<std::size_t>(receiver)].find(key) ==
                  uav_chunks_[static_cast<std::size_t>(receiver)].end()) {
            ++count;
          }
        }
        json << (sender == receiver ? 0U : count);
      }
      std::size_t bs_count{};
      for (const auto &key : uav_chunks_[static_cast<std::size_t>(sender)]) {
        if (bs_chunks_.find(key) == bs_chunks_.end()) {
          ++bs_count;
        }
      }
      json << ',' << bs_count << ']';
    }
    const auto trajectory_summaries = trajectorySummariesSnapshot();
    json << "],\"trajectory_summary\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      const auto &summary =
          trajectory_summaries[static_cast<std::size_t>(drone)];
      const auto &goal = summary.valid
                             ? summary.goal
                             : uav_positions_[static_cast<std::size_t>(drone)];
      json << "{\"goal_position\":[" << goal[0] << ',' << goal[1] << ','
           << goal[2] << "],\"trajectory_length\":" << summary.length
           << ",\"expected_execution_time\":"
           << summary.expected_execution_time << '}';
    }
    json << "],\"map_summary\":{\"representation\":"
            "\"chunk_version_counts\",\"bs_known_chunks\":"
         << bs_chunks_.size() << ",\"local_known_chunks\":[";
    for (int drone = 0; drone < drone_count_; ++drone) {
      if (drone) json << ',';
      json << uav_chunks_[static_cast<std::size_t>(drone)].size();
    }
    json << "]},\"coverage\":0.0"
         << ",\"redundant_exploration_ratio\":"
         << redundantExplorationRatio()
         << ",\"bs_global_map_iou\":" << bs_map_metrics.iou
         << ",\"bs_global_map_coverage\":" << bs_map_metrics.coverage
         << ",\"bs_global_map_known_voxels\":" << bs_map_metrics.known
         << ",\"bs_global_map_total_voxels\":" << bs_map_metrics.total
         << ",\"bs_global_map_occupied_voxels\":"
         << bs_map_metrics.occupied
         << ",\"bs_global_map_gt_intersection_voxels\":"
         << bs_map_metrics.intersection
         << ",\"bs_uplink_prb_slots\":" << bs_uplink_prb_slots_
         << ",\"bs_downlink_prb_slots\":" << bs_downlink_prb_slots_
         << ",\"direct_u2u_prb_slots\":" << direct_u2u_prb_slots_
         << ",\"interval_bs_uplink_prb_slots\":"
         << (bs_uplink_prb_slots_ - rl_last_state_bs_uplink_prb_slots_)
         << ",\"interval_bs_downlink_prb_slots\":"
         << (bs_downlink_prb_slots_ - rl_last_state_bs_downlink_prb_slots_)
         << ",\"interval_direct_u2u_prb_slots\":"
         << (direct_u2u_prb_slots_ - rl_last_state_direct_u2u_prb_slots_)
         << ",\"task_metric_observer_mode\":\""
         << task_metric_observer_mode_ << "\""
         << ",\"task_metric_observer_pending_events\":"
         << taskObserverQueueDepth()
         << ",\"task_metric_observer_events_enqueued\":"
         << task_observer_events_enqueued_.load()
         << ",\"task_metric_observer_events_processed\":"
         << task_observer_events_processed_.load()
         << '}';
    const std::string state_payload = json.str();
    bool committed = false;
    if (shared_communication_) {
      const auto version = shared_communication_->publish(
          state_payload, rl_communication_slot_index_, stamp);
      std::uint64_t ring_sequence{};
      if (shared_transition_ring_) {
        SharedBlockSnapshot physical;
        std::ostringstream record;
        const bool have_physical =
            shared_physical_ && shared_physical_->snapshot(physical);
        record << "{\"communication_version\":" << version
               << ",\"physical_version\":"
               << (have_physical ? physical.version : 0U)
               << ",\"communication\":" << state_payload
               << ",\"physical\":";
        if (have_physical) {
          record << physical.payload;
        } else {
          record << "{}";
        }
        record << '}';
        ring_sequence = shared_transition_ring_->publish(
            record.str(), rl_communication_slot_index_, stamp);
      }
      shared_state_event_->poke();
      committed = true;
      RCLCPP_INFO(
          get_logger(),
          "RACER_SHM_STATE_PUBLISHED communication_version=%llu "
          "transition_ring_sequence=%llu racer_version=%llu "
          "map_version=%llu sim_step=%llu "
          "sim_time=%.9f",
          static_cast<unsigned long long>(version),
          static_cast<unsigned long long>(ring_sequence),
          static_cast<unsigned long long>(rl_state_sequence_),
          static_cast<unsigned long long>(rl_state_sequence_),
          static_cast<unsigned long long>(rl_communication_slot_index_),
          stamp);
    } else {
      const std::filesystem::path output_path(rl_bs_state_path_);
      if (!output_path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(output_path.parent_path(), error);
      }
      const auto temporary = output_path.string() + ".tmp";
      std::ofstream output(temporary, std::ios::trunc);
      output << state_payload;
      output.close();
      committed = output &&
                  std::rename(temporary.c_str(), output_path.string().c_str()) ==
                      0;
    }
    if (committed) {
      rl_last_state_bs_uplink_prb_slots_ = bs_uplink_prb_slots_;
      rl_last_state_bs_downlink_prb_slots_ = bs_downlink_prb_slots_;
      rl_last_state_direct_u2u_prb_slots_ = direct_u2u_prb_slots_;
    }
  }

  std::shared_ptr<DeliveryFlow> makeCentralFlow(
      int origin, std::size_t topic_index, std::size_t bytes,
      double stamp) const {
    auto flow = std::make_shared<DeliveryFlow>();
    flow->origin_sender = origin;
    flow->topic_index = topic_index;
    flow->born_at = stamp;
    flow->bytes = bytes;
    flow->delivered.assign(static_cast<std::size_t>(drone_count_), false);
    return flow;
  }

  bool periodicBsUploadLinksUsable(int uav, double stamp) {
    if (uav < 0 || uav >= drone_count_) return false;
    const LinkKey downlink{ap_node_id_, uav};
    const LinkKey uplink{uav, ap_node_id_};
    // Keep both directions in the asynchronous Sionna active-link set even
    // while a request is suppressed. Otherwise a missing cached sample could
    // prevent the very refresh that would make the next opportunity usable.
    markActiveLink(downlink.sender, downlink.receiver);
    markActiveLink(uplink.sender, uplink.receiver);
    if (force_bs_perfect_delivery_ || mode_ == "ideal") return true;
    return usable(linkFor(downlink, stamp)) && usable(linkFor(uplink, stamp));
  }

  void schedulePeriodicBsUploadRequests(double stamp) {
    if (!bs_periodic_upload_request_enabled_) return;
    if (!bs_periodic_upload_clock_initialized_) {
      bs_periodic_upload_clock_initialized_ = true;
      bs_next_periodic_upload_request_s_ =
          stamp + bs_periodic_upload_request_period_s_;
    }
    while (stamp + 1.0e-9 >= bs_next_periodic_upload_request_s_) {
      ++bs_periodic_upload_request_rounds_;
      for (int receiver = 0; receiver < drone_count_; ++receiver) {
        ++bs_periodic_upload_request_opportunities_;
        const auto receiver_index = static_cast<std::size_t>(receiver);
        if (bs_periodic_upload_request_pending_[receiver_index]) {
          ++bs_periodic_upload_requests_suppressed_pending_control_;
          continue;
        }
        if (!periodicBsUploadLinksUsable(receiver, stamp)) {
          ++bs_periodic_upload_requests_suppressed_no_link_;
          continue;
        }
        if (bs_max_inflight_chunks_per_uav_ > 0 &&
            bsUploadInflightChunkCount(receiver) >=
                static_cast<std::size_t>(bs_max_inflight_chunks_per_uav_)) {
          ++bs_periodic_upload_requests_suppressed_backpressure_;
          continue;
        }
        if (enqueue(
                ap_node_id_, receiver, 0U, nullptr, nullptr,
                RouteStage::kBsControl, receiver, stamp, nullptr,
                bs_control_bytes_, 0U, false, receiver, 0U,
                bs_periodic_upload_request_rounds_, nullptr, true)) {
          ++bs_periodic_upload_requests_scheduled_;
          bs_periodic_upload_request_pending_[receiver_index] = true;
        } else {
          ++bs_periodic_upload_request_enqueue_failures_;
        }
      }
      bs_next_periodic_upload_request_s_ +=
          bs_periodic_upload_request_period_s_;
    }
  }

  void beginBsTurn(double stamp) {
    if (!bs_round_robin_enabled_ || bs_turn_active_) return;
    bs_active_uav_ = bs_next_uav_;
    bs_next_uav_ = (bs_next_uav_ + 1) % drone_count_;
    bs_turn_active_ = true;
    bs_turn_started_at_ = stamp;
    bs_control_finished_ = false;
    ++bs_round_robin_turns_;
    ++bs_uav_turns_[bs_active_uav_];

    const bool control_enqueued = enqueue(
        ap_node_id_, bs_active_uav_, 0U, nullptr, nullptr,
        RouteStage::kBsControl, bs_active_uav_, stamp, nullptr,
        bs_control_bytes_);
    if (!control_enqueued) bs_control_finished_ = true;

    const auto candidates = sortedChunks(
        bs_chunks_, uav_chunks_[static_cast<std::size_t>(bs_active_uav_)]);
    const std::size_t limit =
        bs_max_downlink_chunks_per_turn_ > 0
            ? std::min(
                  candidates.size(),
                  static_cast<std::size_t>(
                      bs_max_downlink_chunks_per_turn_))
            : candidates.size();
    const auto topic_index = chunkDataTopicIndex();
    for (std::size_t offset = 0; offset < limit; ++offset) {
      const auto &key = candidates[offset];
      const auto &cached = chunk_repository_.at(key);
      auto routed = retargetChunkMessage(
          cached.message, ap_node_id_, bs_active_uav_);
      if (!routed) continue;
      auto flow = makeCentralFlow(key.owner - 1, topic_index, cached.bytes,
                                  stamp);
      if (enqueue(ap_node_id_, bs_active_uav_, topic_index, routed,
                  flow, RouteStage::kBsDownlink, bs_active_uav_, stamp,
                  &key, cached.bytes)) {
        ++bs_missing_chunks_scheduled_downlink_;
      }
    }
  }

  void scheduleBsUpload(double stamp) {
    if (!bs_turn_active_ || bs_active_uav_ < 0) return;
    const auto candidates = sortedChunks(
        uav_chunks_[static_cast<std::size_t>(bs_active_uav_)], bs_chunks_);
    const std::size_t limit =
        bs_max_uplink_chunks_per_turn_ > 0
            ? std::min(
                  candidates.size(),
                  static_cast<std::size_t>(
                      bs_max_uplink_chunks_per_turn_))
            : candidates.size();
    const auto topic_index = chunkDataTopicIndex();
    for (std::size_t offset = 0; offset < limit; ++offset) {
      const auto &key = candidates[offset];
      if (hasPendingBsUplink(key)) continue;
      const auto &cached = chunk_repository_.at(key);
      auto routed = retargetChunkMessage(
          cached.message, bs_active_uav_, ap_node_id_);
      if (!routed) continue;
      auto flow = makeCentralFlow(key.owner - 1, topic_index, cached.bytes,
                                  stamp);
      if (enqueue(bs_active_uav_, ap_node_id_, topic_index, routed,
                  flow, RouteStage::kBsUplink, -1, stamp, &key,
                  cached.bytes, 0U, false, bs_active_uav_)) {
        ++bs_incremental_chunks_scheduled_uplink_;
      }
    }
  }

  bool bsTurnQueuesEmpty() const {
    if (!bs_turn_active_ || bs_active_uav_ < 0) return true;
    const auto downlink = queues_.find({ap_node_id_, bs_active_uav_});
    const auto uplink = queues_.find({bs_active_uav_, ap_node_id_});
    return (downlink == queues_.end() || downlink->second.packets.empty()) &&
           (uplink == queues_.end() || uplink->second.packets.empty());
  }

  void finishBsPacket(const PendingPacket &packet, bool delivered,
                      double stamp) {
    if (packet.route == RouteStage::kBsControl) {
      if (packet.periodic_bs_upload_request) {
        if (packet.final_receiver >= 0 &&
            packet.final_receiver < drone_count_) {
          bs_periodic_upload_request_pending_[static_cast<std::size_t>(
              packet.final_receiver)] = false;
        }
        if (delivered) {
          ++bs_periodic_upload_requests_delivered_;
          schedulePeriodicBsUpload(
              packet.final_receiver, stamp, packet.action_step_id);
        } else {
          ++bs_periodic_upload_requests_failed_;
        }
        return;
      }
      bs_control_finished_ = true;
      if (delivered) {
        ++bs_upload_grants_delivered_;
        scheduleBsUpload(stamp);
      } else {
        ++bs_upload_grants_failed_;
      }
    }
  }

  void maybeFinishBsTurn(double stamp) {
    if (!bs_turn_active_ || !bs_control_finished_ ||
        stamp - bs_turn_started_at_ + 1.0e-12 < bs_min_turn_s_ ||
        !bsTurnQueuesEmpty()) {
      return;
    }
    bs_turn_active_ = false;
    bs_active_uav_ = -1;
  }

  bool bsLinkMayTransmit(const LinkKey &key) const {
    if (rl_bs_scheduler_enabled_) return true;
    if (!bs_round_robin_enabled_ ||
        (key.sender != ap_node_id_ && key.receiver != ap_node_id_)) {
      return true;
    }
    if (!bs_turn_active_ || bs_active_uav_ < 0) return false;
    const int uav = key.sender == ap_node_id_ ? key.receiver : key.sender;
    return uav == bs_active_uav_;
  }

  const LinkQuality *linkFor(const LinkKey &key, double stamp) const {
    const auto found = links_.find(key);
    if (found == links_.end()) return nullptr;
    if (timeSeconds(found->second.valid_until) + 1.0e-9 < stamp) {
      return nullptr;
    }
    return &found->second;
  }

  bool packetChannel(const PendingPacket &packet, const LinkKey &key,
                     double stamp, double &effective_snr_db) const {
    // An action-attributed packet must never silently fall back to links_.
    if (packet.action_id != 0U && !packet.action_channel_snapshot) return false;
    if (packet.action_channel_snapshot) {
      const auto &snapshot = *packet.action_channel_snapshot;
      if (packet.action_id == 0U ||
          packet.channel_version != snapshot.channel_version ||
          packet.action_step_id != snapshot.step_id || key.sender < 0 ||
          key.receiver < 0 || key.sender >= radio_node_count_ ||
          key.receiver >= radio_node_count_) {
        return false;
      }
      const auto index = static_cast<std::size_t>(key.sender) *
                             static_cast<std::size_t>(radio_node_count_) +
                         static_cast<std::size_t>(key.receiver);
      if (index >= snapshot.available.size() ||
          index >= snapshot.effective_snr_db.size() ||
          snapshot.available[index] == 0U) {
        return false;
      }
      effective_snr_db =
          static_cast<double>(snapshot.effective_snr_db[index]);
      return std::isfinite(effective_snr_db);
    }
    const auto *link = linkFor(key, stamp);
    if (!usable(link)) return false;
    effective_snr_db = radioSnrDb(link, radioModel(packet.route));
    return std::isfinite(effective_snr_db);
  }

  bool usable(const LinkQuality *link) const {
    if (mode_ == "ideal") return true;
    return link != nullptr && std::isfinite(link->snr_db) &&
           link->model != "unavailable";
  }

  double snr(const LinkQuality *link) const {
    return mode_ == "ideal" ? 40.0 : static_cast<double>(link->snr_db);
  }

  bool expired(const PendingPacket &packet, double stamp) const {
    // A perfect BS action must remain deliverable even when finite BS
    // serialization/queueing carries it beyond the application's normal TTL.
    // Other routes retain their original TTL semantics.
    if (perfectBsRoute(packet.route)) return false;
    // BS control is its own protocol class. It must not accidentally inherit
    // DroneState's 0.3 s freshness deadline merely because the synthetic
    // control packet carries no ROS payload and uses topic slot zero.
    const double ttl = packet.route == RouteStage::kBsControl
                           ? bs_control_ttl_s_
                           : kTopicPolicies[packet.topic_index].ttl_s;
    return ttl > 0.0 && stamp - packet.born_at > ttl;
  }

  void countBsNoLinkDrop(RouteStage route) {
    if (route == RouteStage::kBsControl) {
      ++bs_control_dropped_no_link_;
    } else if (route == RouteStage::kBsUplink) {
      ++bs_uplink_dropped_no_link_;
    } else if (route == RouteStage::kBsDownlink) {
      ++bs_downlink_dropped_no_link_;
    }
  }

  void countBsPerDrop(RouteStage route) {
    if (route == RouteStage::kBsControl) {
      ++bs_control_dropped_per_;
    } else if (route == RouteStage::kBsUplink) {
      ++bs_uplink_dropped_per_;
    } else if (route == RouteStage::kBsDownlink) {
      ++bs_downlink_dropped_per_;
    }
  }

  void countBsTtlDrop(RouteStage route) {
    if (route == RouteStage::kBsControl) {
      ++bs_control_dropped_ttl_;
    } else if (route == RouteStage::kBsUplink) {
      ++bs_uplink_dropped_ttl_;
    } else if (route == RouteStage::kBsDownlink) {
      ++bs_downlink_dropped_ttl_;
    }
  }

  std::deque<PendingPacket>::iterator removePacket(
      LinkQueue &queue, std::deque<PendingPacket>::iterator packet) {
    if (packet == queue.packets.end()) return packet;
    updateRouteQueueCache(*packet, false);
    queue.bytes -= std::min(queue.bytes, packet->bytes);
    return queue.packets.erase(packet);
  }

  std::mt19937 &linkRng(const LinkKey &key) {
    return link_rngs_.at(key);
  }

  PacketStartResult startPacket(const LinkKey &key, PendingPacket &packet,
                                double stamp,
                                double bs_airtime_horizon) {
    if (packet.transmitting || packet.delivery_at > stamp) {
      return PacketStartResult::kDeferred;
    }
    const bool perfect_bs = perfectBsRoute(packet.route);
    double link_snr{};
    if (!perfect_bs && !packetChannel(packet, key, stamp, link_snr)) {
      if (packet.action_channel_snapshot) {
        ++rl_action_bound_transmission_attempts_;
      }
      return PacketStartResult::kNoLink;
    }
    const auto &radio = radioModel(packet.route);
    if (perfect_bs) link_snr = 40.0;
    double transmission_started_at = std::max(stamp, packet.delivery_at);
    if (usesBsRadio(packet.route)) {
      // Serialization uses the shared BS radio, but propagation does not.
      // Admit several packets to the same 20 ms PHY window so the link can
      // pipeline them instead of waiting for each packet to arrive first.
      transmission_started_at =
          std::max(transmission_started_at, bs_radio_available_at_);
    }
    const double serialization_delay =
        radio.serializationDelay(link_snr, packet.bytes);
    if (usesBsRadio(packet.route) &&
        transmission_started_at >= bs_airtime_horizon - 1.0e-12) {
      return PacketStartResult::kDeferred;
    }
    if (packet.action_channel_snapshot) {
      ++rl_action_bound_transmission_attempts_;
    }
    const std::string selected_mcs(radio.selectMcs(link_snr).name);
    ++mcs_counts_[selected_mcs];
    if (usesBsRadio(packet.route)) {
      ++bs_mcs_counts_[selected_mcs];
    } else {
      ++uav_mcs_counts_[selected_mcs];
    }
    cumulative_initial_tbler_ +=
        perfect_bs ? 0.0 : radio.transportBlockErrorRate(link_snr);
    ++initial_tbler_samples_;
    const double random_jitter =
        jitter_s_ <= 0.0
            ? 0.0
            : std::uniform_real_distribution<double>(-jitter_s_, jitter_s_)(
                  linkRng(key));
    if (usesBsRadio(packet.route)) {
      bs_radio_available_at_ = transmission_started_at + serialization_delay;
    }
    packet.delivery_at =
        transmission_started_at +
        serialization_delay +
        std::max(0.0, base_latency_s_ + random_jitter);
    const auto occupied_slots = static_cast<std::uint64_t>(std::max(
        1.0, std::ceil(serialization_delay / radio.slotDuration())));
    const auto prb_slots = occupied_slots *
        static_cast<std::uint64_t>(radio.config().resource_blocks);
    if (packet.route == RouteStage::kBsUplink ||
        packet.route == RouteStage::kApUplink) {
      bs_uplink_prb_slots_ += prb_slots;
      if (packet.reserved_pair_control) {
        pair_control_reserved_uplink_prb_slots_ += prb_slots;
      }
    } else if (packet.route == RouteStage::kBsDownlink ||
               packet.route == RouteStage::kBsControl ||
               packet.route == RouteStage::kApDownlink) {
      bs_downlink_prb_slots_ += prb_slots;
      if (packet.reserved_pair_control) {
        pair_control_reserved_downlink_prb_slots_ += prb_slots;
      }
    } else if (packet.route == RouteStage::kDirect) {
      direct_u2u_prb_slots_ += prb_slots;
    }
    packet.transmitting = true;
    return PacketStartResult::kStarted;
  }

  bool deliverToUav(const PendingPacket &packet, int receiver,
                    double stamp) {
    if (!packet.flow || receiver < 0 || receiver >= drone_count_) return false;
    const auto receiver_index = static_cast<std::size_t>(receiver);
    if (packet.has_chunk_key &&
        uav_chunks_[receiver_index].find(packet.chunk_key) !=
            uav_chunks_[receiver_index].end()) {
      ++duplicates_suppressed_;
      return false;
    }
    if (packet.flow->delivered[receiver_index]) {
      ++duplicates_suppressed_;
      return false;
    }
    publishers_[packet.topic_index][static_cast<std::size_t>(receiver)]->
        publish(*packet.message);
    packet.flow->delivered[receiver_index] = true;
    if (packet.has_chunk_key) {
      insertKnownUavChunk(receiver, packet.chunk_key);
    }
    const int information_sender =
        packet.current_sender >= 0 && packet.current_sender < drone_count_
            ? packet.current_sender
            : (packet.flow ? packet.flow->origin_sender : -1);
    if (information_sender >= 0 && information_sender < drone_count_) {
      last_uav_info_received_s_[static_cast<std::size_t>(information_sender)]
                               [receiver_index] = stamp;
    }
    if (packet.route != RouteStage::kBsDownlink ||
        packet.peer_message_version > 0U) {
      ++logical_delivered_packets_;
      logical_delivered_bytes_ += packet.bytes;
      cumulative_end_to_end_delay_s_ += stamp - packet.flow->born_at;
    }
    if (packet.route == RouteStage::kApDownlink) {
      ++ap_relay_wins_;
    } else if (packet.route == RouteStage::kBsDownlink) {
      if (packet.has_chunk_key) {
        ++bs_missing_chunks_delivered_downlink_;
      } else if (packet.peer_message_version > 0U) {
        ++bs_peer_downlink_delivered_;
        ++bs_peer_downlink_delivered_by_topic_[packet.topic_index];
      }
    } else {
      ++direct_delivery_wins_;
    }
    observePairControlDelivery(packet, receiver, stamp);
    return true;
  }

  void completeSuccessfulPacket(const LinkKey &key,
                                const PendingPacket &packet,
                                double stamp) {
    if (packet.action_channel_snapshot) {
      ++rl_action_bound_packets_delivered_;
    }
    ++delivered_packets_;
    delivered_bytes_ += packet.bytes;
    cumulative_delay_s_ += stamp - packet.enqueued_at;
    if (packet.reserved_pair_control) {
      const double hop_delay_s = stamp - packet.enqueued_at;
      pair_control_reserved_cumulative_hop_delay_s_ += hop_delay_s;
      pair_control_reserved_max_hop_delay_s_ =
          std::max(pair_control_reserved_max_hop_delay_s_, hop_delay_s);
      ++pair_control_reserved_delivered_packets_;
      pair_control_reserved_delivered_bytes_ += packet.bytes;
      if (packet.route == RouteStage::kBsUplink) {
        ++pair_control_reserved_uplinks_delivered_;
      } else if (packet.route == RouteStage::kBsDownlink) {
        ++pair_control_reserved_downlinks_delivered_;
        const double end_to_end_delay_s = stamp - packet.born_at;
        pair_control_reserved_cumulative_end_to_end_delay_s_ +=
            end_to_end_delay_s;
        pair_control_reserved_max_end_to_end_delay_s_ = std::max(
            pair_control_reserved_max_end_to_end_delay_s_,
            end_to_end_delay_s);
      }
    }
    if (packet.route == RouteStage::kDirect) {
      ++direct_delivered_packets_;
      deliverToUav(packet, packet.final_receiver, stamp);
      return;
    }
    if (packet.route == RouteStage::kApDownlink) {
      ++ap_downlink_delivered_packets_;
      deliverToUav(packet, packet.final_receiver, stamp);
      return;
    }

    if (packet.route == RouteStage::kBsControl) {
      ++bs_control_delivered_packets_;
      finishBsPacket(packet, true, stamp);
      return;
    }
    if (packet.route == RouteStage::kBsDownlink) {
      ++bs_downlink_delivered_packets_;
      deliverToUav(packet, packet.final_receiver, stamp);
      return;
    }
    if (packet.route == RouteStage::kBsUplink) {
      ++bs_uplink_delivered_packets_;
      const int information_sender =
          packet.current_sender >= 0 && packet.current_sender < drone_count_
              ? packet.current_sender
              : (packet.flow ? packet.flow->origin_sender : -1);
      if (information_sender >= 0 && information_sender < drone_count_) {
        last_bs_info_received_s_[static_cast<std::size_t>(information_sender)] =
            stamp;
      }
      if (packet.has_chunk_key && insertKnownBsChunk(packet.chunk_key)) {
        ++bs_incremental_chunks_received_uplink_;
      }
      if (packet.peer_message_version > 0U && packet.flow &&
          packet.flow->origin_sender >= 0 &&
          packet.flow->origin_sender < drone_count_) {
        auto &cached =
            bs_latest_peer_messages_[static_cast<std::size_t>(
                packet.flow->origin_sender)][packet.topic_index];
        if (packet.peer_message_version >= cached.version) {
          cached.message = packet.message;
          cached.flow = packet.flow;
          cached.bytes = packet.bytes;
          cached.born_at = packet.born_at;
          cached.version = packet.peer_message_version;
        }
        ++bs_peer_uplink_delivered_;
        ++bs_peer_uplink_delivered_by_topic_[packet.topic_index];
      }
      if (kTopicPolicies.at(packet.topic_index).key == "chunk_data" &&
          packet.message) {
        if (task_metric_observer_mode_ == "inline") {
          racer_fidelity_msgs::msg::ChunkData chunk;
          if (deserialize(packet.message, chunk)) {
            applyTaskChunkObservation(chunk, false, true);
          }
        } else if (task_metric_observer_mode_ == "async") {
          enqueueTaskObservation(
              {TaskObservationKind::kBsChunk, information_sender,
               packet.message, nullptr});
        }
      }
      return;
    }

    ++ap_uplink_delivered_packets_;
    if (kTopicPolicies.at(packet.topic_index).key == "recovery_status") {
      if (ap_recovery_status_publisher_ && packet.message) {
        ap_recovery_status_publisher_->publish(*packet.message);
        ++ap_global_updates_received_;
        ++logical_delivered_packets_;
        logical_delivered_bytes_ += packet.bytes;
        if (packet.flow) {
          cumulative_end_to_end_delay_s_ +=
              stamp - packet.flow->born_at;
        }
      }
      return;
    }
    if (!packet.flow || packet.flow->ap_received) return;
    packet.flow->ap_received = true;
    ++ap_global_updates_received_;
    ap_latest_messages_[static_cast<std::size_t>(
        packet.flow->origin_sender)][packet.topic_index] = packet.message;
    for (int receiver = 0; receiver < drone_count_; ++receiver) {
      if (receiver == packet.flow->origin_sender ||
          packet.flow->delivered[static_cast<std::size_t>(receiver)]) {
        continue;
      }
      if (enqueue(ap_node_id_, receiver, packet.topic_index,
                  packet.message, packet.flow, RouteStage::kApDownlink,
                  receiver, packet.flow->born_at,
                  packet.has_chunk_key ? &packet.chunk_key : nullptr)) {
        ++ap_selective_forwards_enqueued_;
      }
    }
    (void)key;
  }

  // Slot-based TDD OFDMA on the shared BS band. Each active peer gets a
  // disjoint PRB range in the selected direction. We keep power spectral
  // density constant (power proportional to allocated PRBs), so the existing
  // full-band channel SNR/MCS remains valid and there is no artificial SNR
  // gain or full-band power duplication when the BS serves several peers.
  void processBsOfdma(double stamp, double horizon) {
    const double slot_s = bs_model_.slotDuration();
    double cursor = std::max(bs_ofdma_next_at_,
        std::ceil((stamp - 1.0e-9) / slot_s) * slot_s);
    struct Head {
      LinkKey key;
      std::deque<PendingPacket>::iterator packet;
      double snr{};
    };
    // Never replay past slots against queues that arrived at the current
    // clock, and never allocate a slot twice when /clock is paused/repeated.
    while (cursor + slot_s <= horizon + 1.0e-9) {
      std::vector<Head> uplinks;
      std::vector<Head> downlinks;
      for (auto &[key, queue] : queues_) {
        if (!bsLinkMayTransmit(key)) continue;
        // Remove stale/unavailable attempts before computing fair shares.
        // Do this in a separate pass: deque erase invalidates its iterators.
        for (auto packet = queue.packets.begin(); packet != queue.packets.end();) {
          if (!usesBsRadio(packet->route) || packet->transmitting ||
              !eventDrivenPacketMayUseBs(*packet) || packet->delivery_at > cursor) {
            ++packet;
            continue;
          }
          const bool ttl = expired(*packet, cursor);
          double snr{};
          const bool available = perfectBsRoute(packet->route) ||
                                 packetChannel(*packet, key, cursor, snr);
          if (!ttl && available) { ++packet; continue; }
          const PendingPacket failed = *packet;
          finishBsPacket(failed, false, cursor);
          if (ttl) {
            ++dropped_ttl_;
            countBsTtlDrop(failed.route);
          } else {
            if (packet->action_channel_snapshot) ++rl_action_bound_transmission_attempts_;
            ++dropped_no_link_;
            countBsNoLinkDrop(failed.route);
          }
          packet = removePacket(queue, packet);
        }
        auto selected = queue.packets.end();
        int priority = std::numeric_limits<int>::min();
        for (auto packet = queue.packets.begin(); packet != queue.packets.end(); ++packet) {
          if (!usesBsRadio(packet->route) || packet->transmitting ||
              !eventDrivenPacketMayUseBs(*packet) || packet->delivery_at > cursor) continue;
          const int candidate_priority = packetPriority(
              packet->route, packet->topic_index, packet->reserved_pair_control);
          if (selected == queue.packets.end() || candidate_priority > priority ||
              (candidate_priority == priority && packet->enqueued_at < selected->enqueued_at)) {
            selected = packet;
            priority = candidate_priority;
          }
        }
        if (selected == queue.packets.end()) continue;
        double snr = 40.0;
        if (!perfectBsRoute(selected->route)) packetChannel(*selected, key, cursor, snr);
        auto &heads = key.receiver == ap_node_id_ ? uplinks : downlinks;
        heads.push_back({key, selected, snr});
      }
      std::vector<int> up_peers, down_peers;
      for (const auto &head : uplinks) up_peers.push_back(head.key.sender);
      for (const auto &head : downlinks) down_peers.push_back(head.key.receiver);
      const auto slot = bs_ofdma_scheduler_.allocate(up_peers, down_peers);
      ++bs_ofdma_slot_index_;
      ++bs_ofdma_slots_processed_;
      if (!slot.allocations.empty()) {
        ++bs_ofdma_active_slots_;
        if (slot.uplink) ++bs_ofdma_uplink_slots_;
        else ++bs_ofdma_downlink_slots_;
        bs_ofdma_peak_parallel_links_ = std::max<std::uint64_t>(
            bs_ofdma_peak_parallel_links_, slot.allocations.size());
      }
      auto &heads = slot.uplink ? uplinks : downlinks;
      int slot_prbs = 0;
      for (const auto &grant : slot.allocations) {
        const auto found = std::find_if(heads.begin(), heads.end(),
            [&](const Head &head) {
              return (slot.uplink ? head.key.sender : head.key.receiver) == grant.sender;
            });
        if (found == heads.end()) throw std::logic_error("missing BS OFDMA queue");
        auto &packet = *found->packet;
        if (!packet.bs_ofdma_started) {
          packet.bs_ofdma_started = true;
          packet.bs_ofdma_remaining_bits = 8.0 * static_cast<double>(packet.bytes);
          if (packet.action_channel_snapshot) ++rl_action_bound_transmission_attempts_;
          const std::string mcs(bs_model_.selectMcs(found->snr).name);
          ++mcs_counts_[mcs];
          ++bs_mcs_counts_[mcs];
          cumulative_initial_tbler_ += perfectBsRoute(packet.route)
              ? 0.0 : bs_model_.transportBlockErrorRate(found->snr);
          ++initial_tbler_samples_;
        }
        packet.bs_ofdma_remaining_bits = std::max(0.0,
            packet.bs_ofdma_remaining_bits - bs_model_.bitsPerSlot(found->snr, grant.prb_count));
        slot_prbs += grant.prb_count;
        bs_ofdma_allocated_prb_slots_ += grant.prb_count;
        if (slot.uplink) {
          bs_uplink_prb_slots_ += grant.prb_count;
          if (packet.reserved_pair_control) pair_control_reserved_uplink_prb_slots_ += grant.prb_count;
        } else {
          bs_downlink_prb_slots_ += grant.prb_count;
          if (packet.reserved_pair_control) pair_control_reserved_downlink_prb_slots_ += grant.prb_count;
        }
        if (bs_ofdma_diagnostic_logging_) {
          RCLCPP_INFO(get_logger(),
              "RACER_BS_OFDMA_GRANT slot=%lu time_s=%.9f uplink=%d sender=%d receiver=%d "
              "first_prb=%d prbs=%d remaining_bits=%.3f attempt=%d",
              static_cast<unsigned long>(bs_ofdma_slot_index_), cursor,
              slot.uplink, found->key.sender, found->key.receiver,
              grant.first_prb, grant.prb_count, packet.bs_ofdma_remaining_bits, packet.attempts);
        }
        if (packet.bs_ofdma_remaining_bits <= 0.0) {
          const double jitter = jitter_s_ <= 0.0 ? 0.0
              : std::uniform_real_distribution<double>(-jitter_s_, jitter_s_)(linkRng(found->key));
          packet.transmitting = true;
          packet.delivery_at = cursor + slot_s + std::max(0.0, base_latency_s_ + jitter);
        }
      }
      bs_ofdma_max_allocated_prbs_per_slot_ = std::max(
          bs_ofdma_max_allocated_prbs_per_slot_, slot_prbs);
      cursor += slot_s;
    }
    bs_ofdma_next_at_ = cursor;
  }

  // Independent CSMA collision domain for all UAV<->BS traffic. The BS is
  // one transmitter (not one contender per destination). U2U uses its own band.
  void processBsCsma(double stamp, double horizon) {
    const auto count = static_cast<std::size_t>(radio_node_count_);
    if (bs_csma_backoff_.size() != count) {
      bs_csma_backoff_.assign(count, -1);
      bs_csma_cw_.assign(count, csma_cw_min_);
    }
    const double slot = bs_model_.slotDuration();
    double cursor = std::max({stamp, bs_csma_next_at_, bs_radio_available_at_});
    while (cursor < horizon - 1.0e-12) {
      struct Candidate {
        LinkKey key{};
        LinkQueue *queue{};
        std::deque<PendingPacket>::iterator packet{};
        int priority{std::numeric_limits<int>::min()};
      };
      for (auto &[key, queue] : queues_) {
        for (auto packet = queue.packets.begin(); packet != queue.packets.end();) {
          if (!packet->transmitting && usesBsRadio(packet->route) && expired(*packet, stamp)) {
            finishBsPacket(*packet, false, stamp);
            ++dropped_ttl_;
            countBsTtlDrop(packet->route);
            packet = removePacket(queue, packet);
          } else ++packet;
        }
      }
      std::vector<Candidate> heads(count);
      for (auto &[key, queue] : queues_) {
        if (!bsLinkMayTransmit(key)) continue;
        for (auto packet = queue.packets.begin(); packet != queue.packets.end();) {
          if (!usesBsRadio(packet->route) || !eventDrivenPacketMayUseBs(*packet) ||
              packet->transmitting || packet->delivery_at > stamp) { ++packet; continue; }
          const int priority = packetPriority(packet->route, packet->topic_index,
                                              packet->reserved_pair_control);
          auto &head = heads.at(static_cast<std::size_t>(key.sender));
          if (!head.queue || priority > head.priority ||
              (priority == head.priority && packet->enqueued_at < head.packet->enqueued_at) ||
              (priority == head.priority && packet->enqueued_at == head.packet->enqueued_at &&
               key.receiver < head.key.receiver)) {
            head = {key, &queue, packet, priority};
          }
          ++packet;
        }
      }
      std::vector<std::size_t> active, contenders;
      for (std::size_t sender = 0; sender < count; ++sender) {
        if (!heads[sender].queue) { bs_csma_backoff_[sender] = -1; continue; }
        active.push_back(sender);
        if (bs_csma_backoff_[sender] < 0) {
          bs_csma_backoff_[sender] = std::uniform_int_distribution<int>(
              0, bs_csma_cw_[sender])(linkRng(heads[sender].key));
        }
      }
      if (active.empty()) { bs_csma_guard_ = 0; break; }
      if (bs_csma_guard_ < csma_difs_slots_) {
        ++bs_csma_guard_;
        ++bs_csma_idle_slots_;
        cursor += slot;
        continue;
      }
      for (auto sender : active) {
        if (bs_csma_backoff_[sender] == 0) contenders.push_back(sender);
      }
      if (contenders.empty()) {
        for (auto sender : active) --bs_csma_backoff_[sender];
        ++bs_csma_idle_slots_;
        cursor += slot;
        continue;
      }
      const bool collision = contenders.size() > 1;
      double busy_until = cursor;
      std::size_t started{};
      for (auto sender : contenders) {
        auto &head = heads[sender];
        // Simultaneous contenders serialize over the same interval. Every
        // attempted packet consumes resources and collisions fail on retirement.
        bs_radio_available_at_ = cursor;
        const auto result = startPacket(head.key, *head.packet, cursor, horizon);
        bs_csma_backoff_[sender] = -1;
        if (result == PacketStartResult::kStarted) {
          head.packet->bs_csma_collision = collision;
          busy_until = std::max(busy_until, bs_radio_available_at_);
          ++started;
        } else if (result == PacketStartResult::kNoLink) {
          finishBsPacket(*head.packet, false, stamp);
          ++dropped_no_link_;
          countBsNoLinkDrop(head.packet->route);
          removePacket(*head.queue, head.packet);
        }
      }
      if (collision && started > 0) {
        ++bs_csma_collision_events_;
        bs_csma_collided_packets_ += started;
      } else if (started > 0) ++bs_csma_successful_contentions_;
      bs_csma_guard_ = 0;
      bs_radio_available_at_ = busy_until;
      cursor = std::max(cursor + slot, busy_until);
    }
    bs_csma_next_at_ = cursor;
  }

  void schedulerTick() {
    const double stamp = now().seconds();
    if (rl_bs_synchronous_mode_ && rl_sync_scheduler_have_stamp_) {
      if (stamp + 1.0e-9 < rl_sync_scheduler_last_stamp_s_) {
        throw std::runtime_error(
            "simulation time moved backwards in communication scheduler");
      }
      if (stamp <= rl_sync_scheduler_last_stamp_s_ + 1.0e-9) {
        // Wall timers continue firing during PPO/Qwen work.  An identical
        // /clock stamp must be a complete no-op so queues, retries, packet
        // delivery and resource counters remain frozen with Isaac.
        return;
      }
    }
    if (shared_uav_radio_enabled_) {
      advanceUavOfdma(stamp);
      if (!ap_enabled_) return;
    }
    // Reserved transaction stages enter the normal BS queues before the RL
    // action for this tick is applied. This preserves their control priority
    // even when the current action happens to select the same message.
    scheduleReservedPairControl(stamp);
    schedulePeriodicBsUploadRequests(stamp);
    const bool rl_state_due = rl_bs_event_driven_one_shot_
        ? advanceEventDrivenOneShotClock(stamp)
        : (rl_bs_synchronous_mode_ ? advanceSynchronousRlClock(stamp)
                                   : beginRlDecision(stamp));
    if (!rl_bs_scheduler_enabled_) beginBsTurn(stamp);
    // First retire every packet whose propagation delay has elapsed.  A
    // packet remains charged to its link queue while in flight, so queue
    // capacity and duplicate suppression continue to cover the whole trip.
    for (auto &[key, queue] : queues_) {
      if (!bsLinkMayTransmit(key)) continue;
      for (auto packet = queue.packets.begin();
           packet != queue.packets.end();) {
        if (!eventDrivenPacketMayUseBs(*packet) ||
            !packet->transmitting || packet->delivery_at > stamp) {
          ++packet;
          continue;
        }
        if (expired(*packet, stamp)) {
          const PendingPacket failed = *packet;
          finishBsPacket(failed, false, stamp);
          ++dropped_ttl_;
          countBsTtlDrop(failed.route);
          packet = removePacket(queue, packet);
          continue;
        }
        const bool perfect_bs = perfectBsRoute(packet->route);
        double link_snr{};
        if (!perfect_bs && !packetChannel(*packet, key, stamp, link_snr)) {
          const PendingPacket failed = *packet;
          finishBsPacket(failed, false, stamp);
          ++dropped_no_link_;
          countBsNoLinkDrop(failed.route);
          packet = removePacket(queue, packet);
          continue;
        }
        const auto &radio = radioModel(packet->route);
        const double per =
            mode_ == "ideal" || perfect_bs
                ? 0.0
                : radio.packetErrorRate(link_snr, packet->bytes);
        const bool transmission_failed =
            packet->bs_csma_collision || std::uniform_real_distribution<double>(0.0, 1.0)(linkRng(key)) <
            per;
        const bool reliable =
            packet->route == RouteStage::kBsControl ||
            kTopicPolicies[packet->topic_index].reliable;
        const bool bs_route = packet->route == RouteStage::kBsControl ||
                              packet->route == RouteStage::kBsUplink ||
                              packet->route == RouteStage::kBsDownlink;
        const int retry_limit = bs_route ? bs_max_retries_ : max_retries_;
        if (bs_channel_access_mode_ == "csma" && usesBsRadio(packet->route)) {
          const auto sender = static_cast<std::size_t>(key.sender);
          bs_csma_backoff_[sender] = -1;
          bs_csma_cw_[sender] = transmission_failed
              ? static_cast<int>(std::min<long long>(csma_cw_max_, 2LL * (bs_csma_cw_[sender] + 1LL) - 1LL))
              : csma_cw_min_;
        }
        if (transmission_failed && reliable &&
            packet->attempts < retry_limit) {
          ++packet->attempts;
          ++retried_packets_;
          packet->transmitting = false;
          packet->bs_csma_collision = false;
          packet->bs_ofdma_started = false;
          packet->bs_ofdma_remaining_bits = 0.0;
          packet->delivery_at = stamp + retry_backoff_s_;
          ++packet;
          continue;
        }
        if (transmission_failed) {
          const PendingPacket failed = *packet;
          finishBsPacket(failed, false, stamp);
          ++dropped_per_;
          countBsPerDrop(failed.route);
          packet = removePacket(queue, packet);
          continue;
        }
        // Copy before delivery because AP/control completion may enqueue a
        // packet onto another pre-created link queue.
        const PendingPacket completed = *packet;
        completeSuccessfulPacket(key, completed, stamp);
        packet = removePacket(queue, packet);
      }
    }

    // Assign the current 20 ms BS airtime globally.  Serialization is shared
    // across all UAV-BS queues, while already-serialized packets can remain
    // in propagation concurrently.  This realizes the same aggregate PHY
    // budget used by scheduleRlActionWithinSlotBudgets().
    const double bs_airtime_horizon =
        stamp + std::max(rl_bs_communication_slot_s_, 1.0e-6);
    if (bs_channel_access_mode_ == "csma") processBsCsma(stamp, bs_airtime_horizon);
    if (bs_channel_access_mode_ == "ofdma") processBsOfdma(stamp, bs_airtime_horizon);
    while (bs_channel_access_mode_ != "csma") {
      LinkQueue *selected_queue{};
      LinkKey selected_key{};
      std::deque<PendingPacket>::iterator selected_packet{};
      int selected_priority = std::numeric_limits<int>::min();
      double selected_enqueued_at = std::numeric_limits<double>::infinity();
      for (auto &[key, queue] : queues_) {
        if (!bsLinkMayTransmit(key)) continue;
        const bool non_bs_transmission_active = std::any_of(
            queue.packets.begin(), queue.packets.end(),
            [this](const PendingPacket &candidate) {
              return candidate.transmitting &&
                     !usesBsRadio(candidate.route);
            });
        for (auto packet = queue.packets.begin();
             packet != queue.packets.end(); ++packet) {
          if ((bs_channel_access_mode_ == "ofdma" && usesBsRadio(packet->route)) ||
              !eventDrivenPacketMayUseBs(*packet) ||
              packet->transmitting || packet->delivery_at > stamp ||
              (!usesBsRadio(packet->route) &&
               non_bs_transmission_active)) {
            continue;
          }
          const int priority = packetPriority(
              packet->route, packet->topic_index,
              packet->reserved_pair_control);
          if (!selected_queue || priority > selected_priority ||
              (priority == selected_priority &&
               packet->enqueued_at < selected_enqueued_at)) {
            selected_queue = &queue;
            selected_key = key;
            selected_packet = packet;
            selected_priority = priority;
            selected_enqueued_at = packet->enqueued_at;
          }
        }
      }
      if (!selected_queue) break;
      if (expired(*selected_packet, stamp)) {
        const PendingPacket failed = *selected_packet;
        finishBsPacket(failed, false, stamp);
        ++dropped_ttl_;
        countBsTtlDrop(failed.route);
        removePacket(*selected_queue, selected_packet);
        continue;
      }
      const auto result = startPacket(
          selected_key, *selected_packet, stamp, bs_airtime_horizon);
      if (result == PacketStartResult::kStarted) continue;
      if (result == PacketStartResult::kNoLink) {
        const PendingPacket failed = *selected_packet;
        finishBsPacket(failed, false, stamp);
        ++dropped_no_link_;
        countBsNoLinkDrop(failed.route);
        removePacket(*selected_queue, selected_packet);
        continue;
      }
      // The shared serialization window is full. Remaining BS packets stay
      // queued for the next physical slot and retain their action snapshot.
      break;
    }
    if (!rl_bs_scheduler_enabled_) maybeFinishBsTurn(stamp);
    if (rl_state_due) {
      writeRlState(
          rl_bs_synchronous_mode_ ? stamp : rl_pending_boundary_stamp_s_);
      if (rl_bs_event_driven_one_shot_) {
        if (rl_completed_interval_available_) clearEventDrivenAction();
      } else if (!rl_bs_synchronous_mode_ && shared_communication_) {
        completeRlDecisionBoundary();
      }
    }
    if (rl_bs_synchronous_mode_ && rl_sync_clock_initialized_) {
      rl_sync_scheduler_have_stamp_ = true;
      rl_sync_scheduler_last_stamp_s_ = stamp;
    }
  }

  void publishStatistics() {
    const double task_stamp = now().seconds();
    if (rl_bs_synchronous_mode_) {
      if (rl_sync_boundary_frozen_) return;
      if (rl_have_statistics_sim_stamp_ &&
          task_stamp - rl_last_statistics_sim_stamp_s_ < 1.0 - 1.0e-9) {
        return;
      }
      rl_have_statistics_sim_stamp_ = true;
      rl_last_statistics_sim_stamp_s_ = task_stamp;
    }
    const auto task_step = static_cast<std::uint64_t>(std::llround(
        std::max(0.0, task_stamp) / rl_bs_decision_period_s_));
    const double task_time_s =
        static_cast<double>(task_step) * rl_bs_decision_period_s_;
    BsGlobalMapMetrics bs_map_metrics;
    TaskQualitySample task_quality;
    bool have_task_metric{};
    if (!shared_task_metric_ring_) {
      // Preserve the explicitly selected legacy file bridge. Production
      // shared-memory training never takes this synchronous diagnostic path.
      bs_map_metrics = bsGlobalMapMetrics();
      task_quality = {task_time_s, redundantExplorationRatio(),
                      bs_map_metrics.iou, bs_map_metrics.coverage,
                      task_step, task_stamp};
      have_task_metric = true;
      exportObservedOccupiedVoxels();
      validateIncrementalCaches();
    } else {
      std::lock_guard<std::mutex> lock(task_metric_snapshot_mutex_);
      bs_map_metrics = latest_bs_map_metrics_;
      task_quality = latest_task_quality_;
      have_task_metric = task_metric_compute_count_ > 0U;
    }
    racer_sionna_interfaces::msg::CommStatistics message;
    message.stamp = now();
    if (have_task_metric) {
      task_quality.bs_uplink_prb_slots = bs_uplink_prb_slots_;
      task_quality.bs_downlink_prb_slots = bs_downlink_prb_slots_;
      task_quality.direct_u2u_prb_slots = direct_u2u_prb_slots_;
      if (task_quality_history_.empty() ||
          task_quality_history_.back().task_step < task_quality.task_step) {
        task_quality_history_.push_back(task_quality);
      } else if (task_quality_history_.back().task_step ==
                 task_quality.task_step) {
        task_quality_history_.back() = task_quality;
      }
    }
    message.attempted_packets = attempted_packets_;
    message.delivered_packets = delivered_packets_;
    message.dropped_no_link = dropped_no_link_;
    message.dropped_per = dropped_per_;
    message.dropped_queue = dropped_queue_;
    message.dropped_ttl = dropped_ttl_;
    message.retried_packets = retried_packets_;
    message.attempted_bytes = attempted_bytes_;
    message.delivered_bytes = delivered_bytes_;
    for (const auto &[key, queue] : queues_) {
      (void)key;
      message.queued_packets += queue.packets.size();
      message.queued_bytes += queue.bytes;
    }
    for (const auto &queue : uav_udp_queues_) {
      message.queued_packets += queue.datagrams.size();
      message.queued_bytes += queue.bytes;
    }
    message.queued_packets += completed_uav_datagrams_.size();
    message.mean_delivery_delay_ms =
        delivered_packets_ == 0U
            ? 0.0F
            : static_cast<float>(1000.0 * cumulative_delay_s_ /
                                 delivered_packets_);
    statistics_publisher_->publish(message);

    const double logical_delivery_ratio =
        logical_attempted_packets_ == 0U
            ? 0.0
            : static_cast<double>(logical_delivered_packets_) /
                  static_cast<double>(logical_attempted_packets_);
    const double mean_end_to_end_delay_ms =
        logical_delivered_packets_ == 0U
            ? 0.0
            : 1000.0 * cumulative_end_to_end_delay_s_ /
                  static_cast<double>(logical_delivered_packets_);
    const double mean_initial_tbler =
        initial_tbler_samples_ == 0U
            ? 0.0
            : cumulative_initial_tbler_ /
                  static_cast<double>(initial_tbler_samples_);
    const auto &total_phy = model_.config();
    const auto &phy = directRadioModel().config();
    const auto &bs_phy = bs_model_.config();
    const auto &fixed_mcs = directRadioModel().selectMcs(0.0);
    const bool uses_fixed_mcs = phy.fixed_mcs_index >= 0;
    const auto initial_assignment_ack_count = static_cast<std::uint64_t>(
        std::count(initial_assignment_epoch_acks_.begin(),
                   initial_assignment_epoch_acks_.end(), true));
    const auto perfect_direct_attempts =
        lossless_nearest_forwarded_packets_ +
        (initial_assignment_perfect_via_bs_
             ? 0U
             : initial_assignment_perfect_forwarded_packets_);
    const auto sionna_direct_attempted_packets =
        direct_attempted_packets_ >= perfect_direct_attempts
            ? direct_attempted_packets_ - perfect_direct_attempts
            : 0U;
    std::ostringstream json;
    json << std::setprecision(std::numeric_limits<double>::max_digits10);
    json << "{\"network_topology\":\"" << topology_
         << "\",\"nearest_neighbor_count\":" << nearest_neighbor_count_
         << ",\"lossless_nearest_neighbor_count\":"
         << lossless_nearest_neighbor_count_
         << ",\"lossless_communication_range_m\":"
         << lossless_communication_range_m_
         << ",\"lossless_control_only\":"
         << (lossless_control_only_ ? "true" : "false")
         << ",\"directed_message_unicast\":"
         << (directed_message_unicast_ ? "true" : "false")
         << ",\"uav_udp_directed_unicast\":"
         << (uav_udp_directed_unicast_ ? "true" : "false")
         << ",\"chunk_data_pre_enqueue_dedup\":"
         << (chunk_data_pre_enqueue_dedup_ ? "true" : "false")
         << ",\"chunk_data_max_pending_per_link\":"
         << chunk_data_max_pending_per_link_
         << ",\"directed_unicast_messages\":"
         << directed_unicast_messages_
         << ",\"invalid_directed_messages_dropped\":"
         << invalid_directed_messages_dropped_
         << ",\"chunk_enqueue_duplicates_suppressed\":"
         << chunk_enqueue_duplicates_suppressed_
         << ",\"chunk_enqueue_budget_drops\":"
         << chunk_enqueue_budget_drops_
         << ",\"communication_range_m\":" << communication_range_m_
         << ",\"ideal_direct_enabled\":"
         << (ideal_direct_enabled_ ? "true" : "false")
         << ",\"preserve_ideal_direct_with_bs\":"
         << (preserve_ideal_direct_with_bs_ ? "true" : "false")
         << ",\"initial_assignment_perfect_delivery_enabled\":"
         << (initial_assignment_perfect_delivery_ ? "true" : "false")
         << ",\"initial_assignment_perfect_via_bs\":"
         << (initial_assignment_perfect_via_bs_ ? "true" : "false")
         << ",\"initial_assignment_perfect_delivery_complete\":"
         << (initial_assignment_perfect_complete_ ? "true" : "false")
         << ",\"initial_assignment_perfect_epoch\":"
         << initial_assignment_perfect_epoch_
         << ",\"initial_assignment_perfect_acks\":"
         << initial_assignment_ack_count
         << ",\"initial_assignment_perfect_completed_at_s\":"
         << initial_assignment_perfect_completed_at_s_
         << ",\"initial_assignment_perfect_drone_state_messages\":"
         << initial_assignment_perfect_drone_state_messages_
         << ",\"initial_assignment_perfect_global_assignment_messages\":"
         << initial_assignment_perfect_global_assignment_messages_
         << ",\"initial_assignment_perfect_forwarded_packets\":"
         << initial_assignment_perfect_forwarded_packets_
         << ",\"initial_assignment_perfect_bs_uplinks\":"
         << initial_assignment_perfect_bs_uplinks_
         << ",\"initial_assignment_perfect_bs_downlinks\":"
         << initial_assignment_perfect_bs_downlinks_
         << ",\"uav_transport\":\"UDP\""
         << ",\"shared_uav_ofdma_enabled\":"
         << (shared_uav_radio_enabled_ && !uav_csma_enabled_ ? "true" : "false")
         << ",\"shared_uav_radio_enabled\":"
         << (shared_uav_radio_enabled_ ? "true" : "false")
         << ",\"bs_channel_access_mode\":\"" << bs_channel_access_mode_ << "\""
         << ",\"bs_downlink_source_owned_chunks_only\":" << (bs_downlink_source_owned_chunks_only_ ? "true" : "false")
         << ",\"bs_ofdma_enabled\":" << (bs_channel_access_mode_ == "ofdma" ? "true" : "false")
         << ",\"bs_ofdma_total_prbs\":" << bs_model_.config().resource_blocks
         << ",\"bs_ofdma_slot_duration_ms\":" << 1000.0 * bs_model_.slotDuration()
         << ",\"bs_ofdma_scheduler\":\"round_robin_equal_share\""
         << ",\"bs_ofdma_duplex\":\"tdd_alternating_backlogged_directions\""
         << ",\"bs_ofdma_power_policy\":\"constant_psd_proportional_to_prbs\""
         << ",\"bs_ofdma_slots_processed\":" << bs_ofdma_slots_processed_
         << ",\"bs_ofdma_active_slots\":" << bs_ofdma_active_slots_
         << ",\"bs_ofdma_uplink_slots\":" << bs_ofdma_uplink_slots_
         << ",\"bs_ofdma_downlink_slots\":" << bs_ofdma_downlink_slots_
         << ",\"bs_ofdma_allocated_prb_slots\":" << bs_ofdma_allocated_prb_slots_
         << ",\"bs_ofdma_max_allocated_prbs_per_slot\":" << bs_ofdma_max_allocated_prbs_per_slot_
         << ",\"bs_ofdma_peak_parallel_links\":" << bs_ofdma_peak_parallel_links_
         << ",\"bs_csma_idle_slots\":" << bs_csma_idle_slots_
         << ",\"bs_csma_successful_contentions\":" << bs_csma_successful_contentions_
         << ",\"bs_csma_collision_events\":" << bs_csma_collision_events_
         << ",\"bs_csma_collided_packets\":" << bs_csma_collided_packets_
         << ",\"uav_channel_access_mode\":\""
         << uav_channel_access_mode_ << "\""
         << ",\"uav_udp_header_bytes\":" << udp_header_bytes_
         << ",\"uav_udp_retransmissions\":0"
         << ",\"uav_ofdma_total_prbs\":"
         << directRadioModel().config().resource_blocks
         << ",\"uav_ofdma_slot_duration_ms\":"
         << 1000.0 * directRadioModel().slotDuration()
         << ",\"uav_ofdma_scheduler\":\""
         << (uav_csma_enabled_ ? "disabled_for_csma"
                               : "round_robin_equal_share")
         << "\""
         << ",\"uav_ofdma_half_duplex\":true"
         << ",\"uav_csma_cw_min\":" << csma_cw_min_
         << ",\"uav_csma_cw_max\":" << csma_cw_max_
         << ",\"uav_csma_difs_slots\":" << csma_difs_slots_
         << ",\"uav_ofdma_diagnostic_logging\":"
         << (ofdma_diagnostic_logging_ ? "true" : "false")
         << ",\"uav_ofdma_log_packet_stride\":"
         << ofdma_log_packet_stride_
         << ",\"uav_udp_datagrams_enqueued\":"
         << uav_udp_datagrams_enqueued_
         << ",\"uav_udp_unicast_datagrams\":"
         << uav_udp_unicast_datagrams_
         << ",\"uav_udp_multicast_datagrams\":"
         << uav_udp_multicast_datagrams_
         << ",\"uav_udp_datagrams_dropped_queue\":"
         << uav_udp_datagrams_dropped_queue_
         << ",\"uav_udp_datagrams_dropped_ttl\":"
         << uav_udp_datagrams_dropped_ttl_
         << ",\"uav_udp_receiver_attempts\":"
         << uav_udp_receiver_attempts_
         << ",\"uav_udp_receiver_successes\":"
         << uav_udp_receiver_successes_
         << ",\"uav_udp_receiver_per_failures\":"
         << uav_udp_receiver_per_failures_
         << ",\"uav_udp_receiver_no_link_failures\":"
         << uav_udp_receiver_no_link_failures_
         << ",\"uav_udp_receiver_half_duplex_failures\":"
         << uav_udp_receiver_half_duplex_failures_
         << ",\"uav_udp_physical_bytes_enqueued\":"
         << uav_udp_physical_bytes_enqueued_
         << ",\"uav_udp_physical_bytes_transmitted\":"
         << uav_udp_physical_bytes_transmitted_
         << ",\"uav_physical_transmissions_started\":"
         << uav_physical_transmissions_started_
         << ",\"uav_physical_transmissions_completed\":"
         << uav_physical_transmissions_completed_
         << ",\"uav_physical_transmissions_with_success\":"
         << uav_physical_transmissions_with_success_
         << ",\"uav_tx_power_applications\":"
         << uav_tx_power_applications_
         << ",\"uav_ofdma_slots_processed\":"
         << uav_ofdma_slots_processed_
         << ",\"uav_ofdma_active_slots\":"
         << uav_ofdma_active_slots_
         << ",\"uav_ofdma_prb_allocation_events\":"
         << uav_ofdma_prb_allocation_events_
         << ",\"uav_ofdma_allocated_prb_slots\":"
         << uav_ofdma_allocated_prb_slots_
         << ",\"uav_ofdma_peak_active_senders\":"
         << uav_ofdma_peak_active_senders_
         << ",\"uav_ofdma_max_allocated_prbs_per_slot\":"
         << uav_ofdma_max_allocated_prbs_per_slot_
         << ",\"uav_csma_idle_slots\":" << uav_csma_idle_slots_
         << ",\"uav_csma_backoff_slots\":" << uav_csma_backoff_slots_
         << ",\"uav_csma_busy_slots\":" << uav_csma_busy_slots_
         << ",\"uav_csma_successful_contentions\":"
         << uav_csma_successful_contentions_
         << ",\"uav_csma_collision_events\":"
         << uav_csma_collision_events_
         << ",\"uav_csma_collided_datagrams\":"
         << uav_csma_collided_datagrams_
         << ",\"uav_udp_receiver_collision_failures\":"
         << uav_udp_receiver_collision_failures_
         << ",\"ideal_coalesce_window_ms\":"
         << 1000.0 * ideal_coalesce_window_s_
         << ",\"ideal_coalesced_messages\":"
         << ideal_coalesced_messages_
         << ",\"perfect_forwarding_calls\":"
         << perfect_forward_profile_calls_
         << ",\"perfect_forwarding_receivers\":"
         << perfect_forward_profile_receivers_
         << ",\"perfect_forwarding_mean_ms\":"
         << (perfect_forward_profile_calls_ == 0U ? 0.0 :
             perfect_forward_profile_total_ms_ /
                 static_cast<double>(perfect_forward_profile_calls_))
         << ",\"perfect_forwarding_max_ms\":"
         << perfect_forward_profile_max_ms_
         << ",\"ideal_logical_messages\":" << ideal_logical_messages_
         << ",\"ideal_statistical_transport_blocks\":"
         << ideal_statistical_transport_blocks_
         << ",\"ideal_statistical_bytes\":"
         << ideal_statistical_bytes_
         << ",\"active_link_publications\":"
         << active_link_publications_
         << ",\"active_link_samples_published\":"
         << active_link_samples_published_
         << ",\"ideal_direct_forwarded_packets\":"
         << ideal_direct_forwarded_packets_
         << ",\"lossless_nearest_forwarded_packets\":"
         << lossless_nearest_forwarded_packets_
         << ",\"lossless_range_forwarded_packets\":"
         << lossless_range_forwarded_packets_
         << ",\"lossless_nearest_position_unavailable_events\":"
         << lossless_nearest_position_unavailable_events_
         << ",\"sionna_direct_attempted_packets\":"
         << sionna_direct_attempted_packets
         << ",\"nearest_filtered_receivers\":" << nearest_filtered_receivers_
         << ",\"range_filtered_receivers\":" << range_filtered_receivers_
         << ",\"nearest_position_unavailable\":"
         << position_unavailable_receivers_
         << ",\"position_unavailable_receivers\":"
         << position_unavailable_receivers_
         << ",\"ap_enabled\":" << (ap_enabled_ ? "true" : "false")
         << ",\"ap_node_id\":" << (ap_enabled_ ? ap_node_id_ : -1)
         << ",\"attempted_packets\":" << attempted_packets_
         << ",\"delivered_packets\":" << delivered_packets_
         << ",\"dropped_no_link\":" << dropped_no_link_
         << ",\"dropped_per\":" << dropped_per_
         << ",\"dropped_queue\":" << dropped_queue_
         << ",\"dropped_ttl\":" << dropped_ttl_
         << ",\"retried_packets\":" << retried_packets_
         << ",\"attempted_bytes\":" << attempted_bytes_
         << ",\"delivered_bytes\":" << delivered_bytes_
         << ",\"queued_packets\":" << message.queued_packets
         << ",\"queued_bytes\":" << message.queued_bytes
         << ",\"mean_delivery_delay_ms\":"
         << message.mean_delivery_delay_ms
         << ",\"logical_attempted_packets\":"
         << logical_attempted_packets_
         << ",\"logical_delivered_packets\":"
         << logical_delivered_packets_
         << ",\"logical_attempted_bytes\":" << logical_attempted_bytes_
         << ",\"logical_delivered_bytes\":" << logical_delivered_bytes_
         << ",\"logical_delivery_ratio\":" << logical_delivery_ratio
         << ",\"mean_end_to_end_delay_ms\":" << mean_end_to_end_delay_ms
         << ",\"direct_attempted_packets\":" << direct_attempted_packets_
         << ",\"direct_delivered_packets\":" << direct_delivered_packets_
         << ",\"ap_uplink_attempted_packets\":"
         << ap_uplink_attempted_packets_
         << ",\"ap_uplink_delivered_packets\":"
         << ap_uplink_delivered_packets_
         << ",\"ap_downlink_attempted_packets\":"
         << ap_downlink_attempted_packets_
         << ",\"ap_downlink_delivered_packets\":"
         << ap_downlink_delivered_packets_
         << ",\"ap_global_updates_received\":"
         << ap_global_updates_received_
         << ",\"ap_selective_forwards_enqueued\":"
         << ap_selective_forwards_enqueued_
         << ",\"ap_relay_wins\":" << ap_relay_wins_
         << ",\"direct_delivery_wins\":" << direct_delivery_wins_
         << ",\"duplicates_suppressed\":" << duplicates_suppressed_
         << ",\"bs_round_robin_enabled\":"
         << (bs_round_robin_enabled_ ? "true" : "false")
         << ",\"rl_bs_scheduler_enabled\":"
         << (rl_bs_scheduler_enabled_ ? "true" : "false")
         << ",\"force_bs_perfect_delivery\":"
         << (force_bs_perfect_delivery_ ? "true" : "false")
         << ",\"bs_periodic_upload_request_enabled\":"
         << (bs_periodic_upload_request_enabled_ ? "true" : "false")
         << ",\"bs_periodic_upload_request_period_ms\":"
         << 1.0e3 * bs_periodic_upload_request_period_s_
         << ",\"bs_periodic_upload_chunks_per_request\":"
         << bs_periodic_upload_chunks_per_request_
         << ",\"bs_max_inflight_chunks_per_uav\":"
         << bs_max_inflight_chunks_per_uav_
         << ",\"bs_payload_precache_enabled\":"
         << (bs_payload_precache_enabled_ ? "true" : "false")
         << ",\"bs_payload_precache_batch_size\":"
         << bs_payload_precache_batch_size_
         << ",\"bs_max_payload_precache_inflight\":"
         << bs_max_payload_precache_inflight_
         << ",\"bs_payload_precache_timeout_ms\":"
         << 1.0e3 * bs_payload_precache_timeout_s_
         << ",\"bs_max_uplink_chunks_per_rl_slot\":"
         << bs_max_uplink_chunks_per_rl_slot_
         << ",\"bs_max_downlink_chunks_per_rl_slot\":"
         << bs_max_downlink_chunks_per_rl_slot_
         << ",\"bs_downlink_newest_missing_chunk_first\":"
         << (bs_downlink_newest_missing_chunk_first_ ? "true" : "false")
         << ",\"bs_chunk_budget_admission_fraction\":"
         << kBsMapAdmissionFraction
         << ",\"bs_chunk_budget_reference_bytes\":"
         << kRepresentativeMapChunkBytes
         << ",\"bs_adaptive_chunk_budget_slots\":"
         << bs_adaptive_chunk_budget_slots_
         << ",\"bs_adaptive_budget_channel_samples\":"
         << bs_adaptive_budget_channel_samples_
         << ",\"bs_adaptive_budget_unavailable_links\":"
         << bs_adaptive_budget_unavailable_links_
         << ",\"bs_last_adaptive_uplink_chunk_budget\":"
         << bs_last_adaptive_uplink_chunk_budget_
         << ",\"bs_last_adaptive_downlink_chunk_budget\":"
         << bs_last_adaptive_downlink_chunk_budget_
         << ",\"bs_last_adaptive_uplink_chunks_selected\":"
         << bs_last_adaptive_uplink_chunks_selected_
         << ",\"bs_last_adaptive_downlink_chunks_selected\":"
         << bs_last_adaptive_downlink_chunks_selected_
         << ",\"bs_adaptive_uplink_chunk_budget_total\":"
         << bs_adaptive_uplink_chunk_budget_total_
         << ",\"bs_adaptive_downlink_chunk_budget_total\":"
         << bs_adaptive_downlink_chunk_budget_total_
         << ",\"bs_adaptive_uplink_chunks_selected_total\":"
         << bs_adaptive_uplink_chunks_selected_total_
         << ",\"bs_adaptive_downlink_chunks_selected_total\":"
         << bs_adaptive_downlink_chunks_selected_total_
         << ",\"bs_adaptive_budget_mcs_counts\":{\"QPSK\":"
         << bs_adaptive_budget_mcs_counts_["QPSK"]
         << ",\"16QAM\":" << bs_adaptive_budget_mcs_counts_["16QAM"]
         << ",\"64QAM\":" << bs_adaptive_budget_mcs_counts_["64QAM"]
         << ",\"256QAM\":" << bs_adaptive_budget_mcs_counts_["256QAM"]
         << "}"
         << ",\"bs_uplink_slot_budget_exhaustions\":"
         << bs_uplink_slot_budget_exhaustions_
         << ",\"bs_downlink_slot_budget_exhaustions\":"
         << bs_downlink_slot_budget_exhaustions_
         << ",\"bs_control_ttl_s\":" << bs_control_ttl_s_
         << ",\"bs_periodic_upload_request_rounds\":"
         << bs_periodic_upload_request_rounds_
         << ",\"bs_periodic_upload_request_opportunities\":"
         << bs_periodic_upload_request_opportunities_
         << ",\"bs_periodic_upload_requests_scheduled\":"
         << bs_periodic_upload_requests_scheduled_
         << ",\"bs_periodic_upload_request_enqueue_failures\":"
         << bs_periodic_upload_request_enqueue_failures_
         << ",\"bs_periodic_upload_requests_suppressed_pending_control\":"
         << bs_periodic_upload_requests_suppressed_pending_control_
         << ",\"bs_periodic_upload_requests_suppressed_no_link\":"
         << bs_periodic_upload_requests_suppressed_no_link_
         << ",\"bs_periodic_upload_requests_suppressed_backpressure\":"
         << bs_periodic_upload_requests_suppressed_backpressure_
         << ",\"bs_periodic_upload_requests_delivered\":"
         << bs_periodic_upload_requests_delivered_
         << ",\"bs_periodic_upload_requests_failed\":"
         << bs_periodic_upload_requests_failed_
         << ",\"bs_periodic_upload_triggers\":"
         << bs_periodic_upload_triggers_
         << ",\"bs_all_to_all_relay_enabled\":"
         << (bs_all_to_all_relay_enabled_ ? "true" : "false")
         << ",\"pair_control_reservation_enabled\":"
         << (pair_control_reservation_enabled_ ? "true" : "false")
         << ",\"pair_control_reservation_s\":"
         << pair_control_reservation_s_
         << ",\"pair_control_reservations_active\":"
         << pair_control_reservations_.size()
         << ",\"pair_control_reservations_started\":"
         << pair_control_reservations_started_
         << ",\"pair_control_reservations_completed\":"
         << pair_control_reservations_completed_
         << ",\"pair_control_reservations_terminated\":"
         << pair_control_reservations_terminated_
         << ",\"pair_control_reservations_expired\":"
         << pair_control_reservations_expired_
         << ",\"pair_control_reserved_prepared_delivered\":"
         << pair_control_reserved_prepared_delivered_
         << ",\"pair_control_reserved_commits_delivered\":"
         << pair_control_reserved_commits_delivered_
         << ",\"pair_control_reserved_uplinks_scheduled\":"
         << pair_control_reserved_uplinks_scheduled_
         << ",\"pair_control_reserved_uplinks_delivered\":"
         << pair_control_reserved_uplinks_delivered_
         << ",\"pair_control_reserved_downlinks_scheduled\":"
         << pair_control_reserved_downlinks_scheduled_
         << ",\"pair_control_reserved_downlinks_delivered\":"
         << pair_control_reserved_downlinks_delivered_
         << ",\"pair_control_reserved_uplink_prb_slots\":"
         << pair_control_reserved_uplink_prb_slots_
         << ",\"pair_control_reserved_downlink_prb_slots\":"
         << pair_control_reserved_downlink_prb_slots_
         << ",\"pair_control_reserved_delivered_packets\":"
         << pair_control_reserved_delivered_packets_
         << ",\"pair_control_reserved_delivered_bytes\":"
         << pair_control_reserved_delivered_bytes_
         << ",\"pair_control_reserved_mean_hop_delay_ms\":"
         << (pair_control_reserved_delivered_packets_ > 0U
                 ? 1.0e3 * pair_control_reserved_cumulative_hop_delay_s_ /
                       static_cast<double>(
                           pair_control_reserved_delivered_packets_)
                 : 0.0)
         << ",\"pair_control_reserved_max_hop_delay_ms\":"
         << 1.0e3 * pair_control_reserved_max_hop_delay_s_
         << ",\"pair_control_reserved_mean_end_to_end_delay_ms\":"
         << (pair_control_reserved_downlinks_delivered_ > 0U
                 ? 1.0e3 *
                       pair_control_reserved_cumulative_end_to_end_delay_s_ /
                       static_cast<double>(
                           pair_control_reserved_downlinks_delivered_)
                 : 0.0)
         << ",\"pair_control_reserved_max_end_to_end_delay_ms\":"
         << 1.0e3 * pair_control_reserved_max_end_to_end_delay_s_
         << ",\"rl_bs_synchronous_mode\":"
         << (rl_bs_synchronous_mode_ ? "true" : "false")
         << ",\"rl_bs_event_driven_one_shot\":"
         << (rl_bs_event_driven_one_shot_ ? "true" : "false")
         << ",\"rl_event_applied_action_count\":"
         << rl_event_applied_action_count_
         << ",\"rl_event_completed_transition_count\":"
         << rl_event_completed_transition_count_
         << ",\"rl_event_rejected_action_count\":"
         << rl_event_rejected_action_count_
         << ",\"rl_event_last_delta_t_s\":" << rl_event_delta_t_s_
         << ",\"rl_bs_communication_slot_ms\":"
         << 1.0e3 * rl_bs_communication_slot_s_
         << ",\"rl_bs_decision_period_ms\":"
         << 1.0e3 * rl_bs_decision_period_s_
         << ",\"rl_bs_repeat_action_each_slot\":"
         << (rl_bs_repeat_action_each_slot_ ? "true" : "false")
         << ",\"rl_bs_slots_per_decision\":" << rl_slots_per_decision_
         << ",\"rl_communication_slot_index\":"
         << rl_communication_slot_index_
         << ",\"rl_decision_index\":" << rl_decision_index_
         << ",\"rl_last_action_held_slots\":"
         << rl_last_action_held_slots_
         << ",\"rl_missing_action_slots\":" << rl_missing_action_slots_
         << ",\"rl_bs_action_epoch\":" << rl_action_epoch_
         << ",\"rl_action_channel_version\":"
         << rl_action_source_communication_version_
         << ",\"rl_action_channel_snapshot_step_id\":"
         << (rl_action_channel_snapshot_
                 ? rl_action_channel_snapshot_->step_id
                 : 0U)
         << ",\"rl_action_channel_snapshots_cached\":"
         << action_channel_snapshots_.size()
         << ",\"rl_action_bound_packets_enqueued\":"
         << rl_action_bound_packets_enqueued_
         << ",\"rl_action_bound_transmission_attempts\":"
         << rl_action_bound_transmission_attempts_
         << ",\"rl_action_bound_packets_delivered\":"
         << rl_action_bound_packets_delivered_
         << ",\"rl_rejected_missing_channel_snapshot_actions\":"
         << rl_rejected_missing_channel_snapshot_actions_
         << ",\"rl_rejected_mismatched_channel_snapshot_actions\":"
         << rl_rejected_mismatched_channel_snapshot_actions_
         << ",\"rl_bs_state_sequence\":" << rl_state_sequence_
         << ",\"bs_uplink_prb_slots\":" << bs_uplink_prb_slots_
         << ",\"bs_downlink_prb_slots\":" << bs_downlink_prb_slots_
         << ",\"direct_u2u_prb_slots\":" << direct_u2u_prb_slots_
         << ",\"bs_active_uav\":" << bs_active_uav_
         << ",\"bs_round_robin_turns\":" << bs_round_robin_turns_
         << ",\"bs_upload_grants_delivered\":"
         << bs_upload_grants_delivered_
         << ",\"bs_upload_grants_failed\":" << bs_upload_grants_failed_
         << ",\"bs_incremental_chunks_scheduled_uplink\":"
         << bs_incremental_chunks_scheduled_uplink_
         << ",\"bs_incremental_chunks_received_uplink\":"
         << bs_incremental_chunks_received_uplink_
         << ",\"bs_chunk_payload_request_messages\":"
         << bs_chunk_payload_request_messages_
         << ",\"bs_chunk_payload_chunks_requested\":"
         << bs_chunk_payload_chunks_requested_
         << ",\"bs_chunk_payload_responses_received\":"
         << bs_chunk_payload_responses_received_
         << ",\"bs_chunk_payload_request_duplicates_suppressed\":"
         << bs_chunk_payload_request_duplicates_suppressed_
         << ",\"bs_chunk_payload_request_timeouts\":"
         << bs_chunk_payload_request_timeouts_
         << ",\"bs_chunk_payload_unsolicited_responses\":"
         << bs_chunk_payload_unsolicited_responses_
         << ",\"bs_upload_selection_backpressure_skips\":"
         << bs_upload_selection_backpressure_skips_
         << ",\"payload_precache_queued\":"
         << payload_precache_queued_
         << ",\"payload_precache_request_messages\":"
         << payload_precache_request_messages_
         << ",\"payload_precache_chunks_requested\":"
         << payload_precache_chunks_requested_
         << ",\"payload_precache_responses_received\":"
         << payload_precache_responses_received_
         << ",\"payload_precache_timeouts\":"
         << payload_precache_timeouts_
         << ",\"payload_precache_late_responses\":"
         << payload_precache_late_responses_
         << ",\"payload_precache_satisfied_by_peer_data\":"
         << payload_precache_satisfied_by_peer_data_
         << ",\"payload_direct_pushes_received\":"
         << payload_direct_pushes_received_
         << ",\"payload_direct_pushes_rejected\":"
         << payload_direct_pushes_rejected_
         << ",\"payload_precache_pending\":"
         << pending_payload_precache_keys_.size()
         << ",\"payload_precache_pending_peak\":"
         << payload_precache_pending_peak_
         << ",\"payload_precache_inflight\":"
         << inflight_payload_precache_.size()
         << ",\"payload_precache_inflight_peak\":"
         << payload_precache_inflight_peak_
         << ",\"payload_cache_entries\":"
         << chunk_repository_.size()
         << ",\"bs_missing_chunks_scheduled_downlink\":"
         << bs_missing_chunks_scheduled_downlink_
         << ",\"bs_missing_chunks_delivered_downlink\":"
         << bs_missing_chunks_delivered_downlink_
         << ",\"bs_peer_data_matches_uav_topics\":"
         << (bs_round_robin_enabled_ ? "true" : "false")
         << ",\"bs_relay_topic_policy\":\"chunk_data_drone_state_trajectory_global_assignment\""
         << ",\"bs_pair_control_relay_enabled\":"
         << (pair_control_reservation_enabled_ ? "true" : "false")
         << ",\"bs_peer_uplink_scheduled\":"
         << bs_peer_uplink_scheduled_
         << ",\"bs_peer_uplink_delivered\":"
         << bs_peer_uplink_delivered_
         << ",\"bs_peer_downlink_scheduled\":"
         << bs_peer_downlink_scheduled_
         << ",\"bs_peer_downlink_delivered\":"
         << bs_peer_downlink_delivered_
         << ",\"bs_peer_topic_counters\":{";
    for (std::size_t topic = 0; topic < kTopicPolicies.size(); ++topic) {
      if (topic) json << ',';
      json << '\"' << kTopicPolicies[topic].key << "\":{"
           << "\"uplink_scheduled\":"
           << bs_peer_uplink_scheduled_by_topic_[topic]
           << ",\"uplink_delivered\":"
           << bs_peer_uplink_delivered_by_topic_[topic]
           << ",\"downlink_scheduled\":"
           << bs_peer_downlink_scheduled_by_topic_[topic]
           << ",\"downlink_delivered\":"
           << bs_peer_downlink_delivered_by_topic_[topic] << '}';
    }
    json << '}'
         << ",\"bs_known_map_chunks\":" << bs_chunks_.size()
         << ",\"redundant_exploration_ratio\":"
         << task_quality.redundancy
         << ",\"bs_global_map_iou\":" << bs_map_metrics.iou
         << ",\"bs_global_map_coverage\":" << bs_map_metrics.coverage
         << ",\"bs_global_map_coverage_definition\":\"known BS SDF "
            "voxels / configured planning-box voxels\""
         << ",\"bs_global_map_known_voxels\":" << bs_map_metrics.known
         << ",\"bs_global_map_total_voxels\":" << bs_map_metrics.total
         << ",\"bs_global_map_occupied_voxels\":"
         << bs_map_metrics.occupied
         << ",\"bs_global_map_gt_intersection_voxels\":"
         << bs_map_metrics.intersection
         << ",\"ground_truth_occupied_voxels\":"
         << ground_truth_occupied_voxels_.size()
         << ",\"task_metric_observer_mode\":\""
         << task_metric_observer_mode_ << "\""
         << ",\"task_metric_observer_pending_events\":"
         << taskObserverQueueDepth()
         << ",\"task_metric_observer_peak_queue_depth\":"
         << task_observer_peak_queue_depth_
         << ",\"task_metric_observer_events_enqueued\":"
         << task_observer_events_enqueued_.load()
         << ",\"task_metric_observer_events_processed\":"
         << task_observer_events_processed_.load()
         << ",\"task_quality_history\":[";
    for (std::size_t index = 0; index < task_quality_history_.size(); ++index) {
      if (index) json << ',';
      const auto &sample = task_quality_history_[index];
      json << "{\"time_s\":" << sample.time_s
           << ",\"task_step\":" << sample.task_step
           << ",\"task_time_s\":" << sample.time_s
           << ",\"source_sim_time_s\":" << sample.source_sim_time_s
           << ",\"redundant_exploration_ratio\":" << sample.redundancy
           << ",\"bs_global_map_iou\":" << sample.map_iou
           << ",\"bs_global_map_coverage\":" << sample.map_coverage
           << ",\"bs_uplink_prb_slots\":"
           << sample.bs_uplink_prb_slots
           << ",\"bs_downlink_prb_slots\":"
           << sample.bs_downlink_prb_slots
           << ",\"direct_u2u_prb_slots\":"
           << sample.direct_u2u_prb_slots << '}';
    }
    json << ']'
         << ",\"bs_control_attempted_packets\":"
         << bs_control_attempted_packets_
         << ",\"bs_control_delivered_packets\":"
         << bs_control_delivered_packets_
         << ",\"bs_control_dropped_no_link\":"
         << bs_control_dropped_no_link_
         << ",\"bs_control_dropped_per\":"
         << bs_control_dropped_per_
         << ",\"bs_control_dropped_ttl\":"
         << bs_control_dropped_ttl_
         << ",\"bs_uplink_attempted_packets\":"
         << bs_uplink_attempted_packets_
         << ",\"bs_uplink_delivered_packets\":"
         << bs_uplink_delivered_packets_
         << ",\"bs_uplink_dropped_no_link\":"
         << bs_uplink_dropped_no_link_
         << ",\"bs_uplink_dropped_per\":"
         << bs_uplink_dropped_per_
         << ",\"bs_uplink_dropped_ttl\":"
         << bs_uplink_dropped_ttl_
         << ",\"bs_downlink_attempted_packets\":"
         << bs_downlink_attempted_packets_
         << ",\"bs_downlink_delivered_packets\":"
         << bs_downlink_delivered_packets_
         << ",\"bs_downlink_dropped_no_link\":"
         << bs_downlink_dropped_no_link_
         << ",\"bs_downlink_dropped_per\":"
         << bs_downlink_dropped_per_
         << ",\"bs_downlink_dropped_ttl\":"
         << bs_downlink_dropped_ttl_
         << ",\"phy\":{\"carrier_frequency_hz\":"
         << carrier_frequency_hz_
         << ",\"bandwidth_hz\":" << total_phy.bandwidth_hz
         << ",\"total_bandwidth_hz\":" << total_phy.bandwidth_hz
         << ",\"spectrum_partition_enabled\":"
         << (ap_enabled_ ? "true" : "false")
         << ",\"subcarrier_spacing_hz\":"
         << total_phy.subcarrier_spacing_hz
         << ",\"resource_blocks\":" << total_phy.resource_blocks
         << ",\"uav_broadcast_bandwidth_hz\":" << phy.bandwidth_hz
         << ",\"uav_broadcast_resource_blocks\":" << phy.resource_blocks
         << ",\"bs_bandwidth_hz\":" << bs_phy.bandwidth_hz
         << ",\"bs_resource_blocks\":" << bs_phy.resource_blocks
         << ",\"waveform\":\"CP-OFDM\",\"fec\":\"5G_NR_LDPC\""
         << ",\"channel_small_scale_fading\":\"none\""
         << ",\"bs_tx_power_dbm\":" << bs_tx_power_dbm_
         << ",\"uav_tx_power_dbm\":" << uav_tx_power_dbm_
         << ",\"max_retries\":" << max_retries_
         << ",\"bs_max_retries\":" << bs_max_retries_
         << ",\"random_seed\":" << random_seed_
         << ",\"bs_upa\":\"" << bs_array_rows_ << "x"
         << bs_array_cols_ << "\""
         << ",\"uav_upa\":\"" << uav_array_rows_ << "x"
         << uav_array_cols_ << "\""
         << ",\"target_initial_tbler\":"
         << phy.target_initial_tbler
         << ",\"fixed_mcs_index\":" << phy.fixed_mcs_index
         << ",\"fixed_mcs_modulation\":\""
         << (uses_fixed_mcs ? fixed_mcs.name : "adaptive") << "\""
         << ",\"fixed_mcs_code_rate\":"
         << (uses_fixed_mcs ? fixed_mcs.code_rate : -1.0)
         << ",\"fixed_mcs_target_snr_db\":"
         << (uses_fixed_mcs ? fixed_mcs.target_snr_db : -1.0)
         << ",\"bs_mcs_mode\":\"adaptive\""
         << ",\"bs_fixed_mcs_index\":" << bs_phy.fixed_mcs_index
         << ",\"mean_selected_initial_tbler\":" << mean_initial_tbler
         << ",\"mcs_counts\":{\"QPSK\":" << mcs_counts_["QPSK"]
         << ",\"16QAM\":" << mcs_counts_["16QAM"]
         << ",\"64QAM\":" << mcs_counts_["64QAM"]
         << ",\"256QAM\":" << mcs_counts_["256QAM"] << "}"
         << ",\"uav_mcs_counts\":{\"QPSK\":" << uav_mcs_counts_["QPSK"]
         << ",\"16QAM\":" << uav_mcs_counts_["16QAM"]
         << ",\"64QAM\":" << uav_mcs_counts_["64QAM"]
         << ",\"256QAM\":" << uav_mcs_counts_["256QAM"] << "}"
         << ",\"bs_mcs_counts\":{\"QPSK\":" << bs_mcs_counts_["QPSK"]
         << ",\"16QAM\":" << bs_mcs_counts_["16QAM"]
         << ",\"64QAM\":" << bs_mcs_counts_["64QAM"]
         << ",\"256QAM\":" << bs_mcs_counts_["256QAM"] << "}}"
         << ",\"sionna_exact_samples\":"
         << link_model_counts_["sionna_exact"]
         << ",\"sionna_cache_corrected_samples\":"
         << link_model_counts_["sionna_cache_corrected"]
         << ",\"radio_map_cache_samples\":"
         << link_model_counts_["radio_map_cache"] << "}";
    RCLCPP_INFO(get_logger(), "RACER_SIONNA_STATS %s", json.str().c_str());
  }

  std::string mode_;
  int drone_count_{};
  std::string topology_;
  int nearest_neighbor_count_{};
  int lossless_nearest_neighbor_count_{};
  double lossless_communication_range_m_{};
  bool lossless_control_only_{false};
  bool directed_message_unicast_{false};
  bool uav_udp_directed_unicast_{true};
  bool chunk_data_pre_enqueue_dedup_{false};
  int chunk_data_max_pending_per_link_{};
  double communication_range_m_{};
  double ideal_coalesce_window_s_{};
  bool preserve_ideal_direct_with_bs_{false};
  bool initial_assignment_perfect_delivery_{false};
  bool initial_assignment_perfect_via_bs_{false};
  bool initial_assignment_perfect_complete_{false};
  std::uint64_t initial_assignment_perfect_epoch_{};
  std::vector<bool> initial_assignment_epoch_acks_;
  double initial_assignment_perfect_completed_at_s_{};
  double active_link_hold_s_{};
  double active_link_publish_period_s_{};
  double carrier_frequency_hz_{};
  double bs_tx_power_dbm_{};
  double uav_tx_power_dbm_{};
  int bs_array_rows_{};
  int bs_array_cols_{};
  int uav_array_rows_{};
  int uav_array_cols_{};
  bool ap_enabled_{false};
  bool bs_round_robin_enabled_{false};
  bool nearest_neighbors_enabled_{false};
  bool distance_radius_enabled_{false};
  bool ideal_direct_enabled_{false};
  int ap_node_id_{};
  int radio_node_count_{};
  int random_seed_{};
  double base_latency_s_{};
  double jitter_s_{};
  std::size_t queue_capacity_bytes_{};
  int udp_header_bytes_{};
  std::string bs_channel_access_mode_;
  bool bs_downlink_source_owned_chunks_only_{false};
  std::vector<int> bs_csma_backoff_, bs_csma_cw_;
  double bs_csma_next_at_{};
  int bs_csma_guard_{};
  std::uint64_t bs_csma_idle_slots_{}, bs_csma_successful_contentions_{};
  std::uint64_t bs_csma_collision_events_{}, bs_csma_collided_packets_{};
  std::string uav_channel_access_mode_;
  int csma_cw_min_{15};
  int csma_cw_max_{1023};
  int csma_difs_slots_{2};
  bool uav_csma_enabled_{false};
  bool ofdma_diagnostic_logging_{true};
  int ofdma_log_packet_stride_{1000};
  int max_retries_{};
  int bs_max_retries_{};
  double retry_backoff_s_{};
  double bs_min_turn_s_{};
  std::size_t bs_control_bytes_{};
  int bs_max_downlink_chunks_per_turn_{};
  int bs_max_uplink_chunks_per_turn_{};
  int bs_max_uplink_chunks_per_rl_slot_{};
  int bs_max_downlink_chunks_per_rl_slot_{};
  bool bs_downlink_newest_missing_chunk_first_{false};
  bool rl_bs_scheduler_enabled_{false};
  bool force_bs_perfect_delivery_{false};
  bool bs_periodic_upload_request_enabled_{false};
  double bs_periodic_upload_request_period_s_{0.2};
  int bs_periodic_upload_chunks_per_request_{8};
  int bs_max_inflight_chunks_per_uav_{32};
  bool bs_payload_precache_enabled_{false};
  int bs_payload_precache_batch_size_{8};
  int bs_max_payload_precache_inflight_{16};
  double bs_payload_precache_timeout_s_{5.0};
  double bs_control_ttl_s_{2.0};
  bool bs_all_to_all_relay_enabled_{false};
  bool pair_control_reservation_enabled_{false};
  double pair_control_reservation_s_{1.8};
  bool rl_bs_synchronous_mode_{false};
  bool rl_bs_event_driven_one_shot_{false};
  std::string rl_bs_action_path_;
  std::string rl_bs_state_path_;
  std::string rl_shared_memory_root_;
  double rl_bs_communication_slot_s_{};
  double rl_bs_decision_period_s_{};
  bool rl_bs_repeat_action_each_slot_{true};
  double rl_llm_state_period_s_{};
  std::string ground_truth_occupied_voxels_path_;
  std::string observed_occupied_voxels_path_;
  bool require_ground_truth_map_{false};
  std::string task_metric_observer_mode_;
  double bs_map_resolution_{};
  std::array<double, 3> bs_map_size_{};
  double bs_map_ground_height_{};
  std::array<double, 3> bs_map_box_min_{};
  std::array<double, 3> bs_map_box_max_{};
  std::array<std::int64_t, 3> bs_map_voxel_num_{};
  std::array<std::int64_t, 3> bs_map_box_begin_{};
  std::array<std::int64_t, 3> bs_map_box_end_{};
  std::array<double, 3> bs_map_origin_{};
  std::uint64_t bs_full_map_voxels_{};
  std::size_t bs_planning_box_voxels_{};
  LinkModel model_;
  LinkModel uav_broadcast_model_;
  LinkModel bs_model_;
  BsOfdmaScheduler bs_ofdma_scheduler_;
  bool bs_ofdma_diagnostic_logging_{false};
  double bs_ofdma_next_at_{};
  std::uint64_t bs_ofdma_slot_index_{};
  std::uint64_t bs_ofdma_slots_processed_{};
  std::uint64_t bs_ofdma_active_slots_{};
  std::uint64_t bs_ofdma_uplink_slots_{};
  std::uint64_t bs_ofdma_downlink_slots_{};
  std::uint64_t bs_ofdma_allocated_prb_slots_{};
  std::uint64_t bs_ofdma_peak_parallel_links_{};
  int bs_ofdma_max_allocated_prbs_per_slot_{};
  SharedOfdmaScheduler ofdma_scheduler_;
  SharedOfdmaScheduler uav_broadcast_ofdma_scheduler_;
  bool shared_uav_radio_enabled_{false};
  double bs_radio_available_at_{};

  std::unordered_map<LinkKey, LinkQuality, LinkKeyHash> links_;
  std::unordered_map<std::uint64_t,
                     std::shared_ptr<const ActionChannelSnapshot>>
      action_channel_snapshots_;
  std::deque<std::uint64_t> action_channel_snapshot_versions_;
  std::unordered_map<LinkKey, LinkQueue, LinkKeyHash> queues_;
  std::vector<UavUdpQueue> uav_udp_queues_;
  std::deque<UdpDatagram> completed_uav_datagrams_;
  std::vector<std::mt19937> uav_sender_rngs_;
  std::vector<int> csma_backoff_slots_;
  std::vector<int> csma_contention_windows_;
  int csma_transmitting_sender_{-1};
  int csma_idle_guard_slots_{};
  std::uint64_t csma_collision_busy_slots_remaining_{};
  std::size_t csma_last_collision_senders_{};
  std::unordered_map<LinkKey, std::mt19937, LinkKeyHash> link_rngs_;
  std::unordered_map<LinkKey, double, LinkKeyHash> active_links_;
  bool active_links_dirty_{false};
  bool have_active_link_publish_sim_stamp_{false};
  double last_active_link_publish_sim_stamp_s_{};
  std::unordered_map<std::string, std::uint64_t> link_model_counts_;
  std::unordered_map<std::string, std::uint64_t> mcs_counts_;
  std::unordered_map<std::string, std::uint64_t> uav_mcs_counts_;
  std::unordered_map<std::string, std::uint64_t> bs_mcs_counts_;
  std::unordered_map<ChunkKey, CachedChunk, ChunkKeyHash> chunk_repository_;
  std::unordered_map<ChunkKey, std::uint64_t, ChunkKeyHash>
      bs_chunk_arrival_sequences_;
  std::uint64_t bs_chunk_arrival_sequence_{};
  std::vector<std::unordered_set<ChunkKey, ChunkKeyHash>> uav_chunks_;
  // Ordered materialized set differences make each PHY-slot selection cost
  // proportional to its quota instead of to the accumulated global map.
  std::vector<std::set<ChunkKey>> ordered_bs_missing_chunks_;
  std::vector<std::set<ChunkKey>> ordered_uav_missing_chunks_;
  std::vector<std::vector<ChunkKey>> downlink_priority_tables_;
  std::vector<std::array<double, 3>> downlink_priority_positions_;
  std::vector<bool> downlink_priority_position_valid_;
  std::vector<std::uint64_t> downlink_priority_versions_;
  std::uint64_t downlink_inventory_version_{};
  std::vector<
      std::unordered_map<ChunkKey, PendingBsChunkRequest, ChunkKeyHash>>
      pending_bs_chunk_requests_;
  std::deque<ChunkKey> pending_payload_precache_;
  std::unordered_set<ChunkKey, ChunkKeyHash> pending_payload_precache_keys_;
  std::unordered_map<ChunkKey, InflightPayloadPrecache, ChunkKeyHash>
      inflight_payload_precache_;
  std::vector<bool> bs_periodic_upload_request_pending_;
  std::vector<std::uint64_t> bs_uplink_budget_action_id_;
  std::vector<std::size_t> bs_uplink_chunks_selected_for_action_;
  std::vector<std::vector<std::uint64_t>> cached_pair_missing_bytes_;
  std::vector<std::vector<std::uint64_t>> cached_pair_missing_chunks_;
  std::vector<std::uint64_t> cached_bs_missing_bytes_;
  std::vector<std::uint64_t> cached_bs_missing_chunks_;
  std::vector<std::uint64_t> cached_bs_uav_missing_bytes_;
  std::vector<std::uint64_t> cached_bs_uav_missing_chunks_;
  std::vector<std::uint64_t> cached_uplink_queue_bytes_;
  std::vector<std::vector<std::uint64_t>> cached_relay_queue_bytes_;
  std::vector<std::unordered_set<std::uint32_t>> uav_observed_voxels_;
  std::vector<std::array<double, 3>> uav_positions_;
  std::vector<std::array<double, 3>> uav_velocities_;
  std::vector<double> uav_yaws_;
  std::vector<bool> uav_position_valid_;
  std::vector<std::vector<double>> last_uav_info_received_s_;
  std::vector<double> last_bs_info_received_s_;
  std::vector<TrajectorySummary> trajectory_summaries_;
  std::unordered_set<ChunkKey, ChunkKeyHash> bs_chunks_;
  std::unordered_set<std::uint32_t> bs_occupied_voxels_;
  std::vector<std::uint64_t> bs_known_voxel_bits_;
  std::unordered_set<std::uint32_t> bs_frontier_voxels_;
  std::size_t bs_known_planning_voxels_{};
  std::unordered_set<std::uint32_t> ground_truth_occupied_voxels_;
  // Async Task Metric incremental caches. Boundary metric publication reads
  // only these counts; it never traverses the growing global/UAV maps.
  std::unordered_map<std::uint32_t, std::uint32_t>
      uav_observation_refcounts_;
  std::size_t redundant_observation_sum_{};
  std::size_t bs_gt_intersection_voxels_{};
  std::vector<TaskQualitySample> task_quality_history_;
  TaskQualitySample latest_task_quality_;
  BsGlobalMapMetrics latest_bs_map_metrics_;
  bool have_last_llm_bs_coverage_{false};
  double last_llm_bs_coverage_{};
  bool rl_have_last_event_llm_state_time_{false};
  double rl_last_event_llm_state_time_s_{};
  mutable std::mutex task_metric_mutex_;
  mutable std::mutex task_metric_snapshot_mutex_;
  mutable std::mutex task_observer_queue_mutex_;
  std::condition_variable task_observer_condition_;
  std::deque<TaskObservation> task_observer_queue_;
  std::thread task_observer_thread_;
  bool task_observer_stopping_{false};
  std::size_t task_observer_peak_queue_depth_{};
  std::atomic<std::uint64_t> task_observer_events_enqueued_{};
  std::atomic<std::uint64_t> task_observer_events_processed_{};
  std::mutex communication_mutex_;
  std::mutex fast_state_queue_mutex_;
  std::deque<std::shared_ptr<FastBoundarySnapshot>> fast_state_queue_;
  rclcpp::CallbackGroup::SharedPtr scheduler_callback_group_;
  rclcpp::CallbackGroup::SharedPtr io_callback_group_;
  rclcpp::CallbackGroup::SharedPtr payload_cache_callback_group_;
  rclcpp::CallbackGroup::SharedPtr fast_state_callback_group_;
  rclcpp::CallbackGroup::SharedPtr telemetry_callback_group_;
  std::vector<std::vector<rclcpp::GenericPublisher::SharedPtr>> publishers_;
  std::vector<rclcpp::GenericSubscription::SharedPtr> subscriptions_;
  std::vector<rclcpp::GenericSubscription::SharedPtr>
      payload_cache_subscriptions_;
  std::vector<std::vector<std::shared_ptr<rclcpp::SerializedMessage>>>
      ap_latest_messages_;
  std::vector<std::vector<CachedPeerMessage>> uav_latest_peer_messages_;
  std::vector<std::vector<CachedPeerMessage>> bs_latest_peer_messages_;
  std::unordered_map<std::uint64_t, PairControlReservation>
      pair_control_reservations_;
  rclcpp::Subscription<LinkQualityArray>::SharedPtr link_subscription_;
  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>
      odometry_subscriptions_;
  rclcpp::Publisher<racer_sionna_interfaces::msg::CommStatistics>::SharedPtr
      statistics_publisher_;
  rclcpp::Publisher<LinkQualityArray>::SharedPtr active_link_publisher_;
  rclcpp::GenericPublisher::SharedPtr ap_recovery_status_publisher_;
  rclcpp::GenericSubscription::SharedPtr ap_recovery_command_subscription_;
  rclcpp::TimerBase::SharedPtr scheduler_timer_;
  rclcpp::TimerBase::SharedPtr fast_state_timer_;
  rclcpp::TimerBase::SharedPtr active_link_timer_;
  rclcpp::TimerBase::SharedPtr statistics_timer_;

  bool bs_turn_active_{false};
  bool bs_control_finished_{false};
  int bs_active_uav_{-1};
  int bs_next_uav_{0};
  double bs_turn_started_at_{};
  std::unordered_map<int, std::uint64_t> bs_uav_turns_;
  bool rl_have_decision_stamp_{false};
  double rl_last_decision_s_{};
  bool rl_async_clock_initialized_{false};
  bool rl_completed_interval_available_{false};
  std::uint64_t rl_async_observed_slot_{};
  std::uint64_t rl_late_boundary_recoveries_{};
  double rl_pending_boundary_stamp_s_{};
  std::unique_ptr<EventDrivenOneShotGate> event_driven_gate_;
  double rl_event_interval_start_sim_time_s_{};
  double rl_event_delta_t_s_{};
  double rl_event_action_age_s_{};
  std::uint64_t rl_event_last_state_slot_{};
  std::uint64_t rl_event_applied_action_count_{};
  std::uint64_t rl_event_completed_transition_count_{};
  std::uint64_t rl_event_rejected_action_count_{};
  std::uint64_t rl_event_executing_action_version_{};
  bool rl_sync_clock_initialized_{false};
  double rl_last_sync_stamp_s_{};
  bool rl_sync_boundary_frozen_{false};
  bool rl_sync_scheduler_have_stamp_{false};
  double rl_sync_scheduler_last_stamp_s_{};
  bool rl_have_statistics_sim_stamp_{false};
  double rl_last_statistics_sim_stamp_s_{};
  std::uint64_t rl_slots_per_decision_{1U};
  std::uint64_t rl_communication_slot_index_{};
  std::uint64_t rl_decision_index_{};
  std::uint64_t rl_action_decision_index_{};
  std::uint64_t rl_current_action_held_slots_{};
  std::uint64_t rl_last_action_held_slots_{};
  std::uint64_t rl_missing_action_slots_{};
  std::uint64_t rl_last_state_bs_uplink_prb_slots_{};
  std::uint64_t rl_last_state_bs_downlink_prb_slots_{};
  std::uint64_t rl_last_state_direct_u2u_prb_slots_{};
  bool rl_have_action_{false};
  bool rl_have_latest_action_{false};
  std::uint64_t rl_action_epoch_{};
  std::uint64_t rl_latest_action_epoch_{};
  std::uint64_t rl_state_sequence_{};
  std::uint64_t rl_shared_action_version_{};
  std::uint64_t rl_action_source_physical_version_{};
  std::uint64_t rl_action_source_communication_version_{};
  std::uint64_t rl_action_source_guidance_id_{};
  std::uint64_t rl_action_source_sim_step_{};
  std::uint64_t rl_action_source_step_id_{};
  std::uint64_t rl_action_policy_version_{};
  std::uint64_t rl_action_generated_wall_time_ns_{};
  double rl_action_source_sim_time_s_{};
  std::uint64_t rl_latest_action_source_physical_version_{};
  std::uint64_t rl_latest_action_source_communication_version_{};
  std::uint64_t rl_latest_action_source_guidance_id_{};
  std::uint64_t rl_latest_action_source_sim_step_{};
  std::uint64_t rl_latest_action_source_step_id_{};
  std::uint64_t rl_latest_action_source_slot_{};
  std::uint64_t rl_latest_policy_version_{};
  std::uint64_t rl_latest_action_generated_wall_time_ns_{};
  double rl_latest_action_source_sim_time_s_{};
  double rl_action_applied_sim_time_s_{};
  std::shared_ptr<const ActionChannelSnapshot> rl_action_channel_snapshot_;
  std::shared_ptr<const ActionChannelSnapshot>
      rl_latest_action_channel_snapshot_;
  std::uint64_t rl_rejected_missing_channel_snapshot_actions_{};
  std::uint64_t rl_rejected_mismatched_channel_snapshot_actions_{};
  std::unique_ptr<SharedJsonBlock> shared_physical_;
  std::unique_ptr<SharedJsonBlock> shared_physical_fast_;
  std::unique_ptr<SharedJsonBlock> shared_communication_;
  std::unique_ptr<SharedJsonRing> shared_transition_ring_;
  std::unique_ptr<SharedJsonRing> shared_task_metric_ring_;
  std::unique_ptr<SharedJsonBlock> shared_guidance_fast_;
  std::unique_ptr<SharedJsonBlock> shared_state_event_;
  std::unique_ptr<SharedJsonBlock> shared_action_;
  std::unique_ptr<SharedJsonBlock> shared_action_ack_;
  std::unique_ptr<SharedJsonBlock> shared_status_;
  std::vector<std::vector<bool>> rl_relay_action_;
  std::vector<bool> rl_upload_action_;
  std::vector<std::vector<bool>> rl_latest_relay_action_;
  std::vector<bool> rl_latest_upload_action_;

  double fast_rl_state_build_total_ms_{};
  std::uint64_t fast_rl_state_build_count_{};
  double fast_rl_state_build_max_ms_{};
  double fast_rl_state_publish_total_ms_{};
  std::uint64_t fast_rl_state_publish_count_{};
  double fast_rl_state_publish_max_ms_{};
  double action_apply_ack_total_ms_{};
  std::uint64_t action_apply_ack_count_{};
  double action_apply_ack_max_ms_{};
  std::uint64_t rl_action_bound_packets_enqueued_{};
  std::uint64_t rl_action_bound_transmission_attempts_{};
  std::uint64_t rl_action_bound_packets_delivered_{};
  double task_metric_compute_total_ms_{};
  std::uint64_t task_metric_compute_count_{};
  double task_metric_compute_max_ms_{};
  double missing_bytes_update_total_ms_{};
  std::uint64_t missing_bytes_update_count_{};
  double queue_cache_update_total_ms_{};
  std::uint64_t queue_cache_update_count_{};

  std::uint64_t attempted_packets_{};
  std::uint64_t delivered_packets_{};
  std::uint64_t dropped_no_link_{};
  std::uint64_t dropped_per_{};
  std::uint64_t dropped_queue_{};
  std::uint64_t dropped_ttl_{};
  std::uint64_t retried_packets_{};
  std::uint64_t attempted_bytes_{};
  std::uint64_t delivered_bytes_{};
  double cumulative_delay_s_{};

  std::uint64_t logical_attempted_packets_{};
  std::uint64_t logical_delivered_packets_{};
  std::uint64_t logical_attempted_bytes_{};
  std::uint64_t logical_delivered_bytes_{};
  double cumulative_end_to_end_delay_s_{};
  std::uint64_t direct_attempted_packets_{};
  std::uint64_t direct_delivered_packets_{};
  std::uint64_t ap_uplink_attempted_packets_{};
  std::uint64_t ap_uplink_delivered_packets_{};
  std::uint64_t ap_downlink_attempted_packets_{};
  std::uint64_t ap_downlink_delivered_packets_{};
  std::uint64_t ap_global_updates_received_{};
  std::uint64_t ap_selective_forwards_enqueued_{};
  std::uint64_t ap_relay_wins_{};
  std::uint64_t direct_delivery_wins_{};
  std::uint64_t duplicates_suppressed_{};
  std::uint64_t nearest_filtered_receivers_{};
  std::uint64_t range_filtered_receivers_{};
  std::uint64_t position_unavailable_receivers_{};
  std::uint64_t ideal_coalesced_messages_{};
  std::uint64_t ideal_direct_forwarded_packets_{};
  std::uint64_t ideal_logical_messages_{};
  std::uint64_t ideal_statistical_transport_blocks_{};
  std::uint64_t ideal_statistical_bytes_{};
  std::uint64_t perfect_forward_profile_calls_{};
  std::uint64_t perfect_forward_profile_receivers_{};
  double perfect_forward_profile_total_ms_{};
  double perfect_forward_profile_max_ms_{};
  std::uint64_t active_link_publications_{};
  std::uint64_t active_link_samples_published_{};
  std::uint64_t lossless_nearest_forwarded_packets_{};
  std::uint64_t lossless_range_forwarded_packets_{};
  std::uint64_t lossless_nearest_position_unavailable_events_{};
  std::uint64_t initial_assignment_perfect_drone_state_messages_{};
  std::uint64_t initial_assignment_perfect_global_assignment_messages_{};
  std::uint64_t initial_assignment_perfect_forwarded_packets_{};
  std::uint64_t initial_assignment_perfect_bs_uplinks_{};
  std::uint64_t initial_assignment_perfect_bs_downlinks_{};
  std::uint64_t directed_unicast_messages_{};
  std::uint64_t invalid_directed_messages_dropped_{};
  std::uint64_t chunk_enqueue_duplicates_suppressed_{};
  std::uint64_t chunk_enqueue_budget_drops_{};
  double cumulative_initial_tbler_{};
  std::uint64_t initial_tbler_samples_{};
  bool uav_ofdma_timeline_initialized_{false};
  double uav_ofdma_next_slot_start_{};
  std::uint64_t uav_ofdma_slot_index_{};
  std::uint64_t uav_udp_next_packet_id_{};
  std::uint64_t uav_udp_datagrams_enqueued_{};
  std::uint64_t uav_udp_unicast_datagrams_{};
  std::uint64_t uav_udp_multicast_datagrams_{};
  std::uint64_t uav_udp_datagrams_dropped_queue_{};
  std::uint64_t uav_udp_datagrams_dropped_ttl_{};
  std::uint64_t uav_udp_receiver_attempts_{};
  std::uint64_t uav_udp_receiver_successes_{};
  std::uint64_t uav_udp_receiver_per_failures_{};
  std::uint64_t uav_udp_receiver_no_link_failures_{};
  std::uint64_t uav_udp_receiver_half_duplex_failures_{};
  std::uint64_t uav_udp_physical_bytes_enqueued_{};
  std::uint64_t uav_udp_physical_bytes_transmitted_{};
  std::uint64_t uav_physical_transmissions_started_{};
  std::uint64_t uav_physical_transmissions_completed_{};
  std::uint64_t uav_physical_transmissions_with_success_{};
  std::uint64_t uav_tx_power_applications_{};
  std::uint64_t uav_ofdma_slots_processed_{};
  std::uint64_t uav_ofdma_active_slots_{};
  std::uint64_t uav_ofdma_prb_allocation_events_{};
  std::uint64_t uav_ofdma_allocated_prb_slots_{};
  std::uint64_t uav_ofdma_peak_active_senders_{};
  int uav_ofdma_max_allocated_prbs_per_slot_{};
  std::uint64_t uav_csma_idle_slots_{};
  std::uint64_t uav_csma_backoff_slots_{};
  std::uint64_t uav_csma_busy_slots_{};
  std::uint64_t uav_csma_successful_contentions_{};
  std::uint64_t uav_csma_collision_events_{};
  std::uint64_t uav_csma_collided_datagrams_{};
  std::uint64_t uav_udp_receiver_collision_failures_{};
  std::uint64_t bs_round_robin_turns_{};
  std::uint64_t bs_upload_grants_delivered_{};
  std::uint64_t bs_upload_grants_failed_{};
  bool bs_periodic_upload_clock_initialized_{false};
  double bs_next_periodic_upload_request_s_{0.2};
  std::uint64_t bs_periodic_upload_request_rounds_{};
  std::uint64_t bs_periodic_upload_request_opportunities_{};
  std::uint64_t bs_periodic_upload_requests_scheduled_{};
  std::uint64_t bs_periodic_upload_request_enqueue_failures_{};
  std::uint64_t bs_periodic_upload_requests_suppressed_pending_control_{};
  std::uint64_t bs_periodic_upload_requests_suppressed_no_link_{};
  std::uint64_t bs_periodic_upload_requests_suppressed_backpressure_{};
  std::uint64_t bs_periodic_upload_requests_delivered_{};
  std::uint64_t bs_periodic_upload_requests_failed_{};
  std::uint64_t bs_periodic_upload_triggers_{};
  std::uint64_t bs_incremental_chunks_scheduled_uplink_{};
  std::uint64_t bs_incremental_chunks_received_uplink_{};
  std::uint64_t bs_chunk_payload_request_messages_{};
  std::uint64_t bs_chunk_payload_chunks_requested_{};
  std::uint64_t bs_chunk_payload_responses_received_{};
  std::uint64_t bs_chunk_payload_request_duplicates_suppressed_{};
  std::uint64_t bs_chunk_payload_request_timeouts_{};
  std::uint64_t bs_chunk_payload_unsolicited_responses_{};
  std::uint64_t bs_upload_selection_backpressure_skips_{};
  std::uint64_t payload_precache_queued_{};
  std::uint64_t payload_precache_request_messages_{};
  std::uint64_t payload_precache_chunks_requested_{};
  std::uint64_t payload_precache_responses_received_{};
  std::uint64_t payload_precache_timeouts_{};
  std::uint64_t payload_precache_late_responses_{};
  std::uint64_t payload_precache_satisfied_by_peer_data_{};
  std::uint64_t payload_direct_pushes_received_{};
  std::uint64_t payload_direct_pushes_rejected_{};
  std::size_t payload_precache_pending_peak_{};
  std::size_t payload_precache_inflight_peak_{};
  std::uint64_t bs_adaptive_chunk_budget_slots_{};
  std::uint64_t bs_adaptive_budget_channel_samples_{};
  std::uint64_t bs_adaptive_budget_unavailable_links_{};
  std::size_t bs_last_adaptive_uplink_chunk_budget_{};
  std::size_t bs_last_adaptive_downlink_chunk_budget_{};
  std::size_t bs_last_adaptive_uplink_chunks_selected_{};
  std::size_t bs_last_adaptive_downlink_chunks_selected_{};
  std::uint64_t bs_adaptive_uplink_chunk_budget_total_{};
  std::uint64_t bs_adaptive_downlink_chunk_budget_total_{};
  std::uint64_t bs_adaptive_uplink_chunks_selected_total_{};
  std::uint64_t bs_adaptive_downlink_chunks_selected_total_{};
  std::unordered_map<std::string, std::uint64_t>
      bs_adaptive_budget_mcs_counts_;
  std::uint64_t bs_uplink_slot_budget_exhaustions_{};
  std::uint64_t bs_downlink_slot_budget_exhaustions_{};
  std::uint64_t bs_missing_chunks_scheduled_downlink_{};
  std::uint64_t bs_missing_chunks_delivered_downlink_{};
  std::uint64_t uav_peer_message_sequence_{};
  std::uint64_t bs_peer_uplink_scheduled_{};
  std::uint64_t bs_peer_uplink_delivered_{};
  std::uint64_t bs_peer_downlink_scheduled_{};
  std::uint64_t bs_peer_downlink_delivered_{};
  std::vector<std::uint64_t> bs_peer_uplink_scheduled_by_topic_;
  std::vector<std::uint64_t> bs_peer_uplink_delivered_by_topic_;
  std::vector<std::uint64_t> bs_peer_downlink_scheduled_by_topic_;
  std::vector<std::uint64_t> bs_peer_downlink_delivered_by_topic_;
  std::uint64_t bs_control_attempted_packets_{};
  std::uint64_t bs_control_delivered_packets_{};
  std::uint64_t bs_uplink_attempted_packets_{};
  std::uint64_t bs_uplink_delivered_packets_{};
  std::uint64_t bs_downlink_attempted_packets_{};
  std::uint64_t bs_downlink_delivered_packets_{};
  std::uint64_t bs_control_dropped_no_link_{};
  std::uint64_t bs_control_dropped_per_{};
  std::uint64_t bs_control_dropped_ttl_{};
  std::uint64_t bs_uplink_dropped_no_link_{};
  std::uint64_t bs_uplink_dropped_per_{};
  std::uint64_t bs_uplink_dropped_ttl_{};
  std::uint64_t bs_downlink_dropped_no_link_{};
  std::uint64_t bs_downlink_dropped_per_{};
  std::uint64_t bs_downlink_dropped_ttl_{};
  std::uint64_t bs_uplink_prb_slots_{};
  std::uint64_t bs_downlink_prb_slots_{};
  std::uint64_t direct_u2u_prb_slots_{};
  std::uint64_t pair_control_reservations_started_{};
  std::uint64_t pair_control_reservations_completed_{};
  std::uint64_t pair_control_reservations_terminated_{};
  std::uint64_t pair_control_reservations_expired_{};
  std::uint64_t pair_control_reserved_prepared_delivered_{};
  std::uint64_t pair_control_reserved_commits_delivered_{};
  std::uint64_t pair_control_reserved_uplinks_scheduled_{};
  std::uint64_t pair_control_reserved_uplinks_delivered_{};
  std::uint64_t pair_control_reserved_downlinks_scheduled_{};
  std::uint64_t pair_control_reserved_downlinks_delivered_{};
  std::uint64_t pair_control_reserved_uplink_prb_slots_{};
  std::uint64_t pair_control_reserved_downlink_prb_slots_{};
  std::uint64_t pair_control_reserved_delivered_packets_{};
  std::uint64_t pair_control_reserved_delivered_bytes_{};
  double pair_control_reserved_cumulative_hop_delay_s_{};
  double pair_control_reserved_max_hop_delay_s_{};
  double pair_control_reserved_cumulative_end_to_end_delay_s_{};
  double pair_control_reserved_max_end_to_end_delay_s_{};
};

}  // namespace racer_sionna_comm

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto proxy = std::make_shared<racer_sionna_comm::CommunicationProxy>();
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 4U);
  executor.add_node(proxy);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
