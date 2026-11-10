#include <racer_sionna_comm/link_model.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace racer_sionna_comm {

LinkModel::LinkModel(LinkModelConfig config) : config_(config) {
  const double occupied_bandwidth =
      12.0 * static_cast<double>(config_.resource_blocks) *
      config_.subcarrier_spacing_hz;
  if (config_.bandwidth_hz <= 0.0 ||
      config_.subcarrier_spacing_hz <= 0.0 || config_.resource_blocks <= 0 ||
      occupied_bandwidth > config_.bandwidth_hz ||
      config_.data_re_efficiency <= 0.0 ||
      config_.data_re_efficiency > 1.0 ||
      config_.target_initial_tbler <= 0.0 ||
      config_.target_initial_tbler >= 1.0 || config_.tbler_slope_db <= 0.0 ||
      config_.transport_block_bytes == 0U ||
      (config_.fixed_mcs_index != -1 && config_.fixed_mcs_index != 14 &&
       config_.fixed_mcs_index != 20)) {
    throw std::invalid_argument("invalid link model configuration");
  }
}

const McsEntry &LinkModel::selectMcs(double snr_db) const {
  if (config_.fixed_mcs_index == 14) {
    return kFixedMcs14;
  }
  if (config_.fixed_mcs_index == 20) {
    return kFixedMcs20;
  }
  const McsEntry *selected = &kMcsTable.front();
  for (const auto &candidate : kMcsTable) {
    if (snr_db + 1.0e-12 < candidate.target_snr_db) break;
    selected = &candidate;
  }
  return *selected;
}

double LinkModel::transportBlockErrorRate(double snr_db) const {
  const auto &mcs = selectMcs(snr_db);
  // Each MCS threshold is the calibrated SNR at which the initial LDPC
  // transport-block error rate is the configured target.  The logistic curve
  // represents the 5G NR LDPC waterfall without adding a fading process.
  const double odds_at_target =
      (1.0 - config_.target_initial_tbler) /
      config_.target_initial_tbler;
  const double exponent = std::clamp(
      (snr_db - mcs.target_snr_db) / config_.tbler_slope_db, -60.0, 60.0);
  return std::clamp(1.0 / (1.0 + odds_at_target * std::exp(exponent)),
                    0.0, 1.0);
}

double LinkModel::bitRate(double snr_db) const {
  return bitRate(snr_db, config_.resource_blocks);
}

double LinkModel::bitRate(double snr_db, int resource_blocks) const {
  if (resource_blocks <= 0 || resource_blocks > config_.resource_blocks) {
    throw std::invalid_argument("allocated resource blocks are out of range");
  }
  const auto &mcs = selectMcs(snr_db);
  const double slots_per_second =
      1000.0 * config_.subcarrier_spacing_hz / 15.0e3;
  constexpr double kOfdmSymbolsPerNormalCpSlot = 14.0;
  const double data_resource_elements_per_second =
      12.0 * static_cast<double>(resource_blocks) *
      kOfdmSymbolsPerNormalCpSlot * slots_per_second *
      config_.data_re_efficiency;
  return std::max(1.0e3,
                  data_resource_elements_per_second *
                      static_cast<double>(mcs.modulation_order) *
                      mcs.code_rate);
}

double LinkModel::bitsPerSlot(double snr_db, int resource_blocks) const {
  return bitRate(snr_db, resource_blocks) * slotDuration();
}

double LinkModel::slotDuration() const {
  return 1.0e-3 * 15.0e3 / config_.subcarrier_spacing_hz;
}

double LinkModel::serializationDelay(double snr_db, std::size_t bytes) const {
  const double unquantized =
      8.0 * static_cast<double>(bytes) / bitRate(snr_db);
  const double slot = slotDuration();
  return std::max(slot, std::ceil(unquantized / slot) * slot);
}

double LinkModel::packetErrorRate(double snr_db, std::size_t bytes) const {
  const double tbler = transportBlockErrorRate(snr_db);
  const std::size_t transport_blocks = transportBlockCount(bytes);
  return std::clamp(
      1.0 - std::pow(1.0 - tbler, static_cast<double>(transport_blocks)),
      0.0, 1.0);
}

double LinkModel::expectedReliablePacketAirtime(
    double snr_db, std::size_t bytes, int max_retries) const {
  if (max_retries < 0) {
    throw std::invalid_argument("maximum retries must be non-negative");
  }
  const double per = packetErrorRate(snr_db, bytes);
  double expected_attempts = 1.0;
  double retry_probability = per;
  for (int retry = 0; retry < max_retries; ++retry) {
    expected_attempts += retry_probability;
    retry_probability *= per;
  }
  return serializationDelay(snr_db, bytes) * expected_attempts;
}

std::size_t LinkModel::reliablePacketCapacity(
    double snr_db, std::size_t bytes, int max_retries,
    double available_airtime_s) const {
  if (!std::isfinite(available_airtime_s) || available_airtime_s < 0.0) {
    throw std::invalid_argument("available airtime must be finite and non-negative");
  }
  if (available_airtime_s == 0.0) return 0U;
  const double expected_airtime_s =
      expectedReliablePacketAirtime(snr_db, bytes, max_retries);
  return static_cast<std::size_t>(std::max(
      0.0,
      std::floor(available_airtime_s / expected_airtime_s + 1.0e-9)));
}

std::size_t LinkModel::transportBlockCount(std::size_t bytes) const noexcept {
  return std::max<std::size_t>(
      1U, (bytes + config_.transport_block_bytes - 1U) /
              config_.transport_block_bytes);
}

}  // namespace racer_sionna_comm
