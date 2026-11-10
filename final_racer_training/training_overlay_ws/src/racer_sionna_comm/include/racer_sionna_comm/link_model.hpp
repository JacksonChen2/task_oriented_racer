#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace racer_sionna_comm {

struct LinkModelConfig {
  double bandwidth_hz{100.0e6};
  double subcarrier_spacing_hz{120.0e3};
  int resource_blocks{66};
  double data_re_efficiency{0.82};
  double target_initial_tbler{0.10};
  double tbler_slope_db{1.35};
  std::size_t transport_block_bytes{1200};
  // -1 keeps the original SNR-adaptive abstraction.  Fixed MCS entries use
  // 3GPP TS 38.214 table 5.1.3.1-1 modulation orders and target code rates.
  int fixed_mcs_index{-1};
};

struct McsEntry {
  std::string_view name;
  int modulation_order{};
  double code_rate{};
  double target_snr_db{};
};

class LinkModel {
 public:
  explicit LinkModel(LinkModelConfig config);

  const McsEntry &selectMcs(double snr_db) const;
  double transportBlockErrorRate(double snr_db) const;
  double bitRate(double snr_db) const;
  double bitRate(double snr_db, int resource_blocks) const;
  double bitsPerSlot(double snr_db, int resource_blocks) const;
  double serializationDelay(double snr_db, std::size_t bytes) const;
  double packetErrorRate(double snr_db, std::size_t bytes) const;
  double expectedReliablePacketAirtime(
      double snr_db, std::size_t bytes, int max_retries) const;
  std::size_t reliablePacketCapacity(
      double snr_db, std::size_t bytes, int max_retries,
      double available_airtime_s) const;
  std::size_t transportBlockCount(std::size_t bytes) const noexcept;
  double slotDuration() const;
  const LinkModelConfig &config() const noexcept { return config_; }

  static constexpr std::array<McsEntry, 4> kMcsTable{{
      {"QPSK", 2, 0.4902, 1.0},
      {"16QAM", 4, 0.4785, 7.0},
      {"64QAM", 6, 0.6504, 14.0},
      {"256QAM", 8, 0.7363, 21.0},
  }};
  static constexpr McsEntry kFixedMcs14{
      "16QAM", 4, 616.0 / 1024.0, 9.0};
  static constexpr McsEntry kFixedMcs20{
      "64QAM", 6, 567.0 / 1024.0, 13.0};

 private:
  LinkModelConfig config_;
};

}  // namespace racer_sionna_comm
