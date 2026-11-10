#pragma once

#include <racer_sionna_comm/shared_ofdma_scheduler.hpp>

namespace racer_sionna_comm {

// One shared BS band, half-duplex TDD. Within each PHY slot several peers
// use disjoint PRBs; UL and DL never overlap. Each direction retains its own
// rotating equal-share allocator, including when there are more peers than PRBs.
class BsOfdmaScheduler {
 public:
  explicit BsOfdmaScheduler(int total_prbs)
      : uplink_(total_prbs), downlink_(total_prbs) {}

  struct Slot {
    bool uplink{true};
    std::vector<PrbAllocation> allocations;
  };

  Slot allocate(const std::vector<int> &uplink_peers,
                const std::vector<int> &downlink_peers) {
    if (uplink_peers.empty() && downlink_peers.empty()) return {};
    const bool uplink = downlink_peers.empty() ||
                        (!uplink_peers.empty() && prefer_uplink_);
    prefer_uplink_ = !uplink;
    return {uplink, uplink ? uplink_.allocate(uplink_peers)
                           : downlink_.allocate(downlink_peers)};
  }

 private:
  SharedOfdmaScheduler uplink_;
  SharedOfdmaScheduler downlink_;
  bool prefer_uplink_{true};
};

}  // namespace racer_sionna_comm
