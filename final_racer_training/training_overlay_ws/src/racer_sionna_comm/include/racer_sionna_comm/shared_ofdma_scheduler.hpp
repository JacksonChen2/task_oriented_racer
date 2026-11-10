#pragma once

#include <vector>

namespace racer_sionna_comm {

struct PrbAllocation {
  int sender{};
  int first_prb{};
  int prb_count{};
};

// Deterministic, interference-free OFDMA allocation for the distributed
// UAV-to-UAV resource pool. Every returned allocation is disjoint and their
// total never exceeds total_prbs. The rotating start sender distributes both
// remainder PRBs and service fairly across slots.
class SharedOfdmaScheduler {
 public:
  explicit SharedOfdmaScheduler(int total_prbs);

  std::vector<PrbAllocation> allocate(
      const std::vector<int> &active_senders);
  int totalPrbs() const noexcept { return total_prbs_; }
  int nextSender() const noexcept { return next_sender_; }

 private:
  int total_prbs_{};
  int next_sender_{};
};

}  // namespace racer_sionna_comm
