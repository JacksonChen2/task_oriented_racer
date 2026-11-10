#include <racer_sionna_comm/shared_ofdma_scheduler.hpp>

#include <algorithm>
#include <stdexcept>

namespace racer_sionna_comm {

SharedOfdmaScheduler::SharedOfdmaScheduler(int total_prbs)
    : total_prbs_(total_prbs) {
  if (total_prbs_ <= 0) {
    throw std::invalid_argument("total PRBs must be positive");
  }
}

std::vector<PrbAllocation> SharedOfdmaScheduler::allocate(
    const std::vector<int> &active_senders) {
  std::vector<int> ordered = active_senders;
  std::sort(ordered.begin(), ordered.end());
  ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
  ordered.erase(
      std::remove_if(ordered.begin(), ordered.end(),
                     [](int sender) { return sender < 0; }),
      ordered.end());
  if (ordered.empty()) return {};

  auto first = std::lower_bound(ordered.begin(), ordered.end(), next_sender_);
  if (first == ordered.end()) first = ordered.begin();
  std::rotate(ordered.begin(), first, ordered.end());

  const int scheduled = std::min<int>(total_prbs_, ordered.size());
  const int equal_share = total_prbs_ / scheduled;
  const int remainder = total_prbs_ % scheduled;
  std::vector<PrbAllocation> allocations;
  allocations.reserve(static_cast<std::size_t>(scheduled));
  int first_prb = 0;
  for (int index = 0; index < scheduled; ++index) {
    const int count = equal_share + (index < remainder ? 1 : 0);
    allocations.push_back({ordered[static_cast<std::size_t>(index)],
                           first_prb, count});
    first_prb += count;
  }

  // When there is a remainder, the next sender after the extra-PRB group gets
  // first choice next slot. With an even split, rotate by one sender.
  const int advance = remainder > 0 ? remainder : 1;
  next_sender_ = ordered[static_cast<std::size_t>(advance % ordered.size())];
  return allocations;
}

}  // namespace racer_sionna_comm
