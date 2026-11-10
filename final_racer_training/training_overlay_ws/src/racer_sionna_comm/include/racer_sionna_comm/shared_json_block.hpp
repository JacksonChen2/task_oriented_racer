#pragma once

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <linux/futex.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

namespace racer_sionna_comm {

// Must remain byte-identical to agentic_crpo.shared_ipc._HEADER.
#pragma pack(push, 1)
struct SharedBlockHeader {
  char magic[8];
  std::uint32_t abi_version;
  std::uint32_t capacity;
  std::uint64_t version;
  std::uint32_t active_index;
  std::uint32_t lengths[2];
  std::uint32_t checksums[2];
  std::uint32_t futex_word;
  std::uint64_t sim_steps[2];
  double sim_times[2];
  std::uint32_t writer_pid;
  std::uint32_t flags;
};
#pragma pack(pop)

static_assert(sizeof(SharedBlockHeader) == 88U);
static_assert(offsetof(SharedBlockHeader, futex_word) == 44U);

struct SharedBlockSnapshot {
  std::uint64_t version{};
  std::uint64_t sim_step{};
  double sim_time_s{};
  std::uint32_t writer_pid{};
  std::string payload;
};

class SharedJsonBlock {
 public:
  static constexpr std::size_t kHeaderBytes = 128U;

  explicit SharedJsonBlock(const std::string &path) : path_(path) {
    try {
      fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
      if (fd_ < 0) fail("open");
      struct stat status {};
      if (::fstat(fd_, &status) != 0) fail("fstat");
      mapped_bytes_ = static_cast<std::size_t>(status.st_size);
      if (mapped_bytes_ < kHeaderBytes + 2U) {
        throw std::runtime_error("shared block is too small: " + path_);
      }
      mapping_ = ::mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd_, 0);
      if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        fail("mmap");
      }
      header_ = static_cast<SharedBlockHeader *>(mapping_);
      static constexpr char kMagic[8] = {
          'F', 'R', 'S', 'H', 'M', '0', '1', '\0'};
      if (std::memcmp(header_->magic, kMagic, sizeof(kMagic)) != 0 ||
          header_->abi_version != 1U ||
          mapped_bytes_ != kHeaderBytes + 2U * header_->capacity) {
        throw std::runtime_error("shared block ABI mismatch: " + path_);
      }
    } catch (...) {
      if (mapping_ != nullptr) ::munmap(mapping_, mapped_bytes_);
      if (fd_ >= 0) ::close(fd_);
      mapping_ = nullptr;
      fd_ = -1;
      throw;
    }
  }

  SharedJsonBlock(const SharedJsonBlock &) = delete;
  SharedJsonBlock &operator=(const SharedJsonBlock &) = delete;

  ~SharedJsonBlock() {
    if (mapping_ != nullptr) ::munmap(mapping_, mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
  }

  std::uint64_t version() const {
    lock(LOCK_SH);
    const auto value = header_->version;
    unlock();
    return value;
  }

  bool snapshot(SharedBlockSnapshot &output) const {
    lock(LOCK_SH);
    const auto version = header_->version;
    if (version == 0U) {
      unlock();
      return false;
    }
    const auto active = header_->active_index;
    if (active > 1U || header_->lengths[active] > header_->capacity) {
      unlock();
      throw std::runtime_error("invalid shared payload slot: " + path_);
    }
    const auto *source = static_cast<const char *>(mapping_) + kHeaderBytes +
                         active * header_->capacity;
    std::string payload(source, source + header_->lengths[active]);
    if (crc32(payload) != header_->checksums[active]) {
      unlock();
      throw std::runtime_error("shared payload checksum mismatch: " + path_);
    }
    output.version = version;
    output.sim_step = header_->sim_steps[active];
    output.sim_time_s = header_->sim_times[active];
    output.writer_pid = header_->writer_pid;
    output.payload = std::move(payload);
    unlock();
    return true;
  }

  std::uint64_t publish(const std::string &payload, std::uint64_t sim_step,
                        double sim_time_s, std::uint32_t flags = 0U) {
    if (payload.size() > header_->capacity) {
      throw std::runtime_error("shared payload exceeds capacity: " + path_);
    }
    lock(LOCK_EX);
    const std::uint32_t inactive = 1U - header_->active_index;
    auto *destination = static_cast<char *>(mapping_) + kHeaderBytes +
                        inactive * header_->capacity;
    std::memcpy(destination, payload.data(), payload.size());
    header_->lengths[inactive] = static_cast<std::uint32_t>(payload.size());
    header_->checksums[inactive] = crc32(payload);
    header_->sim_steps[inactive] = sim_step;
    header_->sim_times[inactive] = sim_time_s;
    header_->active_index = inactive;
    ++header_->version;
    ++header_->futex_word;
    header_->writer_pid = static_cast<std::uint32_t>(::getpid());
    header_->flags = flags;
    const auto committed = header_->version;
    unlock();
    if (::syscall(SYS_futex, &header_->futex_word, FUTEX_WAKE, INT_MAX,
                  nullptr, nullptr, 0) < 0) {
      fail("futex wake");
    }
    return committed;
  }

  void poke() {
    lock(LOCK_EX);
    ++header_->futex_word;
    unlock();
    if (::syscall(SYS_futex, &header_->futex_word, FUTEX_WAKE, INT_MAX,
                  nullptr, nullptr, 0) < 0) {
      fail("futex wake");
    }
  }

 private:
  static std::uint32_t crc32(const std::string &data) {
    std::uint32_t value = 0xFFFFFFFFU;
    for (const auto byte : data) {
      value ^= static_cast<std::uint8_t>(byte);
      for (int bit = 0; bit < 8; ++bit) {
        const std::uint32_t mask =
            static_cast<std::uint32_t>(-(static_cast<int>(value & 1U)));
        value = (value >> 1U) ^ (0xEDB88320U & mask);
      }
    }
    return value ^ 0xFFFFFFFFU;
  }

  [[noreturn]] void fail(const char *operation) const {
    throw std::runtime_error(std::string(operation) + " failed for " + path_ +
                             ": " + std::strerror(errno));
  }

  void lock(int operation) const {
    while (::flock(fd_, operation) != 0) {
      if (errno != EINTR) fail("flock");
    }
  }

  void unlock() const {
    while (::flock(fd_, LOCK_UN) != 0) {
      if (errno != EINTR) fail("flock unlock");
    }
  }

  std::string path_;
  int fd_{-1};
  void *mapping_{nullptr};
  std::size_t mapped_bytes_{};
  SharedBlockHeader *header_{nullptr};
};

// Must remain byte-identical to agentic_crpo.shared_ipc._RING_HEADER and
// _RING_SLOT_HEADER. The ring is single-writer and never waits for a reader.
#pragma pack(push, 1)
struct SharedRingHeader {
  char magic[8];
  std::uint32_t abi_version;
  std::uint32_t slot_capacity;
  std::uint32_t slot_count;
  std::uint32_t slot_header_size;
  std::uint64_t write_sequence;
  std::uint32_t futex_word;
  std::uint32_t writer_pid;
};

struct SharedRingSlotHeader {
  std::uint64_t sequence;
  std::uint64_t sim_step;
  double sim_time_s;
  std::uint32_t length;
  std::uint32_t checksum;
};
#pragma pack(pop)

static_assert(sizeof(SharedRingHeader) == 40U);
static_assert(offsetof(SharedRingHeader, futex_word) == 32U);
static_assert(sizeof(SharedRingSlotHeader) == 32U);

class SharedJsonRing {
 public:
  static constexpr std::size_t kHeaderBytes = 128U;

  explicit SharedJsonRing(const std::string &path) : path_(path) {
    try {
      fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
      if (fd_ < 0) fail("open");
      struct stat status {};
      if (::fstat(fd_, &status) != 0) fail("fstat");
      mapped_bytes_ = static_cast<std::size_t>(status.st_size);
      mapping_ = ::mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd_, 0);
      if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        fail("mmap");
      }
      header_ = static_cast<SharedRingHeader *>(mapping_);
      static constexpr char kMagic[8] = {
          'F', 'R', 'R', 'N', 'G', '0', '1', '\0'};
      slot_stride_ = sizeof(SharedRingSlotHeader) + header_->slot_capacity;
      if (std::memcmp(header_->magic, kMagic, sizeof(kMagic)) != 0 ||
          header_->abi_version != 1U ||
          header_->slot_header_size != sizeof(SharedRingSlotHeader) ||
          header_->slot_count < 2U ||
          mapped_bytes_ !=
              kHeaderBytes + header_->slot_count * slot_stride_) {
        throw std::runtime_error("shared ring ABI mismatch: " + path_);
      }
    } catch (...) {
      if (mapping_ != nullptr) ::munmap(mapping_, mapped_bytes_);
      if (fd_ >= 0) ::close(fd_);
      mapping_ = nullptr;
      fd_ = -1;
      throw;
    }
  }

  SharedJsonRing(const SharedJsonRing &) = delete;
  SharedJsonRing &operator=(const SharedJsonRing &) = delete;

  ~SharedJsonRing() {
    if (mapping_ != nullptr) ::munmap(mapping_, mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
  }

  std::uint64_t publish(const std::string &payload, std::uint64_t sim_step,
                        double sim_time_s) {
    if (payload.size() > header_->slot_capacity) {
      throw std::runtime_error("shared ring payload exceeds slot capacity: " +
                               path_);
    }
    lock(LOCK_EX);
    const std::uint64_t sequence = header_->write_sequence + 1U;
    const std::size_t index = static_cast<std::size_t>(
        (sequence - 1U) % header_->slot_count);
    auto *slot = reinterpret_cast<SharedRingSlotHeader *>(
        static_cast<char *>(mapping_) + kHeaderBytes + index * slot_stride_);
    auto *destination = reinterpret_cast<char *>(slot) + sizeof(*slot);
    std::memcpy(destination, payload.data(), payload.size());
    slot->sequence = sequence;
    slot->sim_step = sim_step;
    slot->sim_time_s = sim_time_s;
    slot->length = static_cast<std::uint32_t>(payload.size());
    slot->checksum = crc32(payload);
    header_->write_sequence = sequence;
    ++header_->futex_word;
    header_->writer_pid = static_cast<std::uint32_t>(::getpid());
    unlock();
    if (::syscall(SYS_futex, &header_->futex_word, FUTEX_WAKE, INT_MAX,
                  nullptr, nullptr, 0) < 0) {
      fail("futex wake");
    }
    return sequence;
  }

 private:
  static std::uint32_t crc32(const std::string &data) {
    std::uint32_t value = 0xFFFFFFFFU;
    for (const auto byte : data) {
      value ^= static_cast<std::uint8_t>(byte);
      for (int bit = 0; bit < 8; ++bit) {
        const std::uint32_t mask =
            static_cast<std::uint32_t>(-(static_cast<int>(value & 1U)));
        value = (value >> 1U) ^ (0xEDB88320U & mask);
      }
    }
    return value ^ 0xFFFFFFFFU;
  }

  [[noreturn]] void fail(const char *operation) const {
    throw std::runtime_error(std::string(operation) + " failed for " + path_ +
                             ": " + std::strerror(errno));
  }

  void lock(int operation) const {
    while (::flock(fd_, operation) != 0) {
      if (errno != EINTR) fail("flock");
    }
  }

  void unlock() const {
    while (::flock(fd_, LOCK_UN) != 0) {
      if (errno != EINTR) fail("flock unlock");
    }
  }

  std::string path_;
  int fd_{-1};
  void *mapping_{nullptr};
  std::size_t mapped_bytes_{};
  std::size_t slot_stride_{};
  SharedRingHeader *header_{nullptr};
};

}  // namespace racer_sionna_comm
