#include <plan_env/map_ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <plan_env/sdf_map.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace fast_planner {

MapROS::MapROS() = default;
MapROS::~MapROS() = default;

void MapROS::setMap(SDFMap *map) { map_ = map; }

void MapROS::init() {
  node_.param("map_ros/frame_id", frame_id_, std::string("map"));
  node_.param("map_ros/traversed_clearance_radius",
      traversed_clearance_radius_, 0.0);
  node_.param("map_ros/coverage_diagnostic_period",
      coverage_diagnostic_period_, 2.0);
  node_.param("map_ros/esdf_max_update_rate_hz",
      esdf_max_update_rate_hz_, 2.0);
  node_.param("map_ros/esdf_profile_period_s",
      esdf_profile_period_s_, 2.0);
  if (!std::isfinite(esdf_max_update_rate_hz_) ||
      esdf_max_update_rate_hz_ <= 0.0) {
    ROS_WARN("Invalid ESDF maximum update rate %.3f Hz; using 2 Hz",
        esdf_max_update_rate_hz_);
    esdf_max_update_rate_hz_ = 2.0;
  }
  cloud_sub_ = node_.subscribe("/racer/sensor_points", 5,
      &MapROS::cloudCallback, this,
      ros::TransportHints().tcpNoDelay().bestEffort());
  odom_sub_ = node_.subscribe("/odom_world", 5,
      &MapROS::odometryCallback, this, ros::TransportHints().tcpNoDelay());
  esdf_timer_ = node_.createTimer(
      ros::Duration(1.0 / esdf_max_update_rate_hz_),
      &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(
      ros::Duration(0.2), &MapROS::visCallback, this);
  if (coverage_diagnostic_period_ > 0.0) {
    coverage_pub_ = node_.advertise<std_msgs::msg::String>(
        "~/mapping_coverage", 1);
    coverage_bitmap_pub_ = node_.advertise<std_msgs::msg::UInt8MultiArray>(
        "~/mapping_known_bitmap", 1);
    const Eigen::Vector3i coverage_size =
        map_->mp_->box_max_ - map_->mp_->box_min_;
    const std::uint64_t coverage_total =
        static_cast<std::uint64_t>(coverage_size.x()) *
        static_cast<std::uint64_t>(coverage_size.y()) *
        static_cast<std::uint64_t>(coverage_size.z());
    coverage_known_bitmap_.assign(
        static_cast<std::size_t>((coverage_total + 7U) / 8U), 0U);
    coverage_timer_ = node_.createTimer(
        ros::Duration(coverage_diagnostic_period_),
        &MapROS::coverageCallback, this);
  }
}

void MapROS::odometryCallback(const nav_msgs::OdometryConstPtr &message) {
  camera_pos_ << message->pose.pose.position.x, message->pose.pose.position.y,
      message->pose.pose.position.z;
  have_odom_ = true;
}

void MapROS::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &message) {
  if (!map_ || !have_odom_) return;
  pcl::fromROSMsg(*message, point_cloud_);
  if (point_cloud_.empty()) return;
  map_->inputPointCloud(point_cloud_, static_cast<int>(point_cloud_.size()), camera_pos_);
  // Keep occupancy and its inflated collision mask current at the sensor
  // rate. Only the Euclidean distance transform is rate-limited below.
  updateInflatedOccupancy();
  markESDFDirty();
  const std::size_t previous_size = traversed_positions_.size();
  recordTraversedPosition(camera_pos_);
  if (return_corridor_enabled_ && traversed_positions_.size() > previous_size) {
    Eigen::Vector3d update_min = camera_pos_;
    Eigen::Vector3d update_max = camera_pos_;
    const std::size_t finish = traversed_positions_.size() - 1;
    const std::size_t start = finish == 0 ? finish : finish - 1;
    clearCorridorSegment(traversed_positions_[start],
        traversed_positions_[finish], update_min, update_max);
    finishCorridorUpdate(update_min, update_max, false);
  }
}

void MapROS::updateInflatedOccupancy() {
  if (!map_ || !local_updated_) return;
  map_->clearAndInflateLocalMap();
  local_updated_ = false;
}

void MapROS::markESDFDirty() {
  if (!esdf_dirty_) esdf_dirty_since_ = std::chrono::steady_clock::now();
  esdf_dirty_ = true;
}

void MapROS::recordTraversedPosition(const Eigen::Vector3d &position) {
  if (traversed_clearance_radius_ <= 0.0 || !map_) return;
  if (traversed_positions_.empty() ||
      (position - traversed_positions_.back()).norm() >=
          0.5 * map_->mp_->resolution_)
    traversed_positions_.push_back(position);
}

void MapROS::clearCorridorSegment(const Eigen::Vector3d &segment_start,
    const Eigen::Vector3d &position, Eigen::Vector3d &update_min,
    Eigen::Vector3d &update_max) {
  if (traversed_clearance_radius_ <= 0.0 || !map_) return;

  const double resolution = map_->mp_->resolution_;
  const double segment_length = (position - segment_start).norm();
  const int sample_count = std::max(
      1, static_cast<int>(std::ceil(segment_length / (0.5 * resolution))));
  const int voxel_radius = static_cast<int>(
      std::ceil(traversed_clearance_radius_ / resolution));

  for (int sample = 0; sample <= sample_count; ++sample) {
    const double ratio = static_cast<double>(sample) / sample_count;
    const Eigen::Vector3d center =
        segment_start + ratio * (position - segment_start);
    Eigen::Vector3i center_index;
    map_->posToIndex(center, center_index);
    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
      for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
        for (int dz = -voxel_radius; dz <= voxel_radius; ++dz) {
          Eigen::Vector3i index = center_index + Eigen::Vector3i(dx, dy, dz);
          if (!map_->isInMap(index)) continue;
          Eigen::Vector3d voxel_center;
          map_->indexToPos(index, voxel_center);
          if ((voxel_center - center).norm() > traversed_clearance_radius_)
            continue;
          const int address = map_->toAddress(index);
          const double free_log_odds = map_->mp_->clamp_min_log_;
          map_->md_->occupancy_buffer_[address] = free_log_odds;
          map_->md_->occupancy_buffer_inflate_[address] = 0;
          update_min = update_min.cwiseMin(voxel_center);
          update_max = update_max.cwiseMax(voxel_center);
        }
  }
}

void MapROS::finishCorridorUpdate(const Eigen::Vector3d &update_min,
    const Eigen::Vector3d &update_max, bool synchronous) {
  Eigen::Vector3d bound_inflate(
      map_->mp_->local_bound_inflate_, map_->mp_->local_bound_inflate_, 0.0);
  Eigen::Vector3i local_min, local_max;
  map_->posToIndex(update_min - bound_inflate, local_min);
  map_->posToIndex(update_max + bound_inflate, local_max);
  map_->boundIndex(local_min);
  map_->boundIndex(local_max);
  map_->md_->local_bound_min_ = map_->md_->local_bound_min_.cwiseMin(local_min);
  map_->md_->local_bound_max_ = map_->md_->local_bound_max_.cwiseMax(local_max);
  map_->md_->update_min_ = map_->md_->update_min_.cwiseMin(update_min);
  map_->md_->update_max_ = map_->md_->update_max_.cwiseMax(update_max);
  map_->md_->all_min_ = map_->md_->all_min_.cwiseMin(update_min);
  map_->md_->all_max_ = map_->md_->all_max_.cwiseMax(update_max);
  local_updated_ = true;
  markESDFDirty();
  if (synchronous) {
    updateInflatedOccupancy();
    updateESDF(true);
  }
}

void MapROS::prepareReturnCorridor() {
  if (traversed_clearance_radius_ <= 0.0 || !map_ ||
      traversed_positions_.empty())
    return;
  if (return_corridor_enabled_) return;

  // This boundary is deliberately dormant throughout exploration.  Enabling
  // it earlier would expose the unknown surface of the measured swept tube as
  // artificial frontiers and therefore alter original RACER task selection.
  return_corridor_enabled_ = true;
  recordTraversedPosition(camera_pos_);
  Eigen::Vector3d update_min = traversed_positions_.front();
  Eigen::Vector3d update_max = traversed_positions_.front();
  for (std::size_t index = 1; index < traversed_positions_.size(); ++index)
    clearCorridorSegment(traversed_positions_[index - 1],
        traversed_positions_[index], update_min, update_max);
  if (traversed_positions_.size() == 1)
    clearCorridorSegment(traversed_positions_.front(),
        traversed_positions_.front(), update_min, update_max);
  finishCorridorUpdate(update_min, update_max, true);
}

void SDFMap::prepareReturnCorridor() { mr_->prepareReturnCorridor(); }

bool MapROS::updateESDFForPlanning(SDFMap *map) {
  if (!map || !map->mr_) return false;
  return map->mr_->updateESDF(true);
}

bool MapROS::updateESDF(bool planner_forced) {
  if (!map_) return false;
  updateInflatedOccupancy();
  if (!esdf_dirty_) return false;

  const auto now = std::chrono::steady_clock::now();
  if (!planner_forced && have_esdf_update_wall_) {
    const double elapsed_s = std::chrono::duration<double>(
        now - last_esdf_update_wall_).count();
    if (elapsed_s + 1.0e-9 < 1.0 / esdf_max_update_rate_hz_) return false;
  }
  const double dirty_age_ms = 1000.0 * std::chrono::duration<double>(
      now - esdf_dirty_since_).count();
  const auto started = std::chrono::steady_clock::now();
  map_->updateESDF3d();
  const double current_ms = 1000.0 * std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  esdf_dirty_ = false;
  have_esdf_update_wall_ = true;
  last_esdf_update_wall_ = std::chrono::steady_clock::now();
  ++esdf_updates_;
  if (planner_forced)
    ++esdf_planner_updates_;
  else
    ++esdf_timer_updates_;
  esdf_total_ms_ += current_ms;
  esdf_max_ms_ = std::max(esdf_max_ms_, current_ms);
  reportESDFProfile(current_ms, planner_forced, dirty_age_ms);
  return true;
}

void MapROS::reportESDFProfile(double current_ms, bool planner_forced,
    double dirty_age_ms) {
  const auto now = std::chrono::steady_clock::now();
  const bool report_due = last_esdf_profile_wall_.time_since_epoch().count() == 0 ||
      std::chrono::duration<double>(now - last_esdf_profile_wall_).count() >=
          esdf_profile_period_s_;
  if (!report_due) return;
  last_esdf_profile_wall_ = now;
  ROS_INFO(
      "RACER_ESDF_PROFILE updates=%llu timer_updates=%llu planner_updates=%llu "
      "current_ms=%.6f mean_ms=%.6f max_ms=%.6f dirty_age_ms=%.3f "
      "reason=%s max_rate_hz=%.3f",
      static_cast<unsigned long long>(esdf_updates_),
      static_cast<unsigned long long>(esdf_timer_updates_),
      static_cast<unsigned long long>(esdf_planner_updates_), current_ms,
      esdf_updates_ == 0 ? 0.0 : esdf_total_ms_ / esdf_updates_, esdf_max_ms_,
      dirty_age_ms, planner_forced ? "planner" : "timer",
      esdf_max_update_rate_hz_);
}

void MapROS::updateESDFCallback(const ros::TimerEvent &) {
  updateESDF(false);
}

void MapROS::visCallback(const ros::TimerEvent &) {}

void MapROS::coverageCallback(const ros::TimerEvent &) {
  if (!map_) return;

  // Read-only acceptance instrumentation.  This uses the same UNKNOWN test
  // as SDFMap::getOccupancy and never writes the occupancy buffers or any
  // planner input.  The configured planning box is half-open, matching
  // SDFMap::isInBox.
  const Eigen::Vector3i begin = map_->mp_->box_min_;
  const Eigen::Vector3i end = map_->mp_->box_max_;
  const Eigen::Vector3i map_begin = Eigen::Vector3i::Zero();
  const Eigen::Vector3i map_end = map_->mp_->map_voxel_num_;
  const Eigen::Vector3i scan_begin = begin.cwiseMax(map_begin);
  const Eigen::Vector3i scan_end = end.cwiseMin(map_end);
  std::uint64_t known = 0;
  const std::uint64_t size_y =
      static_cast<std::uint64_t>(end.y() - begin.y());
  const std::uint64_t size_z =
      static_cast<std::uint64_t>(end.z() - begin.z());
  const std::uint64_t total =
      static_cast<std::uint64_t>(end.x() - begin.x()) *
      size_y * size_z;
  if (coverage_known_bitmap_.size() !=
      static_cast<std::size_t>((total + 7U) / 8U)) {
    coverage_known_bitmap_.assign(
        static_cast<std::size_t>((total + 7U) / 8U), 0U);
  }
  for (int x = scan_begin.x(); x < scan_end.x(); ++x)
    for (int y = scan_begin.y(); y < scan_end.y(); ++y)
      for (int z = scan_begin.z(); z < scan_end.z(); ++z) {
        const double occupancy =
            map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)];
        if (occupancy < map_->mp_->clamp_min_log_ - 1.0e-3) continue;
        ++known;
        const std::uint64_t compact_address =
            (static_cast<std::uint64_t>(x - begin.x()) * size_y +
             static_cast<std::uint64_t>(y - begin.y())) *
                size_z +
            static_cast<std::uint64_t>(z - begin.z());
        coverage_known_bitmap_[static_cast<std::size_t>(
            compact_address / 8U)] |= static_cast<std::uint8_t>(
                1U << (compact_address % 8U));
      }

  const double ratio = total == 0 ? 0.0 :
      static_cast<double>(known) / static_cast<double>(total);
  std_msgs::msg::String message;
  std::ostringstream payload;
  payload << "{\"known_voxels\":" << known
          << ",\"total_voxels\":" << total
          << ",\"ratio\":" << std::setprecision(12) << ratio << "}";
  message.data = payload.str();
  coverage_pub_.publish(message);
  std_msgs::msg::UInt8MultiArray bitmap;
  bitmap.layout.dim.resize(1);
  bitmap.layout.dim[0].label = "planning_box_known_bitmap";
  bitmap.layout.dim[0].size = static_cast<std::uint32_t>(total);
  bitmap.layout.dim[0].stride =
      static_cast<std::uint32_t>(coverage_known_bitmap_.size());
  bitmap.data = coverage_known_bitmap_;
  coverage_bitmap_pub_.publish(bitmap);
  ROS_INFO("RACER_MAP_COVERAGE known=%llu total=%llu ratio=%.6f",
      static_cast<unsigned long long>(known),
      static_cast<unsigned long long>(total), ratio);
}

}  // namespace fast_planner
