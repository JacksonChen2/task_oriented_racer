#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using Vec3 = std::array<double, 3>;
using Vec4 = std::array<double, 4>;
using Mat3 = std::array<double, 9>;
using Clock = std::chrono::steady_clock;

constexpr double kMass = 0.98;
constexpr double kGravity = 9.81;
constexpr Vec3 kInertia{2.64e-3, 2.64e-3, 4.96e-3};
constexpr double kArmLength = 0.26;
constexpr double kThrustCoefficient = 8.98132e-9;
constexpr double kMomentCoefficient = 1.169367864e-10;
constexpr double kMotorTimeConstant = 1.0 / 30.0;
constexpr double kMinimumRpm = 1200.0;
constexpr double kMaximumRpm = 35000.0;
constexpr double kQuadraticDragCoefficient =
    0.1 * 3.14159265358979323846 * kArmLength * kArmLength;
constexpr Vec3 kAttitudeGain{1.0, 1.0, 1.0};
constexpr Vec3 kAngularRateGain{0.07, 0.07, 0.1};
constexpr Vec3 kVelocityGain{3.4, 3.4, 4.0};

struct LowerConstraint {
  double distance;
  Vec3 normal;
  double bound;
};

struct UpperConstraint {
  double distance;
  Vec3 direction;
  double bound;
};

struct PointBuffer {
  py::array owner;
  const char *data{nullptr};
  py::ssize_t rows{0};
  py::ssize_t row_stride{0};
  py::ssize_t column_stride{0};
  bool is_float32{false};
};

struct SweepConstraint {
  Vec3 direction;
  double distance;
};

struct Wrench {
  Vec3 force;
  Vec3 torque;
  Vec4 command_rpm;
  Vec4 motor_rpm;
  Vec4 motor_thrust;
};

double elapsed_ms(const Clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double dot(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const Vec3 &value) { return std::sqrt(dot(value, value)); }

Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

Vec3 limit_norm(Vec3 value, double maximum) {
  const double value_norm = norm(value);
  if (value_norm > maximum && maximum > 0.0) {
    const double scale = maximum / value_norm;
    for (double &entry : value) {
      entry *= scale;
    }
  }
  return value;
}

Vec3 normalized_or(const Vec3 &value, const Vec3 &fallback,
                   double epsilon = 1.0e-12) {
  const double value_norm = norm(value);
  if (value_norm < epsilon) {
    return fallback;
  }
  return {value[0] / value_norm, value[1] / value_norm, value[2] / value_norm};
}

Vec3 mat_vec(const Mat3 &matrix, const Vec3 &vector) {
  return {matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
          matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
          matrix[6] * vector[0] + matrix[7] * vector[1] +
              matrix[8] * vector[2]};
}

Vec3 mat_transpose_vec(const Mat3 &matrix, const Vec3 &vector) {
  return {matrix[0] * vector[0] + matrix[3] * vector[1] + matrix[6] * vector[2],
          matrix[1] * vector[0] + matrix[4] * vector[1] + matrix[7] * vector[2],
          matrix[2] * vector[0] + matrix[5] * vector[1] +
              matrix[8] * vector[2]};
}

Mat3 quaternion_matrix(const Vec4 &quaternion) {
  double w = quaternion[0];
  double x = quaternion[1];
  double y = quaternion[2];
  double z = quaternion[3];
  const double magnitude = std::sqrt(w * w + x * x + y * y + z * z);
  if (magnitude < 1.0e-12) {
    return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  }
  w /= magnitude;
  x /= magnitude;
  y /= magnitude;
  z /= magnitude;
  return {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
          2.0 * (x * z + y * w),       2.0 * (x * y + z * w),
          1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
          2.0 * (x * z - y * w),       2.0 * (y * z + x * w),
          1.0 - 2.0 * (x * x + y * y)};
}

Mat3 desired_rotation(const Vec3 &force_world, double desired_yaw) {
  Vec3 desired_z = normalized_or(force_world, {0.0, 0.0, 1.0}, 1.0e-9);
  const Vec3 heading{std::cos(desired_yaw), std::sin(desired_yaw), 0.0};
  Vec3 desired_y = cross(desired_z, heading);
  if (norm(desired_y) < 1.0e-9) {
    desired_y = {0.0, 1.0, 0.0};
  }
  desired_y = normalized_or(desired_y, {0.0, 1.0, 0.0});
  const Vec3 desired_x = cross(desired_y, desired_z);
  // np.column_stack((desired_x, desired_y, desired_z)), row major.
  return {desired_x[0], desired_y[0], desired_z[0], desired_x[1], desired_y[1],
          desired_z[1], desired_x[2], desired_y[2], desired_z[2]};
}

Vec3 read_vec3(const py::detail::unchecked_reference<double, 2> &array,
               py::ssize_t row) {
  return {array(row, 0), array(row, 1), array(row, 2)};
}

Vec4 read_vec4(const py::detail::unchecked_reference<double, 2> &array,
               py::ssize_t row) {
  return {array(row, 0), array(row, 1), array(row, 2), array(row, 3)};
}

double point_value(const PointBuffer &points, py::ssize_t row,
                   py::ssize_t column) {
  const char *address =
      points.data + row * points.row_stride + column * points.column_stride;
  if (points.is_float32) {
    return static_cast<double>(*reinterpret_cast<const float *>(address));
  }
  return *reinterpret_cast<const double *>(address);
}

std::vector<LowerConstraint> build_point_constraints(const Vec3 &position,
                                                     const Vec3 &velocity,
                                                     const PointBuffer &points,
                                                     double clearance) {
  struct Candidate {
    double distance;
    py::ssize_t index;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<std::size_t>(points.rows));
  for (py::ssize_t row = 0; row < points.rows; ++row) {
    const Vec3 point{point_value(points, row, 0), point_value(points, row, 1),
                     point_value(points, row, 2)};
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
        !std::isfinite(point[2])) {
      continue;
    }
    const Vec3 delta{position[0] - point[0], position[1] - point[1],
                     position[2] - point[2]};
    const double distance = norm(delta);
    if (distance > 1.0e-4 && distance <= 2.2) {
      candidates.push_back({distance, row});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.distance < b.distance;
            });
  if (candidates.size() > 64) {
    candidates.resize(64);
  }

  std::vector<LowerConstraint> constraints;
  constraints.reserve(candidates.size());
  for (const Candidate &candidate : candidates) {
    const Vec3 point{point_value(points, candidate.index, 0),
                     point_value(points, candidate.index, 1),
                     point_value(points, candidate.index, 2)};
    const Vec3 normal{(position[0] - point[0]) / candidate.distance,
                      (position[1] - point[1]) / candidate.distance,
                      (position[2] - point[2]) / candidate.distance};
    const double approach_speed = std::max(0.0, -dot(normal, velocity));
    const double stopping_allowance =
        approach_speed * approach_speed / (2.0 * std::max(0.1, 0.6)) +
        0.20 * approach_speed;
    const double lower_bound =
        -0.8 * (candidate.distance - (clearance + stopping_allowance));
    constraints.push_back({candidate.distance, normal, lower_bound});
  }
  std::stable_sort(constraints.begin(), constraints.end(),
                   [](const LowerConstraint &a, const LowerConstraint &b) {
                     return a.distance > b.distance;
                   });
  return constraints;
}

std::vector<LowerConstraint> build_aabb_constraints(
    const Vec3 &position, const Vec3 &velocity,
    const py::detail::unchecked_reference<double, 2> &obstacle_minimums,
    const py::detail::unchecked_reference<double, 2> &obstacle_maximums,
    double clearance) {
  std::vector<LowerConstraint> constraints;
  constraints.reserve(static_cast<std::size_t>(obstacle_minimums.shape(0)));
  for (py::ssize_t row = 0; row < obstacle_minimums.shape(0); ++row) {
    Vec3 closest{};
    Vec3 minimum{};
    Vec3 maximum{};
    for (int axis = 0; axis < 3; ++axis) {
      minimum[axis] = obstacle_minimums(row, axis);
      maximum[axis] = obstacle_maximums(row, axis);
      closest[axis] =
          std::min(std::max(position[axis], minimum[axis]), maximum[axis]);
    }
    Vec3 delta{position[0] - closest[0], position[1] - closest[1],
               position[2] - closest[2]};
    double distance = norm(delta);
    Vec3 normal{};
    double signed_distance = 0.0;
    if (distance > 1.0e-9) {
      normal = {delta[0] / distance, delta[1] / distance, delta[2] / distance};
      signed_distance = distance;
    } else {
      std::array<double, 6> face_distances{
          position[0] - minimum[0], position[1] - minimum[1],
          position[2] - minimum[2], maximum[0] - position[0],
          maximum[1] - position[1], maximum[2] - position[2]};
      const auto face_it =
          std::min_element(face_distances.begin(), face_distances.end());
      const int face = static_cast<int>(face_it - face_distances.begin());
      normal[face % 3] = face < 3 ? -1.0 : 1.0;
      signed_distance = -*face_it;
    }
    if (signed_distance > 1.0) {
      continue;
    }
    const double approach_speed = std::max(0.0, -dot(normal, velocity));
    const double stopping_allowance =
        approach_speed * approach_speed / (2.0 * std::max(0.1, 0.7)) +
        0.08 * approach_speed;
    const double lower_bound =
        -1.2 * (signed_distance - (clearance + stopping_allowance));
    constraints.push_back({signed_distance, normal, lower_bound});
  }
  std::stable_sort(constraints.begin(), constraints.end(),
                   [](const LowerConstraint &a, const LowerConstraint &b) {
                     return a.distance > b.distance;
                   });
  return constraints;
}

std::vector<LowerConstraint> build_flight_constraints(const Vec3 &position,
                                                      const Vec3 &velocity,
                                                      const Vec3 &minimum,
                                                      const Vec3 &maximum,
                                                      double clearance) {
  std::vector<LowerConstraint> constraints;
  constraints.reserve(6);
  for (int axis = 0; axis < 3; ++axis) {
    for (int side = 0; side < 2; ++side) {
      const double distance = side == 0 ? position[axis] - minimum[axis]
                                        : maximum[axis] - position[axis];
      Vec3 normal{0.0, 0.0, 0.0};
      normal[axis] = side == 0 ? 1.0 : -1.0;
      const double approach_speed = std::max(0.0, -dot(normal, velocity));
      const double stopping_allowance =
          approach_speed * approach_speed / (2.0 * std::max(0.1, 0.7)) +
          0.10 * approach_speed;
      const double lower_bound =
          -1.2 * (distance - (clearance + stopping_allowance));
      constraints.push_back({distance, normal, lower_bound});
    }
  }
  std::stable_sort(constraints.begin(), constraints.end(),
                   [](const LowerConstraint &a, const LowerConstraint &b) {
                     return a.distance > b.distance;
                   });
  return constraints;
}

std::vector<UpperConstraint>
build_sweep_constraints(const Vec3 &velocity,
                        const std::vector<SweepConstraint> &sweeps,
                        double clearance) {
  std::vector<UpperConstraint> constraints;
  constraints.reserve(sweeps.size());
  for (const SweepConstraint &sweep : sweeps) {
    const double direction_norm = norm(sweep.direction);
    if (direction_norm < 1.0e-8 || !std::isfinite(sweep.distance) ||
        sweep.distance < 0.0) {
      continue;
    }
    const Vec3 direction{sweep.direction[0] / direction_norm,
                         sweep.direction[1] / direction_norm,
                         sweep.direction[2] / direction_norm};
    const double approach_speed = std::max(0.0, dot(direction, velocity));
    const double stopping_allowance =
        approach_speed * approach_speed / (2.0 * std::max(0.1, 0.6)) +
        0.20 * approach_speed;
    const double upper_bound =
        0.8 * (sweep.distance - (clearance + stopping_allowance));
    constraints.push_back({sweep.distance, direction, upper_bound});
  }
  std::stable_sort(constraints.begin(), constraints.end(),
                   [](const UpperConstraint &a, const UpperConstraint &b) {
                     return a.distance > b.distance;
                   });
  return constraints;
}

Vec3 project_lower(Vec3 result, const std::vector<LowerConstraint> &constraints,
                   double speed_limit, int iterations) {
  result = limit_norm(result, speed_limit);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (const LowerConstraint &constraint : constraints) {
      const double violation =
          constraint.bound - dot(constraint.normal, result);
      if (violation > 0.0) {
        for (int axis = 0; axis < 3; ++axis) {
          result[axis] += violation * constraint.normal[axis];
        }
      }
    }
    result = limit_norm(result, speed_limit);
  }
  return result;
}

Vec3 project_upper(Vec3 result, const std::vector<UpperConstraint> &constraints,
                   double speed_limit, int iterations) {
  result = limit_norm(result, speed_limit);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (const UpperConstraint &constraint : constraints) {
      const double violation =
          dot(constraint.direction, result) - constraint.bound;
      if (violation > 0.0) {
        for (int axis = 0; axis < 3; ++axis) {
          result[axis] -= violation * constraint.direction[axis];
        }
      }
    }
    result = limit_norm(result, speed_limit);
  }
  return result;
}

Vec3 project_external(Vec3 result,
                      const std::vector<LowerConstraint> &point_constraints,
                      const std::vector<LowerConstraint> &flight_constraints,
                      const std::vector<UpperConstraint> &sweep_constraints,
                      double speed_limit) {
  result = project_lower(result, point_constraints, speed_limit, 4);
  if (!flight_constraints.empty()) {
    result = project_lower(result, flight_constraints, speed_limit, 3);
  }
  return project_upper(result, sweep_constraints, speed_limit, 4);
}

Vec3 cbf_swarm_filter(
    Vec3 result, py::ssize_t own_index, const Vec3 &position,
    const Vec3 &own_velocity,
    const py::detail::unchecked_reference<double, 2> &positions,
    const py::detail::unchecked_reference<double, 2> &velocities,
    double safe_distance, double speed_limit) {
  result = limit_norm(result, speed_limit);
  for (int iteration = 0; iteration < 6; ++iteration) {
    for (py::ssize_t peer_index = 0; peer_index < positions.shape(0);
         ++peer_index) {
      if (peer_index == own_index) {
        continue;
      }
      const Vec3 peer_position = read_vec3(positions, peer_index);
      const Vec3 peer_velocity = read_vec3(velocities, peer_index);
      Vec3 relative{position[0] - peer_position[0],
                    position[1] - peer_position[1],
                    position[2] - peer_position[2]};
      double distance = norm(relative);
      if (distance < 1.0e-6) {
        relative = {1.0, 0.0, 0.0};
        distance = 1.0e-6;
      }
      const Vec3 normal{relative[0] / distance, relative[1] / distance,
                        relative[2] / distance};
      const Vec3 relative_velocity{own_velocity[0] - peer_velocity[0],
                                   own_velocity[1] - peer_velocity[1],
                                   own_velocity[2] - peer_velocity[2]};
      const double approach_speed =
          std::max(0.0, -dot(normal, relative_velocity));
      const double stopping_allowance =
          approach_speed * approach_speed / (2.0 * std::max(0.1, 0.6)) +
          0.20 * approach_speed;
      const double dynamic_safe_distance = safe_distance + stopping_allowance;
      const double h =
          distance * distance - dynamic_safe_distance * dynamic_safe_distance;
      const double lower_bound =
          dot(normal, peer_velocity) - 0.8 * h / (2.0 * distance);
      const double violation = lower_bound - dot(normal, result);
      if (violation > 0.0) {
        for (int axis = 0; axis < 3; ++axis) {
          result[axis] += violation * normal[axis];
        }
      }
    }
    result = limit_norm(result, speed_limit);
  }
  return result;
}

Wrench velocity_motor_wrench(const Vec3 &desired_velocity,
                             const Vec3 &current_velocity,
                             const Vec4 &orientation,
                             const Vec3 &angular_velocity_world,
                             double desired_yaw, const Vec4 &current_motor_rpm,
                             double dt) {
  const Mat3 rotation = quaternion_matrix(orientation);
  Vec3 acceleration{};
  for (int axis = 0; axis < 3; ++axis) {
    acceleration[axis] = kVelocityGain[axis] *
                         (desired_velocity[axis] - current_velocity[axis]) /
                         kMass;
  }
  const double acceleration_norm = norm(acceleration);
  if (acceleration_norm > 1.4) {
    const double scale = 1.4 / acceleration_norm;
    for (double &entry : acceleration) {
      entry *= scale;
    }
  }
  const Vec3 desired_force_world{kMass * acceleration[0],
                                 kMass * acceleration[1],
                                 kMass * (acceleration[2] + kGravity)};
  const Mat3 desired = desired_rotation(desired_force_world, desired_yaw);

  Mat3 error_matrix{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double desired_t_rotation = 0.0;
      double rotation_t_desired = 0.0;
      for (int index = 0; index < 3; ++index) {
        desired_t_rotation +=
            desired[index * 3 + row] * rotation[index * 3 + column];
        rotation_t_desired +=
            rotation[index * 3 + row] * desired[index * 3 + column];
      }
      error_matrix[row * 3 + column] =
          0.5 * (desired_t_rotation - rotation_t_desired);
    }
  }
  const Vec3 attitude_error{error_matrix[7], error_matrix[2], error_matrix[3]};
  const Vec3 angular_velocity_body =
      mat_transpose_vec(rotation, angular_velocity_world);
  const Vec3 inertia_omega{kInertia[0] * angular_velocity_body[0],
                           kInertia[1] * angular_velocity_body[1],
                           kInertia[2] * angular_velocity_body[2]};
  const Vec3 gyroscopic = cross(angular_velocity_body, inertia_omega);
  Vec3 requested_moment{};
  for (int axis = 0; axis < 3; ++axis) {
    requested_moment[axis] =
        -kAttitudeGain[axis] * attitude_error[axis] -
        kAngularRateGain[axis] * angular_velocity_body[axis] + gyroscopic[axis];
  }

  double trace = 0.0;
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      trace += desired[row * 3 + column] * rotation[row * 3 + column];
    }
  }
  const double attitude_error_function = 0.5 * (3.0 - trace);
  const Vec3 body_z{rotation[2], rotation[5], rotation[8]};
  const double total_thrust =
      attitude_error_function < 1.0
          ? std::max(0.0, dot(desired_force_world, body_z))
          : 0.0;
  const double roll = requested_moment[0];
  const double pitch = requested_moment[1];
  const double yaw = requested_moment[2];
  const Vec4 rpm_squared{total_thrust / (4.0 * kThrustCoefficient) -
                             pitch / (2.0 * kArmLength * kThrustCoefficient) +
                             yaw / (4.0 * kMomentCoefficient),
                         total_thrust / (4.0 * kThrustCoefficient) +
                             pitch / (2.0 * kArmLength * kThrustCoefficient) +
                             yaw / (4.0 * kMomentCoefficient),
                         total_thrust / (4.0 * kThrustCoefficient) +
                             roll / (2.0 * kArmLength * kThrustCoefficient) -
                             yaw / (4.0 * kMomentCoefficient),
                         total_thrust / (4.0 * kThrustCoefficient) -
                             roll / (2.0 * kArmLength * kThrustCoefficient) -
                             yaw / (4.0 * kMomentCoefficient)};

  Wrench wrench{};
  const double decay = std::exp(-dt / kMotorTimeConstant);
  Vec4 squared{};
  for (int motor = 0; motor < 4; ++motor) {
    wrench.command_rpm[motor] = std::min(
        std::max(std::sqrt(std::max(0.0, rpm_squared[motor])), kMinimumRpm),
        kMaximumRpm);
    wrench.motor_rpm[motor] =
        wrench.command_rpm[motor] +
        (current_motor_rpm[motor] - wrench.command_rpm[motor]) * decay;
    squared[motor] = wrench.motor_rpm[motor] * wrench.motor_rpm[motor];
    wrench.motor_thrust[motor] = kThrustCoefficient * squared[motor];
  }
  wrench.force = {0.0, 0.0,
                  kThrustCoefficient *
                      (squared[0] + squared[1] + squared[2] + squared[3])};
  wrench.torque = {kThrustCoefficient * kArmLength * (squared[2] - squared[3]),
                   kThrustCoefficient * kArmLength * (squared[1] - squared[0]),
                   kMomentCoefficient *
                       (squared[0] + squared[1] - squared[2] - squared[3])};
  const double speed = norm(current_velocity);
  const Vec3 drag_world{
      -kQuadraticDragCoefficient * speed * current_velocity[0],
      -kQuadraticDragCoefficient * speed * current_velocity[1],
      -kQuadraticDragCoefficient * speed * current_velocity[2]};
  const Vec3 drag_local = mat_transpose_vec(rotation, drag_world);
  for (int axis = 0; axis < 3; ++axis) {
    wrench.force[axis] += drag_local[axis];
  }
  return wrench;
}

void require_shape(const py::buffer_info &info, py::ssize_t rows,
                   py::ssize_t columns, const char *name) {
  if (info.ndim != 2 || info.shape[0] != rows || info.shape[1] != columns) {
    throw py::value_error(std::string(name) + " must have shape [N," +
                          std::to_string(columns) + "]");
  }
}

PointBuffer point_buffer_from_object(py::handle value, py::ssize_t index) {
  py::array array = py::array::ensure(value);
  if (!array) {
    throw py::type_error("safety_points entries must be NumPy arrays");
  }
  const py::buffer_info info = array.request();
  if (info.ndim != 2 || info.shape[1] != 3) {
    throw py::value_error("safety_points[" + std::to_string(index) +
                          "] must have shape [M,3]");
  }
  const bool float32 = info.format == py::format_descriptor<float>::format();
  const bool float64 = info.format == py::format_descriptor<double>::format();
  if (!float32 && !float64) {
    throw py::type_error(
        "safety point arrays must have float32 or float64 dtype");
  }
  return {std::move(array), static_cast<const char *>(info.ptr),
          info.shape[0],    info.strides[0],
          info.strides[1],  float32};
}

py::dict solve_control_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> commands,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> orientations,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        linear_velocities,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        angular_velocities,
    py::array_t<double, py::array::c_style | py::array::forcecast> motor_rpm,
    py::sequence safety_points, py::sequence sweep_constraints,
    py::array_t<double, py::array::c_style | py::array::forcecast> yaw_commands,
    double dt, double speed_limit, double obstacle_clearance,
    double sweep_clearance, double swarm_safe_distance, bool external_scene,
    py::object flight_minimum, py::object flight_maximum,
    py::object obstacle_minimums, py::object obstacle_maximums) {
  const py::buffer_info command_info = commands.request();
  if (command_info.ndim != 2 || command_info.shape[1] != 3) {
    throw py::value_error("commands must have shape [N,3]");
  }
  const py::ssize_t count = command_info.shape[0];
  require_shape(positions.request(), count, 3, "positions");
  require_shape(orientations.request(), count, 4, "orientations");
  require_shape(linear_velocities.request(), count, 3, "linear_velocities");
  require_shape(angular_velocities.request(), count, 3, "angular_velocities");
  require_shape(motor_rpm.request(), count, 4, "motor_rpm");
  if (yaw_commands.ndim() != 1 || yaw_commands.shape(0) != count) {
    throw py::value_error("yaw_commands must have shape [N]");
  }
  if (py::len(safety_points) != count || py::len(sweep_constraints) != count) {
    throw py::value_error(
        "safety_points and sweep_constraints must have N entries");
  }

  std::vector<PointBuffer> point_buffers;
  std::vector<std::vector<SweepConstraint>> sweep_buffers(
      static_cast<std::size_t>(count));
  point_buffers.reserve(static_cast<std::size_t>(count));
  for (py::ssize_t index = 0; index < count; ++index) {
    point_buffers.push_back(
        point_buffer_from_object(safety_points[index], index));
    py::sequence entries =
        py::reinterpret_borrow<py::sequence>(sweep_constraints[index]);
    auto &output = sweep_buffers[static_cast<std::size_t>(index)];
    output.reserve(static_cast<std::size_t>(py::len(entries)));
    for (py::handle entry_handle : entries) {
      py::sequence entry = py::reinterpret_borrow<py::sequence>(entry_handle);
      if (py::len(entry) != 2) {
        throw py::value_error(
            "each sweep constraint must be (direction, distance)");
      }
      py::sequence direction = py::reinterpret_borrow<py::sequence>(entry[0]);
      if (py::len(direction) != 3) {
        throw py::value_error("sweep direction must have three values");
      }
      output.push_back(
          {{py::cast<double>(direction[0]), py::cast<double>(direction[1]),
            py::cast<double>(direction[2])},
           py::cast<double>(entry[1])});
    }
  }

  bool has_flight_volume =
      !flight_minimum.is_none() && !flight_maximum.is_none();
  Vec3 flight_min{};
  Vec3 flight_max{};
  if (has_flight_volume) {
    const auto lower = py::cast<std::vector<double>>(flight_minimum);
    const auto upper = py::cast<std::vector<double>>(flight_maximum);
    if (lower.size() != 3 || upper.size() != 3) {
      throw py::value_error("flight volume bounds must have three values");
    }
    std::copy_n(lower.begin(), 3, flight_min.begin());
    std::copy_n(upper.begin(), 3, flight_max.begin());
  }

  py::array_t<double, py::array::c_style | py::array::forcecast>
      obstacle_min_array;
  py::array_t<double, py::array::c_style | py::array::forcecast>
      obstacle_max_array;
  if (!external_scene) {
    if (obstacle_minimums.is_none() || obstacle_maximums.is_none()) {
      throw py::value_error(
          "AABB bounds are required without an external scene");
    }
    obstacle_min_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            obstacle_minimums);
    obstacle_max_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            obstacle_maximums);
    if (!obstacle_min_array || !obstacle_max_array ||
        obstacle_min_array.ndim() != 2 || obstacle_min_array.shape(1) != 3 ||
        obstacle_max_array.ndim() != 2 || obstacle_max_array.shape(1) != 3 ||
        obstacle_min_array.shape(0) != obstacle_max_array.shape(0)) {
      throw py::value_error("obstacle bounds must both have shape [K,3]");
    }
  } else {
    obstacle_min_array = py::array_t<double>(
        py::array::ShapeContainer{py::ssize_t{0}, py::ssize_t{3}});
    obstacle_max_array = py::array_t<double>(
        py::array::ShapeContainer{py::ssize_t{0}, py::ssize_t{3}});
  }

  py::array_t<double> applied_commands({count, py::ssize_t{3}});
  py::array_t<double> forces({count, py::ssize_t{3}});
  py::array_t<double> torques({count, py::ssize_t{3}});
  py::array_t<double> command_rpms({count, py::ssize_t{4}});
  py::array_t<double> output_motor_rpms({count, py::ssize_t{4}});
  py::array_t<double> motor_thrusts({count, py::ssize_t{4}});
  py::array_t<bool> interventions({count});
  py::array_t<double> constraint_build_ms({count});
  py::array_t<double> obstacle_projection_ms({count});
  py::array_t<double> swarm_cbf_ms({count});
  py::array_t<double> so3_ms({count});

  const auto command_values = commands.unchecked<2>();
  const auto position_values = positions.unchecked<2>();
  const auto orientation_values = orientations.unchecked<2>();
  const auto velocity_values = linear_velocities.unchecked<2>();
  const auto angular_velocity_values = angular_velocities.unchecked<2>();
  const auto input_motor_values = motor_rpm.unchecked<2>();
  const auto yaw_values = yaw_commands.unchecked<1>();
  const auto obstacle_min_values = obstacle_min_array.unchecked<2>();
  const auto obstacle_max_values = obstacle_max_array.unchecked<2>();
  auto applied_values = applied_commands.mutable_unchecked<2>();
  auto force_values = forces.mutable_unchecked<2>();
  auto torque_values = torques.mutable_unchecked<2>();
  auto command_rpm_values = command_rpms.mutable_unchecked<2>();
  auto output_motor_values = output_motor_rpms.mutable_unchecked<2>();
  auto thrust_values = motor_thrusts.mutable_unchecked<2>();
  auto intervention_values = interventions.mutable_unchecked<1>();
  auto constraint_time_values = constraint_build_ms.mutable_unchecked<1>();
  auto obstacle_time_values = obstacle_projection_ms.mutable_unchecked<1>();
  auto cbf_time_values = swarm_cbf_ms.mutable_unchecked<1>();
  auto so3_time_values = so3_ms.mutable_unchecked<1>();

  int thread_count = 1;
#ifdef _OPENMP
  thread_count =
      std::max(1, std::min(static_cast<int>(count), omp_get_max_threads()));
#endif
  {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static)                                      \
    num_threads(thread_count) if (count > 1)
    for (py::ssize_t index = 0; index < count; ++index) {
      const Vec3 requested = read_vec3(command_values, index);
      const Vec3 position = read_vec3(position_values, index);
      const Vec3 velocity = read_vec3(velocity_values, index);
      const Vec4 orientation = read_vec4(orientation_values, index);
      const Vec3 angular_velocity = read_vec3(angular_velocity_values, index);
      const Vec4 input_motor_rpm = read_vec4(input_motor_values, index);

      const auto constraint_started = Clock::now();
      std::vector<LowerConstraint> obstacle_constraints;
      std::vector<LowerConstraint> flight_constraints;
      std::vector<UpperConstraint> sweep_constraint_values;
      if (external_scene) {
        obstacle_constraints = build_point_constraints(
            position, velocity, point_buffers[static_cast<std::size_t>(index)],
            obstacle_clearance);
        if (has_flight_volume) {
          flight_constraints = build_flight_constraints(
              position, velocity, flight_min, flight_max, obstacle_clearance);
        }
        sweep_constraint_values = build_sweep_constraints(
            velocity, sweep_buffers[static_cast<std::size_t>(index)],
            sweep_clearance);
      } else {
        obstacle_constraints =
            build_aabb_constraints(position, velocity, obstacle_min_values,
                                   obstacle_max_values, obstacle_clearance);
      }
      constraint_time_values(index) = elapsed_ms(constraint_started);

      const auto first_obstacle_started = Clock::now();
      Vec3 applied =
          external_scene
              ? project_external(requested, obstacle_constraints,
                                 flight_constraints, sweep_constraint_values,
                                 speed_limit)
              : project_lower(requested, obstacle_constraints, speed_limit, 4);
      obstacle_time_values(index) = elapsed_ms(first_obstacle_started);

      const auto cbf_started = Clock::now();
      applied =
          cbf_swarm_filter(applied, index, position, velocity, position_values,
                           velocity_values, swarm_safe_distance, speed_limit);
      cbf_time_values(index) = elapsed_ms(cbf_started);

      // CBF can point a velocity back toward an obstacle. The second obstacle
      // projection is therefore intentionally retained. Its half spaces are
      // command-independent and are exactly the same objects built above.
      const auto second_obstacle_started = Clock::now();
      applied =
          external_scene
              ? project_external(applied, obstacle_constraints,
                                 flight_constraints, sweep_constraint_values,
                                 speed_limit)
              : project_lower(applied, obstacle_constraints, speed_limit, 4);
      obstacle_time_values(index) += elapsed_ms(second_obstacle_started);

      const auto so3_started = Clock::now();
      const Wrench wrench = velocity_motor_wrench(
          applied, velocity, orientation, angular_velocity, yaw_values(index),
          input_motor_rpm, dt);
      so3_time_values(index) = elapsed_ms(so3_started);

      Vec3 command_delta{};
      for (int axis = 0; axis < 3; ++axis) {
        applied_values(index, axis) = applied[axis];
        force_values(index, axis) = wrench.force[axis];
        torque_values(index, axis) = wrench.torque[axis];
        command_delta[axis] = applied[axis] - requested[axis];
      }
      for (int motor = 0; motor < 4; ++motor) {
        command_rpm_values(index, motor) = wrench.command_rpm[motor];
        output_motor_values(index, motor) = wrench.motor_rpm[motor];
        thrust_values(index, motor) = wrench.motor_thrust[motor];
      }
      intervention_values(index) = norm(command_delta) > 1.0e-3;
    }
  }

  py::dict result;
  result["applied_commands"] = std::move(applied_commands);
  result["forces"] = std::move(forces);
  result["torques"] = std::move(torques);
  result["command_rpm"] = std::move(command_rpms);
  result["motor_rpm"] = std::move(output_motor_rpms);
  result["motor_thrust"] = std::move(motor_thrusts);
  result["intervened"] = std::move(interventions);
  result["constraint_build_ms"] = std::move(constraint_build_ms);
  result["obstacle_projection_ms"] = std::move(obstacle_projection_ms);
  result["swarm_cbf_ms"] = std::move(swarm_cbf_ms);
  result["so3_ms"] = std::move(so3_ms);
  result["openmp_threads"] = thread_count;
  return result;
}

} // namespace

PYBIND11_MODULE(racer_control_batch_cpp, module) {
  module.doc() =
      "Batched, source-faithful RACER safety/CBF/SO3 control orchestration";
  module.def("solve_control_batch", &solve_control_batch, py::arg("commands"),
             py::arg("positions"), py::arg("orientations"),
             py::arg("linear_velocities"), py::arg("angular_velocities"),
             py::arg("motor_rpm"), py::arg("safety_points"),
             py::arg("sweep_constraints"), py::arg("yaw_commands"),
             py::arg("dt"), py::arg("speed_limit"),
             py::arg("obstacle_clearance"), py::arg("sweep_clearance"),
             py::arg("swarm_safe_distance"), py::arg("external_scene"),
             py::arg("flight_minimum") = py::none(),
             py::arg("flight_maximum") = py::none(),
             py::arg("obstacle_minimums") = py::none(),
             py::arg("obstacle_maximums") = py::none());
}
