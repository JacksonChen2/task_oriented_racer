"""Crazyflie 2.x six-DOF thrust/attitude controller and motor allocation."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


MASS = 0.027
GRAVITY = 9.81
ARM_LENGTH = 0.046
MAX_MOTOR_THRUST = 0.16
YAW_TORQUE_PER_THRUST = 0.006
INERTIA = np.diag((1.43e-5, 1.43e-5, 2.89e-5))


@dataclass
class CrazyflieWrench:
    local_force: np.ndarray
    local_torque: np.ndarray
    motor_thrusts: np.ndarray
    desired_rotation: np.ndarray


def quaternion_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _vee(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        (matrix[2, 1], matrix[0, 2], matrix[1, 0]), dtype=float
    )


def allocate_motors(thrust: float, torque: Sequence[float]) -> np.ndarray:
    """Allocate an X-quad wrench while preserving collective thrust.

    Collective thrust has priority over attitude torque. If a requested torque
    would saturate a motor, all differential components are scaled together
    around ``thrust / 4`` instead of independently clipping motors.
    """

    tx, ty, tz = (float(value) for value in torque)
    lever = ARM_LENGTH / math.sqrt(2.0)
    # [sum thrust, roll, pitch, yaw] = mixing @ motors.
    mixing = np.asarray(
        (
            (1.0, 1.0, 1.0, 1.0),
            (lever, -lever, -lever, lever),
            (-lever, -lever, lever, lever),
            (YAW_TORQUE_PER_THRUST, -YAW_TORQUE_PER_THRUST,
             YAW_TORQUE_PER_THRUST, -YAW_TORQUE_PER_THRUST),
        )
    )
    requested = np.linalg.solve(
        mixing, np.asarray((thrust, tx, ty, tz))
    )
    collective = float(np.clip(thrust / 4.0, 0.0, MAX_MOTOR_THRUST))
    differential = requested - thrust / 4.0
    scale = 1.0
    for value in differential:
        if value > 1.0e-12:
            scale = min(scale, (MAX_MOTOR_THRUST - collective) / value)
        elif value < -1.0e-12:
            scale = min(scale, collective / -value)
    return np.clip(
        collective + max(0.0, scale) * differential,
        0.0,
        MAX_MOTOR_THRUST,
    )


def velocity_wrench(
    desired_velocity: Sequence[float],
    current_velocity: Sequence[float],
    quaternion_wxyz: Sequence[float],
    angular_velocity: Sequence[float],
    desired_yaw: float,
    maximum_acceleration: float = 4.0,
    velocity_gain: float = 2.4,
    attitude_gain: float = 0.0010,
    rate_gain: float = 0.00060,
) -> CrazyflieWrench:
    """Geometric velocity/attitude controller suitable for a PhysX rigid body.

    ``angular_velocity`` follows the ROS/Isaac convention and is expressed in
    the world frame.  Attitude error and commanded rotor torque are body-frame
    quantities, so the measured rate must be rotated into the body frame before
    applying rate feedback.  Omitting this transform is harmless near zero yaw
    but cross-couples roll and pitch after a sustained heading change.
    """

    rotation = quaternion_matrix(quaternion_wxyz)
    error = np.asarray(desired_velocity, dtype=float) - np.asarray(
        current_velocity, dtype=float
    )
    acceleration = velocity_gain * error
    norm = float(np.linalg.norm(acceleration))
    if norm > maximum_acceleration:
        acceleration *= maximum_acceleration / norm
    desired_force_world = MASS * (
        acceleration + np.asarray((0.0, 0.0, GRAVITY))
    )
    desired_z = desired_force_world / max(
        1.0e-9, float(np.linalg.norm(desired_force_world))
    )
    heading = np.asarray((math.cos(desired_yaw), math.sin(desired_yaw), 0.0))
    desired_y = np.cross(desired_z, heading)
    if np.linalg.norm(desired_y) < 1.0e-8:
        desired_y = np.asarray((0.0, 1.0, 0.0))
    desired_y /= np.linalg.norm(desired_y)
    desired_x = np.cross(desired_y, desired_z)
    desired_rotation = np.column_stack((desired_x, desired_y, desired_z))
    error_matrix = 0.5 * (
        desired_rotation.T @ rotation - rotation.T @ desired_rotation
    )
    attitude_error = _vee(error_matrix)
    angular_velocity_body = rotation.T @ np.asarray(
        angular_velocity, dtype=float
    )
    local_torque = (
        -attitude_gain * attitude_error
        - rate_gain * angular_velocity_body
    )
    # Crazyflie's rotor drag provides substantially less yaw authority than
    # roll/pitch leverage. Prevent heading control from starving lift.
    local_torque[2] *= 0.20
    local_torque = np.clip(local_torque, (-0.003, -0.003, -0.0008),
                           (0.003, 0.003, 0.0008))
    thrust = float(np.dot(desired_force_world, rotation[:, 2]))
    thrust = float(np.clip(thrust, 0.0, 4.0 * MAX_MOTOR_THRUST))
    motors = allocate_motors(thrust, local_torque)
    # Reconstruct the feasible wrench after individual motor saturation.
    feasible_thrust = float(np.sum(motors))
    lever = ARM_LENGTH / math.sqrt(2.0)
    feasible_torque = np.asarray(
        (
            lever * (motors[0] - motors[1] - motors[2] + motors[3]),
            lever * (-motors[0] - motors[1] + motors[2] + motors[3]),
            YAW_TORQUE_PER_THRUST
            * (motors[0] - motors[1] + motors[2] - motors[3]),
        )
    )
    return CrazyflieWrench(
        local_force=np.asarray((0.0, 0.0, feasible_thrust)),
        local_torque=feasible_torque,
        motor_thrusts=motors,
        desired_rotation=desired_rotation,
    )
