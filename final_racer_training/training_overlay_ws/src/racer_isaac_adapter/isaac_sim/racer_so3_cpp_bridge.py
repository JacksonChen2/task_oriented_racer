"""Source-faithful RACER SO3 plant and low-level velocity adapter.

The C++ exploration agent publishes a world-frame velocity target.  This
module converts that target into the desired force/orientation used by the
upstream SO3 controller, evaluates the upstream attitude controller and mixer,
integrates the first-order motor states, and returns the actual rotor wrench.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


MASS = 0.98
GRAVITY = 9.81
INERTIA = np.diag((2.64e-3, 2.64e-3, 4.96e-3))
ARM_LENGTH = 0.26
PROPELLER_RADIUS = 0.062
THRUST_COEFFICIENT = 8.98132e-9
MOMENT_COEFFICIENT = 1.169367864e-10
MOTOR_TIME_CONSTANT = 1.0 / 30.0
MINIMUM_RPM = 1200.0
MAXIMUM_RPM = 35000.0
QUADRATIC_DRAG_COEFFICIENT = 0.1 * math.pi * ARM_LENGTH**2

# Effective values after gains_hummingbird.yaml and simulator.xml overrides.
ATTITUDE_GAIN = np.asarray((1.0, 1.0, 1.0))
ANGULAR_RATE_GAIN = np.asarray((0.07, 0.07, 0.1))
VELOCITY_GAIN = np.asarray((3.4, 3.4, 4.0))


@dataclass
class RacerSO3Wrench:
    local_force: np.ndarray
    local_torque: np.ndarray
    command_rpm: np.ndarray
    motor_rpm: np.ndarray
    motor_thrusts: np.ndarray
    desired_rotation: np.ndarray
    desired_force_world: np.ndarray


def quaternion_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=float,
    )


def _vee(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        (matrix[2, 1], matrix[0, 2], matrix[1, 0]), dtype=float
    )


def hover_rpm() -> float:
    return math.sqrt(MASS * GRAVITY / (4.0 * THRUST_COEFFICIENT))


def _desired_rotation(force_world: np.ndarray, desired_yaw: float) -> np.ndarray:
    norm = float(np.linalg.norm(force_world))
    desired_z = (
        force_world / norm
        if norm > 1.0e-9
        else np.asarray((0.0, 0.0, 1.0))
    )
    desired_heading = np.asarray(
        (math.cos(desired_yaw), math.sin(desired_yaw), 0.0)
    )
    desired_y = np.cross(desired_z, desired_heading)
    if float(np.linalg.norm(desired_y)) < 1.0e-9:
        desired_y = np.asarray((0.0, 1.0, 0.0))
    desired_y /= np.linalg.norm(desired_y)
    desired_x = np.cross(desired_y, desired_z)
    return np.column_stack((desired_x, desired_y, desired_z))


def motor_commands_for_wrench(
    total_thrust: float, body_moment: Sequence[float]
) -> np.ndarray:
    """Invert the exact upstream plus-quad mixer and clamp input RPM."""

    roll, pitch, yaw = (float(value) for value in body_moment)
    rpm_squared = np.asarray(
        (
            total_thrust / (4.0 * THRUST_COEFFICIENT)
            - pitch / (2.0 * ARM_LENGTH * THRUST_COEFFICIENT)
            + yaw / (4.0 * MOMENT_COEFFICIENT),
            total_thrust / (4.0 * THRUST_COEFFICIENT)
            + pitch / (2.0 * ARM_LENGTH * THRUST_COEFFICIENT)
            + yaw / (4.0 * MOMENT_COEFFICIENT),
            total_thrust / (4.0 * THRUST_COEFFICIENT)
            + roll / (2.0 * ARM_LENGTH * THRUST_COEFFICIENT)
            - yaw / (4.0 * MOMENT_COEFFICIENT),
            total_thrust / (4.0 * THRUST_COEFFICIENT)
            - roll / (2.0 * ARM_LENGTH * THRUST_COEFFICIENT)
            - yaw / (4.0 * MOMENT_COEFFICIENT),
        )
    )
    raw = np.sqrt(np.maximum(0.0, rpm_squared))
    return np.clip(raw, MINIMUM_RPM, MAXIMUM_RPM)


def advance_motor_rpm(
    current_rpm: Sequence[float], command_rpm: Sequence[float], dt: float
) -> np.ndarray:
    target = np.clip(
        np.asarray(command_rpm, dtype=float), MINIMUM_RPM, MAXIMUM_RPM
    )
    current = np.asarray(current_rpm, dtype=float)
    decay = math.exp(-float(dt) / MOTOR_TIME_CONSTANT)
    return target + (current - target) * decay


def rotor_wrench(motor_rpm: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    squared = np.square(np.asarray(motor_rpm, dtype=float))
    force = np.asarray(
        (0.0, 0.0, THRUST_COEFFICIENT * float(np.sum(squared)))
    )
    torque = np.asarray(
        (
            THRUST_COEFFICIENT
            * ARM_LENGTH
            * (squared[2] - squared[3]),
            THRUST_COEFFICIENT
            * ARM_LENGTH
            * (squared[1] - squared[0]),
            MOMENT_COEFFICIENT
            * (squared[0] + squared[1] - squared[2] - squared[3]),
        )
    )
    return force, torque


def velocity_motor_wrench(
    desired_velocity: Sequence[float],
    current_velocity: Sequence[float],
    quaternion_wxyz: Sequence[float],
    angular_velocity_world: Sequence[float],
    desired_yaw: float,
    current_motor_rpm: Sequence[float],
    dt: float,
    maximum_acceleration: float = 1.4,
) -> RacerSO3Wrench:
    """Evaluate the RACER controller/motor plant for one PhysX substep."""

    rotation = quaternion_matrix(quaternion_wxyz)
    velocity_error = np.asarray(desired_velocity, dtype=float) - np.asarray(
        current_velocity, dtype=float
    )
    acceleration = VELOCITY_GAIN * velocity_error / MASS
    acceleration_norm = float(np.linalg.norm(acceleration))
    if acceleration_norm > maximum_acceleration:
        acceleration *= maximum_acceleration / acceleration_norm
    desired_force_world = MASS * (
        acceleration + np.asarray((0.0, 0.0, GRAVITY))
    )
    desired_rotation = _desired_rotation(desired_force_world, desired_yaw)

    error_matrix = 0.5 * (
        desired_rotation.T @ rotation - rotation.T @ desired_rotation
    )
    attitude_error = _vee(error_matrix)
    angular_velocity_body = rotation.T @ np.asarray(
        angular_velocity_world, dtype=float
    )
    gyroscopic = np.cross(
        angular_velocity_body, INERTIA @ angular_velocity_body
    )
    requested_moment = (
        -ATTITUDE_GAIN * attitude_error
        - ANGULAR_RATE_GAIN * angular_velocity_body
        + gyroscopic
    )

    attitude_error_function = 0.5 * (
        3.0 - float(np.trace(desired_rotation.T @ rotation))
    )
    total_thrust = (
        float(np.dot(desired_force_world, rotation[:, 2]))
        if attitude_error_function < 1.0
        else 0.0
    )
    command_rpm = motor_commands_for_wrench(
        max(0.0, total_thrust), requested_moment
    )
    motor_rpm = advance_motor_rpm(current_motor_rpm, command_rpm, dt)
    local_force, local_torque = rotor_wrench(motor_rpm)

    velocity_world = np.asarray(current_velocity, dtype=float)
    speed = float(np.linalg.norm(velocity_world))
    drag_world = -QUADRATIC_DRAG_COEFFICIENT * speed * velocity_world
    local_force = local_force + rotation.T @ drag_world

    return RacerSO3Wrench(
        local_force=local_force,
        local_torque=local_torque,
        command_rpm=command_rpm,
        motor_rpm=motor_rpm,
        motor_thrusts=THRUST_COEFFICIENT * np.square(motor_rpm),
        desired_rotation=desired_rotation,
        desired_force_world=desired_force_world,
    )
