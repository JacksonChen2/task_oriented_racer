#!/usr/bin/env python3
"""Source-faithful propulsion and rigid-body helpers for the RACER SO3 plant.

This module deliberately has no Isaac Sim or ROS dependency.  It implements
the equations in the original RACER ``Quadrotor.cpp`` and mixer in
``quadrotor_simulator_so3.cpp`` so the same functions can be unit-tested and
used by an Isaac physics callback or a ROS 2 plant adapter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple


Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "racer_so3_model.json"


def load_model(path: Path | str | None = None) -> dict:
    model_path = Path(path) if path is not None else default_model_path()
    with model_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def clamp_motor_commands(command_rpm: Iterable[float], model: dict) -> Vector4:
    propulsion = model["propulsion"]
    lower = float(propulsion["minimum_rpm"])
    upper = float(propulsion["maximum_rpm"])
    values = tuple(min(upper, max(lower, float(value))) for value in command_rpm)
    if len(values) != 4:
        raise ValueError("RACER SO3 requires exactly four motor commands")
    return values  # type: ignore[return-value]


def motor_rpm_derivative(
    current_rpm: Sequence[float], command_rpm: Sequence[float], model: dict
) -> Vector4:
    """Return the original first-order motor derivative in RPM/s."""
    if len(current_rpm) != 4 or len(command_rpm) != 4:
        raise ValueError("RACER SO3 requires exactly four motor states")
    target = clamp_motor_commands(command_rpm, model)
    tau = float(model["propulsion"]["motor_time_constant_s"])
    return tuple(
        (target[index] - float(current_rpm[index])) / tau for index in range(4)
    )  # type: ignore[return-value]


def advance_motor_rpm(
    current_rpm: Sequence[float],
    command_rpm: Sequence[float],
    dt: float,
    model: dict,
) -> Vector4:
    """Advance the motor state using the exact solution of the source ODE."""
    if len(current_rpm) != 4 or len(command_rpm) != 4:
        raise ValueError("RACER SO3 requires exactly four motor states")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    target = clamp_motor_commands(command_rpm, model)
    tau = float(model["propulsion"]["motor_time_constant_s"])
    decay = math.exp(-float(dt) / tau)
    return tuple(
        target[index] + (float(current_rpm[index]) - target[index]) * decay
        for index in range(4)
    )  # type: ignore[return-value]


def rotor_thrusts(rpm: Sequence[float], model: dict) -> Vector4:
    if len(rpm) != 4:
        raise ValueError("RACER SO3 requires exactly four motor states")
    kf = float(model["propulsion"]["thrust_coefficient_N_per_rpm2"])
    return tuple(kf * float(value) ** 2 for value in rpm)  # type: ignore[return-value]


def rotor_wrench_body(rpm: Sequence[float], model: dict) -> Tuple[Vector3, Vector3]:
    """Return body-frame force and moment using the upstream mixer signs."""
    if len(rpm) != 4:
        raise ValueError("RACER SO3 requires exactly four motor states")
    values_sq = tuple(float(value) ** 2 for value in rpm)
    propulsion = model["propulsion"]
    kf = float(propulsion["thrust_coefficient_N_per_rpm2"])
    km = float(propulsion["moment_coefficient_Nm_per_rpm2"])
    arm = float(propulsion["arm_length_m"])

    thrust = kf * sum(values_sq)
    roll = kf * arm * (values_sq[2] - values_sq[3])
    pitch = kf * arm * (values_sq[1] - values_sq[0])
    yaw = km * (values_sq[0] + values_sq[1] - values_sq[2] - values_sq[3])
    return (0.0, 0.0, thrust), (roll, pitch, yaw)


def motor_rpm_for_wrench(
    total_thrust_n: float, body_moment_nm: Sequence[float], model: dict
) -> Vector4:
    """Invert the exact RACER mixer and clamp its resulting RPM commands."""
    if len(body_moment_nm) != 3:
        raise ValueError("body_moment_nm must have roll, pitch, and yaw")
    propulsion = model["propulsion"]
    kf = float(propulsion["thrust_coefficient_N_per_rpm2"])
    km = float(propulsion["moment_coefficient_Nm_per_rpm2"])
    arm = float(propulsion["arm_length_m"])
    roll, pitch, yaw = (float(value) for value in body_moment_nm)

    rpm_sq = (
        total_thrust_n / (4.0 * kf) - pitch / (2.0 * arm * kf) + yaw / (4.0 * km),
        total_thrust_n / (4.0 * kf) + pitch / (2.0 * arm * kf) + yaw / (4.0 * km),
        total_thrust_n / (4.0 * kf) + roll / (2.0 * arm * kf) - yaw / (4.0 * km),
        total_thrust_n / (4.0 * kf) - roll / (2.0 * arm * kf) - yaw / (4.0 * km),
    )
    raw_rpm = tuple(math.sqrt(max(0.0, value)) for value in rpm_sq)
    return clamp_motor_commands(raw_rpm, model)


def quadratic_drag_world(velocity_world_m_s: Sequence[float], model: dict) -> Vector3:
    """Return the source model's quadratic world-frame drag force."""
    if len(velocity_world_m_s) != 3:
        raise ValueError("velocity_world_m_s must contain three components")
    velocity = tuple(float(value) for value in velocity_world_m_s)
    speed = math.sqrt(sum(value * value for value in velocity))
    if speed <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    coefficient = float(model["aerodynamics"]["quadratic_drag_coefficient"])
    scale = -coefficient * speed
    return tuple(scale * value for value in velocity)  # type: ignore[return-value]


def hover_rpm(model: dict) -> float:
    mass = float(model["rigid_body"]["mass_kg"])
    gravity = float(model["rigid_body"]["gravity_m_s2"])
    kf = float(model["propulsion"]["thrust_coefficient_N_per_rpm2"])
    return math.sqrt(mass * gravity / (4.0 * kf))


if __name__ == "__main__":
    parameters = load_model()
    rpm = hover_rpm(parameters)
    force, moment = rotor_wrench_body((rpm, rpm, rpm, rpm), parameters)
    print(f"hover_rpm={rpm:.9f}")
    print(f"body_force_N={force}")
    print(f"body_moment_Nm={moment}")
