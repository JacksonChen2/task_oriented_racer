from __future__ import annotations

import math
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from racer_so3_dynamics import (  # noqa: E402
    advance_motor_rpm,
    hover_rpm,
    load_model,
    motor_rpm_for_wrench,
    quadratic_drag_world,
    rotor_wrench_body,
)


MODEL = load_model()


def test_hover_exactly_balances_weight():
    rpm = hover_rpm(MODEL)
    force, torque = rotor_wrench_body((rpm,) * 4, MODEL)
    expected = MODEL["rigid_body"]["mass_kg"] * MODEL["rigid_body"]["gravity_m_s2"]
    assert math.isclose(force[2], expected, rel_tol=1.0e-12)
    assert torque == (0.0, 0.0, 0.0)


def test_mixer_round_trip_away_from_saturation():
    requested_force = 12.0
    requested_torque = (0.08, -0.06, 0.01)
    rpm = motor_rpm_for_wrench(requested_force, requested_torque, MODEL)
    force, torque = rotor_wrench_body(rpm, MODEL)
    assert math.isclose(force[2], requested_force, rel_tol=1.0e-12)
    for actual, expected in zip(torque, requested_torque):
        assert math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_motor_time_constant():
    target = 20000.0
    tau = MODEL["propulsion"]["motor_time_constant_s"]
    current = (1200.0,) * 4
    after_tau = advance_motor_rpm(current, (target,) * 4, tau, MODEL)
    expected = target + (1200.0 - target) / math.e
    for value in after_tau:
        assert math.isclose(value, expected, rel_tol=1.0e-12)


def test_quadratic_drag_opposes_world_velocity():
    drag = quadratic_drag_world((3.0, 4.0, 0.0), MODEL)
    coefficient = MODEL["aerodynamics"]["quadratic_drag_coefficient"]
    assert drag == (-15.0 * coefficient, -20.0 * coefficient, -0.0)
