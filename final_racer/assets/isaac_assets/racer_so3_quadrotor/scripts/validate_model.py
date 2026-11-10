#!/usr/bin/env python3
"""Validate consistency between the RACER SO3 URDF and model parameters."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from racer_so3_dynamics import hover_rpm, load_model, rotor_wrench_body


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "urdf" / "racer_so3_quadrotor.urdf"
DEFAULT_MODEL = ROOT / "config" / "racer_so3_model.json"


def close(actual: float, expected: float, tolerance: float = 1.0e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r}")


def validate(
    urdf_path: Path,
    model_path: Path,
    expected_visuals: int = 12,
    expected_collisions: int = 7,
) -> dict:
    model = load_model(model_path)
    robot = ET.parse(urdf_path).getroot()
    if robot.tag != "robot" or robot.attrib.get("name") != model["model_name"]:
        raise AssertionError("URDF robot name does not match the model configuration")

    base = robot.find("./link[@name='base_link']")
    if base is None:
        raise AssertionError("URDF is missing base_link")
    mass_element = base.find("./inertial/mass")
    inertia_element = base.find("./inertial/inertia")
    if mass_element is None or inertia_element is None:
        raise AssertionError("base_link is missing an inertial definition")

    rigid = model["rigid_body"]
    close(float(mass_element.attrib["value"]), float(rigid["mass_kg"]))
    matrix = rigid["inertia_kg_m2"]
    for attribute, expected in (
        ("ixx", matrix[0][0]),
        ("ixy", matrix[0][1]),
        ("ixz", matrix[0][2]),
        ("iyy", matrix[1][1]),
        ("iyz", matrix[1][2]),
        ("izz", matrix[2][2]),
    ):
        close(float(inertia_element.attrib[attribute]), float(expected))

    rotor_positions = [entry["position_body_m"] for entry in model["rotors"]]
    expected_positions = (
        [0.26, 0.0, 0.0],
        [-0.26, 0.0, 0.0],
        [0.0, 0.26, 0.0],
        [0.0, -0.26, 0.0],
    )
    if rotor_positions != list(expected_positions):
        raise AssertionError("rotor positions do not implement the upstream plus mixer")

    rotation = model["frames"]["camera_optical"]["rotation_body_from_camera"]
    for row in rotation:
        close(sum(value * value for value in row), 1.0)
    for first in range(3):
        for second in range(first + 1, 3):
            close(sum(rotation[first][i] * rotation[second][i] for i in range(3)), 0.0)

    computed_hover = hover_rpm(model)
    close(computed_hover, float(model["derived"]["hover_rpm_per_motor"]), 1.0e-11)
    force, moment = rotor_wrench_body((computed_hover,) * 4, model)
    close(force[2], float(rigid["mass_kg"]) * float(rigid["gravity_m_s2"]), 1.0e-11)
    for value in moment:
        close(value, 0.0)

    visuals = base.findall("visual")
    collisions = base.findall("collision")
    if len(visuals) != expected_visuals:
        raise AssertionError(
            f"expected {expected_visuals} primitive visuals, found {len(visuals)}"
        )
    if len(collisions) != expected_collisions:
        raise AssertionError(
            f"expected {expected_collisions} primitive collisions, found {len(collisions)}"
        )
    if base.findall(".//mesh"):
        raise AssertionError("portable model must not depend on external mesh files")

    return {
        "status": "ok",
        "urdf": str(urdf_path),
        "model": str(model_path),
        "mass_kg": float(rigid["mass_kg"]),
        "inertia_diagonal_kg_m2": [matrix[0][0], matrix[1][1], matrix[2][2]],
        "hover_rpm": computed_hover,
        "visual_count": len(visuals),
        "collision_count": len(collisions),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expected-visuals", type=int, default=12)
    parser.add_argument("--expected-collisions", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    arguments = parser.parse_args()
    report = validate(
        arguments.urdf.resolve(),
        arguments.model.resolve(),
        arguments.expected_visuals,
        arguments.expected_collisions,
    )
    if arguments.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            "RACER SO3 model validation passed: "
            f"mass={report['mass_kg']} kg, hover={report['hover_rpm']:.3f} RPM, "
            f"{report['visual_count']} visuals, {report['collision_count']} collisions"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ET.ParseError, OSError, ValueError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
