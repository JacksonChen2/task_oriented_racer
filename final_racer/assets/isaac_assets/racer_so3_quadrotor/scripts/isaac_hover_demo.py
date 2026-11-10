#!/usr/bin/env python3
"""Minimal Isaac Sim hover check using the original RACER motor equations.

This is a plant demonstration, not the full RACER geometric controller.  It
commands the analytically derived hover RPM, integrates the original first-order
motor state, applies source-faithful thrust/torque, and adds the original
quadratic translational drag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "usd" / "racer_so3_quadrotor.usd"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--physics-hz", type=float, default=1000.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


ARGS = parse_arguments()
APP = SimulationApp({"headless": ARGS.headless})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Usd, UsdPhysics

from racer_so3_dynamics import (
    advance_motor_rpm,
    hover_rpm,
    load_model,
    quadratic_drag_world,
    rotor_wrench_body,
)


def find_rigid_body(root_path: str) -> str:
    root = get_current_stage().GetPrimAtPath(root_path)
    matches = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one rigid body, found {[str(p.GetPath()) for p in matches]}")
    return str(matches[0].GetPath())


def rotation_body_to_world(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return the active rotation matrix for Isaac's scalar-first quaternion."""
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def main() -> None:
    usd_path = ARGS.usd.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"{usd_path} does not exist; run import_urdf_to_usd.py first"
        )
    if ARGS.duration <= 0.0 or ARGS.physics_hz <= 0.0:
        raise ValueError("duration and physics-hz must be positive")

    dt = 1.0 / ARGS.physics_hz
    world = World(physics_dt=dt, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    root_path = "/World/RacerSO3"
    add_reference_to_stage(str(usd_path), root_path)
    body_path = find_rigid_body(root_path)
    body = world.scene.add(
        SingleRigidPrim(
            prim_path=body_path,
            name="racer_so3_body",
            position=np.asarray((0.0, 0.0, 1.0)),
            orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
        )
    )
    world.reset()
    # Reset/startup may advance a small number of gravity-only substeps before
    # this external plant begins applying motor forces.
    body.set_linear_velocity(np.zeros(3, dtype=np.float32))
    body.set_angular_velocity(np.zeros(3, dtype=np.float32))

    model = load_model()
    target = hover_rpm(model)
    # Start at hover RPM so this test isolates force/inertia import from spool-up drop.
    rpm = (target, target, target, target)
    initial = body.get_world_pose()[0].copy()
    steps = int(round(ARGS.duration * ARGS.physics_hz))
    for index in range(steps):
        rpm = advance_motor_rpm(rpm, (target,) * 4, dt, model)
        body_force, body_torque = rotor_wrench_body(rpm, model)
        _, orientation = body.get_world_pose()
        drag_world = np.asarray(
            quadratic_drag_world(body.get_linear_velocity(), model),
            dtype=np.float64,
        )
        drag_body = rotation_body_to_world(orientation).T @ drag_world
        total_body_force = np.asarray(body_force, dtype=np.float64) + drag_body
        # Submit one force/torque command per step. Separate apply_forces calls
        # overwrite one another in the tensor API instead of accumulating.
        body._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=np.asarray(total_body_force, dtype=np.float32).reshape((1, 3)),
            torques=np.asarray(body_torque, dtype=np.float32).reshape((1, 3)),
            is_global=False,
        )
        # A rendered World.step() lets Kit advance multiple 1 kHz substeps in
        # one 60 Hz application update, while this external force is valid for
        # only one substep. Step PhysX explicitly so every substep receives the
        # matching motor force.
        world.step(render=False)

    final = body.get_world_pose()[0]
    print(
        "RACER SO3 hover check: "
        f"initial={initial.tolist()}, final={final.tolist()}, "
        f"delta={(final - initial).tolist()}, hover_rpm={target:.6f}",
        flush=True,
    )


try:
    main()
finally:
    APP.close()
