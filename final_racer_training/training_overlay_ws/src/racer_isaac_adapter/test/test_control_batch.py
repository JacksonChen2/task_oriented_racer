#!/usr/bin/env python3
"""Numerical equivalence and wall-time profile for the batch controller."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np


ISAAC_SIM_DIR = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ISAAC_SIM_DIR))

from racer_control_batch_cpp import solve_control_batch  # noqa: E402
from racer_so3_cpp_bridge import velocity_motor_wrench  # noqa: E402
from safety_cpp_bridge import (  # noqa: E402
    aabb_obstacle_filter,
    cbf_swarm_filter,
    flight_volume_filter,
    pointcloud_obstacle_constraints,
    project_velocity_constraints,
    sweep_obstacle_filter,
)


N = 10
DT = 0.001
SPEED_LIMIT = 2.0
OBSTACLE_CLEARANCE = 0.332
SWEEP_CLEARANCE = 0.06
SWARM_SAFE_DISTANCE = 1.296
FLIGHT_MINIMUM = (-8.0, -8.0, 0.35)
FLIGHT_MAXIMUM = (8.0, 8.0, 5.0)


def external_projection(
    preferred, position, velocity, point_constraints, sweeps
):
    result = project_velocity_constraints(
        preferred, point_constraints, SPEED_LIMIT
    )
    result = flight_volume_filter(
        result,
        position,
        FLIGHT_MINIMUM,
        FLIGHT_MAXIMUM,
        clearance=OBSTACLE_CLEARANCE,
        speed_limit=SPEED_LIMIT,
        current_velocity=velocity,
    )
    return np.asarray(
        sweep_obstacle_filter(
            result,
            sweeps,
            speed_limit=SPEED_LIMIT,
            current_velocity=velocity,
            clearance=SWEEP_CLEARANCE,
        ),
        dtype=float,
    )


def reference_job(job):
    (
        index,
        commands,
        positions,
        orientations,
        velocities,
        angular_velocities,
        rpms,
        points,
        sweeps,
        yaws,
        external_scene,
        obstacles,
    ) = job
    requested = commands[index]
    position = positions[index]
    velocity = velocities[index]
    peers = [
        (peer, positions[peer], velocities[peer])
        for peer in range(len(positions))
        if peer != index
    ]
    if external_scene:
        point_constraints = pointcloud_obstacle_constraints(
            position,
            points[index],
            clearance=OBSTACLE_CLEARANCE,
            current_velocity=velocity,
        )
        applied = external_projection(
            requested, position, velocity, point_constraints, sweeps[index]
        )
    else:
        point_constraints = None
        applied = np.asarray(
            aabb_obstacle_filter(
                requested,
                position,
                obstacles,
                clearance=OBSTACLE_CLEARANCE,
                speed_limit=SPEED_LIMIT,
                current_velocity=velocity,
            ),
            dtype=float,
        )
    applied = np.asarray(
        cbf_swarm_filter(
            applied,
            position,
            peers,
            safe_distance=SWARM_SAFE_DISTANCE,
            speed_limit=SPEED_LIMIT,
            current_velocity=velocity,
        ),
        dtype=float,
    )
    if external_scene:
        applied = external_projection(
            applied, position, velocity, point_constraints, sweeps[index]
        )
    else:
        applied = np.asarray(
            aabb_obstacle_filter(
                applied,
                position,
                obstacles,
                clearance=OBSTACLE_CLEARANCE,
                speed_limit=SPEED_LIMIT,
                current_velocity=velocity,
            ),
            dtype=float,
        )
    wrench = velocity_motor_wrench(
        applied,
        velocity,
        orientations[index],
        angular_velocities[index],
        yaws[index],
        rpms[index],
        DT,
    )
    return (
        applied,
        wrench.local_force,
        wrench.local_torque,
        wrench.command_rpm,
        wrench.motor_rpm,
        wrench.motor_thrusts,
        np.linalg.norm(applied - requested) > 1.0e-3,
    )


def make_case():
    rng = np.random.default_rng(120934)
    positions = np.column_stack(
        (
            np.linspace(-3.0, 3.0, N),
            rng.uniform(-2.5, 2.5, N),
            rng.uniform(0.7, 3.8, N),
        )
    )
    commands = rng.uniform(-1.6, 1.6, (N, 3))
    velocities = rng.uniform(-1.0, 1.0, (N, 3))
    angular_velocities = rng.uniform(-0.35, 0.35, (N, 3))
    orientations = rng.normal(size=(N, 4))
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    rpms = rng.uniform(12000.0, 22000.0, (N, 4))
    yaws = rng.uniform(-np.pi, np.pi, N)
    points = []
    sweeps = []
    for index in range(N):
        directions = rng.normal(size=(180 + index, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        distances = rng.uniform(0.25, 2.8, len(directions))
        hits = positions[index] + directions * distances[:, None]
        if index % 2:
            hits = hits.astype(np.float32)
        points.append(hits)
        sweep_directions = rng.normal(size=(7, 3))
        sweep_distances = rng.uniform(0.05, 2.4, 7)
        sweeps.append(
            [
                (sweep_directions[row], sweep_distances[row])
                for row in range(7)
            ]
        )
    obstacles = [
        SimpleNamespace(
            minimum=np.asarray((-1.4, -0.7, 0.0)),
            maximum=np.asarray((-0.8, 1.5, 3.6)),
        ),
        SimpleNamespace(
            minimum=np.asarray((1.1, -2.0, 0.0)),
            maximum=np.asarray((1.7, -0.4, 4.0)),
        ),
        SimpleNamespace(
            minimum=np.asarray((-4.0, 2.7, 0.0)),
            maximum=np.asarray((4.0, 3.0, 4.5)),
        ),
    ]
    return (
        commands,
        positions,
        orientations,
        velocities,
        angular_velocities,
        rpms,
        points,
        sweeps,
        yaws,
        obstacles,
    )


def call_batch(case, external_scene):
    (
        commands,
        positions,
        orientations,
        velocities,
        angular_velocities,
        rpms,
        points,
        sweeps,
        yaws,
        obstacles,
    ) = case
    obstacle_minimums = np.asarray(
        [obstacle.minimum for obstacle in obstacles], dtype=float
    )
    obstacle_maximums = np.asarray(
        [obstacle.maximum for obstacle in obstacles], dtype=float
    )
    return solve_control_batch(
        commands,
        positions,
        orientations,
        velocities,
        angular_velocities,
        rpms,
        points,
        sweeps,
        yaws,
        DT,
        SPEED_LIMIT,
        OBSTACLE_CLEARANCE,
        SWEEP_CLEARANCE,
        SWARM_SAFE_DISTANCE,
        external_scene,
        FLIGHT_MINIMUM if external_scene else None,
        FLIGHT_MAXIMUM if external_scene else None,
        obstacle_minimums if not external_scene else None,
        obstacle_maximums if not external_scene else None,
    )


def reference_results(case, external_scene, executor=None):
    *_, obstacles = case
    jobs = [
        (index, *case[:-1], external_scene, obstacles) for index in range(N)
    ]
    if executor is None:
        output = list(map(reference_job, jobs))
    else:
        output = list(executor.map(reference_job, jobs))
    return {
        "applied_commands": np.asarray([item[0] for item in output]),
        "forces": np.asarray([item[1] for item in output]),
        "torques": np.asarray([item[2] for item in output]),
        "command_rpm": np.asarray([item[3] for item in output]),
        "motor_rpm": np.asarray([item[4] for item in output]),
        "motor_thrust": np.asarray([item[5] for item in output]),
        "intervened": np.asarray([item[6] for item in output]),
    }


def check_mode(case, external_scene):
    reference = reference_results(case, external_scene)
    batch = call_batch(case, external_scene)
    maximum_absolute_error = {}
    for key in (
        "applied_commands",
        "forces",
        "torques",
        "command_rpm",
        "motor_rpm",
        "motor_thrust",
    ):
        difference = np.abs(reference[key] - batch[key])
        maximum_absolute_error[key] = float(np.max(difference))
        np.testing.assert_allclose(
            batch[key], reference[key], rtol=2.0e-11, atol=2.0e-11
        )
    np.testing.assert_array_equal(batch["intervened"], reference["intervened"])
    return maximum_absolute_error


def profile(case):
    iterations = 40
    with ThreadPoolExecutor(max_workers=8) as executor:
        reference_results(case, True, executor)
        started = time.perf_counter()
        for _ in range(iterations):
            reference_results(case, True, executor)
        reference_ms = 1000.0 * (time.perf_counter() - started) / iterations
    call_batch(case, True)
    started = time.perf_counter()
    for _ in range(iterations):
        call_batch(case, True)
    batch_ms = 1000.0 * (time.perf_counter() - started) / iterations
    return {
        "uav_count": N,
        "iterations": iterations,
        "python_threadpool_reference_mean_step_ms": reference_ms,
        "cpp_batch_mean_step_ms": batch_ms,
        "speedup": reference_ms / batch_ms,
    }


def main():
    case = make_case()
    report = {
        "external_scene_max_abs_error": check_mode(case, True),
        "aabb_scene_max_abs_error": check_mode(case, False),
        "profile": profile(case),
    }
    print("RACER_CONTROL_BATCH_CORRECTNESS " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
