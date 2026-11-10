"""Three-dimensional ESDF and inter-UAV collision avoidance."""

import math
from typing import Any, Iterable, Sequence, Tuple

import numpy as np


# The Isaac bridge only uses the geometry-based filters below. Keeping this
# type alias local makes the vendored bridge independent of the Python RACER
# voxel-map package while preserving the ESDF helper's public signature.
VoxelMap = Any
Vector3 = Tuple[float, float, float]


def limit_norm(vector: Sequence[float], maximum: float) -> Vector3:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > maximum > 0.0:
        value *= maximum / norm
    return tuple(float(item) for item in value)


def cbf_swarm_filter(
    preferred: Sequence[float],
    position: Sequence[float],
    peers: Iterable[
        Tuple[int, Sequence[float], Sequence[float]]
    ],
    safe_distance: float,
    speed_limit: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    alpha: float = 0.8,
    guaranteed_deceleration: float = 0.6,
    response_time: float = 0.20,
    iterations: int = 6,
) -> Vector3:
    """Project velocity onto stopping-distance pairwise 3-D barriers."""

    result = np.asarray(limit_norm(preferred, speed_limit), dtype=float)
    own = np.asarray(position, dtype=float)
    own_velocity = np.asarray(current_velocity, dtype=float)
    constraints = list(peers)
    for _ in range(iterations):
        for _, peer_position, peer_velocity in constraints:
            relative = own - np.asarray(peer_position, dtype=float)
            distance = float(np.linalg.norm(relative))
            if distance < 1.0e-6:
                relative = np.asarray((1.0, 0.0, 0.0))
                distance = 1.0e-6
            normal = relative / distance
            peer = np.asarray(peer_velocity, dtype=float)
            approach_speed = max(
                0.0, -float(np.dot(normal, own_velocity - peer))
            )
            stopping_allowance = (
                approach_speed * approach_speed
                / (2.0 * max(0.1, guaranteed_deceleration))
                + response_time * approach_speed
            )
            dynamic_safe_distance = safe_distance + stopping_allowance
            # h = ||p_i-p_j||^2-d_safe^2. Enforce h_dot >= -alpha h.
            h = (
                distance * distance
                - dynamic_safe_distance * dynamic_safe_distance
            )
            lower_bound = float(np.dot(normal, peer)) - alpha * h / (
                2.0 * distance
            )
            violation = lower_bound - float(np.dot(normal, result))
            if violation > 0.0:
                result += violation * normal
        result = np.asarray(limit_norm(result, speed_limit))
    return tuple(float(value) for value in result)


def esdf_obstacle_filter(
    preferred: Sequence[float],
    position: Sequence[float],
    voxel_map: VoxelMap,
    clearance: float,
    speed_limit: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    alpha: float = 0.8,
    guaranteed_deceleration: float = 1.2,
    response_time: float = 0.12,
) -> Vector3:
    """Enforce an execution clearance using an ESDF velocity barrier.

    The relatively low barrier gain deliberately begins braking before the
    geometric limit.  A rigid-body quadrotor cannot reverse velocity
    instantaneously, so a high-gain first-order barrier can be mathematically
    valid for a point mass while still allowing the real vehicle to overshoot
    into a wall.
    """

    result = np.asarray(limit_norm(preferred, speed_limit), dtype=float)
    distance = voxel_map.distance_at(position, unknown_is_occupied=False)
    if not math.isfinite(distance):
        return (0.0, 0.0, 0.0)
    gradient = voxel_map.esdf_gradient(position, unknown_is_occupied=False)
    norm = float(np.linalg.norm(gradient))
    if norm < 1.0e-8:
        return tuple(float(value) for value in result)
    normal = gradient / norm
    normal_speed = float(
        np.dot(normal, np.asarray(current_velocity, dtype=float))
    )
    approach_speed = max(0.0, -normal_speed)
    stopping_allowance = (
        approach_speed * approach_speed
        / (2.0 * max(0.1, guaranteed_deceleration))
        + response_time * approach_speed
    )
    dynamic_clearance = clearance + stopping_allowance
    lower_bound = -alpha * (distance - dynamic_clearance)
    violation = lower_bound - float(np.dot(normal, result))
    if violation > 0.0:
        result += violation * normal
    return limit_norm(result, speed_limit)


def aabb_obstacle_filter(
    preferred: Sequence[float],
    position: Sequence[float],
    obstacles: Iterable[object],
    clearance: float,
    speed_limit: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    alpha: float = 1.2,
    guaranteed_deceleration: float = 0.7,
    response_time: float = 0.08,
    activation_distance: float = 1.0,
    iterations: int = 4,
) -> Vector3:
    """High-rate rigid-body velocity barrier for axis-aligned collision boxes.

    RACER's local trajectory planner runs at a much lower rate than a flight
    controller.  This projection is intended for that flight-controller layer:
    it adds stopping distance to the configured body clearance and enforces
    all nearby collision-shape half-space constraints every physics step.
    ``obstacles`` need only expose three-element ``minimum`` and ``maximum``
    properties, so this routine is independent of the acceptance scenario.
    """

    own = np.asarray(position, dtype=float)
    velocity = np.asarray(current_velocity, dtype=float)
    constraints = []
    for obstacle in obstacles:
        minimum = np.asarray(obstacle.minimum, dtype=float)
        maximum = np.asarray(obstacle.maximum, dtype=float)
        closest = np.clip(own, minimum, maximum)
        delta = own - closest
        distance = float(np.linalg.norm(delta))
        if distance > 1.0e-9:
            normal = delta / distance
            signed_distance = distance
        else:
            # Inside a box, choose the closest face and point out of it.
            face_distances = np.concatenate((own - minimum, maximum - own))
            face = int(np.argmin(face_distances))
            normal = np.zeros(3)
            normal[face % 3] = -1.0 if face < 3 else 1.0
            signed_distance = -float(face_distances[face])
        if signed_distance > activation_distance:
            continue
        approach_speed = max(0.0, -float(np.dot(normal, velocity)))
        stopping_allowance = (
            approach_speed * approach_speed
            / (2.0 * max(0.1, guaranteed_deceleration))
            + response_time * approach_speed
        )
        dynamic_clearance = clearance + stopping_allowance
        lower_bound = -alpha * (signed_distance - dynamic_clearance)
        constraints.append((signed_distance, normal, lower_bound))

    result = np.asarray(limit_norm(preferred, speed_limit), dtype=float)
    # If a narrow passage makes conservative clearances mutually infeasible,
    # the closest surface must retain final authority.  Applying constraints
    # far-to-near makes the last projection in every pass the most urgent one;
    # near-to-far allowed a farther opposing wall to project the command back
    # toward an already contacting rotor/arm envelope.
    constraints.sort(key=lambda item: item[0], reverse=True)
    for _ in range(iterations):
        for _, normal, lower_bound in constraints:
            violation = lower_bound - float(np.dot(normal, result))
            if violation > 0.0:
                result += violation * normal
        result = np.asarray(limit_norm(result, speed_limit), dtype=float)
    return tuple(float(value) for value in result)


def pointcloud_obstacle_filter(
    preferred: Sequence[float],
    position: Sequence[float],
    points_world: Sequence[Sequence[float]],
    clearance: float,
    speed_limit: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    alpha: float = 0.8,
    guaranteed_deceleration: float = 0.6,
    response_time: float = 0.20,
    activation_distance: float = 2.2,
    maximum_constraints: int = 64,
    iterations: int = 4,
) -> Vector3:
    """Project velocity away from recently observed 3-D obstacle returns.

    This is the external-USD counterpart to :func:`aabb_obstacle_filter`.
    It uses only hit points produced by the vehicle's simulated lidar, so it
    does not require privileged knowledge of an arbitrary scene's geometry.
    Points are kept in the shared world frame by the Isaac bridge.
    """

    constraints = pointcloud_obstacle_constraints(
        position,
        points_world,
        clearance,
        current_velocity=current_velocity,
        alpha=alpha,
        guaranteed_deceleration=guaranteed_deceleration,
        response_time=response_time,
        activation_distance=activation_distance,
        maximum_constraints=maximum_constraints,
    )
    return project_velocity_constraints(
        preferred,
        constraints,
        speed_limit,
        iterations=iterations,
    )


def pointcloud_obstacle_constraints(
    position: Sequence[float],
    points_world: Sequence[Sequence[float]],
    clearance: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    alpha: float = 0.8,
    guaranteed_deceleration: float = 0.6,
    response_time: float = 0.20,
    activation_distance: float = 2.2,
    maximum_constraints: int = 64,
):
    """Build the point-cloud half spaces independently of a command.

    A control tick projects first the planner command and then the swarm-safe
    command against the same obstacle geometry.  Keeping construction
    separate avoids sorting the same point cloud twice while preserving the
    original projection order and numerical result.
    """

    points = np.asarray(points_world, dtype=float).reshape((-1, 3))
    if not len(points):
        return []
    own = np.asarray(position, dtype=float)
    velocity = np.asarray(current_velocity, dtype=float)
    delta = own - points
    distances = np.linalg.norm(delta, axis=1)
    valid = (
        np.all(np.isfinite(points), axis=1)
        & (distances > 1.0e-4)
        & (distances <= activation_distance)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    indices = indices[np.argsort(distances[indices])[:maximum_constraints]]
    constraints = []
    for index in indices:
        distance = float(distances[index])
        normal = delta[index] / distance
        approach_speed = max(0.0, -float(np.dot(normal, velocity)))
        stopping_allowance = (
            approach_speed * approach_speed
            / (2.0 * max(0.1, guaranteed_deceleration))
            + response_time * approach_speed
        )
        dynamic_clearance = clearance + stopping_allowance
        lower_bound = -alpha * (distance - dynamic_clearance)
        constraints.append((distance, normal, lower_bound))

    # Preserve the nearest observed surface as the final constraint when a
    # conservative clearance cannot fit between two opposing surfaces.
    constraints.sort(key=lambda item: item[0], reverse=True)
    return constraints


def project_velocity_constraints(
    preferred: Sequence[float],
    constraints,
    speed_limit: float,
    iterations: int = 4,
) -> Vector3:
    """Project a velocity through precomputed obstacle half spaces."""

    result = np.asarray(limit_norm(preferred, speed_limit), dtype=float)
    for _ in range(iterations):
        for _, normal, lower_bound in constraints:
            violation = lower_bound - float(np.dot(normal, result))
            if violation > 0.0:
                result += violation * normal
        result = np.asarray(limit_norm(result, speed_limit), dtype=float)
    return tuple(float(value) for value in result)


def sweep_obstacle_filter(
    preferred: Sequence[float],
    sweep_constraints: Iterable[Tuple[Sequence[float], float]],
    speed_limit: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    clearance: float = 0.06,
    alpha: float = 0.8,
    guaranteed_deceleration: float = 0.6,
    response_time: float = 0.20,
    iterations: int = 4,
) -> Vector3:
    """Enforce free travel reported by a rigid-envelope PhysX sweep.

    ``distance`` is the distance that the already-inflated vehicle sphere may
    travel along ``direction`` before impact.  Unlike treating a sweep hit as
    an ordinary point-cloud return, this preserves the actual swept-volume
    semantics and remains valid beside thin rack meshes.  The fixed clearance
    is only a contact-offset/controller-lag reserve; stopping distance is
    computed separately from measured velocity.
    """

    result = np.asarray(limit_norm(preferred, speed_limit), dtype=float)
    velocity = np.asarray(current_velocity, dtype=float)
    constraints = []
    for direction, distance in sweep_constraints:
        direction_value = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction_value))
        travel = float(distance)
        if norm < 1.0e-8 or not math.isfinite(travel) or travel < 0.0:
            continue
        direction_value /= norm
        approach_speed = max(0.0, float(np.dot(direction_value, velocity)))
        stopping_allowance = (
            approach_speed * approach_speed
            / (2.0 * max(0.1, guaranteed_deceleration))
            + response_time * approach_speed
        )
        dynamic_clearance = clearance + stopping_allowance
        upper_bound = alpha * (travel - dynamic_clearance)
        constraints.append((travel, direction_value, upper_bound))

    # As with the point constraints, preserve the nearest swept obstacle as
    # final authority if opposing conservative constraints are infeasible.
    constraints.sort(key=lambda item: item[0], reverse=True)
    for _ in range(iterations):
        for _, direction, upper_bound in constraints:
            violation = float(np.dot(direction, result)) - upper_bound
            if violation > 0.0:
                result -= violation * direction
        result = np.asarray(limit_norm(result, speed_limit), dtype=float)
    return tuple(float(value) for value in result)


def flight_volume_filter(
    preferred: Sequence[float],
    position: Sequence[float],
    minimum: Sequence[float],
    maximum: Sequence[float],
    clearance: float,
    speed_limit: float,
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    alpha: float = 1.2,
    guaranteed_deceleration: float = 0.7,
    response_time: float = 0.10,
    iterations: int = 3,
) -> Vector3:
    """Keep a vehicle inside a configured 3-D flight volume.

    Lidar alone can have a blind cone directly above/below the sensor. A
    configured room geofence therefore remains authoritative for floor,
    ceiling, and outer walls while lidar handles arbitrary interior meshes.
    """

    result = np.asarray(limit_norm(preferred, speed_limit), dtype=float)
    own = np.asarray(position, dtype=float)
    lower = np.asarray(minimum, dtype=float)
    upper = np.asarray(maximum, dtype=float)
    velocity = np.asarray(current_velocity, dtype=float)
    constraints = []
    for axis in range(3):
        for distance, direction in (
            (float(own[axis] - lower[axis]), 1.0),
            (float(upper[axis] - own[axis]), -1.0),
        ):
            normal = np.zeros(3, dtype=float)
            normal[axis] = direction
            approach_speed = max(0.0, -float(np.dot(normal, velocity)))
            stopping_allowance = (
                approach_speed * approach_speed
                / (2.0 * max(0.1, guaranteed_deceleration))
                + response_time * approach_speed
            )
            dynamic_clearance = clearance + stopping_allowance
            lower_bound = -alpha * (distance - dynamic_clearance)
            constraints.append((distance, normal, lower_bound))
    # As above, an infeasible volume corner is resolved in favour of the
    # closest boundary instead of letting a farther face undo its projection.
    constraints.sort(key=lambda item: item[0], reverse=True)
    for _ in range(iterations):
        for _, normal, lower_bound in constraints:
            violation = lower_bound - float(np.dot(normal, result))
            if violation > 0.0:
                result += violation * normal
        result = np.asarray(limit_norm(result, speed_limit), dtype=float)
    return tuple(float(value) for value in result)


def emergency_separation(
    velocity: Sequence[float],
    position: Sequence[float],
    peer_positions: Iterable[Sequence[float]],
    emergency_distance: float,
    speed_limit: float,
) -> Vector3:
    result = np.asarray(velocity, dtype=float)
    own = np.asarray(position, dtype=float)
    for peer in peer_positions:
        delta = own - np.asarray(peer, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance < emergency_distance:
            if distance < 1.0e-6:
                delta, distance = np.asarray((1.0, 0.0, 0.0)), 1.0
            strength = speed_limit * (emergency_distance - distance) / max(
                emergency_distance, 1.0e-6
            )
            result += strength * delta / distance
    return limit_norm(result, speed_limit)


def predicted_path_conflict(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    safe_distance: float,
    time_tolerance: float = 0.35,
) -> bool:
    """Check time-stamped ``(t,x,y,z)`` trajectories for 3-D conflict."""

    for own in first:
        if len(own) < 4:
            continue
        for peer in second:
            if len(peer) < 4 or abs(float(own[0]) - float(peer[0])) > time_tolerance:
                continue
            if float(np.linalg.norm(np.asarray(own[1:4]) - np.asarray(peer[1:4]))) < safe_distance:
                return True
    return False
