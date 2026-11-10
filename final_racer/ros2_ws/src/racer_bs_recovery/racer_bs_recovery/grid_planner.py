"""Sparse known-free voxel fusion and conservative A* for BS recovery."""

# This module deliberately has no dependency on racer_original_core.

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Key = Tuple[int, int, int]
FREE = 0
OCCUPIED = 1


@dataclass(frozen=True)
class PlannedRecovery:
    path: Tuple[Tuple[float, float, float], ...]
    frontier: Tuple[float, float, float]
    score: float


class SparseVoxelMap:
    """A small BS-side map that never treats unobserved space as traversable."""

    _NEIGHBORS: Tuple[Key, ...] = (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    )

    def __init__(
        self,
        resolution: float = 0.30,
        origin: Sequence[float] = (-28.1, -27.0, 0.0),
        clearance: float = 0.45,
    ) -> None:
        if resolution <= 0.0 or clearance < 0.0:
            raise ValueError("invalid voxel-map dimensions")
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.clearance = float(clearance)
        self.cells: Dict[Key, int] = {}

    def key(self, point: Sequence[float]) -> Key:
        value = np.floor(
            (np.asarray(point, dtype=float) - self.origin) / self.resolution
        ).astype(np.int64)
        return int(value[0]), int(value[1]), int(value[2])

    def center(self, key: Key) -> Tuple[float, float, float]:
        value = self.origin + (np.asarray(key, dtype=float) + 0.5) * self.resolution
        return float(value[0]), float(value[1]), float(value[2])

    def integrate_ray(
        self, sensor: Sequence[float], endpoint: Sequence[float], hit: bool
    ) -> None:
        start = np.asarray(sensor, dtype=float)
        end = np.asarray(endpoint, dtype=float)
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if not math.isfinite(distance) or distance < 1.0e-6:
            return
        count = max(1, int(math.ceil(distance / (0.5 * self.resolution))))
        for ratio in np.linspace(0.0, 1.0, count, endpoint=False):
            cell = self.key(start + ratio * delta)
            if self.cells.get(cell) != OCCUPIED:
                self.cells[cell] = FREE
        end_key = self.key(end)
        if hit:
            self.cells[end_key] = OCCUPIED
        elif self.cells.get(end_key) != OCCUPIED:
            self.cells[end_key] = FREE

    def traversable(self, key: Key) -> bool:
        if self.cells.get(key) != FREE:
            return False
        radius = int(math.ceil(self.clearance / self.resolution))
        radius_sq = (self.clearance / self.resolution) ** 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > radius_sq:
                        continue
                    if self.cells.get(
                        (key[0] + dx, key[1] + dy, key[2] + dz)
                    ) == OCCUPIED:
                        return False
        return True

    def nearest_traversable(
        self, point: Sequence[float], radius_m: float
    ) -> Optional[Key]:
        target = self.key(point)
        radius = max(1, int(math.ceil(radius_m / self.resolution)))
        best: Optional[Key] = None
        best_cost = math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    candidate = (target[0] + dx, target[1] + dy, target[2] + dz)
                    if not self.traversable(candidate):
                        continue
                    cost = dx * dx + dy * dy + dz * dz
                    if cost < best_cost:
                        best, best_cost = candidate, cost
        return best

    def astar(
        self, start_point: Sequence[float], goal: Key, max_expansions: int = 60000
    ) -> Optional[List[Key]]:
        start = self.nearest_traversable(start_point, 1.0)
        if start is None or not self.traversable(goal):
            return None
        queue: List[Tuple[float, float, Key]] = []
        heapq.heappush(queue, (0.0, 0.0, start))
        parent: Dict[Key, Optional[Key]] = {start: None}
        cost: Dict[Key, float] = {start: 0.0}
        expansions = 0
        while queue and expansions < max_expansions:
            _, current_cost, current = heapq.heappop(queue)
            if current == goal:
                result: List[Key] = []
                cursor: Optional[Key] = current
                while cursor is not None:
                    result.append(cursor)
                    cursor = parent[cursor]
                result.reverse()
                return result
            if current_cost > cost.get(current, math.inf):
                continue
            expansions += 1
            for offset in self._NEIGHBORS:
                nxt = tuple(current[i] + offset[i] for i in range(3))
                if not self.traversable(nxt):
                    continue
                candidate_cost = current_cost + self.resolution
                if candidate_cost >= cost.get(nxt, math.inf):
                    continue
                cost[nxt] = candidate_cost
                parent[nxt] = current
                heuristic = self.resolution * math.sqrt(
                    sum((nxt[i] - goal[i]) ** 2 for i in range(3))
                )
                heapq.heappush(
                    queue, (candidate_cost + heuristic, candidate_cost, nxt)
                )
        return None

    def plan_to_frontiers(
        self,
        start: Sequence[float],
        frontier_clusters: Iterable[np.ndarray],
        staging_radius: float,
        maximum_motion: float,
    ) -> Optional[PlannedRecovery]:
        best: Optional[PlannedRecovery] = None
        for cluster in frontier_clusters:
            points = np.asarray(cluster, dtype=float).reshape((-1, 3))
            if not len(points):
                continue
            frontier = np.median(points, axis=0)
            goal = self.nearest_traversable(frontier, staging_radius)
            if goal is None:
                continue
            keys = self.astar(start, goal)
            if not keys or len(keys) < 2:
                continue
            centers = [self.center(key) for key in keys]
            clipped = [centers[0]]
            distance = 0.0
            for point in centers[1:]:
                step = float(np.linalg.norm(np.asarray(point) - np.asarray(clipped[-1])))
                if distance + step > maximum_motion:
                    break
                clipped.append(point)
                distance += step
            if len(clipped) < 2:
                continue
            # Prefer high-gain clusters, then shorter known-free paths.
            score = 0.04 * min(len(points), 500) - distance
            candidate = PlannedRecovery(
                tuple(clipped), tuple(float(v) for v in frontier), score
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best
