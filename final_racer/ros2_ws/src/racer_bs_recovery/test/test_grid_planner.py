"""Unit tests for the standalone known-free recovery planner."""

import numpy as np

from racer_bs_recovery.grid_planner import OCCUPIED, SparseVoxelMap


def make_corridor():
    grid = SparseVoxelMap(resolution=1.0, origin=(0.0, 0.0, 0.0), clearance=0.0)
    for x in range(8):
        grid.cells[(x, 0, 0)] = 0
    return grid


def test_astar_uses_only_observed_free_cells():
    grid = make_corridor()
    path = grid.astar((0.5, 0.5, 0.5), (7, 0, 0))
    assert path is not None
    assert all(grid.cells[cell] == 0 for cell in path)
    grid.cells.pop((4, 0, 0))
    assert grid.astar((0.5, 0.5, 0.5), (7, 0, 0)) is None


def test_ray_marks_free_then_hit_occupied():
    grid = SparseVoxelMap(resolution=0.5, origin=(0.0, 0.0, 0.0), clearance=0.0)
    grid.integrate_ray((0.1, 0.1, 0.1), (2.1, 0.1, 0.1), True)
    assert grid.cells[grid.key((0.1, 0.1, 0.1))] == 0
    assert grid.cells[grid.key((2.1, 0.1, 0.1))] == OCCUPIED


def test_frontier_plan_is_bounded():
    grid = make_corridor()
    cluster = np.asarray(((7.5, 0.5, 0.5), (7.5, 0.6, 0.5)))
    plan = grid.plan_to_frontiers(
        (0.5, 0.5, 0.5), [cluster], staging_radius=1.0, maximum_motion=3.1
    )
    assert plan is not None
    assert 2 <= len(plan.path) <= 4
    assert np.linalg.norm(np.asarray(plan.path[-1]) - np.asarray(plan.path[0])) <= 3.1
