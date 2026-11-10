#!/usr/bin/env python3

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "sionna_channel_node.py"
)
SPEC = importlib.util.spec_from_file_location("sionna_channel_node", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reciprocal_layout_traces_each_unordered_link_once():
    directed = {(left, right) for left in range(4) for right in range(4) if left != right}
    sources, receivers, routes = MODULE.SionnaChannelNode._reciprocal_solver_layout(
        directed
    )

    assert len(routes) == 6
    assert len(sources) == 3
    assert set(routes) == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    assert all(source in sources for source, _ in routes.values())
    assert all(receiver in receivers for _, receiver in routes.values())


def test_sparse_star_uses_one_ray_shooting_source():
    directed = {(0, 1), (1, 0), (0, 3), (3, 0), (0, 7)}
    sources, receivers, routes = MODULE.SionnaChannelNode._reciprocal_solver_layout(
        directed
    )

    assert sources == [0]
    assert receivers == [1, 3, 7]
    assert routes == {(0, 1): (0, 1), (0, 3): (0, 3), (0, 7): (0, 7)}


def test_ten_uavs_plus_bs_share_reciprocal_geometry():
    directed = {
        (sender, receiver)
        for sender in range(11)
        for receiver in range(11)
        if sender != receiver
    }
    sources, receivers, routes = MODULE.SionnaChannelNode._reciprocal_solver_layout(
        directed
    )

    assert len(directed) == 110
    assert len(routes) == 55
    assert len(sources) == 10
    assert len(receivers) == 10
    assert all(
        (left, right) in routes
        for left in range(10)
        for right in range(left + 1, 11)
    )
