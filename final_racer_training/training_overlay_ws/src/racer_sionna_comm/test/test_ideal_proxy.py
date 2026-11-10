#!/usr/bin/env python3
"""Exercise the serialized ROS boundary without touching message contents."""

import os
import json
import random
import re
import signal
import subprocess
import time

import pytest
import rclpy
from nav_msgs.msg import Odometry
from racer_fidelity_msgs.msg import (
    ChunkData,
    ChunkStamps,
    DroneState,
    GlobalGridAssignment,
    IdxList,
    PairOpt,
    PairOptResponse,
)
from racer_recovery_core.msg import RecoveryCommand, RecoveryStatus
from racer_sionna_interfaces.msg import LinkQuality, LinkQualityArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


def configure_ros_domain(test_offset: int, random_low: int, random_high: int) -> None:
    """Use a caller-reserved domain range when integration tests run in parallel."""
    base = os.environ.get("RACER_TEST_ROS_DOMAIN_ID_BASE")
    domain_id = int(base) + test_offset if base is not None else random.randint(
        random_low, random_high
    )
    if not 0 <= domain_id <= 232:
        raise ValueError(f"invalid ROS domain ID for test: {domain_id}")
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)


@pytest.mark.timeout(30)
def test_ideal_proxy_preserves_order_and_excludes_self():
    configure_ros_domain(0, 201, 229)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=ideal", "-p", "drone_count:=2",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_ideal_proxy_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = []
    self_received = []
    node.create_subscription(
        DroneState,
        "/racer_sionna/rx/drone_1/drone_state",
        lambda message: received.append((message.stamp, list(message.grid_ids))),
        qos,
    )
    node.create_subscription(
        DroneState,
        "/racer_sionna/rx/drone_0/drone_state",
        lambda message: self_received.append(message.stamp),
        qos,
    )
    publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1

        for sequence in (1, 2, 3):
            message = DroneState()
            message.drone_id = 1
            message.stamp = float(sequence)
            message.grid_ids = [sequence, sequence + 10]
            publisher.publish(message)

        deadline = time.monotonic() + 8.0
        while len(received) < 3 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert received == [
            (1.0, [1, 11]),
            (2.0, [2, 12]),
            (3.0, [3, 13]),
        ]
        assert self_received == []
        # Allow the proxy's one-second statistics timer to emit a snapshot
        # before SIGINT so the fast-path accounting can be asserted below.
        time.sleep(1.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["ideal_logical_messages"] == 3
    assert statistics["perfect_forwarding_calls"] == 3
    assert statistics["perfect_forwarding_receivers"] == 3
    assert statistics["attempted_packets"] == 3
    assert statistics["delivered_packets"] == 3
    assert statistics["queued_packets"] == 0
    assert statistics["dropped_per"] == 0
    assert statistics["ideal_statistical_bytes"] == statistics["attempted_bytes"]
    assert statistics["ideal_statistical_transport_blocks"] >= 3


@pytest.mark.timeout(30)
def test_initial_assignment_perfect_window_closes_after_all_epoch_acks():
    configure_ros_domain(13, 80, 109)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=3",
            "-p", "network_topology:=distributed",
            "-p", "initial_assignment_perfect_delivery:=true",
            "-p", "max_retries:=0", "-p", "base_latency_ms:=0.0",
            "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_initial_assignment_perfect_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received_states = {receiver: [] for receiver in range(3)}
    received_assignments = {receiver: [] for receiver in range(3)}
    for receiver in range(3):
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received_states[receiver].append(
                message.stamp
            ),
            qos,
        )
        node.create_subscription(
            GlobalGridAssignment,
            f"/racer_sionna/rx/drone_{receiver}/global_assignment",
            lambda message, receiver=receiver: received_assignments[receiver].append(
                message.epoch
            ),
            qos,
        )
    state_publishers = [
        node.create_publisher(
            DroneState, f"/racer_sionna/tx/drone_{sender}/drone_state", qos
        )
        for sender in range(3)
    ]
    assignment_publisher = node.create_publisher(
        GlobalGridAssignment,
        "/racer_sionna/tx/drone_0/global_assignment",
        qos,
    )

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            any(publisher.get_subscription_count() < 1 for publisher in state_publishers)
            or assignment_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert all(publisher.get_subscription_count() == 1 for publisher in state_publishers)
        assert assignment_publisher.get_subscription_count() == 1

        for sender, publisher in enumerate(state_publishers):
            state = DroneState()
            state.drone_id = sender + 1
            state.stamp = float(sender + 1)
            state.assignment_epoch = 0
            publisher.publish(state)

        assignment = GlobalGridAssignment()
        assignment.coordinator_id = 1
        assignment.epoch = 1
        assignment.offsets = [0, 1, 2, 3]
        assignment.grid_ids = [10, 20, 30]
        assignment_publisher.publish(assignment)
        deadline = time.monotonic() + 5.0
        while (
            sum(len(values) for values in received_assignments.values()) < 2
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        assert received_assignments == {0: [], 1: [1], 2: [1]}

        for sender, publisher in enumerate(state_publishers):
            state = DroneState()
            state.drone_id = sender + 1
            state.stamp = float(sender + 11)
            state.assignment_epoch = 1
            publisher.publish(state)

        deadline = time.monotonic() + 5.0
        while (
            sum(len(values) for values in received_states.values()) < 12
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert sum(len(values) for values in received_states.values()) == 12

        post_window = DroneState()
        post_window.drone_id = 1
        post_window.stamp = 99.0
        post_window.assignment_epoch = 1
        state_publishers[0].publish(post_window)
        deadline = time.monotonic() + 1.3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert all(99.0 not in values for values in received_states.values())
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["initial_assignment_perfect_delivery_enabled"] is True
    assert statistics["initial_assignment_perfect_delivery_complete"] is True
    assert statistics["initial_assignment_perfect_epoch"] == 1
    assert statistics["initial_assignment_perfect_acks"] == 3
    assert statistics["initial_assignment_perfect_drone_state_messages"] == 6
    assert statistics["initial_assignment_perfect_global_assignment_messages"] == 1
    assert statistics["initial_assignment_perfect_forwarded_packets"] == 14
    assert statistics["sionna_direct_attempted_packets"] == 2


@pytest.mark.timeout(30)
def test_initial_assignment_perfect_via_bs_uses_reliable_two_hop_window():
    configure_ros_domain(14, 80, 109)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=3",
            "-p", "network_topology:=bs_round_robin",
            "-p", "initial_assignment_perfect_delivery:=true",
            "-p", "initial_assignment_perfect_via_bs:=true",
            "-p", "max_retries:=0", "-p", "base_latency_ms:=0.0",
            "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_initial_assignment_perfect_bs_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received_states = {receiver: [] for receiver in range(3)}
    received_assignments = {receiver: [] for receiver in range(3)}
    for receiver in range(3):
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received_states[receiver].append(
                message.stamp
            ),
            qos,
        )
        node.create_subscription(
            GlobalGridAssignment,
            f"/racer_sionna/rx/drone_{receiver}/global_assignment",
            lambda message, receiver=receiver: received_assignments[receiver].append(
                message.epoch
            ),
            qos,
        )
    state_publishers = [
        node.create_publisher(
            DroneState, f"/racer_sionna/tx/drone_{sender}/drone_state", qos
        )
        for sender in range(3)
    ]
    assignment_publisher = node.create_publisher(
        GlobalGridAssignment,
        "/racer_sionna/tx/drone_0/global_assignment",
        qos,
    )

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            any(publisher.get_subscription_count() < 1 for publisher in state_publishers)
            or assignment_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert all(publisher.get_subscription_count() == 1 for publisher in state_publishers)
        assert assignment_publisher.get_subscription_count() == 1

        for sender, publisher in enumerate(state_publishers):
            state = DroneState()
            state.drone_id = sender + 1
            state.stamp = float(sender + 1)
            state.assignment_epoch = 0
            publisher.publish(state)

        assignment = GlobalGridAssignment()
        assignment.coordinator_id = 1
        assignment.epoch = 1
        assignment.offsets = [0, 1, 2, 3]
        assignment.grid_ids = [10, 20, 30]
        assignment_publisher.publish(assignment)
        deadline = time.monotonic() + 5.0
        while (
            sum(len(values) for values in received_assignments.values()) < 2
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        assert received_assignments == {0: [], 1: [1], 2: [1]}

        for sender, publisher in enumerate(state_publishers):
            state = DroneState()
            state.drone_id = sender + 1
            state.stamp = float(sender + 11)
            state.assignment_epoch = 1
            publisher.publish(state)

        deadline = time.monotonic() + 5.0
        while (
            sum(len(values) for values in received_states.values()) < 12
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        assert sum(len(values) for values in received_states.values()) == 12
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["initial_assignment_perfect_delivery_enabled"] is True
    assert statistics["initial_assignment_perfect_via_bs"] is True
    assert statistics["initial_assignment_perfect_delivery_complete"] is True
    assert statistics["initial_assignment_perfect_epoch"] == 1
    assert statistics["initial_assignment_perfect_acks"] == 3
    assert statistics["initial_assignment_perfect_bs_uplinks"] == 7
    assert statistics["initial_assignment_perfect_bs_downlinks"] == 14
    assert statistics["initial_assignment_perfect_forwarded_packets"] == 14
    assert statistics["bs_uplink_delivered_packets"] == 7
    assert statistics["bs_downlink_delivered_packets"] == 14
    assert statistics["direct_attempted_packets"] == 0


@pytest.mark.timeout(30)
def test_active_link_hints_are_immediate_then_half_ttl_keepalives():
    configure_ros_domain(8, 120, 149)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=2",
            "-p", "max_retries:=0", "-p", "base_latency_ms:=0.0",
            "-p", "jitter_ms:=0.0", "-p", "active_link_hold_s:=1.0",
            "-p", "active_link_publish_period_s:=0.02",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_active_link_hint_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    hints = []
    node.create_subscription(
        LinkQualityArray,
        "/racer_sionna/active_links",
        lambda message: hints.append(
            [(link.sender_id, link.receiver_id) for link in message.links]
        ),
        qos,
    )
    publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    try:
        deadline = time.monotonic() + 10.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1
        message = DroneState()
        message.drone_id = 1
        message.stamp = 1.0
        publisher.publish(message)
        deadline = time.monotonic() + 1.35
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        nonempty = [hint for hint in hints if hint]
        assert 2 <= len(nonempty) <= 4
        assert all(hint == [(0, 1)] for hint in nonempty)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


@pytest.mark.timeout(30)
def test_bs_links_are_active_before_any_bs_packet_is_enqueued():
    configure_ros_domain(12, 40, 69)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "active_link_hold_s:=1.0",
            "-p", "active_link_publish_period_s:=0.02",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_proactive_bs_link_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    hints = []
    node.create_subscription(
        LinkQualityArray,
        "/racer_sionna/active_links",
        lambda message: hints.append(
            {(link.sender_id, link.receiver_id) for link in message.links}
        ),
        qos,
    )
    try:
        expected = {(0, 2), (2, 0), (1, 2), (2, 1)}
        deadline = time.monotonic() + 5.0
        while expected not in hints and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert expected in hints
        assert process.poll() is None
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


@pytest.mark.timeout(30)
def test_shared_ofdma_multicast_is_one_physical_tx_with_independent_per():
    configure_ros_domain(9, 160, 189)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=10",
            "-p", "network_topology:=distributed",
            "-p", "fixed_mcs_index:=20", "-p", "tx_power_dbm:=20.0",
            "-p", "max_retries:=0", "-p", "base_latency_ms:=0.0",
            "-p", "jitter_ms:=0.0",
            "-p", "uav_ofdma_diagnostic_logging:=true",
            "-p", "uav_ofdma_log_packet_stride:=1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_shared_ofdma_multicast_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = {receiver: [] for receiver in range(1, 10)}
    for receiver in received:
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received[receiver].append(
                message.stamp
            ),
            qos,
        )
    publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )

    def publish_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=2.0)).to_msg()
        for receiver in range(1, 10):
            link = LinkQuality()
            link.stamp = message.stamp
            link.valid_until = valid_until
            link.sender_id = 0
            link.receiver_id = receiver
            link.snr_db = 100.0 if receiver % 2 == 1 else -100.0
            link.model = "deterministic_test_path"
            message.links.append(link)
        link_publisher.publish(message)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1
        for _ in range(5):
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)

        state = DroneState()
        state.drone_id = 1
        state.stamp = 314.0
        publisher.publish(state)
        deadline = time.monotonic() + 5.0
        while (
            any(not received[receiver] for receiver in (1, 3, 5, 7, 9))
            and time.monotonic() < deadline
        ):
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert received == {
            1: [314.0], 2: [], 3: [314.0], 4: [], 5: [314.0],
            6: [], 7: [314.0], 8: [], 9: [314.0],
        }
        time.sleep(1.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["shared_uav_ofdma_enabled"] is True
    assert statistics["uav_transport"] == "UDP"
    assert statistics["uav_udp_retransmissions"] == 0
    assert statistics["uav_udp_datagrams_enqueued"] == 1
    assert statistics["uav_udp_multicast_datagrams"] == 1
    assert statistics["uav_udp_unicast_datagrams"] == 0
    assert statistics["uav_physical_transmissions_started"] == 1
    assert statistics["uav_physical_transmissions_completed"] == 1
    assert statistics["uav_tx_power_applications"] == 1
    assert statistics["uav_ofdma_prb_allocation_events"] == 1
    assert statistics["uav_ofdma_allocated_prb_slots"] == 66
    assert statistics["uav_ofdma_max_allocated_prbs_per_slot"] == 66
    assert statistics["uav_udp_receiver_attempts"] == 9
    assert statistics["uav_udp_receiver_successes"] == 5
    assert statistics["uav_udp_receiver_per_failures"] == 4
    assert statistics["uav_udp_receiver_no_link_failures"] == 0
    assert statistics["uav_udp_receiver_half_duplex_failures"] == 0
    slot_logs = re.findall(r"RACER_UAV_OFDMA_SLOT[^\n]+", output)
    receiver_logs = re.findall(r"RACER_UAV_OFDMA_RX[^\n]+", output)
    assert len(slot_logs) == 1
    assert "allocated_prbs=66" in slot_logs[0]
    assert "receivers=9" in slot_logs[0]
    assert "tx_power_dbm=20.000" in slot_logs[0]
    assert len(receiver_logs) == 9
    assert any("receiver=2" in line and "result=per_failure" in line
               for line in receiver_logs)


@pytest.mark.timeout(30)
def test_shared_ofdma_enforces_half_duplex_for_overlapping_senders():
    configure_ros_domain(10, 190, 220)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=3",
            "-p", "network_topology:=distributed",
            "-p", "fixed_mcs_index:=20", "-p", "tx_power_dbm:=20.0",
            "-p", "max_retries:=0", "-p", "base_latency_ms:=0.0",
            "-p", "jitter_ms:=0.0",
            "-p", "uav_ofdma_diagnostic_logging:=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_shared_ofdma_half_duplex_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = {receiver: [] for receiver in (0, 1, 2)}
    for receiver in received:
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received[receiver].append(
                message.stamp
            ),
            qos,
        )
    publishers = [
        node.create_publisher(
            DroneState, f"/racer_sionna/tx/drone_{sender}/drone_state", qos
        )
        for sender in (0, 1)
    ]
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )

    def publish_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=2.0)).to_msg()
        for sender, receiver in ((0, 1), (0, 2), (1, 0), (1, 2)):
            link = LinkQuality()
            link.stamp = message.stamp
            link.valid_until = valid_until
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 100.0
            link.model = "deterministic_test_path"
            message.links.append(link)
        link_publisher.publish(message)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            any(publisher.get_subscription_count() < 1
                for publisher in publishers)
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert all(publisher.get_subscription_count() == 1
                   for publisher in publishers)
        for _ in range(5):
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)

        for sender, stamp in ((0, 401.0), (1, 402.0)):
            state = DroneState()
            state.drone_id = sender + 1
            state.stamp = stamp
            # Keep both physical transmissions active across many slots so
            # each sending UAV necessarily overlaps the other's multicast.
            state.grid_ids = list(range(20000))
            publishers[sender].publish(state)
        deadline = time.monotonic() + 5.0
        while len(received[2]) < 2 and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert sorted(received[2]) == [401.0, 402.0]
        assert received[0] == []
        assert received[1] == []
        time.sleep(1.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["uav_physical_transmissions_started"] == 2
    assert statistics["uav_physical_transmissions_completed"] == 2
    assert statistics["uav_tx_power_applications"] == 2
    assert statistics["uav_ofdma_peak_active_senders"] == 2
    assert statistics["uav_ofdma_max_allocated_prbs_per_slot"] == 66
    assert statistics["uav_udp_receiver_attempts"] == 4
    assert statistics["uav_udp_receiver_successes"] == 2
    assert statistics["uav_udp_receiver_half_duplex_failures"] == 2


@pytest.mark.timeout(30)
def test_csma_zero_contention_window_reports_shared_channel_collisions():
    configure_ros_domain(21, 221, 232)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=3",
            "-p", "network_topology:=distributed",
            "-p", "uav_channel_access_mode:=csma",
            "-p", "uav_csma_cw_min:=0", "-p", "uav_csma_cw_max:=0",
            "-p", "uav_csma_difs_slots:=0",
            "-p", "fixed_mcs_index:=20", "-p", "tx_power_dbm:=20.0",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "uav_ofdma_diagnostic_logging:=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_csma_collision_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    publishers = [
        node.create_publisher(
            DroneState, f"/racer_sionna/tx/drone_{sender}/drone_state", qos
        )
        for sender in (0, 1)
    ]
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )

    def publish_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=2.0)).to_msg()
        for sender, receiver in ((0, 1), (0, 2), (1, 0), (1, 2)):
            link = LinkQuality()
            link.stamp = message.stamp
            link.valid_until = valid_until
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 100.0
            link.model = "deterministic_test_path"
            message.links.append(link)
        link_publisher.publish(message)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            any(publisher.get_subscription_count() < 1
                for publisher in publishers)
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert all(publisher.get_subscription_count() == 1
                   for publisher in publishers)
        for sequence in range(20):
            for sender, publisher in enumerate(publishers):
                state = DroneState()
                state.drone_id = sender + 1
                state.stamp = 500.0 + sequence
                state.grid_ids = list(range(2000))
                publisher.publish(state)
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.001)
        time.sleep(1.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["uav_channel_access_mode"] == "csma"
    assert statistics["shared_uav_radio_enabled"] is True
    assert statistics["shared_uav_ofdma_enabled"] is False
    assert statistics["uav_csma_collision_events"] >= 1
    assert statistics["uav_csma_collided_datagrams"] >= 2
    assert statistics["uav_udp_receiver_collision_failures"] >= 2
    assert statistics["uav_ofdma_max_allocated_prbs_per_slot"] == 66


@pytest.mark.timeout(30)
def test_nearest_neighbor_topology_delivers_only_to_closest_uav():
    configure_ros_domain(5, 90, 119)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=ideal", "-p", "drone_count:=4",
            "-p", "network_topology:=nearest_neighbors",
            "-p", "nearest_neighbor_count:=1",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_nearest_neighbor_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = {receiver: [] for receiver in (1, 2, 3)}
    for receiver in received:
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received[receiver].append(
                message.stamp
            ),
            qos,
        )
    state_publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    odometry_publishers = [
        node.create_publisher(Odometry, f"/drone_{drone}/odom", 10)
        for drone in range(4)
    ]
    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while state_publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert state_publisher.get_subscription_count() == 1
        for _ in range(10):
            for drone, x_position in enumerate((0.0, 1.0, 3.0, 10.0)):
                odometry = Odometry()
                odometry.pose.pose.position.x = x_position
                odometry_publishers[drone].publish(odometry)
            rclpy.spin_once(node, timeout_sec=0.05)
        message = DroneState()
        message.drone_id = 1
        message.stamp = 23.0
        state_publisher.publish(message)
        deadline = time.monotonic() + 5.0
        while not received[1] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(1.0)
        assert received == {1: [23.0], 2: [], 3: []}
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["nearest_neighbor_count"] == 1
    assert statistics["nearest_filtered_receivers"] >= 2
    assert statistics["dropped_per"] == 0


@pytest.mark.timeout(30)
def test_lossless_nearest_override_keeps_other_receivers_on_sionna_path():
    configure_ros_domain(7, 30, 59)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=4",
            "-p", "network_topology:=distributed",
            "-p", "lossless_nearest_neighbor_count:=1",
            "-p", "max_retries:=0",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_lossless_nearest_override_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = {receiver: [] for receiver in (1, 2, 3)}
    for receiver in received:
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received[receiver].append(
                message.stamp
            ),
            qos,
        )
    state_publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    odometry_publishers = [
        node.create_publisher(Odometry, f"/drone_{drone}/odom", 10)
        for drone in range(4)
    ]

    def publish_sionna_link():
        now = node.get_clock().now()
        links = LinkQualityArray()
        links.stamp = now.to_msg()
        link = LinkQuality()
        link.stamp = links.stamp
        link.valid_until = (now + Duration(seconds=1.0)).to_msg()
        link.sender_id = 0
        link.receiver_id = 3
        link.snr_db = 100.0
        link.model = "sionna_exact"
        links.links.append(link)
        # Receiver 2 intentionally has no Sionna link. Receiver 1 is closest
        # and must be delivered losslessly without any link-quality sample.
        link_publisher.publish(links)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            state_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_sionna_link()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert state_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1
        for _ in range(10):
            for drone, x_position in enumerate((0.0, 1.0, 3.0, 10.0)):
                odometry = Odometry()
                odometry.pose.pose.position.x = x_position
                odometry_publishers[drone].publish(odometry)
            publish_sionna_link()
            rclpy.spin_once(node, timeout_sec=0.05)
        message = DroneState()
        message.drone_id = 1
        message.stamp = 67.0
        state_publisher.publish(message)
        deadline = time.monotonic() + 5.0
        while (
            (not received[1] or not received[3])
            and time.monotonic() < deadline
        ):
            publish_sionna_link()
            rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(1.0)
        assert received == {1: [67.0], 2: [], 3: [67.0]}
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["lossless_nearest_neighbor_count"] == 1
    assert statistics["lossless_nearest_forwarded_packets"] == 1
    assert statistics["sionna_direct_attempted_packets"] == 2
    assert statistics["direct_attempted_packets"] == 3
    assert statistics["direct_delivered_packets"] == 2
    assert statistics["dropped_no_link"] == 1
    assert statistics["sionna_exact_samples"] > 0


@pytest.mark.timeout(35)
def test_selective_nearest_control_and_directed_chunk_budget():
    configure_ros_domain(8, 120, 149)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=4",
            "-p", "network_topology:=distributed",
            "-p", "lossless_nearest_neighbor_count:=2",
            "-p", "lossless_control_only:=true",
            "-p", "directed_message_unicast:=true",
            "-p", "chunk_data_pre_enqueue_dedup:=true",
            "-p", "chunk_data_max_pending_per_link:=2",
            "-p", "max_retries:=0",
            "-p", "base_latency_ms:=500.0", "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_selective_nearest_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    states = {receiver: [] for receiver in (1, 2, 3)}
    chunks = {receiver: [] for receiver in (1, 2, 3)}
    for receiver in states:
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: states[receiver].append(message.stamp),
            qos,
        )
        node.create_subscription(
            ChunkData,
            f"/racer_sionna/rx/drone_{receiver}/chunk_data",
            lambda message, receiver=receiver: chunks[receiver].append(message.idx),
            qos,
        )
    state_publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    odometry_publishers = [
        node.create_publisher(Odometry, f"/drone_{drone}/odom", 10)
        for drone in range(4)
    ]

    def publish_link():
        now = node.get_clock().now()
        links = LinkQualityArray()
        links.stamp = now.to_msg()
        link = LinkQuality()
        link.stamp = links.stamp
        link.valid_until = (now + Duration(seconds=1.0)).to_msg()
        link.sender_id = 0
        link.receiver_id = 3
        link.snr_db = 100.0
        link.model = "sionna_exact"
        links.links.append(link)
        link_publisher.publish(links)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            state_publisher.get_subscription_count() < 1
            or chunk_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert state_publisher.get_subscription_count() == 1
        assert chunk_publisher.get_subscription_count() == 1
        for _ in range(10):
            for drone, x_position in enumerate((0.0, 1.0, 2.0, 10.0)):
                odometry = Odometry()
                odometry.pose.pose.position.x = x_position
                odometry_publishers[drone].publish(odometry)
            rclpy.spin_once(node, timeout_sec=0.05)

        state = DroneState()
        state.drone_id = 1
        state.stamp = 101.0
        state_publisher.publish(state)
        deadline = time.monotonic() + 3.0
        while (
            (not states[1] or not states[2])
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
        assert states == {1: [101.0], 2: [101.0], 3: []}

        publish_link()
        rclpy.spin_once(node, timeout_sec=0.1)
        for index in (1, 2, 3):
            chunk = ChunkData()
            chunk.from_drone_id = 1
            chunk.to_drone_id = 4
            chunk.chunk_drone_id = 1
            chunk.idx = index
            chunk.voxel_adrs = [index]
            chunk.voxel_occ = [1]
            chunk_publisher.publish(chunk)
        deadline = time.monotonic() + 4.0
        while len(chunks[3]) < 2 and time.monotonic() < deadline:
            publish_link()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert chunks == {1: [], 2: [], 3: [1, 2]}

        duplicate = ChunkData()
        duplicate.from_drone_id = 1
        duplicate.to_drone_id = 4
        duplicate.chunk_drone_id = 1
        duplicate.idx = 1
        duplicate.voxel_adrs = [1]
        duplicate.voxel_occ = [1]
        chunk_publisher.publish(duplicate)
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            publish_link()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert chunks == {1: [], 2: [], 3: [1, 2]}
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["lossless_control_only"] is True
    assert statistics["directed_message_unicast"] is True
    assert statistics["chunk_data_pre_enqueue_dedup"] is True
    assert statistics["lossless_nearest_forwarded_packets"] == 2
    assert statistics["directed_unicast_messages"] >= 4
    assert statistics["chunk_enqueue_budget_drops"] >= 1
    assert statistics["chunk_enqueue_duplicates_suppressed"] >= 1


@pytest.mark.timeout(30)
def test_distance_radius_topology_enforces_hard_3d_range():
    configure_ros_domain(6, 60, 89)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=ideal", "-p", "drone_count:=4",
            "-p", "network_topology:=distance_radius",
            "-p", "communication_range_m:=4.0",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_distance_radius_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = {receiver: [] for receiver in (1, 2, 3)}
    for receiver in received:
        node.create_subscription(
            DroneState,
            f"/racer_sionna/rx/drone_{receiver}/drone_state",
            lambda message, receiver=receiver: received[receiver].append(
                message.stamp
            ),
            qos,
        )
    state_publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    odometry_publishers = [
        node.create_publisher(Odometry, f"/drone_{drone}/odom", 10)
        for drone in range(4)
    ]
    positions = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 4.0),
        (2.4, 2.4, 2.4),
        (4.01, 0.0, 0.0),
    )
    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while state_publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert state_publisher.get_subscription_count() == 1
        for _ in range(10):
            for publisher, (x_position, y_position, z_position) in zip(
                odometry_publishers, positions
            ):
                odometry = Odometry()
                odometry.pose.pose.position.x = x_position
                odometry.pose.pose.position.y = y_position
                odometry.pose.pose.position.z = z_position
                publisher.publish(odometry)
            rclpy.spin_once(node, timeout_sec=0.05)
        message = DroneState()
        message.drone_id = 1
        message.stamp = 41.0
        state_publisher.publish(message)
        deadline = time.monotonic() + 5.0
        while not received[1] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(1.0)
        assert received == {1: [41.0], 2: [], 3: []}
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["network_topology"] == "distance_radius"
    assert statistics["communication_range_m"] == 4.0
    assert statistics["range_filtered_receivers"] >= 2
    assert statistics["dropped_queue"] == 0


@pytest.mark.timeout(30)
def test_ap_assisted_proxy_gathers_and_deduplicates():
    configure_ros_domain(1, 180, 199)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=ideal", "-p", "drone_count:=3",
            "-p", "network_topology:=ap_assisted",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_ap_proxy_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    received = {1: [], 2: []}
    node.create_subscription(
        DroneState,
        "/racer_sionna/rx/drone_1/drone_state",
        lambda message: received[1].append(message.stamp),
        qos,
    )
    node.create_subscription(
        DroneState,
        "/racer_sionna/rx/drone_2/drone_state",
        lambda message: received[2].append(message.stamp),
        qos,
    )
    publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1
        message = DroneState()
        message.drone_id = 1
        message.stamp = 17.0
        publisher.publish(message)
        deadline = time.monotonic() + 8.0
        while any(len(values) < 1 for values in received.values()) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        # Leave enough time for any slower duplicate AP copy to arrive.
        duplicate_deadline = time.monotonic() + 1.2
        while time.monotonic() < duplicate_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert received == {1: [17.0], 2: [17.0]}
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["network_topology"] == "ap_assisted"
    assert statistics["ap_global_updates_received"] >= 1
    assert statistics["logical_delivered_packets"] == 2


@pytest.mark.timeout(30)
def test_bs_round_robin_uploads_only_incremental_map_chunks():
    configure_ros_domain(2, 160, 179)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=ideal", "-p", "drone_count:=3",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "bs_min_turn_ms:=5.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_bs_round_robin_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1
        message = ChunkData()
        message.from_drone_id = 1
        message.to_drone_id = 2
        message.chunk_drone_id = 1
        message.idx = 1
        message.voxel_adrs = [17, 23]
        message.voxel_occ = [0, 1]
        publisher.publish(message)
        time.sleep(1.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["network_topology"] == "bs_round_robin"
    assert statistics["bs_round_robin_enabled"] is True
    assert statistics["bs_round_robin_turns"] >= 3
    assert statistics["bs_upload_grants_delivered"] >= 3
    assert statistics["bs_incremental_chunks_received_uplink"] == 1
    assert statistics["bs_known_map_chunks"] == 1
    assert statistics["ap_uplink_attempted_packets"] == 0


@pytest.mark.timeout(30)
def test_bs_round_robin_repairs_a_missing_direct_map_chunk():
    configure_ros_domain(3, 140, 159)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "bs_min_turn_ms:=5.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_bs_repair_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    received = []
    node.create_subscription(
        ChunkData,
        "/racer_sionna/rx/drone_1/chunk_data",
        lambda message: received.append((message.chunk_drone_id, message.idx)),
        qos,
    )

    def publish_bs_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=1.0)).to_msg()
        # Node 2 is the BS. Intentionally omit both direct UAV links so the
        # only possible delivery from UAV 0 to UAV 1 is the scheduled BS path.
        for sender, receiver in ((0, 2), (2, 0), (1, 2), (2, 1)):
            link = LinkQuality()
            link.stamp = message.stamp
            link.valid_until = valid_until
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 40.0
            link.model = "deterministic_test_path"
            message.links.append(link)
        link_publisher.publish(message)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            chunk_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert chunk_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1
        for _ in range(5):
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        chunk = ChunkData()
        chunk.from_drone_id = 1
        chunk.to_drone_id = 2
        chunk.chunk_drone_id = 1
        chunk.idx = 7
        chunk.voxel_adrs = [3, 9]
        chunk.voxel_occ = [1, 0]
        chunk_publisher.publish(chunk)
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert received == [(1, 7)]
        time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["direct_delivered_packets"] == 0
    assert statistics["bs_incremental_chunks_received_uplink"] == 1
    assert statistics["bs_missing_chunks_delivered_downlink"] == 1
    assert statistics["uav_ofdma_total_prbs"] == 33
    phy = statistics["phy"]
    assert phy["bandwidth_hz"] == 100_000_000.0
    assert phy["spectrum_partition_enabled"] is True
    assert phy["uav_broadcast_bandwidth_hz"] == 50_000_000.0
    assert phy["uav_broadcast_resource_blocks"] == 33
    assert phy["fixed_mcs_index"] == 14
    assert phy["uav_mcs_counts"]["16QAM"] > 0
    assert phy["bs_bandwidth_hz"] == 50_000_000.0
    assert phy["bs_resource_blocks"] == 33
    assert phy["bs_mcs_mode"] == "adaptive"
    assert phy["bs_fixed_mcs_index"] == -1
    assert phy["bs_mcs_counts"]["256QAM"] > 0


@pytest.mark.timeout(30)
def test_rl_bs_file_bridge_relays_maps_state_and_excludes_pair_control(tmp_path):
    configure_ros_domain(11, 90, 119)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    ground_truth_path = tmp_path / "ground_truth_occupied_voxels.txt"
    ground_truth_path.write_text("1\n3\n")
    observed_path = tmp_path / "observed_occupied_voxels.txt"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "max_retries:=0", "-p", "bs_max_retries:=0",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "force_bs_perfect_delivery:=true",
            "-p", "rl_bs_decision_period_ms:=10.0",
            "-p", f"rl_bs_action_path:={action_path}",
            "-p", f"rl_bs_state_path:={state_path}",
            "-p", f"ground_truth_occupied_voxels_path:={ground_truth_path}",
            "-p", f"observed_occupied_voxels_path:={observed_path}",
            "-p", "require_ground_truth_map:=true",
            "-p", "task_metric_observer_mode:=async",
            "-p", "sdf_map.resolution:=1.0",
            "-p", "sdf_map.map_size_x:=1.0",
            "-p", "sdf_map.map_size_y:=1.0",
            "-p", "sdf_map.map_size_z:=4.0",
            "-p", "sdf_map.ground_height:=0.0",
            "-p", "sdf_map.box_min_x:=-0.5",
            "-p", "sdf_map.box_min_y:=-0.5",
            "-p", "sdf_map.box_min_z:=0.0",
            "-p", "sdf_map.box_max_x:=0.5",
            "-p", "sdf_map.box_max_y:=0.5",
            "-p", "sdf_map.box_max_z:=4.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_rl_bs_file_bridge_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    state_publisher = node.create_publisher(
        DroneState, "/racer_sionna/tx/drone_0/drone_state", qos
    )
    pair_publisher = node.create_publisher(
        PairOpt, "/racer_sionna/tx/drone_0/pair_opt", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    received = []
    received_states = []
    received_pair_control = []
    node.create_subscription(
        ChunkData,
        "/racer_sionna/rx/drone_1/chunk_data",
        lambda message: received.append((message.chunk_drone_id, message.idx)),
        qos,
    )
    node.create_subscription(
        DroneState,
        "/racer_sionna/rx/drone_1/drone_state",
        lambda message: received_states.append(message.assignment_epoch),
        qos,
    )
    node.create_subscription(
        PairOpt,
        "/racer_sionna/rx/drone_1/pair_opt",
        lambda message: received_pair_control.append(message.transaction_id),
        qos,
    )

    def publish_bs_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        # No direct or BS link is published. The explicit one-run override
        # must make only BS routes usable and lossless.
        link_publisher.publish(message)

    def wait_for(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        payload = None
        while time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.02)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if predicate(payload):
                return payload
        raise AssertionError(f"timed out waiting for RL state; last={payload}")

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            chunk_publisher.get_subscription_count() < 1
            or pair_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert chunk_publisher.get_subscription_count() == 1
        chunk = ChunkData()
        chunk.from_drone_id = 1
        chunk.to_drone_id = 2
        chunk.chunk_drone_id = 1
        chunk.idx = 41
        # The BS receives one occupied and two free known voxels. Coverage
        # therefore counts 3/4 planning-box voxels, while occupied-map IoU
        # remains 1/(1+2-1) = 0.5.
        chunk.voxel_adrs = [1, 2, 0]
        chunk.voxel_occ = [1, 0, 0]
        chunk_publisher.publish(chunk)
        state = DroneState()
        state.drone_id = 1
        state.assignment_epoch = 23
        state.grid_ids = [4, 8]
        state_publisher.publish(state)
        pair = PairOpt()
        pair.from_drone_id = 1
        pair.to_drone_id = 2
        pair.phase = PairOpt.PHASE_PROPOSE
        pair.transaction_id = 901
        pair_publisher.publish(pair)
        # Wait until the proxy has observed the source chunk before publishing
        # the one-shot action epoch. Sleeping here is racy under the full test
        # suite because DDS delivery and the 10 ms scheduler run independently.
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [1, 0]
        )
        for _ in range(5):
            state_publisher.publish(state)
            rclpy.spin_once(node, timeout_sec=0.02)

        # N=2 canonical bits: B01, B10, u0, u1. Upload sender 0 once.
        action_path.write_text("1 2 0 0 1 0\n")
        upload_state = wait_for(
            lambda value: value.get("action_epoch") == 1
            and value.get("map_summary", {}).get("bs_known_chunks") == 1
        )
        assert upload_state["bs_uplink_prb_slots"] > 0
        assert upload_state["channel_snr_db"][0][2] == 40.0
        assert upload_state["channel_snr_db"][2][1] == 40.0
        assert upload_state["channel_snr_db"][0][1] == -120.0
        assert upload_state["bs_downlink_prb_slots"] == 0
        assert upload_state["redundant_exploration_ratio"] == 0.0
        assert upload_state["bs_global_map_iou"] == 0.5
        assert upload_state["bs_global_map_coverage"] == 0.75
        assert upload_state["bs_global_map_known_voxels"] == 3
        assert upload_state["bs_global_map_total_voxels"] == 4
        assert upload_state["task_time_s"] == pytest.approx(
            upload_state["task_step"]
            * upload_state["task_step_duration_s"]
        )

        # Select route B[0,1]; BS sends every cached chunk UAV 1 lacks.
        action_path.write_text("2 2 1 0 0 0\n")
        relay_state = wait_for(
            lambda value: value.get("action_epoch") == 2
            and value.get("bs_downlink_prb_slots", 0) > 0
        )
        deadline = time.monotonic() + 3.0
        while (not received or not received_states) and time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert received == [(1, 41)]
        assert received_states == [23]
        assert received_pair_control == []
        assert relay_state["bs_uplink_prb_slots"] == upload_state["bs_uplink_prb_slots"]
        time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["rl_bs_scheduler_enabled"] is True
    assert statistics["direct_delivered_packets"] == 0
    assert statistics["bs_incremental_chunks_received_uplink"] == 1
    assert statistics["bs_missing_chunks_delivered_downlink"] == 1
    assert statistics["bs_peer_data_matches_uav_topics"] is True
    assert statistics["bs_relay_topic_policy"] == (
        "chunk_data_drone_state_trajectory_global_assignment"
    )
    assert statistics["bs_pair_control_relay_enabled"] is False
    assert statistics["bs_peer_uplink_delivered"] >= 1
    assert statistics["bs_peer_downlink_delivered"] >= 1
    pair_counters = statistics["bs_peer_topic_counters"]
    assert pair_counters["pair_opt"] == {
        "uplink_scheduled": 0,
        "uplink_delivered": 0,
        "downlink_scheduled": 0,
        "downlink_delivered": 0,
    }
    assert pair_counters["pair_opt_res"] == {
        "uplink_scheduled": 0,
        "uplink_delivered": 0,
        "downlink_scheduled": 0,
        "downlink_delivered": 0,
    }
    assert pair_counters["drone_state"]["uplink_delivered"] >= 1
    assert pair_counters["drone_state"]["downlink_delivered"] >= 1
    assert statistics["bs_uplink_prb_slots"] > 0
    assert statistics["bs_downlink_prb_slots"] > 0
    assert statistics["redundant_exploration_ratio"] == 0.0
    assert statistics["bs_global_map_iou"] == 0.5
    cache_checks = re.findall(
        r"RACER_INCREMENTAL_CACHE_VALIDATION mismatches=(\d+)", output
    )
    assert cache_checks and set(cache_checks) == {"0"}
    assert statistics["bs_global_map_coverage"] == 0.75
    assert statistics["bs_global_map_coverage_definition"] == (
        "known BS SDF voxels / configured planning-box voxels"
    )
    assert statistics["bs_global_map_known_voxels"] == 3
    assert statistics["bs_global_map_total_voxels"] == 4
    assert statistics["ground_truth_occupied_voxels"] == 2
    assert statistics["task_metric_observer_mode"] == "async"
    assert statistics["task_metric_observer_events_enqueued"] > 0
    assert (
        statistics["task_metric_observer_events_processed"]
        == statistics["task_metric_observer_events_enqueued"]
    )
    assert statistics["task_quality_history"]
    assert all(
        "task_step" in item
        and "task_time_s" in item
        and "bs_global_map_coverage" in item
        for item in statistics["task_quality_history"]
    )
    assert observed_path.read_text().splitlines() == ["1"]


@pytest.mark.timeout(35)
def test_rl_bs_forwards_foreign_origin_chunk_and_keeps_hops_independent(tmp_path):
    """A relay may upload a foreign chunk; a failed UL cannot block cached DL."""
    configure_ros_domain(14, 50, 79)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=3",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "max_retries:=0", "-p", "bs_max_retries:=0",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "force_bs_perfect_delivery:=false",
            "-p", "rl_bs_decision_period_ms:=10.0",
            "-p", f"rl_bs_action_path:={action_path}",
            "-p", f"rl_bs_state_path:={state_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_rl_bs_multihop_map_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    chunk_publishers = [
        node.create_publisher(
            ChunkData, f"/racer_sionna/tx/drone_{sender}/chunk_data", qos
        )
        for sender in range(3)
    ]
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    received = []
    node.create_subscription(
        ChunkData,
        "/racer_sionna/rx/drone_2/chunk_data",
        lambda message: received.append(
            (
                message.from_drone_id,
                message.to_drone_id,
                message.chunk_drone_id,
                message.idx,
            )
        ),
        qos,
    )

    def publish_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=1.0)).to_msg()
        # Node 3 is the BS. UAV 1 (zero-based index 1) can upload and the BS
        # can reach UAV 2. UAV 0 has no uplink at all.
        for sender, receiver in ((1, 3), (3, 2)):
            link = LinkQuality()
            link.stamp = message.stamp
            link.valid_until = valid_until
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 40.0
            link.model = "deterministic_test_path"
            message.links.append(link)
        link_publisher.publish(message)

    def wait_for(predicate, timeout=6.0):
        deadline = time.monotonic() + timeout
        payload = None
        while time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.02)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if predicate(payload):
                return payload
        raise AssertionError(f"timed out waiting for RL state; last={payload}")

    def write_action(epoch, relay_pairs=(), uploads=()):
        relay_pairs = set(relay_pairs)
        uploads = set(uploads)
        bits = []
        for sender in range(3):
            for receiver in range(3):
                if sender != receiver:
                    bits.append(int((sender, receiver) in relay_pairs))
        bits.extend(int(sender in uploads) for sender in range(3))
        action_path.write_text(
            f"{epoch} 3 " + " ".join(str(bit) for bit in bits) + "\n"
        )

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            any(
                publisher.get_subscription_count() < 1
                for publisher in chunk_publishers
            )
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert all(
            publisher.get_subscription_count() == 1
            for publisher in chunk_publishers
        )
        assert link_publisher.get_subscription_count() == 1

        # UAV 1 currently holds a chunk originally mapped by UAV 0. This is
        # the exact owner != current_sender case that the former filter lost.
        cached = ChunkData()
        cached.from_drone_id = 2
        cached.to_drone_id = 3
        cached.chunk_drone_id = 1
        cached.idx = 101
        cached.voxel_adrs = [11]
        cached.voxel_occ = [1]
        chunk_publishers[1].publish(cached)
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [0, 1, 0]
        )

        # Standalone u[1] uploads C_1 - C_BS. The immutable owner remains 0.
        write_action(1, uploads={1})
        uploaded = wait_for(
            lambda value: value.get("action_epoch") == 1
            and value.get("map_summary", {}).get("bs_known_chunks") == 1
        )
        assert [row[-1] for row in uploaded["information_version_gap"]] == [
            0, 0, 0
        ]

        # UAV 0 owns a newer chunk that BS has not seen. B[0,2] attempts that
        # unavailable uplink and, independently, sends BS's old cached chunk
        # to UAV 2 over the usable downlink.
        newest = ChunkData()
        newest.from_drone_id = 1
        newest.to_drone_id = 3
        newest.chunk_drone_id = 1
        newest.idx = 102
        newest.voxel_adrs = [12]
        newest.voxel_occ = [0]
        chunk_publishers[0].publish(newest)
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [1, 1, 0]
        )
        write_action(2, relay_pairs={(0, 2)})
        independent = wait_for(
            lambda value: value.get("action_epoch") == 2
            and value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [1, 1, 1]
        )
        assert independent["map_summary"]["bs_known_chunks"] == 1
        assert independent["information_version_gap"][0][-1] == 1
        assert received == [(4, 3, 1, 101)]
        time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["bs_incremental_chunks_received_uplink"] == 1
    assert statistics["bs_known_map_chunks"] == 1
    assert statistics["bs_missing_chunks_delivered_downlink"] == 1
    assert statistics["dropped_no_link"] >= 1
    cache_checks = re.findall(
        r"RACER_INCREMENTAL_CACHE_VALIDATION mismatches=(\d+)", output
    )
    assert cache_checks and set(cache_checks) == {"0"}


@pytest.mark.timeout(35)
def test_rl_upload_materializes_uncached_racer_chunks_and_caps_each_slot(tmp_path):
    """A slot pipelines many chunks while each keeps its propagation delay."""
    configure_ros_domain(16, 1, 29)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=1",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=250.0", "-p", "jitter_ms:=0.0",
            "-p", "max_retries:=0", "-p", "bs_max_retries:=0",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "force_bs_perfect_delivery:=false",
            "-p", "rl_bs_decision_period_ms:=10.0",
            "-p", f"rl_bs_action_path:={action_path}",
            "-p", f"rl_bs_state_path:={state_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_rl_bs_on_demand_chunk_test")
    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
    stamp_publisher = node.create_publisher(
        ChunkStamps, "/racer_sionna/tx/drone_0/chunk_stamps", qos
    )
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    requested = []

    def publish_links():
        now = node.get_clock().now()
        links = LinkQualityArray()
        links.stamp = now.to_msg()
        link = LinkQuality()
        link.stamp = links.stamp
        link.valid_until = (now + Duration(seconds=1.0)).to_msg()
        link.sender_id = 0
        link.receiver_id = 1
        link.snr_db = 40.0
        link.model = "deterministic_test_path"
        links.links.append(link)
        link_publisher.publish(links)

    def on_request(message):
        if message.from_drone_id != 0:
            return
        keys = []
        for owner, ranges in enumerate(message.idx_lists, start=1):
            for offset in range(0, len(ranges.ids), 2):
                keys.extend(
                    (owner, index)
                    for index in range(
                        ranges.ids[offset], ranges.ids[offset + 1] + 1
                    )
                )
        requested.extend(keys)
        # Simulate MultiMapManager::sendChunks(): materialized data re-enters
        # the Proxy on the normal UAV TX topic with the BS-local target marker.
        for owner, index in keys:
            chunk = ChunkData()
            chunk.from_drone_id = 1
            chunk.to_drone_id = 0
            chunk.chunk_drone_id = owner
            chunk.idx = index
            chunk.voxel_adrs = [1000 + index]
            chunk.voxel_occ = [index % 2]
            chunk_publisher.publish(chunk)

    node.create_subscription(
        ChunkStamps,
        "/racer_sionna/rx/drone_0/chunk_stamps",
        on_request,
        qos,
    )

    def wait_for(predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        payload = None
        while time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.02)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if predicate(payload):
                return payload
        raise AssertionError(f"timed out waiting for RL state; last={payload}")

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            stamp_publisher.get_subscription_count() < 1
            or chunk_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert stamp_publisher.get_subscription_count() == 1
        assert chunk_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1

        inventory = ChunkStamps()
        inventory.from_drone_id = 1
        inventory.time = 1.0
        owner = IdxList()
        owner.ids = [1, 40]
        inventory.idx_lists = [owner]
        stamp_publisher.publish(inventory)
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [40]
        )

        # One action is held over several scheduler slots. Admission follows
        # the physical slot quota without an extra default 32-chunk window.
        upload_started = time.monotonic()
        action_path.write_text("1 1 1\n")
        first = wait_for(
            lambda value: value.get("action_epoch") == 1
            and value.get("map_summary", {}).get("bs_known_chunks", 0) >= 32
        )
        assert first["map_summary"]["local_known_chunks"] == [40]
        settle_deadline = time.monotonic() + 0.4
        while time.monotonic() < settle_deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert len(requested) == 40
        assert len(set(requested)) == 40

        # A new action finds no remaining deficit and cannot duplicate prior
        # requests or uploads.
        action_path.write_text("2 1 1\n")
        wait_for(
            lambda value: value.get("action_epoch") == 2
            and value.get("map_summary", {}).get("bs_known_chunks") == 40
        )
        # Forty packets finish close to one 250 ms propagation delay. A
        # stop-and-wait link would require roughly 40 * 250 ms here.
        assert time.monotonic() - upload_started < 2.0
        assert len(requested) == 40
        assert len(set(requested)) == 40
        time.sleep(1.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["bs_chunk_payload_request_messages"] == 1
    assert statistics["bs_chunk_payload_chunks_requested"] == 40
    assert statistics["bs_chunk_payload_responses_received"] == 40
    assert statistics["bs_chunk_payload_request_timeouts"] == 0
    assert statistics["bs_chunk_payload_unsolicited_responses"] == 0
    assert statistics["bs_incremental_chunks_received_uplink"] == 40
    assert statistics["bs_known_map_chunks"] == 40
    assert statistics["bs_uplink_attempted_packets"] == 40
    assert statistics["bs_uplink_delivered_packets"] == 40
    assert statistics["bs_adaptive_uplink_chunks_selected_total"] == 40
    assert statistics["bs_max_inflight_chunks_per_uav"] == 0
    assert statistics["bs_upload_selection_backpressure_skips"] == 0
    assert statistics["bs_adaptive_budget_mcs_counts"]["256QAM"] > 0


@pytest.mark.timeout(30)
def test_pushed_payload_bypasses_fetch_but_waits_for_rl_upload_action(tmp_path):
    """Generation pushes payload locally; missing pushes use fetch fallback."""
    configure_ros_domain(19, 1, 29)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=1",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "bandwidth_hz:=4000000.0", "-p", "resource_blocks:=2",
            "-p", "uav_broadcast_bandwidth_hz:=2000000.0",
            "-p", "uav_broadcast_resource_blocks:=1",
            "-p", "bs_bandwidth_hz:=2000000.0", "-p", "bs_resource_blocks:=1",
            "-p", "max_retries:=0", "-p", "bs_max_retries:=0",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "force_bs_perfect_delivery:=false",
            "-p", "bs_payload_precache_enabled:=true",
            "-p", "rl_bs_decision_period_ms:=10.0",
            "-p", f"rl_bs_action_path:={action_path}",
            "-p", f"rl_bs_state_path:={state_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_deferred_payload_fetch_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    stamp_publisher = node.create_publisher(
        ChunkStamps, "/racer_sionna/tx/drone_0/chunk_stamps", qos
    )
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    cache_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/cache/drone_0/chunk_data", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    requested = []

    def on_request(message):
        if message.from_drone_id != 0:
            return
        for owner, ranges in enumerate(message.idx_lists, start=1):
            for offset in range(0, len(ranges.ids), 2):
                requested.extend(
                    (owner, index)
                    for index in range(
                        ranges.ids[offset], ranges.ids[offset + 1] + 1
                    )
                )

    node.create_subscription(
        ChunkStamps,
        "/racer_sionna/rx/drone_0/chunk_stamps",
        on_request,
        qos,
    )

    def publish_link():
        now = node.get_clock().now()
        links = LinkQualityArray()
        links.stamp = now.to_msg()
        link = LinkQuality()
        link.stamp = links.stamp
        link.valid_until = (now + Duration(seconds=1.0)).to_msg()
        link.sender_id = 0
        link.receiver_id = 1
        link.snr_db = 40.0
        link.model = "deterministic_test_path"
        links.links.append(link)
        link_publisher.publish(links)

    def wait_for(predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        payload = None
        while time.monotonic() < deadline:
            publish_link()
            rclpy.spin_once(node, timeout_sec=0.02)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if predicate(payload):
                return payload
        raise AssertionError(f"timed out waiting for RL state; last={payload}")

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            stamp_publisher.get_subscription_count() < 1
            or chunk_publisher.get_subscription_count() < 1
            or cache_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_link()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert stamp_publisher.get_subscription_count() == 1
        assert chunk_publisher.get_subscription_count() == 1
        assert cache_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1

        pushed_chunk = ChunkData()
        pushed_chunk.from_drone_id = 1
        pushed_chunk.to_drone_id = -1
        pushed_chunk.chunk_drone_id = 1
        pushed_chunk.idx = 1
        # This real payload needs more than one 20 ms slot on the deliberately
        # narrow BS carrier. It must serialize across slots without blocking
        # the following repair-fetched chunk forever.
        pushed_chunk.voxel_adrs = list(range(3000))
        pushed_chunk.voxel_occ = [1] * 3000
        cache_publisher.publish(pushed_chunk)
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [1]
        )

        inventory = ChunkStamps()
        inventory.from_drone_id = 1
        inventory.time = 1.0
        owner = IdxList()
        owner.ids = [1, 2]
        inventory.idx_lists = [owner]
        stamp_publisher.publish(inventory)
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [2]
        )

        # Chunk 1 was pushed with its payload and needs no request. Chunk 2
        # deliberately omits the push and exercises the repair fetch.
        wait_for(lambda value: requested)
        assert requested == [(1, 2)]

        action_path.write_text("1 1 0\n")
        wait_for(lambda value: value.get("action_epoch") == 1)
        chunk = ChunkData()
        chunk.from_drone_id = 1
        chunk.to_drone_id = 0
        chunk.chunk_drone_id = 1
        chunk.idx = 2
        chunk.voxel_adrs = [1002]
        chunk.voxel_occ = [1]
        chunk_publisher.publish(chunk)
        # Materialization is local simulator plumbing. It must not bypass the
        # current zero upload action or count as BS delivery.
        settle_deadline = time.monotonic() + 0.3
        latest = None
        while time.monotonic() < settle_deadline:
            publish_link()
            rclpy.spin_once(node, timeout_sec=0.02)
            try:
                latest = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        assert latest is not None
        assert latest["map_summary"]["bs_known_chunks"] == 0

        action_path.write_text("2 1 1\n")
        wait_for(
            lambda value: value.get("action_epoch") == 2
            and value.get("map_summary", {}).get("bs_known_chunks") == 2
        )
        assert requested == [(1, 2)]
        time.sleep(1.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches, output
    statistics = json.loads(matches[-1])
    assert statistics["bs_chunk_payload_chunks_requested"] == 1
    assert statistics["bs_chunk_payload_responses_received"] == 1
    assert statistics["payload_precache_queued"] == 1
    assert statistics["payload_precache_chunks_requested"] == 1
    assert statistics["payload_precache_responses_received"] == 1
    assert statistics["payload_precache_timeouts"] == 0
    assert statistics["payload_direct_pushes_received"] == 1
    assert statistics["payload_direct_pushes_rejected"] == 0
    assert statistics["bs_incremental_chunks_scheduled_uplink"] == 2
    assert statistics["bs_incremental_chunks_received_uplink"] == 2
    assert statistics["bs_uplink_attempted_packets"] == 2


@pytest.mark.timeout(30)
def test_periodic_bs_upload_is_link_aware_and_backpressured():
    """A fast polling clock must leave only one bounded chunk batch in flight."""
    configure_ros_domain(18, 1, 29)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=1",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "bs_max_retries:=0",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "bs_periodic_upload_request_enabled:=true",
            "-p", "bs_periodic_upload_request_period_ms:=10.0",
            "-p", "bs_periodic_upload_chunks_per_request:=4",
            "-p", "bs_max_inflight_chunks_per_uav:=4",
            "-p", "bs_control_ttl_s:=2.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_periodic_upload_backpressure_test")
    qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
    stamp_publisher = node.create_publisher(
        ChunkStamps, "/racer_sionna/tx/drone_0/chunk_stamps", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    requested = []

    def on_request(message):
        if message.from_drone_id != 0:
            return
        for owner, ranges in enumerate(message.idx_lists, start=1):
            for offset in range(0, len(ranges.ids), 2):
                requested.extend(
                    (owner, index)
                    for index in range(
                        ranges.ids[offset], ranges.ids[offset + 1] + 1
                    )
                )

    node.create_subscription(
        ChunkStamps,
        "/racer_sionna/rx/drone_0/chunk_stamps",
        on_request,
        qos,
    )

    def publish_links():
        now = node.get_clock().now()
        links = LinkQualityArray()
        links.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=1.0)).to_msg()
        for sender, receiver in ((0, 1), (1, 0)):
            link = LinkQuality()
            link.stamp = links.stamp
            link.valid_until = valid_until
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 40.0
            link.model = "deterministic_test_path"
            links.links.append(link)
        link_publisher.publish(links)

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            stamp_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert stamp_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1

        inventory = ChunkStamps()
        inventory.from_drone_id = 1
        inventory.time = 1.0
        owner = IdxList()
        owner.ids = [1, 100]
        inventory.idx_lists = [owner]
        stamp_publisher.publish(inventory)

        deadline = time.monotonic() + 1.4
        while time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.01)
        # Intentionally do not materialize a response. The first four pending
        # payloads must apply backpressure to every later 10 ms opportunity.
        assert len(requested) == 4
        assert len(set(requested)) == 4
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["bs_periodic_upload_request_opportunities"] > 4
    assert statistics["bs_periodic_upload_requests_scheduled"] >= 1
    assert statistics["bs_periodic_upload_requests_delivered"] >= 1
    assert statistics["bs_periodic_upload_requests_suppressed_backpressure"] > 0
    assert statistics["bs_chunk_payload_chunks_requested"] == 4
    assert statistics["bs_chunk_payload_request_timeouts"] == 0


@pytest.mark.timeout(30)
def test_rl_relay_action_reuses_on_demand_uplink_materialization(tmp_path):
    """B[i,j] uses the same true-inventory uplink path as standalone u[i]."""
    configure_ros_domain(17, 1, 29)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "max_retries:=0", "-p", "bs_max_retries:=0",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "force_bs_perfect_delivery:=false",
            "-p", "rl_bs_decision_period_ms:=10.0",
            "-p", f"rl_bs_action_path:={action_path}",
            "-p", f"rl_bs_state_path:={state_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_rl_relay_on_demand_chunk_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    stamp_publisher = node.create_publisher(
        ChunkStamps, "/racer_sionna/tx/drone_0/chunk_stamps", qos
    )
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    requested = []

    def publish_links():
        now = node.get_clock().now()
        links = LinkQualityArray()
        links.stamp = now.to_msg()
        for sender, receiver in ((0, 2), (2, 1)):
            link = LinkQuality()
            link.stamp = links.stamp
            link.valid_until = (now + Duration(seconds=1.0)).to_msg()
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 40.0
            link.model = "deterministic_test_path"
            links.links.append(link)
        link_publisher.publish(links)

    def on_request(message):
        if message.from_drone_id != 0:
            return
        assert list(message.idx_lists[0].ids) == []
        assert list(message.idx_lists[1].ids) == [7, 7]
        requested.append((2, 7))
        chunk = ChunkData()
        chunk.from_drone_id = 1
        chunk.to_drone_id = 0
        chunk.chunk_drone_id = 2
        chunk.idx = 7
        chunk.voxel_adrs = [1007]
        chunk.voxel_occ = [1]
        chunk_publisher.publish(chunk)

    node.create_subscription(
        ChunkStamps,
        "/racer_sionna/rx/drone_0/chunk_stamps",
        on_request,
        qos,
    )

    def wait_for(predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        payload = None
        while time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.02)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if predicate(payload):
                return payload
        raise AssertionError(f"timed out waiting for RL state; last={payload}")

    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            stamp_publisher.get_subscription_count() < 1
            or chunk_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert stamp_publisher.get_subscription_count() == 1
        assert chunk_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1

        inventory = ChunkStamps()
        inventory.from_drone_id = 1
        inventory.time = 1.0
        owner = IdxList()
        owner.ids = [7, 7]
        # UAV 0 owns a chunk whose immutable origin is UAV 1.
        inventory.idx_lists = [IdxList(), owner]
        stamp_publisher.publish(inventory)
        wait_for(
            lambda value: value.get("map_summary", {}).get(
                "local_known_chunks"
            ) == [1, 0]
        )

        # B[0,1]=1, B[1,0]=0, u[0]=u[1]=0.  The relay row alone must trigger
        # the same on-demand 0->BS upload path.
        action_path.write_text("1 2 1 0 0 0\n")
        wait_for(
            lambda value: value.get("action_epoch") == 1
            and value.get("map_summary", {}).get("bs_known_chunks") == 1
        )
        assert requested == [(2, 7)]
        time.sleep(1.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches
    statistics = json.loads(matches[-1])
    assert statistics["bs_chunk_payload_chunks_requested"] == 1
    assert statistics["bs_chunk_payload_responses_received"] == 1
    assert statistics["bs_incremental_chunks_received_uplink"] == 1
    assert statistics["bs_uplink_attempted_packets"] == 1
    assert statistics["bs_uplink_delivered_packets"] == 1


@pytest.mark.timeout(30)
def test_bs_pair_control_reservation_completes_after_rl_deselects_pair(tmp_path):
    configure_ros_domain(15, 30, 59)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=sionna", "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "max_retries:=0", "-p", "bs_max_retries:=2",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "force_bs_perfect_delivery:=false",
            "-p", "pair_control_reservation_enabled:=true",
            "-p", "pair_control_reservation_s:=1.8",
            "-p", "rl_bs_decision_period_ms:=10.0",
            "-p", f"rl_bs_action_path:={action_path}",
            "-p", f"rl_bs_state_path:={state_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_pair_control_reservation_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    propose_publisher = node.create_publisher(
        PairOpt, "/racer_sionna/tx/drone_0/pair_opt", qos
    )
    response_publisher = node.create_publisher(
        PairOptResponse, "/racer_sionna/tx/drone_1/pair_opt_res", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )
    proposals_at_responder = []
    responses_at_proposer = []
    node.create_subscription(
        PairOpt,
        "/racer_sionna/rx/drone_1/pair_opt",
        lambda message: proposals_at_responder.append(
            (message.transaction_id, message.phase)
        ),
        qos,
    )
    node.create_subscription(
        PairOptResponse,
        "/racer_sionna/rx/drone_0/pair_opt_res",
        lambda message: responses_at_proposer.append(
            (message.transaction_id, message.status)
        ),
        qos,
    )

    def publish_bs_links():
        now = node.get_clock().now()
        message = LinkQualityArray()
        message.stamp = now.to_msg()
        valid_until = (now + Duration(seconds=1.0)).to_msg()
        # No UAV-UAV link is available. Every delivered transaction stage must
        # consume the normal, lossy Sionna BS uplink and downlink resources.
        for sender, receiver in ((0, 2), (2, 0), (1, 2), (2, 1)):
            link = LinkQuality()
            link.stamp = message.stamp
            link.valid_until = valid_until
            link.sender_id = sender
            link.receiver_id = receiver
            link.snr_db = 40.0
            link.model = "deterministic_test_path"
            message.links.append(link)
        link_publisher.publish(message)

    def spin_until(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.01)
            if predicate():
                return
        raise AssertionError("timed out waiting for reserved control stage")

    def wait_for_action_epoch(epoch, timeout=5.0):
        payload = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.01)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("action_epoch") == epoch:
                return payload
        raise AssertionError(f"timed out waiting for action {epoch}; last={payload}")

    def wait_for_usable_channel_snapshot(timeout=5.0):
        payload = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.01)
            try:
                payload = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            channel = payload.get("channel_snr_db", [])
            if (
                len(channel) == 3
                and all(
                    channel[sender][receiver] > 30.0
                    for sender, receiver in ((0, 2), (2, 0), (1, 2), (2, 1))
                )
            ):
                return payload
        raise AssertionError(
            f"timed out waiting for usable channel snapshot; last={payload}"
        )

    transaction_id = 77
    output = ""
    try:
        deadline = time.monotonic() + 10.0
        while (
            propose_publisher.get_subscription_count() < 1
            or response_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            publish_bs_links()
            rclpy.spin_once(node, timeout_sec=0.05)
        assert propose_publisher.get_subscription_count() == 1
        assert response_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1

        # N=2 canonical bits: B01, B10, u0, u1. RL schedules only the initial
        # proposer upload and its relay to the responder.
        action_state = wait_for_usable_channel_snapshot()
        action_path.write_text(
            "1 2 1 0 1 0 "
            f"{action_state['channel_version']} {action_state['task_step']}\n"
        )
        wait_for_action_epoch(1)
        proposal = PairOpt()
        proposal.from_drone_id = 1
        proposal.to_drone_id = 2
        proposal.phase = PairOpt.PHASE_PROPOSE
        proposal.transaction_id = transaction_id
        proposal.from_assignment_epoch = 3
        proposal.to_assignment_epoch = 4
        proposal.new_assignment_epoch = 5
        proposal.commitment_until = 100.0
        propose_publisher.publish(proposal)
        spin_until(
            lambda: (transaction_id, PairOpt.PHASE_PROPOSE)
            in proposals_at_responder
        )

        # Deselect every BS action before generating PREPARED. The remaining
        # three messages can now move only through the temporary reservation.
        action_state = wait_for_usable_channel_snapshot()
        action_path.write_text(
            "2 2 0 0 0 0 "
            f"{action_state['channel_version']} {action_state['task_step']}\n"
        )
        wait_for_action_epoch(2)
        prepared = PairOptResponse()
        prepared.from_drone_id = 2
        prepared.to_drone_id = 1
        prepared.status = PairOptResponse.STATUS_PREPARED
        prepared.transaction_id = transaction_id
        prepared.responder_assignment_epoch = 4
        response_publisher.publish(prepared)
        spin_until(
            lambda: (transaction_id, PairOptResponse.STATUS_PREPARED)
            in responses_at_proposer
        )

        commit = PairOpt()
        commit.from_drone_id = 1
        commit.to_drone_id = 2
        commit.phase = PairOpt.PHASE_COMMIT
        commit.transaction_id = transaction_id
        commit.from_assignment_epoch = 3
        commit.to_assignment_epoch = 4
        commit.new_assignment_epoch = 5
        commit.commitment_until = 100.0
        propose_publisher.publish(commit)
        spin_until(
            lambda: (transaction_id, PairOpt.PHASE_COMMIT)
            in proposals_at_responder
        )

        committed = PairOptResponse()
        committed.from_drone_id = 2
        committed.to_drone_id = 1
        committed.status = PairOptResponse.STATUS_COMMITTED
        committed.transaction_id = transaction_id
        committed.responder_assignment_epoch = 5
        response_publisher.publish(committed)
        spin_until(
            lambda: (transaction_id, PairOptResponse.STATUS_COMMITTED)
            in responses_at_proposer
        )
        # Leave a full statistics-timer period after the final transaction;
        # loaded CI executors can dispatch the 1 Hz wall timer slightly late.
        time.sleep(1.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)

    matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)
    assert matches, output
    statistics = json.loads(matches[-1])
    assert statistics["force_bs_perfect_delivery"] is False
    assert statistics["pair_control_reservation_enabled"] is True
    assert statistics["pair_control_reservation_s"] == pytest.approx(1.8)
    assert statistics["pair_control_reservations_started"] == 1
    assert statistics["pair_control_reservations_completed"] == 1
    assert statistics["pair_control_reservations_active"] == 0
    assert statistics["pair_control_reservations_expired"] == 0
    assert statistics["pair_control_reserved_prepared_delivered"] == 1
    assert statistics["pair_control_reserved_commits_delivered"] == 1
    assert statistics["pair_control_reserved_uplinks_delivered"] >= 3
    assert statistics["pair_control_reserved_downlinks_delivered"] >= 3
    assert statistics["pair_control_reserved_delivered_packets"] >= 6
    assert statistics["pair_control_reserved_delivered_bytes"] > 0
    assert statistics["pair_control_reserved_uplink_prb_slots"] > 0
    assert statistics["pair_control_reserved_downlink_prb_slots"] > 0
    assert statistics["pair_control_reserved_mean_hop_delay_ms"] > 0.0
    assert statistics["pair_control_reserved_mean_end_to_end_delay_ms"] > 0.0
    assert statistics["bs_uplink_prb_slots"] >= statistics[
        "pair_control_reserved_uplink_prb_slots"
    ]
    assert statistics["bs_downlink_prb_slots"] >= statistics[
        "pair_control_reserved_downlink_prb_slots"
    ]
    assert statistics["phy"]["bs_bandwidth_hz"] == 50_000_000.0


@pytest.mark.timeout(30)
def test_bs_round_robin_routes_recovery_control_through_ap_links():
    configure_ros_domain(4, 120, 139)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "mode:=ideal", "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "bs_min_turn_ms:=5.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_sionna_recovery_control_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    status_received = []
    command_received = []
    node.create_subscription(
        RecoveryStatus,
        "/racer_ap_recovery/status_uplink",
        lambda message: status_received.append(
            (message.drone_id, message.episode_id)
        ),
        qos,
    )
    node.create_subscription(
        RecoveryCommand,
        "/racer_sionna/rx/drone_0/recovery_command",
        lambda message: command_received.append(
            (message.drone_id, message.partner_id)
        ),
        qos,
    )
    status_publisher = node.create_publisher(
        RecoveryStatus,
        "/racer_sionna/tx/drone_0/recovery_status",
        qos,
    )
    command_publisher = node.create_publisher(
        RecoveryCommand,
        "/racer_ap_recovery/command_downlink",
        qos,
    )
    try:
        deadline = time.monotonic() + 10.0
        while (
            status_publisher.get_subscription_count() < 1
            or command_publisher.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert status_publisher.get_subscription_count() == 1
        assert command_publisher.get_subscription_count() == 1

        status = RecoveryStatus()
        status.drone_id = 1
        status.episode_id = 9
        status.phase = RecoveryStatus.PHASE_WAITING_FOR_AP
        status.request_repartition = True
        status_publisher.publish(status)

        command = RecoveryCommand()
        command.drone_id = 1
        command.episode_id = 9
        command.assignment_epoch = 3
        command.action = RecoveryCommand.ACTION_REALLOCATE
        command.partner_id = 2
        command_publisher.publish(command)

        deadline = time.monotonic() + 8.0
        while (
            not status_received or not command_received
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert status_received == [(1, 9)]
        assert command_received == [(1, 2)]
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
