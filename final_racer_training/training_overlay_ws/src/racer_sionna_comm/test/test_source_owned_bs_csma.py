"""Real ROS integration tests for owner filtering across BS access modes."""
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
@pytest.mark.parametrize("access,collide", [
    ("csma", False), ("csma", True), ("scheduled", False), ("ofdma", False),
])
def test_source_owned_bs_csma_relays_nearest_chunk_first(tmp_path, access, collide):
    """Verify source filtering and distance order, or forced collision retries."""
    configure_ros_domain(27, 40, 49)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "communication_state.json"
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            *(["-p", "uav_csma_cw_min:=0", "-p", "uav_csma_cw_max:=0",
               "-p", "bs_max_retries:=1"] if collide else []),
            "-p", f"bs_channel_access_mode:={access}",
            "-p", "bs_downlink_source_owned_chunks_only:=true",
            "-p", "mode:=sionna", "-p", "drone_count:=3",
            "-p", "network_topology:=bs_round_robin",
            "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
            "-p", "max_retries:=0", "-p", f"bs_max_retries:={1 if collide else 0}",
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
    odometry = node.create_publisher(Odometry, "/drone_2/odom", qos)
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
        pairs = (((0, 3), (1, 3), (3, 2)) if collide else ((1, 3), (3, 2)))
        if access == "ofdma":
            pairs += ((3, 0),)
        for sender, receiver in pairs:
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

        if collide:
            for sender in (0, 1):
                chunk = ChunkData()
                chunk.from_drone_id = sender + 1
                chunk.to_drone_id = 3
                chunk.chunk_drone_id = sender + 1
                chunk.idx = 1
                chunk.voxel_adrs = [11]
                chunk.voxel_occ = [1]
                chunk_publishers[sender].publish(chunk)
            wait_for(lambda value: value.get("map_summary", {}).get("local_known_chunks") == [1, 1, 0])
            # Zero contention window forces simultaneous contenders. Neither
            # may succeed, including the reliable retransmission.
            write_action(1, uploads={0, 1})
            for _ in range(25):
                publish_links()
                rclpy.spin_once(node, timeout_sec=0.02)
        else:
            # Source 1 owns two chunks; the other owner is closer but must be
            # excluded. Index order disagrees with distance order.
            for owner, idx, address in ((1, 101, 800), (1, 201, 11), (2, 301, 11)):
                chunk = ChunkData()
                chunk.from_drone_id = 2
                chunk.to_drone_id = 3
                chunk.chunk_drone_id = owner
                chunk.idx = idx
                chunk.voxel_adrs = [address]
                chunk.voxel_occ = [1]
                chunk_publishers[1].publish(chunk)
            position = Odometry()
            position.pose.pose.position.x = -10.95
            position.pose.pose.position.y = -17.95
            position.pose.pose.position.z = 1.15
            position.pose.pose.orientation.w = 1.0
            for _ in range(5):
                odometry.publish(position)
                rclpy.spin_once(node, timeout_sec=0.02)
            wait_for(lambda value: value.get("map_summary", {}).get("local_known_chunks") == [0, 3, 0])
            write_action(1, uploads={1})
            wait_for(lambda value: value.get("map_summary", {}).get("bs_known_chunks") == 3)
            # B[0,2] may only downlink owner 1, despite owner 2 already cached.
            write_action(2, relay_pairs={(0, 2)})
            wait_for(lambda value: value.get("action_epoch") == 2 and
                     value.get("map_summary", {}).get("local_known_chunks") == [0, 3, 2])
            assert received == [(4, 3, 1, 201), (4, 3, 1, 101)]
            if access == "ofdma":
                # One BS transmitter serves two receivers on disjoint PRBs.
                # Both lack owner 2's chunk, while the original source keeps it.
                write_action(3, relay_pairs={(1, 0), (1, 2)})
                wait_for(lambda value: value.get("action_epoch") == 3 and
                         value.get("map_summary", {}).get("local_known_chunks") == [1, 3, 3])
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
    if collide:
        assert statistics["bs_csma_collision_events"] >= 2
        assert statistics["bs_csma_collided_packets"] >= 4
        assert statistics["retried_packets"] >= 2
        assert statistics["bs_incremental_chunks_received_uplink"] == 0
    else:
        assert statistics["bs_incremental_chunks_received_uplink"] == 3
        assert statistics["bs_known_map_chunks"] == 3
        assert statistics["bs_missing_chunks_delivered_downlink"] == (4 if access == "ofdma" else 2)
        assert statistics["bs_channel_access_mode"] == access
        assert statistics["bs_downlink_source_owned_chunks_only"] is True
        if access == "csma":
            assert statistics["bs_csma_successful_contentions"] >= 5
            assert statistics["bs_csma_idle_slots"] > 0
        elif access == "ofdma":
            assert statistics["bs_ofdma_peak_parallel_links"] == 2
            assert statistics["bs_ofdma_allocated_prb_slots"] == (
                statistics["bs_uplink_prb_slots"] + statistics["bs_downlink_prb_slots"]
            )
            assert statistics["bs_ofdma_max_allocated_prbs_per_slot"] <= 33
    cache_checks = re.findall(
        r"RACER_INCREMENTAL_CACHE_VALIDATION mismatches=(\d+)", output
    )
    assert cache_checks and set(cache_checks) == {"0"}

