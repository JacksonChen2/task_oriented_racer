#!/usr/bin/env python3
"""Cross-language coverage for the fixed-clock Fast RL State ABI."""

import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid

import pytest
import rclpy
from nav_msgs.msg import Odometry
from racer_fidelity_msgs.msg import ChunkData
from racer_sionna_interfaces.msg import LinkQuality, LinkQualityArray
from rosgraph_msgs.msg import Clock
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data


TRAINING_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TRAINING_ROOT / "agentic_crpo"))

from agentic_crpo.fast_state import (  # noqa: E402
    decode_fast_state,
    encode_physical_fast,
)
from agentic_crpo.shared_ipc import (  # noqa: E402
    SharedMemoryLayout,
    create_layout,
)


def _clock_message(seconds: float) -> Clock:
    message = Clock()
    whole = int(seconds)
    message.clock.sec = whole
    message.clock.nanosec = int(round((seconds - whole) * 1.0e9))
    return message


def _wait_records(ring, count: int, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    records = []
    while time.monotonic() < deadline:
        records = ring.read_after(0)
        if len(records) >= count:
            return records
        time.sleep(0.005)
    raise AssertionError(f"expected {count} Fast State records, got {len(records)}")


@pytest.mark.timeout(40)
def test_cpp_fast_state_is_step_aligned_and_holds_active_action():
    root = Path("/dev/shm") / f"final_racer_test_{uuid.uuid4().hex}"
    layout = SharedMemoryLayout.from_value(root)
    capacities = {name: 16 * 1024 for name in layout.JSON_BLOCKS}
    capacities.update(
        {"transition_ring": 16 * 1024, "task_metric_ring": 32 * 1024}
    )
    blocks = create_layout(
        layout,
        capacities=capacities,
        ring_slots={"transition_ring": 32, "task_metric_ring": 32},
    )
    domain_id = random.randint(190, 229)
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    process = subprocess.Popen(
        [
            "ros2",
            "run",
            "racer_sionna_comm",
            "racer_sionna_communication_proxy",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "mode:=ideal",
            "-p",
            "drone_count:=2",
            "-p",
            "network_topology:=bs_round_robin",
            "-p",
            "rl_bs_scheduler_enabled:=true",
            "-p",
            "rl_bs_synchronous_mode:=false",
            "-p",
            "rl_bs_communication_slot_ms:=20.0",
            "-p",
            "rl_bs_decision_period_ms:=100.0",
            "-p",
            "rl_llm_state_period_ms:=100.0",
            "-p",
            f"rl_shared_memory_root:={root}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    rclpy.init()
    node = Node("racer_shared_async_proxy_test")
    clock_publisher = node.create_publisher(Clock, "/clock", 10)
    odometry_publishers = [
        node.create_publisher(
            Odometry, f"/drone_{index}/odom", qos_profile_sensor_data
        )
        for index in range(2)
    ]
    output = ""
    try:
        discovery_deadline = time.monotonic() + 5.0
        while (
            any(
                publisher.get_subscription_count() < 1
                for publisher in odometry_publishers
            )
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
        assert all(
            publisher.get_subscription_count() >= 1
            for publisher in odometry_publishers
        )
        for _ in range(8):
            for index, publisher in enumerate(odometry_publishers):
                odometry = Odometry()
                odometry.pose.pose.position.x = float(index)
                odometry.pose.pose.position.z = 1.0
                odometry.pose.pose.orientation.w = 1.0
                publisher.publish(odometry)
            clock_publisher.publish(_clock_message(0.0))
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.01)

        initial_record = _wait_records(blocks["transition_ring"], 1)[0]
        initial = decode_fast_state(initial_record.payload, 2)
        assert initial_record.sim_step == 0
        assert initial_record.sim_time_s == pytest.approx(0.0)
        assert initial.values["step_id"] == 0
        assert initial.values["communication_slot_index"] == 0

        llm_deadline = time.monotonic() + 5.0
        llm_payload = None
        while time.monotonic() < llm_deadline:
            value = blocks["communication"].snapshot_json()
            if value is not None:
                _, llm_payload = value
                break
            time.sleep(0.005)
        assert llm_payload is not None
        assert "bs_map_summary" in llm_payload
        assert "uav_bs_missing_bytes" in llm_payload
        assert "bs_uav_missing_bytes" in llm_payload
        assert "position_sources" in llm_payload
        assert {
            "map_summary",
            "coverage",
            "coverage_delta",
            "information_version_gap",
            "bs_global_map_iou",
            "local_known_chunks",
        }.isdisjoint(llm_payload)

        # Reproduce a loaded proxy lagging behind Isaac: the physical block is
        # already terminal at 5 s while communication is still producing the
        # 0.1/0.2 s boundaries. The future terminal bit must not end RL early.
        blocks["physical_fast"].publish_bytes(
            encode_physical_fast(
                n_uavs=2,
                sim_step=500,
                sim_time_s=5.0,
                coverage=0.5,
                coverage_delta=0.0,
                terminated=False,
                truncated=True,
            ),
            sim_step=500,
            sim_time_s=5.0,
        )

        # N=2: relay 0->1, relay 1->0, upload 0, upload 1.
        action_payload = " ".join(
            (
                "1", "2", "0", str(initial.values["sequence"]),
                "0", "0", "7", str(time.time_ns()), "0", "0",
                "1", "0", "1", "0",
            )
        ) + "\n"
        blocks["action"].publish_bytes(
            action_payload.encode("ascii"), sim_step=0, sim_time_s=0.0,
            version=1,
        )

        for tick in range(1, 21):
            clock_publisher.publish(_clock_message(0.01 * tick))
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.01)

        records = _wait_records(blocks["transition_ring"], 3)
        decoded = [decode_fast_state(record.payload, 2) for record in records]
        assert [record.sim_step for record in records[:3]] == [0, 1, 2]
        assert [record.sim_time_s for record in records[:3]] == pytest.approx(
            [0.0, 0.1, 0.2]
        )
        assert not decoded[1].values["truncated"]
        assert not decoded[2].values["truncated"]
        executed = decoded[2]
        assert executed.values["communication_slot_index"] == 10
        assert executed.values["rl_decision_index"] == 2
        assert executed.values["action_held_slots"] == 5
        assert executed.values["action_version"] == 1
        assert executed.values["policy_version"] == 7
        assert executed.relay_action.tolist() == [[0, 1], [0, 0]]
        assert executed.upload_action.tolist() == [1, 0]
        acknowledgement = blocks["action_ack"].snapshot_json()
        assert acknowledgement is not None
        _, acknowledgement_payload = acknowledgement
        assert acknowledgement_payload["action_id"] == 1
        assert acknowledgement_payload["source_step_id"] == 0
        assert acknowledgement_payload["channel_snapshot_step_id"] == 0
        assert acknowledgement_payload["channel_version"] == int(
            initial.values["sequence"]
        )

        # A delayed executor callback must not kill the proxy or skip the
        # canonical 100 ms step. Recovery emits at most one boundary per
        # callback, allowing other callback groups to run between snapshots.
        clock_publisher.publish(_clock_message(0.35))
        rclpy.spin_once(node, timeout_sec=0.02)
        recovered = _wait_records(blocks["transition_ring"], 4)
        assert [record.sim_step for record in recovered[:4]] == [0, 1, 2, 3]
        assert process.poll() is None

        clock_publisher.publish(_clock_message(0.40))
        rclpy.spin_once(node, timeout_sec=0.02)
        recovered = _wait_records(blocks["transition_ring"], 5)
        assert [record.sim_step for record in recovered[:5]] == [0, 1, 2, 3, 4]
        assert [record.sim_time_s for record in recovered[:5]] == pytest.approx(
            [0.0, 0.1, 0.2, 0.3, 0.4]
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
        for block in blocks.values():
            block.close()
        shutil.rmtree(root, ignore_errors=True)
    assert process.returncode in (0, -signal.SIGINT), output


@pytest.mark.timeout(40)
def test_queued_rl_packet_uses_the_action_bound_channel_snapshot():
    root = Path("/dev/shm") / f"final_racer_test_{uuid.uuid4().hex}"
    layout = SharedMemoryLayout.from_value(root)
    capacities = {name: 16 * 1024 for name in layout.JSON_BLOCKS}
    capacities.update(
        {"transition_ring": 16 * 1024, "task_metric_ring": 32 * 1024}
    )
    blocks = create_layout(
        layout,
        capacities=capacities,
        ring_slots={"transition_ring": 32, "task_metric_ring": 32},
    )
    domain_id = random.randint(150, 189)
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_sionna_comm",
            "racer_sionna_communication_proxy", "--ros-args",
            "-p", "use_sim_time:=true",
            "-p", "mode:=sionna",
            "-p", "drone_count:=2",
            "-p", "network_topology:=bs_round_robin",
            "-p", "rl_bs_scheduler_enabled:=true",
            "-p", "rl_bs_synchronous_mode:=false",
            "-p", "rl_bs_communication_slot_ms:=20.0",
            "-p", "rl_bs_decision_period_ms:=100.0",
            "-p", "base_latency_ms:=0.0",
            "-p", "jitter_ms:=0.0",
            "-p", "bs_max_retries:=0",
            "-p", f"rl_shared_memory_root:={root}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    rclpy.init()
    node = Node("racer_action_bound_channel_test")
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
    clock_publisher = node.create_publisher(Clock, "/clock", 10)
    odometry_publishers = [
        node.create_publisher(
            Odometry, f"/drone_{index}/odom", qos_profile_sensor_data
        )
        for index in range(2)
    ]
    chunk_publisher = node.create_publisher(
        ChunkData, "/racer_sionna/tx/drone_0/chunk_data", qos
    )
    link_publisher = node.create_publisher(
        LinkQualityArray, "/racer_sionna/link_quality", qos
    )

    def publish_uplink(*, available: bool) -> None:
        links = LinkQualityArray()
        link = LinkQuality()
        link.sender_id = 0
        link.receiver_id = 2
        link.snr_db = 100.0 if available else -120.0
        link.model = "deterministic_test_path" if available else "unavailable"
        link.valid_until.sec = 10
        links.links.append(link)
        link_publisher.publish(links)

    output = ""
    try:
        discovery_deadline = time.monotonic() + 7.0
        while (
            chunk_publisher.get_subscription_count() < 1
            or link_publisher.get_subscription_count() < 1
            or any(
                publisher.get_subscription_count() < 1
                for publisher in odometry_publishers
            )
        ) and time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert chunk_publisher.get_subscription_count() == 1
        assert link_publisher.get_subscription_count() == 1

        # Load UAV 0's inventory and a usable uplink before the t=0 Fast State
        # can be emitted. Odometry is deliberately withheld until both inputs
        # have been processed.
        chunk = ChunkData()
        chunk.from_drone_id = 1
        chunk.to_drone_id = 2
        chunk.chunk_drone_id = 1
        chunk.idx = 1
        chunk.voxel_adrs = [1]
        chunk.voxel_occ = [1]
        for _ in range(8):
            # Re-publish the same immutable chunk until the subscriber callback
            # has certainly run; inventory dedup keeps this idempotent.
            chunk_publisher.publish(chunk)
            publish_uplink(available=True)
            clock_publisher.publish(_clock_message(0.0))
            rclpy.spin_once(node, timeout_sec=0.02)

        for _ in range(8):
            for index, publisher in enumerate(odometry_publishers):
                odometry = Odometry()
                odometry.pose.pose.position.x = float(index)
                odometry.pose.pose.position.z = 1.0
                odometry.pose.pose.orientation.w = 1.0
                publisher.publish(odometry)
            publish_uplink(available=True)
            clock_publisher.publish(_clock_message(0.0))
            rclpy.spin_once(node, timeout_sec=0.02)

        initial_record = _wait_records(blocks["transition_ring"], 1)[0]
        initial = decode_fast_state(initial_record.payload, 2)
        assert initial.bs_missing_bytes[0] > 0
        assert initial.channel_snr_db[0, 2] > 90.0

        # Upload-only action u[0]=1, bound to the usable t=0 snapshot.
        action_payload = " ".join(
            (
                "1", "2", "0", str(initial.values["sequence"]),
                "0", "0", "5", str(time.time_ns()), "0", "0",
                "0", "0", "1", "0",
            )
        ) + "\n"
        blocks["action"].publish_bytes(
            action_payload.encode("ascii"), sim_step=0, sim_time_s=0.0,
            version=1,
        )

        # Sionna may now update the live table to no-link. The queued action
        # must still use the high-SNR snapshot it consumed at inference time.
        for _ in range(8):
            publish_uplink(available=False)
            clock_publisher.publish(_clock_message(0.0))
            rclpy.spin_once(node, timeout_sec=0.02)
        for tick in range(1, 31):
            publish_uplink(available=False)
            clock_publisher.publish(_clock_message(0.01 * tick))
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.004)

        records = _wait_records(blocks["transition_ring"], 3)
        decoded = [decode_fast_state(record.payload, 2) for record in records]
        assert decoded[1].channel_snr_db[0, 2] == pytest.approx(-120.0)
        assert decoded[2].bs_missing_bytes[0] == 0

        acknowledgement = blocks["action_ack"].snapshot_json()
        assert acknowledgement is not None
        _, acknowledgement_payload = acknowledgement
        assert acknowledgement_payload["action_id"] == 1
        assert acknowledgement_payload["source_step_id"] == 0
        assert acknowledgement_payload["channel_version"] == int(
            initial.values["sequence"]
        )

        # Exercise the opposite transition as well: bind action 2 to a
        # no-link state, then make the live table usable before execution.
        # A newly generated chunk must still fail its upload.
        records = _wait_records(blocks["transition_ring"], 4)
        unavailable_state = decode_fast_state(records[3].payload, 2)
        assert unavailable_state.channel_snr_db[0, 2] == pytest.approx(-120.0)
        action_payload = " ".join(
            (
                "2", "2", "0", str(unavailable_state.values["sequence"]),
                "0", "0", "5", str(time.time_ns()), "0.3", "3",
                "0", "0", "1", "0",
            )
        ) + "\n"
        blocks["action"].publish_bytes(
            action_payload.encode("ascii"), sim_step=15, sim_time_s=0.3,
            version=2,
        )
        for tick in range(31, 41):
            publish_uplink(available=True)
            clock_publisher.publish(_clock_message(0.01 * tick))
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.004)

        second_acknowledgement = blocks["action_ack"].snapshot_json()
        assert second_acknowledgement is not None
        _, second_acknowledgement_payload = second_acknowledgement
        assert second_acknowledgement_payload["action_id"] == 2
        assert second_acknowledgement_payload["source_step_id"] == 3
        assert second_acknowledgement_payload["channel_version"] == int(
            unavailable_state.values["sequence"]
        )

        second_chunk = ChunkData()
        second_chunk.from_drone_id = 1
        second_chunk.to_drone_id = 2
        second_chunk.chunk_drone_id = 1
        second_chunk.idx = 2
        second_chunk.voxel_adrs = [2]
        second_chunk.voxel_occ = [1]
        chunk_publisher.publish(second_chunk)
        for _ in range(5):
            publish_uplink(available=True)
            clock_publisher.publish(_clock_message(0.4))
            rclpy.spin_once(node, timeout_sec=0.02)
        for tick in range(41, 66):
            publish_uplink(available=True)
            clock_publisher.publish(_clock_message(0.01 * tick))
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.004)
        final_records = _wait_records(blocks["transition_ring"], 6)
        final_state = decode_fast_state(final_records[5].payload, 2)
        assert final_state.channel_snr_db[0, 2] > 90.0
        assert final_state.bs_missing_bytes[0] > 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
        for block in blocks.values():
            block.close()
        shutil.rmtree(root, ignore_errors=True)
    assert process.returncode in (0, -signal.SIGINT), output
