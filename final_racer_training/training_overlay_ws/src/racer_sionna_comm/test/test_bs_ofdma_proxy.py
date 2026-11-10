"""Real proxy tests: concurrent grants, packet fragmentation, retries and clock freeze."""
import json
import os
import random
import re
import signal
import subprocess
import time

import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from racer_fidelity_msgs.msg import ChunkData
from racer_sionna_interfaces.msg import LinkQuality, LinkQualityArray


@pytest.mark.timeout(30)
@pytest.mark.parametrize("fail", [False, True])
def test_bs_ofdma_concurrent_uploads_and_retries(tmp_path, fail):
    os.environ["ROS_DOMAIN_ID"] = str(random.randint(70, 89))
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "proxy.log"
    command = [
        "ros2", "run", "racer_sionna_comm", "racer_sionna_communication_proxy",
        "--ros-args", "-p", "use_sim_time:=true", "-p", "mode:=sionna",
        "-p", "drone_count:=2", "-p", "network_topology:=bs_round_robin",
        "-p", "bs_channel_access_mode:=ofdma",
        "-p", "bs_ofdma_diagnostic_logging:=true",
        "-p", "bs_resource_blocks:=2", "-p", "resource_blocks:=35",
        "-p", "rl_bs_scheduler_enabled:=true",
        "-p", "base_latency_ms:=0.0", "-p", "jitter_ms:=0.0",
        "-p", "bs_max_retries:=1", "-p", "retry_backoff_ms:=0.0",
        "-p", "initial_assignment_perfect_delivery:=false",
        "-p", "queue_capacity_bytes:=1048576",
                "-p", f"rl_bs_action_path:={action_path}",
        "-p", f"rl_bs_state_path:={state_path}",
    ]
    if fail:
        # At QPSK, the two-PRB pool needs a longer admission window to admit
        # two representative chunks including the reliable retry budget.
        command += ["-p", "rl_bs_communication_slot_ms:=80.0",
                    "-p", "rl_bs_decision_period_ms:=80.0"]
    with output_path.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    rclpy.init()
    node = Node("bs_ofdma_proxy_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    clock = node.create_publisher(Clock, "/clock", 10)
    links = node.create_publisher(LinkQualityArray, "/racer_sionna/link_quality", qos)
    chunks = [node.create_publisher(ChunkData, f"/racer_sionna/tx/drone_{i}/chunk_data", qos)
              for i in range(2)]

    def tick(seconds):
        message = Clock()
        ns = round(seconds * 1e9)
        message.clock.sec, message.clock.nanosec = divmod(ns, 10**9)
        clock.publish(message)
        channel = LinkQualityArray()
        channel.stamp = message.clock
        for sender in range(2):
            link = LinkQuality()
            link.sender_id = sender
            link.receiver_id = 2
            link.stamp = message.clock
            link.valid_until.sec = 20
            # Available but below QPSK threshold: PER rounds to 1 at -80 dB.
            link.snr_db = -80.0 if fail else 40.0
            link.model = "deterministic_ofdma_test"
            channel.links.append(link)
        links.publish(channel)
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.01)

    def state():
        try:
            return json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    try:
        deadline = time.monotonic() + 8
        while (any(p.get_subscription_count() == 0 for p in chunks) or
               links.get_subscription_count() == 0 or clock.get_subscription_count() == 0):
            assert time.monotonic() < deadline, output_path.read_text()
            tick(1.0)
        for _ in range(5):
            tick(1.0)
        payloads = []
        for i, publisher in enumerate(chunks):
            chunk = ChunkData()
            chunk.from_drone_id = i + 1
            chunk.to_drone_id = 2 - i
            chunk.chunk_drone_id = i + 1
            chunk.idx = 1
            # Large packets require many PHY slots in the two-PRB test pool.
            chunk.voxel_adrs = list(range(8000))
            chunk.voxel_occ = [1] * 8000
            payloads.append(chunk)
            publisher.publish(chunk)
        deadline = time.monotonic() + 4
        sim_time = 1.0
        while state().get("map_summary", {}).get("local_known_chunks") != [1, 1]:
            assert time.monotonic() < deadline, output_path.read_text()
            for publisher, chunk in zip(chunks, payloads):
                publisher.publish(chunk)
            sim_time += 0.02
            tick(sim_time)
        action_path.write_text("1 2 0 0 1 1\n")
        # Freeze while both packets are still partially serialized: the two-PRB
        # pool cannot complete these payloads in one logical 20 ms window.
        frozen_time = sim_time
        deadline = time.monotonic() + 4
        while state().get("action_epoch") != 1:
            assert time.monotonic() < deadline, output_path.read_text()[-4000:]
            frozen_time += 0.02
            tick(frozen_time)
        for _ in range(6):
            tick(frozen_time)
        before = output_path.read_text()
        assert "RACER_BS_OFDMA_GRANT" in before
        for _ in range(8):
            tick(frozen_time)
        frozen = output_path.read_text()
        assert frozen.count("RACER_BS_OFDMA_GRANT") == before.count("RACER_BS_OFDMA_GRANT")
        assert state().get("map_summary", {}).get("bs_known_chunks") == 0
        for i in range(1, 100):
            tick(frozen_time + i * 0.02)
        # Statistics are emitted by a one-second wall timer. Keep /clock
        # frozen so its final snapshot includes every charged grant.
        time.sleep(1.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    output = output_path.read_text()
    assert process.returncode in (0, -signal.SIGINT), output[-4000:]
    stats = [json.loads(s) for s in re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", output)]
    assert stats, output[-4000:]
    final = stats[-1]
    assert final["bs_channel_access_mode"] == "ofdma"
    assert final["bs_ofdma_peak_parallel_links"] == 2
    assert final["bs_ofdma_max_allocated_prbs_per_slot"] <= 2
    assert final["bs_ofdma_allocated_prb_slots"] == (
        final["bs_uplink_prb_slots"] + final["bs_downlink_prb_slots"]
    )
    grants = re.findall(
        r"RACER_BS_OFDMA_GRANT slot=(\d+) time_s=\S+ uplink=(\d) sender=(\d+) "
        r"receiver=(\d+) first_prb=(\d+) prbs=(\d+) remaining_bits=([\d.]+) attempt=(\d+)", output)
    assert grants, output[-4000:]
    assert sum(int(g[5]) for g in grants) == final["bs_ofdma_allocated_prb_slots"]
    slots = {}
    for slot, direction, sender, receiver, start, count, remaining, attempt in grants:
        slots.setdefault(slot, []).append((int(direction), int(sender), int(receiver),
                                          int(start), int(count)))
    for allocations in slots.values():
        assert len({a[0] for a in allocations}) == 1
        occupied = set()
        for _, _, _, first, count in allocations:
            prbs = set(range(first, first + count))
            assert not occupied & prbs
            occupied |= prbs
        assert len(occupied) <= 2
    assert any(len(a) == 2 for a in slots.values())
    assert any(float(g[6]) > 0 for g in grants), "packet fragmentation was not exercised"
    if fail:
        assert final["retried_packets"] >= 2
        assert final["bs_incremental_chunks_received_uplink"] == 0
        assert any(int(g[7]) == 1 and float(g[6]) > 0 for g in grants)
    else:
        assert final["bs_incremental_chunks_received_uplink"] == 2
