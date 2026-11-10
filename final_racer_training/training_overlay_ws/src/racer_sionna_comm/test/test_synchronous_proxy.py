#!/usr/bin/env python3
"""ROS integration coverage for the 20 ms / 100 ms synchronous scheduler."""

import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import pytest
import rclpy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def _clock_message(seconds: float) -> Clock:
    message = Clock()
    whole = int(seconds)
    message.clock.sec = whole
    message.clock.nanosec = int(round((seconds - whole) * 1.0e9))
    return message


def _wait_state(path: Path, decision: int, timeout_s: float = 4.0):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            last = json.loads(path.read_text(encoding="utf-8"))
            if int(last.get("rl_decision_index", -1)) == decision:
                return last
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.005)
    raise AssertionError(f"no synchronous state decision={decision}; last={last}")


def _write_action(path: Path, epoch: int, decision: int, bits: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        f"{epoch} 2 {decision} {bits}\n", encoding="ascii"
    )
    temporary.replace(path)


@pytest.mark.timeout(30)
def test_action_is_held_for_five_slots_and_state_freezes_without_clock(tmp_path):
    domain_id = random.randint(190, 229)
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    action_path = tmp_path / "action.txt"
    state_path = tmp_path / "state.json"
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
            "rl_bs_synchronous_mode:=true",
            "-p",
            "rl_bs_communication_slot_ms:=20.0",
            "-p",
            "rl_bs_decision_period_ms:=100.0",
            "-p",
            f"rl_bs_action_path:={action_path}",
            "-p",
            f"rl_bs_state_path:={state_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    rclpy.init()
    node = Node("racer_synchronous_proxy_test")
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
            any(publisher.get_subscription_count() < 1 for publisher in odometry_publishers)
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

        initial = _wait_state(state_path, 0)
        assert initial["communication_slot_index"] == 0
        assert initial["action_held_slots"] == 0

        # Two relay bits followed by two upload bits.
        _write_action(action_path, 1, 0, "1 0 1 1")
        for tick in range(1, 11):
            clock_publisher.publish(_clock_message(0.01 * tick))
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.008)
        first = _wait_state(state_path, 1)
        assert first["sim_time_s"] == pytest.approx(0.1)
        assert first["communication_slot_index"] == 5
        assert first["rl_decision_index"] == 1
        assert first["action_decision_index"] == 0
        assert first["action_held_slots"] == 5
        assert first["slots_per_decision"] == 5

        frozen_bytes = state_path.read_bytes()
        frozen_mtime = state_path.stat().st_mtime_ns
        freeze_deadline = time.monotonic() + 0.35
        while time.monotonic() < freeze_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert state_path.read_bytes() == frozen_bytes
        assert state_path.stat().st_mtime_ns == frozen_mtime

        _write_action(action_path, 2, 1, "0 1 0 0")
        for tick in range(11, 21):
            clock_publisher.publish(_clock_message(0.01 * tick))
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.008)
        second = _wait_state(state_path, 2)
        assert second["sim_time_s"] == pytest.approx(0.2)
        assert second["communication_slot_index"] == 10
        assert second["action_decision_index"] == 1
        assert second["action_held_slots"] == 5
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
    assert process.returncode in (0, -signal.SIGINT), output
