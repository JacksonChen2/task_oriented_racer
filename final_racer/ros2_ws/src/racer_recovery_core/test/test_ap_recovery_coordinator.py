#!/usr/bin/env python3

import os
import random
import signal
import subprocess
import time

import pytest
import rclpy
from racer_recovery_core.msg import RecoveryCommand, RecoveryStatus
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


@pytest.mark.timeout(25)
def test_ap_coordinator_selects_nearest_healthy_partner():
    os.environ["ROS_DOMAIN_ID"] = str(random.randint(90, 119))
    process = subprocess.Popen(
        [
            "ros2", "run", "racer_recovery_core",
            "racer_ap_recovery_coordinator", "--ros-args",
            "-p", "drone_count:=3", "-p", "status_timeout_s:=5.0",
            "-p", "command_cooldown_s:=0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    rclpy.init()
    node = Node("racer_ap_recovery_coordinator_test")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    commands = []
    node.create_subscription(
        RecoveryCommand,
        "/racer_ap_recovery/command_downlink",
        lambda message: commands.append(message),
        qos,
    )
    publisher = node.create_publisher(
        RecoveryStatus, "/racer_ap_recovery/status_uplink", qos
    )
    try:
        deadline = time.monotonic() + 8.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert publisher.get_subscription_count() == 1

        for drone_id, x in ((2, 2.0), (3, 8.0)):
            status = RecoveryStatus()
            status.drone_id = drone_id
            status.phase = RecoveryStatus.PHASE_NORMAL
            status.position.x = x
            status.grid_ids = [10 + drone_id]
            publisher.publish(status)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)

        failed = RecoveryStatus()
        failed.drone_id = 1
        failed.episode_id = 7
        failed.phase = RecoveryStatus.PHASE_WAITING_FOR_AP
        failed.position.x = 0.0
        failed.grid_ids = [4, 5]
        failed.failed_grid_id = 4
        failed.request_repartition = True
        publisher.publish(failed)

        deadline = time.monotonic() + 5.0
        while not commands and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert len(commands) == 1
        command = commands[0]
        assert command.drone_id == 1
        assert command.episode_id == 7
        assert command.action == RecoveryCommand.ACTION_REALLOCATE
        assert command.partner_id == 2
        assert command.blocked_grid_id == 4
        assert command.blocked_until_s > 0.0
        assert command.assignment_epoch > 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
