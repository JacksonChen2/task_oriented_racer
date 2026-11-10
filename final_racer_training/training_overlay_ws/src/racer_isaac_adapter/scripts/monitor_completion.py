#!/usr/bin/env python3
"""Close Isaac after every unchanged RACER FSM reports FINISH."""

import os
from pathlib import Path
import re
import subprocess
import sys
import time


log_path = Path(sys.argv[1])
drone_count = int(sys.argv[2])
launch_pid = int(sys.argv[3])
finish_pattern = re.compile(
    r"racer_(?:original|recovery)_exploration_(\d+).*"
    r"(?:finish exploration|state: FINISH)"
)
finished = set()


def launch_is_alive() -> bool:
    try:
        os.kill(launch_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


with log_path.open("r", errors="replace") as stream:
    while launch_is_alive():
        line = stream.readline()
        if not line:
            time.sleep(0.25)
            continue
        match = finish_pattern.search(line)
        if match is None:
            continue
        finished.add(int(match.group(1)))
        if len(finished) < drone_count:
            continue
        # This monitor observes logs only.  The message closes Isaac after the
        # original FSMs have already made and logged their FINISH decisions.
        subprocess.run(
            [
                "timeout", "15", "ros2", "topic", "pub", "--once",
                "/racer_3d/mission_complete", "std_msgs/msg/String",
                "{data: 'true'}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        break
