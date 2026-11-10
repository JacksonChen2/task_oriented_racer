"""Detect dead nested workers and stalled training independently of readers."""

from pathlib import Path


def process_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] not in {"Z", "X"}
    except FileNotFoundError:
        return False


class RuntimeHealth:
    def __init__(self, started: float, startup_timeout: float = 180.0,
                 progress_timeout: float = 60.0):
        self.started = started
        self.startup_timeout = startup_timeout
        self.progress_timeout = progress_timeout
        self.progress = {}

    def check(self, blocks, now: float):
        status = blocks["status_racer"].snapshot_json()
        if status is not None:
            pid = int(status[1]["pid"])
            if not process_alive(pid):
                return f"communication_proxy_dead_pid_{pid}"
        action_version = blocks["action"].version
        previous_action = self.progress.get("action", (0, self.started))[0]
        actor_just_started = previous_action == 0 and action_version > 0
        for name in ("physical", "transition_ring", "action"):
            version = blocks[name].version
            previous, changed = self.progress.get(name, (0, self.started))
            if version != previous or (actor_just_started and name != "action"):
                # Physical and transition seed records can be published long
                # before model warmup completes.  Rebase their progress clocks
                # when the first action arrives so startup latency is not
                # mistaken for a post-action stall.
                changed = now
            self.progress[name] = (version, changed)
            if version == 0:
                if now - self.started > self.startup_timeout:
                    return f"startup_timeout_{name}"
            elif now - changed > self.progress_timeout:
                # Initial physical/state snapshots can precede model warmup.
                if blocks["action"].version == 0 and now - self.started <= self.startup_timeout:
                    continue
                return f"progress_timeout_{name}"
        return None
