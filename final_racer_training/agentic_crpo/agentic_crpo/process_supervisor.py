"""Lightweight supervisor for the four independent runtime processes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import uuid
from typing import IO, Any

from .runtime_health import RuntimeHealth
from .result_assembly import assemble
from .config import load_config
from .shared_ipc import SharedMemoryLayout, create_layout


CAPACITIES = {
    "physical": 256 * 1024,
    "physical_fast": 16 * 1024,
    "communication": 1024 * 1024,
    "state_event": 4 * 1024,
    "guidance": 128 * 1024,
    "guidance_fast": 16 * 1024,
    "action": 32 * 1024,
    "action_ack": 32 * 1024,
    "shutdown": 4 * 1024,
    "status_isaac": 4 * 1024,
    "status_racer": 4 * 1024,
    "status_llm": 4 * 1024,
    "status_rl": 4 * 1024,
    # Fast state is a fixed binary ABI. The independent metric ring may carry
    # low-rate JSON because it is never decoded on the actor inference path.
    "transition_ring": 16 * 1024,
    "task_metric_ring": 32 * 1024,
}

RING_SLOTS = {"transition_ring": 4096, "task_metric_ring": 4096}


@dataclass
class Child:
    name: str
    process: subprocess.Popen[bytes]
    log: IO[bytes]
    pidfd: int | None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _signal_group(child: Child, sig: signal.Signals) -> None:
    # The launch-group leader can exit before all ROS descendants have handled
    # SIGTERM.  Popen.poll() only reports on that leader, so conditioning this
    # call on it being alive can leave re-parented nodes running into the next
    # episode.  The process group remains addressable while any descendant is
    # alive; killpg raises ProcessLookupError once the complete group is gone.
    try:
        os.killpg(child.process.pid, sig)
    except ProcessLookupError:
        pass


def _is_clean_boundary_exit(name: str, status: int | None) -> bool:
    """Return whether a process can normally finish a finite episode.

    Isaac owns the simulation horizon, while RL owns consumption of the
    terminal transition.  Once both clocks reach the final boundary, either
    process may be reaped first by the supervisor.
    """

    return status == 0 and name in {"isaac", "rl"}


def _spawn(
    name: str,
    command: list[str],
    output_dir: Path,
    environment: dict[str, str],
) -> Child:
    log = (output_dir / f"process_{name}.log").open("wb", buffering=0)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    # pidfd_open is unavailable in some of the pinned Python interpreters used
    # by the training environment.  Keep pidfds as the efficient Linux path,
    # but fall back to Popen.poll() rather than failing after the child has
    # already been spawned (which would also orphan its ROS process group).
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd: int | None = None
    if pidfd_open is not None:
        try:
            pidfd = pidfd_open(process.pid)
        except OSError:
            pidfd = None
    return Child(name, process, log, pidfd)


def run(args: argparse.Namespace) -> int:
    config = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not config.is_file() or not runner.is_file():
        raise FileNotFoundError("config or split-process runner is missing")

    root = (
        Path(args.shared_memory_root).expanduser().resolve()
        if args.shared_memory_root
        else Path("/dev/shm")
        / f"final_racer_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    layout = SharedMemoryLayout.from_value(root)
    blocks = create_layout(
        layout, capacities=CAPACITIES, ring_slots=RING_SLOTS
    )
    blocks["shutdown"].publish_json(
        {"shutdown": False, "supervisor_pid": os.getpid()}
    )

    environment = os.environ.copy()
    loaded_config = load_config(str(config))
    llm_period_ms = 20.0 * float(
        loaded_config["environment"].get("high_level_interval", 250)
    )
    package_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = package_root + (
        ":" + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RACER_SHARED_MEMORY_ROOT": str(layout.root),
            "RACER_RL_SYNC_ENABLED": "false",
            "RACER_RL_BS_COMMUNICATION_SLOT_MS": "20.0",
            "RACER_RL_BS_DECISION_PERIOD_MS": "100.0",
            "RACER_RL_LLM_STATE_PERIOD_MS": format(llm_period_ms, ".17g"),
            "RACER_SINGLE_GPU_PAUSE_REQUEST_PATH": "",
            "RACER_SINGLE_GPU_PAUSE_ACK_PATH": "",
            "RACER_RESULT_DIR": str(output_dir / "sim"),
        }
    )
    Path(environment["RACER_RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
    common_train = [
        sys.executable,
        "-m",
        "agentic_crpo.train_crpo",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir / "training"),
        "--total-timesteps",
        str(args.total_timesteps),
    ]
    if args.perfect_reference:
        common_train += ["--perfect-reference", args.perfect_reference]
    for option, value in (
        ("--learning-rate", args.learning_rate),
        ("--n-steps", args.n_steps),
        ("--batch-size", args.batch_size),
        ("--n-epochs", args.n_epochs),
        ("--activation-fn", args.activation_fn),
        ("--gamma-task", args.gamma_task),
    ):
        if value is not None:
            common_train += [option, str(value)]
    if args.resume:
        common_train += ["--resume", str(Path(args.resume).resolve())]

    commands = {
        "racer": [str(runner)],
        "isaac": [str(runner)],
        "llm": [
            sys.executable,
            "-m",
            "agentic_crpo.llm_worker",
            "--config",
            str(config),
            "--shared-memory-root",
            str(layout.root),
        ],
        "rl": common_train,
    }
    children: dict[str, Child] = {}
    selector = selectors.DefaultSelector()
    wake_read, wake_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    selector.register(wake_read, selectors.EVENT_READ, "signal")
    interrupted = False

    def on_signal(_number: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        try:
            os.write(wake_write, b"x")
        except BlockingIOError:
            pass

    old_handlers = {
        sig: signal.signal(sig, on_signal)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    reason = "unknown"
    first_name = ""
    first_status: int | None = None
    final_boundary_sync: dict[str, Any] = {
        "attempted": False,
        "completed": False,
    }
    started_ns = time.time_ns()
    try:
        for name in ("racer", "isaac", "llm", "rl"):
            child_environment = environment.copy()
            if name in {"racer", "isaac"}:
                child_environment["RACER_PROCESS_ROLE"] = name
            child = _spawn(
                name,
                commands[name],
                output_dir,
                child_environment,
            )
            children[name] = child
            if child.pidfd is not None:
                selector.register(child.pidfd, selectors.EVENT_READ, name)
        _atomic_json(
            output_dir / "process_manifest.json",
            {
                "supervisor_pid": os.getpid(),
                "shared_memory_root": str(layout.root),
                "started_wall_time_ns": started_ns,
                "processes": {
                    name: {
                        "pid": child.process.pid,
                        "command": commands[name],
                        "process_group": child.process.pid,
                    }
                    for name, child in children.items()
                },
            },
        )
        # With pidfds this normally wakes immediately on child exit.  The
        # bounded timeout also supports older Python runtimes without pidfds
        # and keeps signal handling responsive in both cases.
        health = RuntimeHealth(time.monotonic())
        while reason == "unknown":
            events = selector.select(timeout=0.25)
            if any(key.data == "signal" for key, _ in events) or interrupted:
                reason = "supervisor_signal"
                break
            for name, child in children.items():
                status = child.process.poll()
                if status is not None:
                    first_name = name
                    first_status = status
                    reason = f"{name}_exit_{status}"
                    break

            if reason == "unknown":
                failure = health.check(blocks, time.monotonic())
                if failure:
                    first_name, first_status = "health", 1
                    reason = failure
                    print(f"RACER_RUNTIME_FAILURE {failure}", flush=True)

        # Isaac force-publishes its terminal physical state before exiting,
        # and RL may consume that terminal boundary and exit before Isaac is
        # reaped. Give the independently running proxy a bounded chance to
        # commit the matching final fixed-clock boundary in either ordering.
        if _is_clean_boundary_exit(first_name, first_status):
            final_boundary_sync["attempted"] = True
            physical_value = blocks["physical"].snapshot_json()
            if physical_value is not None:
                physical_time = float(
                    physical_value[1].get(
                        "sim_time_s", physical_value[0].sim_time_s
                    )
                )
                decision_interval_s = (
                    float(environment["RACER_RL_BS_DECISION_PERIOD_MS"])
                    / 1000.0
                )
                target_time = (
                    round(physical_time / decision_interval_s)
                    * decision_interval_s
                )
                final_boundary_sync.update(
                    {
                        "physical_sim_time_s": physical_time,
                        "target_sim_time_s": target_time,
                    }
                )
                # The transition boundary is light; the map metric is allowed
                # to trail it. Keep RACER alive long enough for the final cost
                # record so a partial rollout never updates with pending cost.
                sync_deadline = time.monotonic() + min(
                    30.0, float(args.shutdown_grace_s)
                )
                transition_ready = False
                metric_ready = False
                while time.monotonic() < sync_deadline:
                    for name, field in (
                        ("transition_ring", "communication_sim_time_s"),
                        ("task_metric_ring", "task_metric_sim_time_s"),
                    ):
                        ring = blocks[name]
                        ring_version = ring.version
                        records = (
                            [] if ring_version == 0
                            else ring.read_after(
                                ring_version - 1, max_items=1
                            )
                        )
                        if not records:
                            continue
                        value = float(records[0].sim_time_s)
                        final_boundary_sync[field] = value
                        if name == "transition_ring":
                            transition_ready = value + 1.0e-6 >= target_time
                        else:
                            metric_ready = value + 1.0e-6 >= target_time
                    if transition_ready and metric_ready:
                        final_boundary_sync["completed"] = True
                        break
                    if children["racer"].process.poll() is not None:
                        break
                    time.sleep(0.005)

        blocks["shutdown"].publish_json(
            {
                "shutdown": True,
                "reason": reason,
                "supervisor_pid": os.getpid(),
                "wall_time_ns": time.time_ns(),
            }
        )
        for block in blocks.values():
            block.poke()

        # RACER has no algorithm-facing shutdown reader; stop its launch group
        # after the shared shutdown has woken model waiters. LLM/RL get a grace
        # period to finish their current inference/update and flush artifacts.
        if first_name != "racer" and "racer" in children:
            _signal_group(children["racer"], signal.SIGTERM)
        if first_name not in {"isaac", ""} and "isaac" in children:
            _signal_group(children["isaac"], signal.SIGTERM)

        deadline = time.monotonic() + float(args.shutdown_grace_s)
        for name in ("rl", "llm", "isaac", "racer"):
            child = children[name]
            if child.process.poll() is not None:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                child.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                break
        for child in children.values():
            _signal_group(child, signal.SIGTERM)
        term_deadline = time.monotonic() + 5.0
        for child in children.values():
            if child.process.poll() is None:
                try:
                    child.process.wait(
                        timeout=max(0.0, term_deadline - time.monotonic())
                    )
                except subprocess.TimeoutExpired:
                    pass
        for child in children.values():
            _signal_group(child, signal.SIGKILL)
        for child in children.values():
            child.process.wait()

        result_path = assemble(output_dir, environment)
        final = {
            "reason": reason,
            "first_exit_process": first_name,
            "first_exit_status": first_status,
            "interrupted": interrupted,
            "started_wall_time_ns": started_ns,
            "completed_wall_time_ns": time.time_ns(),
            "exit_statuses": {
                name: child.process.returncode
                for name, child in children.items()
            },
            "final_versions": {
                name: block.version for name, block in blocks.items()
            },
            "assembled_result": (
                None if result_path is None else str(result_path)
            ),
            "final_boundary_sync": final_boundary_sync,
        }
        _atomic_json(output_dir / "supervisor_result.json", final)
        if interrupted:
            return 130
        # A normal episode is ended by Isaac or by RL after consuming the
        # terminal transition/reaching its explicit target. Their independent
        # process teardown order must not change the campaign result.
        return (
            0
            if _is_clean_boundary_exit(first_name, first_status)
            and children["rl"].process.returncode == 0
            and result_path is not None
            else 1
        )
    finally:
        running = [
            child for child in children.values()
            if child.process.poll() is None
        ]
        if running:
            try:
                blocks["shutdown"].publish_json(
                    {
                        "shutdown": True,
                        "reason": "supervisor_finalizer",
                        "supervisor_pid": os.getpid(),
                    }
                )
                for block in blocks.values():
                    block.poke()
            except Exception:
                pass
            for child in running:
                _signal_group(child, signal.SIGTERM)
            for child in running:
                try:
                    child.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    _signal_group(child, signal.SIGKILL)
                    child.process.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        selector.close()
        os.close(wake_read)
        os.close(wake_write)
        for child in children.values():
            child.log.close()
            if child.pidfd is not None:
                os.close(child.pidfd)
        for block in blocks.values():
            block.close()
        if layout.root.exists():
            for path in layout.root.iterdir():
                path.unlink(missing_ok=True)
            layout.root.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--runner",
        default=str(
            Path(__file__).resolve().parents[2]
            / "runtime"
            / "run_final_racer_training.sh"
        ),
    )
    parser.add_argument("--shared-memory-root")
    parser.add_argument("--shutdown-grace-s", type=float, default=120.0)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--perfect-reference")
    parser.add_argument("--resume")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--activation-fn")
    parser.add_argument("--gamma-task", type=float)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
