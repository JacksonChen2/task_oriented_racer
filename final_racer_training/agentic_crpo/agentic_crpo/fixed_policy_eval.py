"""Run one real event-driven mission with a non-learning sanity policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import queue
import shutil
import threading
from typing import Any

import numpy as np

from .action_mapping import ActionMapping
from .backend import MissionEndedError
from .config import load_config, save_resolved_config
from .factory import make_env
from .shared_ipc import open_blocks


POLICY_MODES = ("zero", "one", "bernoulli_05", "round_robin_row")


def fixed_action(
    mode: str,
    action_dim: int,
    rng: np.random.Generator,
    decision_index: int = 0,
) -> np.ndarray:
    """Return one action without consulting observations or learned weights."""

    if mode == "zero":
        return np.zeros(action_dim, dtype=np.int8)
    if mode == "one":
        return np.ones(action_dim, dtype=np.int8)
    if mode == "bernoulli_05":
        return rng.binomial(1, 0.5, size=action_dim).astype(np.int8)
    if mode == "round_robin_row":
        n_uavs = int(round(action_dim**0.5))
        if n_uavs * n_uavs != action_dim:
            raise ValueError("round_robin_row requires an N^2 action space")
        sender = int(decision_index) % n_uavs
        matrix = np.zeros((n_uavs + 1, n_uavs), dtype=np.int8)
        matrix[sender, :] = 1
        matrix[sender, sender] = 0
        # The compressed N^2 action represents u_ii as the sender's upload
        # bit in the final matrix row. Together with the N-1 relay bits this
        # realizes u_ij=1 for every j during sender i's configured turn.
        matrix[n_uavs, sender] = 1
        return ActionMapping(n_uavs).encode(matrix)
    raise ValueError(f"unknown fixed policy: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICY_MODES, required=True)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--perfect-reference", type=Path)
    # Accepted for compatibility with the existing process supervisor. They
    # are deliberately ignored: this entry point creates no PPO model.
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--activation-fn")
    parser.add_argument("--gamma-task", type=float)
    parser.add_argument("--resume")
    return parser.parse_args()


def _ack_payload(block: Any) -> dict[str, Any]:
    snapshot = block.snapshot_json()
    return {} if snapshot is None else dict(snapshot[1])


def _curve_points(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not rows:
        return []
    points: list[dict[str, float]] = []
    for target in range(0, 301, 30):
        row = min(rows, key=lambda item: abs(float(item["sim_time_s"]) - target))
        points.append(
            {
                "target_time_s": float(target),
                "sim_time_s": float(row["sim_time_s"]),
                "reward": float(row["reward"]),
                "reward_cumulative": float(row["reward_cumulative"]),
                "L_task": float(row["L_task"]),
                "L_task_time_weighted": float(row["L_task_time_weighted"]),
                "joint_coverage": float(row["joint_coverage"]),
                "bs_coverage": float(row["bs_coverage"]),
            }
        )
    return points


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.perfect_reference is not None:
        config["environment"]["backend"]["perfect_reference_result"] = str(
            args.perfect_reference.expanduser().resolve()
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action_seed = int(config["seed"] if args.action_seed is None else args.action_seed)
    config.setdefault("evaluation", {}).update(
        {
            "type": "fixed_policy_sanity",
            "policy": args.policy,
            "action_seed": action_seed,
            "training_enabled": False,
            "qwen_enabled": False,
        }
    )
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    shutil.copy2(config["_config_path"], output_dir / "input_config.yaml")

    env = make_env(config, disable_qwen=True)
    ack_block = open_blocks(env.backend.layout, ("action_ack",))["action_ack"]
    rng = np.random.default_rng(action_seed)
    action_dim = int(env.mapping.action_dim)
    published_actions: dict[int, np.ndarray] = {}
    published_senders: dict[int, int] = {}
    raw_rows: queue.Queue[dict[str, Any] | None] = queue.Queue()
    resolved_rows: list[dict[str, Any]] = []
    resolver_errors: list[BaseException] = []
    csv_path = output_dir / "fixed_policy_trace.csv"
    fields = [
        "step_id", "sim_time_s", "delta_t_s", "policy", "action_seed",
        "round_robin_sender", "next_round_robin_sender", "startup_warmup",
        "requested_action_ones", "executed_action_ones",
        "published_action_version", "executed_action_version",
        "ack_action_version", "version_match", "ack_match", "action_match",
        "stale_action", "action_age_s", "reward", "reward_cumulative",
        "C_BS", "C_BS_ul", "C_BS_dl", "C_no_link", "C_reward_resource",
        "L_task", "L_task_time_weighted", "constraint_cost",
        "joint_coverage", "bs_coverage", "raw_coverage",
    ]

    def resolve_metrics() -> None:
        cumulative_reward = 0.0
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                while True:
                    first = raw_rows.get()
                    if first is None:
                        break
                    batch = [first]
                    while len(batch) < 128:
                        try:
                            item = raw_rows.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            raw_rows.put(None)
                            break
                        batch.append(item)
                    metrics = env.resolve_pending_costs(
                        [int(item["step_id"]) for item in batch]
                    )
                    for row, metric in zip(batch, metrics):
                        cumulative_reward += float(row["reward"])
                        row.update(
                            {
                                "reward_cumulative": cumulative_reward,
                                "L_task": float(metric["L_task"]),
                                "L_task_time_weighted": float(
                                    metric["L_task_time_weighted"]
                                ),
                                "constraint_cost": float(metric["cost"]),
                                "joint_coverage": float(metric["C_joint_map"]),
                                "bs_coverage": float(metric["C_BS_map"]),
                            }
                        )
                        writer.writerow({name: row[name] for name in fields})
                        resolved_rows.append(row)
                    handle.flush()
        except BaseException as error:
            resolver_errors.append(error)

    resolver = threading.Thread(target=resolve_metrics, name="metric-resolver")
    resolver.start()
    terminal = False
    try:
        env.reset(seed=int(config["seed"]))
        while not terminal:
            if resolver_errors:
                raise RuntimeError("metric resolver failed") from resolver_errors[0]
            source = env.cycle_status()
            decision_index = int(source["episode_step"])
            action = fixed_action(
                args.policy, action_dim, rng, decision_index=decision_index
            )
            published = int(env.publish_action(action, 0, source_context=source)["action_version"])
            published_actions[published] = action.copy()
            published_senders[published] = decision_index % env.n_uavs + 1
            try:
                _observation, reward, terminated, truncated, info = env.step(action)
            except MissionEndedError:
                break
            executed = np.asarray(info["executed_action"], dtype=np.int8).reshape(-1)
            ack = _ack_payload(ack_block)
            executed_version = int(info["action_version"])
            ack_version = int(ack.get("action_id", -1))
            expected_executed = published_actions.get(executed_version)
            startup_warmup = executed_version == 0 and expected_executed is None
            version_match = startup_warmup or expected_executed is not None
            ack_match = ack_version == published
            action_match = startup_warmup or bool(
                expected_executed is not None
                and np.array_equal(executed, expected_executed)
            )
            row = {
                "step_id": int(info["cost_step_id"]),
                "sim_time_s": float(info["simulation_time_s"]),
                "delta_t_s": float(info["transition_sim_time_s"]),
                "policy": args.policy,
                "action_seed": action_seed,
                "round_robin_sender": (
                    published_senders.get(executed_version, 0)
                    if args.policy == "round_robin_row"
                    else 0
                ),
                "next_round_robin_sender": (
                    decision_index % env.n_uavs + 1
                    if args.policy == "round_robin_row"
                    else 0
                ),
                "startup_warmup": int(startup_warmup),
                "requested_action_ones": int(action.sum()),
                "executed_action_ones": int(executed.sum()),
                "published_action_version": published,
                "executed_action_version": executed_version,
                "ack_action_version": ack_version,
                "version_match": int(version_match),
                "ack_match": int(ack_match),
                "action_match": int(action_match),
                "stale_action": int(not (version_match and ack_match and action_match)),
                "action_age_s": float(info["action_age"]),
                "reward": float(reward),
                "C_BS": float(info["C_BS"]),
                "C_BS_ul": float(info["C_BS_ul"]),
                "C_BS_dl": float(info["C_BS_dl"]),
                "C_no_link": float(info["C_no_link"]),
                "C_reward_resource": float(info["C_reward_resource"]),
                "raw_coverage": float(info["coverage"]),
            }
            raw_rows.put(row)
            terminal = bool(terminated or truncated)
    finally:
        raw_rows.put(None)
        resolver.join()
        ack_block.close()
        env.close()
    if resolver_errors:
        raise RuntimeError("metric resolver failed") from resolver_errors[0]
    if not resolved_rows:
        raise RuntimeError("fixed-policy evaluation produced no transitions")
    resolved_rows.sort(key=lambda item: int(item["step_id"]))
    final = resolved_rows[-1]
    summary = {
        "evaluation": "fixed_policy_sanity",
        "training_enabled": False,
        "policy": args.policy,
        "environment_seed": int(config["seed"]),
        "action_seed": action_seed,
        "action_dim": action_dim,
        "transitions": len(resolved_rows),
        "simulation_time_s": float(final["sim_time_s"]),
        "total_reward": float(sum(float(row["reward"]) for row in resolved_rows)),
        "mean_reward": float(np.mean([float(row["reward"]) for row in resolved_rows])),
        "total_C_BS": float(sum(float(row["C_BS"]) for row in resolved_rows)),
        "total_C_no_link": float(sum(float(row["C_no_link"]) for row in resolved_rows)),
        "total_C_reward_resource": float(
            sum(float(row["C_reward_resource"]) for row in resolved_rows)
        ),
        "final_L_task": float(final["L_task"]),
        "time_weighted_L_task": float(final["L_task_time_weighted"]),
        "final_joint_coverage": float(final["joint_coverage"]),
        "final_bs_coverage": float(final["bs_coverage"]),
        "final_raw_coverage": float(final["raw_coverage"]),
        "requested_action_ones_mean": float(
            np.mean([float(row["requested_action_ones"]) for row in resolved_rows])
        ),
        "executed_action_ones_mean": float(
            np.mean([float(row["executed_action_ones"]) for row in resolved_rows])
        ),
        "version_mismatches": int(sum(not bool(row["version_match"]) for row in resolved_rows)),
        "ack_mismatches": int(sum(not bool(row["ack_match"]) for row in resolved_rows)),
        "action_mismatches": int(sum(not bool(row["action_match"]) for row in resolved_rows)),
        "stale_actions": int(sum(int(row["stale_action"]) for row in resolved_rows)),
        "startup_warmup_transitions": int(
            sum(int(row["startup_warmup"]) for row in resolved_rows)
        ),
        "max_action_age_s": float(max(float(row["action_age_s"]) for row in resolved_rows)),
        "curve_30s": _curve_points(resolved_rows),
        "trace_csv": str(csv_path),
    }
    (output_dir / "fixed_policy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RACER_FIXED_POLICY_EVAL_OK " + json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
