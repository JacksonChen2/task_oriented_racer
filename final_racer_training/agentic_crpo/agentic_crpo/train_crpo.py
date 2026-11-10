"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from torch import nn

from .config import load_config, save_resolved_config
from .constraint import print_calibration, resolve_constraint
from .crpo_policy import DUAL_ENCODER_ARCHITECTURE
from .crpo_ppo import CONSTRAINT_COST_VERSION, CRPOPPO
from .episode_tracker import EpisodeCostTracker
from .factory import make_env


_ACTIVATION_FUNCTIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def _configure_rl_pytorch_threads() -> None:
    """Keep PyTorch from competing with the saturated simulation CPUs."""

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to fix RL PyTorch CPU thread counts at one")
    print(
        "RACER_RL_TORCH_THREADS "
        + json.dumps(
            {
                "intra_op_threads": torch.get_num_threads(),
                "inter_op_threads": torch.get_num_interop_threads(),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def _activation_class(name: str) -> type[nn.Module]:
    key = name.strip().lower()
    try:
        return _ACTIVATION_FUNCTIONS[key]
    except KeyError as error:
        supported = ", ".join(sorted(_ACTIVATION_FUNCTIONS))
        raise ValueError(
            f"unsupported PPO activation_fn={name!r}; choose one of: {supported}"
        ) from error


def _prepare_resume_for_new_episode(model: CRPOPPO) -> None:
    """Keep learned state but force a fresh synchronized environment reset.

    SB3 serializes ``_last_obs``.  Reusing that terminal observation in a new
    Isaac process would bypass FileBridgeBackend.reset() and disconnect the
    policy from the new episode's exact s_0 boundary.  Mission-stop flags are
    likewise per-process state, not checkpoint state.
    """

    model.ended_on_mission_boundary = False
    model._stop_before_next_rollout = False
    model._last_obs = None
    model._last_original_obs = None
    model._last_episode_starts = None
    tracker = getattr(model, "episode_cost_tracker", None)
    if tracker is not None:
        tracker.reset_active()


def _migrate_constraint_state_for_resume(
    model: CRPOPPO,
    *,
    expected_estimator: str,
    episode_cost_window: int,
    telescoping_tolerance: float,
) -> dict[str, object] | None:
    """Reset statistics when a checkpoint used an older cost definition."""

    saved_cost_version = getattr(model, "constraint_cost_version", None)
    requires_migration = (
        model.constraint_estimator != expected_estimator
        or saved_cost_version != CONSTRAINT_COST_VERSION
    )
    if not requires_migration:
        return None
    if expected_estimator != "time_weighted_mean":
        raise ValueError(
            "resume checkpoint constraint definition mismatch: "
            f"checkpoint={model.constraint_estimator}, "
            f"config={expected_estimator}"
        )
    migration = {
        "checkpoint": model.constraint_estimator,
        "configured": expected_estimator,
        "checkpoint_cost_version": saved_cost_version,
        "configured_cost_version": CONSTRAINT_COST_VERSION,
        "historical_constraint_records_cleared": True,
    }
    model.constraint_estimator = expected_estimator
    model.constraint_cost_version = CONSTRAINT_COST_VERSION
    model.episode_cost_window = int(episode_cost_window)
    model.telescoping_tolerance = float(telescoping_tolerance)
    model.episode_cost_tracker = EpisodeCostTracker(
        model.n_envs,
        model.episode_cost_window,
        model.telescoping_tolerance,
    )
    return migration


def _mission_has_ended(env: object) -> bool:
    snapshot = getattr(env, "snapshot", None)
    if snapshot is not None and (snapshot.terminated or snapshot.truncated):
        return True

    # The communication proxy can exit just before publishing its final RL
    # telemetry. Isaac writes mission_state independently, so consult that
    # authoritative terminal flag when the in-memory snapshot is one slot old.
    backend = getattr(env, "backend", None)
    mission_path = getattr(backend, "mission_state_path", None)
    if mission_path is None:
        return False
    try:
        mission = json.loads(Path(mission_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(mission.get("terminated", False) or mission.get("truncated", False))


def _learn_until_target_or_mission_end(
    model: CRPOPPO,
    env: object,
    *,
    total_timesteps: int,
    callback: CheckpointCallback,
    reset_num_timesteps: bool,
) -> bool:
    """Train until the requested target, or preserve work at a mission boundary.

    Stable-Baselines3 automatically resets a vectorized environment after a
    terminal transition.  A live file bridge has no next episode once Isaac
    has finished, so that automatic reset eventually times out waiting for new
    telemetry.  Treat the timeout as a clean boundary only when training made
    progress and the last mission snapshot is explicitly terminal/truncated.
    """

    starting_timesteps = int(model.num_timesteps)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    except TimeoutError:
        mission_ended = _mission_has_ended(env)
        if int(model.num_timesteps) <= starting_timesteps or not mission_ended:
            raise
        print(
            "Isaac mission ended before the requested training target; "
            f"saving {model.num_timesteps} completed timesteps."
        )
        return True
    # The synchronous collector can stop cleanly without throwing: it first
    # trains a non-empty partial rollout at the terminal boundary, then makes
    # the following collect_rollouts() return False.  Preserve that explicit
    # mission-boundary result in the run metadata.
    return bool(getattr(model, "ended_on_mission_boundary", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--perfect-reference",
        type=Path,
        help="override environment.backend.perfect_reference_result",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="override training.total_timesteps for short integrations",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument(
        "--gamma-task",
        type=float,
        help="manual constraint threshold; overrides baseline-gap calibration",
    )
    parser.add_argument(
        "--activation-fn",
        choices=sorted(_ACTIVATION_FUNCTIONS),
    )
    parser.add_argument(
        "--output-dir", type=Path, help="override training.output_dir"
    )
    parser.add_argument(
        "--mock-smoke",
        action="store_true",
        help="force the mock backend, disable Qwen, and use smoke timesteps",
    )
    return parser.parse_args()


def main() -> None:
    _configure_rl_pytorch_threads()
    args = parse_args()
    config = load_config(args.config)
    if args.gamma_task is not None:
        config.setdefault("constraint", {})["gamma_task"] = args.gamma_task
    if args.perfect_reference is not None:
        config["environment"]["backend"]["perfect_reference_result"] = str(
            args.perfect_reference.expanduser().resolve()
        )
    if args.total_timesteps is not None:
        if args.total_timesteps < 1:
            raise ValueError("--total-timesteps must be positive")
        config["training"]["total_timesteps"] = args.total_timesteps
    ppo_overrides = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "activation_fn": args.activation_fn,
    }
    for name, value in ppo_overrides.items():
        if value is not None:
            config["crpo"]["ppo"][name] = value
    ppo_config = config["crpo"]["ppo"]
    if float(ppo_config["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    for name in ("n_steps", "batch_size", "n_epochs"):
        if int(ppo_config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(ppo_config["batch_size"]) > int(ppo_config["n_steps"]):
        raise ValueError("batch_size cannot exceed n_steps for one environment")
    activation_fn = _activation_class(
        str(ppo_config.get("activation_fn", "silu"))
    )
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(
            args.output_dir.expanduser().resolve()
        )
    training = config["training"]
    experiment_variant = str(
        config.get("algorithm", {}).get(
            "variant",
            "final_racer_qwen8b_fp8_resource_ltask025_perfect_direct_dual_encoder_bs_mask",
        )
    )
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    constraint = resolve_constraint(config)
    config.setdefault("constraint", {})["resolved_gamma_task"] = (
        constraint.gamma_task
    )
    print_calibration(constraint)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    shutil.copy2(config["_config_path"], output_dir / "input_config.yaml")
    env = make_env(
        config, force_mock=args.mock_smoke, disable_qwen=args.mock_smoke
    )
    crpo = config["crpo"]
    ppo = crpo["ppo"]
    rollout_steps = (
        min(32, int(ppo["n_steps"]))
        if args.mock_smoke
        else int(ppo["n_steps"])
    )
    batch_size = (
        min(16, int(ppo["batch_size"]), rollout_steps)
        if args.mock_smoke
        else int(ppo["batch_size"])
    )
    training_epochs = (
        min(2, int(ppo["n_epochs"]))
        if args.mock_smoke
        else int(ppo["n_epochs"])
    )
    if args.resume is not None:
        # Runtime settings override saved hyperparameters while retaining weights
        # and optimizer state. Pass buffer settings before load builds the buffer.
        model = CRPOPPO.load(
            args.resume, env=env, device=training["device"],
            learning_rate=float(ppo["learning_rate"]),
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=training_epochs,
            gamma=float(crpo["gamma_reward"]),
            gamma_cost=float(crpo["gamma_cost"]),
            gae_lambda=float(crpo["gae_lambda_reward"]),
            rollout_buffer_kwargs={
                "gamma_cost": float(crpo["gamma_cost"]),
                "gae_lambda_cost": float(crpo["gae_lambda_cost"]),
            },
            ent_coef=float(ppo["ent_coef"]),
        )
        # Checkpoints created before synchronous collection do not contain
        # these bookkeeping attributes.  They are operational state rather
        # than learned parameters, so initialize them safely on resume.
        if not hasattr(model, "ended_on_mission_boundary"):
            model.ended_on_mission_boundary = False
        if not hasattr(model, "partial_rollout_updates"):
            model.partial_rollout_updates = 0
        if not hasattr(model, "sync_update_records"):
            model.sync_update_records = []
        if not hasattr(model, "_stop_before_next_rollout"):
            model._stop_before_next_rollout = False
        _prepare_resume_for_new_episode(model)
        saved_variant = getattr(model, "experiment_variant", None)
        if saved_variant is None:
            raise ValueError(
                "the resume checkpoint predates algorithm-variant metadata; "
                "do not reuse a legacy checkpoint for this Qwen8B-FP8 variant"
            )
        if saved_variant is not None and saved_variant != experiment_variant:
            raise ValueError(
                "resume checkpoint algorithm variant mismatch: "
                f"checkpoint={saved_variant}, config={experiment_variant}"
            )
        expected_estimator = str(crpo["constraint_estimator"])
        migration = _migrate_constraint_state_for_resume(
            model,
            expected_estimator=expected_estimator,
            episode_cost_window=int(crpo["episode_cost_window"]),
            telescoping_tolerance=float(
                config["constraint"].get("telescoping_tolerance", 1.0e-6)
            ),
        )
        if migration is not None:
            print(
                "RACER_CONSTRAINT_ESTIMATOR_MIGRATION "
                + json.dumps(migration, separators=(",", ":")),
                flush=True,
            )
        model.gamma_task = constraint.gamma_task
        model.eta = constraint.eta
    else:
        model = CRPOPPO(
            env,
            learning_rate=float(ppo["learning_rate"]),
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=training_epochs,
            gamma_reward=float(crpo["gamma_reward"]),
            gamma_cost=float(crpo["gamma_cost"]),
            gae_lambda_reward=float(crpo["gae_lambda_reward"]),
            gae_lambda_cost=float(crpo["gae_lambda_cost"]),
            gamma_task=constraint.gamma_task,
            eta=constraint.eta,
            episode_cost_window=int(crpo["episode_cost_window"]),
            telescoping_tolerance=float(
                config["constraint"].get("telescoping_tolerance", 1.0e-6)
            ),
            constraint_estimator=str(crpo["constraint_estimator"]),
            cost_vf_coef=float(ppo["cost_vf_coef"]),
            clip_range=float(ppo["clip_range"]),
            ent_coef=float(ppo["ent_coef"]),
            vf_coef=float(ppo["reward_vf_coef"]),
            max_grad_norm=float(ppo["max_grad_norm"]),
            target_kl=ppo.get("target_kl"),
            normalize_advantage=bool(ppo["normalize_advantage"]),
            policy_kwargs={
                "features_extractor_kwargs": {
                    "observation_layout": env.small_builder.observation_layout,
                },
                "net_arch": ppo["net_arch"],
                "activation_fn": activation_fn,
            },
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=int(config["seed"]),
            device=str(training["device"]),
            verbose=int(training.get("verbose", 1)),
        )
        model.experiment_variant = experiment_variant
    interval = int(training["checkpoint_interval"])
    callback = CheckpointCallback(
        save_freq=max(1, interval),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="crpo",
    )
    timesteps = int(
        training.get("smoke_timesteps", 64)
        if args.mock_smoke
        else training["total_timesteps"]
    )
    ended_on_mission_boundary = _learn_until_target_or_mission_end(
        model,
        env,
        total_timesteps=timesteps,
        callback=callback,
        reset_num_timesteps=args.resume is None,
    )
    final_path = output_dir / "crpo_final"
    model.save(final_path)
    metadata = {
        "n_uavs": int(config["n_uavs"]),
        "algorithm_variant": experiment_variant,
        "policy_architecture": DUAL_ENCODER_ARCHITECTURE,
        "observation_dim": env.small_builder.observation_dim,
        "physical_state_dim": model.policy.features_extractor.physical_state_dim,
        "llm_guidance_dim": model.policy.features_extractor.guidance_dim,
        "action_dim": env.mapping.action_dim,
        "num_timesteps": int(model.num_timesteps),
        "ended_on_mission_boundary": ended_on_mission_boundary,
        "partial_rollout_updates": model.partial_rollout_updates,
        "synchronous_ppo_updates": model.sync_update_records,
        "reward_updates": model.reward_updates,
        "cost_updates": model.cost_updates,
        "last_mode": model.crpo_mode,
        "last_J_C_hat": model.j_cost_hat,
        "constraint_estimator": model.constraint_estimator,
        "constraint_estimator_source": model.constraint_estimator_source,
        "constraint_cost_version": model.constraint_cost_version,
        "constraint_calibration": constraint.as_dict(),
        "ppo_hyperparameters": ppo,
        "qwen_config": config["qwen"],
        "reward_config": config.get(
            "reward", {"type": "negative_bs_resource", "scale": 1.0}
        ),
        "constraint_config": config["constraint"],
        "normalization": config["environment"]["normalization"],
        "episode_statistics": [
            asdict(record) for record in model.episode_cost_tracker.records
        ],
    }
    (output_dir / "training_state.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env.close()


if __name__ == "__main__":
    main()
