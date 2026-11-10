"""Training entry point for the separate soft-LLM-action-prior variant."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil

from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from .config import load_config, save_resolved_config
from .constraint import print_calibration, resolve_constraint
from .factory import make_env
from .llm_prior_crpo_ppo import LLMPriorCRPOPPO
from .no_link_schedule import no_link_mask_for_episode
from .train_crpo import (
    _ACTIVATION_FUNCTIONS,
    _activation_class,
    _configure_rl_pytorch_threads,
    _learn_until_target_or_mission_end,
    _migrate_constraint_state_for_resume,
    _prepare_resume_for_new_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--perfect-reference", type=Path)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--gamma-task", type=float)
    parser.add_argument(
        "--activation-fn", choices=sorted(_ACTIVATION_FUNCTIONS)
    )
    parser.add_argument("--relay-beta", type=float)
    parser.add_argument("--upload-beta", type=float)
    parser.add_argument("--prior-clip", type=float)
    parser.add_argument("--initial-gate", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mock-smoke", action="store_true")
    return parser.parse_args()


def _prior_config(
    config: dict[str, object], args: argparse.Namespace
) -> dict[str, float]:
    raw = config.get("llm_action_prior")
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        raise ValueError(
            "the LLM-prior training entry requires "
            "llm_action_prior.enabled=true"
        )
    overrides = {
        "relay_beta": args.relay_beta,
        "upload_beta": args.upload_beta,
        "clip": args.prior_clip,
        "initial_gate": args.initial_gate,
    }
    for name, value in overrides.items():
        if value is not None:
            raw[name] = value
    required = ("relay_beta", "upload_beta", "clip", "initial_gate")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"missing llm_action_prior settings: {missing}")
    result = {name: float(raw[name]) for name in required}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("LLM-prior settings must be finite")
    if result["relay_beta"] < 0.0 or result["upload_beta"] < 0.0:
        raise ValueError("LLM-prior beta values must be non-negative")
    if result["clip"] <= 0.0:
        raise ValueError("LLM-prior clip must be positive")
    if not 0.0 < result["initial_gate"] < 1.0:
        raise ValueError("LLM-prior initial_gate must be in (0, 1)")
    return result


def _verify_resumed_prior(
    model: LLMPriorCRPOPPO, prior: dict[str, float]
) -> None:
    actual = {
        "relay_beta": model.policy.llm_prior_relay_beta,
        "upload_beta": model.policy.llm_prior_upload_beta,
        "clip": model.policy.llm_prior_clip,
        "initial_gate": model.policy.llm_prior_initial_gate,
    }
    mismatches = {
        name: (actual[name], configured)
        for name, configured in prior.items()
        if not math.isclose(
            actual[name], configured, rel_tol=0.0, abs_tol=1.0e-12
        )
    }
    if mismatches:
        raise ValueError(
            "resume checkpoint LLM-prior settings differ from config: "
            f"{mismatches}"
        )


def _mask_no_link_actions(config: dict[str, object]) -> bool:
    actor = config.get("actor", {})
    if not isinstance(actor, dict):
        raise ValueError("actor config must be a mapping")
    enabled = actor.get("mask_no_link_actions", False)
    if not isinstance(enabled, bool):
        raise ValueError("actor.mask_no_link_actions must be a boolean")
    return enabled


def main() -> None:
    _configure_rl_pytorch_threads()
    args = parse_args()
    config = load_config(args.config)
    prior = _prior_config(config, args)
    episode_text = os.environ.get("RACER_QWEN8_EPISODE_NUMBER")
    episode_number = int(episode_text) if episode_text is not None else None
    actor_config = config.setdefault("actor", {})
    mask_no_link_actions, no_link_logit_penalty = no_link_mask_for_episode(
        actor_config, episode_number
    )
    # Keep configured options reloadable; effective values belong to this episode.
    config["training"]["effective_no_link_options"] = {
        "mask_no_link_actions": mask_no_link_actions,
        "no_link_logit_penalty": no_link_logit_penalty,
    }
    print("RACER_NO_LINK_EPISODE_OPTIONS " + json.dumps({
        "episode": episode_number,
        "schedule_enabled": actor_config["no_link_logit_schedule"]["enabled"],
        **config["training"]["effective_no_link_options"],
    }), flush=True)
    if episode_number is not None:
        config["training"]["episode_number"] = episode_number
    if not bool(config["environment"].get("include_global_guidance", True)):
        raise ValueError(
            "LLM action prior requires environment.include_global_guidance=true"
        )
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
    for name, value in {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "activation_fn": args.activation_fn,
    }.items():
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
            "variant", "final_racer_soft_llm_action_prior_crpo"
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
        model = LLMPriorCRPOPPO.load(
            args.resume,
            env=env,
            device=training["device"],
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
        _verify_resumed_prior(model, prior)
        previous_penalty = float(
            getattr(model.policy, "no_link_logit_penalty", 0.0)
        )
        if previous_penalty != no_link_logit_penalty:
            model.set_no_link_logit_penalty(no_link_logit_penalty)
            print(
                "RACER_NO_LINK_LOGIT_PENALTY "
                + json.dumps(
                    {
                        "episode": episode_number,
                        "previous": previous_penalty,
                        "current": no_link_logit_penalty,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        previous_mask_mode = bool(model.policy.mask_no_link_actions)
        mask_mode_changed = previous_mask_mode != mask_no_link_actions
        if mask_mode_changed:
            model.set_mask_no_link_actions(mask_no_link_actions)
            print(
                "RACER_ACTION_MASK_MODE_CHANGE "
                + json.dumps(
                    {
                        "previous": previous_mask_mode,
                        "current": mask_no_link_actions,
                        "policy_architecture": (
                            model.policy_architecture_version
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        for name, default in (
            ("ended_on_mission_boundary", False),
            ("partial_rollout_updates", 0),
            ("sync_update_records", []),
            ("_stop_before_next_rollout", False),
        ):
            if not hasattr(model, name):
                setattr(model, name, default)
        _prepare_resume_for_new_episode(model)
        saved_variant = getattr(model, "experiment_variant", None)
        if (
            saved_variant is None
            or (
                saved_variant != experiment_variant
                and not mask_mode_changed
            )
        ):
            raise ValueError(
                "resume checkpoint algorithm variant mismatch: "
                f"checkpoint={saved_variant}, config={experiment_variant}"
            )
        if saved_variant != experiment_variant:
            print(
                "RACER_VARIANT_MIGRATION "
                + json.dumps(
                    {
                        "checkpoint": saved_variant,
                        "config": experiment_variant,
                        "reason": "action_mask_mode_change",
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            model.experiment_variant = experiment_variant
        migration = _migrate_constraint_state_for_resume(
            model,
            expected_estimator=str(crpo["constraint_estimator"]),
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
        # Resume restores the checkpoint's PPO settings. Apply the requested
        # KL limit from this episode's config so scheduled changes take effect.
        model.target_kl = ppo.get("target_kl")
    else:
        model = LLMPriorCRPOPPO(
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
                "llm_prior_relay_beta": prior["relay_beta"],
                "llm_prior_upload_beta": prior["upload_beta"],
                "llm_prior_clip": prior["clip"],
                "llm_prior_initial_gate": prior["initial_gate"],
                "initial_action_probability": float(
                    config.get("actor", {}).get("initial_action_probability", 0.5)
                ),
                "mask_no_link_actions": mask_no_link_actions,
                "no_link_logit_penalty": no_link_logit_penalty,
            },
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=int(config["seed"]),
            device=str(training["device"]),
            verbose=int(training.get("verbose", 1)),
        )
        model.experiment_variant = experiment_variant

    callback = CheckpointCallback(
        save_freq=max(1, int(training["checkpoint_interval"])),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="llm_prior_crpo",
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
    episode_metrics = env.episode_metrics(complete=ended_on_mission_boundary)
    (output_dir / "episode_metrics.json").write_text(
        json.dumps(episode_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "RACER_EPISODE_METRICS "
        + json.dumps(episode_metrics, separators=(",", ":")),
        flush=True,
    )
    final_path = output_dir / "llm_prior_crpo_final"
    model.save(final_path)
    metadata = {
        "n_uavs": int(config["n_uavs"]),
        "algorithm_variant": experiment_variant,
        "policy_architecture": model.policy_architecture_version,
        "mask_no_link_actions": bool(model.policy.mask_no_link_actions),
        "no_link_logit_penalty": float(model.policy.no_link_logit_penalty),
        "no_link_logit_schedule": actor_config["no_link_logit_schedule"],
        "episode_number": episode_number,
        "observation_dim": env.small_builder.observation_dim,
        "physical_state_dim": model.policy.features_extractor.physical_state_dim,
        "llm_guidance_dim": model.policy.features_extractor.guidance_dim,
        "action_dim": env.mapping.action_dim,
        "num_timesteps": int(model.num_timesteps),
        "ended_on_mission_boundary": ended_on_mission_boundary,
        "episode_metrics": episode_metrics,
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
        "llm_action_prior": config["llm_action_prior"],
        "learned_prior_gate_bias": float(
            model.policy.llm_prior_gate_net.bias.detach().cpu().item()
        ),
        "learned_prior_gate_weight_norm": float(
            torch.linalg.vector_norm(
                model.policy.llm_prior_gate_net.weight.detach()
            ).cpu().item()
        ),
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
