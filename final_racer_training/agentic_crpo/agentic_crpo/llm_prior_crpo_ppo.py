"""Independent CRPO/PPO variant using the soft LLM-logit-prior policy."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from .crpo_buffer import CRPORolloutBuffer
from .crpo_policy import TASK_PHASE_DUAL_CRITIC_ARCHITECTURE
from .crpo_ppo import CONSTRAINT_COST_VERSION, CRPOPPO, _CRPODecisionRecord
from .episode_tracker import EpisodeCostTracker
from .llm_prior_policy import (
    LLM_PRIOR_ARCHITECTURE,
    LLM_PRIOR_UNMASKED_ARCHITECTURE,
    LLMPriorCRPOActorCriticPolicy,
    llm_prior_architecture,
)


class LLMPriorCRPOPPO(CRPOPPO):
    """The existing CRPO update with an independently versioned policy.

    Inheriting the rollout and optimizer logic keeps the CRPO equations exactly
    aligned with the baseline.  Construction deliberately bypasses
    ``CRPOPPO.__init__`` because that constructor correctly and intentionally
    hard-codes the baseline policy class.
    """

    policy: LLMPriorCRPOActorCriticPolicy
    rollout_buffer: CRPORolloutBuffer

    @classmethod
    def load(
        cls,
        path: str | Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> "LLMPriorCRPOPPO":
        """Accept only checkpoints created by this LLM-prior variant."""

        stream_position: int | None = None
        if isinstance(path, io.BufferedIOBase):
            if not path.seekable():
                raise ValueError(
                    "checkpoint streams must be seekable for architecture "
                    "validation"
                )
            stream_position = path.tell()
        data, params, _ = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=False,
        )
        if stream_position is not None:
            path.seek(stream_position)
        if data is None or params is None:
            raise ValueError("checkpoint does not contain model data/parameters")
        saved_architecture = data.get("policy_architecture_version")
        supported_architectures = {
            LLM_PRIOR_ARCHITECTURE,
            LLM_PRIOR_UNMASKED_ARCHITECTURE,
        }
        if saved_architecture not in supported_architectures:
            raise ValueError(
                "incompatible checkpoint architecture: expected one of "
                f"{sorted(supported_architectures)!r}, got "
                f"{saved_architecture!r}. "
                "Baseline CRPO checkpoints cannot initialize the explicit "
                "LLM action-prior policy; start a new training run."
            )
        policy_state = params.get("policy")
        required_parameters = {
            "features_extractor.physical_encoder.0.weight",
            "features_extractor.guidance_encoder.0.weight",
            "features_extractor.fusion_encoder.0.weight",
            "llm_prior_gate_net.weight",
            "llm_prior_gate_net.bias",
        }
        if not isinstance(policy_state, dict):
            raise ValueError("checkpoint is missing the policy state dictionary")
        missing = sorted(required_parameters.difference(policy_state))
        if missing:
            raise ValueError(
                "checkpoint architecture metadata names the LLM-prior policy, "
                f"but parameters are missing: {missing}"
            )

        # Calling PPO's classmethod implementation with this subclass avoids
        # the baseline CRPOPPO.load() architecture guard while retaining SB3's
        # standard restore path.
        model = PPO.load.__func__(
            cls,
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        if not isinstance(model, cls):
            raise TypeError("SB3 restored an unexpected algorithm class")
        if model.policy.architecture_version != saved_architecture:
            raise ValueError(
                "checkpoint action-mask mode disagrees with its architecture "
                f"metadata: policy={model.policy.architecture_version!r}, "
                f"checkpoint={saved_architecture!r}"
            )
        model._initialize_policy_runtime(force=True)
        return model

    def __init__(
        self,
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma_reward: float = 1.0,
        gamma_cost: float = 1.0,
        gae_lambda_reward: float = 0.95,
        gae_lambda_cost: float = 0.95,
        gamma_task: float = 1.0,
        eta: float = 0.0,
        episode_cost_window: int = 20,
        telescoping_tolerance: float = 1.0e-6,
        constraint_estimator: str = "time_weighted_mean",
        cost_vf_coef: float = 0.5,
        policy_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if constraint_estimator not in (
            "episode_return",
            "critic_estimate",
            "mean_step_cost",
            "time_weighted_mean",
        ):
            raise ValueError(
                "constraint_estimator must be episode_return, "
                "critic_estimate, mean_step_cost, or time_weighted_mean"
            )
        self.gamma_cost = float(gamma_cost)
        if (
            constraint_estimator == "time_weighted_mean"
            and not np.isclose(self.gamma_cost, 1.0)
        ):
            raise ValueError(
                "time_weighted_mean requires gamma_cost=1 so the fixed-horizon "
                "constraint return remains undiscounted"
            )
        self.gae_lambda_cost = float(gae_lambda_cost)
        self.gamma_task = float(gamma_task)
        self.eta = float(eta)
        self.episode_cost_window = int(episode_cost_window)
        self.telescoping_tolerance = float(telescoping_tolerance)
        self.constraint_estimator = constraint_estimator
        self.constraint_cost_version = (
            None
            if kwargs.get("_init_setup_model", True) is False
            else CONSTRAINT_COST_VERSION
        )
        self.cost_vf_coef = float(cost_vf_coef)
        mask_no_link_actions = bool(
            (policy_kwargs or {}).get("mask_no_link_actions", False)
        )
        self.policy_architecture_version = llm_prior_architecture(
            mask_no_link_actions
        )
        self.critic_architecture_version = TASK_PHASE_DUAL_CRITIC_ARCHITECTURE
        self.reward_updates = 0
        self.cost_updates = 0
        self.crpo_mode = "reward"
        self.constraint_estimator_source = "uninitialized"
        self.j_cost_hat = 0.0
        self._last_cost_value_estimate: np.ndarray | None = None
        self.ended_on_mission_boundary = False
        self.partial_rollout_updates = 0
        self.sync_update_records: list[dict[str, Any]] = []
        self._stop_before_next_rollout = False
        self.policy_version = 0
        self._behavior_decision_records: dict[
            tuple[int, int], _CRPODecisionRecord
        ] = {}
        kwargs.pop("policy", None)
        PPO.__init__(
            self,
            LLMPriorCRPOActorCriticPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma_reward,
            gae_lambda=gae_lambda_reward,
            rollout_buffer_class=CRPORolloutBuffer,
            rollout_buffer_kwargs={
                "gamma_cost": self.gamma_cost,
                "gae_lambda_cost": self.gae_lambda_cost,
            },
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
        if hasattr(self, "n_envs"):
            self.episode_cost_tracker = EpisodeCostTracker(
                self.n_envs,
                self.episode_cost_window,
                self.telescoping_tolerance,
                check_telescoping=(
                    self.constraint_estimator != "mean_step_cost"
                ),
            )
        if hasattr(self, "policy"):
            self._initialize_policy_runtime(force=True)

    def set_mask_no_link_actions(self, enabled: bool) -> None:
        """Change only the hard feasibility mask and refresh behavior policy."""

        self.policy.set_mask_no_link_actions(enabled)
        self.policy_kwargs = dict(self.policy_kwargs)
        self.policy_kwargs["mask_no_link_actions"] = enabled
        self.policy_architecture_version = llm_prior_architecture(enabled)
        self._initialize_policy_runtime(force=True)

    def set_no_link_logit_penalty(self, penalty: float) -> None:
        """Update the learner and behavior distributions between episodes."""

        self.policy.set_no_link_logit_penalty(penalty)
        self.policy_kwargs = dict(self.policy_kwargs)
        self.policy_kwargs["no_link_logit_penalty"] = float(penalty)
        self._initialize_policy_runtime(force=True)

    def train(self) -> None:
        """Run the unchanged CRPO update, then log explicit prior influence."""

        super().train()
        observations = getattr(self.rollout_buffer, "observations", None)
        if observations is None or np.size(observations) == 0:
            return
        with th.no_grad():
            obs = th.as_tensor(
                observations, device=self.device, dtype=th.float32
            )
            features = self.policy.extract_features(obs)
            available = self.policy.bs_link_availability(obs)
            latent_pi = self.policy._actor_latent(features, available)
            base_logits = self.policy.action_net(latent_pi)
            residual, bias, gate = self.policy.llm_prior_residual(obs, features)
            final_logits = base_logits + residual
            feasible = (
                self.policy.action_mask(obs)
                if self.policy.mask_no_link_actions
                else th.ones_like(final_logits, dtype=th.bool)
            )

            base_probability = th.sigmoid(base_logits).clamp(1.0e-7, 1 - 1.0e-7)
            final_probability = th.sigmoid(final_logits).clamp(
                1.0e-7, 1 - 1.0e-7
            )
            bit_kl = (
                final_probability
                * (final_probability.log() - base_probability.log())
                + (1.0 - final_probability)
                * (
                    (1.0 - final_probability).log()
                    - (1.0 - base_probability).log()
                )
            )
            feasible_float = feasible.to(dtype=bit_kl.dtype)
            feasible_count = feasible_float.sum().clamp_min(1.0)
            sample_kl = (bit_kl * feasible_float).sum(dim=-1)
            informative = (bias.abs() > 1.0e-6) & feasible

            self.logger.record(
                "llm_prior/gate_mean", float(gate.mean().cpu().item())
            )
            self.logger.record(
                "llm_prior/abs_logit_residual_mean",
                float(
                    (residual.abs() * feasible_float).sum().cpu().item()
                    / feasible_count.cpu().item()
                ),
            )
            self.logger.record(
                "llm_prior/bernoulli_product_kl_mean",
                float(sample_kl.mean().cpu().item()),
            )
            self.logger.record(
                "llm_prior/informative_feasible_fraction",
                float(
                    informative.to(dtype=th.float32).sum().cpu().item()
                    / feasible_count.cpu().item()
                ),
            )
            self.logger.record(
                "llm_prior/relay_beta",
                self.policy.llm_prior_relay_beta,
            )
            self.logger.record(
                "llm_prior/upload_beta",
                self.policy.llm_prior_upload_beta,
            )


__all__ = ["LLMPriorCRPOPPO"]
