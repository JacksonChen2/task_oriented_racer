"""CRPO policy variant with a soft, explicit LLM action-logit prior.

This module is deliberately separate from :mod:`agentic_crpo.crpo_policy`.
The existing dual-encoder policy remains the baseline and is not modified.
"""

from __future__ import annotations

import math

import torch as th
from torch import nn

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from .crpo_policy import CRPOActorCriticPolicy, bernoulli_entropy_statistics


LLM_PRIOR_MASKED_ARCHITECTURE = (
    "physical_guidance_dual_encoder_bs_mask_soft_llm_logit_prior_v1"
)
LLM_PRIOR_UNMASKED_ARCHITECTURE = (
    "physical_guidance_dual_encoder_unmasked_soft_llm_logit_prior_v1"
)
# Backwards-compatible name used by the existing masked-policy tests and
# checkpoints.
LLM_PRIOR_ARCHITECTURE = LLM_PRIOR_MASKED_ARCHITECTURE


def llm_prior_architecture(mask_no_link_actions: bool) -> str:
    return (
        LLM_PRIOR_MASKED_ARCHITECTURE
        if mask_no_link_actions
        else LLM_PRIOR_UNMASKED_ARCHITECTURE
    )


class LLMPriorCRPOActorCriticPolicy(CRPOActorCriticPolicy):
    """Add a bounded LLM-derived residual to the Bernoulli action logits.

    The first ``N(N-1)`` residuals come from the off-diagonal entries of
    ``W_task`` and the final ``N`` residuals come from ``omega_sem``.  Both
    blocks are centered and standardized, so the prior changes relative link
    preference rather than globally increasing the number of active links.

    The prior is soft: an LLM-unselected action receives a finite logit shift
    and can still be selected by the learned policy. No-link hard masking is
    optional and disabled by default. When enabled, it is applied after the
    soft prior. An opt-in episode schedule can supply a finite no-link penalty
    before switching to the hard mask; both are disabled by default.
    """

    llm_prior_gate_net: nn.Linear

    def __init__(
        self,
        *args: object,
        llm_prior_relay_beta: float = 0.5,
        llm_prior_upload_beta: float = 0.5,
        llm_prior_clip: float = 2.0,
        llm_prior_initial_gate: float = 0.5,
        initial_action_probability: float = 0.5,
        mask_no_link_actions: bool = False,
        no_link_logit_penalty: float = 0.0,
        **kwargs: object,
    ) -> None:
        for name, value in (
            ("llm_prior_relay_beta", llm_prior_relay_beta),
            ("llm_prior_upload_beta", llm_prior_upload_beta),
            ("llm_prior_clip", llm_prior_clip),
            ("llm_prior_initial_gate", llm_prior_initial_gate),
            ("initial_action_probability", initial_action_probability),
            ("no_link_logit_penalty", no_link_logit_penalty),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if llm_prior_relay_beta < 0.0 or llm_prior_upload_beta < 0.0:
            raise ValueError("LLM-prior beta values must be non-negative")
        if llm_prior_clip <= 0.0:
            raise ValueError("llm_prior_clip must be positive")
        if not 0.0 < llm_prior_initial_gate < 1.0:
            raise ValueError("llm_prior_initial_gate must be in (0, 1)")
        if not 0.0 < initial_action_probability < 1.0:
            raise ValueError("initial_action_probability must be in (0, 1)")
        if not isinstance(mask_no_link_actions, bool):
            raise TypeError("mask_no_link_actions must be a boolean")
        if no_link_logit_penalty < 0.0:
            raise ValueError("no_link_logit_penalty must be non-negative")

        # These values must exist before ActorCriticPolicy calls the virtual
        # _build() method from its constructor.
        self.llm_prior_relay_beta = float(llm_prior_relay_beta)
        self.llm_prior_upload_beta = float(llm_prior_upload_beta)
        self.llm_prior_clip = float(llm_prior_clip)
        self.llm_prior_initial_gate = float(llm_prior_initial_gate)
        self.initial_action_probability = float(initial_action_probability)
        self.mask_no_link_actions = mask_no_link_actions
        self.no_link_logit_penalty = float(no_link_logit_penalty)
        super().__init__(*args, **kwargs)
        self.architecture_version = llm_prior_architecture(
            self.mask_no_link_actions
        )

    def set_mask_no_link_actions(self, enabled: bool) -> None:
        """Switch the hard no-link logit mask without changing parameters."""

        if not isinstance(enabled, bool):
            raise TypeError("mask_no_link_actions must be a boolean")
        self.mask_no_link_actions = enabled
        self.architecture_version = llm_prior_architecture(enabled)

    def set_no_link_logit_penalty(self, penalty: float) -> None:
        """Set the finite penalty resolved for the current episode."""
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError("no_link_logit_penalty must be finite and non-negative")
        self.no_link_logit_penalty = float(penalty)

    def _current_action_mask(self, obs: PyTorchObs) -> th.Tensor | None:
        return self.action_mask(obs) if self.mask_no_link_actions else None

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # At zero LLM residual, every fresh Bernoulli bit starts at the same
        # configured probability. Existing checkpoints restore their learned
        # action-net parameters after construction and are not reinitialized.
        nn.init.zeros_(self.action_net.weight)
        nn.init.constant_(
            self.action_net.bias,
            math.log(
                self.initial_action_probability
                / (1.0 - self.initial_action_probability)
            ),
        )
        self.llm_prior_gate_net = nn.Linear(
            self.features_extractor.features_dim,
            1,
            device=self.device,
        )
        # Begin with one state-independent, configured trust value. Training
        # can then learn when the global guidance should have more or less
        # influence without ever converting it into a hard action constraint.
        nn.init.zeros_(self.llm_prior_gate_net.weight)
        nn.init.constant_(
            self.llm_prior_gate_net.bias,
            math.log(
                self.llm_prior_initial_gate
                / (1.0 - self.llm_prior_initial_gate)
            ),
        )
        # Include the new gate only in the actor optimizer.
        self._build_independent_optimizers(lr_schedule)

    @staticmethod
    def _standardize_preferences(
        values: th.Tensor,
        *,
        clip: float,
    ) -> th.Tensor:
        """Center/scale the last axis and map a neutral row exactly to zero."""

        if values.shape[-1] == 0:
            return values
        centered = values - values.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).sqrt()
        normalized = th.where(
            scale > 1.0e-6,
            centered / scale.clamp_min(1.0e-6),
            th.zeros_like(centered),
        )
        return normalized.clamp(min=-clip, max=clip)

    def llm_prior_bias(self, obs: PyTorchObs) -> th.Tensor:
        """Return the bounded 100-D semantic preference in action-bit order."""

        if not isinstance(obs, th.Tensor):
            raise TypeError("LLM-prior policy requires a tensor observation")
        extractor = self.features_extractor
        if obs.shape[-1] != extractor.observation_dim:
            raise ValueError(
                f"expected observation width {extractor.observation_dim}, "
                f"got {obs.shape[-1]}"
            )
        n_uavs = extractor.n_uavs
        guidance = obs[
            ...,
            extractor.physical_state_dim : (
                extractor.physical_state_dim + extractor.guidance_dim
            ),
        ]
        task_dependency = guidance[..., : n_uavs * n_uavs].reshape(
            *guidance.shape[:-1], n_uavs, n_uavs
        )
        semantic_importance = guidance[..., n_uavs * n_uavs :]

        if n_uavs == 1:
            relay_bias = guidance.new_empty(*guidance.shape[:-1], 0)
        else:
            relay_values = task_dependency[
                ..., self._pair_off_diagonal
            ].reshape(*guidance.shape[:-1], n_uavs, n_uavs - 1)
            relay_bias = self._standardize_preferences(
                relay_values,
                clip=self.llm_prior_clip,
            ).reshape(*guidance.shape[:-1], n_uavs * (n_uavs - 1))
        upload_bias = self._standardize_preferences(
            semantic_importance,
            clip=self.llm_prior_clip,
        )
        bias = th.cat((relay_bias, upload_bias), dim=-1)
        expected = n_uavs * n_uavs
        if bias.shape[-1] != expected:
            raise RuntimeError(
                f"LLM-prior width {bias.shape[-1]} does not match {expected}"
            )
        if not bool(th.all(th.isfinite(bias))):
            raise FloatingPointError("LLM action prior contains NaN or Inf")
        return bias

    def llm_prior_gate(self, features: th.Tensor) -> th.Tensor:
        """Return the learned, per-observation trust gate in ``(0, 1)``."""

        gate = th.sigmoid(self.llm_prior_gate_net(features))
        if not bool(th.all(th.isfinite(gate))):
            raise FloatingPointError("LLM-prior gate contains NaN or Inf")
        return gate

    def llm_prior_residual(
        self,
        obs: PyTorchObs,
        features: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(scaled residual, normalized bias, learned gate)``."""

        bias = self.llm_prior_bias(obs)
        n_uavs = self.features_extractor.n_uavs
        relay_dim = n_uavs * (n_uavs - 1)
        scales = th.cat(
            (
                bias.new_full((relay_dim,), self.llm_prior_relay_beta),
                bias.new_full((n_uavs,), self.llm_prior_upload_beta),
            )
        )
        gate = self.llm_prior_gate(features)
        return gate * bias * scales, bias, gate

    def _prior_masked_action_logits(
        self,
        obs: PyTorchObs,
        features: th.Tensor,
        latent_pi: th.Tensor,
        action_mask: th.Tensor | None,
    ) -> th.Tensor:
        raw_logits = self.action_net(latent_pi)
        prior_residual, _bias, _gate = self.llm_prior_residual(obs, features)
        prior_logits = raw_logits + prior_residual
        if not bool(th.all(th.isfinite(prior_logits))):
            raise FloatingPointError(
                "LLM-prior policy action logits contain NaN or Inf"
            )
        if not self.mask_no_link_actions:
            if self.no_link_logit_penalty == 0.0:
                return prior_logits
            soft_mask = self.action_mask(obs)
            if prior_logits.shape != soft_mask.shape:
                raise ValueError("action logits and no-link mask shapes differ")
            # Preserve the original schedule's finite probability floor.
            penalized = (prior_logits - self.no_link_logit_penalty).clamp_min(-30.0)
            return th.where(soft_mask, prior_logits, penalized)
        if action_mask is None or prior_logits.shape != action_mask.shape:
            raise ValueError("masked policy requires an action mask matching its logits")
        # Match sampling, cached behavior logits and PPO likelihood evaluation.
        # The LLM prior cannot re-enable an unavailable physical direction.
        return prior_logits.masked_fill(
            ~action_mask, th.finfo(prior_logits.dtype).min
        )

    def _prior_action_distribution(
        self,
        obs: PyTorchObs,
        features: th.Tensor,
        latent_pi: th.Tensor,
        action_mask: th.Tensor | None,
    ) -> Distribution:
        return self._distribution_from_masked_logits(
            self._prior_masked_action_logits(
                obs, features, latent_pi, action_mask
            )
        )

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        critic_tau: th.Tensor | float | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        available = self.bs_link_availability(obs)
        latent_pi = self._actor_latent(features, available)
        values = self.predict_reward_values(obs, critic_tau)
        distribution = self._prior_action_distribution(
            obs, features, latent_pi, self._current_action_mask(obs)
        )
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        critic_tau: th.Tensor | float | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("LLM-prior policy requires a tensor observation")
        features = self.extract_features(obs)
        available = self.bs_link_availability(obs)
        latent_pi = self._actor_latent(features, available)
        distribution = self._prior_action_distribution(
            obs, features, latent_pi, self._current_action_mask(obs)
        )
        return (
            self.predict_reward_values(obs, critic_tau),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        if not isinstance(obs, th.Tensor):
            raise TypeError("LLM-prior policy requires a tensor observation")
        features = self.extract_features(obs)
        available = self.bs_link_availability(obs)
        latent_pi = self._actor_latent(features, available)
        return self._prior_action_distribution(
            obs, features, latent_pi, self._current_action_mask(obs)
        )

    def forward_crpo_with_masked_logits(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        critic_tau: th.Tensor | float | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Sample and cache the exact LLM-prior behavior distribution."""

        features = self.extract_features(obs)
        available = self.bs_link_availability(obs)
        latent_pi = self._actor_latent(features, available)
        reward_values = self.predict_reward_values(obs, critic_tau)
        cost_values = self.predict_cost_values(obs, critic_tau)
        masked_logits = self._prior_masked_action_logits(
            obs, features, latent_pi, self._current_action_mask(obs)
        )
        distribution = self._distribution_from_masked_logits(masked_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return (
            actions,
            reward_values,
            cost_values,
            log_prob,
            masked_logits,
        )

    def evaluate_actions_crpo(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        critic_tau: th.Tensor | float | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("LLM-prior policy requires a tensor observation")
        features = self.extract_features(obs)
        available = self.bs_link_availability(obs)
        latent_pi = self._actor_latent(features, available)
        distribution = self._prior_action_distribution(
            obs, features, latent_pi, self._current_action_mask(obs)
        )
        return (
            self.predict_reward_values(obs, critic_tau),
            self.predict_cost_values(obs, critic_tau),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def evaluate_actor_actions_for_optimization(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("LLM-prior policy requires a tensor observation")
        features = self.extract_features(obs)
        available = self.bs_link_availability(obs)
        latent_pi = self._actor_latent(features, available)
        mask = self._current_action_mask(obs)
        logits = self._prior_masked_action_logits(obs, features, latent_pi, mask)
        distribution = self._distribution_from_masked_logits(logits)
        entropy_sum, entropy_mean = bernoulli_entropy_statistics(logits, mask)
        return distribution.log_prob(actions), entropy_sum, entropy_mean


__all__ = [
    "LLM_PRIOR_ARCHITECTURE",
    "LLM_PRIOR_MASKED_ARCHITECTURE",
    "LLM_PRIOR_UNMASKED_ARCHITECTURE",
    "LLMPriorCRPOActorCriticPolicy",
    "llm_prior_architecture",
]
