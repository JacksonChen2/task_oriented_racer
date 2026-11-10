"""Vanilla matrix-CRPO implemented as a local SB3 PPO subclass."""

from __future__ import annotations

from collections import defaultdict, deque
import copy
from dataclasses import dataclass
import io
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .crpo_buffer import CRPORolloutBuffer
from .crpo_policy import (
    CRPOActorCriticPolicy,
    DUAL_ENCODER_ARCHITECTURE,
    TASK_PHASE_DUAL_CRITIC_ARCHITECTURE,
)
from .backend import MissionEndedError
from .episode_tracker import EpisodeCostTracker


CONSTRAINT_COST_VERSION = "time_weighted_trapezoid_fixed_horizon_v1"


def _gradient_norm(parameters: list[th.nn.Parameter]) -> float:
    gradients = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(th.stack(gradients).norm(2).item())


@dataclass(frozen=True)
class _CRPODecisionRecord:
    """CPU rollout data and identity produced by exactly one inference."""

    proposed_action: np.ndarray
    env_action: np.ndarray
    observation: np.ndarray
    old_log_prob: np.ndarray
    reward_value: np.ndarray
    cost_value: np.ndarray
    critic_tau: np.ndarray
    masked_logits: np.ndarray
    action_version: np.ndarray
    step_id: np.ndarray
    sim_timestamp: np.ndarray
    policy_version: int
    state_version: int
    communication_state_version: int
    update_in_progress: bool
    cycle_status: dict[str, Any]
    inference_started_perf_ns: int
    inference_completed_perf_ns: int
    action_published_perf_ns: int
    inference_started_wall_ns: int
    inference_completed_wall_ns: int
    action_published_wall_ns: int


class _InferenceWorker:
    """Latest-only policy worker with a non-blocking collector handoff.

    The two arrays are swapped under a short condition critical section.  A
    newly observed state replaces an older state that has not started
    inference yet; the state already being evaluated is never mutated.
    """

    def __init__(
        self,
        observation_template: np.ndarray,
        infer: Callable[
            [np.ndarray, dict[str, Any], int], _CRPODecisionRecord
        ],
    ) -> None:
        self._infer = infer
        self._condition = threading.Condition()
        self._pending_observation = np.empty_like(observation_template)
        self._working_observation = np.empty_like(observation_template)
        self._pending_status: dict[str, Any] | None = None
        self._pending_policy_version = 0
        self._pending_token = -1
        self._next_token = 0
        self._results: deque[_CRPODecisionRecord] = deque()
        self._errors: deque[BaseException] = deque()
        self._timelines: dict[int, dict[str, int]] = {}
        self._running = False
        self.replaced_pending_states = 0
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="crpo-policy-inference",
            daemon=True,
        )
        self._thread.start()

    def submit_latest(
        self,
        observation: np.ndarray,
        cycle_status: dict[str, Any],
        policy_version: int,
    ) -> int:
        """Publish the newest state without ever waiting for a free slot."""

        with self._condition:
            token = self._next_token
            if self._pending_status is not None:
                self.replaced_pending_states += 1
                self._timelines.pop(self._pending_token, None)
            np.copyto(self._pending_observation, observation)
            self._pending_token = token
            self._pending_status = dict(cycle_status)
            self._pending_policy_version = int(policy_version)
            self._timelines[token] = {
                "submitted_perf_ns": time.perf_counter_ns(),
                "started_perf_ns": 0,
                "completed_perf_ns": 0,
            }
            self._next_token += 1
            self._condition.notify_all()
            return token

    def drain_results(self) -> list[_CRPODecisionRecord]:
        """Return completed decisions immediately; never wait for inference."""

        with self._condition:
            if self._errors:
                error = self._errors.popleft()
                raise error
            values = list(self._results)
            self._results.clear()
            return values

    def first_result(self) -> _CRPODecisionRecord:
        """Establish the initial behavior policy before collection starts.

        This one-time startup barrier cannot block a *next* inference: there
        is no previous policy decision or trainable transition yet.  Isaac
        and the Proxy remain free-running and their ordered ring absorbs any
        boundaries produced during model startup.
        """

        with self._condition:
            while not self._results and not self._errors:
                self._condition.wait()
            if self._errors:
                raise self._errors.popleft()
            return self._results.popleft()

    def timeline(self, token: int | None) -> dict[str, int]:
        if token is None:
            return {}
        with self._condition:
            return dict(self._timelines.get(token, {}))

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending_status = None
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending_status is None:
                    if self._stopping:
                        return
                    self._condition.wait()
                token = self._pending_token
                cycle_status = dict(self._pending_status)
                policy_version = self._pending_policy_version
                self._pending_status = None
                self._working_observation, self._pending_observation = (
                    self._pending_observation,
                    self._working_observation,
                )
                self._running = True
                started_perf_ns = time.perf_counter_ns()
                self._timelines[token]["started_perf_ns"] = started_perf_ns
                cycle_status["_inference_submitted_perf_ns"] = (
                    self._timelines[token]["submitted_perf_ns"]
                )
            try:
                result = self._infer(
                    self._working_observation, cycle_status, policy_version
                )
                error = None
            except BaseException as caught:
                result = None
                error = caught
            with self._condition:
                completed_perf_ns = time.perf_counter_ns()
                self._timelines[token]["completed_perf_ns"] = (
                    completed_perf_ns
                )
                self._running = False
                if result is not None:
                    self._results.append(result)
                if error is not None:
                    self._errors.append(error)
                self._condition.notify_all()


class _UpdateInferencePump:
    """Keep frozen-policy control alive while the learner updates.

    Update-window transitions are consumed from the ordered runtime ring so
    that every inference sees the latest available state.  They are retained
    only as runtime diagnostics and are never exposed to the PPO buffer.
    """

    def __init__(
        self,
        env: VecEnv,
        observation: np.ndarray,
        policy_version: int,
        infer: Callable[
            [np.ndarray, dict[str, Any], int], _CRPODecisionRecord
        ],
        cycle_status: Callable[[], dict[str, Any]],
    ) -> None:
        self.env = env
        self.policy_version = int(policy_version)
        self._infer = infer
        self._cycle_status = cycle_status
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest_observation = np.asarray(observation).copy()
        self.latest_episode_starts = np.zeros(env.num_envs, dtype=bool)
        self.transition_count = 0
        self.max_action_age = 0.0
        self.mission_ended = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="crpo-update-behavior-pump",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self) -> None:
        self._stop.set()
        self._thread.join()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        with self._lock:
            return (
                self.latest_observation.copy(),
                self.latest_episode_starts.copy(),
                int(self.transition_count),
                float(self.max_action_age),
            )

    @staticmethod
    def _state_key(status: dict[str, Any]) -> tuple[int, int]:
        return (
            int(
                status.get(
                    "consumed_physical_version",
                    status.get("task_step", -1),
                )
            ),
            int(
                status.get(
                    "consumed_communication_version",
                    status.get("rl_decision_index", -1),
                )
            ),
        )

    def _run(self) -> None:
        worker = _InferenceWorker(
            self.latest_observation,
            self._infer,
        )
        latest_decision: _CRPODecisionRecord | None = None
        last_state_key: tuple[int, int] | None = None
        latest_action_age = 0.0
        try:
            initial_status = self._cycle_status()
            initial_status["_update_in_progress"] = True
            initial_status["_previous_action_age"] = latest_action_age
            last_state_key = self._state_key(initial_status)
            worker.submit_latest(
                self.latest_observation,
                initial_status,
                self.policy_version,
            )
            latest_decision = worker.first_result()

            while not self._stop.is_set():
                try:
                    new_obs, _rewards, dones, infos = self.env.step(
                        latest_decision.env_action
                    )
                except MissionEndedError:
                    self.mission_ended = True
                    break
                ages = [float(info.get("action_age", 0.0)) for info in infos]
                latest_action_age = max(ages, default=0.0)
                with self._lock:
                    self.latest_observation = np.asarray(new_obs).copy()
                    self.latest_episode_starts = np.asarray(dones).copy()
                    self.transition_count += self.env.num_envs
                    self.max_action_age = max(
                        self.max_action_age, latest_action_age
                    )
                print(
                    "RACER_RL_UPDATE_TRANSITION "
                    + json.dumps(
                        {
                            "state_version": int(
                                infos[0].get("state_sequence", 0)
                            ),
                            "state_timestamp": float(
                                infos[0].get("simulation_time_s", 0.0)
                            ),
                            "action_id": int(
                                infos[0].get("action_version", 0)
                            ),
                            "step_id": int(infos[0].get("step_id", 0)),
                            "policy_version": int(
                                infos[0].get(
                                    "policy_version", self.policy_version
                                )
                            ),
                            "update_in_progress": True,
                            "action_timestamp": float(
                                infos[0].get("sim_timestamp", 0.0)
                            ),
                            "action_age": latest_action_age,
                            "added_to_rollout_buffer": False,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if np.any(dones):
                    self.mission_ended = True
                    break

                status = self._cycle_status()
                status["_update_in_progress"] = True
                status["_previous_action_age"] = latest_action_age
                state_key = self._state_key(status)
                if state_key != last_state_key:
                    worker.submit_latest(
                        new_obs, status, self.policy_version
                    )
                    last_state_key = state_key
                for completed in worker.drain_results():
                    latest_decision = completed
        except BaseException as caught:
            self.error = caught
        finally:
            worker.close()
            try:
                worker.drain_results()
            except BaseException as caught:
                if self.error is None:
                    self.error = caught


def _bernoulli_log_prob_from_masked_logits(
    masked_logits: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    """Score MultiBinary actions from cached CPU logits without PyTorch."""

    logits = np.asarray(masked_logits, dtype=np.float32)
    action_bits = np.asarray(actions).reshape(logits.shape)
    if logits.ndim != 2:
        raise ValueError(f"expected two-dimensional logits, got {logits.shape}")
    if action_bits.shape != logits.shape:
        raise ValueError(
            f"action shape {action_bits.shape} does not match logits "
            f"{logits.shape}"
        )
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("cached action logits contain NaN or Inf")
    if np.any((action_bits != 0) & (action_bits != 1)):
        raise ValueError("MultiBinary actions must contain only zero or one")
    # A masked Bernoulli bit is a deterministic zero and contributes exactly
    # zero to the joint log probability. Feeding float32.min through a sum is
    # unsafe when a stale, incompatible action is paired with a newer mask.
    masked = logits == np.finfo(np.float32).min
    if np.any(masked & (action_bits != 0)):
        raise FloatingPointError(
            "executed action is incompatible with its behavior-policy mask"
        )
    safe_logits = np.where(masked, np.float32(0.0), logits)
    zero = np.float32(0.0)
    per_bit = np.where(
        action_bits != 0,
        -np.logaddexp(zero, -safe_logits),
        -np.logaddexp(zero, safe_logits),
    )
    per_bit[masked] = zero
    result = per_bit.sum(axis=1, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("cached action log probability is non-finite")
    return result


def select_crpo_mode(j_cost_hat: float, gamma_task: float, eta: float) -> str:
    """Pure switching rule, kept separate for deterministic unit testing."""

    return "reward" if j_cost_hat <= gamma_task + eta else "cost"


class CRPOPPO(PPO):
    """PPO with reward/cost critics and CRPO's mode-switching actor update."""

    policy: CRPOActorCriticPolicy
    rollout_buffer: CRPORolloutBuffer

    def _bootstrap_discount(
        self, gamma: float, info: dict[str, Any]
    ) -> float:
        """Discount one truncated-transition bootstrap target.

        The fixed-clock implementation intentionally returns the configured
        per-transition discount unchanged. Experimental variable-time
        algorithms may override this hook without changing the baseline.
        """

        del info
        return float(gamma)

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
    ) -> "CRPOPPO":
        """Reject pre-dual-encoder checkpoints before SB3 restores weights.

        SB3 contains a compatibility fallback that may retry policy loading
        with ``exact_match=False``.  That behavior is unsafe across this
        deliberate feature-extractor change, so architecture compatibility is
        checked before delegating to SB3's normal loader.
        """

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
        if saved_architecture != DUAL_ENCODER_ARCHITECTURE:
            raise ValueError(
                "incompatible checkpoint architecture: expected "
                f"{DUAL_ENCODER_ARCHITECTURE!r}, got "
                f"{saved_architecture!r}. Legacy single-encoder checkpoints "
                "cannot initialize the Physical State + LLM Guidance dual "
                "encoder; start a new training run."
            )
        policy_state = params.get("policy")
        required_encoder_parameters = {
            "features_extractor.physical_encoder.0.weight",
            "features_extractor.guidance_encoder.0.weight",
            "features_extractor.fusion_encoder.0.weight",
        }
        if not isinstance(policy_state, dict):
            raise ValueError("checkpoint is missing the policy state dictionary")
        missing = sorted(required_encoder_parameters.difference(policy_state))
        if missing:
            raise ValueError(
                "checkpoint architecture metadata names the dual encoder, "
                f"but encoder parameters are missing: {missing}"
            )

        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        model._initialize_policy_runtime(force=True)
        return model

    @staticmethod
    def _migrate_legacy_optimizer_state(
        policy: CRPOActorCriticPolicy,
        saved_optimizer_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract actor moments from the old unified optimizer by name."""

        current = policy.optimizer.state_dict()
        saved_groups = saved_optimizer_state.get("param_groups", [])
        current_groups = current.get("param_groups", [])
        if len(saved_groups) != len(current_groups):
            return current

        legacy_names = [
            name
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        ]
        saved_ids = [
            parameter_id
            for group in saved_groups
            for parameter_id in group.get("params", [])
        ]
        if len(saved_ids) == sum(len(group["params"]) for group in current_groups):
            return saved_optimizer_state
        if len(saved_ids) != len(legacy_names):
            return current
        saved_name_by_id = dict(zip(saved_ids, legacy_names))

        name_by_parameter_id = {
            id(parameter): name for name, parameter in policy.named_parameters()
        }
        current_name_by_id: dict[int, str] = {}
        for optimizer_group, state_group in zip(
            policy.optimizer.param_groups, current_groups
        ):
            for parameter, parameter_id in zip(
                optimizer_group["params"], state_group["params"]
            ):
                name = name_by_parameter_id.get(id(parameter))
                if name is not None:
                    current_name_by_id[int(parameter_id)] = name
        current_id_by_name = {
            name: parameter_id
            for parameter_id, name in current_name_by_id.items()
        }

        migrated_state: dict[int, Any] = {}
        for saved_id, value in saved_optimizer_state.get("state", {}).items():
            name = saved_name_by_id.get(saved_id)
            current_id = current_id_by_name.get(name) if name else None
            if current_id is not None:
                migrated_state[current_id] = value
        migrated_groups = []
        for saved_group, current_group in zip(saved_groups, current_groups):
            migrated_group = dict(saved_group)
            migrated_group["params"] = list(current_group["params"])
            migrated_groups.append(migrated_group)
        return {"state": migrated_state, "param_groups": migrated_groups}

    def set_parameters(
        self,
        load_path_or_dict: str | dict[str, Any],
        exact_match: bool = True,
        device: th.device | str = "auto",
    ) -> None:
        """Migrate pre-phase checkpoints without weakening exact loading."""

        if not isinstance(load_path_or_dict, dict):
            super().set_parameters(load_path_or_dict, exact_match, device)
            return
        policy_state = load_path_or_dict.get("policy")
        legacy_critics = isinstance(policy_state, dict) and not any(
            key.startswith("reward_critic.") for key in policy_state
        )
        migrated = dict(load_path_or_dict)
        saved_optimizer = migrated.get("policy.optimizer")
        missing_critic_optimizers = any(
            name not in migrated
            for name in (
                "policy.reward_critic_optimizer",
                "policy.constraint_critic_optimizer",
            )
        )
        if isinstance(saved_optimizer, dict) and missing_critic_optimizers:
            migrated["policy.optimizer"] = self._migrate_legacy_optimizer_state(
                self.policy, saved_optimizer
            )
        migrated.setdefault(
            "policy.reward_critic_optimizer",
            self.policy.reward_critic_optimizer.state_dict(),
        )
        migrated.setdefault(
            "policy.constraint_critic_optimizer",
            self.policy.constraint_critic_optimizer.state_dict(),
        )
        super().set_parameters(migrated, exact_match, device)
        if missing_critic_optimizers:
            print(
                "RACER_CHECKPOINT_OPTIMIZER_MIGRATION "
                + json.dumps(
                    {
                        "critic_architecture": TASK_PHASE_DUAL_CRITIC_ARCHITECTURE,
                        "legacy_critic_parameters": legacy_critics,
                        "critic_optimizer_state": "fresh",
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

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
        self.policy_architecture_version = DUAL_ENCODER_ARCHITECTURE
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
        # Records are immutable within one rollout. They are cleared at every
        # policy swap so an action from theta_k cannot be scored as theta_k+1.
        self._behavior_decision_records: dict[
            tuple[int, int], _CRPODecisionRecord
        ] = {}
        # BaseAlgorithm.load reconstructs subclasses with a serialized
        # ``policy=...`` keyword. This class always owns the CRPO policy.
        kwargs.pop("policy", None)
        super().__init__(
            CRPOActorCriticPolicy,
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
        # SB3 ``load()`` first constructs with ``_init_setup_model=False`` and
        # restores serialized attributes afterwards. In that path ``n_envs``
        # does not exist yet and the saved tracker must remain authoritative.
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

    @property
    def behavior_policy(self) -> CRPOActorCriticPolicy:
        """Frozen model used by every real-time action inference."""

        self._initialize_policy_runtime()
        return self._behavior_policy

    @property
    def learner_policy(self) -> CRPOActorCriticPolicy:
        """Independent optimizer-owned policy used by PPO/CRPO updates."""

        return self.policy

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Route public inference through the frozen behavior policy too."""

        behavior_policy, _version = self._acquire_behavior_policy()
        try:
            return behavior_policy.predict(
                observation,
                state,
                episode_start,
                deterministic,
            )
        finally:
            self._release_behavior_policy()

    @staticmethod
    def _clone_frozen_policy(
        policy: CRPOActorCriticPolicy,
    ) -> CRPOActorCriticPolicy:
        # SB3 caches the last distribution logits, which may be a non-leaf
        # tensor that PyTorch refuses to deepcopy. Neither that cache nor Adam
        # belongs in a behavior model, so detach both temporarily while making
        # the parameter-disjoint snapshot. This consumes no sampling RNG.
        optimizers = (
            policy.optimizer,
            policy.reward_critic_optimizer,
            policy.constraint_critic_optimizer,
        )
        distribution = getattr(policy.action_dist, "distribution", None)
        policy.optimizer = None  # type: ignore[assignment]
        policy.reward_critic_optimizer = None  # type: ignore[assignment]
        policy.constraint_critic_optimizer = None  # type: ignore[assignment]
        policy.action_dist.distribution = None
        try:
            cloned = copy.deepcopy(policy)
        finally:
            (
                policy.optimizer,
                policy.reward_critic_optimizer,
                policy.constraint_critic_optimizer,
            ) = optimizers
            policy.action_dist.distribution = distribution
        cloned.set_training_mode(False)
        for parameter in cloned.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        if cloned.device.type == "cuda":
            th.cuda.current_stream(cloned.device).synchronize()
        return cloned

    @staticmethod
    def _policy_parameters_are_disjoint(
        first: CRPOActorCriticPolicy,
        second: CRPOActorCriticPolicy,
    ) -> bool:
        first_storage = {
            (parameter.device.type, parameter.device.index, parameter.data_ptr())
            for parameter in first.parameters()
        }
        return all(
            (
                parameter.device.type,
                parameter.device.index,
                parameter.data_ptr(),
            )
            not in first_storage
            for parameter in second.parameters()
        )

    def _initialize_policy_runtime(self, *, force: bool = False) -> None:
        if not hasattr(self, "policy"):
            return
        if force or not hasattr(self, "_behavior_condition"):
            self._behavior_condition = threading.Condition()
            self._behavior_active_inferences = 0
            self._behavior_swap_pending = False
            self._update_runtime_lock = threading.Lock()
            self._update_in_progress = False
            self._update_inference_stats: dict[str, Any] = {}
            self._inference_cuda_stream = (
                th.cuda.Stream(device=self.device, priority=-1)
                if self.device.type == "cuda"
                else None
            )
        if force or not hasattr(self, "_behavior_policy"):
            self._behavior_policy = self._clone_frozen_policy(self.policy)
            self._behavior_policy_version = int(
                getattr(self, "policy_version", 0)
            )
        if not self._policy_parameters_are_disjoint(
            self.policy, self._behavior_policy
        ):
            raise RuntimeError(
                "behavior_policy and learner_policy share parameter storage"
            )

    def _acquire_behavior_policy(
        self,
    ) -> tuple[CRPOActorCriticPolicy, int]:
        self._initialize_policy_runtime()
        with self._behavior_condition:
            while self._behavior_swap_pending:
                self._behavior_condition.wait()
            self._behavior_active_inferences += 1
            return self._behavior_policy, self._behavior_policy_version

    def _release_behavior_policy(self) -> None:
        with self._behavior_condition:
            self._behavior_active_inferences -= 1
            if self._behavior_active_inferences < 0:
                raise RuntimeError("negative active behavior inference count")
            self._behavior_condition.notify_all()

    def _swap_behavior_policy(
        self,
        new_policy: CRPOActorCriticPolicy,
        new_version: int,
    ) -> tuple[int, int]:
        if not self._policy_parameters_are_disjoint(self.policy, new_policy):
            raise RuntimeError(
                "new behavior_policy shares learner parameter storage"
            )
        with self._behavior_condition:
            self._behavior_swap_pending = True
            while self._behavior_active_inferences:
                self._behavior_condition.wait()
            old_version = int(self._behavior_policy_version)
            self._behavior_policy = new_policy
            self._behavior_policy_version = int(new_version)
            self._behavior_swap_pending = False
            self._behavior_condition.notify_all()
        return old_version, int(new_version)

    def _excluded_save_params(self) -> list[str]:
        # Action versions restart from one with every new simulator process;
        # carrying runtime behavior records into the next episode would make
        # their keys collide with unrelated actions.
        return super()._excluded_save_params() + [
            "_behavior_decision_records",
            "_behavior_policy",
            "_behavior_condition",
            "_behavior_active_inferences",
            "_behavior_swap_pending",
            "_behavior_policy_version",
            "_update_runtime_lock",
            "_update_in_progress",
            "_update_inference_stats",
            "_inference_cuda_stream",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return (
            [
                "policy",
                "policy.optimizer",
                "policy.reward_critic_optimizer",
                "policy.constraint_critic_optimizer",
            ],
            [],
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CRPORolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if self._stop_before_next_rollout:
            return False
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_policy_version = int(self.policy_version)
        rollout_buffer.reset()
        n_steps = 0
        callback.on_rollout_start()
        metric_values: dict[str, list[float]] = defaultdict(list)
        partial_rollout = False
        pending_rollout_rows: list[dict[str, Any]] = []
        try:
            decoupled_values = env.env_method(
                "supports_decoupled_transition_collection"
            )
        except (AttributeError, RuntimeError):
            decoupled_values = []
        decoupled_collection = bool(
            decoupled_values and all(bool(value) for value in decoupled_values)
        )
        inference_worker: _InferenceWorker | None = None
        pending_inference_token: int | None = None
        latest_inference_cache: _CRPODecisionRecord | None = None
        decision_records = self._behavior_decision_records

        def remember_decision(record: _CRPODecisionRecord) -> None:
            for env_index, version in enumerate(record.action_version):
                decision_records[(env_index, int(version))] = record
            limit = max(64, 2 * n_rollout_steps * env.num_envs)
            while len(decision_records) > limit:
                del decision_records[next(iter(decision_records))]
        last_submitted_state_key: tuple[int, int] | None = None
        if decoupled_collection:
            inference_worker = _InferenceWorker(
                self._last_obs,
                lambda observation, status, version: (
                    self._perform_policy_inference(
                        observation,
                        status,
                        version,
                        publish_action=True,
                    )
                ),
            )
            initial_status = self._cycle_status()
            last_submitted_state_key = (
                int(
                    initial_status.get(
                        "consumed_physical_version",
                        initial_status.get("task_step", -1),
                    )
                ),
                int(
                    initial_status.get(
                        "consumed_communication_version",
                        initial_status.get("rl_decision_index", -1),
                    )
                ),
            )
            pending_inference_token = inference_worker.submit_latest(
                self._last_obs, initial_status, rollout_policy_version
            )
            initial_decision = inference_worker.first_result()
            latest_inference_cache = initial_decision
            remember_decision(initial_decision)

        while n_steps < n_rollout_steps:
            if inference_worker is None:
                inference_handoff_wait_ms = 0.0
                collector_blocked_on_inference = False
                inference_cache = self._perform_policy_inference(
                    self._last_obs,
                    self._cycle_status(),
                    rollout_policy_version,
                    publish_action=False,
                )
                env.env_method("set_policy_version", rollout_policy_version)
                cycle_status = inference_cache.cycle_status
                clipped_actions = inference_cache.env_action
            else:
                try:
                    completed_inferences = inference_worker.drain_results()
                except MissionEndedError:
                    inference_worker.close()
                    inference_worker = None
                    self.ended_on_mission_boundary = True
                    if n_steps == 0:
                        callback.on_rollout_end()
                        return False
                    partial_rollout = True
                    self._stop_before_next_rollout = True
                    break
                except BaseException:
                    inference_worker.close()
                    raise
                for completed in completed_inferences:
                    latest_inference_cache = completed
                    remember_decision(completed)
                inference_handoff_wait_ms = 0.0
                collector_blocked_on_inference = False
                if latest_inference_cache is None:
                    clipped_actions = np.zeros(
                        (env.num_envs, self.action_space.n), dtype=np.int8
                    )
                    cycle_status = self._cycle_status()
                else:
                    clipped_actions = latest_inference_cache.env_action
                    cycle_status = latest_inference_cache.cycle_status
            try:
                new_obs, rewards, dones, infos = env.step(clipped_actions)
            except MissionEndedError:
                self.ended_on_mission_boundary = True
                if inference_worker is not None:
                    inference_worker.close()
                    inference_worker = None
                if n_steps == 0:
                    callback.on_rollout_end()
                    return False
                # The final transition was already returned and inserted on
                # the preceding iteration.  Finalize the non-empty prefix and
                # train it instead of discarding it because n_steps < 384.
                partial_rollout = True
                self._stop_before_next_rollout = True
                break
            except BaseException:
                if inference_worker is not None:
                    inference_worker.close()
                raise
            if inference_worker is not None:
                # The boundary wait normally leaves ample time for inference
                # to finish.  Draining is a lock-only poll and can never wait
                # for the policy thread.
                completed_inferences = inference_worker.drain_results()
                for completed in completed_inferences:
                    latest_inference_cache = completed
                    remember_decision(completed)
                next_status = self._cycle_status()
                next_state_key = (
                    int(
                        next_status.get(
                            "consumed_physical_version",
                            next_status.get("task_step", -1),
                        )
                    ),
                    int(
                        next_status.get(
                            "consumed_communication_version",
                            next_status.get("rl_decision_index", -1),
                        )
                    ),
                )
                if (
                    n_steps + 1 < n_rollout_steps
                    and not np.any(dones)
                    and next_state_key != last_submitted_state_key
                ):
                    pending_inference_token = (
                        inference_worker.submit_latest(
                            new_obs, next_status, rollout_policy_version
                        )
                    )
                    last_submitted_state_key = next_state_key
            transition_processing_started_ns = time.perf_counter_ns()
            costs = np.asarray(
                [float(info.get("cost", 0.0)) for info in infos], np.float32
            )
            cost_pending = np.asarray(
                [bool(info.get("cost_pending", False)) for info in infos],
                dtype=bool,
            )
            executed_actions_np = np.stack(
                [
                    np.asarray(
                        info.get("executed_action", clipped_actions[index]),
                        dtype=np.int8,
                    )
                    for index, info in enumerate(infos)
                ]
            )
            if inference_worker is not None:
                # Publication precedes construction of the immutable record
                # by only a few CPU instructions.  A second lock-only poll
                # closes that narrow race without waiting on the worker.
                for completed in inference_worker.drain_results():
                    latest_inference_cache = completed
                    remember_decision(completed)
            fallback_cache = (
                inference_cache
                if inference_worker is None
                else latest_inference_cache
            )
            if fallback_cache is None:
                raise RuntimeError(
                    "transition boundary completed before the first policy "
                    "inference; collector did not block, but no cached CPU "
                    "distribution exists for stale-action handling"
                )
            rollout_observations = self._last_obs.copy()
            reward_values_np = fallback_cache.reward_value.copy()
            cost_values_np = fallback_cache.cost_value.copy()
            critic_taus_np = fallback_cache.critic_tau.copy()
            log_probs_np = fallback_cache.old_log_prob.copy()
            masked_logits_np = fallback_cache.masked_logits.copy()
            selected_records: list[_CRPODecisionRecord] = []
            fresh_flags: list[bool] = []
            for index, info in enumerate(infos):
                executed_version = int(info.get("action_version", -1))
                selected = (
                    inference_cache
                    if inference_worker is None
                    else decision_records.get((index, executed_version))
                )
                behavior_match = bool(
                    selected is not None
                    and selected.policy_version == rollout_policy_version
                    and int(info.get("policy_version", 0))
                    == rollout_policy_version
                    and selected.action_version[index] >= 0
                    and executed_version == selected.action_version[index]
                    and np.array_equal(
                        executed_actions_np[index],
                        selected.proposed_action[index],
                    )
                )
                fresh = bool(
                    behavior_match
                    and selected is not None
                    and selected.step_id[index] >= 0
                    and int(info.get("action_source_step_id", -1))
                    == selected.step_id[index]
                    and 0
                    <= int(info.get("step_id", -1))
                    - int(selected.step_id[index])
                    <= 1
                    and float(info.get("action_age", float("inf")))
                    <= float(info.get("rl_decision_interval_s", 0.1))
                    + 1.0e-9
                    and selected.sim_timestamp[index] >= 0.0
                    and np.isclose(
                        float(info.get("action_source_sim_time_s", -1.0)),
                        selected.sim_timestamp[index],
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                )
                # A held action still belongs to the policy record that
                # generated it. A newer observation may carry a different
                # channel mask and assign probability zero to the action.
                if behavior_match:
                    assert selected is not None
                    rollout_observations[index] = selected.observation[index]
                    reward_values_np[index] = selected.reward_value[index]
                    cost_values_np[index] = selected.cost_value[index]
                    critic_taus_np[index] = selected.critic_tau[index]
                    log_probs_np[index] = selected.old_log_prob[index]
                    masked_logits_np[index] = selected.masked_logits[index]
                    selected_records.append(selected)
                else:
                    selected_records.append(fallback_cache)
                fresh_flags.append(fresh)
            fresh_actions = np.asarray(fresh_flags, dtype=bool)
            stale_actions = ~fresh_actions
            inference_cache = selected_records[0]
            cycle_status = inference_cache.cycle_status
            executed_policy_versions = np.asarray(
                [int(info.get("policy_version", 0)) for info in infos],
                dtype=np.int64,
            )
            if decoupled_collection and not np.all(
                executed_policy_versions == rollout_policy_version
            ):
                print(
                    "RACER_RL_TRANSITION_EXCLUDED "
                    + json.dumps(
                        {
                            "reason": "policy_version_mismatch",
                            "rollout_policy_version": rollout_policy_version,
                            "executed_policy_versions": (
                                executed_policy_versions.tolist()
                            ),
                            "action_versions": [
                                int(info.get("action_version", 0))
                                for info in infos
                            ],
                            "state_versions": [
                                int(info.get("state_sequence", 0))
                                for info in infos
                            ],
                            "update_in_progress": False,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                # These boundaries may have been queued while theta_k was
                # controlling during an update. They remain available to
                # runtime metrics but can never enter theta_{k+1}'s buffer.
                self._last_obs = new_obs
                self._last_episode_starts = dones
                continue
            stale_log_prob_from_logits_ms = 0.0
            raw_costs = costs.copy()
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                if inference_worker is not None:
                    inference_worker.close()
                return False
            self._update_info_buffer(infos, dones)
            if not np.any(cost_pending):
                self.episode_cost_tracker.observe(raw_costs, dones, infos)
            n_steps += 1

            if np.any(stale_actions):
                # This is mathematically identical to evaluating the current
                # masked Bernoulli distribution at the held action, but uses
                # only CPU data cached by the first inference. It cannot
                # contend with the next policy forward or synchronize CUDA.
                stale_log_prob_started_ns = time.perf_counter_ns()
                stale_log_probs_np = _bernoulli_log_prob_from_masked_logits(
                    masked_logits_np[stale_actions],
                    executed_actions_np[stale_actions],
                )
                stale_log_prob_from_logits_ms = (
                    time.perf_counter_ns() - stale_log_prob_started_ns
                ) / 1.0e6
                if np.any(fresh_actions):
                    log_probs_np = log_probs_np.copy()
                    log_probs_np[stale_actions] = stale_log_probs_np
                else:
                    log_probs_np = stale_log_probs_np

            for info_index, info in enumerate(infos):
                mapping = {
                    "training/reward": rewards[info_index],
                    "training/C_BS": info.get("C_BS", 0.0),
                    "training/L_task": info.get("L_task", 0.0),
                    "training/D_traj": info.get("D_traj", 0.0),
                    "training/D_cov": info.get("D_cov", 0.0),
                    "training/D_red": info.get("D_red", 0.0),
                    "training/D_map": info.get("D_map", 0.0),
                    "training/action_ones": info.get("action_ones", 0.0),
                    "training/relay_action_ones": info.get(
                        "relay_action_ones", 0.0
                    ),
                    "training/upload_action_ones": info.get(
                        "upload_action_ones", 0.0
                    ),
                    "training/coverage": info.get("coverage", 0.0),
                    "training/episode_length": info.get(
                        "episode_length", 0.0
                    ),
                    "qwen/inference_latency": info.get("qwen_latency", 0.0),
                    "qwen/W_task_mean": info.get("qwen_W_task_mean", 0.0),
                    "qwen/omega_sem_entropy": info.get(
                        "qwen_omega_sem_entropy", 0.0
                    ),
                    "qwen/parse_failures": info.get("qwen_parse_failures", 0.0),
                    "qwen/single_gpu_pause_count": info.get(
                        "qwen_single_gpu_pause_count", 0.0
                    ),
                    "qwen/single_gpu_pause_ack_wait_s": info.get(
                        "qwen_single_gpu_pause_ack_wait_s", 0.0
                    ),
                    "constraint/D_traj": info.get("D_traj", 0.0),
                    "constraint/D_cov": info.get("D_cov", 0.0),
                    "constraint/D_red": info.get("D_red", 0.0),
                    "constraint/D_map": info.get("D_map", 0.0),
                    "constraint/L_task": info.get("L_task", 0.0),
                    "constraint/L_task_previous": info.get(
                        "L_task_previous", 0.0
                    ),
                    "constraint/delta_t_s": info.get("delta_t_s", 0.0),
                    "constraint/L_task_interval_mean": info.get(
                        "tilde_L_task", 0.0
                    ),
                    "constraint/L_task_time_weighted": info.get(
                        "L_task_time_weighted", 0.0
                    ),
                    "constraint/elapsed_time_s": info.get(
                        "constraint_elapsed_time_s", 0.0
                    ),
                    "constraint/Gamma_task": self.gamma_task,
                    "constraint/value": info.get("constraint_value", 0.0),
                    "constraint/step_cost": info.get("cost", 0.0),
                    "constraint/per_transition_cost": info.get(
                        "constraint_cost", 0.0
                    ),
                    "constraint/cumulative_return": info.get(
                        "constraint_return_cumulative", 0.0
                    ),
                    "constraint/violation": info.get("constraint_value", 0.0)
                    - self.gamma_task,
                    "constraint/relay_fanout_proposed_max": info.get(
                        "relay_fanout_proposed_max", 0.0
                    ),
                    "constraint/relay_fanout_executed_max": info.get(
                        "relay_fanout_executed_max", 0.0
                    ),
                    "constraint/relay_fanout_violating_senders": info.get(
                        "relay_fanout_violating_senders", 0.0
                    ),
                    "constraint/relay_fanout_projection_drops": info.get(
                        "relay_fanout_projection_drops", 0.0
                    ),
                    "task/coverage": info.get("coverage", 0.0),
                    "task/coverage_pc": info.get("coverage_pc", 0.0),
                    "task/redundancy": info.get("redundancy", 0.0),
                    "task/redundancy_pc": info.get("redundancy_pc", 0.0),
                    "task/map_iou": info.get("map_iou", 0.0),
                    "task/map_iou_pc": info.get("map_iou_pc", 0.0),
                    "task/bs_global_map_coverage": info.get(
                        "bs_global_map_coverage", 0.0
                    ),
                    "task/C_joint_coverage": info.get(
                        "C_joint_map", 0.0
                    ),
                    "task/C_BS_coverage": info.get("C_BS_map", 0.0),
                    "task/task_step": info.get("task_step", 0.0),
                    "task/task_time_s": info.get("task_time_s", 0.0),
                    "task/trajectory_deviation": info.get(
                        "trajectory_deviation_m", 0.0
                    ),
                    "communication/C_BS": info.get("C_BS", 0.0),
                    "communication/C_no_link": info.get(
                        "C_no_link", 0.0
                    ),
                    "communication/C_reward_resource": info.get(
                        "C_reward_resource", info.get("C_BS", 0.0)
                    ),
                    "communication/C_U2U": info.get("C_U2U", 0.0),
                }
                if cost_pending[info_index]:
                    # Async task metrics are exact only after the existing
                    # rollout-end backfill. Avoid averaging placeholders with
                    # the resolved constraint statistics below.
                    for name in (
                        "training/L_task",
                        "constraint/L_task",
                        "constraint/L_task_previous",
                        "constraint/delta_t_s",
                        "constraint/L_task_interval_mean",
                        "constraint/L_task_time_weighted",
                        "constraint/elapsed_time_s",
                        "constraint/value",
                        "constraint/step_cost",
                        "constraint/per_transition_cost",
                        "constraint/cumulative_return",
                        "constraint/violation",
                    ):
                        mapping.pop(name, None)
                if dones[info_index] and not cost_pending[info_index]:
                    mapping.update(
                        {
                            "episode/L_task_final": info.get("L_task", 0.0),
                            "episode/L_task_time_weighted": info.get(
                                "L_task_time_weighted", 0.0
                            ),
                            "episode/constraint_return": info.get(
                                "constraint_return_cumulative", 0.0
                            ),
                            "episode/sum_CRPO_cost": (
                                self.episode_cost_tracker.records[-1].sum_cost
                                if self.episode_cost_tracker.records
                                else 0.0
                            ),
                            "episode/C_BS_total": (
                                self.episode_cost_tracker.records[-1].total_bs_resource
                                if self.episode_cost_tracker.records
                                else 0.0
                            ),
                            "episode/constraint_elapsed_time_s": (
                                self.episode_cost_tracker.records[
                                    -1
                                ].constraint_elapsed_time_s
                                if self.episode_cost_tracker.records
                                else 0.0
                            ),
                        }
                    )
                for name, value in mapping.items():
                    metric_values[name].append(float(value))

            actions_np = executed_actions_np
            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)
            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[index]["terminal_observation"]
                    )[0]
                    terminal_tau = th.as_tensor(
                        [self._critic_tau_from_info(infos[index])],
                        device=self.device,
                        dtype=terminal_obs.dtype,
                    )
                    with th.no_grad():
                        rewards[index] += (
                            self._bootstrap_discount(
                                self.gamma, infos[index]
                            )
                            * self.policy.predict_reward_values(
                                terminal_obs, terminal_tau
                            )[0]
                        )
                        costs[index] += (
                            self._bootstrap_discount(
                                self.gamma_cost, infos[index]
                            )
                            * self.policy.predict_cost_values(
                                terminal_obs, terminal_tau
                            )[0]
                        )

            buffer_position = rollout_buffer.pos
            rollout_buffer.add(
                rollout_observations,
                actions_np,
                rewards,
                costs,
                self._last_episode_starts,
                reward_values_np,
                cost_values_np,
                log_probs_np,
                critic_tau=critic_taus_np,
                action_version=np.asarray(
                    [info.get("action_version", 0) for info in infos],
                    dtype=np.int64,
                ),
                policy_version=np.asarray(
                    [record.policy_version for record in selected_records],
                    dtype=np.int64,
                ),
                action_age=np.asarray(
                    [info.get("action_age", 0.0) for info in infos],
                    dtype=np.float64,
                ),
                sim_timestamp=np.asarray(
                    [info.get("sim_timestamp", 0.0) for info in infos],
                    dtype=np.float64,
                ),
                step_id=np.asarray(
                    [info.get("cost_step_id", info.get("step_id", 0))
                     for info in infos],
                    dtype=np.int64,
                ),
                sim_time=np.asarray(
                    [info.get("transition_start_sim_time_s", 0.0)
                     for info in infos],
                    dtype=np.float64,
                ),
                state_version=np.asarray(
                    [record.state_version for record in selected_records],
                    dtype=np.int64,
                ),
                communication_state_version=np.asarray(
                    [
                        record.communication_state_version
                        for record in selected_records
                    ],
                    dtype=np.int64,
                ),
                update_in_progress=np.asarray(
                    [record.update_in_progress for record in selected_records],
                    dtype=bool,
                ),
                action_timestamp=np.asarray(
                    [
                        record.action_published_wall_ns
                        for record in selected_records
                    ],
                    dtype=np.int64,
                ),
                state_timestamp=np.asarray(
                    [
                        float(record.cycle_status.get("sim_time_s", 0.0))
                        for record in selected_records
                    ],
                    dtype=np.float64,
                ),
                cost_ready=~cost_pending,
            )
            if np.any(cost_pending):
                pending_rollout_rows.append(
                    {
                        "position": buffer_position,
                        "pending": cost_pending.copy(),
                        "step_ids": np.asarray(
                            [info.get("cost_step_id", 0) for info in infos],
                            dtype=np.int64,
                        ),
                        # Only the time-limit critic bootstrap has been added
                        # to placeholder costs at this point.
                        "bootstrap": costs.copy(),
                        "dones": dones.copy(),
                        "infos": [dict(info) for info in infos],
                    }
                )
            transition_processing_completed_ns = time.perf_counter_ns()
            cycle_completed_perf_ns = transition_processing_completed_ns
            cycle_completed_wall_time_ns = time.time_ns()
            transition_processing_ms = (
                transition_processing_completed_ns
                - transition_processing_started_ns
            ) / 1.0e6
            rl_cycle_ms = (
                cycle_completed_perf_ns
                - inference_cache.inference_started_perf_ns
            ) / 1.0e6
            next_inference_timeline = (
                inference_worker.timeline(pending_inference_token)
                if inference_worker is not None
                else {}
            )
            next_inference_started_ns = int(
                next_inference_timeline.get("started_perf_ns", 0)
            )
            next_inference_completed_ns = int(
                next_inference_timeline.get("completed_perf_ns", 0)
            )
            next_inference_overlap_ms = 0.0
            if next_inference_started_ns > 0:
                overlap_end_ns = min(
                    transition_processing_completed_ns,
                    next_inference_completed_ns
                    if next_inference_completed_ns > 0
                    else transition_processing_completed_ns,
                )
                next_inference_overlap_ms = max(
                    0.0,
                    (
                        overlap_end_ns
                        - max(
                            transition_processing_started_ns,
                            next_inference_started_ns,
                        )
                    )
                    / 1.0e6,
                )
            next_inference_not_started_during_finalize = bool(
                pending_inference_token is not None
                and next_inference_started_ns == 0
            )
            # The collector never acquires a policy lock. A sub-millisecond
            # finalize may finish before the OS schedules the worker, which
            # means no overlap was observed, not that inference was blocked.
            inference_blocked_by_finalize = False
            transition_kind = (
                "fresh"
                if np.all(fresh_actions)
                else "stale"
                if not np.any(fresh_actions)
                else "mixed"
            )
            metric_values["timing/rl_cycle_ms"].append(rl_cycle_ms)
            metric_values["timing/policy_inference_ms"].append(
                (
                    inference_cache.inference_completed_perf_ns
                    - inference_cache.inference_started_perf_ns
                )
                / 1.0e6
            )
            metric_values["timing/inference_handoff_wait_ms"].append(
                inference_handoff_wait_ms
            )
            metric_values["timing/collector_blocked_on_inference"].append(
                float(collector_blocked_on_inference)
            )
            metric_values["timing/transition_finalize_ms"].append(
                transition_processing_ms
            )
            if np.any(stale_actions):
                metric_values[
                    "timing/stale_log_prob_from_cached_logits_ms"
                ].append(stale_log_prob_from_logits_ms)
            metric_values["timing/next_inference_overlap_ms"].append(
                next_inference_overlap_ms
            )
            metric_values[
                "timing/inference_blocked_by_transition_finalize"
            ].append(float(inference_blocked_by_finalize))
            metric_values[
                "timing/next_inference_not_started_during_finalize"
            ].append(float(next_inference_not_started_during_finalize))
            if transition_kind == "fresh":
                metric_values["timing/fresh_transition_processing_ms"].append(
                    transition_processing_ms
                )
            else:
                metric_values["timing/stale_transition_processing_ms"].append(
                    transition_processing_ms
                )
            transition_log = {
                "rl_cycle_id": cycle_status.get("rl_cycle_id", 0),
                "cycle_started_wall_time_ns": (
                    inference_cache.inference_started_wall_ns
                ),
                "policy_completed_wall_time_ns": (
                    inference_cache.inference_completed_wall_ns
                ),
                "cycle_completed_wall_time_ns": cycle_completed_wall_time_ns,
                "policy_inference_latency_ms": (
                    inference_cache.inference_completed_perf_ns
                    - inference_cache.inference_started_perf_ns
                )
                / 1.0e6,
                "decoupled_inference_collector": decoupled_collection,
                "collector_inference_handoff_wait_ms": (
                    inference_handoff_wait_ms
                ),
                "collector_blocked_on_inference": (
                    collector_blocked_on_inference
                ),
                "transition_finalize_started_perf_ns": (
                    transition_processing_started_ns
                ),
                "transition_finalize_completed_perf_ns": (
                    transition_processing_completed_ns
                ),
                "next_inference_started_perf_ns": next_inference_started_ns,
                "next_inference_completed_perf_ns": (
                    next_inference_completed_ns
                ),
                "next_inference_overlap_ms": next_inference_overlap_ms,
                "next_inference_overlap_observed": bool(
                    next_inference_overlap_ms > 0.0
                ),
                "inference_blocked_by_transition_finalize": (
                    inference_blocked_by_finalize
                ),
                "next_inference_not_started_during_finalize": (
                    next_inference_not_started_during_finalize
                ),
                "transition_waited_for_inference_ms": 0.0,
                "transition_kind": transition_kind,
                "fresh_transition_count": int(np.count_nonzero(fresh_actions)),
                "stale_transition_count": int(np.count_nonzero(stale_actions)),
                "second_policy_forward": False,
                "stale_policy_forward_latency_ms": 0.0,
                "stale_log_prob_from_cached_logits_ms": (
                    stale_log_prob_from_logits_ms
                ),
                "transition_processing_latency_ms": transition_processing_ms,
                "transition_finalize_latency_ms": transition_processing_ms,
                "fresh_transition_processing_latency_ms": (
                    transition_processing_ms
                    if transition_kind == "fresh"
                    else None
                ),
                "stale_transition_processing_latency_ms": (
                    transition_processing_ms
                    if transition_kind != "fresh"
                    else None
                ),
                "rl_cycle_latency_ms": rl_cycle_ms,
                "proposed_action_version": int(
                    inference_cache.action_version[0]
                ),
                "proposed_step_id": int(inference_cache.step_id[0]),
                "proposed_sim_timestamp": float(
                    inference_cache.sim_timestamp[0]
                ),
                "source_physical_version": cycle_status.get(
                    "consumed_physical_version", 0
                ),
                "source_communication_version": cycle_status.get(
                    "consumed_communication_version", 0
                ),
                "source_guidance_id": cycle_status.get("guidance_id", 0),
                "task_step": int(infos[0].get("step_id", 0)),
                "task_time_s": float(
                    infos[0].get("simulation_time_s", 0.0)
                ),
                "bs_global_map_coverage": float(
                    infos[0].get("bs_global_map_coverage", 0.0)
                ),
                "simulation_time_s": float(
                    infos[0].get("simulation_time_s", 0.0)
                ),
                "communication_slot_index": int(
                    infos[0].get("communication_slot_index", 0)
                ),
                "rl_decision_index": int(
                    infos[0].get("rl_decision_index", 0)
                ),
                "action_held_slots": int(
                    infos[0].get("action_held_slots", 0)
                ),
                "action_version": int(infos[0].get("action_version", 0)),
                "policy_version": int(infos[0].get("policy_version", 0)),
                "action_age": float(infos[0].get("action_age", 0.0)),
                "sim_timestamp": float(
                    infos[0].get("sim_timestamp", 0.0)
                ),
                "rollout_buffer_size": rollout_buffer.valid_size,
                "rollout_buffer_capacity": n_rollout_steps,
                "transition_sim_time_s": float(
                    infos[0].get("transition_sim_time_s", 0.0)
                ),
                "done": bool(dones[0]),
            }
            print(
                "RACER_RL_TRANSITION "
                + json.dumps(transition_log, separators=(",", ":")),
                flush=True,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            if np.any(dones):
                # A four-process campaign owns one finite Isaac mission.  Its
                # VecEnv reset can expose the final snapshot again while the
                # producer shuts down, so close the rollout at the first real
                # terminal transition instead of recording that snapshot as
                # a sequence of one-step episodes.
                self.ended_on_mission_boundary = True
                partial_rollout = n_steps < n_rollout_steps
                self._stop_before_next_rollout = True
                break

        if inference_worker is not None:
            inference_worker.close()
            inference_worker = None

        # Task metrics are causally keyed by producer step_id and backfilled
        # only after action collection stops. The actor/proxy never waits;
        # CRPO GAE and network updates explicitly refuse pending costs below.
        if pending_rollout_rows:
            resolved: dict[tuple[int, int], dict[str, Any]] = {}
            for env_index in range(env.num_envs):
                selected = [
                    row for row in pending_rollout_rows
                    if bool(row["pending"][env_index])
                ]
                if not selected:
                    continue
                step_ids = [
                    int(row["step_ids"][env_index]) for row in selected
                ]
                values = env.env_method(
                    "resolve_pending_costs", step_ids, indices=env_index
                )[0]
                if len(values) != len(selected):
                    raise RuntimeError("task cost backfill returned wrong length")
                for row, value in zip(selected, values):
                    resolved[(int(row["position"]), env_index)] = value
            for row in pending_rollout_rows:
                row_costs = rollout_buffer.costs[int(row["position"])].copy()
                tracker_costs = row_costs.copy()
                for env_index in range(env.num_envs):
                    if not bool(row["pending"][env_index]):
                        continue
                    value = resolved[(int(row["position"]), env_index)]
                    row["infos"][env_index].update(value)
                    backfilled = float(value["cost"]) + float(
                        row["bootstrap"][env_index]
                    )
                    rollout_buffer.backfill_cost(
                        int(row["position"]), env_index, backfilled
                    )
                    row_costs[env_index] = backfilled
                    tracker_costs[env_index] = float(value["cost"])
                    for name, key in (
                        ("training/L_task", "L_task"),
                        ("training/D_traj", "D_traj"),
                        ("training/D_cov", "D_cov"),
                        ("training/D_red", "D_red"),
                        ("training/D_map", "D_map"),
                        ("task/C_joint_coverage", "C_joint_map"),
                        ("task/C_BS_coverage", "C_BS_map"),
                        ("constraint/step_cost", "cost"),
                        ("constraint/L_task", "L_task"),
                        (
                            "constraint/L_task_previous",
                            "L_task_previous",
                        ),
                        ("constraint/delta_t_s", "delta_t_s"),
                        (
                            "constraint/L_task_interval_mean",
                            "tilde_L_task",
                        ),
                        ("constraint/value", "constraint_value"),
                        (
                            "constraint/per_transition_cost",
                            "constraint_cost",
                        ),
                        (
                            "constraint/cumulative_return",
                            "constraint_return_cumulative",
                        ),
                        (
                            "constraint/L_task_time_weighted",
                            "L_task_time_weighted",
                        ),
                        (
                            "constraint/elapsed_time_s",
                            "constraint_elapsed_time_s",
                        ),
                    ):
                        metric_values[name].append(float(value.get(key, 0.0)))
                    metric_values["constraint/violation"].append(
                        float(value["constraint_value"]) - self.gamma_task
                    )
                self.episode_cost_tracker.observe(
                    tracker_costs, row["dones"], row["infos"]
                )
                for env_index, done in enumerate(row["dones"]):
                    if not done or not bool(row["pending"][env_index]):
                        continue
                    record = self.episode_cost_tracker.records[-1]
                    metric_values["episode/L_task_final"].append(
                        record.final_task_loss
                    )
                    metric_values["episode/L_task_time_weighted"].append(
                        record.time_weighted_task_loss
                    )
                    metric_values["episode/constraint_return"].append(
                        record.sum_cost
                    )
                    metric_values["episode/sum_CRPO_cost"].append(
                        record.sum_cost
                    )
                    metric_values["episode/C_BS_total"].append(
                        record.total_bs_resource
                    )
                    metric_values["episode/constraint_elapsed_time_s"].append(
                        record.constraint_elapsed_time_s
                    )
            rollout_buffer.assert_costs_ready()

        with th.no_grad():
            last_obs = obs_as_tensor(new_obs, self.device)
            last_critic_taus = th.as_tensor(
                [self._critic_tau_from_info(info) for info in infos],
                device=self.device,
                dtype=last_obs.dtype,
            )
            values = self.policy.predict_reward_values(
                last_obs, last_critic_taus
            )
            cost_values = self.policy.predict_cost_values(
                last_obs, last_critic_taus
            )
        self._last_cost_value_estimate = cost_values.cpu().numpy().flatten()
        rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            last_cost_values=cost_values,
            dones=dones,
        )
        if partial_rollout:
            self.partial_rollout_updates += 1
        self.logger.record("rollout/buffer_size", rollout_buffer.valid_size)
        self.logger.record("rollout/is_partial", int(partial_rollout))
        rollout_rewards = rollout_buffer.rewards[:rollout_buffer.valid_size]
        self.logger.record("rollout/reward_sum", float(np.sum(rollout_rewards)))
        self.logger.record("rollout/reward_mean", float(np.mean(rollout_rewards)))
        for timing_name in (
            "timing/fresh_transition_processing_ms",
            "timing/stale_transition_processing_ms",
            "timing/rl_cycle_ms",
            "timing/policy_inference_ms",
            "timing/inference_handoff_wait_ms",
            "timing/collector_blocked_on_inference",
            "timing/transition_finalize_ms",
            "timing/next_inference_overlap_ms",
            "timing/inference_blocked_by_transition_finalize",
        ):
            samples = metric_values.get(timing_name, [])
            # Preserve the full TensorBoard keys while avoiding collisions in
            # SB3's fixed-width stdout formatter, which truncates long names.
            self.logger.record(
                f"{timing_name}_count", len(samples), exclude="stdout"
            )
            if samples:
                self.logger.record(
                    f"{timing_name}_max",
                    max(samples),
                    exclude="stdout",
                )
        for name, values_list in metric_values.items():
            if values_list:
                self.logger.record(name, float(np.mean(values_list)))
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _constraint_estimate(self) -> tuple[float, str]:
        tracker = self.episode_cost_tracker
        if self.constraint_estimator == "time_weighted_mean":
            active = tracker.lengths > 0
            if np.any(active):
                return (
                    float(np.mean(tracker.constraint_values[active])),
                    "partial_time_weighted_mean",
                )
            if tracker.records:
                return (
                    float(tracker.records[-1].time_weighted_task_loss),
                    "time_weighted_mean",
                )
            return 0.0, "partial_time_weighted_mean"
        if self.constraint_estimator == "mean_step_cost":
            if tracker.records:
                values = [
                    item.sum_cost / max(1, item.episode_length)
                    for item in tracker.records
                ]
                return float(np.mean(values)), "mean_step_cost"
            active = tracker.lengths > 0
            if np.any(active):
                return float(
                    np.mean(tracker.costs[active] / tracker.lengths[active])
                ), "partial_mean_step_cost"
            return 0.0, "partial_mean_step_cost"
        if self.constraint_estimator == "episode_return" and tracker.records:
            return (
                float(np.mean([item.final_task_loss for item in tracker.records])),
                "episode_return",
            )
        estimate = tracker.initial_losses + tracker.costs
        if self._last_cost_value_estimate is not None:
            estimate = estimate + self._last_cost_value_estimate
        return float(np.mean(estimate)), "critic_estimate"

    def _synchronization_status(self) -> dict[str, Any]:
        if self.env is None:
            return {"enabled": False}
        try:
            values = self.env.env_method("synchronization_status")
        except (AttributeError, RuntimeError):
            return {"enabled": False}
        if not values or not isinstance(values[0], dict):
            return {"enabled": False}
        return dict(values[0])

    def _cycle_status(self) -> dict[str, Any]:
        if self.env is None:
            return {}
        try:
            values = self.env.env_method("cycle_status")
        except (AttributeError, RuntimeError):
            return self._synchronization_status()
        if not values or not isinstance(values[0], dict):
            return {}
        return dict(values[0])

    @staticmethod
    def _normalized_critic_tau(
        elapsed_episode_time: float, episode_duration: float
    ) -> float:
        if not np.isfinite(elapsed_episode_time):
            raise FloatingPointError("episode elapsed time is NaN or Inf")
        if not np.isfinite(episode_duration) or episode_duration <= 0.0:
            raise ValueError("episode duration must be finite and positive")
        return float(np.clip(elapsed_episode_time / episode_duration, 0.0, 1.0))

    @classmethod
    def _critic_tau_from_cycle_status(
        cls, cycle_status: dict[str, Any], batch_size: int
    ) -> np.ndarray:
        elapsed = float(
            cycle_status.get(
                "elapsed_episode_time_s",
                cycle_status.get(
                    "task_time_s", cycle_status.get("sim_time_s", 0.0)
                ),
            )
        )
        duration = float(cycle_status.get("episode_duration_s", 300.0))
        tau = cls._normalized_critic_tau(elapsed, duration)
        return np.full(batch_size, tau, dtype=np.float32)

    @classmethod
    def _critic_tau_from_info(cls, info: dict[str, Any]) -> float:
        elapsed = float(
            info.get(
                "simulation_time_s",
                info.get("constraint_elapsed_time_s", 0.0),
            )
        )
        duration = float(
            info.get("constraint_episode_duration_s", 300.0)
        )
        return cls._normalized_critic_tau(elapsed, duration)

    def _perform_policy_inference(
        self,
        observation: np.ndarray,
        cycle_status: dict[str, Any],
        policy_version: int,
        *,
        publish_action: bool,
    ) -> _CRPODecisionRecord:
        """Run one forward and optionally publish without waiting for state."""

        behavior_policy, behavior_policy_version = (
            self._acquire_behavior_policy()
        )
        inference_started_perf_ns = time.perf_counter_ns()
        inference_started_wall_ns = time.time_ns()
        with self._update_runtime_lock:
            update_in_progress = bool(self._update_in_progress)
        if int(policy_version) != int(behavior_policy_version):
            self._release_behavior_policy()
            raise RuntimeError(
                "inference request policy version does not match frozen "
                f"behavior policy: requested={policy_version}, "
                f"behavior={behavior_policy_version}"
            )
        proposed_action_version = int(
            cycle_status.get("next_action_version", -1)
        )
        proposed_step_id = int(
            cycle_status.get(
                "task_step", cycle_status.get("rl_decision_index", -1)
            )
        )
        critic_tau = self._critic_tau_from_cycle_status(
            cycle_status, observation.shape[0]
        )
        print(
            "RACER_RL_CYCLE_BEGIN "
            + json.dumps(
                {
                    "wall_time_ns": inference_started_wall_ns,
                    "perf_time_ns": inference_started_perf_ns,
                    "rl_cycle_id": cycle_status.get("rl_cycle_id", 0),
                    "source_physical_version": cycle_status.get(
                        "consumed_physical_version", 0
                    ),
                    "source_communication_version": cycle_status.get(
                        "consumed_communication_version", 0
                    ),
                    "channel_version": cycle_status.get(
                        "channel_version",
                        cycle_status.get("consumed_communication_version", 0),
                    ),
                    "source_guidance_id": cycle_status.get("guidance_id", 0),
                    "source_sim_time_s": cycle_status.get("sim_time_s"),
                    "proposed_action_version": proposed_action_version,
                    "proposed_step_id": proposed_step_id,
                    "decoupled_inference_worker": publish_action,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        try:
            stream_context = (
                th.cuda.stream(self._inference_cuda_stream)
                if self._inference_cuda_stream is not None
                else None
            )
            if stream_context is None:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(
                        observation, behavior_policy.device
                    )
                    critic_tau_tensor = th.as_tensor(
                        critic_tau,
                        device=behavior_policy.device,
                        dtype=obs_tensor.dtype,
                    )
                    (
                        actions,
                        values,
                        cost_values,
                        log_probs,
                        masked_logits,
                    ) = behavior_policy.forward_crpo_with_masked_logits(
                        obs_tensor, critic_tau=critic_tau_tensor
                    )
                    action_columns = int(
                        actions.reshape(actions.shape[0], -1).shape[1]
                    )
                    rollout_cpu = th.cat(
                        (
                            actions.reshape(actions.shape[0], action_columns),
                            values.reshape(actions.shape[0], 1),
                            cost_values.reshape(actions.shape[0], 1),
                            log_probs.reshape(actions.shape[0], 1),
                            masked_logits.reshape(
                                actions.shape[0], action_columns
                            ),
                        ),
                        dim=1,
                    ).detach().cpu().numpy()
            else:
                with stream_context, th.no_grad():
                    obs_tensor = obs_as_tensor(
                        observation, behavior_policy.device
                    )
                    critic_tau_tensor = th.as_tensor(
                        critic_tau,
                        device=behavior_policy.device,
                        dtype=obs_tensor.dtype,
                    )
                    (
                        actions,
                        values,
                        cost_values,
                        log_probs,
                        masked_logits,
                    ) = behavior_policy.forward_crpo_with_masked_logits(
                        obs_tensor, critic_tau=critic_tau_tensor
                    )
                    action_columns = int(
                        actions.reshape(actions.shape[0], -1).shape[1]
                    )
                    rollout_cpu = th.cat(
                        (
                            actions.reshape(actions.shape[0], action_columns),
                            values.reshape(actions.shape[0], 1),
                            cost_values.reshape(actions.shape[0], 1),
                            log_probs.reshape(actions.shape[0], 1),
                            masked_logits.reshape(
                                actions.shape[0], action_columns
                            ),
                        ),
                        dim=1,
                    ).detach().cpu().numpy()
        finally:
            # The swap condition can proceed only after the complete forward
            # and device-to-host copy have crossed this safe boundary.
            self._release_behavior_policy()
        inference_completed_perf_ns = time.perf_counter_ns()
        inference_completed_wall_ns = time.time_ns()
        proposed_action = rollout_cpu[:, :action_columns].reshape(
            tuple(actions.shape)
        )
        env_action = proposed_action
        if isinstance(self.action_space, spaces.Box):
            if behavior_policy.squash_output:
                env_action = behavior_policy.unscale_action(env_action)
            else:
                env_action = np.clip(
                    env_action, self.action_space.low, self.action_space.high
                )
        action_versions = np.full(
            actions.shape[0], proposed_action_version, dtype=np.int64
        )
        step_ids = np.full(
            actions.shape[0], proposed_step_id, dtype=np.int64
        )
        sim_timestamps = np.full(
            actions.shape[0],
            float(cycle_status.get("sim_time_s", -1.0)),
            dtype=np.float64,
        )
        publish_started_perf_ns = time.perf_counter_ns()
        if publish_action:
            if self.env is None:
                raise RuntimeError("cannot publish an action without an env")
            for env_index in range(actions.shape[0]):
                publication = self.env.env_method(
                    "publish_action",
                    env_action[env_index],
                    policy_version,
                    cycle_status,
                    indices=env_index,
                )[0]
                action_versions[env_index] = int(
                    publication["action_version"]
                )
                step_ids[env_index] = int(publication["step_id"])
                sim_timestamps[env_index] = float(
                    publication["sim_timestamp"]
                )
        action_published_perf_ns = time.perf_counter_ns()
        action_published_wall_ns = time.time_ns()
        inference_latency_ms = (
            inference_completed_perf_ns - inference_started_perf_ns
        ) / 1.0e6
        action_age = float(cycle_status.get("_previous_action_age", 0.0))
        state_version = int(
            cycle_status.get(
                "consumed_physical_version",
                cycle_status.get("task_step", -1),
            )
        )
        communication_state_version = int(
            cycle_status.get(
                "consumed_communication_version",
                cycle_status.get("rl_decision_index", -1),
            )
        )
        with self._update_runtime_lock:
            if self._update_in_progress:
                stats = self._update_inference_stats
                stats["actions_generated"] = int(
                    stats.get("actions_generated", 0)
                ) + int(actions.shape[0])
                stats["max_action_age"] = max(
                    float(stats.get("max_action_age", 0.0)), action_age
                )
                stats["inference_latency_sum_ms"] = float(
                    stats.get("inference_latency_sum_ms", 0.0)
                ) + inference_latency_ms
                stats["inference_latency_max_ms"] = max(
                    float(stats.get("inference_latency_max_ms", 0.0)),
                    inference_latency_ms,
                )
                stats.setdefault("state_versions", set()).add(
                    (state_version, communication_state_version)
                )
        print(
            "[RL ACT] "
            + json.dumps(
                {
                    "state_version": state_version,
                    "communication_state_version": (
                        communication_state_version
                    ),
                    "channel_version": communication_state_version,
                    "state_timestamp": cycle_status.get("sim_time_s"),
                    "action_id": int(action_versions[0]),
                    "step_id": int(step_ids[0]),
                    "policy_version": int(behavior_policy_version),
                    "update_in_progress": update_in_progress,
                    "action_timestamp": action_published_wall_ns,
                    "inference_ms": inference_latency_ms,
                    "action_age": action_age,
                    "uses_learner_parameters": False,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        print(
            "RACER_RL_INFERENCE "
            + json.dumps(
                {
                    "rl_cycle_id": cycle_status.get("rl_cycle_id", 0),
                    "step_id": int(step_ids[0]),
                    "action_version": int(action_versions[0]),
                    "inference_started_perf_ns": inference_started_perf_ns,
                    "inference_completed_perf_ns": (
                        inference_completed_perf_ns
                    ),
                    "action_published_perf_ns": action_published_perf_ns,
                    "inference_started_wall_time_ns": (
                        inference_started_wall_ns
                    ),
                    "inference_completed_wall_time_ns": (
                        inference_completed_wall_ns
                    ),
                    "inference_latency_ms": inference_latency_ms,
                    "policy_version": int(behavior_policy_version),
                    "update_in_progress": update_in_progress,
                    "action_publish_latency_ms": (
                        action_published_perf_ns - publish_started_perf_ns
                    )
                    / 1.0e6,
                    "inference_queue_latency_ms": max(
                        0.0,
                        (
                            inference_started_perf_ns
                            - int(
                                cycle_status.get(
                                    "_inference_submitted_perf_ns",
                                    inference_started_perf_ns,
                                )
                            )
                        )
                        / 1.0e6,
                    ),
                    "waited_for_action_ack": False,
                    "waited_for_next_state": False,
                    "request_ring_blocked": False,
                    "blocked_on_transition_lock": False,
                    "decoupled_worker": publish_action,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return _CRPODecisionRecord(
            proposed_action=proposed_action,
            env_action=env_action,
            observation=np.asarray(observation).copy(),
            reward_value=rollout_cpu[:, action_columns],
            cost_value=rollout_cpu[:, action_columns + 1],
            critic_tau=critic_tau.copy(),
            old_log_prob=rollout_cpu[:, action_columns + 2],
            masked_logits=rollout_cpu[
                :, action_columns + 3 : action_columns * 2 + 3
            ],
            action_version=action_versions,
            step_id=step_ids,
            sim_timestamp=sim_timestamps,
            policy_version=int(behavior_policy_version),
            state_version=state_version,
            communication_state_version=communication_state_version,
            update_in_progress=update_in_progress,
            cycle_status=dict(cycle_status),
            inference_started_perf_ns=inference_started_perf_ns,
            inference_completed_perf_ns=inference_completed_perf_ns,
            action_published_perf_ns=action_published_perf_ns,
            inference_started_wall_ns=inference_started_wall_ns,
            inference_completed_wall_ns=inference_completed_wall_ns,
            action_published_wall_ns=action_published_wall_ns,
        )

    def _train_learner_policy(self) -> dict[str, Any]:
        """Run the unchanged PPO/CRPO mathematics on learner_policy only."""

        update_started_perf_ns = time.perf_counter_ns()
        update_started_wall_ns = time.time_ns()
        rollout_size = self.rollout_buffer.valid_size
        self.rollout_buffer.assert_finite()
        rollout_policy_versions = self.rollout_buffer.policy_versions[
            :rollout_size
        ]
        expected_policy_version = int(self.policy_version)
        if rollout_policy_versions.size and not np.all(
            rollout_policy_versions == expected_policy_version
        ):
            observed = np.unique(rollout_policy_versions).tolist()
            raise RuntimeError(
                "PPO update refused mixed/off-policy rollout versions: "
                f"expected={expected_policy_version}, observed={observed}"
            )
        update_index = self.reward_updates + self.cost_updates + 1
        sync_before = self._synchronization_status()
        update_begin = {
            "update_index": update_index,
            "policy_version_before": int(
                getattr(self, "policy_version", 0)
            ),
            "rollout_buffer_size": rollout_size,
            "rollout_buffer_capacity": self.n_steps,
            "partial_rollout": rollout_size < self.n_steps,
            "simulation_time_s": sync_before.get(
                "telemetry_sim_time_s", sync_before.get("sim_time_s")
            ),
            "communication_slot_index": sync_before.get(
                "telemetry_communication_slot_index",
                sync_before.get("communication_slot_index"),
            ),
            "rl_decision_index": sync_before.get(
                "telemetry_rl_decision_index",
                sync_before.get("rl_decision_index"),
            ),
            "boundary_status": sync_before.get("status", "disabled"),
            "update_started_perf_ns": update_started_perf_ns,
            "update_started_wall_time_ns": update_started_wall_ns,
        }
        print(
            "RACER_PPO_UPDATE_BEGIN "
            + json.dumps(update_begin, separators=(",", ":")),
            flush=True,
        )
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [
                self.policy.optimizer,
                self.policy.reward_critic_optimizer,
                self.policy.constraint_critic_optimizer,
            ]
        )
        actor_parameters, reward_parameters, constraint_parameters = (
            self.policy.optimization_parameters()
        )
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            None
            if self.clip_range_vf is None
            else self.clip_range_vf(self._current_progress_remaining)
        )
        self.j_cost_hat, self.constraint_estimator_source = (
            self._constraint_estimate()
        )
        violation = self.j_cost_hat - self.gamma_task
        self.crpo_mode = select_crpo_mode(
            self.j_cost_hat, self.gamma_task, 0.0
        )
        if self.crpo_mode == "reward":
            self.reward_updates += 1
        else:
            self.cost_updates += 1

        entropy_losses: list[float] = []
        entropy_sum_raw_values: list[float] = []
        entropy_mean_valid_values: list[float] = []
        policy_losses: list[float] = []
        reward_value_losses: list[float] = []
        cost_value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[float] = []
        actor_grad_norms_before: list[float] = []
        actor_grad_norms_after: list[float] = []
        reward_grad_norms_before: list[float] = []
        reward_grad_norms_after: list[float] = []
        constraint_grad_norms_before: list[float] = []
        constraint_grad_norms_after: list[float] = []
        continue_training = True
        final_loss = th.zeros((), device=self.device)
        epochs_executed = 0
        optimizer_steps = 0

        for epoch in range(self.n_epochs):
            epochs_executed = epoch + 1
            epoch_kls: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                log_prob, entropy_sum_raw, entropy_mean_valid = (
                    self.policy.evaluate_actor_actions_for_optimization(
                        rollout_data.observations, actions
                    )
                )
                reward_values = self.policy.predict_reward_values(
                    rollout_data.observations, rollout_data.critic_taus
                )
                cost_values = self.policy.predict_cost_values(
                    rollout_data.observations, rollout_data.critic_taus
                )
                reward_values = reward_values.flatten()
                cost_values = cost_values.flatten()
                advantages = (
                    rollout_data.normalized_advantages
                    if self.crpo_mode == "reward"
                    else -rollout_data.normalized_cost_advantages
                )
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                objective_1 = advantages * ratio
                objective_2 = advantages * th.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range
                )
                policy_loss = -th.min(objective_1, objective_2).mean()
                policy_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1.0) > clip_range).float()).item()
                )

                if clip_range_vf is None:
                    reward_prediction = reward_values
                    cost_prediction = cost_values
                else:
                    reward_prediction = rollout_data.old_values + th.clamp(
                        reward_values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    cost_prediction = rollout_data.old_cost_values + th.clamp(
                        cost_values - rollout_data.old_cost_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                reward_value_loss = F.mse_loss(
                    rollout_data.returns, reward_prediction
                )
                cost_value_loss = F.mse_loss(
                    rollout_data.cost_returns, cost_prediction
                )
                reward_value_losses.append(reward_value_loss.item())
                cost_value_losses.append(cost_value_loss.item())
                entropy_loss = -entropy_mean_valid.mean()
                entropy_losses.append(entropy_loss.item())
                entropy_sum_raw_values.append(entropy_sum_raw.mean().item())
                entropy_mean_valid_values.append(entropy_mean_valid.mean().item())
                actor_loss = policy_loss + self.ent_coef * entropy_loss
                scaled_reward_loss = self.vf_coef * reward_value_loss
                scaled_constraint_loss = self.cost_vf_coef * cost_value_loss
                final_loss = actor_loss
                if not all(
                    bool(th.isfinite(loss))
                    for loss in (
                        actor_loss, scaled_reward_loss, scaled_constraint_loss
                    )
                ):
                    raise FloatingPointError("CRPO actor/critic loss is NaN or Inf")

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean(
                        (th.exp(log_ratio) - 1.0) - log_ratio
                    ).item()
                    epoch_kls.append(approx_kl)
                    approx_kl_divs.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad(set_to_none=True)
                self.policy.reward_critic_optimizer.zero_grad(set_to_none=True)
                self.policy.constraint_critic_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if any(parameter.grad is not None for parameter in reward_parameters + constraint_parameters):
                    raise RuntimeError("actor loss propagated into a critic")
                actor_before = th.nn.utils.clip_grad_norm_(
                    actor_parameters, self.max_grad_norm, error_if_nonfinite=True
                )
                actor_grad_norms_before.append(float(actor_before.item()))
                actor_grad_norms_after.append(_gradient_norm(actor_parameters))
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad(set_to_none=True)

                scaled_reward_loss.backward()
                if any(parameter.grad is not None for parameter in actor_parameters + constraint_parameters):
                    raise RuntimeError("reward critic loss propagated outside reward critic")
                reward_before = th.nn.utils.clip_grad_norm_(
                    reward_parameters, self.max_grad_norm, error_if_nonfinite=True
                )
                reward_grad_norms_before.append(float(reward_before.item()))
                reward_grad_norms_after.append(_gradient_norm(reward_parameters))
                self.policy.reward_critic_optimizer.step()
                self.policy.reward_critic_optimizer.zero_grad(set_to_none=True)

                scaled_constraint_loss.backward()
                if any(parameter.grad is not None for parameter in actor_parameters + reward_parameters):
                    raise RuntimeError("constraint critic loss propagated outside constraint critic")
                constraint_before = th.nn.utils.clip_grad_norm_(
                    constraint_parameters, self.max_grad_norm, error_if_nonfinite=True
                )
                constraint_grad_norms_before.append(float(constraint_before.item()))
                constraint_grad_norms_after.append(_gradient_norm(constraint_parameters))
                self.policy.constraint_critic_optimizer.step()
                optimizer_steps += 1
                # Let the real-time behavior thread enqueue a forward between
                # learner minibatches when both models share one GPU/CPU.
                time.sleep(0)
            self._n_updates += 1
            if not continue_training:
                break

        if not all(
            bool(th.all(th.isfinite(parameter)))
            for parameter in self.policy.parameters()
        ):
            raise FloatingPointError(
                "CRPO optimizer produced a non-finite parameter"
            )

        reward_explained = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        cost_explained = explained_variance(
            self.rollout_buffer.cost_values.flatten(),
            self.rollout_buffer.cost_returns.flatten(),
        )
        mean = lambda values: float(np.mean(values)) if values else 0.0
        self.logger.record("train/entropy_loss", mean(entropy_losses))
        logged_values = {
            "entropy_sum_raw": mean(entropy_sum_raw_values),
            "entropy_mean_valid": mean(entropy_mean_valid_values),
            "actor_grad_norm_before_clip": mean(actor_grad_norms_before),
            "actor_grad_norm_after_clip": mean(actor_grad_norms_after),
            "reward_critic_grad_norm_before_clip": mean(reward_grad_norms_before),
            "reward_critic_grad_norm_after_clip": mean(reward_grad_norms_after),
            "constraint_critic_grad_norm_before_clip": mean(constraint_grad_norms_before),
            "constraint_critic_grad_norm_after_clip": mean(constraint_grad_norms_after),
            "approx_kl": mean(approx_kl_divs),
            "clip_fraction": mean(clip_fractions),
        }
        for name, value in logged_values.items():
            self.logger.record(name, value)
            self.logger.record(f"train/{name}", value)
        self.logger.record("train/policy_gradient_loss", mean(policy_losses))
        self.logger.record("train/reward_value_loss", mean(reward_value_losses))
        self.logger.record(
            "train/constraint_value_loss", mean(cost_value_losses)
        )
        # Preserve the established key for existing dashboards/checkpoints.
        self.logger.record("train/cost_value_loss", mean(cost_value_losses))
        self.logger.record(
            "train/reward_adv_mean", self.rollout_buffer.reward_adv_mean
        )
        self.logger.record(
            "train/reward_adv_std", self.rollout_buffer.reward_adv_std
        )
        self.logger.record(
            "train/constraint_adv_mean",
            self.rollout_buffer.constraint_adv_mean,
        )
        self.logger.record(
            "train/constraint_adv_std", self.rollout_buffer.constraint_adv_std
        )
        self.logger.record(
            "train/critic_tau_mean",
            float(np.mean(self.rollout_buffer.critic_taus)),
        )
        self.logger.record("train/approx_kl", mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", mean(clip_fractions))
        self.logger.record("train/loss", final_loss.item())
        self.logger.record("train/reward_explained_variance", reward_explained)
        self.logger.record("train/cost_explained_variance", cost_explained)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("crpo/mode", 0 if self.crpo_mode == "reward" else 1)
        self.logger.record(
            "crpo/mode_name",
            "reward" if self.crpo_mode == "reward" else "constraint",
        )
        self.logger.record("crpo/J_C_hat", self.j_cost_hat)
        self.logger.record("crpo/Gamma_task", self.gamma_task)
        self.logger.record("constraint/Gamma_task", self.gamma_task)
        self.logger.record("constraint/violation_estimate", violation)
        self.logger.record("crpo/constraint_violation", violation)
        self.logger.record("crpo/reward_updates", self.reward_updates)
        self.logger.record("crpo/cost_updates", self.cost_updates)
        self.logger.record(
            "crpo/estimator_is_episode_return",
            int(self.constraint_estimator_source == "episode_return"),
        )
        self.logger.record(
            "crpo/estimator_is_mean_step_cost",
            int(
                self.constraint_estimator_source
                in {"mean_step_cost", "partial_mean_step_cost"}
            ),
        )
        self.logger.record(
            "crpo/estimator_is_time_weighted_mean",
            int(
                self.constraint_estimator_source
                in {"time_weighted_mean", "partial_time_weighted_mean"}
            ),
        )
        if self.episode_cost_tracker.records:
            errors = [
                abs(item.telescoping_error)
                for item in self.episode_cost_tracker.records
            ]
            self.logger.record("crpo/telescoping_error_max", max(errors))
        return {
            "update_begin": update_begin,
            "sync_before": sync_before,
            "rollout_size": rollout_size,
            "epochs_executed": epochs_executed,
            "optimizer_steps": optimizer_steps,
            "update_started_perf_ns": update_started_perf_ns,
        }

    def train(self) -> None:
        """Train learner asynchronously while frozen behavior keeps acting."""

        self._initialize_policy_runtime()
        old_version = int(self.policy_version)
        if int(self._behavior_policy_version) != old_version:
            raise RuntimeError(
                "behavior/rollout policy version drift before PPO update"
            )

        # learner_policy is a long-lived independent object so Adam state is
        # preserved, while its weights are explicitly rebased to theta_k.
        self.policy.load_state_dict(self._behavior_policy.state_dict())
        if not self._policy_parameters_are_disjoint(
            self.policy, self._behavior_policy
        ):
            raise RuntimeError(
                "optimizer-owned learner shares frozen behavior storage"
            )
        behavior_parameter_versions = tuple(
            parameter._version for parameter in self._behavior_policy.parameters()
        )
        with self._update_runtime_lock:
            self._update_in_progress = True
            self._update_inference_stats = {
                "actions_generated": 0,
                "max_action_age": 0.0,
                "inference_latency_sum_ms": 0.0,
                "inference_latency_max_ms": 0.0,
                "state_versions": set(),
            }
        print(
            "[PPO UPDATE START] "
            + json.dumps(
                {"policy_version": old_version},
                separators=(",", ":"),
            ),
            flush=True,
        )

        pump: _UpdateInferencePump | None = None
        if (
            not self._stop_before_next_rollout
            and self.env is not None
            and self._last_obs is not None
        ):
            try:
                decoupled = self.env.env_method(
                    "supports_decoupled_transition_collection"
                )
            except (AttributeError, RuntimeError):
                decoupled = []
            if decoupled and all(bool(value) for value in decoupled):
                pump = _UpdateInferencePump(
                    self.env,
                    self._last_obs,
                    old_version,
                    lambda observation, status, version: (
                        self._perform_policy_inference(
                            observation,
                            status,
                            version,
                            publish_action=True,
                        )
                    ),
                    self._cycle_status,
                )
                pump.start()

        learner_result: dict[str, Any] = {}
        learner_error: list[BaseException] = []

        def run_learner() -> None:
            try:
                learner_result.update(self._train_learner_policy())
            except BaseException as caught:
                learner_error.append(caught)

        learner_thread = threading.Thread(
            target=run_learner,
            name="crpo-ppo-learner",
            daemon=False,
        )
        learner_thread.start()
        learner_thread.join()

        # Snapshot theta_{k+1} while theta_k remains live. The final swap
        # therefore only changes one Python object reference under the guard.
        next_behavior: CRPOActorCriticPolicy | None = None
        if not learner_error:
            # The update pump continues serving theta_k while this independent
            # theta_{k+1} snapshot is prepared for one reference swap.
            try:
                next_behavior = self._clone_frozen_policy(self.policy)
            except BaseException as caught:
                learner_error.append(caught)
        if pump is not None:
            pump.stop_and_join()
            latest_obs, latest_starts, _transitions, pump_max_age = (
                pump.snapshot()
            )
            self._last_obs = latest_obs
            self._last_original_obs = latest_obs.copy()
            self._last_episode_starts = latest_starts
            if pump.mission_ended:
                self.ended_on_mission_boundary = True
                self._stop_before_next_rollout = True
            if pump.error is not None and not isinstance(
                pump.error, MissionEndedError
            ):
                learner_error.append(pump.error)
            with self._update_runtime_lock:
                self._update_inference_stats["max_action_age"] = max(
                    float(
                        self._update_inference_stats.get(
                            "max_action_age", 0.0
                        )
                    ),
                    pump_max_age,
                )

        behavior_unchanged = behavior_parameter_versions == tuple(
            parameter._version for parameter in self._behavior_policy.parameters()
        )
        if not behavior_unchanged:
            learner_error.append(
                RuntimeError("behavior_policy changed during learner update")
            )
        if learner_error:
            with self._update_runtime_lock:
                self._update_in_progress = False
            raise learner_error[0]
        assert next_behavior is not None
        new_version = old_version + 1
        swapped_from, swapped_to = self._swap_behavior_policy(
            next_behavior, new_version
        )
        if (swapped_from, swapped_to) != (old_version, new_version):
            with self._update_runtime_lock:
                self._update_in_progress = False
            raise RuntimeError("atomic policy swap returned inconsistent versions")
        self.policy_version = new_version
        # No theta_k decision may be reused to score theta_{k+1}'s rollout.
        self._behavior_decision_records.clear()

        with self._update_runtime_lock:
            self._update_in_progress = False
            inference_stats = dict(self._update_inference_stats)
        state_versions = inference_stats.pop("state_versions", set())
        actions_generated = int(inference_stats.get("actions_generated", 0))
        mean_inference_ms = (
            float(inference_stats.get("inference_latency_sum_ms", 0.0))
            / actions_generated
            if actions_generated
            else 0.0
        )

        update_begin = learner_result["update_begin"]
        sync_before = learner_result["sync_before"]
        rollout_size = int(learner_result["rollout_size"])
        refresh_applied = pump is not None
        if (
            pump is None
            and not self._stop_before_next_rollout
            and self.env is not None
            and self._last_obs is not None
        ):
            try:
                refreshed = self.env.env_method("refresh_latest_observation")
            except (AttributeError, RuntimeError, MissionEndedError):
                refreshed = []
            if refreshed:
                refreshed_array = np.asarray(refreshed, dtype=np.float32)
                if refreshed_array.shape == self._last_obs.shape:
                    self._last_obs = refreshed_array
                    self._last_original_obs = refreshed_array.copy()
                    refresh_applied = True
        sync_after = self._synchronization_status()
        end_sim_time = sync_after.get(
            "telemetry_sim_time_s", sync_after.get("sim_time_s")
        )
        end_slot = sync_after.get(
            "telemetry_communication_slot_index",
            sync_after.get("communication_slot_index"),
        )
        end_decision = sync_after.get(
            "telemetry_rl_decision_index",
            sync_after.get("rl_decision_index"),
        )
        sync_enabled = bool(sync_before.get("enabled", False))
        simulation_time_frozen = bool(
            sync_enabled
            and update_begin["simulation_time_s"] == end_sim_time
            and update_begin["communication_slot_index"] == end_slot
            and update_begin["rl_decision_index"] == end_decision
            and sync_before.get("status") in {"paused", "terminal"}
            and sync_after.get("status") in {"paused", "terminal"}
        )
        simulation_time_advanced = bool(
            update_begin["simulation_time_s"] is not None
            and end_sim_time is not None
            and float(end_sim_time)
            > float(update_begin["simulation_time_s"]) + 1.0e-12
        )
        update_completed_perf_ns = time.perf_counter_ns()
        update_completed_wall_ns = time.time_ns()
        ppo_update_latency_ms = (
            update_completed_perf_ns
            - int(learner_result["update_started_perf_ns"])
        ) / 1.0e6
        update_end = {
            **update_begin,
            "simulation_time_s_start": update_begin["simulation_time_s"],
            "simulation_time_s_end": end_sim_time,
            "communication_slot_index_end": end_slot,
            "rl_decision_index_end": end_decision,
            "boundary_status_end": sync_after.get("status", "disabled"),
            "simulation_time_frozen": simulation_time_frozen,
            "simulation_time_advanced": simulation_time_advanced,
            "latest_state_refresh_applied": refresh_applied,
            "policy_version_after": self.policy_version,
            "physical_version_start": sync_before.get("physical_version"),
            "physical_version_end": sync_after.get("physical_version"),
            "communication_version_start": sync_before.get(
                "communication_version"
            ),
            "communication_version_end": sync_after.get(
                "communication_version"
            ),
            "consumed_physical_version_end": sync_after.get(
                "consumed_physical_version"
            ),
            "consumed_communication_version_end": sync_after.get(
                "consumed_communication_version"
            ),
            "skipped_physical_versions_total": sync_after.get(
                "skipped_physical_versions", 0
            ),
            "skipped_communication_versions_total": sync_after.get(
                "skipped_communication_versions", 0
            ),
            "update_completed_perf_ns": update_completed_perf_ns,
            "update_completed_wall_time_ns": update_completed_wall_ns,
            "ppo_update_latency_ms": ppo_update_latency_ms,
            "epochs_executed": int(learner_result["epochs_executed"]),
            "optimizer_steps": int(learner_result["optimizer_steps"]),
            "async_learner": True,
            "behavior_parameters_unchanged": behavior_unchanged,
            "behavior_learner_storage_disjoint": True,
            "policy_swap_success": True,
            "actions_generated_during_update": actions_generated,
            "update_transitions_excluded": (
                pump.transition_count if pump is not None else 0
            ),
            "distinct_update_states_inferred": len(state_versions),
            "update_max_action_age": float(
                inference_stats.get("max_action_age", 0.0)
            ),
            "update_inference_latency_mean_ms": mean_inference_ms,
            "update_inference_latency_max_ms": float(
                inference_stats.get("inference_latency_max_ms", 0.0)
            ),
            "inference_used_updating_parameters": False,
        }
        self.sync_update_records.append(update_end)
        self.logger.record(
            "timing/ppo_update_sim_time_start",
            float(update_begin["simulation_time_s"] or 0.0),
        )
        self.logger.record(
            "timing/ppo_update_sim_time_end", float(end_sim_time or 0.0)
        )
        self.logger.record(
            "timing/ppo_update_sim_time_frozen", int(simulation_time_frozen)
        )
        self.logger.record(
            "timing/ppo_update_sim_time_advanced", int(simulation_time_advanced)
        )
        self.logger.record("timing/rollout_buffer_size", rollout_size)
        self.logger.record(
            "timing/ppo_update_latency_ms", ppo_update_latency_ms
        )
        self.logger.record(
            "timing/update_actions_generated", actions_generated
        )
        self.logger.record(
            "timing/update_action_age_max",
            float(inference_stats.get("max_action_age", 0.0)),
        )
        print(
            "[POLICY SWAP] "
            + json.dumps(
                {"old_version": swapped_from, "new_version": swapped_to},
                separators=(",", ":"),
            ),
            flush=True,
        )
        print(
            "[PPO UPDATE END] "
            + json.dumps(
                {
                    "old_version": old_version,
                    "new_version": new_version,
                    "update_time": ppo_update_latency_ms / 1000.0,
                    "actions_generated_during_update": actions_generated,
                    "max_action_age": float(
                        inference_stats.get("max_action_age", 0.0)
                    ),
                    "inference_latency_mean_ms": mean_inference_ms,
                    "policy_swap_success": True,
                    "inference_used_updating_parameters": False,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        print(
            "RACER_PPO_UPDATE_END "
            + json.dumps(update_end, separators=(",", ":")),
            flush=True,
        )
