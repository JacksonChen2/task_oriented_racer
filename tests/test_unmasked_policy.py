from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from agentic_crpo.config import load_config
from agentic_crpo.llm_prior_policy import LLMPriorCRPOActorCriticPolicy
from agentic_crpo.no_link_schedule import no_link_mask_for_episode

ROOT = Path(__file__).resolve().parents[1]


def test_all_unavailable_links_remain_selectable_and_ppo_likelihood_matches():
    torch.set_num_threads(1)
    policy = LLMPriorCRPOActorCriticPolicy(
        gym.spaces.Box(0, 1, shape=(350,), dtype=np.float32),
        gym.spaces.MultiBinary(100), lambda _: 0.0001,
        features_extractor_kwargs={'observation_layout': 'phaoi_z'},
        initial_action_probability=0.8,
    )
    # Zero is the exact unavailable-link sentinel, with a neutral LLM prior.
    observation = torch.zeros((1, 350))
    distribution = policy.get_distribution(observation)
    assert torch.allclose(distribution.distribution.probs, torch.full((1, 100), 0.8))
    actions = torch.ones((1, 100))
    _, evaluated_log_prob, _ = policy.evaluate_actions(observation, actions)
    assert torch.allclose(evaluated_log_prob, distribution.log_prob(actions))
    assert torch.isfinite(evaluated_log_prob).all()
    policy.set_mask_no_link_actions(True)
    assert torch.count_nonzero(policy.get_distribution(observation).distribution.probs) == 0
    policy.set_mask_no_link_actions(False)
    assert torch.allclose(policy.get_distribution(observation).distribution.probs,
                          torch.full((1, 100), 0.8))
    with pytest.raises(ValueError):
        policy.set_no_link_logit_penalty(-1.0)


def test_config_never_enables_episode_suppression(monkeypatch):
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen-for-config-test')
    monkeypatch.delenv('RACER_QWEN8_PERFECT_RESULT', raising=False)
    for episode in (1, 40, 49, 50, 60, 100, 1000):
        monkeypatch.setenv('RACER_QWEN8_EPISODE_NUMBER', str(episode))
        config = load_config(ROOT / 'final_racer_training/config/train.yaml')
        assert config['actor']['mask_no_link_actions'] is False
        assert config['actor']['no_link_logit_penalty'] == 0.0
        assert config['actor']['no_link_logit_schedule']['enabled'] is False
        assert no_link_mask_for_episode(config['actor'], episode) == (False, 0.0)
        assert config['environment']['backend']['perfect_reference_result'] is None


@pytest.mark.parametrize('enabled', [False, True])
def test_directional_mask_prior_and_ppo_distribution_agree(enabled):
    torch.set_num_threads(1)
    policy = LLMPriorCRPOActorCriticPolicy(
        gym.spaces.Box(0, 1, shape=(350,), dtype=np.float32),
        gym.spaces.MultiBinary(100), lambda _: 0.0001,
        features_extractor_kwargs={'observation_layout': 'phaoi_z'},
        initial_action_probability=0.8, mask_no_link_actions=enabled,
    )
    observation = torch.zeros((1, 350))
    observation[0, 120] = 1.0  # Only UAV 0 has an uplink.
    observation[0, 131] = 1.0e-6  # Weak-but-present BS -> UAV 1 must remain valid.
    observation[0, 132] = 1.0  # BS -> UAV 2 is also available.
    neutral_probs = policy.get_distribution(observation).distribution.probs.detach().clone()
    observation[0, 240:340] = torch.linspace(0, 1, 100)
    observation[0, 340] = 1.0
    feasible = torch.tensor([[j in (1, 2) for i in range(10) for j in range(10) if i != j]
                             + [i == 0 for i in range(10)]])
    distribution = policy.get_distribution(observation)
    probs = distribution.distribution.probs.detach().clone()
    assert not torch.allclose(probs[feasible], neutral_probs[feasible])  # Soft LLM prior survives.
    assert torch.all(probs[feasible] > 0)
    if enabled:
        assert torch.count_nonzero(probs[~feasible]) == 0
    else:
        assert torch.all(probs[~feasible] > 0)
    actions, _, _, cached_log_prob, logits = policy.forward_crpo_with_masked_logits(observation)
    if enabled:
        assert torch.count_nonzero(actions[~feasible]) == 0
    _, ordinary_log_prob, _ = policy.evaluate_actions(observation, actions)
    _, _, crpo_log_prob, _ = policy.evaluate_actions_crpo(observation, actions)
    optimized_log_prob, entropy_sum, entropy_mean = policy.evaluate_actor_actions_for_optimization(observation, actions)
    for value in (ordinary_log_prob, crpo_log_prob, optimized_log_prob):
        assert torch.allclose(value, cached_log_prob)
    assert torch.isfinite(logits).all()
    loss = -optimized_log_prob.mean() - 0.001 * entropy_mean.mean()
    loss.backward()
    assert torch.isfinite(entropy_sum).all()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_all_unavailable_masked_actions_have_finite_zero_entropy():
    policy = LLMPriorCRPOActorCriticPolicy(
        gym.spaces.Box(0, 1, shape=(350,), dtype=np.float32),
        gym.spaces.MultiBinary(100), lambda _: 0.0001,
        features_extractor_kwargs={'observation_layout': 'phaoi_z'},
        mask_no_link_actions=True,
    )
    obs = torch.zeros((1, 350))
    actions, _, log_prob = policy(obs)
    assert torch.count_nonzero(actions) == 0
    optimized, entropy_sum, entropy_mean = policy.evaluate_actor_actions_for_optimization(obs, actions)
    assert torch.allclose(log_prob, optimized)
    assert torch.equal(entropy_sum, torch.zeros_like(entropy_sum))
    assert torch.equal(entropy_mean, torch.zeros_like(entropy_mean))
    (-optimized.mean() - entropy_mean.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_config_mode_stays_constant_across_episodes(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen-for-config-test')
    config = load_config(ROOT / 'final_racer_training/config/train.yaml')
    config['actor']['mask_no_link_actions'] = enabled
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    for episode in (1, 49, 50, 60, 1000):
        monkeypatch.setenv('RACER_QWEN8_EPISODE_NUMBER', str(episode))
        loaded = load_config(path)
        assert loaded['actor']['mask_no_link_actions'] is enabled
        assert loaded['actor']['no_link_logit_penalty'] == 0.0
        assert ('_masked_v1' if enabled else '_unmasked_v1') in loaded['algorithm']['variant']


@pytest.mark.parametrize('change', [
    {'mask_no_link_actions': 'false'}, {'mask_no_link_actions': 1},
    {'no_link_logit_penalty': 0.25},
    {'no_link_logit_schedule': {'enabled': 'false'}},
])
def test_invalid_mask_and_schedule_options_are_rejected(tmp_path, monkeypatch, change):
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen-for-config-test')
    config = load_config(ROOT / 'final_racer_training/config/train.yaml')
    config['actor'].update(change)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)
