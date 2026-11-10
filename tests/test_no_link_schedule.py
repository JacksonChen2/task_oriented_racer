"""Exercise optional scheduling and its actual Bernoulli/PPO behavior."""
import copy
import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from agentic_crpo.no_link_schedule import no_link_mask_for_episode, validate_no_link_options
from agentic_crpo.llm_prior_policy import LLMPriorCRPOActorCriticPolicy


def test_schedule_defaults_and_original_50_episode_boundaries():
    actor = {'mask_no_link_actions': False, 'no_link_logit_schedule': {
        'enabled': True, 'total_episodes': 50, 'max_penalty': 12.0}}
    original = copy.deepcopy(actor)
    expected = {1: (False, 0.0), 2: (False, 0.25), 20: (False, 4.75),
                25: (False, 6.0), 49: (False, 12.0), 50: (True, 0.0), 1000: (True, 0.0)}
    for episode, value in expected.items():
        assert no_link_mask_for_episode(actor, episode) == value
        assert no_link_mask_for_episode({}, episode) == (False, 0.0)
        assert no_link_mask_for_episode({'mask_no_link_actions': True}, episode) == (True, 0.0)
    assert actor == original
    # Merely supplying parameters never opts an old config into scheduling.
    actor['no_link_logit_schedule'].pop('enabled')
    assert no_link_mask_for_episode(actor, 60) == (False, 0.0)
    actor['no_link_logit_schedule'].update(enabled=True, total_episodes=3, max_penalty=4)
    assert [no_link_mask_for_episode(actor, e) for e in (1, 2, 3, 4)] == [
        (False, 0.0), (False, 4.0), (True, 0.0), (True, 0.0)]


@pytest.mark.parametrize('episode', [None, 0, -1, True, 2.5, '2'])
def test_schedule_requires_valid_episode(episode):
    with pytest.raises(ValueError, match='episode number'):
        no_link_mask_for_episode({'no_link_logit_schedule': {'enabled': True}}, episode)


@pytest.mark.parametrize('schedule', [
    None, [], {'enabled': 'false'}, {'enabled': 1}, {'total_episodes': 2},
    {'total_episodes': True}, {'total_episodes': 3.5}, {'max_penalty': 0},
    {'max_penalty': -1}, {'max_penalty': float('nan')}, {'max_penalty': float('inf')},
    {'max_penalty': True}, {'max_penalty': '12'}, {'enable': True},
])
def test_invalid_schedule_rejected(schedule):
    with pytest.raises(ValueError):
        validate_no_link_options({'no_link_logit_schedule': schedule})


def test_constant_hard_mask_and_schedule_are_alternatives():
    with pytest.raises(ValueError, match='constant hard masking OR'):
        validate_no_link_options({'mask_no_link_actions': True,
                                  'no_link_logit_schedule': {'enabled': True}})


def test_penalty_decreases_only_unavailable_probabilities_and_ppo_agrees():
    torch.set_num_threads(1)
    policy = LLMPriorCRPOActorCriticPolicy(
        gym.spaces.Box(0, 1, shape=(350,), dtype=np.float32),
        gym.spaces.MultiBinary(100), lambda _: 0.0001,
        features_extractor_kwargs={'observation_layout': 'phaoi_z'},
        initial_action_probability=0.8,
    )
    obs = torch.zeros((2, 350))
    obs[0, 120] = 1  # UL 0
    obs[0, 131] = 1e-6  # DL 1: weak, available
    obs[0, 132] = 1  # DL 2
    obs[:, 240:340] = torch.linspace(0, 1, 100)  # Non-neutral soft LLM prior
    obs[:, 340] = 1
    feasible = policy.action_mask(obs)
    baseline = policy.get_distribution(obs).distribution.probs.detach().clone()
    prior_probs = baseline
    for penalty in (0.25, 6, 12):
        policy.set_no_link_logit_penalty(penalty)
        dist = policy.get_distribution(obs)
        probs = dist.distribution.probs.detach()
        assert torch.equal(probs[feasible], baseline[feasible])
        assert torch.all(probs[~feasible] < prior_probs[~feasible])
        assert torch.all(probs[~feasible] > 0)
        assert torch.allclose(probs[~feasible],
                              torch.sigmoid(torch.logit(baseline[~feasible]) - penalty))
        prior_probs = probs.clone()
        actions, _, _, behavior_log_prob, logits = policy.forward_crpo_with_masked_logits(obs)
        _, ordinary, _ = policy.evaluate_actions(obs, actions)
        _, _, crpo, _ = policy.evaluate_actions_crpo(obs, actions)
        optimized, entropy_sum, entropy_mean = policy.evaluate_actor_actions_for_optimization(obs, actions)
        for value in (ordinary, crpo, optimized):
            assert torch.allclose(value, behavior_log_prob)
        assert torch.isfinite(logits).all() and torch.isfinite(entropy_sum).all()
        policy.zero_grad()
        (-optimized.mean() - 0.001 * entropy_mean.mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    # Final scheduled hard mask still preserves all feasible soft-prior logits.
    policy.set_no_link_logit_penalty(0)
    policy.set_mask_no_link_actions(True)
    probs = policy.get_distribution(obs).distribution.probs.detach()
    assert torch.equal(probs[feasible], baseline[feasible])
    assert torch.count_nonzero(probs[~feasible]) == 0
    policy.set_mask_no_link_actions(False)
    assert torch.equal(policy.get_distribution(obs).distribution.probs, baseline)
    # The original finite floor prevents soft suppression becoming a hard zero.
    policy.set_no_link_logit_penalty(1e6)
    probs = policy.get_distribution(obs).distribution.probs.detach()
    assert torch.all(probs[~feasible] > 0)
    assert torch.allclose(probs[~feasible], torch.full_like(probs[~feasible], 1/(1+math.exp(30))), atol=0)
    for bad in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            policy.set_no_link_logit_penalty(bad)
