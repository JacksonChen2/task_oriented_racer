"""Exercise campaign configuration and resume without starting ROS processes."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from agentic_crpo.no_link_schedule import no_link_mask_for_episode

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['unmasked', 'masked', 'scheduled'])
def test_campaign_keeps_saved_mode_and_checks_checkpoints(tmp_path, monkeypatch, mode):
    enabled = mode == 'masked'
    spec = importlib.util.spec_from_file_location('portable_campaign', ROOT / 'scripts/campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen-for-config-test')
    reference = tmp_path / 'local_reference.json'
    reference.write_text('{}')
    monkeypatch.setenv('RACER_QWEN8_PERFECT_RESULT', str(reference))
    output = tmp_path / 'campaign'
    calls = []

    class FinishedEpisode:
        def __init__(self, command, *, env, **kwargs):
            calls.append(env.copy())
            assert env['RACER_QWEN8_PERFECT_RESULT'] == str(reference)
            self.pid = 999999
            config = yaml.safe_load(Path(env['RACER_QWEN8_CONFIG']).read_text())
            episode = int(env['RACER_QWEN8_START_EPISODE'])
            folder = output / f'episode_{episode:02d}'
            training = folder / 'training'
            training.mkdir(parents=True)
            mask, penalty = no_link_mask_for_episode(config['actor'], episode)
            (training / 'llm_prior_crpo_final.zip').write_bytes(b'test-checkpoint')
            (training / 'training_state.json').write_text(json.dumps({
                'mask_no_link_actions': mask,
                'no_link_logit_penalty': penalty,
                'no_link_logit_schedule': config['actor']['no_link_logit_schedule'],
            }))
            (folder / 'warehouse_full_bs_round_robin_result.json').write_text('{}')

        def wait(self):
            return 0

    monkeypatch.setattr(campaign.subprocess, 'Popen', FinishedEpisode)
    monkeypatch.setattr(campaign.signal, 'signal', lambda *args: None)
    option = {'masked': '--mask-no-link-actions', 'unmasked': '--no-mask-no-link-actions',
              'scheduled': '--schedule-no-link-actions'}[mode]
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--episodes', '1', '--output', str(output), option])
    campaign.main()
    assert len(calls) == 1
    assert 'RACER_QWEN8_RESUME' not in calls[0]
    assert f'_{mode}_' in calls[0]['RACER_QWEN8_ALGORITHM_LABEL']

    # Omitted CLI flag uses the campaign snapshot, even if the default YAML is off.
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--resume', '--episodes', '2', '--output', str(output)])
    campaign.main()
    assert len(calls) == 2
    assert calls[1]['RACER_QWEN8_RESUME'] == str(output / 'episode_01/training/llm_prior_crpo_final.zip')
    snapshot = output / 'campaign_config.yaml'
    config = yaml.safe_load(snapshot.read_text())
    assert config['actor']['mask_no_link_actions'] is enabled
    assert config['actor']['no_link_logit_schedule']['enabled'] is (mode == 'scheduled')
    assert config['experiment']['episodes'] == 2
    assert json.loads((output / 'progress.json').read_text())['completed_episodes'] == 2

    before = snapshot.read_bytes()
    opposite = {'masked': '--no-mask-no-link-actions', 'unmasked': '--mask-no-link-actions',
                'scheduled': '--no-schedule-no-link-actions'}[mode]
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--resume', '--episodes', '3', '--output', str(output), opposite])
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2
    assert snapshot.read_bytes() == before
    assert len(calls) == 2

    # Resume crosses the hard-mask boundary using the absolute episode number.
    if mode == 'scheduled':
        monkeypatch.setattr(sys, 'argv', ['campaign.py', '--resume', '--episodes', '51', '--output', str(output)])
        campaign.main()
        assert len(calls) == 51
        for episode, mask, penalty in ((2, False, 0.25), (49, False, 12.0), (50, True, 0.0), (51, True, 0.0)):
            data = json.loads((output / f'episode_{episode:02d}/training/training_state.json').read_text())
            assert (data['mask_no_link_actions'], data['no_link_logit_penalty']) == (mask, penalty)
        campaign.main()  # Completed checkpoint validation also uses each episode's schedule.
        assert len(calls) == 51

    previous_call_count = len(calls)
    state = output / 'episode_02/training/training_state.json'
    state.write_text(json.dumps({'mask_no_link_actions': not enabled, 'no_link_logit_penalty': 0.0}))
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--resume', '--episodes', '3', '--output', str(output)])
    with pytest.raises(SystemExit, match='Checkpoint mask mode'):
        campaign.main()
    assert len(calls) == previous_call_count


def test_cli_rejects_conflicting_modes_before_launch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('portable_campaign', ROOT / 'scripts/campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen')
    output = tmp_path / 'conflicting'
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--output', str(output),
                                     '--mask-no-link-actions', '--schedule-no-link-actions'])
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2
    assert not (output / 'campaign_config.yaml').exists()
