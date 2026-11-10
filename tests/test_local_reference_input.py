"""Reference inputs are local requirements, never bundled experiment data."""
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from agentic_crpo.config import load_config, require_reference_input
from agentic_crpo.factory import make_backend

ROOT = Path(__file__).resolve().parents[1]


def test_default_has_no_reference_but_mock_remains_available(monkeypatch):
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen')
    monkeypatch.delenv('RACER_QWEN8_PERFECT_RESULT', raising=False)
    config = load_config(ROOT / 'final_racer_training/config/train.yaml')
    assert config['environment']['backend']['perfect_reference_result'] is None
    assert config['environment']['backend']['require_perfect_reference'] is True
    assert config['environment']['backend']['require_auxiliary_reference_metrics'] is True
    with pytest.raises(FileNotFoundError, match='RACER_QWEN8_PERFECT_RESULT'):
        require_reference_input(config)
    assert make_backend(config, force_mock=True) is not None


@pytest.mark.parametrize('via_environment', [False, True])
def test_local_reference_path_is_used_and_missing_files_are_rejected(tmp_path, monkeypatch, via_environment):
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen')
    monkeypatch.delenv('RACER_QWEN8_PERFECT_RESULT', raising=False)
    config = load_config(ROOT / 'final_racer_training/config/train.yaml')
    reference = tmp_path / 'reference with spaces.json'
    reference.write_text('{}')  # Only path handling is tested here.
    if via_environment:
        monkeypatch.setenv('RACER_QWEN8_PERFECT_RESULT', str(reference))
    else:
        config['environment']['backend']['perfect_reference_result'] = reference.name
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path)
    assert require_reference_input(loaded) == reference
    reference.unlink()
    with pytest.raises(FileNotFoundError, match='does not exist'):
        require_reference_input(loaded)


def test_missing_reference_stops_campaign_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setenv('QWEN_MODEL_PATH', '/tmp/unused-qwen')
    monkeypatch.delenv('RACER_QWEN8_PERFECT_RESULT', raising=False)
    spec = importlib.util.spec_from_file_location('campaign_without_reference', ROOT / 'scripts/campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    output = tmp_path / 'campaign'
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--episodes', '1', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2
    assert not output.exists()
