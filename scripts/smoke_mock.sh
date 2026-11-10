#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
: "${RACER_CRPO_PYTHON:?Set RACER_CRPO_PYTHON}"
test_output="$(mktemp -d /tmp/racer-optional-schedule-smoke.XXXXXX)"
# These unit tests need no ROS pytest plugins (which target a different Python ABI).
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$RACER_CRPO_PYTHON" -m pytest -q -p no:cacheprovider "$RACER_PACKAGE_ROOT/tests"
"$RACER_CRPO_PYTHON" - "$test_output" "$RACER_QWEN8_CONFIG" <<'PY'
import sys
from pathlib import Path
from agentic_crpo.config import load_config, save_resolved_config, set_action_mask_mode
for label, enabled in [('unmasked', False), ('masked', True), ('scheduled', False)]:
    config = load_config(sys.argv[2])
    set_action_mask_mode(config, enabled, schedule_enabled=label == 'scheduled')
    save_resolved_config(config, Path(sys.argv[1]) / label / 'config.yaml')
PY
for mode in unmasked masked; do
  RACER_QWEN8_EPISODE_NUMBER=1 "$RACER_CRPO_PYTHON" \
    -m agentic_crpo.train_event_driven_llm_prior_crpo \
    --config "$test_output/$mode/config.yaml" --mock-smoke --output-dir "$test_output/$mode/episode_01"
  RACER_QWEN8_EPISODE_NUMBER=60 "$RACER_CRPO_PYTHON" \
    -m agentic_crpo.train_event_driven_llm_prior_crpo \
    --config "$test_output/$mode/config.yaml" --mock-smoke \
    --resume "$test_output/$mode/episode_01/llm_prior_crpo_final.zip" \
    --output-dir "$test_output/$mode/episode_60"
done
# Exercise real model save/load and optimizer updates across the schedule boundary.
previous_checkpoint=""
for episode in 1 2 25 49 50 60; do
  resume_args=()
  if [[ -n "$previous_checkpoint" ]]; then
    resume_args=(--resume "$previous_checkpoint")
  fi
  RACER_QWEN8_EPISODE_NUMBER="$episode" "$RACER_CRPO_PYTHON" \
    -m agentic_crpo.train_event_driven_llm_prior_crpo \
    --config "$test_output/scheduled/config.yaml" --mock-smoke \
    "${resume_args[@]}" --output-dir "$test_output/scheduled/episode_$episode"
  previous_checkpoint="$test_output/scheduled/episode_$episode/llm_prior_crpo_final.zip"
done
"$RACER_CRPO_PYTHON" - "$test_output" <<'PY'
import json,sys
from pathlib import Path
from agentic_crpo.config import load_config
from agentic_crpo.no_link_schedule import no_link_mask_for_episode
p=Path(sys.argv[1])
for mode, enabled in [('unmasked', False), ('masked', True)]:
    steps=[]
    for episode in (1,60):
        state=json.loads((p/mode/f'episode_{episode:02d}/training_state.json').read_text())
        assert state['mask_no_link_actions'] is enabled
        assert state['no_link_logit_penalty']==0.0
        assert state['llm_action_prior']['enabled'] is True
        steps.append(state['num_timesteps'])
    assert steps[1]>steps[0]>0
config=load_config(p/'scheduled/config.yaml')
previous_steps=0
for episode in (1,2,25,49,50,60):
    output=p/'scheduled'/f'episode_{episode}'
    state=json.loads((output/'training_state.json').read_text())
    expected=no_link_mask_for_episode(config['actor'],episode)
    assert (state['mask_no_link_actions'],state['no_link_logit_penalty']) == expected
    assert state['no_link_logit_schedule'] == config['actor']['no_link_logit_schedule']
    assert state['llm_action_prior']['enabled'] is True
    assert state['num_timesteps']>previous_steps
    previous_steps=state['num_timesteps']
    resolved=load_config(output/'resolved_config.yaml')
    assert no_link_mask_for_episode(resolved['actor'],episode) == expected
print('OPTIONAL_SCHEDULE_FRESH_AND_RESUME_OK',p)
PY
