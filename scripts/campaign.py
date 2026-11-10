#!/usr/bin/env python3
"""Train with optional constant hard masking or episode suppression (both default off)."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=60, help='Total target episode number')
    parser.add_argument('--duration', type=int, default=300, help='Simulation seconds per episode')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--resume', action='store_true', help='Continue this package\'s existing campaign')
    parser.add_argument('--domain-base', type=int, default=100)
    parser.add_argument('--mask-no-link-actions', action=argparse.BooleanOptionalAction,
                        default=None, help='Override the YAML hard-mask option for a new campaign; default off')
    parser.add_argument('--schedule-no-link-actions', action=argparse.BooleanOptionalAction,
                        default=None, help='Enable gradual suppression then hard masking; default off')
    args = parser.parse_args()
    if args.episodes < 1 or args.duration < 1 or not 1 <= args.domain_base <= 232:
        parser.error('Invalid episode count, duration or ROS domain')
    output = (args.output or ROOT / 'results' / datetime.now().strftime('train_%Y%m%d_%H%M%S')).expanduser().resolve()
    if output.exists() and not args.resume:
        parser.error('Output exists; use --resume to continue or choose a new output')
    from agentic_crpo.config import load_config, save_resolved_config, set_action_mask_mode, require_reference_input
    from agentic_crpo.no_link_schedule import no_link_mask_for_episode
    config = load_config(ROOT / 'final_racer_training/config/train.yaml')
    # Keep the 300-second task reference and cost normalization unchanged.
    # --duration can stop the simulator early for integration checks only.
    config['experiment']['episode_duration_s'] = args.duration
    config['experiment']['episodes'] = args.episodes
    config['training']['output_dir'] = str(output / 'training')
    config.pop('_config_path', None)
    snapshot = output / 'campaign_config.yaml'
    if snapshot.exists():
        previous = load_config(snapshot)
        if int(previous['experiment']['episode_duration_s']) != args.duration:
            parser.error('Resume duration differs from the existing campaign')
        if (args.mask_no_link_actions is not None
                and args.mask_no_link_actions != previous['actor']['mask_no_link_actions']):
            parser.error('Resume mask mode differs from the saved campaign; use a new output directory')
        if (args.schedule_no_link_actions is not None
                and args.schedule_no_link_actions != previous['actor']['no_link_logit_schedule']['enabled']):
            parser.error('Resume schedule differs from the saved campaign; use a new output directory')
        # Preserve the existing campaign configuration, including any intentional edits.
        previous['qwen']['model_path'] = config['qwen']['model_path']
        previous['environment']['backend']['perfect_reference_result'] = config['environment']['backend']['perfect_reference_result']
        previous['training']['output_dir'] = str(output / 'training')
        config = previous
        config.pop('_config_path', None)
    elif args.mask_no_link_actions is not None or args.schedule_no_link_actions is not None:
        try:
            set_action_mask_mode(
                config,
                config['actor']['mask_no_link_actions'] if args.mask_no_link_actions is None else args.mask_no_link_actions,
                schedule_enabled=args.schedule_no_link_actions,
            )
        except ValueError as error:
            parser.error(str(error))
    config['experiment']['episodes'] = args.episodes
    mask_enabled = config['actor']['mask_no_link_actions']
    schedule = config['actor']['no_link_logit_schedule']
    mask_label = 'scheduled' if schedule['enabled'] else ('masked' if mask_enabled else 'unmasked')
    try:
        reference = require_reference_input(config)
    except FileNotFoundError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, snapshot)
    env = os.environ.copy()
    env.update({
        'RACER_QWEN8_CONFIG': str(snapshot), 'RACER_QWEN8_CAMPAIGN': str(output),
        'RACER_QWEN8_PERFECT_RESULT': str(reference),
        'RACER_QWEN8_DURATION': str(args.duration), 'RACER_QWEN8_GAMMA_TASK': '0.15',
        'RACER_QWEN8_LEARNING_RATE': '0.0001', 'RACER_QWEN8_N_STEPS': '256',
        'RACER_QWEN8_BATCH_SIZE': '64', 'RACER_QWEN8_N_EPOCHS': '5',
        'RACER_QWEN8_ACTIVATION_FN': 'silu', 'RACER_QWEN8_TOTAL_TIMESTEPS': '250000',
        'RACER_UAV_CHANNEL_ACCESS_MODE': 'csma', 'RACER_BS_CHANNEL_ACCESS_MODE': 'ofdma',
        'RACER_UAV_TX_POWER_DBM': '23', 'RACER_QWEN8_BS_TX_POWER_DBM': '33',
        'RACER_QWEN8_FIXED_MCS_INDEX': '14',
        'RACER_BS_DOWNLINK_SOURCE_OWNED_CHUNKS_ONLY': 'true',
        'RACER_BS_DOWNLINK_NEWEST_MISSING_CHUNK_FIRST': 'false',
        'RACER_QWEN8_ALGORITHM_LABEL': f'portable_phaoi_z_{mask_label}_bs_ofdma_u2u_csma',
    })
    def progress(status, completed, current=None):
        data = dict(status=status, completed_episodes=completed, current_episode=current,
                    target_episodes=args.episodes, pid=os.getpid(), updated_at=datetime.now().astimezone().isoformat())
        temp = output / 'progress.tmp'
        temp.write_text(json.dumps(data, indent=2) + '\n')
        temp.replace(output / 'progress.json')
    child = None
    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        progress('interrupted', completed)
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    completed = 0
    for episode in range(1, args.episodes + 1):
        ep = output / f'episode_{episode:02d}'
        checkpoint = ep / 'training/llm_prior_crpo_final.zip'
        if ep.exists():
            result = ep / 'warehouse_full_bs_round_robin_result.json'
            state_path = ep / 'training/training_state.json'
            if not (checkpoint.is_file() and result.is_file() and state_path.is_file()):
                progress('blocked_incomplete_episode', completed, episode)
                raise SystemExit(f'Incomplete episode preserved: {ep}. Move it aside before retrying.')
            state = json.loads(state_path.read_text())
            expected_mask, expected_penalty = no_link_mask_for_episode(config['actor'], episode)
            if (state['mask_no_link_actions'] != expected_mask
                    or not math.isclose(state['no_link_logit_penalty'], expected_penalty, abs_tol=1e-12)):
                raise SystemExit('Checkpoint mask mode or logit penalty differs from the saved campaign episode')
            saved_schedule = state.get('no_link_logit_schedule')
            if ((saved_schedule is None and schedule['enabled'])
                    or (saved_schedule is not None and saved_schedule != schedule)):
                raise SystemExit('Checkpoint schedule differs from the saved campaign')
            completed = episode
            continue
        env['RACER_QWEN8_START_EPISODE'] = str(episode)
        env['RACER_QWEN8_EPISODES'] = str(episode)
        # A sequential campaign can reuse one isolated ROS domain indefinitely.
        env['RACER_QWEN8_DOMAIN_BASE'] = str(args.domain_base - episode + 1)
        if int(env['RACER_QWEN8_DOMAIN_BASE']) < 1:
            raise SystemExit('Choose a larger --domain-base for this episode count')
        if episode > 1:
            env['RACER_QWEN8_RESUME'] = str(output / f'episode_{episode-1:02d}/training/llm_prior_crpo_final.zip')
        else:
            env.pop('RACER_QWEN8_RESUME', None)
        progress('running', completed, episode)
        print(f'Running episode {episode}/{args.episodes}: {ep}', flush=True)
        with (output / 'campaign.log').open('a') as log:
            child = subprocess.Popen(['bash', str(ROOT / 'final_racer_training/scripts/run_event_driven_one_shot_campaign.sh')],
                env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        if code or not checkpoint.exists():
            progress('failed', completed, episode)
            raise SystemExit(f'Episode {episode} failed ({code}); see {output / "campaign.log"}')
        completed = episode
    progress('complete', completed)
    print(f'PORTABLE_CAMPAIGN_OK episodes={completed} output={output}', flush=True)


if __name__ == '__main__':
    main()
