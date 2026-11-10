#!/usr/bin/env bash
set -euo pipefail

# Separate A/B entry point. The existing fixed-clock campaign script and its
# 100 ms / five-slot defaults are not modified by invoking this wrapper.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_root="$(realpath "${script_dir}/..")"

export RACER_RL_BS_EVENT_DRIVEN_ONE_SHOT=true
export RACER_QWEN8_SUPERVISOR_MODULE=agentic_crpo.event_driven_llm_prior_process_supervisor
export RACER_QWEN8_CONFIG="${RACER_QWEN8_CONFIG:-${training_root}/config/qwen8b_fp8_llm_action_prior_event_driven_one_shot_reference752763_traj020_red020_constraint012.yaml}"
export RACER_QWEN8_CAMPAIGN="${RACER_QWEN8_CAMPAIGN:-${training_root}/results/llm_action_prior_event_one_shot_$(date +%Y%m%d_%H%M%S)}"
# Match the current comparison configuration. Callers may still override any
# value explicitly, exactly as in the existing campaign entry point.
export RACER_QWEN8_LEARNING_RATE="${RACER_QWEN8_LEARNING_RATE:-0.00005}"
export RACER_QWEN8_N_STEPS="${RACER_QWEN8_N_STEPS:-512}"
export RACER_QWEN8_BATCH_SIZE="${RACER_QWEN8_BATCH_SIZE:-128}"
export RACER_QWEN8_N_EPOCHS="${RACER_QWEN8_N_EPOCHS:-5}"
export RACER_QWEN8_ACTIVATION_FN="${RACER_QWEN8_ACTIVATION_FN:-silu}"
export RACER_QWEN8_GAMMA_TASK="${RACER_QWEN8_GAMMA_TASK:-0.12}"

exec "${script_dir}/run_llm_prior_four_process_campaign.sh" "$@"
