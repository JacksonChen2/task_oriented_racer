#!/usr/bin/env bash
# Source this file; all internal paths follow this package's location.
RACER_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${RACER_PACKAGE_ROOT}/env.local.sh" ]]; then
  source "${RACER_PACKAGE_ROOT}/env.local.sh"
fi
export RACER_PACKAGE_ROOT
export ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
export RACER_FINAL_RACER_ROOT="${RACER_PACKAGE_ROOT}/final_racer"
export RACER_QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"
export RACER_QWEN8_MODEL_PATH="${QWEN_MODEL_PATH:-}"
export RACER_QWEN8_CONFIG="${RACER_PACKAGE_ROOT}/final_racer_training/config/train.yaml"
export PYTHONPATH="${RACER_PACKAGE_ROOT}/final_racer_training/agentic_crpo${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
