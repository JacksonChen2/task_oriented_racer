#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/environment.sh"
exec "${RACER_CRPO_PYTHON:?Set RACER_CRPO_PYTHON in env.local.sh}" \
  "${RACER_PACKAGE_ROOT}/scripts/campaign.py" "$@"
