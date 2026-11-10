#!/usr/bin/env bash

# Configure the training process to use the NVML library installed with the
# active NVIDIA driver.  Old experiment launchers sometimes prepend a copied
# libnvidia-ml.so to LD_LIBRARY_PATH; leaving that entry first makes vLLM see
# an unspecified device after a driver update.
racer_configure_nvidia_driver_runtime() {
  local check_python="$1"
  local ldconfig_command=""
  local system_nvml=""
  local resolved_system_nvml=""
  local original_library_path=""
  local cleaned_library_path=""
  local entry=""
  local candidate=""
  local resolved_candidate=""
  local -a library_entries=()

  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig_command="$(command -v ldconfig)"
  elif [[ -x /sbin/ldconfig ]]; then
    ldconfig_command=/sbin/ldconfig
  else
    printf 'Unable to locate ldconfig for NVIDIA driver discovery.\n' >&2
    return 1
  fi

  # Read the complete ldconfig output instead of exiting awk on the first
  # match. With `set -o pipefail`, an early awk exit closes the pipe while
  # ldconfig is still writing and turns an otherwise successful lookup into
  # SIGPIPE (status 141).
  system_nvml="$(${ldconfig_command} -p 2>/dev/null | awk '
    $1 == "libnvidia-ml.so.1" {
      if (fallback == "") fallback = $NF
      if ($0 ~ /x86-64/ && selected == "") selected = $NF
    }
    END {
      if (selected != "") print selected
      else if (fallback != "") print fallback
    }
  ')"
  if [[ -z "${system_nvml}" || ! -e "${system_nvml}" ]]; then
    printf 'Unable to locate the system libnvidia-ml.so.1.\n' >&2
    return 1
  fi
  if [[ ! -x "${check_python}" ]]; then
    printf 'CUDA validation Python is not executable: %s\n' \
      "${check_python}" >&2
    return 1
  fi

  resolved_system_nvml="$(readlink -f "${system_nvml}")"
  original_library_path="${LD_LIBRARY_PATH:-}"
  IFS=: read -r -a library_entries <<<"${original_library_path}"
  for entry in "${library_entries[@]}"; do
    [[ -n "${entry}" ]] || continue
    candidate="${entry}/libnvidia-ml.so.1"
    if [[ -e "${candidate}" ]]; then
      resolved_candidate="$(readlink -f "${candidate}")"
      if [[ "${resolved_candidate}" != "${resolved_system_nvml}" ]]; then
        printf 'Ignoring stale NVML search path: %s\n' "${entry}" >&2
        continue
      fi
    fi
    cleaned_library_path="${cleaned_library_path:+${cleaned_library_path}:}${entry}"
  done
  if [[ -n "${cleaned_library_path}" ]]; then
    export LD_LIBRARY_PATH="${cleaned_library_path}"
  else
    unset LD_LIBRARY_PATH
  fi

  if ! "${check_python}" - <<'PY'
import torch
from vllm.platforms import current_platform

if current_platform.device_type != "cuda":
    raise SystemExit(
        "vLLM did not detect CUDA through the active system NVML"
    )
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable after loading the system NVML")
print(
    "RACER_NVIDIA_RUNTIME_OK "
    f"cuda_devices={torch.cuda.device_count()} "
    f"device={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY
  then
    printf 'System NVIDIA driver validation failed.\n' >&2
    return 1
  fi
}
