#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${workspace}/third_party/nlopt-2.7.1"
build_dir="${workspace}/third_party/nlopt-build"
install_dir="${workspace}/third_party/install"

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${install_dir}" \
  -DNLOPT_PYTHON=OFF \
  -DNLOPT_GUILE=OFF \
  -DNLOPT_MATLAB=OFF \
  -DNLOPT_OCTAVE=OFF \
  -DNLOPT_SWIG=OFF
cmake --build "${build_dir}" --parallel "${RACER_BUILD_WORKERS:-4}"
cmake --install "${build_dir}"

printf 'NLopt installed in %s\n' "${install_dir}"
