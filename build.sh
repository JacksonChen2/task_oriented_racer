#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/environment.sh"
: "${ISAAC_SIM_ROOT:?Set ISAAC_SIM_ROOT in env.local.sh}"
[[ -x "${ISAAC_SIM_ROOT}/python.sh" && -f "$ROS_SETUP" ]]
# Do not inherit another workspace or a training virtualenv's Python ABI.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH
export PATH="/usr/bin:/bin:${PATH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${RACER_BUILD_WORKERS:-4}"
set +u
source "$ROS_SETUP"
set -u
base="${RACER_PACKAGE_ROOT}/final_racer/ros2_ws"
overlay="${RACER_PACKAGE_ROOT}/final_racer_training/training_overlay_ws"
bash "$base/build_nlopt.sh"
cd "$base"
colcon build --base-paths src --symlink-install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
set +u
source "$base/install/setup.bash"
set -u
cd "$overlay"
colcon build --base-paths src --symlink-install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DISAAC_SIM_ROOT="$ISAAC_SIM_ROOT" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
printf 'PORTABLE_BUILD_OK\n'
