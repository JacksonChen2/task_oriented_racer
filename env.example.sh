# Copy to env.local.sh, then fill in the four external installations.
export ISAAC_SIM_ROOT="/path/to/isaacsim"
export ROS_SETUP="/opt/ros/humble/setup.bash"
export QWEN_MODEL_PATH="/path/to/Qwen3-8B-FP8"
# Required local input for real training; no reference data is published in Git.
export RACER_QWEN8_PERFECT_RESULT="/path/to/local/reference.json"
# Python environment with torch, vLLM and the packages in requirements-training.txt.
export RACER_CRPO_PYTHON="/path/to/training-env/bin/python"
# Directory containing sionna/, mitsuba/ and drjit/ for ROS's Python ABI.
# For Humble on Ubuntu 22.04 this must be Python 3.10 compatible.
export SIONNA_RUNTIME_DIR="/path/to/sionna-runtime-or-site-packages"
export RACER_BUILD_WORKERS=4
export CUDA_VISIBLE_DEVICES=0
