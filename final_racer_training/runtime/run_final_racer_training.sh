#!/usr/bin/env bash
set -euo pipefail

# ROS executables use the system Python ABI even when launched from a venv.
export PATH="/usr/bin:/bin:${PATH}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_root="$(realpath "${script_dir}/..")"
final_root="${RACER_FINAL_RACER_ROOT:-$(realpath "${training_root}/../final_racer")}" 
workspace_dir="${final_root}/ros2_ws"
adapter_overlay="${training_root}/training_overlay_ws/install/setup.bash"
if [[ ! -f "${workspace_dir}/install/setup.bash" ||
      ! -f "${adapter_overlay}" ]]; then
  printf 'Missing final_racer base or training overlays; run scripts/build.sh first.\n' >&2
  exit 2
fi

set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${workspace_dir}/install/setup.bash"
source "${adapter_overlay}"
set -u

adapter_share="$(ros2 pkg prefix racer_isaac_adapter)/share/racer_isaac_adapter"
communication_share="$(ros2 pkg prefix racer_sionna_comm)/share/racer_sionna_comm"
if [[ "$(ros2 pkg prefix racer_original_core)" != "${workspace_dir}/install/racer_original_core" ]] ||
   [[ "$(ros2 pkg prefix racer_sionna_comm)" != "${training_root}/training_overlay_ws/install/racer_sionna_comm" ]] ||
   [[ "$(ros2 pkg prefix racer_isaac_adapter)" != "${training_root}/training_overlay_ws/install/racer_isaac_adapter" ]]; then
  printf 'ROS overlay isolation check failed.\n' >&2
  exit 2
fi
repo_root="${final_root}/assets"
isaac_root="${ISAAC_SIM_ROOT:?Set ISAAC_SIM_ROOT in env.local.sh}"
driver_library_path="${RACER_NVIDIA_DRIVER_LIBRARY_PATH:-}"
isaac_bridge_library_path="${isaac_root}/exts/isaacsim.ros2.bridge/humble/lib"
if [[ -n "${driver_library_path}" ]]; then
  if [[ ! -d "${driver_library_path}" ]]; then
    printf 'RACER_NVIDIA_DRIVER_LIBRARY_PATH is not a directory: %s\n' \
      "${driver_library_path}" >&2
    exit 2
  fi
  isaac_library_path="${driver_library_path}:${isaac_bridge_library_path}"
else
  isaac_library_path="${isaac_bridge_library_path}"
fi
communication_mode="${RACER_COMMUNICATION_MODE:-sionna}"
if [[ -n "${RACER_REQUIRE_SIONNA:-}" ]]; then
  require_sionna="${RACER_REQUIRE_SIONNA}"
else
  require_sionna=$([[ "${communication_mode}" == "ideal" ]] && printf false || printf true)
fi
network_topology="${RACER_NETWORK_TOPOLOGY:-distributed}"
nearest_neighbor_count="${RACER_NEAREST_NEIGHBOR_COUNT:-0}"
lossless_nearest_neighbor_count="${RACER_LOSSLESS_NEAREST_NEIGHBOR_COUNT:-0}"
lossless_communication_range_m="${RACER_LOSSLESS_COMMUNICATION_RANGE_M:-0.0}"
lossless_control_only="${RACER_LOSSLESS_CONTROL_ONLY:-false}"
directed_message_unicast="${RACER_DIRECTED_MESSAGE_UNICAST:-false}"
uav_channel_access_mode="${RACER_UAV_CHANNEL_ACCESS_MODE:-csma}"
uav_csma_cw_min="${RACER_UAV_CSMA_CW_MIN:-15}"
uav_csma_cw_max="${RACER_UAV_CSMA_CW_MAX:-1023}"
uav_csma_difs_slots="${RACER_UAV_CSMA_DIFS_SLOTS:-2}"
chunk_data_pre_enqueue_dedup="${RACER_CHUNK_DATA_PRE_ENQUEUE_DEDUP:-false}"
chunk_data_max_pending_per_link="${RACER_CHUNK_DATA_MAX_PENDING_PER_LINK:-0}"
communication_range_m="${RACER_COMMUNICATION_RANGE_M:-4.0}"
ideal_coalesce_window_ms="${RACER_IDEAL_COALESCE_WINDOW_MS:-20.0}"
preserve_ideal_direct_with_bs="${RACER_PRESERVE_IDEAL_DIRECT_WITH_BS:-false}"
initial_assignment_perfect_delivery="${RACER_INITIAL_ASSIGNMENT_PERFECT_DELIVERY:-false}"
initial_assignment_perfect_via_bs="${RACER_INITIAL_ASSIGNMENT_PERFECT_VIA_BS:-false}"
initial_partition_enabled="${RACER_INITIAL_PARTITION_ENABLED:-false}"
launch_package="${RACER_LAUNCH_PACKAGE:-racer_sionna_comm}"
launch_file="${RACER_LAUNCH_FILE:-original_racer_warehouse_sionna.launch.py}"
algorithm_label="${RACER_ALGORITHM_LABEL:-original_racer_ros1_source_faithful_ros2_cpp_sionna_rt}"
algorithm_variant="${RACER_ALGORITHM_VARIANT:-original}"
exploration_assignment_mode="${RACER_EXPLORATION_ASSIGNMENT_MODE:-}"
lkh_dir="${RACER_LKH_DIR:-/tmp/racer_original_fidelity_sionna_lkh}"
sionna_runtime="${SIONNA_RUNTIME_DIR:-${final_root}/ros2_ws/.sionna_runtime}"
duration="${RACER_FIDELITY_DURATION:-900}"
wall_time_multiplier="${RACER_WALL_TIME_MULTIPLIER:-20}"
wall_time_grace="${RACER_WALL_TIME_GRACE_SECONDS:-300}"
drone_count="${RACER_FIDELITY_DRONE_COUNT:-5}"
trigger_minimum_cloud_frames="${RACER_TRIGGER_MINIMUM_CLOUD_FRAMES:-25}"
trigger_delay_s="${RACER_TRIGGER_DELAY_S:-5.0}"
coverage_update_rate_hz="${RACER_COVERAGE_UPDATE_RATE_HZ:-10.0}"
scenario="${RACER_FIDELITY_SCENARIO:-warehouse_simple}"
headless="${RACER_FIDELITY_HEADLESS:-1}"
visualize="${RACER_FIDELITY_VISUALIZE:-0}"
require_completion="${RACER_REQUIRE_COMPLETION:-1}"
stop_on_completion="${RACER_STOP_ON_COMPLETION:-1}"
coverage_target="${RACER_MAPPING_COVERAGE_TARGET:-0}"
# 640x480 with the upstream skip_pixel=2 produces at most 76,800 samples.
# Keeping them all is necessary for 0.1 m SDF free-space connectivity; a low
# diagnostic budget can leave unknown voxel curtains that the unchanged
# non-optimistic A* correctly refuses to cross.
ray_budget="${RACER_CAMERA_RAY_BUDGET:-76800}"
depth_sensor_backend="${RACER_DEPTH_SENSOR_BACKEND:-warp}"
sensor_profiling="${RACER_SENSOR_PROFILING:-1}"
physics_hz="${RACER_PHYSICS_RATE_HZ:-1000}"
sensor_hz="${RACER_SENSOR_RATE_HZ:-30}"
sensor_worker_count="${RACER_SENSOR_WORKER_COUNT:-1}"
scene_query_rate_hz="${RACER_SCENE_QUERY_RATE_HZ:-50.0}"
startup_free_space_yaw="${RACER_STARTUP_FREE_SPACE_YAW:-0}"
startup_scan_duration="${RACER_STARTUP_SCAN_DURATION:-0.0}"
startup_unknown_corridor_distance="${RACER_STARTUP_UNKNOWN_CORRIDOR_DISTANCE:-0.0}"
startup_corridor_speed="${RACER_STARTUP_CORRIDOR_SPEED:-0.25}"
startup_settle_duration="${RACER_STARTUP_SETTLE_DURATION:-1.0}"
depth_width="${RACER_DEPTH_WIDTH:-640}"
depth_height="${RACER_DEPTH_HEIGHT:-480}"
interactive_hz="${RACER_INTERACTIVE_RENDER_HZ:-30}"
map_points="${RACER_VISUALIZATION_MAX_MAP_POINTS:-12000}"
record_trajectory_history="${RACER_RECORD_TRAJECTORY_HISTORY:-0}"
result_dir="${RACER_RESULT_DIR:-${training_root}/validation}"
bs_channel_access_mode="${RACER_BS_CHANNEL_ACCESS_MODE:-scheduled}"
bs_downlink_source_owned_chunks_only="${RACER_BS_DOWNLINK_SOURCE_OWNED_CHUNKS_ONLY:-false}"
bs_tx_power_dbm="${RACER_BS_TX_POWER_DBM:-40.0}"
uav_tx_power_dbm="${RACER_UAV_TX_POWER_DBM:-23.0}"
carrier_frequency_hz="${RACER_CARRIER_FREQUENCY_HZ:-28000000000.0}"
bandwidth_hz="${RACER_BANDWIDTH_HZ:-100000000.0}"
resource_blocks="${RACER_RESOURCE_BLOCKS:-66}"
uav_broadcast_bandwidth_hz="${RACER_UAV_BROADCAST_BANDWIDTH_HZ:-50000000.0}"
uav_broadcast_resource_blocks="${RACER_UAV_BROADCAST_RESOURCE_BLOCKS:-33}"
bs_bandwidth_hz="${RACER_BS_BANDWIDTH_HZ:-50000000.0}"
bs_resource_blocks="${RACER_BS_RESOURCE_BLOCKS:-33}"
max_retries="${RACER_MAX_RETRIES:-3}"
bs_max_retries="${RACER_BS_MAX_RETRIES:-3}"
fixed_mcs_index="${RACER_FIXED_MCS_INDEX:--1}"
random_seed="${RACER_RANDOM_SEED:-42}"
start_positions="${RACER_START_POSITIONS:-}"
rl_bs_scheduler_enabled="${RACER_RL_BS_SCHEDULER_ENABLED:-false}"
force_bs_perfect_delivery="${RACER_FORCE_BS_PERFECT_DELIVERY:-false}"
bs_periodic_upload_request_enabled="${RACER_BS_PERIODIC_UPLOAD_REQUEST_ENABLED:-false}"
bs_periodic_upload_request_period_ms="${RACER_BS_PERIODIC_UPLOAD_REQUEST_PERIOD_MS:-200.0}"
bs_periodic_upload_chunks_per_request="${RACER_BS_PERIODIC_UPLOAD_CHUNKS_PER_REQUEST:-8}"
# Zero sizes the outstanding window from two current action-bound PHY slots;
# positive values retain a fixed chunk-count cap for controlled experiments.
bs_max_inflight_chunks_per_uav="${RACER_BS_MAX_INFLIGHT_CHUNKS_PER_UAV:-0}"
# Direct generation-time payload pushes are the normal path. This optional
# stamp-time precache worker is retained only for legacy/fallback experiments;
# action-time missing-payload requests remain active when it is disabled.
bs_payload_precache_enabled="${RACER_BS_PAYLOAD_PRECACHE_ENABLED:-false}"
bs_payload_precache_batch_size="${RACER_BS_PAYLOAD_PRECACHE_BATCH_SIZE:-8}"
bs_max_payload_precache_inflight="${RACER_BS_MAX_PAYLOAD_PRECACHE_INFLIGHT:-16}"
bs_payload_precache_timeout_ms="${RACER_BS_PAYLOAD_PRECACHE_TIMEOUT_MS:-5000.0}"
bs_max_uplink_chunks_per_rl_slot="${RACER_BS_MAX_UPLINK_CHUNKS_PER_RL_SLOT:-0}"
bs_max_downlink_chunks_per_rl_slot="${RACER_BS_MAX_DOWNLINK_CHUNKS_PER_RL_SLOT:-0}"
bs_downlink_newest_missing_chunk_first="${RACER_BS_DOWNLINK_NEWEST_MISSING_CHUNK_FIRST:-false}"
bs_control_ttl_s="${RACER_BS_CONTROL_TTL_S:-2.0}"
bs_all_to_all_relay_enabled="${RACER_BS_ALL_TO_ALL_RELAY_ENABLED:-false}"
pair_control_reservation_enabled="${RACER_PAIR_CONTROL_RESERVATION_ENABLED:-false}"
pair_control_reservation_s="${RACER_PAIR_CONTROL_RESERVATION_S:-1.8}"
bs_event_global_min_interval_s="${RACER_BS_EVENT_GLOBAL_MIN_INTERVAL_S:-10.0}"
bs_event_global_retry_interval_s="${RACER_BS_EVENT_GLOBAL_RETRY_INTERVAL_S:-0.5}"
bs_event_global_state_freshness_s="${RACER_BS_EVENT_GLOBAL_STATE_FRESHNESS_S:-2.0}"
bs_event_global_balance_weight_m="${RACER_BS_EVENT_GLOBAL_BALANCE_WEIGHT_M:-2.0}"
bs_event_global_load_gap="${RACER_BS_EVENT_GLOBAL_LOAD_GAP:-3}"
if [[ -z "${exploration_assignment_mode}" ]]; then
  if [[ "${rl_bs_scheduler_enabled}" == "true" &&
        "${network_topology}" == "bs_round_robin" ]]; then
    exploration_assignment_mode="bs_event_global"
  else
    exploration_assignment_mode="original"
  fi
fi
rl_bs_action_path="${RACER_RL_BS_ACTION_PATH:-/tmp/racer_agentic_crpo/action.txt}"
rl_bs_state_path="${RACER_RL_BS_STATE_PATH:-/tmp/racer_agentic_crpo/communication_state.json}"
rl_bs_mission_state_path="${RACER_RL_BS_MISSION_STATE_PATH:-/tmp/racer_agentic_crpo/mission_state.json}"
rl_bs_decision_period_ms="${RACER_RL_BS_DECISION_PERIOD_MS:-20.0}"
rl_bs_repeat_action_each_slot="${RACER_RL_BS_REPEAT_ACTION_EACH_SLOT:-true}"
rl_bs_communication_slot_ms="${RACER_RL_BS_COMMUNICATION_SLOT_MS:-20.0}"
rl_llm_state_period_ms="${RACER_RL_LLM_STATE_PERIOD_MS:-5000.0}"
rl_sync_enabled="${RACER_RL_SYNC_ENABLED:-false}"
rl_bs_event_driven_one_shot="${RACER_RL_BS_EVENT_DRIVEN_ONE_SHOT:-false}"
rl_sync_release_path="${RACER_RL_SYNC_RELEASE_PATH:-}"
rl_sync_ack_path="${RACER_RL_SYNC_ACK_PATH:-}"
rl_sync_maximum_wait_s="${RACER_RL_SYNC_MAXIMUM_WAIT_S:-1200.0}"
rl_sync_poll_s="${RACER_RL_SYNC_POLL_S:-0.002}"
single_gpu_pause_request_path="${RACER_SINGLE_GPU_PAUSE_REQUEST_PATH:-}"
single_gpu_pause_ack_path="${RACER_SINGLE_GPU_PAUSE_ACK_PATH:-}"
single_gpu_maximum_pause_s="${RACER_SINGLE_GPU_MAXIMUM_PAUSE_S:-600.0}"
single_gpu_pause_poll_s="${RACER_SINGLE_GPU_PAUSE_POLL_S:-0.01}"
ground_truth_occupied_voxels_path="${RACER_GROUND_TRUTH_OCCUPIED_VOXELS_PATH:-}"
observed_occupied_voxels_path="${RACER_OBSERVED_OCCUPIED_VOXELS_PATH:-}"
require_ground_truth_map="${RACER_REQUIRE_GROUND_TRUTH_MAP:-false}"
task_metric_observer_mode="${RACER_TASK_METRIC_OBSERVER_MODE:-inline}"
process_role="${RACER_PROCESS_ROLE:-combined}"
shared_memory_root="${RACER_SHARED_MEMORY_ROOT:-}"
if [[ "${process_role}" != "combined" && "${process_role}" != "racer" &&
      "${process_role}" != "isaac" ]]; then
  printf 'RACER_PROCESS_ROLE must be combined, racer, or isaac.\n' >&2
  exit 2
fi
if [[ "${network_topology}" != "distributed" && "${network_topology}" != "nearest_neighbors" && "${network_topology}" != "distance_radius" && "${network_topology}" != "ap_assisted" && "${network_topology}" != "bs_round_robin" ]]; then
  printf 'RACER_NETWORK_TOPOLOGY must be distributed, nearest_neighbors, distance_radius, ap_assisted, or bs_round_robin.\n' >&2
  exit 2
fi
if [[ "${rl_bs_scheduler_enabled}" != "true" && "${rl_bs_scheduler_enabled}" != "false" ]]; then
  printf 'RACER_RL_BS_SCHEDULER_ENABLED must be true or false.\n' >&2
  exit 2
fi
if [[ "${force_bs_perfect_delivery}" != "true" &&
      "${force_bs_perfect_delivery}" != "false" ]]; then
  printf 'RACER_FORCE_BS_PERFECT_DELIVERY must be true or false.\n' >&2
  exit 2
fi
if [[ "${bs_periodic_upload_request_enabled}" != "true" &&
      "${bs_periodic_upload_request_enabled}" != "false" ]]; then
  printf 'RACER_BS_PERIODIC_UPLOAD_REQUEST_ENABLED must be true or false.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' \
  "${bs_periodic_upload_request_period_ms}"; then
  printf 'RACER_BS_PERIODIC_UPLOAD_REQUEST_PERIOD_MS must be finite and positive.\n' >&2
  exit 2
fi
if ! [[ "${bs_periodic_upload_chunks_per_request}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${bs_max_inflight_chunks_per_uav}" =~ ^[0-9]+$ ]] ||
   ! [[ "${bs_max_uplink_chunks_per_rl_slot}" =~ ^[0-9]+$ ]] ||
   ! [[ "${bs_max_downlink_chunks_per_rl_slot}" =~ ^[0-9]+$ ]] ||
   (( bs_max_inflight_chunks_per_uav > 0 && bs_periodic_upload_chunks_per_request > bs_max_inflight_chunks_per_uav )); then
  printf 'BS fixed budgets must be nonnegative; zero selects automatic sizing; periodic chunk budget must be <= a positive per-UAV in-flight limit.\n' >&2
  exit 2
fi
if [[ "${bs_downlink_newest_missing_chunk_first}" != "true" &&
      "${bs_downlink_newest_missing_chunk_first}" != "false" ]]; then
  printf 'RACER_BS_DOWNLINK_NEWEST_MISSING_CHUNK_FIRST must be true or false.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' \
  "${bs_control_ttl_s}"; then
  printf 'RACER_BS_CONTROL_TTL_S must be finite and positive.\n' >&2
  exit 2
fi
if [[ "${bs_all_to_all_relay_enabled}" != "true" &&
      "${bs_all_to_all_relay_enabled}" != "false" ]]; then
  printf 'RACER_BS_ALL_TO_ALL_RELAY_ENABLED must be true or false.\n' >&2
  exit 2
fi
if [[ "${pair_control_reservation_enabled}" != "true" &&
      "${pair_control_reservation_enabled}" != "false" ]]; then
  printf 'RACER_PAIR_CONTROL_RESERVATION_ENABLED must be true or false.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' \
  "${pair_control_reservation_s}"; then
  printf 'RACER_PAIR_CONTROL_RESERVATION_S must be finite and positive.\n' >&2
  exit 2
fi
if [[ "${pair_control_reservation_enabled}" == "true" &&
      ( "${rl_bs_scheduler_enabled}" != "true" ||
        "${network_topology}" != "bs_round_robin" ) ]]; then
  printf 'RACER_PAIR_CONTROL_RESERVATION_ENABLED=true requires the RL BS scheduler and bs_round_robin topology.\n' >&2
  exit 2
fi
if [[ "${force_bs_perfect_delivery}" == "true" &&
      "${network_topology}" != "bs_round_robin" &&
      "${network_topology}" != "ap_assisted" ]]; then
  printf 'RACER_FORCE_BS_PERFECT_DELIVERY=true requires a BS-assisted topology.\n' >&2
  exit 2
fi
if [[ "${preserve_ideal_direct_with_bs}" != "true" &&
      "${preserve_ideal_direct_with_bs}" != "false" ]]; then
  printf 'RACER_PRESERVE_IDEAL_DIRECT_WITH_BS must be true or false.\n' >&2
  exit 2
fi
if [[ "${initial_assignment_perfect_delivery}" != "true" &&
      "${initial_assignment_perfect_delivery}" != "false" ]]; then
  printf 'RACER_INITIAL_ASSIGNMENT_PERFECT_DELIVERY must be true or false.\n' >&2
  exit 2
fi
if [[ "${initial_assignment_perfect_via_bs}" != "true" &&
      "${initial_assignment_perfect_via_bs}" != "false" ]]; then
  printf 'RACER_INITIAL_ASSIGNMENT_PERFECT_VIA_BS must be true or false.\n' >&2
  exit 2
fi
if [[ "${initial_partition_enabled}" != "true" &&
      "${initial_partition_enabled}" != "false" ]]; then
  printf 'RACER_INITIAL_PARTITION_ENABLED must be true or false.\n' >&2
  exit 2
fi
if [[ "${initial_assignment_perfect_via_bs}" == "true" &&
      "${initial_assignment_perfect_delivery}" != "true" ]]; then
  printf 'RACER_INITIAL_ASSIGNMENT_PERFECT_VIA_BS requires RACER_INITIAL_ASSIGNMENT_PERFECT_DELIVERY=true.\n' >&2
  exit 2
fi
if [[ "${initial_assignment_perfect_delivery}" == "true" &&
      "${communication_mode}" == "ideal" ]]; then
  printf 'RACER_INITIAL_ASSIGNMENT_PERFECT_DELIVERY=true requires non-ideal communication.\n' >&2
  exit 2
fi
if [[ "${initial_assignment_perfect_delivery}" == "true" &&
      "${initial_assignment_perfect_via_bs}" != "true" &&
      "${network_topology}" != "distributed" ]]; then
  printf 'Direct initial perfect delivery requires distributed topology.\n' >&2
  exit 2
fi
if [[ "${initial_assignment_perfect_via_bs}" == "true" &&
      "${network_topology}" != "bs_round_robin" &&
      "${network_topology}" != "ap_assisted" ]]; then
  printf 'RACER_INITIAL_ASSIGNMENT_PERFECT_VIA_BS=true requires a BS-assisted topology.\n' >&2
  exit 2
fi
if [[ "${preserve_ideal_direct_with_bs}" == "true" ]] &&
   { [[ "${communication_mode}" != "ideal" ]] ||
     [[ "${network_topology}" != "bs_round_robin" ]] ||
     [[ "${rl_bs_scheduler_enabled}" != "true" ]]; }; then
  printf 'RACER_PRESERVE_IDEAL_DIRECT_WITH_BS=true requires ideal mode, bs_round_robin, and the RL BS scheduler.\n' >&2
  exit 2
fi
if [[ "${rl_sync_enabled}" != "true" && "${rl_sync_enabled}" != "false" ]]; then
  printf 'RACER_RL_SYNC_ENABLED must be true or false.\n' >&2
  exit 2
fi
if [[ "${rl_bs_event_driven_one_shot}" != "true" &&
      "${rl_bs_event_driven_one_shot}" != "false" ]]; then
  printf 'RACER_RL_BS_EVENT_DRIVEN_ONE_SHOT must be true or false.\n' >&2
  exit 2
fi
if [[ "${rl_bs_event_driven_one_shot}" == "true" &&
      ( "${rl_bs_scheduler_enabled}" != "true" ||
        "${rl_sync_enabled}" == "true" ||
        -z "${shared_memory_root}" ) ]]; then
  printf 'Event-driven one-shot mode requires shared-memory RL scheduling and cannot use fixed-clock synchronous mode.\n' >&2
  exit 2
fi
if [[ "${rl_sync_enabled}" == "true" ]]; then
  if [[ "${rl_bs_scheduler_enabled}" != "true" ||
        "${stop_on_completion}" != "0" ]]; then
    printf 'Synchronous RL requires the RL BS scheduler and a fixed exploration horizon.\n' >&2
    exit 2
  fi
  if [[ -z "${rl_sync_release_path}" || -z "${rl_sync_ack_path}" ]]; then
    printf 'Synchronous RL requires release and acknowledgement paths.\n' >&2
    exit 2
  fi
  if ! python3 -c 'import math,sys; comm,decision,max_wait,poll=map(float,sys.argv[1:]); ok=all(math.isfinite(v) and v>0 for v in (comm,decision,max_wait,poll)) and math.isclose(decision,5*comm,rel_tol=0,abs_tol=1e-9); raise SystemExit(not ok)' \
    "${rl_bs_communication_slot_ms}" "${rl_bs_decision_period_ms}" \
    "${rl_sync_maximum_wait_s}" "${rl_sync_poll_s}"; then
    printf 'Synchronous RL requires T_RL=5*T_comm with positive finite intervals.\n' >&2
    exit 2
  fi
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' \
  "${coverage_update_rate_hz}"; then
  printf 'RACER_COVERAGE_UPDATE_RATE_HZ must be finite and positive.\n' >&2
  exit 2
fi
if [[ "${require_ground_truth_map}" != "true" && "${require_ground_truth_map}" != "false" ]]; then
  printf 'RACER_REQUIRE_GROUND_TRUTH_MAP must be true or false.\n' >&2
  exit 2
fi
if [[ "${task_metric_observer_mode}" != "inline" &&
      "${task_metric_observer_mode}" != "async" &&
      "${task_metric_observer_mode}" != "off" ]]; then
  printf 'RACER_TASK_METRIC_OBSERVER_MODE must be inline, async, or off.\n' >&2
  exit 2
fi
if [[ "${require_ground_truth_map}" == "true" && ! -f "${ground_truth_occupied_voxels_path}" ]]; then
  printf 'RACER_REQUIRE_GROUND_TRUTH_MAP=true requires an existing RACER_GROUND_TRUTH_OCCUPIED_VOXELS_PATH.\n' >&2
  exit 2
fi
if [[ -n "${single_gpu_pause_request_path}" || -n "${single_gpu_pause_ack_path}" ]]; then
  if [[ -z "${single_gpu_pause_request_path}" || -z "${single_gpu_pause_ack_path}" ]]; then
    printf 'Both RACER_SINGLE_GPU_PAUSE_REQUEST_PATH and RACER_SINGLE_GPU_PAUSE_ACK_PATH are required.\n' >&2
    exit 2
  fi
  if ! python3 -c 'import math,sys; values=[float(v) for v in sys.argv[1:]]; raise SystemExit(not all(math.isfinite(v) and v > 0.0 for v in values))' "${single_gpu_maximum_pause_s}" "${single_gpu_pause_poll_s}"; then
    printf 'Single-GPU pause timeout and polling interval must be positive finite values.\n' >&2
    exit 2
  fi
fi
if [[ -n "${shared_memory_root}" ]]; then
  # The four-process runtime is deliberately asynchronous. Refuse a mixed
  # configuration instead of allowing a stale environment variable to turn
  # either of the legacy Isaac freeze gates back on.
  if [[ "${rl_sync_enabled}" != "false" ]]; then
    printf 'Shared-memory four-process mode forbids RACER_RL_SYNC_ENABLED=true; Isaac must keep advancing.\n' >&2
    exit 2
  fi
  if [[ -n "${single_gpu_pause_request_path}" || -n "${single_gpu_pause_ack_path}" ]]; then
    printf 'Shared-memory four-process mode forbids single-GPU pause files; GPU contention must not freeze Isaac.\n' >&2
    exit 2
  fi
fi
if [[ "${rl_bs_scheduler_enabled}" == "true" && "${network_topology}" != "bs_round_robin" ]]; then
  printf 'RACER_RL_BS_SCHEDULER_ENABLED=true requires RACER_NETWORK_TOPOLOGY=bs_round_robin.\n' >&2
  exit 2
fi
if [[ "${bs_payload_precache_enabled}" != "true" &&
      "${bs_payload_precache_enabled}" != "false" ]]; then
  printf 'RACER_BS_PAYLOAD_PRECACHE_ENABLED must be true or false.\n' >&2
  exit 2
fi
if ! [[ "${bs_payload_precache_batch_size}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${bs_max_payload_precache_inflight}" =~ ^[1-9][0-9]*$ ]] ||
   (( bs_payload_precache_batch_size > bs_max_payload_precache_inflight )); then
  printf 'Payload precache batch/in-flight limits must be positive and batch must not exceed in-flight.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' "${bs_payload_precache_timeout_ms}"; then
  printf 'RACER_BS_PAYLOAD_PRECACHE_TIMEOUT_MS must be positive and finite.\n' >&2
  exit 2
fi
if [[ "${require_sionna}" != "true" && "${require_sionna}" != "false" ]]; then
  printf 'RACER_REQUIRE_SIONNA must be true or false.\n' >&2
  exit 2
fi
if [[ "${communication_mode}" == "ideal" && "${require_sionna}" != "false" ]]; then
  printf 'Ideal communication mode cannot require Sionna.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' "${ideal_coalesce_window_ms}"; then
  printf 'RACER_IDEAL_COALESCE_WINDOW_MS must be a positive finite number.\n' >&2
  exit 2
fi
if [[ "${network_topology}" == "nearest_neighbors" ]] &&
   { [[ ! "${nearest_neighbor_count}" =~ ^[1-9][0-9]*$ ]] ||
     (( nearest_neighbor_count >= drone_count )); }; then
  printf 'RACER_NEAREST_NEIGHBOR_COUNT must be in [1, drone_count - 1].\n' >&2
  exit 2
fi
if [[ ! "${lossless_nearest_neighbor_count}" =~ ^[0-9]+$ ]] ||
   (( lossless_nearest_neighbor_count >= drone_count )); then
  printf 'RACER_LOSSLESS_NEAREST_NEIGHBOR_COUNT must be in [0, drone_count - 1].\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value >= 0.0))' "${lossless_communication_range_m}"; then
  printf 'RACER_LOSSLESS_COMMUNICATION_RANGE_M must be finite and non-negative.\n' >&2
  exit 2
fi
if [[ "${lossless_control_only}" != "true" && "${lossless_control_only}" != "false" ]]; then
  printf 'RACER_LOSSLESS_CONTROL_ONLY must be true or false.\n' >&2
  exit 2
fi
if [[ "${directed_message_unicast}" != "true" && "${directed_message_unicast}" != "false" ]]; then
  printf 'RACER_DIRECTED_MESSAGE_UNICAST must be true or false.\n' >&2
  exit 2
fi
if [[ "${uav_channel_access_mode}" != "ofdma" &&
      "${uav_channel_access_mode}" != "csma" ]]; then
  printf 'RACER_UAV_CHANNEL_ACCESS_MODE must be ofdma or csma.\n' >&2
  exit 2
fi
if [[ "${bs_channel_access_mode}" != "scheduled" &&
      "${bs_channel_access_mode}" != "csma" &&
      "${bs_channel_access_mode}" != "ofdma" ]]; then
  printf 'RACER_BS_CHANNEL_ACCESS_MODE must be scheduled, csma, or ofdma.\n' >&2
  exit 2
fi
if [[ ! "${uav_csma_cw_min}" =~ ^[0-9]+$ ||
      ! "${uav_csma_cw_max}" =~ ^[0-9]+$ ||
      ! "${uav_csma_difs_slots}" =~ ^[0-9]+$ ]] ||
   (( uav_csma_cw_max < uav_csma_cw_min )); then
  printf 'CSMA contention windows and DIFS slots must be non-negative integers with CWmax >= CWmin.\n' >&2
  exit 2
fi
if [[ "${chunk_data_pre_enqueue_dedup}" != "true" && "${chunk_data_pre_enqueue_dedup}" != "false" ]]; then
  printf 'RACER_CHUNK_DATA_PRE_ENQUEUE_DEDUP must be true or false.\n' >&2
  exit 2
fi
if [[ ! "${chunk_data_max_pending_per_link}" =~ ^[0-9]+$ ]]; then
  printf 'RACER_CHUNK_DATA_MAX_PENDING_PER_LINK must be a non-negative integer.\n' >&2
  exit 2
fi
if { (( lossless_nearest_neighbor_count > 0 )) ||
     python3 -c 'import sys; raise SystemExit(not (float(sys.argv[1]) > 0.0))' "${lossless_communication_range_m}"; } &&
   { [[ "${communication_mode}" == "ideal" ]] ||
     [[ "${network_topology}" != "distributed" ]]; }; then
  printf 'Lossless UAV-link overrides require non-ideal distributed communication.\n' >&2
  exit 2
fi
if [[ "${network_topology}" == "distance_radius" ]] &&
   ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' "${communication_range_m}"; then
  printf 'RACER_COMMUNICATION_RANGE_M must be a positive finite number.\n' >&2
  exit 2
fi
if [[ ! "${wall_time_multiplier}" =~ ^[1-9][0-9]*$ || ! "${wall_time_grace}" =~ ^[0-9]+$ ]]; then
  printf 'RACER_WALL_TIME_MULTIPLIER must be a positive integer and grace must be non-negative.\n' >&2
  exit 2
fi
if [[ ! "${max_retries}" =~ ^[0-9]+$ || ! "${bs_max_retries}" =~ ^[0-9]+$ ]]; then
  printf 'RACER_MAX_RETRIES and RACER_BS_MAX_RETRIES must be non-negative integers.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' "${carrier_frequency_hz}"; then
  printf 'RACER_CARRIER_FREQUENCY_HZ must be a positive finite number.\n' >&2
  exit 2
fi
if [[ ! "${resource_blocks}" =~ ^[1-9][0-9]*$ ]] ||
   ! python3 -c 'import math,sys; bandwidth=float(sys.argv[1]); resource_blocks=int(sys.argv[2]); occupied=12.0*120000.0*resource_blocks; raise SystemExit(not (math.isfinite(bandwidth) and bandwidth > 0.0 and occupied <= bandwidth))' "${bandwidth_hz}" "${resource_blocks}"; then
  printf 'RACER_BANDWIDTH_HZ and RACER_RESOURCE_BLOCKS must define a valid 120 kHz NR allocation.\n' >&2
  exit 2
fi
if [[ ! "${uav_broadcast_resource_blocks}" =~ ^[1-9][0-9]*$ ||
      ! "${bs_resource_blocks}" =~ ^[1-9][0-9]*$ ]] ||
   ! python3 -c 'import math,sys; total=float(sys.argv[1]); uav=float(sys.argv[2]); bs=float(sys.argv[3]); total_rb=int(sys.argv[4]); uav_rb=int(sys.argv[5]); bs_rb=int(sys.argv[6]); valid=all(math.isfinite(x) and x>0 for x in (total,uav,bs)) and abs((uav+bs)-total)<=1e-6*total and uav_rb+bs_rb==total_rb and 12*120000*uav_rb<=uav and 12*120000*bs_rb<=bs; raise SystemExit(not valid)' "${bandwidth_hz}" "${uav_broadcast_bandwidth_hz}" "${bs_bandwidth_hz}" "${resource_blocks}" "${uav_broadcast_resource_blocks}" "${bs_resource_blocks}"; then
  printf 'UAV and BS subbands must exactly partition total bandwidth and PRBs.\n' >&2
  exit 2
fi
if [[ "${fixed_mcs_index}" != "-1" && "${fixed_mcs_index}" != "14" &&
      "${fixed_mcs_index}" != "20" ]]; then
  printf 'RACER_FIXED_MCS_INDEX must be -1 (adaptive), 14, or 20.\n' >&2
  exit 2
fi
if [[ "${exploration_assignment_mode}" != "original" &&
      "${exploration_assignment_mode}" != "bs_event_global" &&
      "${exploration_assignment_mode}" != "global_cooperative" ]]; then
  printf 'RACER_EXPLORATION_ASSIGNMENT_MODE must be original, bs_event_global, or global_cooperative.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; values=map(float,sys.argv[1:]); raise SystemExit(not all(math.isfinite(value) and value > 0.0 for value in values))' \
  "${bs_event_global_min_interval_s}" "${bs_event_global_retry_interval_s}" \
  "${bs_event_global_state_freshness_s}" "${bs_event_global_balance_weight_m}" ||
   [[ ! "${bs_event_global_load_gap}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BS event-global intervals, freshness, balance weight, and load gap must be positive.\n' >&2
  exit 2
fi
if [[ "${exploration_assignment_mode}" == "bs_event_global" ]] &&
   { [[ "${network_topology}" != "bs_round_robin" ]] ||
     [[ "${rl_bs_scheduler_enabled}" != "true" ]] ||
     [[ "${initial_assignment_perfect_delivery}" != "true" ]] ||
     [[ "${initial_assignment_perfect_via_bs}" != "true" ]]; }; then
  printf 'bs_event_global requires bs_round_robin, the RL BS scheduler, and perfect initial assignment via BS.\n' >&2
  exit 2
fi
if [[ "${exploration_assignment_mode}" == "bs_event_global" &&
      "${pair_control_reservation_enabled}" == "true" ]]; then
  printf 'bs_event_global disables BS PairOpt/PairOptResponse and cannot enable pair control reservation.\n' >&2
  exit 2
fi
if [[ ! "${sensor_worker_count}" =~ ^[1-9][0-9]*$ ]] ||
   (( sensor_worker_count > drone_count )); then
  printf 'RACER_SENSOR_WORKER_COUNT must be in [1, drone_count].\n' >&2
  exit 2
fi
if [[ "${depth_sensor_backend}" != "warp" && "${depth_sensor_backend}" != "rtx" ]]; then
  printf 'RACER_DEPTH_SENSOR_BACKEND must be warp or rtx.\n' >&2
  exit 2
fi
if [[ "${sensor_profiling}" != "0" && "${sensor_profiling}" != "1" ]]; then
  printf 'RACER_SENSOR_PROFILING must be 0 or 1.\n' >&2
  exit 2
fi
if [[ ! "${trigger_minimum_cloud_frames}" =~ ^[0-9]+$ ]]; then
  printf 'RACER_TRIGGER_MINIMUM_CLOUD_FRAMES must be a non-negative integer.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value >= 0.0))' "${trigger_delay_s}"; then
  printf 'RACER_TRIGGER_DELAY_S must be finite and non-negative.\n' >&2
  exit 2
fi
if [[ "${startup_free_space_yaw}" != "0" && "${startup_free_space_yaw}" != "1" ]]; then
  printf 'RACER_STARTUP_FREE_SPACE_YAW must be 0 or 1.\n' >&2
  exit 2
fi
if ! python3 -c 'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' "${scene_query_rate_hz}"; then
  printf 'RACER_SCENE_QUERY_RATE_HZ must be a positive finite number.\n' >&2
  exit 2
fi
mkdir -p "${result_dir}" "${lkh_dir}"
run_tag="${scenario}_${network_topology}"
launch_log="${result_dir}/${run_tag}_launch.log"
isaac_log="${result_dir}/${run_tag}_isaac.log"
result_file="${result_dir}/${run_tag}_result.json"
if [[ "${process_role}" != "isaac" ]]; then : > "${launch_log}"; fi
if [[ "${process_role}" != "racer" ]]; then : > "${isaac_log}"; fi

if [[ "${scenario}" == "warehouse_full" ]]; then
  default_scene_usd="${repo_root}/warehouse_scenes/isaac/warehouse_full_with_industrial_ap.usda"
  default_sionna_scene_xml="${repo_root}/sionna_scene/warehouse.xml"
  default_radio_map_cache="${training_root}/cache/no_radio_map_cache.npz"
  default_ap_position_x="-10.02891489217081"
  default_ap_position_y="14.888611215255622"
  default_ap_position_z="7.55"
elif [[ "${scenario}" == "warehouse_loaded_full" ]]; then
  default_scene_usd="${repo_root}/warehouse_scenes/isaac/warehouse_loaded_full_with_industrial_ap.usda"
  default_sionna_scene_xml="${workspace_dir}/src/racer_sionna_comm/assets/warehouse_loaded_sionna/warehouse.xml"
  default_radio_map_cache="${workspace_dir}/src/racer_sionna_comm/assets/warehouse_loaded_sionna/hybrid_radio_cache.npz"
elif [[ "${scenario}" == "warehouse_loaded" || "${scenario}" == "warehouse_loaded_center" ]]; then
  # This layer lives next to its relative warehouse.usd dependency.
  default_scene_usd="${repo_root}/ros2_3d_py_ws/warehouse_loaded_with_industrial_ap.usda"
  default_sionna_scene_xml="${workspace_dir}/src/racer_sionna_comm/assets/warehouse_loaded_sionna/warehouse.xml"
  default_radio_map_cache="${workspace_dir}/src/racer_sionna_comm/assets/warehouse_loaded_sionna/hybrid_radio_cache.npz"
elif [[ "${scenario}" == "warehouse_simple" ]]; then
  default_scene_usd="${repo_root}/ros2_3d_py_ws/warehouse_simple_with_industrial_ap.usda"
  default_sionna_scene_xml="${workspace_dir}/src/racer_sionna_comm/assets/warehouse_simple_with_ap_sionna/warehouse.xml"
  default_radio_map_cache="${workspace_dir}/src/racer_sionna_comm/assets/warehouse_simple_with_ap_sionna/hybrid_radio_cache.npz"
else
  printf 'Unsupported RACER_FIDELITY_SCENARIO: %s\n' "${scenario}" >&2
  exit 2
fi
ap_position_x="${RACER_AP_POSITION_X:-${default_ap_position_x:-}}"
ap_position_y="${RACER_AP_POSITION_Y:-${default_ap_position_y:-}}"
ap_position_z="${RACER_AP_POSITION_Z:-${default_ap_position_z:-}}"
scene_usd="${RACER_SCENE_USD:-${default_scene_usd}}"
sionna_scene_xml="${RACER_SIONNA_SCENE_XML:-${default_sionna_scene_xml}}"
radio_map_cache="${RACER_SIONNA_RADIO_MAP_CACHE:-${default_radio_map_cache}}"
vehicle_usd="${RACER_VEHICLE_USD:-${repo_root}/isaac_assets/racer_so3_quadrotor/usd/crazyflie_with_racer_dynamics.usd}"
if [[ ! -x "${isaac_root}/python.sh" || ! -f "${scene_usd}" || ! -f "${vehicle_usd}" ]]; then
  printf 'Missing Isaac Python, scene USD, or vehicle USD.\n' >&2
  exit 2
fi
if [[ "${communication_mode}" != "ideal" ]]; then
  if [[ ! -d "${sionna_runtime}/sionna" || ! -f "${sionna_scene_xml}" ]]; then
    printf 'Missing isolated Sionna RT runtime or Warehouse XML scene.\n' >&2
    printf 'Run ./setup_sionna_env.sh and ./prepare_warehouse_simple_sionna_scene.sh first.\n' >&2
    exit 2
  fi
  export PYTHONPATH="${sionna_runtime}${PYTHONPATH:+:${PYTHONPATH}}"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-176}"
launch_communication_args=(
  coverage_update_rate_hz:="${coverage_update_rate_hz}"
  communication_mode:="${communication_mode}"
  network_topology:="${network_topology}"
  nearest_neighbor_count:="${nearest_neighbor_count}"
  lossless_nearest_neighbor_count:="${lossless_nearest_neighbor_count}"
  lossless_communication_range_m:="${lossless_communication_range_m}"
  lossless_control_only:="${lossless_control_only}"
  directed_message_unicast:="${directed_message_unicast}"
  uav_channel_access_mode:="${uav_channel_access_mode}"
  uav_csma_cw_min:="${uav_csma_cw_min}"
  uav_csma_cw_max:="${uav_csma_cw_max}"
  uav_csma_difs_slots:="${uav_csma_difs_slots}"
  chunk_data_pre_enqueue_dedup:="${chunk_data_pre_enqueue_dedup}"
  chunk_data_max_pending_per_link:="${chunk_data_max_pending_per_link}"
  communication_range_m:="${communication_range_m}"
  ideal_coalesce_window_ms:="${ideal_coalesce_window_ms}"
  preserve_ideal_direct_with_bs:="${preserve_ideal_direct_with_bs}"
  initial_assignment_perfect_delivery:="${initial_assignment_perfect_delivery}"
  initial_assignment_perfect_via_bs:="${initial_assignment_perfect_via_bs}"
  initial_partition_enabled:="${initial_partition_enabled}"
  require_sionna:="${require_sionna}"
  sionna_scene_xml:="${sionna_scene_xml}"
  bs_channel_access_mode:="${bs_channel_access_mode}"
  bs_downlink_source_owned_chunks_only:="${bs_downlink_source_owned_chunks_only}"
  ap_tx_power_dbm:="${bs_tx_power_dbm}"
  uav_tx_power_dbm:="${uav_tx_power_dbm}"
  carrier_frequency_hz:="${carrier_frequency_hz}"
  bandwidth_hz:="${bandwidth_hz}"
  resource_blocks:="${resource_blocks}"
  uav_broadcast_bandwidth_hz:="${uav_broadcast_bandwidth_hz}"
  uav_broadcast_resource_blocks:="${uav_broadcast_resource_blocks}"
  bs_bandwidth_hz:="${bs_bandwidth_hz}"
  bs_resource_blocks:="${bs_resource_blocks}"
  max_retries:="${max_retries}"
  bs_max_retries:="${bs_max_retries}"
  fixed_mcs_index:="${fixed_mcs_index}"
  rl_bs_scheduler_enabled:="${rl_bs_scheduler_enabled}"
  force_bs_perfect_delivery:="${force_bs_perfect_delivery}"
  bs_periodic_upload_request_enabled:="${bs_periodic_upload_request_enabled}"
  bs_periodic_upload_request_period_ms:="${bs_periodic_upload_request_period_ms}"
  bs_periodic_upload_chunks_per_request:="${bs_periodic_upload_chunks_per_request}"
  bs_max_inflight_chunks_per_uav:="${bs_max_inflight_chunks_per_uav}"
  bs_payload_precache_enabled:="${bs_payload_precache_enabled}"
  bs_payload_precache_batch_size:="${bs_payload_precache_batch_size}"
  bs_max_payload_precache_inflight:="${bs_max_payload_precache_inflight}"
  bs_payload_precache_timeout_ms:="${bs_payload_precache_timeout_ms}"
  bs_max_uplink_chunks_per_rl_slot:="${bs_max_uplink_chunks_per_rl_slot}"
  bs_max_downlink_chunks_per_rl_slot:="${bs_max_downlink_chunks_per_rl_slot}"
  bs_downlink_newest_missing_chunk_first:="${bs_downlink_newest_missing_chunk_first}"
  bs_control_ttl_s:="${bs_control_ttl_s}"
  bs_all_to_all_relay_enabled:="${bs_all_to_all_relay_enabled}"
  pair_control_reservation_enabled:="${pair_control_reservation_enabled}"
  pair_control_reservation_s:="${pair_control_reservation_s}"
  bs_event_global_min_interval_s:="${bs_event_global_min_interval_s}"
  bs_event_global_retry_interval_s:="${bs_event_global_retry_interval_s}"
  bs_event_global_state_freshness_s:="${bs_event_global_state_freshness_s}"
  bs_event_global_balance_weight_m:="${bs_event_global_balance_weight_m}"
  bs_event_global_load_gap:="${bs_event_global_load_gap}"
  rl_bs_synchronous_mode:="${rl_sync_enabled}"
  rl_bs_event_driven_one_shot:="${rl_bs_event_driven_one_shot}"
  rl_bs_action_path:="${rl_bs_action_path}"
  rl_bs_state_path:="${rl_bs_state_path}"
  rl_bs_communication_slot_ms:="${rl_bs_communication_slot_ms}"
  require_ground_truth_map:="${require_ground_truth_map}"
  task_metric_observer_mode:="${task_metric_observer_mode}"
  rl_bs_decision_period_ms:="${rl_bs_decision_period_ms}"
  rl_bs_repeat_action_each_slot:="${rl_bs_repeat_action_each_slot}"
  rl_llm_state_period_ms:="${rl_llm_state_period_ms}"
  random_seed:="${random_seed}"
)
if [[ -n "${shared_memory_root}" ]]; then
  launch_communication_args+=(
    rl_shared_memory_root:="${shared_memory_root}"
  )
fi
if [[ -n "${ground_truth_occupied_voxels_path}" ]]; then
  launch_communication_args+=(
    ground_truth_occupied_voxels_path:="${ground_truth_occupied_voxels_path}"
  )
fi
if [[ -n "${observed_occupied_voxels_path}" ]]; then
  launch_communication_args+=(
    observed_occupied_voxels_path:="${observed_occupied_voxels_path}"
  )
fi
if [[ -n "${ap_position_x}" ]]; then
  launch_communication_args+=(
    ap_position_x:="${ap_position_x}"
    ap_position_y:="${ap_position_y}"
    ap_position_z:="${ap_position_z}"
  )
fi
if [[ -f "${radio_map_cache}" ]]; then
  launch_communication_args+=(radio_map_cache:="${radio_map_cache}")
fi
launch_debug_args=()
if [[ -n "${RACER_DEBUG_LAUNCH_PREFIX:-}" ]]; then
  # Diagnostic-only wrapper used to obtain native backtraces without changing
  # any planner callback, parameter, or normal launch behavior.
  launch_debug_args+=(--launch-prefix "${RACER_DEBUG_LAUNCH_PREFIX}")
  if [[ -n "${RACER_DEBUG_LAUNCH_PREFIX_FILTER:-}" ]]; then
    launch_debug_args+=(--launch-prefix-filter "${RACER_DEBUG_LAUNCH_PREFIX_FILTER}")
  fi
fi
launch_command=(
  ros2 launch "${launch_debug_args[@]}"
  "${launch_package}" "${launch_file}"
  drone_count:="${drone_count}"
  trigger_minimum_cloud_frames:="${trigger_minimum_cloud_frames}"
  trigger_delay_s:="${trigger_delay_s}"
  scenario:="${scenario}"
  algorithm_variant:="${algorithm_variant}"
  exploration_assignment_mode:="${exploration_assignment_mode}"
  lkh_dir:="${lkh_dir}"
  "${launch_communication_args[@]}"
)
if [[ "${process_role}" == "racer" ]]; then
  exec "${launch_command[@]}"
fi

launch_pid=""
completion_monitor_pid=""
if [[ "${process_role}" == "combined" ]]; then
  setsid "${launch_command[@]}" >"${launch_log}" 2>&1 &
  launch_pid=$!
fi

cleanup() {
  trap - EXIT INT TERM HUP
  if [[ -n "${completion_monitor_pid}" ]]; then
    kill "${completion_monitor_pid}" 2>/dev/null || true
    wait "${completion_monitor_pid}" 2>/dev/null || true
  fi
  # The ros2 launch group leader may exit before every child (notably a
  # trajectory server handling an interrupted callback).  Address the already
  # resolved process group unconditionally so an interrupted diagnostic cannot
  # leak DDS participants into the next domain/run.
  if [[ -n "${launch_pid}" ]]; then
    kill -KILL -- "-${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

if [[ "${process_role}" == "combined" && "${stop_on_completion}" == "1" ]]; then
  # Incrementally follow the launch log.  Long source-faithful runs generate
  # hundreds of MB of upstream diagnostics, so re-reading the whole file once
  # per second would consume substantial I/O without changing the criterion.
  python3 "${adapter_share}/scripts/monitor_completion.py" \
    "${launch_log}" "${drone_count}" "${launch_pid}" &
  completion_monitor_pid=$!
fi

if [[ "${process_role}" == "combined" ]]; then sleep 3; fi
isaac_args=(
  --scenario "${scenario}"
  --scene-usd "${scene_usd}"
  --vehicle-model racer_so3
  --vehicle-usd "${vehicle_usd}"
  --duration "${duration}"
  --drone-count "${drone_count}"
  --camera-ray-budget "${ray_budget}"
  --depth-sensor-backend "${depth_sensor_backend}"
  --physics-rate-hz "${physics_hz}"
  --sensor-rate-hz "${sensor_hz}"
  --sensor-worker-count "${sensor_worker_count}"
  --scene-query-rate-hz "${scene_query_rate_hz}"
  --depth-width "${depth_width}"
  --depth-height "${depth_height}"
  --diagnostics
  --mapping-coverage-target "${coverage_target}"
)
if [[ "${stop_on_completion}" == "0" ]]; then
  isaac_args+=(--fixed-exploration-horizon)
fi
if [[ "${sensor_profiling}" == "0" ]]; then
  isaac_args+=(--no-sensor-profiling)
else
  isaac_args+=(--sensor-profiling)
fi
if [[ "${startup_free_space_yaw}" == "1" ]]; then
  isaac_args+=(
    --startup-free-space-yaw
    --startup-scan-duration "${startup_scan_duration}"
    --startup-unknown-corridor-distance "${startup_unknown_corridor_distance}"
    --startup-corridor-speed "${startup_corridor_speed}"
    --startup-settle-duration "${startup_settle_duration}"
  )
fi
if [[ -n "${start_positions}" ]]; then
  read -r -a start_values <<< "${start_positions}"
  if ! python3 - "${drone_count}" "${start_values[@]}" <<'PY'
import math
import sys

drone_count = int(sys.argv[1])
values = [float(value) for value in sys.argv[2:]]
if len(values) != 3 * drone_count or not all(math.isfinite(value) for value in values):
    raise SystemExit(1)
PY
  then
    printf 'RACER_START_POSITIONS must contain exactly three finite values per UAV.\n' >&2
    exit 2
  fi
  isaac_args+=(--starts "${start_values[@]}")
fi
if [[ "${headless}" == "1" && "${visualize}" != "1" ]]; then
  isaac_args+=(--headless --no-animate-propellers)
else
  isaac_args+=(
    --visualize-exploration
    --visualization-max-map-points "${map_points}"
    --interactive-render-hz "${interactive_hz}"
    --propeller-visual-hz 20
    --animate-propellers
  )
fi
if [[ "${record_trajectory_history}" == "1" ]]; then
  isaac_args+=(--record-trajectory-history)
fi
if [[ "${rl_bs_scheduler_enabled}" == "true" ]]; then
  isaac_args+=(--agentic-crpo-state-file "${rl_bs_mission_state_path}")
fi
if [[ -n "${shared_memory_root}" ]]; then
  isaac_args+=(
    --shared-memory-root "${shared_memory_root}"
    --shared-memory-physical-rate-hz 50
  )
fi
if [[ -n "${single_gpu_pause_request_path}" ]]; then
  isaac_args+=(
    --single-gpu-pause-request-file "${single_gpu_pause_request_path}"
    --single-gpu-pause-ack-file "${single_gpu_pause_ack_path}"
    --single-gpu-maximum-pause-s "${single_gpu_maximum_pause_s}"
    --single-gpu-pause-poll-s "${single_gpu_pause_poll_s}"
  )
fi
if [[ "${rl_sync_enabled}" == "true" ]]; then
  isaac_args+=(
    --rl-sync-state-file "${rl_bs_state_path}"
    --rl-sync-action-file "${rl_bs_action_path}"
    --rl-sync-release-file "${rl_sync_release_path}"
    --rl-sync-ack-file "${rl_sync_ack_path}"
    --rl-sync-communication-slot-s "$(python3 -c 'import sys; print(float(sys.argv[1])/1000.0)' "${rl_bs_communication_slot_ms}")"
    --rl-sync-decision-interval-s "$(python3 -c 'import sys; print(float(sys.argv[1])/1000.0)' "${rl_bs_decision_period_ms}")"
    --rl-sync-slots-per-decision 5
    --rl-sync-maximum-wait-s "${rl_sync_maximum_wait_s}"
    --rl-sync-poll-s "${rl_sync_poll_s}"
  )
fi

duration_ceiling="$(python3 -c 'import math,sys; print(math.ceil(float(sys.argv[1])))' "${duration}")"
if [[ "${process_role}" == "isaac" ]]; then
  exec env -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
    ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    PYTHONPATH="${training_root}/agentic_crpo" \
    LD_LIBRARY_PATH="${isaac_library_path}" \
    "${isaac_root}/python.sh" \
    "${adapter_share}/isaac_sim/original_racer_isaac.py" \
    "${isaac_args[@]}"
fi
set +e
env -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH -u PYTHONPATH \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" ROS_DISTRO=humble \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  LD_LIBRARY_PATH="${isaac_library_path}" \
  timeout "$((duration_ceiling * wall_time_multiplier + wall_time_grace))" \
  "${isaac_root}/python.sh" "${adapter_share}/isaac_sim/original_racer_isaac.py" \
  "${isaac_args[@]}" 2>&1 | tee "${isaac_log}"
isaac_status=${PIPESTATUS[0]}
set -e

if [[ -n "${completion_monitor_pid}" ]]; then
  kill "${completion_monitor_pid}" 2>/dev/null || true
  wait "${completion_monitor_pid}" 2>/dev/null || true
  completion_monitor_pid=""
fi

sleep 2
kill -KILL -- "-${launch_pid}" 2>/dev/null || true
wait "${launch_pid}" 2>/dev/null || true

python3 - "${isaac_log}" "${launch_log}" "${result_file}" \
  "${drone_count}" "${require_completion}" "${isaac_status}" \
  "${communication_mode}" "${scenario}" "${network_topology}" \
  "${lossless_nearest_neighbor_count}" \
  "${scene_usd}" "${sionna_scene_xml}" "${algorithm_label}" \
  "${random_seed}" "${communication_range_m}" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

isaac_log, launch_log, result_file = map(Path, sys.argv[1:4])
drone_count = int(sys.argv[4])
require_completion = bool(int(sys.argv[5]))
isaac_status = int(sys.argv[6])
communication_mode = sys.argv[7]
scenario = sys.argv[8]
network_topology = sys.argv[9]
lossless_nearest_neighbor_count = int(sys.argv[10])
scene_usd = sys.argv[11]
sionna_scene_xml = sys.argv[12]
algorithm_label = sys.argv[13]
random_seed = int(sys.argv[14])
communication_range_m = float(sys.argv[15])
lines = isaac_log.read_text(errors="replace").splitlines()
prefix = "RACER_3D_ISAAC_RESULT "
matches = [line[len(prefix):] for line in lines if line.startswith(prefix)]
if not matches:
    raise SystemExit("Isaac result line is missing")
metrics = json.loads(matches[-1])
launch_text = launch_log.read_text(errors="replace")
comm_matches = re.findall(r"RACER_SIONNA_STATS (\{[^\n]+\})", launch_text)
communication_statistics = json.loads(comm_matches[-1]) if comm_matches else {}
channel_profile_matches = re.findall(
    r"RACER_SIONNA_CHANNEL_PROFILE (\{[^\n]+\})", launch_text
)
channel_profile = (
    json.loads(channel_profile_matches[-1])
    if channel_profile_matches
    else {}
)
esdf_pattern = re.compile(
    r"\[racer_original_exploration_(\d+)\]: RACER_ESDF_PROFILE "
    r"updates=(\d+) timer_updates=(\d+) planner_updates=(\d+) "
    r"current_ms=([0-9.eE+-]+) mean_ms=([0-9.eE+-]+) "
    r"max_ms=([0-9.eE+-]+) dirty_age_ms=([0-9.eE+-]+) "
    r"reason=(\w+) max_rate_hz=([0-9.eE+-]+)"
)
latest_esdf_by_agent = {}
for match in esdf_pattern.finditer(launch_text):
    latest_esdf_by_agent[int(match.group(1))] = {
        "updates": int(match.group(2)),
        "timer_updates": int(match.group(3)),
        "planner_updates": int(match.group(4)),
        "current_ms": float(match.group(5)),
        "mean_ms": float(match.group(6)),
        "max_ms": float(match.group(7)),
        "dirty_age_ms": float(match.group(8)),
        "last_reason": match.group(9),
        "max_rate_hz": float(match.group(10)),
    }
if latest_esdf_by_agent:
    total_esdf_updates = sum(
        row["updates"] for row in latest_esdf_by_agent.values()
    )
    metrics["esdf_profile"] = {
        "agents_reported": len(latest_esdf_by_agent),
        "updates": total_esdf_updates,
        "timer_updates": sum(
            row["timer_updates"] for row in latest_esdf_by_agent.values()
        ),
        "planner_forced_updates": sum(
            row["planner_updates"] for row in latest_esdf_by_agent.values()
        ),
        "mean_update_ms": (
            sum(
                row["mean_ms"] * row["updates"]
                for row in latest_esdf_by_agent.values()
            )
            / total_esdf_updates
            if total_esdf_updates
            else 0.0
        ),
        "max_update_ms": max(
            row["max_ms"] for row in latest_esdf_by_agent.values()
        ),
        "configured_max_rate_hz": max(
            row["max_rate_hz"] for row in latest_esdf_by_agent.values()
        ),
        "per_agent_latest": latest_esdf_by_agent,
    }
sionna_ready = (
    communication_mode == "ideal"
    or (
        "loaded Sionna RT scene" in launch_text
        and "Sionna RT initialization failed" not in launch_text
    )
)
exact_link_samples = (
    communication_statistics.get("sionna_exact_samples", 0)
    + communication_statistics.get("sionna_cache_corrected_samples", 0)
)
communication_active = (
    communication_statistics.get("delivered_packets", 0) > 0
    and (communication_mode == "ideal" or (sionna_ready and exact_link_samples > 0))
)
topology_active = (
    communication_statistics.get("network_topology") == network_topology
    and (
        network_topology == "distributed"
        or (
            network_topology == "nearest_neighbors"
            and communication_statistics.get("nearest_neighbor_count", 0) > 0
            and communication_statistics.get("nearest_filtered_receivers", 0) > 0
        )
        or (
            network_topology == "distance_radius"
            and math.isclose(
                communication_statistics.get("communication_range_m", 0.0),
                communication_range_m,
            )
            and communication_statistics.get("range_filtered_receivers", 0) > 0
        )
        or (
            network_topology == "ap_assisted"
            and communication_statistics.get("ap_global_updates_received", 0) > 0
        )
        or (
            network_topology == "bs_round_robin"
            and communication_statistics.get("bs_round_robin_enabled") is True
            and communication_statistics.get("bs_round_robin_turns", 0) >= drone_count
            and communication_statistics.get("bs_upload_grants_delivered", 0) > 0
        )
    )
)
lossless_nearest_override_active = (
    lossless_nearest_neighbor_count == 0
    or (
        communication_mode != "ideal"
        and network_topology == "distributed"
        and communication_statistics.get(
            "lossless_nearest_neighbor_count", 0
        ) == lossless_nearest_neighbor_count
        and communication_statistics.get(
            "lossless_nearest_forwarded_packets", 0
        ) > 0
        and communication_statistics.get(
            "sionna_direct_attempted_packets", 0
        ) > 0
    )
)
finished = sorted({
    int(value)
    for value in re.findall(
        r"racer_(?:original|recovery)_exploration_(\d+).*(?:finish exploration|state: FINISH)",
        launch_text,
    )
})
returned = sorted({
    int(value)
    for value in re.findall(
        r"racer_(?:original|recovery)_exploration_(\d+).*Go back to", launch_text
    )
})
executed = sorted({
    int(value)
    for value in re.findall(
        r"racer_(?:original|recovery)_exploration_(\d+).*from PUB_TRAJ to EXEC_TRAJ",
        launch_text,
    )
})
def count(pattern):
    return len(re.findall(pattern, launch_text))

evidence = {
    "hgrid_tours": count(r"Grid tour:"),
    "frontier_updates": count(r"Frontier num:"),
    "lkh_atsp_solutions": count(r"Best ATSP solution:"),
    "lkh_acvrp_solutions": count(r"Best ACVRP solution:"),
    "pair_requests": count(r"send opt request"),
    "pair_responses": count(r"get response"),
    "kinodynamic_mid_goals": count(r"Mid goal"),
    "nlopt_trajectory_runs": count(r"Traj opt iter num:"),
    "yaw_plans": count(r"Traj: .*yaw:"),
    "lkh_call_failures": count(r"Fail to solve (?:ATSP|ACVRP)"),
    "process_crashes": count(r"process has died|exit code -11|Segmentation"),
    "tracking_recoveries": count(r"\[trackingLostCallback\]"),
    "bounded_recovery_fail_windows": count(r"RACER_RECOVERY plan_fail"),
    "local_reselections": count(r"RACER_RECOVERY local_(?:viewpoint_)?reselect"),
    "local_escapes": count(r"RACER_RECOVERY local_escape"),
    "ap_repartition_requests": count(r"RACER_RECOVERY request_ap_repartition"),
    "ap_repartition_commands": count(r"RACER_RECOVERY command episode="),
    "completed_recoveries": count(r"RACER_RECOVERY complete episode="),
}
normal_completion = len(finished) == drone_count and len(returned) == drone_count
algorithm_pipeline_ok = (
    len(executed) == drone_count
    and evidence["hgrid_tours"] > 0
    and evidence["frontier_updates"] > 0
    and evidence["lkh_atsp_solutions"] > 0
    and (drone_count < 2 or evidence["lkh_acvrp_solutions"] > 0)
    and (drone_count < 2 or evidence["pair_responses"] > 0)
    and evidence["kinodynamic_mid_goals"] > 0
    and evidence["nlopt_trajectory_runs"] > 0
    and evidence["yaw_plans"] > 0
)
min_distance = metrics.get("min_inter_drone")
min_clearance = metrics.get("min_obstacle_clearance")
starts = metrics.get("start_positions", [])
positions = metrics.get("positions", [])
return_errors = [
    math.dist(start, position)
    for start, position in zip(starts, positions)
]
physical_return_ok = (
    len(return_errors) == drone_count
    and all(error < 1.0 for error in return_errors)
)
acceptance = {
    "isaac_exit_ok": isaac_status == 0,
    "zero_collisions": metrics.get("collision_events") == 0,
    "minimum_inter_drone_distance_ok": (
        drone_count < 2 or (min_distance is not None and min_distance >= 1.0)
    ),
    "positive_obstacle_clearance": (
        min_clearance is not None and min_clearance > 0.0
    ),
    "all_agents_executed_original_fsm": len(executed) == drone_count,
    "original_algorithm_pipeline_observed": algorithm_pipeline_ok,
    "lkh_services_healthy": (
        evidence["lkh_call_failures"] == 0
        and evidence["process_crashes"] == 0
    ),
    "normal_completion": normal_completion,
    "physical_return_within_original_1m_threshold": physical_return_ok,
    "communication_proxy_active": communication_active,
    "requested_network_topology_active": topology_active,
    "lossless_nearest_override_active": lossless_nearest_override_active,
    "sionna_rt_active": communication_mode == "ideal" or (
        sionna_ready and exact_link_samples > 0
    ),
}
passed = all(
    value for name, value in acceptance.items()
    if require_completion or name not in (
        "normal_completion",
        "physical_return_within_original_1m_threshold",
    )
)
result = {
    "algorithm": algorithm_label,
    "random_seed": random_seed,
    "scene": scene_usd,
    "sionna_scene": sionna_scene_xml,
    "vehicle": "crazyflie_with_racer_dynamics.usd",
    "drone_count": drone_count,
    "require_completion": require_completion,
    "finished_drone_ids": finished,
    "returned_drone_ids": returned,
    "executed_drone_ids": executed,
    "algorithm_evidence": evidence,
    "return_position_errors_m": return_errors,
    "acceptance": acceptance,
    "communication": {
        "mode": communication_mode,
        "network_topology": network_topology,
        "lossless_nearest_neighbor_count": lossless_nearest_neighbor_count,
        "communication_range_m": communication_range_m,
        "sionna_ready": sionna_ready,
        "exact_link_samples": exact_link_samples,
        "statistics": communication_statistics,
        "channel_profile": channel_profile,
    },
    "passed": passed,
    "metrics": metrics,
}
result_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY
