#!/usr/bin/env python3
"""Isaac Sim boundary for the source-faithful ROS2/C++ RACER port.

This is the self-contained simulator bridge for the C++ RACER nodes.  Isaac
Sim exposes its application API in Python; exploration, mapping, allocation,
planning remain in the original RACER C++ sources.  The default
plant is the generated 0.98 kg RACER SO3 plus-quadrotor; the legacy Crazyflie
profile remains available for comparison.  The RACER profile follows the
upstream ROS 1 simulator rather than the paper's real vehicle: a 1 kHz plant,
200 Hz odometry/IMU, and a 640x480 30 Hz ideal pinhole ray camera. The default
camera backend is a GPU-batched NVIDIA Warp static-mesh ray caster; RTX depth
remains available only as a comparison/fallback backend.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Sequence, Tuple


# Isaac Kit cannot be attached to by ptrace on all target machines.  Keep an
# in-process, read-only stack probe available for diagnosing a stalled bridge.
# SIGUSR1 does not alter simulation or RACER state; it only writes Python
# thread stacks to the existing Isaac log.
faulthandler.register(signal.SIGUSR1, all_threads=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--fixed-exploration-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "continue simulation until --duration even if RACER reports "
            "mission completion or the optional coverage target is reached"
        ),
    )
    parser.add_argument(
        "--mapping-coverage-target",
        type=float,
        default=0.0,
        help=(
            "stop when the union of all UAV RACER SDF maps reaches this "
            "known-voxel fraction; zero disables coverage stopping"
        ),
    )
    parser.add_argument("--drone-count", type=int, default=3)
    parser.add_argument("--physics-rate-hz", type=float, default=1000.0)
    parser.add_argument("--sensor-rate-hz", type=float, default=30.0)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument(
        "--depth-sensor-backend",
        choices=("warp", "rtx"),
        default="warp",
        help=(
            "forward exploration sensor backend; warp uses one GPU-batched "
            "static-mesh raycast and rtx retains the former rendered depth "
            "camera as an explicit comparison/fallback"
        ),
    )
    parser.add_argument(
        "--sensor-profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "emit one RACER_3D_SENSOR_PROFILE JSON record per point-cloud "
            "frame with raycast/acquisition, PointCloud2 construction and "
            "ROS publication wall time"
        ),
    )
    parser.add_argument(
        "--contact-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "poll the former full ContactSensor frames alongside the "
            "event-driven sparse path; intended only for same-trajectory "
            "correctness/profiling tests because it restores the old raw "
            "buffer and polling overhead"
        ),
    )
    parser.add_argument(
        "--scenario", default="acceptance_15x9x2"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--animate-propellers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "drive render-only propeller Xforms from RACER motor RPM; defaults "
            "to enabled in the interactive viewport and disabled headlessly"
        ),
    )
    parser.add_argument(
        "--propeller-visual-hz",
        type=float,
        default=60.0,
        help="maximum USD propeller-transform update rate",
    )
    parser.add_argument(
        "--visualize-exploration",
        action="store_true",
        help=(
            "draw the shared occupied map, planned paths, and flown trails "
            "in the interactive Isaac Sim viewport"
        ),
    )
    parser.add_argument(
        "--record-trajectory-history",
        action="store_true",
        help=(
            "store sampled vehicle poses in the final result for later "
            "visual replay; disabled by default for metric-only headless runs"
        ),
    )
    parser.add_argument(
        "--visualization-max-map-points",
        type=int,
        default=40000,
        help="display-only occupied-voxel cap; does not change the RACER map",
    )
    parser.add_argument(
        "--interactive-render-hz",
        type=float,
        default=30.0,
        help=(
            "minimum wall-clock viewport refresh rate while interactive; "
            "keeps mouse camera input responsive when simulation is slower "
            "than real time"
        ),
    )
    parser.add_argument(
        "--camera-ray-budget",
        type=int,
        default=76800,
        help=(
            "maximum depth samples forwarded per frame; 76800 preserves every "
            "sample from the original 640x480 skip-pixel=2 grid"
        ),
    )
    parser.add_argument(
        "--sensor-worker-count",
        type=int,
        default=1,
        help=(
            "CPU workers used for per-UAV depth/lidar coordinate transforms; "
            "Isaac sensor access and ROS publication remain on the main thread"
        ),
    )
    parser.add_argument(
        "--scene-query-rate-hz",
        type=float,
        default=50.0,
        help="per-UAV PhysX execution-safety sweep refresh rate",
    )
    parser.add_argument(
        "--startup-free-space-yaw",
        action="store_true",
        help=(
            "select each initial yaw using collision-scene sphere sweeps, "
            "then run the configured pre-trigger scan/corridor sequence"
        ),
    )
    parser.add_argument(
        "--startup-scan-duration",
        type=float,
        default=0.0,
        help="seconds used for one stationary 360-degree pre-trigger scan",
    )
    parser.add_argument(
        "--startup-unknown-corridor-distance",
        type=float,
        default=0.0,
        help=(
            "pre-trigger distance allowed through map-UNKNOWN space only "
            "after a collision-scene sphere sweep verifies the corridor"
        ),
    )
    parser.add_argument(
        "--startup-corridor-speed",
        type=float,
        default=0.25,
        help="commanded speed for the verified pre-trigger corridor",
    )
    parser.add_argument(
        "--startup-settle-duration",
        type=float,
        default=1.0,
        help="stationary settling time after the pre-trigger corridor",
    )
    parser.add_argument(
        "--vehicle-model",
        choices=("racer_so3", "crazyflie"),
        default="racer_so3",
        help="PhysX vehicle plant used below the C++ RACER velocity command",
    )
    parser.add_argument(
        "--vehicle-usd",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "isaac_assets"
            / "racer_so3_quadrotor"
            / "usd"
            / "racer_so3_quadrotor_flattened.usd"
        ),
        help="generated RACER SO3 USD asset",
    )
    parser.add_argument(
        "--scene-usd",
        type=Path,
        help=(
            "reference an external collision-enabled USD instead of building "
            "the deterministic acceptance obstacles"
        ),
    )
    parser.add_argument(
        "--starts",
        type=float,
        nargs="+",
        help="flat x y z launch positions; provide three values per vehicle",
    )
    parser.add_argument(
        "--control-probe",
        action="store_true",
        help="command one vehicle at 0.35 m/s for isolated 6-DOF testing",
    )
    parser.add_argument(
        "--control-probe-command",
        type=float,
        nargs=3,
        default=(0.30, 0.20, 0.10),
        metavar=("VX", "VY", "VZ"),
        help="world-frame velocity used with --control-probe",
    )
    parser.add_argument(
        "--agentic-crpo-state-file",
        type=Path,
        help=(
            "optional atomic mission-state file consumed by the external "
            "Qwen+CRPO trainer; disabled by default"
        ),
    )
    parser.add_argument(
        "--shared-memory-root",
        help=(
            "optional /dev/shm child created by the four-process supervisor; "
            "Isaac writes only physical.shm and status_isaac.shm"
        ),
    )
    parser.add_argument(
        "--shared-memory-physical-rate-hz",
        type=float,
        default=50.0,
        help="maximum physical-state publication rate (simulation time)",
    )
    parser.add_argument(
        "--single-gpu-pause-request-file",
        type=Path,
        help="file request used to freeze Isaac while Qwen owns the GPU",
    )
    parser.add_argument(
        "--single-gpu-pause-ack-file",
        type=Path,
        help="atomic Isaac acknowledgement for the single-GPU pause barrier",
    )
    parser.add_argument(
        "--single-gpu-maximum-pause-s",
        type=float,
        default=600.0,
        help="fail-safe wall-time lease for one Qwen pause",
    )
    parser.add_argument(
        "--single-gpu-pause-poll-s",
        type=float,
        default=0.01,
        help="wall-time polling interval while Isaac is frozen",
    )
    parser.add_argument(
        "--rl-sync-state-file",
        type=Path,
        help="100 ms boundary state written by the communication proxy",
    )
    parser.add_argument(
        "--rl-sync-action-file",
        type=Path,
        help="decision-indexed CRPO action file checked before resume",
    )
    parser.add_argument(
        "--rl-sync-release-file",
        type=Path,
        help="trainer release record for a frozen 100 ms RL boundary",
    )
    parser.add_argument(
        "--rl-sync-ack-file",
        type=Path,
        help="Isaac acknowledgement that simulation time is frozen",
    )
    parser.add_argument(
        "--rl-sync-communication-slot-s",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--rl-sync-decision-interval-s",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rl-sync-slots-per-decision",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rl-sync-maximum-wait-s",
        type=float,
        default=1200.0,
        help="wall-time fail-safe only; it never defines an RL timestep",
    )
    parser.add_argument(
        "--rl-sync-poll-s",
        type=float,
        default=0.002,
    )
    return parser.parse_args()


ARGS = parse_arguments()
# With colcon --symlink-install, ``__file__`` resolves to the source tree while
# the compiled pybind11 module remains beside the invoked install-tree script.
# Preserve both locations across SimulationApp's Python-path initialization.
_INVOKED_ISAAC_DIR = Path(sys.argv[0]).absolute().parent
_SOURCE_ISAAC_DIR = Path(__file__).resolve().parent
for _module_dir in (_INVOKED_ISAAC_DIR, _SOURCE_ISAAC_DIR):
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))
if ARGS.scene_usd is not None:
    ARGS.scene_usd = ARGS.scene_usd.expanduser().resolve()
    if not ARGS.scene_usd.is_file():
        raise SystemExit(f"scene USD does not exist: {ARGS.scene_usd}")
if ARGS.vehicle_model == "racer_so3":
    ARGS.vehicle_usd = ARGS.vehicle_usd.expanduser().resolve()
    if not ARGS.vehicle_usd.is_file():
        raise SystemExit(f"vehicle USD does not exist: {ARGS.vehicle_usd}")
if ARGS.camera_ray_budget <= 0:
    raise SystemExit("--camera-ray-budget must be positive")
if ARGS.sensor_worker_count <= 0:
    raise SystemExit("--sensor-worker-count must be positive")
if ARGS.scene_query_rate_hz <= 0.0:
    raise SystemExit("--scene-query-rate-hz must be positive")
if ARGS.startup_scan_duration < 0.0:
    raise SystemExit("--startup-scan-duration must be non-negative")
if ARGS.startup_unknown_corridor_distance < 0.0:
    raise SystemExit("--startup-unknown-corridor-distance must be non-negative")
if ARGS.startup_corridor_speed <= 0.0:
    raise SystemExit("--startup-corridor-speed must be positive")
if ARGS.startup_settle_duration < 0.0:
    raise SystemExit("--startup-settle-duration must be non-negative")
if ARGS.visualization_max_map_points <= 0:
    raise SystemExit("--visualization-max-map-points must be positive")
if ARGS.interactive_render_hz <= 0.0:
    raise SystemExit("--interactive-render-hz must be positive")
if ARGS.propeller_visual_hz <= 0.0:
    raise SystemExit("--propeller-visual-hz must be positive")
if ARGS.shared_memory_physical_rate_hz <= 0.0:
    raise SystemExit("--shared-memory-physical-rate-hz must be positive")
if (ARGS.single_gpu_pause_request_file is None) != (
    ARGS.single_gpu_pause_ack_file is None
):
    raise SystemExit("both single-GPU pause files must be provided")
if ARGS.single_gpu_maximum_pause_s <= 0.0:
    raise SystemExit("--single-gpu-maximum-pause-s must be positive")
if ARGS.single_gpu_pause_poll_s <= 0.0:
    raise SystemExit("--single-gpu-pause-poll-s must be positive")
_RL_SYNC_PATHS = (
    ARGS.rl_sync_state_file,
    ARGS.rl_sync_action_file,
    ARGS.rl_sync_release_file,
    ARGS.rl_sync_ack_file,
)
if any(path is None for path in _RL_SYNC_PATHS) and any(
    path is not None for path in _RL_SYNC_PATHS
):
    raise SystemExit("all four synchronous-RL files must be provided")
if ARGS.shared_memory_root and (
    ARGS.single_gpu_pause_request_file is not None
    or any(path is not None for path in _RL_SYNC_PATHS)
):
    raise SystemExit(
        "four-process shared-memory mode cannot enable either Isaac freeze gate"
    )
if (
    ARGS.rl_sync_communication_slot_s <= 0.0
    or ARGS.rl_sync_decision_interval_s <= 0.0
    or ARGS.rl_sync_slots_per_decision <= 0
    or ARGS.rl_sync_maximum_wait_s <= 0.0
    or ARGS.rl_sync_poll_s <= 0.0
):
    raise SystemExit("synchronous-RL intervals must be positive")
if not math.isclose(
    ARGS.rl_sync_decision_interval_s,
    ARGS.rl_sync_communication_slot_s * ARGS.rl_sync_slots_per_decision,
    rel_tol=0.0,
    abs_tol=1.0e-12,
):
    raise SystemExit(
        "RL decision interval must equal communication slot times slot count"
    )
if ARGS.rl_sync_state_file is not None:
    episode_intervals = ARGS.duration / ARGS.rl_sync_decision_interval_s
    if not math.isclose(
        episode_intervals, round(episode_intervals), rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise SystemExit(
            "synchronous-RL duration must contain an integer number of decisions"
        )
if ARGS.headless and ARGS.visualize_exploration:
    raise SystemExit("--visualize-exploration requires an interactive window")
ANIMATE_PROPELLERS = (
    not ARGS.headless
    if ARGS.animate_propellers is None
    else bool(ARGS.animate_propellers)
)

from isaacsim import SimulationApp  # noqa: E402


simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402


enable_extension("isaacsim.ros2.bridge")
if ARGS.vehicle_model == "racer_so3":
    # Isaac Lab's RayCasterCamera is built on NVIDIA Warp. Isaac Sim ships the
    # same GPU mesh-query runtime even when the separate Isaac Lab package is
    # not installed on the target machine.
    enable_extension("omni.warp.core")
if ARGS.visualize_exploration:
    enable_extension("isaacsim.util.debug_draw")
simulation_app.update()

import numpy as np  # noqa: E402
import omni.syntheticdata as syntheticdata  # noqa: E402
import omni.usd  # noqa: E402
import rclpy  # noqa: E402
if ARGS.shared_memory_root:
    from agentic_crpo.fast_state import encode_physical_fast  # noqa: E402
    from agentic_crpo.shared_ipc import (  # noqa: E402
        SharedMemoryLayout,
        open_blocks,
        open_rings,
    )
from single_gpu_pause_gate import SingleGpuPauseGate  # noqa: E402
from synchronous_rl_gate import SynchronousRlBoundaryGate  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.sensors.physics import ContactSensor  # noqa: E402
from isaacsim.sensors.physx import RotatingLidarPhysX  # noqa: E402
from nav_msgs.msg import Odometry, Path as RosPath  # noqa: E402
from omni.physx import (  # noqa: E402
    get_physx_scene_query_interface,
    get_physx_simulation_interface,
)
from omni.physx.bindings._physx import ContactEventType  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
)
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Imu, PointCloud2  # noqa: E402
from std_msgs.msg import String, UInt8MultiArray  # noqa: E402


_BYTE_POPCOUNT = np.asarray(
    [value.bit_count() for value in range(256)], dtype=np.uint8
)
from visualization_msgs.msg import Marker  # noqa: E402

from crazyflie_cpp_bridge import (  # noqa: E402
    MASS as CRAZYFLIE_MASS,
    quaternion_matrix,
    velocity_wrench,
)
from pointcloud_cpp_bridge import (  # noqa: E402
    create_xyzi_cloud,
    read_xyzi_cloud,
)
if ARGS.vehicle_model == "racer_so3":
    from warp_raycast_camera import WarpRayCasterCameraBatch  # noqa: E402
    from racer_control_batch_cpp import solve_control_batch  # noqa: E402
else:
    WarpRayCasterCameraBatch = None
    solve_control_batch = None
from racer_so3_cpp_bridge import (  # noqa: E402
    MASS as RACER_SO3_MASS,
    hover_rpm as racer_hover_rpm,
    velocity_motor_wrench,
)
from scenario_cpp_bridge import (  # noqa: E402
    DRONE_RADIUS,
    get_scenario,
    obstacle_clearance,
    pairwise_distances,
    point_box_signed_clearance,
)
from safety_cpp_bridge import (  # noqa: E402
    aabb_obstacle_filter,
    cbf_swarm_filter,
    flight_volume_filter,
    pointcloud_obstacle_constraints,
    pointcloud_obstacle_filter,
    project_velocity_constraints,
    sweep_obstacle_filter,
)

if ARGS.visualize_exploration:
    from isaacsim.util.debug_draw import _debug_draw  # noqa: E402


SCENARIO = get_scenario(ARGS.scenario)
if ARGS.starts is None:
    if len(SCENARIO.starts) < ARGS.drone_count:
        raise SystemExit(
            f"scenario {SCENARIO.name!r} provides {len(SCENARIO.starts)} "
            f"starts, but {ARGS.drone_count} vehicles were requested"
        )
    STARTS = SCENARIO.starts[:ARGS.drone_count]
else:
    if len(ARGS.starts) != 3 * ARGS.drone_count:
        raise SystemExit(
            "--starts requires exactly three values per configured vehicle"
        )
    STARTS = tuple(
        tuple(float(value) for value in ARGS.starts[index:index + 3])
        for index in range(0, len(ARGS.starts), 3)
    )


if ARGS.physics_rate_hz <= 0.0 or ARGS.sensor_rate_hz <= 0.0:
    raise ValueError("physics and sensor rates must be positive")
if ARGS.depth_width < 64 or ARGS.depth_height < 48:
    raise ValueError("depth resolution is too small")
PHYSICS_DT = (
    1.0 / ARGS.physics_rate_hz
    if ARGS.vehicle_model == "racer_so3"
    else 0.02
)
ODOM_PERIOD = 1.0 / (200.0 if ARGS.vehicle_model == "racer_so3" else 50.0)
DEPTH_PERIOD = 1.0 / ARGS.sensor_rate_hz
BODY_SIZE = (0.16, 0.16, 0.06)
DEPTH_WIDTH = ARGS.depth_width
DEPTH_HEIGHT = ARGS.depth_height
DEPTH_SCALE_X = DEPTH_WIDTH / 640.0
DEPTH_SCALE_Y = DEPTH_HEIGHT / 480.0
DEPTH_FX = 387.229 * DEPTH_SCALE_X
DEPTH_FY = 387.229 * DEPTH_SCALE_Y
DEPTH_CX = 321.046 * DEPTH_SCALE_X
DEPTH_CY = 243.449 * DEPTH_SCALE_Y
MAPPING_FX = 385.69793701171875 * DEPTH_SCALE_X
MAPPING_FY = 385.69793701171875 * DEPTH_SCALE_Y
MAPPING_CX = 324.0879821777344 * DEPTH_SCALE_X
MAPPING_CY = 239.10362243652344 * DEPTH_SCALE_Y
DEPTH_RENDER_HORIZON = 5.0
DEPTH_MIN_RANGE = 0.2
DEPTH_MAP_RANGE = 4.6
MAPPING_MIN_RAY_LENGTH = 0.5
MAPPING_MAX_RAY_LENGTH = 4.5
DEPTH_FILTER_MARGIN = max(1, round(2 * min(DEPTH_SCALE_X, DEPTH_SCALE_Y)))
DEPTH_SKIP_PIXEL = max(1, round(2 * min(DEPTH_SCALE_X, DEPTH_SCALE_Y)))
# The pinhole sampling lattice and deterministic ray-budget selection do not
# change between frames.  Precomputing them avoids rebuilding and thinning the
# same 640x480 grid for every UAV at every sensor tick.
_depth_sample_v, _depth_sample_u = np.mgrid[
    DEPTH_FILTER_MARGIN:DEPTH_HEIGHT - DEPTH_FILTER_MARGIN:DEPTH_SKIP_PIXEL,
    DEPTH_FILTER_MARGIN:DEPTH_WIDTH - DEPTH_FILTER_MARGIN:DEPTH_SKIP_PIXEL,
]
DEPTH_SAMPLE_ROWS = _depth_sample_v.reshape(-1)
DEPTH_SAMPLE_COLS = _depth_sample_u.reshape(-1)
if len(DEPTH_SAMPLE_ROWS) > ARGS.camera_ray_budget:
    _depth_selected = np.linspace(
        0,
        len(DEPTH_SAMPLE_ROWS) - 1,
        ARGS.camera_ray_budget,
        dtype=np.int64,
    )
    DEPTH_SAMPLE_ROWS = DEPTH_SAMPLE_ROWS[_depth_selected]
    DEPTH_SAMPLE_COLS = DEPTH_SAMPLE_COLS[_depth_selected]
DEPTH_SAMPLE_U = DEPTH_SAMPLE_COLS.astype(np.float32)
DEPTH_SAMPLE_V = DEPTH_SAMPLE_ROWS.astype(np.float32)
CAMERA_TRANSLATION = np.zeros(3)
# The upstream ideal renderer has no carrier geometry.  The generated Isaac
# visual envelope reaches 0.26 + 0.062 m from the body origin, so returns
# inside that envelope are necessarily self returns and must not enter RACER.
SELF_FILTER_RADIUS = 0.322 + 0.01
LIDAR_TRANSLATION = np.asarray((0.0, 0.0, 0.075))
# Use the complete rendered rotor envelope for execution safety.  The seven
# PhysX collision primitives reach about 0.287 m, but a 0.284 m query sphere
# left no allowance for their corners or the visual propeller discs.  The
# planner still receives the unchanged camera point cloud and map inflation;
# this radius exists only below the planner at the Isaac actuator boundary.
VEHICLE_RADIUS = (
    SELF_FILTER_RADIUS if ARGS.vehicle_model == "racer_so3" else DRONE_RADIUS
)
# VEHICLE_RADIUS is the hard, conservative rotor envelope.  Speed-dependent
# stopping allowance is added by pointcloud_obstacle_filter at every control
# step; adding another fixed 0.236 m here double-counted that margin and made a
# physically traversable 0.67 m warehouse gap infeasible at zero speed.
OBSTACLE_CONTROL_CLEARANCE = max(0.30, VEHICLE_RADIUS)
SWARM_CONTROL_DISTANCE = max(0.55, 2.0 * VEHICLE_RADIUS + 0.632)
# The execution interface needs limited tracking authority above the unchanged
# original manager.max_vel=1.5. The planner feed-forward never exceeds that
# value; at most 0.5 m/s is available to the position feedback term to recover
# SO3 attitude/motor lag. Obstacle and peer filters below still reduce this
# actuator bound dynamically using measured velocity and braking distance.
SOURCE_MAX_SPEED = 2.0 if ARGS.vehicle_model == "racer_so3" else 0.35
SAFETY_RAY_MAX_RANGE = 2.4
SAFETY_POINT_VOXEL_SIZE = 0.08
SAFETY_LIDAR_POINT_LIMIT = 1200
SAFETY_LIDAR_HORIZONTAL_RESOLUTION_DEG = 2.5
SAFETY_LIDAR_VERTICAL_RESOLUTION_DEG = 5.0
# The lidar is deliberately coarse and can miss a thin shelf edge or ceiling
# lamp between beams.  A low-level rigid-body sweep closes that geometric gap
# without feeding privileged scene information into RACER's planning map.
SCENE_QUERY_PERIOD = 1.0 / ARGS.scene_query_rate_hz
SCENE_QUERY_RANGE = SAFETY_RAY_MAX_RANGE
# The sphere itself already encloses the full rotor/arm collision geometry.
# Retain an additional free-travel reserve for PhysX contact offset, attitude
# lag and the configured scene-query interval.  The speed-scaled term keeps a
# lower query rate conservative instead of silently increasing travel between
# cached sweeps. This is an actuator margin, not planner-map inflation.
SCENE_QUERY_CLEARANCE = 0.02 + SOURCE_MAX_SPEED * SCENE_QUERY_PERIOD
SOURCE_MAX_YAW_RATE = (
    math.radians(10.0) if ARGS.vehicle_model == "racer_so3" else 0.15
)
ISAAC_SIM_ROOT = Path(
    os.environ.get("ISAAC_SIM_ROOT", str(Path.home() / "isaacsim"))
)
CRAZYFLIE_ASSET = ISAAC_SIM_ROOT / (
    "extscache/"
    "omni.warp.core-1.8.2+lx64/warp/examples/assets/crazyflie.usd"
)
USE_REFERENCE_VISUAL = True
VISUAL_DRONE_COLORS = (
    (1.0, 0.16, 0.10, 1.0),
    (0.15, 1.0, 0.25, 1.0),
    (1.0, 0.82, 0.08, 1.0),
    (0.75, 0.25, 1.0, 1.0),
    (1.0, 0.48, 0.05, 1.0),
)
VISUAL_MAP_COLOR = (0.0, 0.72, 1.0, 0.62)


def _low_level_safety_description() -> str:
    if ARGS.scene_usd is None:
        return "AABB stopping-distance velocity barrier"
    if SCENARIO.safety_min is not None and SCENARIO.safety_max is not None:
        return (
            "low-density 360-degree Warp safety rays, PhysX rigid-body sweep, "
            "and flight-volume stopping-distance barriers"
        )
    return (
        "low-density 360-degree Warp safety rays plus PhysX rigid-body sweep "
        "stopping-distance "
        "velocity barrier"
    )


def _external_obstacle_filter(
    preferred: Sequence[float],
    position: Sequence[float],
    points_world: Sequence[Sequence[float]],
    current_velocity: Sequence[float],
    sweep_constraints=(),
    point_constraints=None,
) -> np.ndarray:
    if point_constraints is None:
        result = pointcloud_obstacle_filter(
            preferred,
            position,
            points_world,
            clearance=OBSTACLE_CONTROL_CLEARANCE,
            speed_limit=SOURCE_MAX_SPEED,
            current_velocity=current_velocity,
        )
    else:
        result = project_velocity_constraints(
            preferred,
            point_constraints,
            SOURCE_MAX_SPEED,
        )
    if SCENARIO.safety_min is not None and SCENARIO.safety_max is not None:
        result = flight_volume_filter(
            result,
            position,
            SCENARIO.safety_min,
            SCENARIO.safety_max,
            clearance=OBSTACLE_CONTROL_CLEARANCE,
            speed_limit=SOURCE_MAX_SPEED,
            current_velocity=current_velocity,
        )
    result = sweep_obstacle_filter(
        result,
        sweep_constraints,
        speed_limit=SOURCE_MAX_SPEED,
        current_velocity=current_velocity,
        clearance=SCENE_QUERY_CLEARANCE,
    )
    return np.asarray(result, dtype=float)


def _solve_control_job(job):
    """Solve one UAV controller without calling Isaac/PhysX APIs.

    The main thread snapshots rigid-body state and scene-query results before
    dispatch.  This keeps all simulator APIs on their owning thread while the
    independent NumPy safety/controller work scales across UAVs.
    """

    (
        drone_id,
        requested_command,
        position,
        orientation,
        velocity,
        angular_velocity,
        yaw_command,
        motor_rpm,
        execution_safety_points,
        sweep_constraints,
        peer_states,
    ) = job
    safety_decision_started = time.perf_counter()
    point_constraints = None
    if ARGS.scene_usd is None:
        applied_command = np.asarray(
            aabb_obstacle_filter(
                requested_command,
                position,
                SCENARIO.obstacles,
                clearance=OBSTACLE_CONTROL_CLEARANCE,
                speed_limit=SOURCE_MAX_SPEED,
                current_velocity=velocity,
            ),
            dtype=float,
        )
    else:
        point_constraints = pointcloud_obstacle_constraints(
            position,
            execution_safety_points,
            clearance=OBSTACLE_CONTROL_CLEARANCE,
            current_velocity=velocity,
        )
        applied_command = _external_obstacle_filter(
            requested_command,
            position,
            execution_safety_points,
            velocity,
            sweep_constraints,
            point_constraints,
        )
    applied_command = np.asarray(
        cbf_swarm_filter(
            applied_command,
            position,
            peer_states,
            safe_distance=SWARM_CONTROL_DISTANCE,
            speed_limit=SOURCE_MAX_SPEED,
            current_velocity=velocity,
        ),
        dtype=float,
    )
    # Pairwise projection can point toward a nearby wall; retain the obstacle
    # barrier as final authority, using the already-built point constraints.
    if ARGS.scene_usd is None:
        applied_command = np.asarray(
            aabb_obstacle_filter(
                applied_command,
                position,
                SCENARIO.obstacles,
                clearance=OBSTACLE_CONTROL_CLEARANCE,
                speed_limit=SOURCE_MAX_SPEED,
                current_velocity=velocity,
            ),
            dtype=float,
        )
    else:
        applied_command = _external_obstacle_filter(
            applied_command,
            position,
            execution_safety_points,
            velocity,
            sweep_constraints,
            point_constraints,
        )
    safety_decision_ms = 1000.0 * (
        time.perf_counter() - safety_decision_started
    )
    if ARGS.vehicle_model == "racer_so3":
        wrench = velocity_motor_wrench(
            applied_command,
            velocity,
            orientation,
            angular_velocity,
            yaw_command,
            motor_rpm,
            PHYSICS_DT,
        )
    else:
        wrench = velocity_wrench(
            applied_command,
            velocity,
            orientation,
            angular_velocity,
            yaw_command,
        )
    intervened = float(
        np.linalg.norm(applied_command - requested_command)
    ) > 1.0e-3
    return (
        drone_id,
        applied_command,
        wrench,
        intervened,
        safety_decision_ms,
    )


def _backend_array_to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "cpu"):
        host = value.cpu()
        if hasattr(host, "numpy"):
            return np.asarray(host.numpy())
    return np.asarray(value)


def _add_fixed_cube(
    world: World,
    path: str,
    center: Sequence[float],
    size: Sequence[float],
    color: Sequence[float],
) -> FixedCuboid:
    return world.scene.add(
        FixedCuboid(
            prim_path=path,
            name=path.rsplit("/", 1)[-1],
            position=np.asarray(center, dtype=float),
            scale=np.asarray(size, dtype=float),
            color=np.asarray(color, dtype=float),
        )
    )


def _add_crazyflie(
    world: World, stage, drone_id: int, start: Sequence[float]
) -> Tuple[
    SingleRigidPrim,
    RotatingLidarPhysX,
    None,
    Tuple[ContactSensor, ...],
]:
    root_path = f"/World/Drones/drone_{drone_id}"
    root = UsdGeom.Xform.Define(stage, root_path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(CRAZYFLIE_MASS)
    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    rigid_api.CreateDisableGravityAttr(False)
    rigid_api.CreateLinearDampingAttr(0.02)
    rigid_api.CreateAngularDampingAttr(0.02)
    PhysxSchema.PhysxContactReportAPI.Apply(
        root.GetPrim()
    ).CreateThresholdAttr(0.0)

    collider = UsdGeom.Cube.Define(stage, root_path + "/body")
    collider.CreateSizeAttr(1.0)
    collider.AddScaleOp().Set(Gf.Vec3f(*BODY_SIZE))
    collider.CreateDisplayOpacityAttr([0.0])
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(
        collider.GetPrim()
    ).CreateThresholdAttr(0.0)

    if USE_REFERENCE_VISUAL and CRAZYFLIE_ASSET.is_file():
        visual = UsdGeom.Xform.Define(stage, root_path + "/crazyflie_visual")
        visual.GetPrim().GetReferences().AddReference(str(CRAZYFLIE_ASSET))
        visual.AddRotateXOp().Set(90.0)

    body = world.scene.add(
        SingleRigidPrim(
            prim_path=root_path,
            name=f"crazyflie_3d_{drone_id}",
            position=np.asarray(start, dtype=float),
            orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
            mass=CRAZYFLIE_MASS,
            reset_xform_properties=True,
        )
    )
    lidar = world.scene.add(
        RotatingLidarPhysX(
            prim_path=root_path + "/lidar",
            name=f"lidar_3d_{drone_id}",
            translation=LIDAR_TRANSLATION,
            rotation_frequency=0.0,
            fov=(360.0, 120.0),
            resolution=(3.0, 5.0),
            # Exclude the Crazyflie's own 0.16 m collision body. Real vehicle
            # drivers apply the same body/self point-cloud mask.
            valid_range=(0.25, 7.0),
        )
    )
    contact = world.scene.add(
        ContactSensor(
            prim_path=root_path + "/body/contact_sensor",
            name=f"contact_3d_{drone_id}",
            dt=PHYSICS_DT,
            min_threshold=1.0e-4,
            max_threshold=1.0e6,
            radius=-1.0,
        )
    )
    return body, lidar, None, (contact,)


def _add_racer_so3(
    world: World, stage, drone_id: int, start: Sequence[float]
) -> Tuple[
    SingleRigidPrim,
    object,
    object,
    Tuple[ContactSensor, ...],
]:
    """Reference the generated RACER asset and attach runtime sensors."""

    root_path = f"/World/Drones/drone_{drone_id}"
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().GetReferences().AddReference(str(ARGS.vehicle_usd))
    body_path = root_path + "/base_link"
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid() or not body_prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        raise RuntimeError(
            f"generated SO3 asset has no rigid body at {body_path}"
        )
    if ARGS.headless:
        # PhysX lidar still needs periodic Kit updates, but compiling three
        # copies of the imported OmniPBR materials can trigger a Vulkan
        # device-lost fault on the 8 GB test GPU. Keep all collision and
        # dynamics prims active while skipping only headless vehicle drawing.
        # The ROS1 ideal renderer never included the carrying vehicle in its
        # depth image, whereas an RTX camera at the source's zero translation
        # would otherwise start inside the imported fuselage mesh.
        UsdGeom.Imageable(
            stage.GetPrimAtPath(body_path + "/visuals")
        ).MakeInvisible()
    PhysxSchema.PhysxContactReportAPI.Apply(
        body_prim
    ).CreateThresholdAttr(0.0)

    body = world.scene.add(
        SingleRigidPrim(
            prim_path=body_path,
            name=f"racer_so3_3d_{drone_id}",
            position=np.asarray(start, dtype=float),
            orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
            reset_xform_properties=True,
        )
    )
    depth_camera = None
    if ARGS.depth_sensor_backend == "rtx":
        depth_camera = world.scene.add(
            Camera(
                prim_path=body_path + "/depth_camera",
                name=f"racer_depth_camera_{drone_id}",
                frequency=ARGS.sensor_rate_hz,
                resolution=(DEPTH_WIDTH, DEPTH_HEIGHT),
                translation=CAMERA_TRANSLATION,
                # Identity in Isaac's robotics camera convention makes camera
                # +Z optical point along body +X, exactly matching cam02body
                # in the upstream pcl_render_node.
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
            )
        )
        depth_camera.set_opencv_pinhole_properties(
            cx=DEPTH_CX,
            cy=DEPTH_CY,
            fx=DEPTH_FX,
            fy=DEPTH_FY,
            pinhole=[0.0] * 12,
        )
        depth_camera.set_clipping_range(
            # The mapper discards source measurements below 0.2 m. Matching
            # that usable near range also prevents imported self geometry
            # inside the ideal camera's blind zone from becoming an obstacle.
            near_distance=DEPTH_MIN_RANGE,
            far_distance=DEPTH_RENDER_HORIZON,
        )
    else:
        # Keep an explicit mount prim for inspection/debugging, but create no
        # Hydra render product. Warp consumes the same body pose and pinhole
        # intrinsics directly.
        mount = UsdGeom.Xform.Define(stage, body_path + "/depth_camera")
        if np.linalg.norm(CAMERA_TRANSLATION) > 0.0:
            mount.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_TRANSLATION))
    # The upstream forward depth camera remains the exploration sensor. The
    # independent execution layer now receives only a low-density 360-degree
    # Warp ray set configured after the static scene mesh is built. Keep an
    # explicit mount Xform for inspection without creating a PhysX lidar.
    safety_mount = UsdGeom.Xform.Define(stage, body_path + "/safety_lidar")
    safety_mount.AddTranslateOp().Set(Gf.Vec3d(*LIDAR_TRANSLATION))
    safety_lidar = None

    collision_prims = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(body_path + "/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(collision_prims) != 7:
        descendants = [
            (
                str(prim.GetPath()),
                prim.GetTypeName(),
                list(prim.GetAppliedSchemas()),
            )
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(body_path + "/")
        ]
        raise RuntimeError(
            "generated SO3 asset must expose seven collision prims below "
            f"{body_path}, found "
            f"{[str(prim.GetPath()) for prim in collision_prims]}; "
            f"descendants={descendants}"
        )
    # Contact reporting remains enabled on every collider. The default
    # statistics path below uses one PhysX contact-event subscription to find
    # the small set of active colliders, then asks only those sensors for the
    # lightweight bool/scalar reading. Legacy full-frame polling is enabled
    # only for an explicit A/B run.
    # Merely omitting add_raw_contact_data_to_frame() is insufficient in
    # Isaac Sim 5.1: ContactSensor.get_current_frame() itself unconditionally
    # fetches the raw contact buffer on every call.
    contacts = []
    for collision_index, collision_prim in enumerate(collision_prims):
        collision_path = str(collision_prim.GetPath())
        PhysxSchema.PhysxContactReportAPI.Apply(
            collision_prim
        ).CreateThresholdAttr(0.0)
        contacts.append(
            world.scene.add(
                ContactSensor(
                    prim_path=collision_path
                    + f"/contact_sensor_{collision_index}",
                    name=f"contact_3d_{drone_id}_{collision_index}",
                    dt=PHYSICS_DT,
                    min_threshold=1.0e-4,
                    max_threshold=1.0e6,
                    radius=-1.0,
                )
            )
        )
    return body, depth_camera, safety_lidar, tuple(contacts)


def _add_vehicle(
    world: World, stage, drone_id: int, start: Sequence[float]
) -> Tuple[SingleRigidPrim, object, object, Tuple[ContactSensor, ...]]:
    if ARGS.vehicle_model == "racer_so3":
        return _add_racer_so3(world, stage, drone_id, start)
    return _add_crazyflie(world, stage, drone_id, start)


def build_world():
    world = World(
        physics_dt=PHYSICS_DT,
        # Keep one physics step per control update. A larger render dt makes
        # World.step() perform unactuated gravity substeps on rendered frames.
        rendering_dt=PHYSICS_DT,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    colors = (
        (0.16, 0.18, 0.21),
        (0.30, 0.35, 0.42),
        (0.38, 0.31, 0.28),
    )
    if ARGS.scene_usd is None:
        for index, obstacle in enumerate(SCENARIO.obstacles):
            _add_fixed_cube(
                world,
                f"/World/Obstacles/{obstacle.name}_{index}",
                obstacle.center,
                obstacle.size,
                colors[min(2, index // 6)],
            )
    else:
        external = UsdGeom.Xform.Define(stage, "/World/ExternalScene")
        external.GetPrim().GetReferences().AddReference(str(ARGS.scene_usd))
    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(2600.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(45.0, -25.0, 20.0))
    camera = UsdGeom.Camera.Define(stage, "/World/OverviewCamera")
    camera.AddTranslateOp().Set(Gf.Vec3d(0.0, -15.0, 12.0))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, 0.0))
    camera.CreateFocalLengthAttr(24.0)

    bodies, range_sensors, safety_sensors, contacts = [], [], [], []
    for drone_id, start in enumerate(STARTS):
        body, range_sensor, safety_sensor, vehicle_contacts = _add_vehicle(
            world, stage, drone_id, start
        )
        bodies.append(body)
        range_sensors.append(range_sensor)
        safety_sensors.append(safety_sensor)
        contacts.append(vehicle_contacts)
    world.reset()
    body_paths = [
        (
            f"/World/Drones/drone_{drone_id}/base_link"
            if ARGS.vehicle_model == "racer_so3"
            else f"/World/Drones/drone_{drone_id}"
        )
        for drone_id in range(len(bodies))
    ]
    # Isaac Sim 5.1's RigidPrim is the vectorized successor to the old
    # RigidPrimView. Keep the World's NumPy frontend so the unchanged RACER
    # controller sees the same float values, while all PhysX I/O below goes
    # through this single native tensor view.
    rigid_body_view = RigidPrim(
        prim_paths_expr=body_paths,
        name="racer_uav_rigid_body_tensor_view",
        reset_xform_properties=False,
    )
    rigid_body_view.initialize()
    if rigid_body_view.count != len(bodies):
        raise RuntimeError(
            "UAV rigid-body tensor view count mismatch: "
            f"expected {len(bodies)}, found {rigid_body_view.count}"
        )
    if list(rigid_body_view.prim_paths) != body_paths:
        raise RuntimeError(
            "UAV rigid-body tensor view order does not match drone IDs: "
            f"{rigid_body_view.prim_paths}"
        )
    rigid_body_view.enable_gravities()
    imported_masses = np.asarray(
        _backend_array_to_numpy(rigid_body_view.get_masses()), dtype=float
    ).reshape(-1)
    if ARGS.vehicle_model == "racer_so3":
        mismatched = np.flatnonzero(
            ~np.isclose(imported_masses, RACER_SO3_MASS, rtol=1.0e-5)
        )
        if len(mismatched):
            raise RuntimeError(
                "SO3 USD mass mismatch for UAV indices "
                f"{mismatched.tolist()}: {imported_masses[mismatched].tolist()}"
            )
    # World.reset() advances initialization physics in Isaac Sim 5.1. Clear
    # that transient for every UAV with one tensor write before the rotor
    # controller starts.
    rigid_body_view.set_velocities(
        np.zeros((len(bodies), 6), dtype=np.float32)
    )
    for range_sensor in range_sensors:
        if (
            ARGS.vehicle_model == "racer_so3"
            and ARGS.depth_sensor_backend == "rtx"
        ):
            range_sensor.add_distance_to_image_plane_to_frame()
        elif ARGS.vehicle_model != "racer_so3":
            range_sensor.add_point_cloud_data_to_frame()
    for safety_sensor in safety_sensors:
        if safety_sensor is not None:
            safety_sensor.add_point_cloud_data_to_frame()
    return (
        world,
        bodies,
        rigid_body_view,
        range_sensors,
        safety_sensors,
        contacts,
    )


def _yaw_from_quaternion(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class PropellerVisualAnimator:
    """Animate render-only blade Xforms without adding physics articulation."""

    DEGREES_PER_RPM_SECOND = 6.0
    AXIS_INDICES = {"X": 0, "Y": 1, "Z": 2}

    def __init__(
        self,
        stage,
        body_paths: Sequence[str],
        enabled: bool,
        update_hz: float,
        strict: bool = False,
    ) -> None:
        self.enabled = False
        self.update_hz = float(update_hz)
        self.update_period = 1.0 / self.update_hz
        self.accumulated_time = 0.0
        self.angles_deg = np.zeros((len(body_paths), 4), dtype=float)
        self.spin_signs = np.zeros((len(body_paths), 4), dtype=float)
        self.entries = []
        self.status = "disabled"
        if not enabled:
            return

        missing = []
        for drone_id, body_path in enumerate(body_paths):
            visuals = stage.GetPrimAtPath(body_path + "/visuals")
            if not visuals.IsValid():
                missing.append(f"{body_path}/visuals")
                continue
            candidates = []
            for prim in Usd.PrimRange(visuals):
                rotor_id_attr = prim.GetAttribute("racer:rotorId")
                if rotor_id_attr.IsValid() and rotor_id_attr.HasAuthoredValueOpinion():
                    candidates.append((int(rotor_id_attr.Get()), prim))
            candidates.sort(key=lambda item: item[0])
            if [item[0] for item in candidates] != [0, 1, 2, 3]:
                missing.append(
                    f"{body_path}/visuals rotor_ids="
                    f"{[item[0] for item in candidates]}"
                )
                continue

            for rotor_id, prim in candidates:
                sign_attr = prim.GetAttribute("racer:spinDirectionSign")
                axis_attr = prim.GetAttribute("racer:visualRotationAxis")
                op_attr = prim.GetAttribute("racer:visualRotationOp")
                sign = int(sign_attr.Get()) if sign_attr.IsValid() else 0
                axis = str(axis_attr.Get()).upper() if axis_attr.IsValid() else ""
                op_name = str(op_attr.Get()) if op_attr.IsValid() else ""
                rotation_attr = prim.GetAttribute(op_name)
                if sign not in (-1, 1):
                    raise RuntimeError(
                        f"invalid propeller spin sign at {prim.GetPath()}: {sign}"
                    )
                if axis not in self.AXIS_INDICES:
                    raise RuntimeError(
                        f"invalid propeller rotation axis at {prim.GetPath()}: {axis}"
                    )
                if not rotation_attr.IsValid() or rotation_attr.Get() is None:
                    raise RuntimeError(
                        f"propeller rotation op {op_name!r} is missing at "
                        f"{prim.GetPath()}"
                    )
                base_rotation = rotation_attr.Get()
                self.spin_signs[drone_id, rotor_id] = sign
                self.entries.append(
                    {
                        "drone_id": drone_id,
                        "rotor_id": rotor_id,
                        "path": str(prim.GetPath()),
                        "attribute": rotation_attr,
                        "axis_index": self.AXIS_INDICES[axis],
                        "base_rotation": tuple(float(value) for value in base_rotation),
                        "vector_type": type(base_rotation),
                    }
                )

        if missing:
            message = "propeller animation metadata unavailable: " + "; ".join(missing)
            if strict:
                raise RuntimeError(message)
            self.status = message
            self.entries = []
            return

        self.enabled = True
        self.status = "ready"
        self._write_angles()

    def _write_angles(self) -> None:
        for entry in self.entries:
            values = list(entry["base_rotation"])
            values[entry["axis_index"]] += self.angles_deg[
                entry["drone_id"], entry["rotor_id"]
            ]
            entry["attribute"].Set(entry["vector_type"](*values))

    def step(self, motor_rpms: Sequence[Sequence[float]], dt: float) -> None:
        if not self.enabled:
            return
        rpm = np.asarray(motor_rpms, dtype=float)
        if rpm.shape != self.angles_deg.shape:
            raise RuntimeError(
                f"propeller RPM shape {rpm.shape} does not match "
                f"visual shape {self.angles_deg.shape}"
            )
        self.angles_deg = np.remainder(
            self.angles_deg
            + self.spin_signs * rpm * self.DEGREES_PER_RPM_SECOND * float(dt),
            360.0,
        )
        self.accumulated_time += float(dt)
        if self.accumulated_time + 1.0e-12 < self.update_period:
            return
        self.accumulated_time %= self.update_period
        self._write_angles()

    def reset(self) -> None:
        if not self.enabled:
            return
        self.angles_deg.fill(0.0)
        self.accumulated_time = 0.0
        self._write_angles()

    def report(self) -> dict:
        return {
            "status": self.status,
            "enabled": self.enabled,
            "propeller_count": len(self.entries),
            "update_hz": self.update_hz,
            "degrees_per_rpm_second": self.DEGREES_PER_RPM_SECOND,
            "paths": [entry["path"] for entry in self.entries],
        }


class IsaacRacer3DBridge(Node):
    def __init__(
        self,
        bodies,
        rigid_body_view,
        range_sensors,
        safety_sensors,
        contacts,
        warp_raycaster=None,
        shared_physical=None,
        shared_physical_fast=None,
        shared_state_event=None,
        shared_status=None,
    ) -> None:
        super().__init__("isaac_racer_3d_bridge")
        self.bodies = bodies
        self.rigid_body_view = rigid_body_view
        self.range_sensors = range_sensors
        self.safety_sensors = safety_sensors
        self.contacts = contacts
        self.warp_raycaster = warp_raycaster
        self.shared_physical = shared_physical
        self.shared_physical_fast = shared_physical_fast
        self.shared_state_event = shared_state_event
        self.shared_status = shared_status
        self.shared_physical_period_s = (
            1.0 / ARGS.shared_memory_physical_rate_hz
        )
        self.shared_physical_last_stamp = -math.inf
        self.shared_physical_last_coverage = 0.0
        self.shared_physical_version = 0
        self.warp_raycaster_report = (
            warp_raycaster.report() if warp_raycaster is not None else None
        )
        self.drone_count = len(bodies)
        if self.rigid_body_view.count != self.drone_count:
            raise RuntimeError(
                "rigid-body tensor view does not cover every UAV"
            )
        self.rigid_body_masses = np.asarray(
            _backend_array_to_numpy(self.rigid_body_view.get_masses()),
            dtype=float,
        ).reshape(-1)
        self.rigid_io_profile = {
            name: {"calls": 0, "total_ms": 0.0, "max_ms": 0.0}
            for name in (
                "initial_sync_state_batch_read",
                "pre_state_batch_read",
                "force_torque_batch_write",
                "post_state_batch_read",
            )
        }
        self.pre_control_positions = np.zeros(
            (self.drone_count, 3), dtype=float
        )
        self.pre_control_orientations = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float),
            (self.drone_count, 1),
        )
        self.pre_control_velocities = np.zeros(
            (self.drone_count, 3), dtype=float
        )
        self.pre_control_angular_velocities = np.zeros(
            (self.drone_count, 3), dtype=float
        )
        self.sensor_executor = (
            ThreadPoolExecutor(
                max_workers=min(ARGS.sensor_worker_count, self.drone_count),
                thread_name_prefix="racer_sensor",
            )
            if ARGS.sensor_worker_count > 1 and self.drone_count > 1
            else None
        )
        self.sensor_worker_count = (
            min(ARGS.sensor_worker_count, self.drone_count)
            if self.sensor_executor is not None
            else 1
        )
        self.commands = np.zeros((self.drone_count, 3), dtype=float)
        self.applied_commands = np.zeros(
            (self.drone_count, 3), dtype=float
        )
        self.safety_points = [np.empty((0, 3), dtype=float) for _ in bodies]
        self.scene_query_points = [
            np.empty((0, 3), dtype=float) for _ in bodies
        ]
        self.execution_safety_points = [
            np.empty((0, 3), dtype=float) for _ in bodies
        ]
        self.scene_query_constraints = [[] for _ in bodies]
        self.last_scene_query = [-math.inf for _ in bodies]
        self.scene_query_updates = 0
        self.scene_query_hits = 0
        self.min_sweep_free_travel = math.inf
        self.scene_query = get_physx_scene_query_interface()
        # Safety points are one current low-density 360-degree Warp scan per
        # UAV. Dense mapping-camera hits never enter this execution-only path;
        # the PhysX swept vehicle envelope remains the thin-obstacle backstop.
        self.yaw_commands = np.zeros(self.drone_count, dtype=float)
        self.yaw_targets = [0.0 for _ in bodies]
        self.startup_yaws = [0.0 for _ in bodies]
        self.startup_corridor_clearances = [0.0 for _ in bodies]
        self.startup_corridor_enabled = [False for _ in bodies]
        self.positions = np.zeros((self.drone_count, 3), dtype=float)
        self.orientations = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float),
            (self.drone_count, 1),
        )
        self.velocities = np.zeros((self.drone_count, 3), dtype=float)
        self.angular_velocities = np.zeros(
            (self.drone_count, 3), dtype=float
        )
        self.accelerations = np.zeros((self.drone_count, 3), dtype=float)
        self.previous_velocities = [None for _ in bodies]
        self.path_lengths = [0.0 for _ in bodies]
        self.agentic_crpo_last_coverage = 0.0
        self.previous_positions = [None for _ in bodies]
        self.motor_thrusts = np.zeros((self.drone_count, 4), dtype=float)
        self.motor_rpms = np.full(
            (self.drone_count, 4),
            racer_hover_rpm() if ARGS.vehicle_model == "racer_so3" else 0.0,
            dtype=float,
        )
        self.control_obstacle_minimums = np.asarray(
            [obstacle.minimum for obstacle in SCENARIO.obstacles],
            dtype=float,
        ).reshape((-1, 3))
        self.control_obstacle_maximums = np.asarray(
            [obstacle.maximum for obstacle in SCENARIO.obstacles],
            dtype=float,
        ).reshape((-1, 3))
        self.propeller_visuals = PropellerVisualAnimator(
            omni.usd.get_context().get_stage(),
            [
                f"/World/Drones/drone_{drone_id}/base_link"
                for drone_id in range(self.drone_count)
            ],
            enabled=(ANIMATE_PROPELLERS and ARGS.vehicle_model == "racer_so3"),
            update_hz=ARGS.propeller_visual_hz,
            strict=(ARGS.animate_propellers is True),
        )
        print(
            "RACER_3D_PROPELLER_VISUALS "
            + json.dumps(self.propeller_visuals.report(), sort_keys=True),
            flush=True,
        )
        self.elapsed = 0.0
        self.last_odom = -math.inf
        self.last_depth = -math.inf
        self.collision_events = 0
        self.contact_active = [False for _ in bodies]
        self.max_contact_force = 0.0
        legacy_sensors_per_uav = (
            7 if ARGS.vehicle_model == "racer_so3" else 1
        )
        self.contact_profile = {
            "fast_path": {
                "steps": 0,
                "light_sensor_reading_calls": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
            },
            "event_callback": {
                "calls": 0,
                "headers": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
            },
            "legacy_reference": {
                "enabled": bool(ARGS.contact_regression),
                "steps": 0,
                "sensor_get_current_frame_calls": 0,
                "expected_sensor_calls_per_step": (
                    self.drone_count * legacy_sensors_per_uav
                ),
                "total_ms": 0.0,
                "max_ms": 0.0,
            },
        }
        self.contact_regression = {
            "compared_steps": 0,
            "active_mismatch_steps": 0,
            "active_mismatch_uav_samples": 0,
            "force_compared_uav_samples": 0,
            "force_absolute_error_sum_n": 0.0,
            "force_max_absolute_error_n": 0.0,
            "legacy_collision_events": 0,
            "legacy_max_contact_force_n": 0.0,
        }
        self.legacy_contact_active = [False for _ in bodies]
        self.contact_pairs_by_sensor = [
            [set() for _ in sensors] for sensors in self.contacts
        ]
        self.contact_sensor_by_collider_path = {}
        self.contact_sensor_by_collider_handle = {}
        for drone_id, sensors in enumerate(self.contacts):
            for sensor_index, sensor in enumerate(sensors):
                collider_path = sensor.prim_path.rsplit("/", 1)[0]
                if collider_path in self.contact_sensor_by_collider_path:
                    raise RuntimeError(
                        "duplicate contact sensor collider path: "
                        f"{collider_path}"
                    )
                self.contact_sensor_by_collider_path[collider_path] = (
                    drone_id,
                    sensor_index,
                )
        self.contact_event_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report_event
            )
        )
        self.min_inter_drone = math.inf
        self.min_obstacle_clearance = math.inf
        self.cloud_frames = 0
        self.sensor_profile_frames = 0
        self.sensor_profile_sums_ms = {
            "raycasting": 0.0,
            "safety_raycasting": 0.0,
            "safety_pointcloud_process": 0.0,
            "rtx_render_step": 0.0,
            "decode_backprojection": 0.0,
            "sensor_postprocess": 0.0,
            "pointcloud2": 0.0,
            "publish": 0.0,
            "total": 0.0,
        }
        self.sensor_profile_max_ms = {
            name: 0.0 for name in self.sensor_profile_sums_ms
        }
        self.pending_rtx_render_step_ms = 0.0
        self.raw_diagnostics_printed = False
        self.control_steps = 0
        self.safety_interventions = 0
        self.safety_decision_profile_frames = 0
        self.safety_decision_profile_sum_ms = 0.0
        self.safety_decision_profile_max_ms = 0.0
        self.safety_decision_worker_max_sum_ms = 0.0
        self.safety_decision_worker_max_max_ms = 0.0
        self.last_safety_decision_ms = 0.0
        self.control_batch_profile_frames = 0
        self.control_batch_profile_sum_ms = 0.0
        self.control_batch_profile_max_ms = 0.0
        self.control_batch_openmp_threads = 0
        self.control_batch_component_sum_ms = {
            name: 0.0
            for name in (
                "constraint_build",
                "obstacle_projection",
                "swarm_cbf",
                "so3",
            )
        }
        self.control_batch_component_max_ms = {
            name: 0.0 for name in self.control_batch_component_sum_ms
        }
        self.mission_complete = False
        self.mapping_coverage = [None for _ in bodies]
        self.mapping_coverage_counts = [None for _ in bodies]
        self.mapping_coverage_history = []
        self.mapping_coverage_history_last_stamp = [
            -math.inf for _ in bodies
        ]
        self.mapping_coverage_joint = None
        self.mapping_coverage_joint_known = 0
        self.mapping_coverage_joint_total = None
        self.mapping_coverage_joint_bitmap = None
        self.mapping_coverage_bitmap_versions = [0 for _ in bodies]
        self.mapping_coverage_joint_history_version = 0
        self.mapping_coverage_joint_history = []
        self.mapping_coverage_target_reached = False
        self.trajectory_history = []
        self.last_trajectory_history_stamp = -math.inf
        self.debug_draw = None
        self.visual_map_points = np.empty((0, 3), dtype=np.float32)
        self.visual_paths = [np.empty((0, 3), dtype=np.float32) for _ in bodies]
        self.visual_markers = {}
        self.visual_trails = [
            [tuple(float(value) for value in STARTS[index])]
            for index in range(self.drone_count)
        ]
        self.visualization_dirty = False
        self.last_visualization_update = -math.inf
        self.last_trail_update = -math.inf
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        cloud_qos = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.odom_publishers = []
        self.imu_publishers = []
        self.cloud_publishers = []
        for drone_id in range(self.drone_count):
            namespace = f"/drone_{drone_id}"
            self.odom_publishers.append(
                self.create_publisher(Odometry, namespace + "/odom", qos)
            )
            self.imu_publishers.append(
                self.create_publisher(Imu, namespace + "/imu", qos)
            )
            self.cloud_publishers.append(
                self.create_publisher(
                    PointCloud2, namespace + "/points", cloud_qos
                )
            )
            self.create_subscription(
                Twist,
                namespace + "/cmd_vel_3d",
                lambda message, index=drone_id: self._command(index, message),
                qos,
            )
        if ARGS.visualize_exploration:
            self.debug_draw = _debug_draw.acquire_debug_draw_interface()
            visual_qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE
            )
            for drone_id in range(self.drone_count):
                for channel in (
                    "trajectory",
                    "frontier",
                    "viewpoints",
                    "travel_traj",
                    "position_cmd",
                    "map_chunks",
                ):
                    self.create_subscription(
                        Marker,
                        f"/drone_{drone_id}/planning_vis/{channel}",
                        lambda message, index=drone_id, name=channel:
                            self._visual_marker(index, name, message),
                        visual_qos,
                    )
        self.metrics_publisher = self.create_publisher(
            String, "/racer_3d/sim_metrics", qos
        )
        # The 200 Hz source-faithful plant can run slower than wall time.
        # Exploration trajectories and peer timeouts must therefore use the
        # physics clock, otherwise their time axis advances while the vehicle
        # has not yet completed the corresponding PhysX motion.
        self.clock_publisher = self.create_publisher(
            Clock, "/clock", QoSProfile(depth=10)
        )
        self.create_subscription(
            String,
            "/racer_3d/mission_complete",
            self._mission_complete,
            qos,
        )
        for drone_id in range(self.drone_count):
            self.create_subscription(
                String,
                (
                    f"/racer_original_exploration_{drone_id + 1}"
                    "/mapping_coverage"
                ),
                lambda message, index=drone_id:
                    self._mapping_coverage(index, message),
                qos,
            )
            self.create_subscription(
                UInt8MultiArray,
                (
                    f"/racer_original_exploration_{drone_id + 1}"
                    "/mapping_known_bitmap"
                ),
                lambda message, index=drone_id:
                    self._mapping_known_bitmap(index, message),
                qos,
            )
        self._read_physics(count_distance=False)

    def _record_rigid_io_profile(self, name: str, elapsed_ms: float) -> None:
        profile = self.rigid_io_profile[name]
        profile["calls"] += 1
        profile["total_ms"] += elapsed_ms
        profile["max_ms"] = max(profile["max_ms"], elapsed_ms)

    def reset_rigid_io_profile(self) -> None:
        for profile in self.rigid_io_profile.values():
            profile["calls"] = 0
            profile["total_ms"] = 0.0
            profile["max_ms"] = 0.0

    def rigid_io_profile_report(self):
        return {
            "view": "isaacsim.core.prims.RigidPrim",
            "backend": "PhysX tensor view with NumPy frontend",
            "uav_count": self.drone_count,
            **{
                name: {
                    "calls": profile["calls"],
                    "mean_ms": (
                        profile["total_ms"] / profile["calls"]
                        if profile["calls"]
                        else 0.0
                    ),
                    "max_ms": profile["max_ms"],
                }
                for name, profile in self.rigid_io_profile.items()
            },
        }

    def reset_contact_profile(self) -> None:
        fast = self.contact_profile["fast_path"]
        fast.update(
            steps=0,
            light_sensor_reading_calls=0,
            total_ms=0.0,
            max_ms=0.0,
        )
        callback = self.contact_profile["event_callback"]
        callback.update(calls=0, headers=0, total_ms=0.0, max_ms=0.0)
        legacy = self.contact_profile["legacy_reference"]
        legacy.update(
            steps=0,
            sensor_get_current_frame_calls=0,
            total_ms=0.0,
            max_ms=0.0,
        )
        self.contact_regression.update(
            compared_steps=0,
            active_mismatch_steps=0,
            active_mismatch_uav_samples=0,
            force_compared_uav_samples=0,
            force_absolute_error_sum_n=0.0,
            force_max_absolute_error_n=0.0,
            legacy_collision_events=0,
            legacy_max_contact_force_n=0.0,
        )
        self.legacy_contact_active = [
            False for _ in range(self.drone_count)
        ]
        self.contact_pairs_by_sensor = [
            [set() for _ in sensors] for sensors in self.contacts
        ]

    def contact_profile_report(self) -> dict:
        fast = self.contact_profile["fast_path"]
        callback = self.contact_profile["event_callback"]
        legacy = self.contact_profile["legacy_reference"]
        compared = self.contact_regression["force_compared_uav_samples"]
        return {
            "backend": (
                "PhysX contact events plus sparse ContactSensor reading"
            ),
            "raw_contact_data_requested": False,
            "uav_count": self.drone_count,
            "fast_path": {
                "steps": fast["steps"],
                "light_sensor_reading_calls": (
                    fast["light_sensor_reading_calls"]
                ),
                "mean_step_ms": (
                    fast["total_ms"] / fast["steps"]
                    if fast["steps"]
                    else 0.0
                ),
                "max_step_ms": fast["max_ms"],
            },
            "event_callback": {
                "calls": callback["calls"],
                "headers": callback["headers"],
                "mean_callback_ms": (
                    callback["total_ms"] / callback["calls"]
                    if callback["calls"]
                    else 0.0
                ),
                "max_callback_ms": callback["max_ms"],
                "amortized_ms_per_physics_step": (
                    callback["total_ms"] / fast["steps"]
                    if fast["steps"]
                    else 0.0
                ),
            },
            "mean_total_contact_handling_ms_per_step": (
                (fast["total_ms"] + callback["total_ms"])
                / fast["steps"]
                if fast["steps"]
                else 0.0
            ),
            "legacy_reference": {
                "enabled": legacy["enabled"],
                "steps": legacy["steps"],
                "sensor_get_current_frame_calls": (
                    legacy["sensor_get_current_frame_calls"]
                ),
                "expected_sensor_calls_per_step": (
                    legacy["expected_sensor_calls_per_step"]
                ),
                "mean_step_ms": (
                    legacy["total_ms"] / legacy["steps"]
                    if legacy["steps"]
                    else None
                ),
                "max_step_ms": (
                    legacy["max_ms"] if legacy["steps"] else None
                ),
            },
            "regression": {
                **self.contact_regression,
                "force_mean_absolute_error_n": (
                    self.contact_regression[
                        "force_absolute_error_sum_n"
                    ] / compared
                    if compared
                    else 0.0
                ),
                "fast_collision_events": self.collision_events,
                "fast_max_contact_force_n": self.max_contact_force,
                "force_definition_note": (
                    "both paths use the maximum scalar reading among that "
                    "UAV's collider sensors; the fast path queries only "
                    "event-active colliders"
                ),
            },
        }

    def control_batch_profile_report(self):
        frames = self.control_batch_profile_frames
        denominator = max(1, frames * self.drone_count)
        return {
            "backend": (
                "pybind11 C++ OpenMP batch with GIL released"
                if ARGS.vehicle_model == "racer_so3"
                else "legacy Python per-UAV controller"
            ),
            "frames": frames,
            "uav_count": self.drone_count,
            "openmp_threads": self.control_batch_openmp_threads,
            "mean_step_wall_ms": (
                self.control_batch_profile_sum_ms / frames
                if frames
                else 0.0
            ),
            "max_step_wall_ms": self.control_batch_profile_max_ms,
            "mean_worker_component_ms": {
                name: total / denominator
                for name, total in self.control_batch_component_sum_ms.items()
            },
            "max_worker_component_ms": self.control_batch_component_max_ms,
        }

    def _read_rigid_body_state_batch(self, profile_name: str):
        """Read all transforms and 6-D velocities from one tensor view."""

        started = time.perf_counter()
        # PhysX exposes transforms and velocities as two native tensor
        # buffers. Each is fetched once for the complete view; importantly,
        # there are no per-prim PhysX getters here.
        positions_value, orientations_value = (
            self.rigid_body_view.get_world_poses(clone=False)
        )
        velocities_value = self.rigid_body_view.get_velocities(clone=False)
        positions = np.array(
            _backend_array_to_numpy(positions_value), dtype=float, copy=True
        ).reshape((self.drone_count, 3))
        orientations = np.array(
            _backend_array_to_numpy(orientations_value),
            dtype=float,
            copy=True,
        ).reshape((self.drone_count, 4))
        velocities = np.array(
            _backend_array_to_numpy(velocities_value), dtype=float, copy=True
        ).reshape((self.drone_count, 6))
        linear_velocities = velocities[:, :3]
        angular_velocities = velocities[:, 3:]
        if not all(
            np.all(np.isfinite(values))
            for values in (
                positions,
                orientations,
                linear_velocities,
                angular_velocities,
            )
        ):
            raise RuntimeError("non-finite UAV state returned by PhysX tensor view")
        self._record_rigid_io_profile(
            profile_name,
            1000.0 * (time.perf_counter() - started),
        )
        return (
            positions,
            orientations,
            linear_velocities,
            angular_velocities,
        )

    def _visual_map(self, message: PointCloud2) -> None:
        points, _ = read_xyzi_cloud(message)
        if len(points) > ARGS.visualization_max_map_points:
            indices = np.linspace(
                0,
                len(points) - 1,
                ARGS.visualization_max_map_points,
                dtype=np.int64,
            )
            points = points[indices]
        self.visual_map_points = np.asarray(points, dtype=np.float32)
        self.visualization_dirty = True

    def _visual_path(self, drone_id: int, message: RosPath) -> None:
        self.visual_paths[drone_id] = np.asarray(
            [
                (
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                )
                for pose in message.poses
            ],
            dtype=np.float32,
        ).reshape((-1, 3))
        self.visualization_dirty = True

    def _visual_marker(
        self, drone_id: int, channel: str, message: Marker
    ) -> None:
        key = (drone_id, channel, message.ns, int(message.id))
        if message.action == Marker.DELETEALL:
            self.visual_markers = {
                item_key: value
                for item_key, value in self.visual_markers.items()
                if item_key[0] != drone_id or item_key[1] != channel
            }
        elif message.action == Marker.DELETE:
            self.visual_markers.pop(key, None)
        else:
            points = np.asarray(
                [(point.x, point.y, point.z) for point in message.points],
                dtype=np.float32,
            ).reshape((-1, 3))
            if not len(points) and message.type in (Marker.CUBE, Marker.SPHERE):
                points = np.asarray(
                    [[message.pose.position.x, message.pose.position.y,
                      message.pose.position.z]], dtype=np.float32
                )
            color = (
                float(message.color.r), float(message.color.g),
                float(message.color.b), max(0.15, float(message.color.a)),
            )
            self.visual_markers[key] = {
                "type": int(message.type),
                "points": points,
                "color": color,
                "width": max(1.0, 20.0 * float(message.scale.x)),
            }
        self.visualization_dirty = True

    @staticmethod
    def _line_segments(points: Sequence[Sequence[float]]):
        if len(points) < 2:
            return [], []
        values = np.asarray(points, dtype=float).reshape((-1, 3))
        return values[:-1].tolist(), values[1:].tolist()

    def update_visualization(self, force: bool = False) -> None:
        if self.debug_draw is None:
            return
        if self.elapsed - self.last_trail_update >= 0.20:
            for drone_id, position in enumerate(self.positions):
                self.visual_trails[drone_id].append(
                    tuple(float(value) for value in position)
                )
                if len(self.visual_trails[drone_id]) > 2500:
                    self.visual_trails[drone_id] = self.visual_trails[drone_id][
                        -2500:
                    ]
            self.last_trail_update = self.elapsed
            self.visualization_dirty = True
        if not self.visualization_dirty:
            return
        if not force and self.elapsed - self.last_visualization_update < 0.20:
            return

        self.debug_draw.clear_points()
        self.debug_draw.clear_lines()
        if len(self.visual_map_points):
            points = self.visual_map_points.tolist()
            self.debug_draw.draw_points(
                points,
                [VISUAL_MAP_COLOR] * len(points),
                [3.0] * len(points),
            )
        marker_points, marker_colors, marker_sizes = [], [], []
        for marker in self.visual_markers.values():
            if marker["type"] in (
                Marker.CUBE, Marker.SPHERE, Marker.CUBE_LIST,
                Marker.SPHERE_LIST, Marker.POINTS,
            ):
                values = marker["points"].tolist()
                marker_points.extend(values)
                marker_colors.extend([marker["color"]] * len(values))
                marker_sizes.extend([marker["width"]] * len(values))
        if marker_points:
            self.debug_draw.draw_points(
                marker_points, marker_colors, marker_sizes
            )
        drone_points = [position.tolist() for position in self.positions]
        drone_colors = [
            VISUAL_DRONE_COLORS[index % len(VISUAL_DRONE_COLORS)]
            for index in range(self.drone_count)
        ]
        self.debug_draw.draw_points(
            drone_points, drone_colors, [14.0] * self.drone_count
        )

        starts, ends, colors, widths = [], [], [], []
        for drone_id in range(self.drone_count):
            color = VISUAL_DRONE_COLORS[
                drone_id % len(VISUAL_DRONE_COLORS)
            ]
            trail_starts, trail_ends = self._line_segments(
                self.visual_trails[drone_id]
            )
            starts.extend(trail_starts)
            ends.extend(trail_ends)
            colors.extend(
                [(color[0], color[1], color[2], 0.55)]
                * len(trail_starts)
            )
            widths.extend([2.0] * len(trail_starts))
            path_starts, path_ends = self._line_segments(
                self.visual_paths[drone_id]
            )
            starts.extend(path_starts)
            ends.extend(path_ends)
            colors.extend([color] * len(path_starts))
            widths.extend([4.0] * len(path_starts))
        for marker in self.visual_markers.values():
            points = marker["points"]
            if marker["type"] == Marker.LINE_LIST:
                count = len(points) - len(points) % 2
                marker_starts = points[:count:2].tolist()
                marker_ends = points[1:count:2].tolist()
            elif marker["type"] == Marker.LINE_STRIP:
                marker_starts, marker_ends = self._line_segments(points)
            else:
                continue
            starts.extend(marker_starts)
            ends.extend(marker_ends)
            colors.extend([marker["color"]] * len(marker_starts))
            widths.extend([marker["width"]] * len(marker_starts))
        if starts:
            self.debug_draw.draw_lines(starts, ends, colors, widths)
        self.last_visualization_update = self.elapsed
        self.visualization_dirty = False

    def _command(self, drone_id: int, message: Twist) -> None:
        self.commands[drone_id] = np.asarray(
            (message.linear.x, message.linear.y, message.linear.z), dtype=float
        )
        self.yaw_targets[drone_id] = float(message.angular.z)

    def _mission_complete(self, message: String) -> None:
        self.mission_complete = message.data.strip().lower() == "true"

    def _mapping_coverage(self, drone_id: int, message: String) -> None:
        try:
            payload = json.loads(message.data)
            ratio = float(payload["ratio"])
            known = int(payload["known_voxels"])
            total = int(payload["total_voxels"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if not (0.0 <= ratio <= 1.0) or known < 0 or total <= 0:
            return
        self.mapping_coverage[drone_id] = ratio
        self.mapping_coverage_counts[drone_id] = {
            "known_voxels": known,
            "total_voxels": total,
        }
        if (
            self.elapsed
            - self.mapping_coverage_history_last_stamp[drone_id]
            >= 1.0
        ):
            self.mapping_coverage_history.append(
                {
                    "time_s": self.elapsed,
                    "drone_id": drone_id,
                    "ratio": ratio,
                    "known_voxels": known,
                    "total_voxels": total,
                }
            )
            self.mapping_coverage_history_last_stamp[drone_id] = self.elapsed

    def _mapping_known_bitmap(
        self, drone_id: int, message: UInt8MultiArray
    ) -> None:
        if len(message.layout.dim) != 1:
            return
        dimension = message.layout.dim[0]
        total = int(dimension.size)
        expected_bytes = (total + 7) // 8
        if (
            dimension.label != "planning_box_known_bitmap"
            or total <= 0
            or int(dimension.stride) != expected_bytes
            or len(message.data) != expected_bytes
        ):
            return
        if self.mapping_coverage_joint_total is None:
            self.mapping_coverage_joint_total = total
            self.mapping_coverage_joint_bitmap = np.zeros(
                expected_bytes, dtype=np.uint8
            )
        if total != self.mapping_coverage_joint_total:
            return
        incoming = np.frombuffer(memoryview(message.data), dtype=np.uint8)
        np.bitwise_or(
            self.mapping_coverage_joint_bitmap,
            incoming,
            out=self.mapping_coverage_joint_bitmap,
        )
        self.mapping_coverage_joint_known = int(
            _BYTE_POPCOUNT[self.mapping_coverage_joint_bitmap].sum(
                dtype=np.uint64
            )
        )
        self.mapping_coverage_joint = (
            self.mapping_coverage_joint_known
            / self.mapping_coverage_joint_total
        )
        self.mapping_coverage_bitmap_versions[drone_id] += 1
        complete_version = min(self.mapping_coverage_bitmap_versions)
        if complete_version > self.mapping_coverage_joint_history_version:
            self.mapping_coverage_joint_history.append(
                {
                    "time_s": self.elapsed,
                    "ratio": self.mapping_coverage_joint,
                    "known_voxels": self.mapping_coverage_joint_known,
                    "total_voxels": self.mapping_coverage_joint_total,
                    "bitmap_version": complete_version,
                }
            )
            self.mapping_coverage_joint_history_version = complete_version
        self.mapping_coverage_target_reached = bool(
            ARGS.mapping_coverage_target > 0.0
            and self.mapping_coverage_joint
            >= ARGS.mapping_coverage_target
        )

    def configure_startup_recovery(self) -> None:
        """Choose truth-checked launch headings without seeding the RACER map."""

        if not ARGS.startup_free_space_yaw or ARGS.scene_usd is None:
            return
        query_range = max(
            SCENE_QUERY_RANGE,
            ARGS.startup_unknown_corridor_distance
            + SCENE_QUERY_CLEARANCE
            + 0.25,
        )
        samples = 72
        startup_orientations = self.orientations.copy()
        for drone_id, position_value in enumerate(self.positions):
            position = np.asarray(position_value, dtype=float)
            own_prefix = f"/World/Drones/drone_{drone_id}/"
            best_yaw = 0.0
            best_clearance = -math.inf
            for sample in range(samples):
                yaw = 2.0 * math.pi * sample / samples
                direction = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
                direction_hits = []

                def report(hit):
                    rigid_body = str(hit.rigid_body)
                    collision = str(getattr(hit, "collision", ""))
                    if not (
                        rigid_body.startswith(own_prefix)
                        or collision.startswith(own_prefix)
                    ):
                        distance = float(getattr(hit, "distance", math.inf))
                        if math.isfinite(distance) and distance >= 0.0:
                            direction_hits.append(distance)
                    return True

                self.scene_query.sweep_sphere_all(
                    VEHICLE_RADIUS,
                    Gf.Vec3f(*position),
                    Gf.Vec3f(*direction),
                    query_range,
                    report,
                )
                clearance = min(direction_hits) if direction_hits else query_range
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_yaw = yaw

            self.startup_yaws[drone_id] = best_yaw
            self.startup_corridor_clearances[drone_id] = best_clearance
            self.startup_corridor_enabled[drone_id] = bool(
                best_clearance
                >= ARGS.startup_unknown_corridor_distance
                + SCENE_QUERY_CLEARANCE
            )
            self.yaw_commands[drone_id] = best_yaw
            self.yaw_targets[drone_id] = best_yaw
            startup_orientations[drone_id] = np.asarray(
                (
                    math.cos(0.5 * best_yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * best_yaw),
                ),
                dtype=float,
            )
        self.rigid_body_view.set_world_poses(
            orientations=startup_orientations.astype(np.float32)
        )
        self.orientations = startup_orientations
        print(
            "RACER_3D_STARTUP_RECOVERY "
            + json.dumps(
                {
                    "mode": "max_free_yaw_scan_truth_checked_unknown_corridor",
                    "yaw_samples": samples,
                    "selected_yaws_rad": self.startup_yaws,
                    "corridor_clearances_m": self.startup_corridor_clearances,
                    "corridor_enabled": self.startup_corridor_enabled,
                    "scan_duration_s": ARGS.startup_scan_duration,
                    "corridor_distance_m": ARGS.startup_unknown_corridor_distance,
                    "corridor_speed_mps": ARGS.startup_corridor_speed,
                    "settle_duration_s": ARGS.startup_settle_duration,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _startup_override(self, drone_id: int):
        if not ARGS.startup_free_space_yaw:
            return None
        scan_end = ARGS.startup_scan_duration
        corridor_duration = (
            ARGS.startup_unknown_corridor_distance
            / ARGS.startup_corridor_speed
        )
        corridor_end = scan_end + corridor_duration
        startup_end = corridor_end + ARGS.startup_settle_duration
        if self.elapsed >= startup_end:
            return None
        yaw = self.startup_yaws[drone_id]
        command = np.zeros(3, dtype=float)
        if self.elapsed < scan_end and scan_end > 0.0:
            yaw += 2.0 * math.pi * self.elapsed / scan_end
        elif (
            self.elapsed < corridor_end
            and self.startup_corridor_enabled[drone_id]
        ):
            command[:2] = ARGS.startup_corridor_speed * np.asarray(
                (math.cos(yaw), math.sin(yaw))
            )
        yaw_rate_limit = SOURCE_MAX_YAW_RATE
        if self.elapsed < scan_end and scan_end > 0.0:
            yaw_rate_limit = max(yaw_rate_limit, 2.0 * math.pi / scan_end)
        return command, yaw, yaw_rate_limit

    def _query_external_safety_points(
        self,
        drone_id: int,
        position: Sequence[float],
        velocity: Sequence[float],
        command: Sequence[float],
    ) -> np.ndarray:
        """Sweep the real vehicle envelope through nearby PhysX geometry.

        This is an execution-only collision sensor.  Its hit points are never
        published on ``/racer/sensor_points`` and therefore cannot change any
        frontier, HGrid, LKH, kinodynamic, B-spline, yaw, or FSM decision.
        """

        directions = [
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((-1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, -1.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.0, 0.0, -1.0)),
        ]
        for candidate in (velocity, command):
            value = np.asarray(candidate, dtype=float)
            norm = float(np.linalg.norm(value))
            if norm > 1.0e-4:
                directions.append(value / norm)

        unique_directions = []
        seen = set()
        for direction in directions:
            key = tuple(np.round(direction, decimals=4))
            if key not in seen:
                seen.add(key)
                unique_directions.append(direction)

        own_prefix = f"/World/Drones/drone_{drone_id}/"
        hits = []
        constraints = []
        for direction in unique_directions:
            direction_hits = []

            def report(hit):
                rigid_body = str(hit.rigid_body)
                collision = str(getattr(hit, "collision", ""))
                if not (
                    rigid_body.startswith(own_prefix)
                    or collision.startswith(own_prefix)
                ):
                    point = np.asarray(hit.position, dtype=float)
                    if np.all(np.isfinite(point)):
                        hits.append(point)
                    distance = float(getattr(hit, "distance", math.inf))
                    if math.isfinite(distance) and distance >= 0.0:
                        direction_hits.append(distance)
                return True

            self.scene_query.sweep_sphere_all(
                VEHICLE_RADIUS,
                Gf.Vec3f(*np.asarray(position, dtype=float)),
                Gf.Vec3f(*direction),
                SCENE_QUERY_RANGE,
                report,
            )
            if direction_hits:
                nearest_travel = min(direction_hits)
                constraints.append(
                    (np.asarray(direction, dtype=float), nearest_travel)
                )
                self.min_sweep_free_travel = min(
                    self.min_sweep_free_travel, nearest_travel
                )

        self.scene_query_updates += 1
        self.scene_query_hits += len(hits)
        self.scene_query_constraints[drone_id] = constraints
        if not hits:
            return np.empty((0, 3), dtype=float)
        points = np.asarray(hits, dtype=float).reshape((-1, 3))
        voxels = np.floor(points / SAFETY_POINT_VOXEL_SIZE).astype(np.int64)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        return points[np.sort(unique_indices)]

    def _execution_safety_points(self, drone_id: int) -> np.ndarray:
        return self.execution_safety_points[drone_id]

    def _refresh_execution_safety_points(self, drone_id: int) -> None:
        """Merge safety sources only when one source changes, not at 1 kHz."""

        sources = (
            self.safety_points[drone_id],
            self.scene_query_points[drone_id],
        )
        nonempty = [points for points in sources if len(points)]
        if not nonempty:
            self.execution_safety_points[drone_id] = np.empty(
                (0, 3), dtype=float
            )
        elif len(nonempty) == 1:
            self.execution_safety_points[drone_id] = nonempty[0]
        else:
            self.execution_safety_points[drone_id] = np.concatenate(
                nonempty, axis=0
            )

    def apply_motor_wrenches(self) -> None:
        self.control_steps += 1
        phase_checkpoint = (
            ARGS.diagnostics
            and (
                self.control_steps in (1, 10, 50, 250, 500, 1000, 1500)
                or self.control_steps % 500 == 0
            )
        )
        (
            self.pre_control_positions,
            self.pre_control_orientations,
            self.pre_control_velocities,
            self.pre_control_angular_velocities,
        ) = self._read_rigid_body_state_batch("pre_state_batch_read")
        states = [
            (
                self.pre_control_positions[drone_id],
                self.pre_control_orientations[drone_id],
                self.pre_control_velocities[drone_id],
            )
            for drone_id in range(self.drone_count)
        ]
        requested_commands = np.empty(
            (self.drone_count, 3), dtype=float
        )
        safety_points_batch = []
        sweep_constraints_batch = []
        control_jobs = []
        for drone_id in range(self.drone_count):
            if phase_checkpoint:
                print(
                    f"RACER_3D_PHASE step={self.control_steps} "
                    f"phase=control_drone_{drone_id}_begin",
                    flush=True,
                )
            requested_command = self.commands[drone_id].copy()
            yaw_rate_limit = SOURCE_MAX_YAW_RATE
            startup_override = self._startup_override(drone_id)
            if startup_override is not None:
                (
                    requested_command,
                    self.yaw_targets[drone_id],
                    yaw_rate_limit,
                ) = startup_override
            yaw_error = (
                self.yaw_targets[drone_id]
                - self.yaw_commands[drone_id]
                + math.pi
            ) % (2.0 * math.pi) - math.pi
            self.yaw_commands[drone_id] += float(
                np.clip(
                    yaw_error,
                    -yaw_rate_limit * PHYSICS_DT,
                    yaw_rate_limit * PHYSICS_DT,
                )
            )
            position, orientation, velocity = states[drone_id]
            if (
                ARGS.scene_usd is not None
                and self.elapsed - self.last_scene_query[drone_id]
                >= SCENE_QUERY_PERIOD - 1.0e-9
            ):
                self.scene_query_points[
                    drone_id
                ] = self._query_external_safety_points(
                    drone_id,
                    position,
                    velocity,
                    requested_command,
                )
                self._refresh_execution_safety_points(drone_id)
                self.last_scene_query[drone_id] = self.elapsed
            if phase_checkpoint:
                print(
                    f"RACER_3D_PHASE step={self.control_steps} "
                    f"phase=control_drone_{drone_id}_scene_query_done",
                    flush=True,
                )
            execution_safety_points = self._execution_safety_points(drone_id)
            requested_commands[drone_id] = requested_command
            safety_points_batch.append(execution_safety_points)
            sweep_constraints_batch.append(
                self.scene_query_constraints[drone_id]
            )
            if ARGS.vehicle_model != "racer_so3":
                peer_states = [
                    (peer_id, peer_position, peer_velocity)
                    for peer_id, (
                        peer_position,
                        _,
                        peer_velocity,
                    ) in enumerate(states)
                    if peer_id != drone_id
                ]
                control_jobs.append(
                    (
                        drone_id,
                        requested_command,
                        position,
                        orientation,
                        velocity,
                        self.pre_control_angular_velocities[drone_id],
                        self.yaw_commands[drone_id],
                        self.motor_rpms[drone_id].copy(),
                        execution_safety_points,
                        self.scene_query_constraints[drone_id],
                        peer_states,
                    )
                )

        safety_decision_batch_started = time.perf_counter()
        batch_result = None
        if ARGS.vehicle_model == "racer_so3":
            batch_result = solve_control_batch(
                requested_commands,
                self.pre_control_positions,
                self.pre_control_orientations,
                self.pre_control_velocities,
                self.pre_control_angular_velocities,
                self.motor_rpms,
                safety_points_batch,
                sweep_constraints_batch,
                self.yaw_commands,
                PHYSICS_DT,
                SOURCE_MAX_SPEED,
                OBSTACLE_CONTROL_CLEARANCE,
                SCENE_QUERY_CLEARANCE,
                SWARM_CONTROL_DISTANCE,
                ARGS.scene_usd is not None,
                SCENARIO.safety_min,
                SCENARIO.safety_max,
                self.control_obstacle_minimums,
                self.control_obstacle_maximums,
            )
            control_results = None
        elif self.sensor_executor is None:
            control_results = list(map(_solve_control_job, control_jobs))
        else:
            control_results = list(
                self.sensor_executor.map(_solve_control_job, control_jobs)
            )
        safety_decision_batch_ms = 1000.0 * (
            time.perf_counter() - safety_decision_batch_started
        )
        if batch_result is not None:
            worker_safety_ms = (
                np.asarray(batch_result["constraint_build_ms"])
                + np.asarray(batch_result["obstacle_projection_ms"])
                + np.asarray(batch_result["swarm_cbf_ms"])
            )
            worker_max_ms = float(
                np.max(worker_safety_ms, initial=0.0)
            )
            self.control_batch_profile_frames += 1
            self.control_batch_profile_sum_ms += safety_decision_batch_ms
            self.control_batch_profile_max_ms = max(
                self.control_batch_profile_max_ms,
                safety_decision_batch_ms,
            )
            self.control_batch_openmp_threads = int(
                batch_result["openmp_threads"]
            )
            for profile_name, result_name in (
                ("constraint_build", "constraint_build_ms"),
                ("obstacle_projection", "obstacle_projection_ms"),
                ("swarm_cbf", "swarm_cbf_ms"),
                ("so3", "so3_ms"),
            ):
                values = np.asarray(batch_result[result_name], dtype=float)
                self.control_batch_component_sum_ms[profile_name] += float(
                    np.sum(values)
                )
                self.control_batch_component_max_ms[profile_name] = max(
                    self.control_batch_component_max_ms[profile_name],
                    float(np.max(values, initial=0.0)),
                )
        else:
            worker_max_ms = max(
                (result[4] for result in control_results), default=0.0
            )
        self.safety_decision_profile_frames += 1
        self.safety_decision_profile_sum_ms += safety_decision_batch_ms
        self.safety_decision_profile_max_ms = max(
            self.safety_decision_profile_max_ms,
            safety_decision_batch_ms,
        )
        self.safety_decision_worker_max_sum_ms += worker_max_ms
        self.safety_decision_worker_max_max_ms = max(
            self.safety_decision_worker_max_max_ms,
            worker_max_ms,
        )
        self.last_safety_decision_ms = safety_decision_batch_ms
        if (
            ARGS.sensor_profiling
            and (self.control_steps == 1 or self.control_steps % 100 == 0)
        ):
            print(
                "RACER_3D_SAFETY_DECISION_PROFILE "
                + json.dumps(
                    {
                        "control_step": self.control_steps,
                        "simulation_time_s": self.elapsed,
                        "uav_count": self.drone_count,
                        "safety_decision_batch_ms": safety_decision_batch_ms,
                        "safety_decision_worker_max_ms": worker_max_ms,
                        "control_backend": (
                            "cpp_openmp_batch"
                            if batch_result is not None
                            else "python_per_uav"
                        ),
                        "control_components_mean_per_uav_ms": (
                            {
                                name: float(
                                    np.mean(batch_result[result_name])
                                )
                                for name, result_name in (
                                    (
                                        "constraint_build",
                                        "constraint_build_ms",
                                    ),
                                    (
                                        "obstacle_projection",
                                        "obstacle_projection_ms",
                                    ),
                                    ("swarm_cbf", "swarm_cbf_ms"),
                                    ("so3", "so3_ms"),
                                )
                            }
                            if batch_result is not None
                            else None
                        ),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
        if batch_result is not None:
            applied_batch = np.asarray(
                batch_result["applied_commands"], dtype=float
            )
            intervention_batch = np.asarray(
                batch_result["intervened"], dtype=bool
            )
            force_batch = np.asarray(
                batch_result["forces"], dtype=np.float32
            )
            torque_batch = np.asarray(
                batch_result["torques"], dtype=np.float32
            )
            self.motor_rpms[...] = batch_result["motor_rpm"]
            self.motor_thrusts[...] = batch_result["motor_thrust"]
        else:
            applied_batch = np.zeros(
                (self.drone_count, 3), dtype=float
            )
            intervention_batch = np.zeros(self.drone_count, dtype=bool)
            force_batch = np.zeros(
                (self.drone_count, 3), dtype=np.float32
            )
            torque_batch = np.zeros(
                (self.drone_count, 3), dtype=np.float32
            )
            for (
                drone_id,
                applied_command,
                wrench,
                intervened,
                _safety_decision_ms,
            ) in control_results:
                applied_batch[drone_id] = applied_command
                intervention_batch[drone_id] = intervened
                force_batch[drone_id] = wrench.local_force
                torque_batch[drone_id] = wrench.local_torque
                self.motor_thrusts[drone_id] = wrench.motor_thrusts

        for drone_id in range(self.drone_count):
            applied_command = applied_batch[drone_id]
            intervened = intervention_batch[drone_id]
            velocity = states[drone_id][2]
            if intervened:
                self.safety_interventions += 1
            self.applied_commands[drone_id] = applied_command
            if (
                ARGS.diagnostics
                and drone_id == 0
                and self.control_steps in (1, 10, 50, 250, 500, 1000, 1500)
            ):
                diag_position = self.pre_control_positions[drone_id]
                diag_orientation = self.pre_control_orientations[drone_id]
                print(
                    "RACER_3D_CONTROL "
                    + json.dumps(
                        {
                            "step": self.control_steps,
                            "mass": float(
                                self.rigid_body_masses[drone_id]
                            ),
                            "position": np.asarray(diag_position).tolist(),
                            "orientation_wxyz": np.asarray(
                                diag_orientation
                            ).tolist(),
                            "velocity": np.asarray(
                                velocity
                            ).tolist(),
                            "requested_command": self.commands[
                                drone_id
                            ].tolist(),
                            "applied_command": applied_command.tolist(),
                            "angular_velocity": self.pre_control_angular_velocities[
                                drone_id
                            ].tolist(),
                            "local_force": force_batch[
                                drone_id
                            ].tolist(),
                            "local_torque": torque_batch[
                                drone_id
                            ].tolist(),
                            "motors": self.motor_thrusts[
                                drone_id
                            ].tolist(),
                            "motor_rpm": (
                                self.motor_rpms[drone_id].tolist()
                                if ARGS.vehicle_model == "racer_so3"
                                else None
                            ),
                            "visual_propeller_angles_deg": (
                                self.propeller_visuals.angles_deg[
                                    drone_id
                                ].tolist()
                                if self.propeller_visuals.enabled
                                else None
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if phase_checkpoint:
                print(
                    f"RACER_3D_PHASE step={self.control_steps} "
                    f"phase=control_drone_{drone_id}_done",
                    flush=True,
                )
        # Match the old local-frame, one-substep force semantics while issuing
        # exactly one PhysX tensor write for all UAVs.
        force_write_started = time.perf_counter()
        self.rigid_body_view.apply_forces_and_torques_at_pos(
            forces=force_batch,
            torques=torque_batch,
            is_global=False,
        )
        self._record_rigid_io_profile(
            "force_torque_batch_write",
            1000.0 * (time.perf_counter() - force_write_started),
        )
        self.propeller_visuals.step(self.motor_rpms, PHYSICS_DT)
        if phase_checkpoint:
            print(
                f"RACER_3D_PHASE step={self.control_steps} "
                "phase=control_all_done",
                flush=True,
            )

    def _read_physics(
        self, count_distance: bool = True, batch_state=None
    ) -> None:
        if batch_state is None:
            batch_state = self._read_rigid_body_state_batch(
                "post_state_batch_read"
            )
        positions, orientations, velocities, angular_velocities = batch_state
        for drone_id, (position, velocity) in enumerate(
            zip(positions, velocities)
        ):
            if count_distance and self.previous_positions[drone_id] is not None:
                step = float(
                    np.linalg.norm(
                        position - self.previous_positions[drone_id]
                    )
                )
                if step < 0.20:
                    self.path_lengths[drone_id] += step
            self.previous_positions[drone_id] = position.copy()
            previous_velocity = self.previous_velocities[drone_id]
            self.accelerations[drone_id] = (
                np.zeros(3, dtype=float)
                if previous_velocity is None
                else (velocity - previous_velocity) / PHYSICS_DT
            )
            self.previous_velocities[drone_id] = velocity.copy()
        self.positions = positions
        self.orientations = orientations
        self.velocities = velocities
        self.angular_velocities = angular_velocities

    def _on_contact_report_event(self, contact_headers, _contact_data) -> None:
        """Maintain active collider pairs without per-step sensor polling."""

        started = time.perf_counter()
        for header in contact_headers:
            pair_key = tuple(
                sorted((int(header.collider0), int(header.collider1)))
            )
            found_or_persisting = header.type in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            )
            for collider_handle in (header.collider0, header.collider1):
                handle = int(collider_handle)
                if handle not in self.contact_sensor_by_collider_handle:
                    collider_path = str(
                        PhysicsSchemaTools.intToSdfPath(collider_handle)
                    )
                    self.contact_sensor_by_collider_handle[handle] = (
                        self.contact_sensor_by_collider_path.get(collider_path)
                    )
                sensor_key = self.contact_sensor_by_collider_handle[handle]
                if sensor_key is None:
                    continue
                drone_id, sensor_index = sensor_key
                pairs = self.contact_pairs_by_sensor[drone_id][sensor_index]
                if found_or_persisting:
                    pairs.add(pair_key)
                else:
                    pairs.discard(pair_key)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        profile = self.contact_profile["event_callback"]
        profile["calls"] += 1
        profile["headers"] += len(contact_headers)
        profile["total_ms"] += elapsed_ms
        profile["max_ms"] = max(profile["max_ms"], elapsed_ms)

    def _read_contact_fast_path(self):
        """Read bool/force only for colliders marked active by PhysX."""

        started = time.perf_counter()
        active = np.zeros(self.drone_count, dtype=bool)
        force = np.zeros(self.drone_count, dtype=float)
        calls = 0
        for drone_id, sensors in enumerate(self.contacts):
            for sensor_index, sensor in enumerate(sensors):
                if not self.contact_pairs_by_sensor[drone_id][sensor_index]:
                    continue
                # ContactSensor.get_current_frame() always fetches the raw
                # contact buffer in Isaac Sim 5.1. The underlying reading API
                # retrieves only is_valid/in_contact/value and preserves the
                # former scalar force definition.
                reading = (
                    sensor._contact_sensor_interface.get_sensor_reading(
                        sensor.prim_path
                    )
                )
                calls += 1
                if not reading.is_valid:
                    continue
                sensor_force = float(reading.value)
                force[drone_id] = max(force[drone_id], sensor_force)
                active[drone_id] = (
                    active[drone_id]
                    or (
                        bool(reading.in_contact)
                        and sensor_force > 1.0e-4
                    )
                )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        profile = self.contact_profile["fast_path"]
        profile["steps"] += 1
        profile["light_sensor_reading_calls"] += calls
        profile["total_ms"] += elapsed_ms
        profile["max_ms"] = max(profile["max_ms"], elapsed_ms)
        return active, force

    def _read_contact_legacy_reference(self):
        """Poll the former sensors only during explicit regression runs."""

        started = time.perf_counter()
        active = np.zeros(self.drone_count, dtype=bool)
        force = np.zeros(self.drone_count, dtype=float)
        calls = 0
        for drone_id, sensors in enumerate(self.contacts):
            frames = [sensor.get_current_frame() for sensor in sensors]
            calls += len(sensors)
            force[drone_id] = max(
                (float(frame.get("force", 0.0)) for frame in frames),
                default=0.0,
            )
            active[drone_id] = any(
                bool(frame.get("in_contact", False))
                and float(frame.get("force", 0.0)) > 1.0e-4
                for frame in frames
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        profile = self.contact_profile["legacy_reference"]
        profile["steps"] += 1
        profile["sensor_get_current_frame_calls"] += calls
        profile["total_ms"] += elapsed_ms
        profile["max_ms"] = max(profile["max_ms"], elapsed_ms)
        return active, force

    def _record_contact_event(self, drone_id: int, force: float) -> None:
        nearest_name = "external_usd"
        center_clearance = None
        body_clearance = None
        if ARGS.scene_usd is None:
            clearances = [
                point_box_signed_clearance(
                    self.positions[drone_id], obstacle
                )
                for obstacle in SCENARIO.obstacles
            ]
            nearest_index = int(np.argmin(clearances))
            nearest_name = SCENARIO.obstacles[nearest_index].name
            center_clearance = clearances[nearest_index]
            body_clearance = center_clearance - DRONE_RADIUS
        print(
            "RACER_3D_CONTACT "
            + json.dumps(
                {
                    "drone_id": drone_id,
                    "elapsed": self.elapsed,
                    "position": self.positions[drone_id].tolist(),
                    "velocity": self.velocities[drone_id].tolist(),
                    "command": self.commands[drone_id].tolist(),
                    "applied_command": self.applied_commands[
                        drone_id
                    ].tolist(),
                    "force": force,
                    "sweep_constraints": [
                        {
                            "direction": direction.tolist(),
                            "free_travel": distance,
                        }
                        for direction, distance
                        in self.scene_query_constraints[drone_id]
                    ],
                    "nearest_obstacle": nearest_name,
                    "center_clearance": center_clearance,
                    "body_clearance": body_clearance,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _update_contact_metrics(self) -> None:
        active, force = self._read_contact_fast_path()
        if ARGS.contact_regression:
            legacy_active, legacy_force = (
                self._read_contact_legacy_reference()
            )
            mismatch = active != legacy_active
            self.contact_regression["compared_steps"] += 1
            self.contact_regression["active_mismatch_steps"] += int(
                bool(np.any(mismatch))
            )
            self.contact_regression[
                "active_mismatch_uav_samples"
            ] += int(np.count_nonzero(mismatch))
            absolute_error = np.abs(force - legacy_force)
            self.contact_regression[
                "force_compared_uav_samples"
            ] += self.drone_count
            self.contact_regression[
                "force_absolute_error_sum_n"
            ] += float(np.sum(absolute_error))
            self.contact_regression[
                "force_max_absolute_error_n"
            ] = max(
                self.contact_regression[
                    "force_max_absolute_error_n"
                ],
                float(np.max(absolute_error, initial=0.0)),
            )
            for drone_id in range(self.drone_count):
                if (
                    legacy_active[drone_id]
                    and not self.legacy_contact_active[drone_id]
                ):
                    self.contact_regression[
                        "legacy_collision_events"
                    ] += 1
                self.legacy_contact_active[drone_id] = bool(
                    legacy_active[drone_id]
                )
            self.contact_regression[
                "legacy_max_contact_force_n"
            ] = max(
                self.contact_regression["legacy_max_contact_force_n"],
                float(np.max(legacy_force, initial=0.0)),
            )

        for drone_id in range(self.drone_count):
            is_active = bool(active[drone_id])
            contact_force = float(force[drone_id])
            # Preserve the existing per-UAV non-contact -> contact edge
            # definition exactly; persistent contact never increments again.
            if is_active and not self.contact_active[drone_id]:
                self.collision_events += 1
                self._record_contact_event(drone_id, contact_force)
            self.contact_active[drone_id] = is_active
            self.max_contact_force = max(
                self.max_contact_force, contact_force
            )

    def _update_metrics(self) -> None:
        distances = list(pairwise_distances(self.positions))
        if distances:
            self.min_inter_drone = min(
                self.min_inter_drone, min(distances)
            )
        if ARGS.scene_usd is None:
            for position in self.positions:
                self.min_obstacle_clearance = min(
                    self.min_obstacle_clearance,
                    obstacle_clearance(
                        position, SCENARIO.obstacles
                    )
                    - VEHICLE_RADIUS,
                )
        else:
            for drone_id, position in enumerate(self.positions):
                points = self._execution_safety_points(drone_id)
                if len(points):
                    self.min_obstacle_clearance = min(
                        self.min_obstacle_clearance,
                        float(
                            np.min(
                                np.linalg.norm(points - position, axis=1)
                            )
                        )
                        - VEHICLE_RADIUS,
                    )
        self._update_contact_metrics()

    def step_observations(self) -> None:
        self.elapsed += PHYSICS_DT
        seconds = int(math.floor(self.elapsed))
        nanoseconds = int(round((self.elapsed - seconds) * 1.0e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        clock = Clock()
        clock.clock.sec = seconds
        clock.clock.nanosec = nanoseconds
        self.clock_publisher.publish(clock)
        post_step_state = self._read_rigid_body_state_batch(
            "post_state_batch_read"
        )
        self._read_physics(batch_state=post_step_state)
        self._update_metrics()
        self.publish_shared_physical()
        # This bridge intentionally does not consume its own /clock topic.
        # Stamp every source message directly with the physics time just
        # published above so ROS2 agents using sim time see one time domain.
        stamp = clock.clock
        if self.elapsed - self.last_odom >= ODOM_PERIOD - 1.0e-9:
            self.last_odom = self.elapsed
            self._publish_odometry_and_imu(stamp)
        if self.elapsed - self.last_depth >= DEPTH_PERIOD - 1.0e-9:
            self.last_depth = self.elapsed
            self._publish_clouds(stamp)
        if (
            ARGS.sensor_profiling
            and (self.control_steps == 1 or self.control_steps % 100 == 0)
        ):
            print(
                "RACER_3D_RIGID_IO_PROFILE "
                + json.dumps(
                    {
                        "control_step": self.control_steps,
                        "simulation_time_s": self.elapsed,
                        **self.rigid_io_profile_report(),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            print(
                "RACER_3D_CONTACT_PROFILE "
                + json.dumps(
                    {
                        "control_step": self.control_steps,
                        "simulation_time_s": self.elapsed,
                        **self.contact_profile_report(),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

    def publish_shared_physical(self, *, force: bool = False) -> None:
        """Publish one compact simulator-owned state without planner work."""

        if self.shared_physical is None:
            return
        if (
            not force
            and self.elapsed - self.shared_physical_last_stamp + 1.0e-12
            < self.shared_physical_period_s
        ):
            return
        coverage = float(self.mapping_coverage_joint or 0.0)
        terminal = bool(
            not ARGS.fixed_exploration_horizon
            and (
                self.mission_complete
                or self.mapping_coverage_target_reached
            )
        )
        truncated = bool(
            self.elapsed >= ARGS.duration - 0.25 * PHYSICS_DT
        )
        payload = {
            "sim_step": int(self.control_steps),
            "sim_time_s": float(self.elapsed),
            "positions": [point.tolist() for point in self.positions],
            "velocities": [value.tolist() for value in self.velocities],
            "yaws": [
                _yaw_from_quaternion(value) for value in self.orientations
            ],
            "fsm_states": [
                "FINISH" if self.mission_complete else "EXPLORE"
                for _ in range(self.drone_count)
            ],
            "coverage": coverage,
            "coverage_delta": (
                coverage - self.shared_physical_last_coverage
            ),
            "path_lengths": list(self.path_lengths),
            "map_summary": {
                "representation": "per_uav_known_voxel_statistics",
                "coverage_per_agent": self.mapping_coverage,
                "coverage_counts_per_agent": self.mapping_coverage_counts,
            },
            "terminated": terminal,
            "truncated": truncated,
        }
        self.shared_physical_version = self.shared_physical.publish_json(
            payload,
            sim_step=self.control_steps,
            sim_time_s=self.elapsed,
        )
        # Fixed-size companion block used by the communication proxy at an
        # exact RL boundary.  It deliberately excludes map/trajectory data.
        if self.shared_physical_fast is not None:
            self.shared_physical_fast.publish_bytes(
                encode_physical_fast(
                    n_uavs=self.drone_count,
                    sim_step=self.control_steps,
                    sim_time_s=self.elapsed,
                    coverage=coverage,
                    coverage_delta=payload["coverage_delta"],
                    terminated=terminal,
                    truncated=truncated,
                ),
                sim_step=self.control_steps,
                sim_time_s=self.elapsed,
            )
        if self.shared_state_event is not None:
            self.shared_state_event.poke()
        self.shared_physical_last_stamp = self.elapsed
        self.shared_physical_last_coverage = coverage
        if force or self.shared_physical_version % 50 == 0:
            print(
                "ISAAC_SHM_STATE "
                + json.dumps(
                    {
                        "physical_version": self.shared_physical_version,
                        "sim_step": self.control_steps,
                        "simulation_time_s": self.elapsed,
                        "coverage": coverage,
                        "terminated": terminal,
                        "truncated": truncated,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
    def publish_initial_time_and_odometry(self) -> None:
        """Publish s_0 without consuming a physics or communication slot."""

        if self.elapsed != 0.0:
            raise RuntimeError("initial synchronous state must be published at t=0")
        clock = Clock()
        clock.clock.sec = 0
        clock.clock.nanosec = 0
        self.clock_publisher.publish(clock)
        initial_state = self._read_rigid_body_state_batch(
            "initial_sync_state_batch_read"
        )
        self._read_physics(batch_state=initial_state)
        self._publish_odometry_and_imu(clock.clock)
        self.last_odom = 0.0

    def publish_current_clock_keepalive(self) -> None:
        """Repeat the frozen /clock sample without advancing simulation.

        The clock topic is not transient-local.  Repeating the exact boundary
        timestamp while the synchronous gate waits closes the DDS delivery
        race in which Isaac has stopped at ``s_t`` but the communication proxy
        missed the one clock sample that would make it publish ``s_t``.
        """

        seconds = int(math.floor(self.elapsed))
        nanoseconds = int(round((self.elapsed - seconds) * 1.0e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        clock = Clock()
        clock.clock.sec = seconds
        clock.clock.nanosec = nanoseconds
        self.clock_publisher.publish(clock)

    def depth_render_due(self) -> bool:
        return self.elapsed - self.last_depth >= DEPTH_PERIOD - PHYSICS_DT

    def _publish_odometry_and_imu(self, stamp) -> None:
        for drone_id in range(self.drone_count):
            position = self.positions[drone_id]
            orientation = self.orientations[drone_id]
            velocity = self.velocities[drone_id]
            angular = self.angular_velocities[drone_id]
            message = Odometry()
            message.header.stamp = stamp
            message.header.frame_id = "map"
            message.child_frame_id = f"drone_{drone_id}/base_link"
            message.pose.pose.position.x = float(position[0])
            message.pose.pose.position.y = float(position[1])
            message.pose.pose.position.z = float(position[2])
            message.pose.pose.orientation.w = float(orientation[0])
            message.pose.pose.orientation.x = float(orientation[1])
            message.pose.pose.orientation.y = float(orientation[2])
            message.pose.pose.orientation.z = float(orientation[3])
            message.twist.twist.linear.x = float(velocity[0])
            message.twist.twist.linear.y = float(velocity[1])
            message.twist.twist.linear.z = float(velocity[2])
            message.twist.twist.angular.x = float(angular[0])
            message.twist.twist.angular.y = float(angular[1])
            message.twist.twist.angular.z = float(angular[2])
            self.odom_publishers[drone_id].publish(message)

            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = f"drone_{drone_id}/base_link"
            imu.orientation = message.pose.pose.orientation
            imu.angular_velocity = message.twist.twist.angular
            acceleration = self.accelerations[drone_id]
            imu.linear_acceleration.x = float(acceleration[0])
            imu.linear_acceleration.y = float(acceleration[1])
            imu.linear_acceleration.z = float(acceleration[2])
            self.imu_publishers[drone_id].publish(imu)

    @staticmethod
    def _legacy_lidar_world_points(
        raw_points: np.ndarray,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(raw_points, dtype=float).reshape((-1, 3))
        finite = np.all(np.isfinite(values), axis=1)
        values = values[finite]
        ranges = np.linalg.norm(values, axis=1)
        valid = (ranges > VEHICLE_RADIUS + 0.035) & (ranges <= 7.05)
        values, ranges = values[valid], ranges[valid]
        rotation = quaternion_matrix(orientation)
        sensor_origin = (
            np.asarray(position, dtype=float) + rotation @ LIDAR_TRANSLATION
        )
        world = values @ rotation.T + sensor_origin
        return world.astype(np.float32), ranges < 6.97

    @staticmethod
    def _depth_world_points(
        depth_image: np.ndarray,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project one upstream-equivalent depth frame into the map frame."""

        depth = np.asarray(depth_image, dtype=np.float32).squeeze()
        if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
        z_measured = depth[DEPTH_SAMPLE_ROWS, DEPTH_SAMPLE_COLS]
        u = DEPTH_SAMPLE_U
        v = DEPTH_SAMPLE_V
        finite_positive = np.isfinite(z_measured) & (z_measured > 0.0)
        keep = ~(finite_positive & (z_measured < DEPTH_MIN_RANGE))
        z_measured, u, v = z_measured[keep], u[keep], v[keep]
        hit = (
            np.isfinite(z_measured)
            & (z_measured > 0.0)
            & (z_measured <= DEPTH_MAP_RANGE)
        )
        z = np.where(hit, z_measured, DEPTH_MAP_RANGE).astype(np.float32)

        # Optical frame: +X right, +Y down, +Z forward.  Upstream cam02body
        # maps that to ROS FLU as [Z, -X, -Y] with zero translation.
        optical_x = (u - MAPPING_CX) * z / MAPPING_FX
        optical_y = (v - MAPPING_CY) * z / MAPPING_FY
        body_points = np.column_stack((z, -optical_x, -optical_y))
        if ARGS.vehicle_model == "racer_so3":
            not_self = (
                np.linalg.norm(body_points, axis=1) > SELF_FILTER_RADIUS
            )
            body_points = body_points[not_self]
            hit = hit[not_self]
        rotation = quaternion_matrix(orientation)
        world = (
            body_points @ rotation.T + np.asarray(position, dtype=float)
        )
        return world.astype(np.float32), hit

    @staticmethod
    def _transform_sensor_capture(capture):
        """Pure NumPy per-UAV work suitable for the sensor worker pool."""

        drone_id, position, orientation, raw, _raw_safety = capture
        if ARGS.vehicle_model == "racer_so3":
            mapping_points, mapping_hit = (
                IsaacRacer3DBridge._depth_world_points(
                    raw, position, orientation
                )
            )
        else:
            mapping_points, mapping_hit = (
                IsaacRacer3DBridge._legacy_lidar_world_points(
                    raw, position, orientation
                )
            )
        return (
            drone_id,
            np.asarray(position, dtype=float),
            tuple(np.asarray(raw).shape),
            mapping_points,
            mapping_hit,
            np.empty((0, 3), dtype=float),
        )

    def _publish_clouds(self, stamp) -> None:
        frame_started = time.perf_counter()
        raycasting_ms = 0.0
        safety_raycasting_ms = 0.0
        safety_pointcloud_process_ms = 0.0
        rtx_render_step_ms = self.pending_rtx_render_step_ms
        self.pending_rtx_render_step_ms = 0.0
        decode_backprojection_ms = 0.0
        # Reuse the single post-world-step tensor snapshot. Ray casting,
        # legacy sensors and PointCloud2 construction all observe the same
        # state as odometry, IMU, metrics and trajectory recording.
        vehicle_positions = self.positions
        vehicle_orientations = self.orientations
        safety_hits_by_drone = {}
        if ARGS.vehicle_model == "racer_so3":
            if self.warp_raycaster is None:
                raise RuntimeError(
                    "RACER SO3 execution safety requires the Warp ray caster"
                )
            (
                compact_safety_points,
                compact_safety_distances,
                safety_raycasting_ms,
                gpu_safety_process_ms,
            ) = self.warp_raycaster.cast_safety(
                vehicle_positions,
                vehicle_orientations,
                quaternion_matrix,
            )
            cpu_safety_process_started = time.perf_counter()
            for drone_id, (points, distances) in enumerate(
                zip(compact_safety_points, compact_safety_distances)
            ):
                # GPU range filtering and voxelization have already reduced
                # the scan. Retain the former conservative nearest-point cap
                # before handing the small current scan to the supervisor.
                if len(points) > SAFETY_LIDAR_POINT_LIMIT:
                    nearest = np.argpartition(
                        distances, SAFETY_LIDAR_POINT_LIMIT - 1
                    )[:SAFETY_LIDAR_POINT_LIMIT]
                    points = points[nearest]
                safety_hits_by_drone[drone_id] = np.asarray(
                    points, dtype=np.float32
                )
            safety_pointcloud_process_ms = (
                gpu_safety_process_ms
                + 1000.0
                * (time.perf_counter() - cpu_safety_process_started)
            )
        if (
            ARGS.vehicle_model == "racer_so3"
            and ARGS.depth_sensor_backend == "warp"
        ):
            (
                points_batch,
                hit_batch,
                _hit_distances,
                raycasting_ms,
            ) = self.warp_raycaster.cast(
                vehicle_positions,
                vehicle_orientations,
                quaternion_matrix,
            )
            transformed = []
            for drone_id, position in enumerate(vehicle_positions):
                transformed.append(
                    (
                        drone_id,
                        position,
                        (self.warp_raycaster.ray_count, 3),
                        points_batch[drone_id],
                        hit_batch[drone_id],
                        safety_hits_by_drone[drone_id],
                    )
                )
        else:
            captures = []
            for drone_id, range_sensor in enumerate(self.range_sensors):
                position = vehicle_positions[drone_id]
                orientation = vehicle_orientations[drone_id]
                if ARGS.vehicle_model == "racer_so3":
                    raw = range_sensor.get_depth()
                    if raw is None:
                        continue
                    raw = _backend_array_to_numpy(raw)
                else:
                    raw = range_sensor.get_current_frame().get("point_cloud")
                    if raw is None:
                        continue
                    raw = _backend_array_to_numpy(raw)
                captures.append(
                    (
                        drone_id,
                        np.asarray(position, dtype=float),
                        np.asarray(orientation, dtype=float),
                        raw,
                        None,
                    )
                )
            transform_started = time.perf_counter()
            if self.sensor_executor is None:
                transformed = list(map(self._transform_sensor_capture, captures))
            else:
                transformed = list(
                    self.sensor_executor.map(
                        self._transform_sensor_capture, captures
                    )
                )
            if ARGS.vehicle_model == "racer_so3":
                transformed = [
                    (*entry[:-1], safety_hits_by_drone[entry[0]])
                    for entry in transformed
                ]
            decode_backprojection_ms = 1000.0 * (
                time.perf_counter() - transform_started
            )

        pointcloud2_ms = 0.0
        publish_ms = 0.0
        published_clouds = 0
        hit_count = 0
        for (
            drone_id,
            position,
            raw_shape,
            mapping_points,
            mapping_hit,
            safety_hits,
        ) in transformed:
            if len(mapping_points) < 12:
                continue
            hit_count += int(np.count_nonzero(mapping_hit))
            camera_hits = np.asarray(mapping_points[mapping_hit], dtype=float)
            # The supervisor sees only the current low-density 360-degree
            # safety scan. Mapping-camera hits remain exclusive to RACER's
            # PointCloud2/SDFMap path and are never copied into safety state.
            self.safety_points[drone_id] = safety_hits
            self._refresh_execution_safety_points(drone_id)
            if self.debug_draw is not None and len(camera_hits):
                combined = np.concatenate(
                    (self.visual_map_points, camera_hits.astype(np.float32)),
                    axis=0,
                )
                voxels = np.floor(combined / 0.12).astype(np.int64)
                _, unique_indices = np.unique(
                    voxels, axis=0, return_index=True
                )
                combined = combined[np.sort(unique_indices)]
                if len(combined) > ARGS.visualization_max_map_points:
                    combined = combined[-ARGS.visualization_max_map_points:]
                self.visual_map_points = combined
                self.visualization_dirty = True
            if ARGS.diagnostics and not self.raw_diagnostics_printed:
                median_error = None
                if ARGS.scene_usd is None:
                    clearances = [
                        abs(
                            obstacle_clearance(
                                point, SCENARIO.obstacles
                            )
                        )
                        for point in mapping_points[
                            ::max(1, len(mapping_points) // 500)
                        ]
                    ]
                    median_error = float(np.median(clearances))
                print(
                    "RACER_3D_SENSOR_RAW "
                    + json.dumps(
                        {
                            "sensor": (
                                (
                                    "warp_raycaster_camera_plus_gpu_360_safety_rays"
                                    if ARGS.depth_sensor_backend == "warp"
                                    else "rtx_depth_plus_gpu_360_safety_rays"
                                )
                                if ARGS.vehicle_model == "racer_so3"
                                else "legacy_rotating_lidar"
                            ),
                            "raw_shape": list(raw_shape),
                            "world_count": len(mapping_points),
                            "camera_ray_budget": (
                                ARGS.camera_ray_budget
                                if ARGS.vehicle_model == "racer_so3"
                                else None
                            ),
                            "median_surface_error_m": median_error,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                self.raw_diagnostics_printed = True
            construct_started = time.perf_counter()
            message = create_xyzi_cloud(
                stamp, "world", mapping_points, mapping_hit
            )
            pointcloud2_ms += 1000.0 * (
                time.perf_counter() - construct_started
            )
            publish_started = time.perf_counter()
            self.cloud_publishers[drone_id].publish(message)
            publish_ms += 1000.0 * (
                time.perf_counter() - publish_started
            )
            self.cloud_frames += 1
            published_clouds += 1

        total_ms = (
            rtx_render_step_ms
            + 1000.0 * (time.perf_counter() - frame_started)
        )
        sensor_postprocess_ms = max(
            0.0,
            total_ms
            - rtx_render_step_ms
            - raycasting_ms
            - safety_raycasting_ms
            - safety_pointcloud_process_ms
            - decode_backprojection_ms
            - pointcloud2_ms
            - publish_ms,
        )
        self.sensor_profile_frames += 1
        profile_values = {
            "raycasting": raycasting_ms,
            "safety_raycasting": safety_raycasting_ms,
            "safety_pointcloud_process": safety_pointcloud_process_ms,
            "rtx_render_step": rtx_render_step_ms,
            "decode_backprojection": decode_backprojection_ms,
            "sensor_postprocess": sensor_postprocess_ms,
            "pointcloud2": pointcloud2_ms,
            "publish": publish_ms,
            "total": total_ms,
        }
        for name, value in profile_values.items():
            self.sensor_profile_sums_ms[name] += value
            self.sensor_profile_max_ms[name] = max(
                self.sensor_profile_max_ms[name], value
            )
        if ARGS.sensor_profiling:
            print(
                "RACER_3D_SENSOR_PROFILE "
                + json.dumps(
                    {
                        "backend": (
                            ARGS.depth_sensor_backend
                            if ARGS.vehicle_model == "racer_so3"
                            else "physx_lidar"
                        ),
                        "frame": self.sensor_profile_frames,
                        "simulation_time_s": self.elapsed,
                        "uav_clouds": published_clouds,
                        "rays_per_uav": (
                            self.warp_raycaster.ray_count
                            if self.warp_raycaster is not None
                            else ARGS.camera_ray_budget
                        ),
                        "hit_points": hit_count,
                        "raycasting_ms": raycasting_ms,
                        "safety_raycasting_ms": safety_raycasting_ms,
                        "safety_pointcloud_process_ms": (
                            safety_pointcloud_process_ms
                        ),
                        "safety_decision_last_batch_ms": (
                            self.last_safety_decision_ms
                        ),
                        "safety_points": int(
                            sum(len(points) for points in self.safety_points)
                        ),
                        "rtx_render_step_ms": rtx_render_step_ms,
                        "decode_backprojection_ms": (
                            decode_backprojection_ms
                        ),
                        "sensor_postprocess_ms": sensor_postprocess_ms,
                        "pointcloud2_construct_ms": pointcloud2_ms,
                        "publish_ms": publish_ms,
                        "sensor_total_ms": total_ms,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

    def shutdown_sensor_workers(self) -> None:
        if self.sensor_executor is not None:
            self.sensor_executor.shutdown(wait=True, cancel_futures=True)
            self.sensor_executor = None

    def publish_metrics(self) -> None:
        if (
            ARGS.record_trajectory_history
            and self.elapsed - self.last_trajectory_history_stamp >= 0.5
        ):
            self.trajectory_history.append(
                {
                    "time_s": self.elapsed,
                    "positions": [
                        position.tolist() for position in self.positions
                    ],
                }
            )
            self.last_trajectory_history_stamp = self.elapsed
        payload = {
            "backend": "isaac_sim_physx_3d",
            "scenario": SCENARIO.name,
            "scene_usd": (
                None if ARGS.scene_usd is None else str(ARGS.scene_usd)
            ),
            "vehicle_model": (
                "RACER SO3 0.98kg plus-quadrotor"
                if ARGS.vehicle_model == "racer_so3"
                else "Crazyflie 2.x 27g six-DOF rigid body"
            ),
            "vehicle_asset_usd": (
                str(ARGS.vehicle_usd)
                if ARGS.vehicle_model == "racer_so3"
                else str(CRAZYFLIE_ASSET)
            ),
            "motion_source": (
                "RACER kf/km mixer, first-order RPM, SO3 torque and drag"
                if ARGS.vehicle_model == "racer_so3"
                else "local rotor thrust and attitude torque"
            ),
            "sensor_source": (
                (
                    "NVIDIA Warp GPU-batched static-mesh pinhole ray caster "
                    "plus GPU-filtered 360-degree Warp safety rays"
                    if ARGS.depth_sensor_backend == "warp"
                    else "Isaac RTX ideal pinhole depth camera plus "
                    "GPU-filtered 360-degree Warp safety rays"
                )
                if ARGS.vehicle_model == "racer_so3"
                else "Isaac RotatingLidarPhysX point cloud"
            ),
            "sensor_parameters": (
                {
                    "width_px": DEPTH_WIDTH,
                    "height_px": DEPTH_HEIGHT,
                    "fx_px": DEPTH_FX,
                    "fy_px": DEPTH_FY,
                    "cx_px": DEPTH_CX,
                    "cy_px": DEPTH_CY,
                    "mapping_fx_px": MAPPING_FX,
                    "mapping_fy_px": MAPPING_FY,
                    "mapping_cx_px": MAPPING_CX,
                    "mapping_cy_px": MAPPING_CY,
                    "render_horizon_m": DEPTH_RENDER_HORIZON,
                    "map_min_range_m": DEPTH_MIN_RANGE,
                    "map_max_range_m": DEPTH_MAP_RANGE,
                    "mapping_min_ray_length_m": MAPPING_MIN_RAY_LENGTH,
                    "mapping_max_ray_length_m": MAPPING_MAX_RAY_LENGTH,
                    "rate_hz": 1.0 / DEPTH_PERIOD,
                    "skip_pixel": DEPTH_SKIP_PIXEL,
                    "point_cloud_ray_budget": ARGS.camera_ray_budget,
                    "depth_sensor_backend": ARGS.depth_sensor_backend,
                    "warp_raycaster": self.warp_raycaster_report,
                    "cpu_sensor_workers": self.sensor_worker_count,
                    "self_return_filter_radius_m": SELF_FILTER_RADIUS,
                    "mount_translation_body_m": CAMERA_TRANSLATION.tolist(),
                    "mount_orientation_body_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "optical_to_body_axes": "[z,-x,-y] -> [x,y,z]",
                    "safety_ray_horizontal_fov_deg": 360.0,
                    "safety_ray_vertical_fov_deg": 180.0,
                    "safety_ray_horizontal_resolution_deg": (
                        SAFETY_LIDAR_HORIZONTAL_RESOLUTION_DEG
                    ),
                    "safety_ray_vertical_resolution_deg": (
                        SAFETY_LIDAR_VERTICAL_RESOLUTION_DEG
                    ),
                    "safety_ray_min_range_m": SELF_FILTER_RADIUS,
                    "safety_ray_max_range_m": SAFETY_RAY_MAX_RANGE,
                    "safety_ray_voxel_size_m": SAFETY_POINT_VOXEL_SIZE,
                    "safety_ray_point_limit_per_uav": (
                        SAFETY_LIDAR_POINT_LIMIT
                    ),
                }
                if ARGS.vehicle_model == "racer_so3"
                else {
                    "type": "legacy_360x120_lidar",
                    "range_m": 7.0,
                }
            ),
            "clock_source": "Isaac physics /clock",
            "elapsed": self.elapsed,
            "collision_events": self.collision_events,
            "physics_contact_events": self.collision_events,
            "max_contact_force": self.max_contact_force,
            "contact_profile": self.contact_profile_report(),
            "min_inter_drone": (
                self.min_inter_drone
                if math.isfinite(self.min_inter_drone)
                else None
            ),
            "min_obstacle_clearance": (
                self.min_obstacle_clearance
                if math.isfinite(self.min_obstacle_clearance)
                else None
            ),
            "path_lengths": self.path_lengths,
            "start_positions": [list(point) for point in STARTS],
            "startup_recovery": {
                "enabled": ARGS.startup_free_space_yaw,
                "selected_yaws_rad": self.startup_yaws,
                "scan_duration_s": ARGS.startup_scan_duration,
                "unknown_corridor_distance_m": (
                    ARGS.startup_unknown_corridor_distance
                ),
                "corridor_clearances_m": self.startup_corridor_clearances,
                "corridor_enabled": self.startup_corridor_enabled,
                "settle_duration_s": ARGS.startup_settle_duration,
            },
            "positions": [point.tolist() for point in self.positions],
            "motor_thrusts_n": [
                values.tolist() for values in self.motor_thrusts
            ],
            "motor_rpm": [
                values.tolist() for values in self.motor_rpms
            ],
            "vehicle_radius_m": VEHICLE_RADIUS,
            "physics_rate_hz": 1.0 / PHYSICS_DT,
            "odometry_rate_hz": 1.0 / ODOM_PERIOD,
            "imu_rate_hz": 1.0 / ODOM_PERIOD,
            "point_cloud_frames": self.cloud_frames,
            "sensor_profile": {
                "frames": self.sensor_profile_frames,
                "mean_ms": {
                    name: (
                        total / self.sensor_profile_frames
                        if self.sensor_profile_frames
                        else 0.0
                    )
                    for name, total in self.sensor_profile_sums_ms.items()
                },
                "max_ms": self.sensor_profile_max_ms,
            },
            "safety_decision_profile": {
                "frames": self.safety_decision_profile_frames,
                "mean_batch_ms": (
                    self.safety_decision_profile_sum_ms
                    / self.safety_decision_profile_frames
                    if self.safety_decision_profile_frames
                    else 0.0
                ),
                "max_batch_ms": self.safety_decision_profile_max_ms,
                "mean_worker_max_ms": (
                    self.safety_decision_worker_max_sum_ms
                    / self.safety_decision_profile_frames
                    if self.safety_decision_profile_frames
                    else 0.0
                ),
                "max_worker_ms": self.safety_decision_worker_max_max_ms,
            },
            "control_batch_profile": self.control_batch_profile_report(),
            "rigid_body_io_profile": self.rigid_io_profile_report(),
            "safety_point_refresh_hz": 1.0 / DEPTH_PERIOD,
            "safety_interventions": self.safety_interventions,
            "scene_query_updates": self.scene_query_updates,
            "scene_query_hits": self.scene_query_hits,
            "scene_query_rate_hz": 1.0 / SCENE_QUERY_PERIOD,
            "scene_query_clearance_m": SCENE_QUERY_CLEARANCE,
            "min_sweep_free_travel": (
                self.min_sweep_free_travel
                if math.isfinite(self.min_sweep_free_travel)
                else None
            ),
            "low_level_safety": _low_level_safety_description(),
        }
        if ARGS.agentic_crpo_state_file is not None:
            coverage = float(self.mapping_coverage_joint or 0.0)
            mission_state = {
                "sim_time_s": self.elapsed,
                "positions": [point.tolist() for point in self.positions],
                "velocities": [value.tolist() for value in self.velocities],
                "yaws": [
                    _yaw_from_quaternion(value)
                    for value in self.orientations
                ],
                "fsm_states": [
                    "FINISH" if self.mission_complete else "EXPLORE"
                    for _ in range(self.drone_count)
                ],
                "coverage": coverage,
                "coverage_delta": coverage - self.agentic_crpo_last_coverage,
                "path_lengths": list(self.path_lengths),
                "map_summary": {
                    "representation": "per_uav_known_voxel_statistics",
                    "coverage_per_agent": self.mapping_coverage,
                    "coverage_counts_per_agent": self.mapping_coverage_counts,
                },
                "terminated": bool(
                    not ARGS.fixed_exploration_horizon
                    and (
                        self.mission_complete
                        or self.mapping_coverage_target_reached
                    )
                ),
                # Floating-point accumulation can place the exact final
                # physics tick a few ulps below ARGS.duration.  Use the same
                # tolerance as the synchronous boundary gate so the final
                # s_T transition is visibly terminal before Isaac enters its
                # permanent terminal barrier.
                "truncated": bool(
                    self.elapsed >= ARGS.duration - 0.25 * PHYSICS_DT
                ),
            }
            self.agentic_crpo_last_coverage = coverage
            state_path = ARGS.agentic_crpo_state_file.expanduser().resolve()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state_path.with_suffix(state_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(mission_state, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, state_path)
        self.metrics_publisher.publish(
            String(
                data=json.dumps(
                    payload, separators=(",", ":"), allow_nan=False
                )
            )
        )


def main() -> None:
    (
        world,
        bodies,
        rigid_body_view,
        range_sensors,
        safety_sensors,
        contacts,
    ) = build_world()
    if ARGS.visualize_exploration:
        start_center = np.mean(np.asarray(STARTS, dtype=float), axis=0)
        camera_target = np.asarray(
            (start_center[0], start_center[1] + 15.0, 3.0), dtype=float
        )
        camera_eye = np.asarray(
            (start_center[0], start_center[1] - 25.0, 25.0), dtype=float
        )
        set_camera_view(camera_eye, camera_target)
        simulation_app.update()
    warp_raycaster = None
    if ARGS.vehicle_model == "racer_so3":
        warp_raycaster = WarpRayCasterCameraBatch(
            omni.usd.get_context().get_stage(),
            camera_count=len(bodies),
            image_width=DEPTH_WIDTH,
            image_height=DEPTH_HEIGHT,
            sample_rows=DEPTH_SAMPLE_ROWS,
            sample_cols=DEPTH_SAMPLE_COLS,
            fx=DEPTH_FX,
            fy=DEPTH_FY,
            cx=DEPTH_CX,
            cy=DEPTH_CY,
            near_depth=DEPTH_MIN_RANGE,
            map_depth=DEPTH_MAP_RANGE,
            render_depth=DEPTH_RENDER_HORIZON,
            mount_translation=CAMERA_TRANSLATION,
            safety_horizontal_resolution_deg=(
                SAFETY_LIDAR_HORIZONTAL_RESOLUTION_DEG
            ),
            safety_vertical_resolution_deg=(
                SAFETY_LIDAR_VERTICAL_RESOLUTION_DEG
            ),
            safety_near_range=SELF_FILTER_RADIUS,
            safety_max_range=SAFETY_RAY_MAX_RANGE,
            safety_voxel_size=SAFETY_POINT_VOXEL_SIZE,
            safety_mount_translation=LIDAR_TRANSLATION,
        )
        print(
            "RACER_3D_WARP_RAYCASTER_READY "
            + json.dumps(warp_raycaster.report(), sort_keys=True),
            flush=True,
        )
    shared_blocks = None
    if ARGS.shared_memory_root:
        shared_layout = SharedMemoryLayout.from_value(
            ARGS.shared_memory_root
        )
        shared_blocks = open_blocks(
            shared_layout,
            ("physical", "physical_fast", "state_event", "status_isaac", "action", "shutdown"),
        )
        shared_blocks.update(open_rings(shared_layout, ("transition_ring",)))
        shared_blocks["status_isaac"].publish_json(
            {"ready": False, "process": "isaac", "pid": os.getpid()}
        )
    rclpy.init()
    bridge = IsaacRacer3DBridge(
        bodies,
        rigid_body_view,
        range_sensors,
        safety_sensors,
        contacts,
        warp_raycaster=warp_raycaster,
        shared_physical=(
            None if shared_blocks is None else shared_blocks["physical"]
        ),
        shared_physical_fast=(
            None if shared_blocks is None else shared_blocks["physical_fast"]
        ),
        shared_state_event=(
            None if shared_blocks is None else shared_blocks["state_event"]
        ),
        shared_status=(
            None if shared_blocks is None else shared_blocks["status_isaac"]
        ),
    )
    if (
        ARGS.vehicle_model == "racer_so3"
        and ARGS.depth_sensor_backend == "rtx"
    ):
        async def wait_for_depth_products() -> None:
            await asyncio.gather(
                *(
                    syntheticdata.sensors.next_render_simulation_async(
                        camera.get_render_product_path(), 10
                    )
                    for camera in range_sensors
                )
            )

        # RTX render products compile asynchronously. Do not begin experiment
        # time until every camera has delivered a correctly sized depth image.
        simulation_app.run_coroutine(wait_for_depth_products())
        for drone_id, camera in enumerate(range_sensors):
            depth = camera.get_depth()
            if depth is None:
                raise RuntimeError(
                    f"depth camera {drone_id} did not produce a frame"
                )
            depth = _backend_array_to_numpy(depth).squeeze()
            if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
                raise RuntimeError(
                    f"depth camera {drone_id} returned {depth.shape}, "
                    f"expected {(DEPTH_HEIGHT, DEPTH_WIDTH)}"
                )
            finite = depth[np.isfinite(depth) & (depth > 0.0)]
            print(
                "RACER_3D_DEPTH_READY "
                + json.dumps(
                    {
                        "drone_id": drone_id,
                        "shape": list(depth.shape),
                        "finite_pixels": int(len(finite)),
                        "minimum_depth_m": (
                            float(np.min(finite)) if len(finite) else None
                        ),
                        "maximum_depth_m": (
                            float(np.max(finite)) if len(finite) else None
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if ARGS.vehicle_model == "racer_so3":
        # Do not spend experiment time before DDS has connected both consumers
        # of every sensor stream: the unchanged exploration node and the
        # launch-readiness trigger.  This is strictly a ROS 2/Isaac transport
        # barrier; it publishes no planner input and changes no RACER decision.
        graph_deadline = time.monotonic() + 60.0
        last_graph_report = -math.inf
        while simulation_app.is_running():
            rclpy.spin_once(bridge, timeout_sec=0.05)
            odom_subscribers = [
                publisher.get_subscription_count()
                for publisher in bridge.odom_publishers
            ]
            cloud_subscribers = [
                publisher.get_subscription_count()
                for publisher in bridge.cloud_publishers
            ]
            graph_ready = all(
                count >= 2
                for count in odom_subscribers + cloud_subscribers
            )
            now_wall = time.monotonic()
            if graph_ready or now_wall - last_graph_report >= 1.0:
                graph_payload = {
                    "expected_subscribers_per_sensor": 2,
                    "metrics_subscribers": (
                        bridge.metrics_publisher.get_subscription_count()
                    ),
                    "odom_subscribers": odom_subscribers,
                    "point_cloud_subscribers": cloud_subscribers,
                    "ready": graph_ready,
                }
                print(
                    (
                        "RACER_3D_ROS_GRAPH "
                        if graph_ready
                        else "RACER_3D_ROS_GRAPH_WAIT "
                    )
                    + json.dumps(graph_payload, sort_keys=True),
                    flush=True,
                )
                last_graph_report = now_wall
            if graph_ready:
                break
            if now_wall >= graph_deadline:
                raise RuntimeError(
                    "ROS graph did not connect two odometry and point-cloud "
                    "subscribers for every vehicle within 60 seconds"
                )
    # Prime range sensors while each gravity-enabled vehicle is held by the
    # same motor/attitude controller used during the actual run.
    for _ in range(8):
        bridge.apply_motor_wrenches()
        world.step(render=False)
    # Sensor prims need warm-up physics, but that warm-up is not part of the
    # experiment. Restore the exact launch state after callbacks are active.
    reset_positions = np.asarray(STARTS, dtype=np.float32)
    reset_orientations = np.tile(
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        (len(bodies), 1),
    )
    rigid_body_view.set_world_poses(
        positions=reset_positions,
        orientations=reset_orientations,
    )
    rigid_body_view.set_velocities(
        np.zeros((len(bodies), 6), dtype=np.float32)
    )
    for drone_id in range(len(bodies)):
        bridge.previous_positions[drone_id] = np.asarray(
            STARTS[drone_id], dtype=float
        )
        bridge.previous_velocities[drone_id] = np.zeros(3, dtype=float)
        bridge.accelerations[drone_id] = np.zeros(3, dtype=float)
        bridge.contact_active[drone_id] = False
        if ARGS.vehicle_model == "racer_so3":
            bridge.motor_rpms[drone_id] = np.full(
                4, racer_hover_rpm(), dtype=float
            )
    bridge.propeller_visuals.reset()
    bridge.collision_events = 0
    bridge.max_contact_force = 0.0
    bridge.elapsed = 0.0
    bridge.last_odom = -math.inf
    bridge.last_depth = -math.inf
    bridge.cloud_frames = 0
    bridge.sensor_profile_frames = 0
    bridge.sensor_profile_sums_ms = {
        name: 0.0 for name in bridge.sensor_profile_sums_ms
    }
    bridge.sensor_profile_max_ms = {
        name: 0.0 for name in bridge.sensor_profile_max_ms
    }
    bridge.control_steps = 0
    bridge.safety_interventions = 0
    bridge.safety_decision_profile_frames = 0
    bridge.safety_decision_profile_sum_ms = 0.0
    bridge.safety_decision_profile_max_ms = 0.0
    bridge.safety_decision_worker_max_sum_ms = 0.0
    bridge.safety_decision_worker_max_max_ms = 0.0
    bridge.last_safety_decision_ms = 0.0
    bridge.control_batch_profile_frames = 0
    bridge.control_batch_profile_sum_ms = 0.0
    bridge.control_batch_profile_max_ms = 0.0
    bridge.control_batch_openmp_threads = 0
    bridge.control_batch_component_sum_ms = {
        name: 0.0 for name in bridge.control_batch_component_sum_ms
    }
    bridge.control_batch_component_max_ms = {
        name: 0.0 for name in bridge.control_batch_component_max_ms
    }
    bridge.reset_rigid_io_profile()
    bridge.reset_contact_profile()
    bridge.scene_query_updates = 0
    bridge.scene_query_hits = 0
    bridge.scene_query_points = [
        np.empty((0, 3), dtype=float) for _ in bodies
    ]
    bridge.execution_safety_points = [
        np.empty((0, 3), dtype=float) for _ in bodies
    ]
    bridge.last_scene_query = [-math.inf for _ in bodies]
    bridge.applied_commands = np.zeros((len(bodies), 3), dtype=float)
    bridge.positions = np.asarray(STARTS, dtype=float)
    bridge.orientations = reset_orientations.astype(float)
    bridge.velocities = np.zeros((len(bodies), 3), dtype=float)
    bridge.angular_velocities = np.zeros((len(bodies), 3), dtype=float)
    bridge.min_inter_drone = math.inf
    bridge.min_obstacle_clearance = math.inf
    bridge.path_lengths = [0.0 for _ in bodies]
    bridge.configure_startup_recovery()
    bridge.update_visualization(force=True)
    if ARGS.control_probe:
        bridge.commands[0] = np.asarray(
            ARGS.control_probe_command, dtype=float
        )
        bridge.yaw_targets[0] = 2.0
    offscreen_camera_textures = []
    if (
        ARGS.visualize_exploration
        and ARGS.vehicle_model == "racer_so3"
        and ARGS.depth_sensor_backend == "rtx"
    ):
        for camera in range_sensors:
            render_product = getattr(camera, "_render_product", None)
            hydra_texture = getattr(
                render_product, "hydra_texture", None
            )
            if hydra_texture is not None:
                offscreen_camera_textures.append(hydra_texture)
        if len(offscreen_camera_textures) != len(range_sensors):
            raise RuntimeError(
                "interactive offscreen-camera gating requires one Hydra "
                "texture per depth camera; found "
                f"{len(offscreen_camera_textures)} for "
                f"{len(range_sensors)} cameras"
            )
        print(
            "RACER_3D_INTERACTIVE_CAMERA_GATING "
            + json.dumps(
                {
                    "camera_count": len(offscreen_camera_textures),
                    "mode": "sensor_frames_only",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    offscreen_cameras_enabled = True
    frame = 0
    last_metrics = -math.inf
    main_wall_started = time.monotonic()
    main_wall_steps = 0
    main_wall_step_sum_ms = 0.0
    main_wall_step_max_ms = 0.0
    main_world_step_sum_ms = 0.0
    main_world_step_max_ms = 0.0
    single_gpu_pause_gate = SingleGpuPauseGate(
        ARGS.single_gpu_pause_request_file,
        ARGS.single_gpu_pause_ack_file,
        maximum_pause_s=ARGS.single_gpu_maximum_pause_s,
        poll_interval_s=ARGS.single_gpu_pause_poll_s,
    )
    synchronous_rl_gate = SynchronousRlBoundaryGate(
        ARGS.rl_sync_state_file,
        ARGS.rl_sync_action_file,
        ARGS.rl_sync_release_file,
        ARGS.rl_sync_ack_file,
        communication_slot_s=ARGS.rl_sync_communication_slot_s,
        decision_interval_s=ARGS.rl_sync_decision_interval_s,
        slots_per_decision=ARGS.rl_sync_slots_per_decision,
        maximum_wait_s=ARGS.rl_sync_maximum_wait_s,
        poll_interval_s=ARGS.rl_sync_poll_s,
        boundary_tolerance_s=max(1.0e-9, 0.25 * PHYSICS_DT),
    )
    if ARGS.shared_memory_root:
        # This is also checked before SimulationApp starts; keep a runtime
        # assertion next to the only two calls that can intentionally wait
        # without advancing world.step().
        if single_gpu_pause_gate.enabled or synchronous_rl_gate.enabled:
            raise RuntimeError(
                "shared-memory training unexpectedly enabled an Isaac freeze gate"
            )
        print(
            "ISAAC_TRAINING_SCHEDULING_MODE "
            + json.dumps(
                {
                    "shared_memory": True,
                    "single_gpu_pause_enabled": False,
                    "synchronous_rl_boundary_enabled": False,
                    "simulation_advances_during_llm_and_ppo": True,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        f"RACER_3D_ISAAC_READY drones={len(bodies)} "
        f"duration={ARGS.duration:.1f} vehicle={ARGS.vehicle_model} "
        f"physics_hz={1.0 / PHYSICS_DT:.0f} motion=rotor_wrench "
        f"sensor_workers={bridge.sensor_worker_count} "
        f"scene_query_hz={1.0 / SCENE_QUERY_PERIOD:.0f} "
        f"propeller_visuals={'on' if bridge.propeller_visuals.enabled else 'off'} "
        f"sensor={((('WarpRayCasterCamera' if ARGS.depth_sensor_backend == 'warp' else 'RTXDepthCamera') + '+GPU360SafetyRays') if ARGS.vehicle_model == 'racer_so3' else 'RotatingLidarPhysX')} "
        f"scene={ARGS.scene_usd or SCENARIO.name}",
        flush=True,
    )
    if bridge.shared_status is not None:
        bridge.shared_status.publish_json(
            {"ready": True, "process": "isaac", "pid": os.getpid()},
            sim_step=bridge.control_steps,
            sim_time_s=bridge.elapsed,
        )
        bridge.publish_shared_physical(force=True)
    if ARGS.visualize_exploration:
        print(
            "RACER_3D_VISUALIZATION_READY "
            "map=cyan trail/path=red,green,yellow,purple,orange "
            "camera=overview",
            flush=True,
        )
    try:
        if shared_blocks is not None:
            # One startup handshake at t=0, before the asynchronous run starts.
            # Repeat real initial odometry until the proxy commits s_0 and the
            # trainer has produced its first action. No physics slots are lost.
            startup_deadline = time.monotonic() + 180.0
            while True:
                stop = shared_blocks["shutdown"].snapshot_json()
                if stop is not None and stop[1].get("shutdown"):
                    raise RuntimeError("training shutdown during startup handshake")
                if (shared_blocks["transition_ring"].version > 0
                        and shared_blocks["action"].version > 0):
                    break
                if not simulation_app.is_running() or time.monotonic() >= startup_deadline:
                    raise RuntimeError("training startup timed out waiting for initial state/action")
                bridge.publish_initial_time_and_odometry()
                rclpy.spin_once(bridge, timeout_sec=0.02)
            print("RACER_TRAINING_STARTUP_READY sim_time=0 state_and_action=true", flush=True)
        last_render_wall = -math.inf
        render_report_wall = time.monotonic()
        render_report_frames = 0
        if synchronous_rl_gate.enabled:
            # State s_0 and the initial mission metrics are published before
            # any physics or communication interval is allowed to advance.
            bridge.publish_initial_time_and_odometry()
            bridge.publish_metrics()
            last_metrics = bridge.elapsed
            synchronous_rl_gate.wait_at_boundary(
                bridge.elapsed,
                simulation_app.is_running,
                terminal=False,
                state_keepalive=bridge.publish_initial_time_and_odometry,
            )
        while (
            simulation_app.is_running()
            and bridge.elapsed < ARGS.duration
            and (
                ARGS.fixed_exploration_horizon
                or not bridge.mission_complete
            )
            and (
                ARGS.fixed_exploration_horizon
                or not bridge.mapping_coverage_target_reached
            )
        ):
            step_started = time.monotonic()
            single_gpu_pause_gate.wait_if_requested(
                bridge.elapsed, simulation_app.is_running
            )
            rclpy.spin_once(bridge, timeout_sec=0.0)
            bridge.apply_motor_wrenches()
            phase_checkpoint = (
                ARGS.diagnostics
                and (
                    bridge.control_steps
                    in (1, 10, 50, 250, 500, 1000, 1500)
                    or bridge.control_steps % 500 == 0
                )
            )
            if phase_checkpoint:
                print(
                    f"RACER_3D_PHASE step={bridge.control_steps} "
                    "phase=before_world_step",
                    flush=True,
                )
            render_sensor = (
                ARGS.depth_sensor_backend == "rtx"
                and bridge.depth_render_due()
                if ARGS.vehicle_model == "racer_so3"
                else frame % max(1, ARGS.render_every) == 0
            )
            # The 1 kHz five-vehicle plant can run much slower than wall
            # time.  If rendering is gated only by the 30 Hz *simulation*
            # sensor clock, Kit may process mouse input just once every few
            # wall seconds.  Keep sensor timing simulation-based, but service
            # the interactive viewport independently in wall time.
            render_interactive = (
                ARGS.visualize_exploration
                and step_started - last_render_wall
                >= 1.0 / ARGS.interactive_render_hz
            )
            render_frame = render_sensor or render_interactive
            # Extra wall-clock renders exist only to service the visible
            # viewport. RTX comparison mode gates its offscreen products;
            # Warp mode has no Hydra camera product at all.
            want_offscreen_cameras = render_sensor
            if (
                offscreen_camera_textures
                and want_offscreen_cameras != offscreen_cameras_enabled
            ):
                for texture in offscreen_camera_textures:
                    texture.set_updates_enabled(want_offscreen_cameras)
                offscreen_cameras_enabled = want_offscreen_cameras
            world_step_started = time.perf_counter()
            world.step(render=render_frame)
            world_step_ms = 1000.0 * (
                time.perf_counter() - world_step_started
            )
            main_world_step_sum_ms += world_step_ms
            main_world_step_max_ms = max(
                main_world_step_max_ms, world_step_ms
            )
            if (
                ARGS.vehicle_model == "racer_so3"
                and ARGS.depth_sensor_backend == "rtx"
                and render_sensor
            ):
                # Isaac exposes RTX work through World.step(render=True).
                # This complete render-bearing step includes one physics tick,
                # hence the explicit render_step name in profiling output.
                bridge.pending_rtx_render_step_ms = world_step_ms
            if phase_checkpoint:
                print(
                    f"RACER_3D_PHASE step={bridge.control_steps} "
                    "phase=after_world_step",
                    flush=True,
                )
            if render_frame:
                # Schedule from frame start so rendering time itself counts
                # toward the wall-clock interval.  If a frame takes longer
                # than the target interval, the next loop services the UI
                # immediately instead of adding another full delay.
                last_render_wall = step_started
                if ARGS.visualize_exploration:
                    render_report_frames += 1
                    report_now = time.monotonic()
                    report_interval = report_now - render_report_wall
                    if report_interval >= 5.0:
                        print(
                            "RACER_3D_INTERACTIVE_RENDER "
                            + json.dumps(
                                {
                                    "measured_fps": (
                                        render_report_frames
                                        / report_interval
                                    ),
                                    "target_fps": (
                                        ARGS.interactive_render_hz
                                    ),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        render_report_wall = report_now
                        render_report_frames = 0
            bridge.step_observations()
            if synchronous_rl_gate.boundary_due(bridge.elapsed):
                # Write the mission-side task state at exactly the same 100 ms
                # boundary as the communication-side state.  World.step and
                # /clock do not run again until the trainer releases it.
                bridge.publish_metrics()
                last_metrics = bridge.elapsed
                synchronous_rl_gate.wait_at_boundary(
                    bridge.elapsed,
                    simulation_app.is_running,
                    terminal=(
                        bridge.elapsed
                        >= ARGS.duration - 0.25 * PHYSICS_DT
                    ),
                    state_keepalive=bridge.publish_current_clock_keepalive,
                )
            if phase_checkpoint:
                print(
                    f"RACER_3D_PHASE step={bridge.control_steps} "
                    "phase=after_observations",
                    flush=True,
                )
            bridge.update_visualization()
            if bridge.elapsed - last_metrics >= 0.5:
                bridge.publish_metrics()
                last_metrics = bridge.elapsed
            frame += 1
            main_step_ms = 1000.0 * (time.monotonic() - step_started)
            main_wall_steps += 1
            main_wall_step_sum_ms += main_step_ms
            main_wall_step_max_ms = max(main_wall_step_max_ms, main_step_ms)
            if ARGS.sensor_profiling and main_wall_steps % 500 == 0:
                main_wall_elapsed_s = time.monotonic() - main_wall_started
                print(
                    "RACER_3D_MAIN_WALL_PROFILE "
                    + json.dumps(
                        {
                            "steps": main_wall_steps,
                            "simulation_time_s": bridge.elapsed,
                            "wall_clock_s": main_wall_elapsed_s,
                            "real_time_factor": (
                                bridge.elapsed / main_wall_elapsed_s
                                if main_wall_elapsed_s > 0.0
                                else 0.0
                            ),
                            "mean_main_step_ms": (
                                main_wall_step_sum_ms / main_wall_steps
                            ),
                            "max_main_step_ms": main_wall_step_max_ms,
                            "mean_world_step_ms": (
                                main_world_step_sum_ms / main_wall_steps
                            ),
                            "max_world_step_ms": main_world_step_max_ms,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            remaining = PHYSICS_DT - (time.monotonic() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        bridge.commands = np.zeros_like(bridge.commands)
        bridge.publish_shared_physical(force=True)
        bridge.publish_metrics()
        # Allow DDS to deliver the final simulator timestamp to the acceptance
        # monitor before tearing down the ROS context.
        for _ in range(5):
            rclpy.spin_once(bridge, timeout_sec=0.02)
        print(
            "RACER_3D_ISAAC_RESULT "
            + json.dumps(
                {
                    "backend": "isaac_sim_physx_3d",
                    "vehicle_model": ARGS.vehicle_model,
                    "elapsed": bridge.elapsed,
                    "vehicle_asset_usd": (
                        str(ARGS.vehicle_usd)
                        if ARGS.vehicle_model == "racer_so3"
                        else str(CRAZYFLIE_ASSET)
                    ),
                    "collision_events": bridge.collision_events,
                    "physics_contact_events": bridge.collision_events,
                    "max_contact_force": bridge.max_contact_force,
                    "contact_profile": bridge.contact_profile_report(),
                    "min_inter_drone": (
                        bridge.min_inter_drone
                        if math.isfinite(bridge.min_inter_drone)
                        else None
                    ),
                    "min_obstacle_clearance": (
                        bridge.min_obstacle_clearance
                        if math.isfinite(bridge.min_obstacle_clearance)
                        else None
                    ),
                    "path_lengths": bridge.path_lengths,
                    "start_positions": [list(point) for point in STARTS],
                    "startup_recovery": {
                        "enabled": ARGS.startup_free_space_yaw,
                        "selected_yaws_rad": bridge.startup_yaws,
                        "scan_duration_s": ARGS.startup_scan_duration,
                        "unknown_corridor_distance_m": (
                            ARGS.startup_unknown_corridor_distance
                        ),
                        "corridor_clearances_m": (
                            bridge.startup_corridor_clearances
                        ),
                        "corridor_enabled": bridge.startup_corridor_enabled,
                        "settle_duration_s": ARGS.startup_settle_duration,
                    },
                    "positions": [
                        point.tolist() for point in bridge.positions
                    ],
                    "point_cloud_frames": bridge.cloud_frames,
                    "mapping_coverage_definition": (
                        "union of known FREE+OCCUPIED SDF voxels across all "
                        "UAV maps / configured planning-box voxels"
                    ),
                    "mapping_coverage_target": ARGS.mapping_coverage_target,
                    "mapping_coverage_target_reached": (
                        bridge.mapping_coverage_target_reached
                    ),
                    "mapping_coverage_per_agent": bridge.mapping_coverage,
                    "mapping_coverage_counts_per_agent": (
                        bridge.mapping_coverage_counts
                    ),
                    "mapping_coverage_history": (
                        bridge.mapping_coverage_history
                    ),
                    "mapping_coverage_joint_history": (
                        bridge.mapping_coverage_joint_history
                    ),
                    "mapping_coverage_joint_counts": {
                        "known_voxels": bridge.mapping_coverage_joint_known,
                        "total_voxels": (
                            bridge.mapping_coverage_joint_total
                        ),
                        "bitmap_sources_received": sum(
                            version > 0
                            for version in
                            bridge.mapping_coverage_bitmap_versions
                        ),
                    },
                    "trajectory_history": bridge.trajectory_history,
                    "mapping_coverage_joint": (
                        bridge.mapping_coverage_joint
                    ),
                    "stop_reason": (
                        "duration"
                        if ARGS.fixed_exploration_horizon
                        and bridge.elapsed >= ARGS.duration
                        else "mapping_coverage_target"
                        if bridge.mapping_coverage_target_reached
                        else "original_fsm_completion"
                        if bridge.mission_complete
                        else "duration"
                        if bridge.elapsed >= ARGS.duration
                        else "application_closed"
                    ),
                    "sensor_source": (
                        (
                            "warp_gpu_batched_camera_plus_gpu_360_safety_rays"
                            if ARGS.depth_sensor_backend == "warp"
                            else "rtx_depth_plus_gpu_360_safety_rays"
                        )
                        if ARGS.vehicle_model == "racer_so3"
                        else "legacy_rotating_lidar"
                    ),
                    "depth_sensor_backend": (
                        ARGS.depth_sensor_backend
                        if ARGS.vehicle_model == "racer_so3"
                        else None
                    ),
                    "warp_raycaster": bridge.warp_raycaster_report,
                    "sensor_profile": {
                        "frames": bridge.sensor_profile_frames,
                        "mean_ms": {
                            name: (
                                total / bridge.sensor_profile_frames
                                if bridge.sensor_profile_frames
                                else 0.0
                            )
                            for name, total
                            in bridge.sensor_profile_sums_ms.items()
                        },
                        "max_ms": bridge.sensor_profile_max_ms,
                    },
                    "safety_decision_profile": {
                        "frames": bridge.safety_decision_profile_frames,
                        "mean_batch_ms": (
                            bridge.safety_decision_profile_sum_ms
                            / bridge.safety_decision_profile_frames
                            if bridge.safety_decision_profile_frames
                            else 0.0
                        ),
                        "max_batch_ms": (
                            bridge.safety_decision_profile_max_ms
                        ),
                        "mean_worker_max_ms": (
                            bridge.safety_decision_worker_max_sum_ms
                            / bridge.safety_decision_profile_frames
                            if bridge.safety_decision_profile_frames
                            else 0.0
                        ),
                        "max_worker_ms": (
                            bridge.safety_decision_worker_max_max_ms
                        ),
                    },
                    "control_batch_profile": (
                        bridge.control_batch_profile_report()
                    ),
                    "rigid_body_io_profile": (
                        bridge.rigid_io_profile_report()
                    ),
                    "main_wall_clock_profile": {
                        "steps": main_wall_steps,
                        "wall_clock_s": (
                            time.monotonic() - main_wall_started
                        ),
                        "real_time_factor": (
                            bridge.elapsed
                            / max(
                                time.monotonic() - main_wall_started,
                                1.0e-9,
                            )
                        ),
                        "mean_main_step_ms": (
                            main_wall_step_sum_ms / main_wall_steps
                            if main_wall_steps
                            else 0.0
                        ),
                        "max_main_step_ms": main_wall_step_max_ms,
                        "mean_world_step_ms": (
                            main_world_step_sum_ms / main_wall_steps
                            if main_wall_steps
                            else 0.0
                        ),
                        "max_world_step_ms": main_world_step_max_ms,
                    },
                    "single_gpu_pause": single_gpu_pause_gate.report(),
                    "synchronous_rl": synchronous_rl_gate.report(),
                    "physics_rate_hz": 1.0 / PHYSICS_DT,
                    "odometry_rate_hz": 1.0 / ODOM_PERIOD,
                    "sensor_rate_hz": 1.0 / DEPTH_PERIOD,
                    "camera_ray_budget": ARGS.camera_ray_budget,
                    "cpu_sensor_workers": bridge.sensor_worker_count,
                    "safety_point_refresh_hz": 1.0 / DEPTH_PERIOD,
                    "safety_interventions": bridge.safety_interventions,
                    "scene_query_updates": bridge.scene_query_updates,
                    "scene_query_hits": bridge.scene_query_hits,
                    "scene_query_rate_hz": 1.0 / SCENE_QUERY_PERIOD,
                    "scene_query_clearance_m": SCENE_QUERY_CLEARANCE,
                    "min_sweep_free_travel": (
                        bridge.min_sweep_free_travel
                        if math.isfinite(bridge.min_sweep_free_travel)
                        else None
                    ),
                    "low_level_safety": _low_level_safety_description(),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        if bridge.shared_status is not None:
            bridge.shared_status.publish_json(
                {
                    "ready": False,
                    "stopping": True,
                    "process": "isaac",
                    "pid": os.getpid(),
                },
                sim_step=bridge.control_steps,
                sim_time_s=bridge.elapsed,
            )
        bridge.shutdown_sensor_workers()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if shared_blocks is not None:
            for block in shared_blocks.values():
                block.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
