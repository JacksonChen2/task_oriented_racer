# RACER SO3 quadrotor asset for Isaac Sim

This directory turns the mathematical SO3 quadrotor plant shipped with RACER
into a portable, importable Isaac Sim asset.

The URDF preserves the upstream mass, inertia, rotor locations, propeller
radius, and body-frame convention. The accompanying model configuration and
Python helper preserve its thrust/moment mixer, RPM limits, first-order motor
response, and quadratic translational drag.

## What is exact and what is inferred

Source-faithful values:

- mass: `0.98 kg`;
- inertia: `diag(2.64e-3, 2.64e-3, 4.96e-3) kg m^2`;
- plus-layout rotor centers: `(+X, -X, +Y, -Y) * 0.26 m`;
- propeller radius: `0.062 m`;
- thrust coefficient: `8.98132e-9 N/RPM^2`;
- yaw moment coefficient: `1.169367864e-10 Nm/RPM^2`;
- motor time constant: `1/30 s`;
- RPM range: `1200..35000`;
- world-frame quadratic drag used by `Quadrotor.cpp`;
- zero-translation, forward-looking ROS optical camera transform.

The ROS1 renderer and mapper use slightly different checked-in intrinsics.
The renderer loads `camera.yaml` (`fx=fy=387.229`, `cx=321.046`,
`cy=243.449`); `single_drone_exploration.xml` defaults the mapping projection
to `fx=fy=385.69794`, `cx=324.08798`, `cy=239.10362`. Both are recorded in
the model configuration so a source-profile integration can preserve this
upstream mismatch. The corresponding simulation ranges are a 5.0 m render
horizon, 0.2--4.6 m depth filter and 0.5--4.5 m occupancy-ray interval.

Inferred values:

- central body, arm cross-section, and motor-can dimensions;
- all collision geometry;
- colors and visual appearance.

The original RACER simulator contains no CAD, URDF, motor mesh, contact model,
or physical camera housing. It also clamps altitude at `z=0` instead of
performing 3-D collision detection. Consequently this is a dynamics-faithful
primitive model, not an exact reconstruction of the authors' custom aircraft.

## Files

- `urdf/racer_so3_quadrotor.urdf`: one floating rigid body with exact mass and
  inertia, primitive visuals, and conservative collision geometry.
- `urdf/racer_so3_quadrotor_crazyflie_style.urdf`: a second, Crazyflie-inspired
  38-primitive visual variant. It retains the same single rigid link, inertia,
  source `+` rotor locations and seven collision primitives.
- `config/racer_so3_model.json`: authoritative equations, frames, parameters,
  provenance, inferred geometry, and derived hover values.
- `config/racer_so3_crazyflie_style.json`: visual-variant manifest and invariant
  list; dynamics continue to come from `racer_so3_model.json`.
- `usd/crazyflie_with_racer_dynamics.usd`: the actual embedded Crazyflie mesh,
  transformed into the RACER plus layout and attached to the unchanged RACER
  physical body.
- `config/crazyflie_with_racer_dynamics.json`: mesh conversion, provenance and
  physics-invariant manifest.
- `config/racer_so3_ros2.yaml`: convenient ROS 2 parameter projection.
- `scripts/racer_so3_dynamics.py`: dependency-free upstream motor/mixer/drag
  equations.
- `scripts/import_urdf_to_usd.py`: Isaac Sim 5.1 URDF-to-USD importer.
- `scripts/isaac_hover_demo.py`: minimal PhysX force application check.
- `scripts/validate_model.py` and `tests/test_dynamics.py`: consistency tests.

## Validate without Isaac Sim

From this directory:

```bash
python3 scripts/validate_model.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

## Import through the Isaac Sim GUI

1. Start Isaac Sim.
2. Open **File > Import** and select
   `urdf/racer_so3_quadrotor.urdf`.
3. Set:
   - **Fix Base Link**: off;
   - **Import Inertia Tensor**: on;
   - **Collision From Visuals**: off;
   - **Distance Scale**: `1.0`;
   - **Merge Fixed Joints**: on.
4. Save the imported stage as a USD asset.

The URDF is intentionally mesh-free, so it has no asset-path or texture
dependencies.

The Crazyflie-style variant follows the reference aircraft's layered PCB,
underslung battery, motor-basket, electronics and thin two-blade-propeller
appearance. It does not copy the reference mesh and it does not change RACER's
source-faithful `+` rotor layout into Crazyflie's `X` layout.

The separate `crazyflie_with_racer_dynamics.usd` variant does embed the actual
Crazyflie example mesh installed with Isaac Sim. The mesh is rendering-only:
its Y-up X layout is scaled and rotated so the visible propeller axes land on
the four RACER plus-layout rotor centers. Its propellers are independently
resized to the RACER `0.062 m` radius. The rigid body, seven collisions, rotor
frames and all `racer:*` dynamics attributes are copied unchanged from the
validated RACER flattened asset.

Each embedded propeller parent also has `racer:rotorId`,
`racer:spinDirectionSign`, `racer:visualRotationAxis` and
`racer:visualRotationOp` metadata. The ROS2/Isaac bridge uses these attributes
to discover all four blades and integrate their display angle from the actual
motor RPM. The blade Xforms have no Physics API: animation is visual-only and
cannot change the validated rigid-body or collision behavior.

## Reproducible command-line import

Run with the Python launcher belonging to the installed Isaac Sim:

```bash
/home/jackson/isaacsim/python.sh scripts/import_urdf_to_usd.py
```

The default output is `usd/racer_so3_quadrotor.usd`, accompanied by
`usd/racer_so3_quadrotor.import_report.json`. Isaac Sim 5.1 also creates
relative composition layers under `usd/configuration`; move or distribute the
whole `usd` directory rather than the top-level USD alone.

For referencing multiple vehicles below paths such as `/World/Drones/drone_0`,
use `usd/racer_so3_quadrotor_flattened.usd`. The flattened asset avoids the
absolute-path composition opinions generated by the URDF importer.

Generate the Crazyflie-style variant reproducibly with:

```bash
/home/jackson/isaacsim/python.sh scripts/import_urdf_to_usd.py \
  --urdf urdf/racer_so3_quadrotor_crazyflie_style.urdf \
  --output usd/racer_so3_quadrotor_crazyflie_style.usd \
  --headless
```

Its portable multi-instance asset is
`usd/racer_so3_quadrotor_crazyflie_style_flattened.usd`. Select it in the C++
RACER runner without changing controller parameters:

```bash
RACER_SO3_VEHICLE_USD="$PWD/usd/racer_so3_quadrotor_crazyflie_style_flattened.usd" \
  /path/to/ros2_3d_C_ws/src/racer_3d_cpp/scripts/run_isaac_test.sh
```

Generate the embedded-mesh variant with the lightweight USD Python bundled in
Isaac Sim:

```bash
USD_LIB=/home/jackson/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_LIB" \
LD_LIBRARY_PATH="$USD_LIB/bin:/home/jackson/isaacsim/kit/python/lib:/home/jackson/isaacsim/kit" \
/home/jackson/isaacsim/kit/python/bin/python3 \
  scripts/build_crazyflie_racer_dynamics_usd.py
```

The generated USD embeds all twelve source meshes, so it has no runtime
reference to the source Crazyflie USD. Select it in the C++ runner with:

```bash
RACER_SO3_VEHICLE_USD="$PWD/usd/crazyflie_with_racer_dynamics.usd" \
RACER_3D_ANIMATE_PROPELLERS=1 \
  /path/to/ros2_3d_C_ws/src/racer_3d_cpp/scripts/run_isaac_test.sh
```

Interactive runs enable the propeller animator automatically. Headless runs
disable it by default; force it on only for diagnostics with
`RACER_3D_ANIMATE_PROPELLERS=1`. The default visual transform rate is 60 Hz and
can be changed with `RACER_3D_PROPELLER_VISUAL_HZ`.

Validate both variants without starting Isaac:

```bash
python3 scripts/validate_model.py
python3 scripts/validate_model.py \
  --urdf urdf/racer_so3_quadrotor_crazyflie_style.urdf \
  --expected-visuals 38
```

To exercise gravity, imported inertia, and source-equivalent hover thrust:

```bash
/home/jackson/isaacsim/python.sh scripts/isaac_hover_demo.py \
  --usd usd/racer_so3_quadrotor.usd --duration 3
```

## Important runtime requirement

URDF describes rigid-body geometry and inertia; it cannot express

```text
thrust = kf * RPM^2
motor_dot = (command_RPM - RPM) / tau
yaw_torque = +/- km * RPM^2
```

Therefore importing the URDF alone produces a passive falling rigid body.
Attach `racer_so3_dynamics.py` to an Isaac physics callback, or reproduce those
equations in the ROS 2/Isaac bridge. `isaac_hover_demo.py` is the minimal
reference implementation.

For RACER integration, keep the original command boundary:

```text
PositionCommand(position, velocity, acceleration, yaw, yaw_rate)
  -> SO3 position/attitude controller
  -> desired total thrust and body moment
  -> RACER mixer and first-order motor states
  -> local PhysX force and torque on base_link
```

Do not replace it with a velocity-only `Twist` interface if source-level
trajectory tracking fidelity is required.
