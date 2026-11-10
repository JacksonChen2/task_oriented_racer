# RACER AP Recovery

`racer_recovery_core` is an independent fork of `racer_original_core`. It keeps
the original RACER frontier, HGrid, A*/kinodynamic trajectory, LKH and pairwise
`allocateGrids()` paths, then adds a bounded recovery state machine around the
original `PLAN_TRAJ` failure branch.

The source-faithful package remains available as `racer_original_core`; this
package does not replace its executables or launch file.

## Recovery sequence

1. A failed plan is retried with a configurable backoff instead of at 100 Hz.
2. A recovery episode starts only when the same goal has failed for both the
   configured duration and count, while measured odometry made insufficient
   progress.
3. The first intervention tries another live-map-validated viewpoint from the
   existing local RACER tour. If none is available, it rotates the first
   assigned HGrid task to select another local target. Neither path marks any
   voxel or frontier explored.
4. A second persistent failure plans a normal RACER trajectory toward a recent
   known-free, ESDF-cleared point on the executed path.
5. If local recovery fails, the UAV sends `RecoveryStatus` over its simulated
   UAV-to-AP link. The AP chooses the nearest fresh, healthy partner and sends a
   `RecoveryCommand` over the simulated AP-to-UAV link.
6. The failed UAV runs the original pairwise `allocateGrids()` path with that
   partner. The failed grid is leased away from the stuck UAV for the recovery
   transaction; it is not deleted and can be explored by the partner.

Recovery commands carry an episode and assignment epoch so delayed commands
cannot overwrite newer assignments. Missing AP or pair responses time out into
another bounded local attempt instead of leaving the FSM in an infinite wait.

## Build and test

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to racer_sionna_comm --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select racer_recovery_core racer_sionna_comm
colcon test-result --verbose
```

The tests cover AP partner selection and bidirectional transport of recovery
status/commands through the Sionna communication proxy.

## Warehouse Loaded run

From the workspace root:

```bash
./run_warehouse_loaded_bs_recovery_900s.sh
```

For the same recovery algorithm without the AP coordinator:

```bash
./run_warehouse_loaded_distributed_recovery_900s.sh
```

This uses the same `warehouse_loaded_with_industrial_ap.usda`, Sionna RT link
model, five-UAV Crazyflie dynamics, 150 Hz physics, 30 Hz sensors and 19,200 ray
budget as the latest paired run. Override the usual `RACER_*` environment
variables when another fidelity setting is required.

The recovery launch itself is:

```bash
ros2 launch racer_recovery_core recovery_racer_warehouse_sionna.launch.py \
  scenario:=warehouse_loaded network_topology:=bs_round_robin drone_count:=5
```

The direct launch expects the Isaac plant, `/clock` and Sionna channel inputs to
be started separately; the wrapper above starts the complete experiment.

## Fidelity boundary

Normal successful planning follows the copied original RACER code. Once a
recovery episode intervenes, the run must be reported as `RACER + Recovery`
rather than unmodified RACER. For a communication-only A/B test, use the same
local recovery policy in both arms. For an end-to-end architecture comparison,
compare distributed local recovery against local recovery plus AP repartition.
