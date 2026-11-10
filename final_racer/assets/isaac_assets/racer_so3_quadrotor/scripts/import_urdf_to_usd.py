#!/usr/bin/env python3
"""Import the RACER SO3 URDF into Isaac Sim 5.1 and save a portable USD."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "urdf" / "racer_so3_quadrotor.urdf"
DEFAULT_MODEL = ROOT / "config" / "racer_so3_model.json"
DEFAULT_USD = ROOT / "usd" / "racer_so3_quadrotor.usd"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_USD)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


ARGS = parse_arguments()
APP = SimulationApp({"headless": ARGS.headless})

import omni.kit.commands
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


def find_rigid_body(stage: Usd.Stage, root_path: str) -> Usd.Prim:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"imported root prim does not exist: {root_path}")
    candidates = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI)
    ]
    if len(candidates) != 1:
        paths = [str(prim.GetPath()) for prim in candidates]
        raise RuntimeError(f"expected one rigid body below {root_path}, found {paths}")
    return candidates[0]


def define_frame(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    name: str,
    translation: list[float],
    quaternion_wxyz: list[float],
) -> Usd.Prim:
    frame = UsdGeom.Xform.Define(stage, parent_path.AppendChild(name))
    frame.AddTranslateOp().Set(Gf.Vec3d(*translation))
    frame.AddOrientOp().Set(
        Gf.Quatf(
            quaternion_wxyz[0],
            Gf.Vec3f(
                quaternion_wxyz[1],
                quaternion_wxyz[2],
                quaternion_wxyz[3],
            ),
        )
    )
    return frame.GetPrim()


def set_scalar(prim: Usd.Prim, name: str, value: float) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Double, custom=True).Set(float(value))


def normalize_imported_cube_extents(stage: Usd.Stage) -> int:
    """Repair double-scaled bounds authored by the Isaac 5.1 URDF importer.

    Imported URDF boxes use a parent Xform scale for their physical size. The
    importer also authors a child Cube extent containing that same size, so
    bounds consumers apply it twice even though rendering and PhysX correctly
    use ``Cube.size * parent scale``. A canonical unit extent preserves the
    rendered/collision geometry and restores correct USD bounds.
    """

    repaired = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cube):
            continue
        cube = UsdGeom.Cube(prim)
        half = 0.5 * float(cube.GetSizeAttr().Get() or 1.0)
        expected = [Gf.Vec3f(-half, -half, -half), Gf.Vec3f(half, half, half)]
        extent = cube.GetExtentAttr()
        if extent.Get() != expected:
            extent.Set(expected)
            repaired += 1
    return repaired


def author_racer_metadata(
    stage: Usd.Stage, rigid_body: Usd.Prim, model: dict
) -> None:
    body_path = rigid_body.GetPath()
    parameters_path = body_path.AppendChild("racer_frames")
    UsdGeom.Scope.Define(stage, parameters_path)

    identity = [1.0, 0.0, 0.0, 0.0]
    for rotor in model["rotors"]:
        define_frame(
            stage,
            parameters_path,
            f"rotor_{rotor['id']}",
            rotor["position_body_m"],
            identity,
        )
    define_frame(
        stage,
        parameters_path,
        "camera_optical_frame",
        model["frames"]["camera_optical"]["translation_body_m"],
        model["frames"]["camera_optical"]["quaternion_body_from_camera_wxyz"],
    )
    define_frame(
        stage,
        parameters_path,
        "imu_link",
        model["frames"]["imu"]["translation_body_m"],
        model["frames"]["imu"]["quaternion_body_from_imu_wxyz"],
    )

    rigid = model["rigid_body"]
    propulsion = model["propulsion"]
    aerodynamics = model["aerodynamics"]
    set_scalar(rigid_body, "racer:massKg", rigid["mass_kg"])
    set_scalar(rigid_body, "racer:gravityMps2", rigid["gravity_m_s2"])
    set_scalar(rigid_body, "racer:armLengthM", propulsion["arm_length_m"])
    set_scalar(
        rigid_body,
        "racer:propellerRadiusM",
        propulsion["propeller_radius_m"],
    )
    set_scalar(
        rigid_body,
        "racer:thrustCoefficientNPerRpm2",
        propulsion["thrust_coefficient_N_per_rpm2"],
    )
    set_scalar(
        rigid_body,
        "racer:momentCoefficientNmPerRpm2",
        propulsion["moment_coefficient_Nm_per_rpm2"],
    )
    set_scalar(
        rigid_body,
        "racer:motorTimeConstantS",
        propulsion["motor_time_constant_s"],
    )
    set_scalar(rigid_body, "racer:minimumRpm", propulsion["minimum_rpm"])
    set_scalar(rigid_body, "racer:maximumRpm", propulsion["maximum_rpm"])
    set_scalar(
        rigid_body,
        "racer:quadraticDragCoefficient",
        aerodynamics["quadratic_drag_coefficient"],
    )

    # The source uses its own quadratic translational drag and no angular drag.
    # Zero PhysX damping prevents the runtime adapter from double-counting it.
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_body)
    physx_body.CreateLinearDampingAttr(0.0)
    physx_body.CreateAngularDampingAttr(0.0)
    physx_body.CreateDisableGravityAttr(False)


def main() -> dict:
    urdf_path = ARGS.urdf.resolve()
    model_path = ARGS.model.resolve()
    output_path = ARGS.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("r", encoding="utf-8") as stream:
        model = json.load(stream)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Isaac URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = True
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.distance_scale = 1.0
    import_config.collision_from_visuals = False

    status, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=str(output_path),
        get_articulation_root=False,
    )
    if not status:
        raise RuntimeError("Isaac URDFParseAndImportFile failed")

    # With dest_path set, Isaac writes a new USD stage instead of replacing the
    # currently open application stage. Reopen that asset before authoring RACER
    # frames and metadata.
    APP.update()
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"could not open the generated USD stage: {output_path}")
    rigid_body = find_rigid_body(stage, imported_path)
    body_path = rigid_body.GetPath()
    author_racer_metadata(stage, rigid_body, model)

    stage.SetDefaultPrim(stage.GetPrimAtPath(imported_path))
    stage.GetRootLayer().Save()

    # Isaac's generated geometry lives at sibling roots (/visuals and
    # /colliders), reached through absolute-path references. Referencing only
    # the robot default prim therefore loses that geometry. Start from the
    # composed physics layer, copy its concrete geometry below base_link, and
    # remove the importer-only sibling roots.
    flattened_path = output_path.with_name(
        output_path.stem + "_flattened.usd"
    )
    physics_path = (
        output_path.parent
        / "configuration"
        / f"{output_path.stem}_physics.usd"
    )
    physics_stage = Usd.Stage.Open(str(physics_path))
    if physics_stage is None:
        raise RuntimeError(f"could not open importer physics layer: {physics_path}")
    normalized_cube_extents = normalize_imported_cube_extents(physics_stage)
    physics_stage.GetRootLayer().Save()
    portable_layer = physics_stage.Flatten()
    portable_stage = Usd.Stage.Open(portable_layer)
    portable_body_path = Sdf.Path(imported_path).AppendChild("base_link")
    for source, destination in (
        (
            Sdf.Path("/visuals/base_link"),
            portable_body_path.AppendChild("visuals"),
        ),
        (
            Sdf.Path("/colliders/base_link"),
            portable_body_path.AppendChild("collisions"),
        ),
    ):
        portable_stage.RemovePrim(destination)
        if not Sdf.CopySpec(
            portable_layer, source, portable_layer, destination
        ):
            raise RuntimeError(f"could not copy {source} to {destination}")
    for importer_root in ("/visuals", "/colliders", "/meshes"):
        portable_stage.RemovePrim(Sdf.Path(importer_root))
    portable_rigid_body = find_rigid_body(portable_stage, imported_path)
    author_racer_metadata(portable_stage, portable_rigid_body, model)
    portable_stage.SetDefaultPrim(
        portable_stage.GetPrimAtPath(imported_path)
    )
    if not portable_layer.Export(str(flattened_path)):
        raise RuntimeError(f"could not export flattened USD: {flattened_path}")
    APP.update()

    mass_api = UsdPhysics.MassAPI.Get(portable_stage, portable_body_path)
    mass_value = mass_api.GetMassAttr().Get()
    diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get()
    report = {
        "status": "ok",
        "isaac_imported_root": imported_path,
        "rigid_body_prim": str(body_path),
        "usd": str(output_path),
        "flattened_usd": str(flattened_path),
        "mass_kg": float(mass_value),
        "diagonal_inertia_kg_m2": [float(value) for value in diagonal_inertia],
        "fixed_base": False,
        "distance_scale": 1.0,
        "source_geometry": "primitive URDF geometry; no external meshes",
        "normalized_cube_extents": normalized_cube_extents,
    }
    report_path = output_path.with_suffix(".import_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


try:
    report = main()
    print(json.dumps(report, indent=2), flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    APP.close()
