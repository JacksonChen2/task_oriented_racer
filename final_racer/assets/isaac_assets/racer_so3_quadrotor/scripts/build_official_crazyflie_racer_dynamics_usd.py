#!/usr/bin/env python3
"""Embed the unmodified Isaac Sim Crazyflie visual in the RACER vehicle.

The generated USD inherits the complete physical body from
``crazyflie_with_racer_dynamics.usd``.  Its existing visual branch is replaced
with an exact Sdf copy of Isaac Sim's installed Crazyflie hierarchy.  Only one
uniform model scale plus rigid coordinate conversion/placement is authored on
the visual parent; mesh points, topology, child transforms, and relative part
sizes are not edited.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DYNAMICS = ROOT / "usd" / "crazyflie_with_racer_dynamics.usd"
DEFAULT_OUTPUT = ROOT / "usd" / "crazyflie_official_visual_with_racer_dynamics.usd"

ROOT_PATH = Sdf.Path("/racer_so3_quadrotor")
BODY_PATH = ROOT_PATH.AppendChild("base_link")
VISUALS_PATH = BODY_PATH.AppendChild("visuals")
OFFICIAL_COPY_PATH = VISUALS_PATH.AppendChild("crazyflie_official")
SOURCE_ROOT_PATH = Sdf.Path("/root")
SOURCE_BOARD = "/root/body/board"
SOURCE_TOP_BOARD = "/root/body/battery_holder"
SOURCE_PINS = "/root/body/pins"
SOURCE_MOTORS = "/root/body/motors"
SOURCE_PROPELLERS = {
    0: "/root/propeller_cw_front/propeller",
    1: "/root/propeller_cw_back/propeller",
    2: "/root/propeller_ccw_back/propeller",
    3: "/root/propeller_ccw_front/propeller",
}
ROTOR_SPIN_SIGNS = {0: -1, 1: -1, 2: 1, 3: 1}


def locate_official_asset() -> Path:
    roots = []
    if os.environ.get("ISAAC_SIM_ROOT"):
        roots.append(Path(os.environ["ISAAC_SIM_ROOT"]))
    roots.extend(
        (
            Path.home() / "software" / "isaacsim",
            Path.home() / "isaacsim",
        )
    )
    relative = Path(
        "extscache/omni.warp.core-1.8.2+lx64/warp/examples/assets/crazyflie.usd"
    )
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    for root in roots:
        cache = root / "extscache"
        if cache.is_dir():
            matches = sorted(cache.glob("omni.warp.core-*/warp/examples/assets/crazyflie.usd"))
            if matches:
                return matches[-1]
    return roots[0] / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, default=locate_official_asset())
    parser.add_argument("--dynamics-source", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def world_bound_center(stage: Usd.Stage, path: str) -> Gf.Vec3d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"required prim is missing: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return Gf.Vec3d(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint())


def target_arm_length(stage: Usd.Stage) -> float:
    rotor = stage.GetPrimAtPath(BODY_PATH.AppendPath("racer_frames/rotor_0"))
    if not rotor.IsValid():
        raise RuntimeError("RACER rotor_0 frame is missing")
    transform = UsdGeom.Xformable(rotor).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = transform.Transform(Gf.Vec3d(0.0))
    body = stage.GetPrimAtPath(BODY_PATH)
    body_transform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_origin = body_transform.Transform(Gf.Vec3d(0.0))
    length = (origin - body_origin).GetLength()
    if length <= 0.0:
        raise RuntimeError("RACER arm length is not positive")
    return float(length)


def official_visual_transform(
    official_stage: Usd.Stage, dynamics_stage: Usd.Stage
) -> tuple[Gf.Matrix4d, dict]:
    """Return one similarity transform; it cannot distort the official model."""

    board = world_bound_center(official_stage, SOURCE_BOARD)
    rotor = world_bound_center(official_stage, SOURCE_PROPELLERS[0])
    radial = math.hypot(float(rotor[0] - board[0]), float(rotor[2] - board[2]))
    if radial <= 1e-9:
        raise RuntimeError("official Crazyflie rotor landmark is degenerate")

    arm_length = target_arm_length(dynamics_stage)
    uniform_scale = arm_length / radial

    # Source is Y-up.  After converting Y-up to Z-up, choose the yaw from the
    # actual official rotor center so rotor 0 lands exactly on body +X.  This is
    # a rigid rotation and preserves all official proportions.
    yaw_deg = math.degrees(
        math.atan2(float(rotor[2] - board[2]), float(rotor[0] - board[0]))
    )
    similarity = (
        Gf.Matrix4d().SetScale(Gf.Vec3d(uniform_scale))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg))
    )
    mapped_board = similarity.Transform(board)
    matrix = similarity * Gf.Matrix4d().SetTranslate(-mapped_board)
    if matrix.Transform(board).GetLength() > 1e-9:
        raise RuntimeError("failed to place official Crazyflie body center at the origin")

    mapped = {
        str(rotor_id): [
            float(value)
            for value in matrix.Transform(world_bound_center(official_stage, path))
        ]
        for rotor_id, path in SOURCE_PROPELLERS.items()
    }
    return matrix, {
        "policy": "exact official hierarchy under one uniform similarity transform",
        "source_up_axis": str(UsdGeom.GetStageUpAxis(official_stage)),
        "target_up_axis": "Z",
        "uniform_scale_xyz": [uniform_scale, uniform_scale, uniform_scale],
        "yaw_after_axis_conversion_deg": yaw_deg,
        "official_board_center_m": [float(value) for value in board],
        "mapped_visual_propeller_centers_m": mapped,
        "mesh_points_edited": False,
        "mesh_topology_edited": False,
        "child_transforms_edited": False,
        "propellers_resized_separately": False,
    }


def copied_path(source_path: str) -> Sdf.Path:
    relative = Sdf.Path(source_path).MakeRelativePath(SOURCE_ROOT_PATH)
    return OFFICIAL_COPY_PATH.AppendPath(relative)


def author_animation_metadata(stage: Usd.Stage) -> dict[str, dict]:
    metadata = {}
    for rotor_id, mesh_path in SOURCE_PROPELLERS.items():
        parent = stage.GetPrimAtPath(copied_path(mesh_path).GetParentPath())
        if not parent.IsValid():
            raise RuntimeError(f"copied propeller parent is missing for rotor {rotor_id}")
        sign = ROTOR_SPIN_SIGNS[rotor_id]
        parent.CreateAttribute("racer:rotorId", Sdf.ValueTypeNames.Int, custom=True).Set(
            rotor_id
        )
        parent.CreateAttribute(
            "racer:spinDirectionSign", Sdf.ValueTypeNames.Int, custom=True
        ).Set(sign)
        parent.CreateAttribute(
            "racer:spinDirection", Sdf.ValueTypeNames.String, custom=True
        ).Set("cw" if sign < 0 else "ccw")
        parent.CreateAttribute(
            "racer:visualRotationAxis", Sdf.ValueTypeNames.Token, custom=True
        ).Set("Y")
        parent.CreateAttribute(
            "racer:visualRotationOp", Sdf.ValueTypeNames.Token, custom=True
        ).Set("xformOp:rotateXYZ")
        metadata[str(rotor_id)] = {
            "path": str(parent.GetPath()),
            "spin_direction_sign": sign,
            "local_rotation_axis": "Y",
        }
    return metadata


def author_custom_materials(stage: Usd.Stage) -> dict:
    """Author the requested Crazyflie-inspired render-only materials."""

    material_path = VISUALS_PATH.AppendPath("Looks/BlackBoardAndPropellers")
    shader_path = material_path.AppendChild("PreviewSurface")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.008, 0.008, 0.010)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.32)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    black_target_paths = [copied_path(SOURCE_BOARD), copied_path(SOURCE_TOP_BOARD)] + [
        copied_path(path) for path in SOURCE_PROPELLERS.values()
    ]
    for path in black_target_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"black material target is not a mesh: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    gold_material_path = VISUALS_PATH.AppendPath("Looks/GoldPins")
    gold_shader_path = gold_material_path.AppendChild("PreviewSurface")
    gold_material = UsdShade.Material.Define(stage, gold_material_path)
    gold_shader = UsdShade.Shader.Define(stage, gold_shader_path)
    gold_shader.CreateIdAttr("UsdPreviewSurface")
    gold_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.83, 0.48, 0.12)
    )
    gold_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.23)
    gold_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.82)
    gold_material.CreateSurfaceOutput().ConnectToSource(
        gold_shader.ConnectableAPI(), "surface"
    )
    pins_path = copied_path(SOURCE_PINS)
    pins = stage.GetPrimAtPath(pins_path)
    if not pins.IsValid() or not pins.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"gold material target is not a mesh: {pins_path}")
    UsdShade.MaterialBindingAPI.Apply(pins).Bind(gold_material)

    motor_material_path = VISUALS_PATH.AppendPath("Looks/PaleGoldMotors")
    motor_shader_path = motor_material_path.AppendChild("PreviewSurface")
    motor_material = UsdShade.Material.Define(stage, motor_material_path)
    motor_shader = UsdShade.Shader.Define(stage, motor_shader_path)
    motor_shader.CreateIdAttr("UsdPreviewSurface")
    motor_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.72, 0.52, 0.25)
    )
    motor_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    motor_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.68)
    motor_material.CreateSurfaceOutput().ConnectToSource(
        motor_shader.ConnectableAPI(), "surface"
    )
    motors_path = copied_path(SOURCE_MOTORS)
    motors = stage.GetPrimAtPath(motors_path)
    if not motors.IsValid() or not motors.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"pale-gold material target is not a mesh: {motors_path}")
    UsdShade.MaterialBindingAPI.Apply(motors).Bind(motor_material)

    return {
        "black": {
            "material_path": str(material_path),
            "diffuse_color_linear_rgb": [0.008, 0.008, 0.010],
            "targets": [str(path) for path in black_target_paths],
        },
        "gold": {
            "material_path": str(gold_material_path),
            "diffuse_color_linear_rgb": [0.83, 0.48, 0.12],
            "metallic": 0.82,
            "target": str(pins_path),
        },
        "pale_gold_motors": {
            "material_path": str(motor_material_path),
            "diffuse_color_linear_rgb": [0.72, 0.52, 0.25],
            "metallic": 0.68,
            "target": str(motors_path),
        },
    }


def verify_exact_visual_copy(source: Usd.Stage, output: Usd.Stage) -> dict:
    source_meshes = [prim for prim in source.Traverse() if prim.IsA(UsdGeom.Mesh)]
    for source_prim in source_meshes:
        destination = output.GetPrimAtPath(copied_path(str(source_prim.GetPath())))
        if not destination.IsValid() or not destination.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"copied mesh is missing: {destination.GetPath()}")
        source_mesh = UsdGeom.Mesh(source_prim)
        destination_mesh = UsdGeom.Mesh(destination)
        for getter in (
            "GetPointsAttr",
            "GetFaceVertexCountsAttr",
            "GetFaceVertexIndicesAttr",
            "GetNormalsAttr",
            "GetExtentAttr",
        ):
            if getattr(source_mesh, getter)().Get() != getattr(destination_mesh, getter)().Get():
                raise RuntimeError(
                    f"official mesh data changed during copy: {source_prim.GetPath()} {getter}"
                )
    return {
        "mesh_count": len(source_meshes),
        "mesh_points_and_topology_exact": True,
    }


def verify_visual_has_no_physics(stage: Usd.Stage) -> None:
    visual = stage.GetPrimAtPath(VISUALS_PATH)
    unexpected = []
    for prim in Usd.PrimRange(visual):
        if any(
            prim.HasAPI(api)
            for api in (
                UsdPhysics.RigidBodyAPI,
                UsdPhysics.MassAPI,
                UsdPhysics.CollisionAPI,
                UsdPhysics.ArticulationRootAPI,
            )
        ):
            unexpected.append(str(prim.GetPath()))
    if unexpected:
        raise RuntimeError(f"render-only visual contains physics APIs: {unexpected}")


def main() -> None:
    args = parse_args()
    official_path = args.official_source.resolve()
    dynamics_path = args.dynamics_source.resolve()
    output_path = args.output.resolve()
    if not official_path.is_file():
        raise FileNotFoundError(f"official Isaac Sim Crazyflie asset not found: {official_path}")
    if not dynamics_path.is_file():
        raise FileNotFoundError(f"RACER dynamics source not found: {dynamics_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    official_stage = Usd.Stage.Open(str(official_path))
    dynamics_stage = Usd.Stage.Open(str(dynamics_path))
    if official_stage is None or dynamics_stage is None:
        raise RuntimeError("failed to open an input USD")
    if official_stage.GetDefaultPrim().GetPath() != SOURCE_ROOT_PATH:
        raise RuntimeError("unexpected official Crazyflie default prim")

    matrix, transform_details = official_visual_transform(official_stage, dynamics_stage)

    if output_path.exists():
        output_path.unlink()
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    dynamics_root = dynamics_stage.GetDefaultPrim()
    if not Sdf.CopySpec(
        dynamics_stage.GetRootLayer(),
        dynamics_root.GetPath(),
        stage.GetRootLayer(),
        ROOT_PATH,
    ):
        raise RuntimeError("failed to copy the RACER vehicle")

    # This is the only removed branch.  Rigid body, mass/inertia, collisions,
    # RACER rotor frames, camera/IMU frames, and propulsion metadata remain the
    # exact authored specs from the source dynamics USD.
    stage.RemovePrim(VISUALS_PATH)
    visuals = UsdGeom.Xform.Define(stage, VISUALS_PATH)
    visuals.AddTransformOp().Set(matrix)
    if not Sdf.CopySpec(
        official_stage.GetRootLayer(),
        SOURCE_ROOT_PATH,
        stage.GetRootLayer(),
        OFFICIAL_COPY_PATH,
    ):
        raise RuntimeError("failed to embed the official Crazyflie hierarchy")

    transform_details["propeller_animation"] = author_animation_metadata(stage)
    transform_details["visual_integrity"] = verify_exact_visual_copy(
        official_stage, stage
    )
    transform_details["custom_materials"] = author_custom_materials(stage)
    verify_visual_has_no_physics(stage)

    visuals.GetPrim().CreateAttribute(
        "racer:visualSource", Sdf.ValueTypeNames.String, custom=True
    ).Set(str(official_path))
    visuals.GetPrim().CreateAttribute(
        "racer:visualTransformDetails", Sdf.ValueTypeNames.String, custom=True
    ).Set(json.dumps(transform_details, sort_keys=True))
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PATH))
    stage.GetRootLayer().customLayerData = {
        "generator": str(Path(__file__).resolve()),
        "officialVisualSource": str(official_path),
        "dynamicsSource": str(dynamics_path),
        "visualPolicy": "official mesh and hierarchy copied exactly; uniform scale only",
    }
    stage.GetRootLayer().Save()

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "official_visual_source": str(official_path),
                "dynamics_source": str(dynamics_path),
                "transform": transform_details,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
