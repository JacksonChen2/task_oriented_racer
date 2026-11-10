#!/usr/bin/env python3
"""Build a portable Crazyflie-mesh vehicle with RACER SO3 dynamics.

The installed Crazyflie USD contributes rendering meshes only.  Its Y-up X
layout is converted to the RACER Z-up plus layout, while rigid-body, collision,
rotor-frame, and propulsion metadata come from the RACER model and its known
good flattened USD.  No physics schema is copied from the visual source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    os.environ.get("ISAAC_SIM_ROOT", str(Path.home() / "isaacsim"))
) / (
    "extscache/"
    "omni.warp.core-1.8.2+lx64/warp/examples/assets/crazyflie.usd"
)
DEFAULT_TEMPLATE = ROOT / "usd" / "racer_so3_quadrotor_flattened.usd"
DEFAULT_MODEL = ROOT / "config" / "racer_so3_model.json"
DEFAULT_OUTPUT = ROOT / "usd" / "crazyflie_with_racer_dynamics.usd"

ROOT_PATH = Sdf.Path("/racer_so3_quadrotor")
BODY_PATH = ROOT_PATH.AppendChild("base_link")
VISUALS_PATH = BODY_PATH.AppendChild("visuals")
SOURCE_COPY_PATH = VISUALS_PATH.AppendChild("crazyflie_mesh")

SOURCE_PROPELLERS = {
    0: "/root/propeller_cw_front/propeller",
    1: "/root/propeller_cw_back/propeller",
    2: "/root/propeller_ccw_back/propeller",
    3: "/root/propeller_ccw_front/propeller",
}
ROTOR_SPIN_SIGNS = {0: -1, 1: -1, 2: 1, 3: 1}
SOURCE_BOARD = "/root/body/board"
TARGET_PROPELLER_Z_M = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--physics-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def world_bound_center(stage: Usd.Stage, path: str) -> Gf.Vec3d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"required visual prim is missing: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    return Gf.Vec3d(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint())


def visual_transform(source_stage: Usd.Stage, arm_length: float) -> tuple[Gf.Matrix4d, dict]:
    """Map source propeller centers onto the four RACER plus-layout centers."""

    rotor = world_bound_center(source_stage, SOURCE_PROPELLERS[0])
    board = world_bound_center(source_stage, SOURCE_BOARD)
    if min(abs(rotor[0]), abs(rotor[2]), abs(rotor[1] - board[1])) <= 1e-9:
        raise RuntimeError("Crazyflie visual landmarks are degenerate")

    # The source is Y-up and its two horizontal axes differ by about 0.7%.
    # Separate X/Z scale factors remove that small asymmetry.  After the Y-up
    # to Z-up conversion and +45 degree yaw, each visible propeller center lies
    # exactly on one RACER cardinal-axis rotor frame.
    scale_x = arm_length / (math.sqrt(2.0) * abs(float(rotor[0])))
    scale_source_z = arm_length / (math.sqrt(2.0) * abs(float(rotor[2])))
    scale_up = TARGET_PROPELLER_Z_M / float(rotor[1] - board[1])
    translate_z = -float(board[1]) * scale_up

    scale = Gf.Matrix4d().SetScale(
        Gf.Vec3d(scale_x, scale_up, scale_source_z)
    )
    y_up_to_z_up = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0)
    )
    x_to_plus = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 45.0)
    )
    center_on_body = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(0.0, 0.0, translate_z)
    )
    matrix = scale * y_up_to_z_up * x_to_plus * center_on_body

    target_positions = {
        0: Gf.Vec3d(arm_length, 0.0, TARGET_PROPELLER_Z_M),
        1: Gf.Vec3d(-arm_length, 0.0, TARGET_PROPELLER_Z_M),
        2: Gf.Vec3d(0.0, arm_length, TARGET_PROPELLER_Z_M),
        3: Gf.Vec3d(0.0, -arm_length, TARGET_PROPELLER_Z_M),
    }
    mapped_positions = {}
    for rotor_id, path in SOURCE_PROPELLERS.items():
        mapped = matrix.Transform(world_bound_center(source_stage, path))
        error = (mapped - target_positions[rotor_id]).GetLength()
        if error > 1e-6:
            raise RuntimeError(
                f"visual rotor {rotor_id} misses RACER frame by {error:.9g} m"
            )
        mapped_positions[str(rotor_id)] = [float(value) for value in mapped]

    details = {
        "source_up_axis": str(UsdGeom.GetStageUpAxis(source_stage)),
        "target_up_axis": "Z",
        "scale_source_xyz": [scale_x, scale_up, scale_source_z],
        "yaw_after_axis_conversion_deg": 45.0,
        "translation_target_xyz_m": [0.0, 0.0, translate_z],
        "target_propeller_height_m": TARGET_PROPELLER_Z_M,
        "mapped_visual_propeller_centers_m": mapped_positions,
    }
    return matrix, details


def resize_copied_propellers(
    stage: Usd.Stage,
    source_stage: Usd.Stage,
    matrix: Gf.Matrix4d,
    target_radius: float,
) -> dict[str, float]:
    """Match each visual blade envelope to the RACER propeller radius."""

    factors = {}
    for rotor_id, source_path in SOURCE_PROPELLERS.items():
        source_mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath(source_path))
        points = source_mesh.GetPointsAttr().Get()
        if not points:
            raise RuntimeError(f"propeller mesh has no points: {source_path}")
        unscaled_radius = max(
            math.hypot(*matrix.TransformDir(Gf.Vec3d(point))[:2])
            for point in points
        )
        factor = target_radius / unscaled_radius
        copied_parent = stage.GetPrimAtPath(
            SOURCE_COPY_PATH.AppendPath(
                Sdf.Path(source_path).GetParentPath().MakeRelativePath(Sdf.Path("/root"))
            )
        )
        scale_attr = copied_parent.GetAttribute("xformOp:scale")
        if not scale_attr.IsValid():
            raise RuntimeError(f"copied propeller has no scale op: {copied_parent.GetPath()}")
        scale_attr.Set(Gf.Vec3d(factor, factor, factor))
        factors[str(rotor_id)] = factor
    return factors


def author_propeller_animation_metadata(stage: Usd.Stage) -> dict[str, dict]:
    """Mark the render-only propeller Xforms for RPM-driven animation."""

    metadata = {}
    for rotor_id, source_path in SOURCE_PROPELLERS.items():
        relative_parent = Sdf.Path(source_path).GetParentPath().MakeRelativePath(
            Sdf.Path("/root")
        )
        propeller = stage.GetPrimAtPath(
            SOURCE_COPY_PATH.AppendPath(relative_parent)
        )
        if not propeller.IsValid():
            raise RuntimeError(f"copied propeller parent is missing: {propeller.GetPath()}")
        sign = ROTOR_SPIN_SIGNS[rotor_id]
        propeller.CreateAttribute(
            "racer:rotorId", Sdf.ValueTypeNames.Int, custom=True
        ).Set(rotor_id)
        propeller.CreateAttribute(
            "racer:spinDirectionSign", Sdf.ValueTypeNames.Int, custom=True
        ).Set(sign)
        propeller.CreateAttribute(
            "racer:spinDirection", Sdf.ValueTypeNames.String, custom=True
        ).Set("cw" if sign < 0 else "ccw")
        # Source Crazyflie is Y-up. The parent visual transform maps this local
        # +Y axis to RACER body +Z, so xformOp:rotateXYZ's Y component spins the
        # blade around its visible shaft without introducing another rigid body.
        propeller.CreateAttribute(
            "racer:visualRotationAxis", Sdf.ValueTypeNames.Token, custom=True
        ).Set("Y")
        propeller.CreateAttribute(
            "racer:visualRotationOp", Sdf.ValueTypeNames.Token, custom=True
        ).Set("xformOp:rotateXYZ")
        metadata[str(rotor_id)] = {
            "path": str(propeller.GetPath()),
            "spin_direction": "cw" if sign < 0 else "ccw",
            "spin_direction_sign": sign,
            "local_rotation_axis": "Y",
        }
    body = stage.GetPrimAtPath(BODY_PATH)
    body.CreateAttribute(
        "racer:visualPropellerCount", Sdf.ValueTypeNames.Int, custom=True
    ).Set(len(metadata))
    return metadata


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    template_path = args.physics_template.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("r", encoding="utf-8") as stream:
        model = json.load(stream)

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"could not open Crazyflie source: {source_path}")
    template_stage = Usd.Stage.Open(str(template_path))
    if template_stage is None:
        raise RuntimeError(f"could not open RACER physics template: {template_path}")

    source_root = source_stage.GetDefaultPrim()
    template_root = template_stage.GetDefaultPrim()
    template_collisions = (
        template_root.GetPath().AppendChild("base_link").AppendChild("collisions")
    )
    if not template_stage.GetPrimAtPath(template_collisions).IsValid():
        raise RuntimeError(f"template collision root is missing: {template_collisions}")

    matrix, transform_details = visual_transform(
        source_stage, float(model["propulsion"]["arm_length_m"])
    )

    if output_path.exists():
        output_path.unlink()
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT_PATH)
    template_body = template_root.GetPath().AppendChild("base_link")
    if not Sdf.CopySpec(
        template_stage.GetRootLayer(),
        template_body,
        stage.GetRootLayer(),
        BODY_PATH,
    ):
        raise RuntimeError("failed to copy the complete RACER physical body")

    # Keep every physical and RACER metadata field from the validated template,
    # but replace its primitive visual branch with the embedded Crazyflie mesh.
    stage.RemovePrim(VISUALS_PATH)
    visuals = UsdGeom.Xform.Define(stage, VISUALS_PATH)
    visuals.AddTransformOp().Set(matrix)
    if not Sdf.CopySpec(
        source_stage.GetRootLayer(),
        source_root.GetPath(),
        stage.GetRootLayer(),
        SOURCE_COPY_PATH,
    ):
        raise RuntimeError("failed to embed the Crazyflie visual hierarchy")
    transform_details["propeller_mesh_scale_factors"] = resize_copied_propellers(
        stage,
        source_stage,
        matrix,
        float(model["propulsion"]["propeller_radius_m"]),
    )
    transform_details["propeller_animation"] = author_propeller_animation_metadata(
        stage
    )

    # The visual source currently has no physics APIs.  Keep this assertion so
    # an upstream asset change cannot silently introduce a second rigid body or
    # visual-mesh collisions into the generated vehicle.
    copied_visual = stage.GetPrimAtPath(SOURCE_COPY_PATH)
    unexpected_physics = []
    for prim in Usd.PrimRange(copied_visual):
        if any(
            prim.HasAPI(api)
            for api in (
                UsdPhysics.RigidBodyAPI,
                UsdPhysics.MassAPI,
                UsdPhysics.CollisionAPI,
                UsdPhysics.ArticulationRootAPI,
            )
        ):
            unexpected_physics.append(str(prim.GetPath()))
    if unexpected_physics:
        raise RuntimeError(f"visual source contains physics APIs: {unexpected_physics}")

    visuals.GetPrim().CreateAttribute(
        "racer:visualSource", Sdf.ValueTypeNames.String, custom=True
    ).Set(str(source_path))
    visuals.GetPrim().CreateAttribute(
        "racer:visualTransformDetails", Sdf.ValueTypeNames.String, custom=True
    ).Set(json.dumps(transform_details, sort_keys=True))
    root.GetPrim().CreateAttribute(
        "racer:modelVariant", Sdf.ValueTypeNames.String, custom=True
    ).Set("Crazyflie mesh with original RACER SO3 dynamics")

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "visualSource": str(source_path),
        "physicsTemplate": str(template_path),
        "dynamicsModel": str(model_path),
        "generator": str(Path(__file__).resolve()),
    }
    stage.GetRootLayer().Save()

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "visual_source": str(source_path),
                "physics_template": str(template_path),
                "model": str(model_path),
                "transform": transform_details,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
