#!/usr/bin/env python3
"""Create a silver/black RACER Crazyflie with a front-facing depth camera."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "usd" / "crazyflie_with_racer_dynamics.usd"
DEFAULT_OUTPUT = ROOT / "usd" / "crazyflie_silver_depth_camera.usd"

ROBOT_PATH = Sdf.Path("/racer_so3_quadrotor")
BODY_PATH = ROBOT_PATH.AppendChild("base_link")
MESH_PATH = BODY_PATH.AppendPath("visuals/crazyflie_mesh")
LOOKS_PATH = ROBOT_PATH.AppendChild("Looks")
CAMERA_VISUAL_PATH = BODY_PATH.AppendChild("depth_camera_visual")
CAMERA_FRAME_PATH = BODY_PATH.AppendPath("racer_frames/camera_optical_frame")
USD_CAMERA_PATH = BODY_PATH.AppendPath("racer_frames/depth_camera_usd")
CAMERA_YAW_DEG = 45.0
CAMERA_LENS_OFFSET_X_M = 0.0385
CAMERA_CENTER_Z_M = 0.059

BLACK_MESH_NAMES = {
    "board",
    "propeller",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def create_preview_material(
    stage: Usd.Stage,
    name: str,
    color: Gf.Vec3f,
    *,
    metallic: float,
    roughness: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, LOOKS_PATH.AppendChild(name))
    shader = UsdShade.Shader.Define(
        stage, LOOKS_PATH.AppendPath(f"{name}/PreviewSurface")
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def add_cube(
    stage: Usd.Stage,
    path: Sdf.Path,
    translation: Gf.Vec3d,
    dimensions: Gf.Vec3f,
    material: UsdShade.Material,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(translation)
    cube.AddScaleOp().Set(dimensions)
    cube.CreateDisplayColorPrimvar().Set([material_color(material)])
    bind_material(cube.GetPrim(), material)
    return cube


def add_cylinder(
    stage: Usd.Stage,
    path: Sdf.Path,
    translation: Gf.Vec3d,
    *,
    radius: float,
    height: float,
    material: UsdShade.Material,
) -> UsdGeom.Cylinder:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.x)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set(translation)
    cylinder.CreateDisplayColorPrimvar().Set([material_color(material)])
    bind_material(cylinder.GetPrim(), material)
    return cylinder


def material_color(material: UsdShade.Material) -> Gf.Vec3f:
    shader = UsdShade.Shader(
        material.GetPrim().GetChild("PreviewSurface")
    )
    return shader.GetInput("diffuseColor").Get()


def author_depth_camera(
    stage: Usd.Stage,
    black: UsdShade.Material,
    lens_black: UsdShade.Material,
) -> None:
    camera_root = UsdGeom.Xform.Define(stage, CAMERA_VISUAL_PATH)
    camera_root.AddRotateZOp().Set(CAMERA_YAW_DEG)
    camera_root.GetPrim().CreateAttribute(
        "racer:sensorType", Sdf.ValueTypeNames.Token, custom=True
    ).Set("depth_camera")
    camera_root.GetPrim().CreateAttribute(
        "racer:yawBodyDeg", Sdf.ValueTypeNames.Float, custom=True
    ).Set(CAMERA_YAW_DEG)
    camera_root.GetPrim().CreateAttribute(
        "racer:mountDescription", Sdf.ValueTypeNames.String, custom=True
    ).Set("stereo depth camera flush-mounted on top, yawed +45 degrees about body Z")

    # A RealSense-style black stereo enclosure centered above the UAV and
    # facing body +X. Its bottom face is at z=0.036 m, touching the highest
    # central body features rather than floating on separate supports.
    add_cube(
        stage,
        CAMERA_VISUAL_PATH.AppendChild("black_housing"),
        Gf.Vec3d(0.0, 0.0, CAMERA_CENTER_Z_M),
        Gf.Vec3f(0.050, 0.110, 0.046),
        black,
    )
    add_cube(
        stage,
        CAMERA_VISUAL_PATH.AppendChild("black_front_bezel"),
        Gf.Vec3d(0.027, 0.0, CAMERA_CENTER_Z_M),
        Gf.Vec3f(0.006, 0.102, 0.039),
        black,
    )

    for name, y_value in (("left_lens", 0.033), ("right_lens", -0.033)):
        add_cylinder(
            stage,
            CAMERA_VISUAL_PATH.AppendChild(name),
            Gf.Vec3d(0.033, y_value, CAMERA_CENTER_Z_M),
            radius=0.013,
            height=0.008,
            material=lens_black,
        )
        add_cylinder(
            stage,
            CAMERA_VISUAL_PATH.AppendChild(f"{name}_inner"),
            Gf.Vec3d(0.0375, y_value, CAMERA_CENTER_Z_M),
            radius=0.008,
            height=0.002,
            material=black,
        )

    add_cylinder(
        stage,
        CAMERA_VISUAL_PATH.AppendChild("ir_projector"),
        Gf.Vec3d(0.033, 0.0, CAMERA_CENTER_Z_M),
        radius=0.006,
        height=0.008,
        material=lens_black,
    )

    # Move the existing ROS optical frame to the physical lens center. Its
    # original orientation (+Z optical forward -> +X body) remains untouched.
    optical_frame = stage.GetPrimAtPath(CAMERA_FRAME_PATH)
    if not optical_frame.IsValid():
        raise RuntimeError(f"missing camera optical frame: {CAMERA_FRAME_PATH}")
    translate = optical_frame.GetAttribute("xformOp:translate")
    if not translate.IsValid():
        translate = UsdGeom.Xformable(optical_frame).AddTranslateOp().GetAttr()
        order = optical_frame.GetAttribute("xformOpOrder").Get() or []
        if "xformOp:translate" not in order:
            optical_frame.GetAttribute("xformOpOrder").Set(
                ["xformOp:translate", *order]
            )
    yaw_rad = math.radians(CAMERA_YAW_DEG)
    lens_position = Gf.Vec3d(
        CAMERA_LENS_OFFSET_X_M * math.cos(yaw_rad),
        CAMERA_LENS_OFFSET_X_M * math.sin(yaw_rad),
        CAMERA_CENTER_Z_M,
    )
    translate.Set(lens_position)
    orient = optical_frame.GetAttribute("xformOp:orient")
    if not orient.IsValid() or orient.Get() is None:
        raise RuntimeError(f"missing camera optical orientation: {CAMERA_FRAME_PATH}")
    original_quatf = orient.Get()
    original_quatd = Gf.Quatd(
        original_quatf.GetReal(), Gf.Vec3d(original_quatf.GetImaginary())
    )
    yaw_quatd = Gf.Quatd(
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), CAMERA_YAW_DEG).GetQuat()
    )
    rotated_quatd = yaw_quatd * original_quatd
    orient.Set(
        Gf.Quatf(
            float(rotated_quatd.GetReal()),
            Gf.Vec3f(rotated_quatd.GetImaginary()),
        )
    )
    optical_frame.CreateAttribute(
        "racer:depthCameraBaselineM", Sdf.ValueTypeNames.Float, custom=True
    ).Set(0.066)
    optical_frame.CreateAttribute(
        "racer:depthRangeM", Sdf.ValueTypeNames.Float2, custom=True
    ).Set(Gf.Vec2f(0.15, 12.0))

    # Also provide a standard renderable USD Camera. USD cameras look down -Z,
    # so a -90 degree Y rotation points it along local +X; the preceding body-Z
    # rotation then applies the requested +45 degree yaw.
    usd_camera = UsdGeom.Camera.Define(stage, USD_CAMERA_PATH)
    usd_camera.AddTranslateOp().Set(lens_position)
    usd_camera.AddRotateZOp().Set(CAMERA_YAW_DEG)
    usd_camera.AddRotateYOp().Set(-90.0)
    usd_camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    usd_camera.CreateFocalLengthAttr(18.0)
    usd_camera.CreateHorizontalApertureAttr(20.955)
    usd_camera.CreateVerticalApertureAttr(15.2908)
    usd_camera.CreateClippingRangeAttr(Gf.Vec2f(0.15, 12.0))
    usd_camera.GetPrim().CreateAttribute(
        "racer:sensorType", Sdf.ValueTypeNames.Token, custom=True
    ).Set("depth_camera")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"could not open source USD: {source}")
    if not source_stage.Export(str(output)):
        raise RuntimeError(f"could not export USD copy: {output}")

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"could not reopen output USD: {output}")

    silver = create_preview_material(
        stage,
        "SilverMetal",
        Gf.Vec3f(0.62, 0.68, 0.75),
        metallic=0.90,
        roughness=0.22,
    )
    black = create_preview_material(
        stage,
        "Black",
        Gf.Vec3f(0.008, 0.010, 0.014),
        metallic=0.20,
        roughness=0.30,
    )
    lens_black = create_preview_material(
        stage,
        "LensBlack",
        Gf.Vec3f(0.003, 0.006, 0.010),
        metallic=0.05,
        roughness=0.08,
    )

    mesh_root = stage.GetPrimAtPath(MESH_PATH)
    if not mesh_root.IsValid():
        raise RuntimeError(f"missing Crazyflie mesh root: {MESH_PATH}")

    recolored = []
    for prim in Usd.PrimRange(mesh_root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        target_material = black if prim.GetName() in BLACK_MESH_NAMES else silver
        bind_material(prim, target_material)
        UsdGeom.Gprim(prim).CreateDisplayColorPrimvar().Set(
            [material_color(target_material)]
        )
        recolored.append((str(prim.GetPath()), target_material.GetPrim().GetName()))

    author_depth_camera(stage, black, lens_black)

    root = stage.GetDefaultPrim()
    root.CreateAttribute(
        "racer:modelVariant", Sdf.ValueTypeNames.String, custom=True
    ).Set("Silver RACER Crazyflie with black propellers, board, and depth camera")
    root.CreateAttribute(
        "racer:appearanceSource", Sdf.ValueTypeNames.Asset, custom=True
    ).Set(Sdf.AssetPath(str(source)))
    stage.GetRootLayer().customLayerData = {
        **dict(stage.GetRootLayer().customLayerData),
        "appearanceVariant": "silver_black_depth_camera",
        "appearanceGenerator": str(Path(__file__).resolve()),
        "appearanceSource": str(source),
    }
    stage.GetRootLayer().Save()

    print(f"created: {output}")
    print(f"recolored meshes: {len(recolored)}")
    for path, material_name in recolored:
        print(f"  {material_name}: {path}")
    print(f"depth camera visual: {CAMERA_VISUAL_PATH}")
    print(f"depth camera optical frame: {CAMERA_FRAME_PATH}")
    print(f"standard USD camera: {USD_CAMERA_PATH}")


if __name__ == "__main__":
    main()
