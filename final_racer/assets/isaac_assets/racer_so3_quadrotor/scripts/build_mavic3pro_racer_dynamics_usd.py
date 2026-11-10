#!/usr/bin/env python3
"""Build a Mavic 3 Pro-inspired visual shell on unchanged RACER dynamics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "usd" / "crazyflie_with_racer_dynamics.usd"
DEFAULT_OUTPUT = ROOT / "usd" / "mavic3pro_visual_with_racer_dynamics.usd"

ROBOT_PATH = Sdf.Path("/racer_so3_quadrotor")
BODY_PATH = ROBOT_PATH.AppendChild("base_link")
VISUALS_PATH = BODY_PATH.AppendChild("visuals")
LOOKS_PATH = ROBOT_PATH.AppendChild("Looks")

ROTOR_RADIUS_M = 0.26
X_LAYOUT_OFFSET_M = ROTOR_RADIUS_M / math.sqrt(2.0)
PROPELLER_RADIUS_M = 0.062

# X-layout appearance. IDs retain the RACER spin pairing: 0/1 CW, 2/3 CCW.
VISUAL_ROTORS = {
    0: ((X_LAYOUT_OFFSET_M, -X_LAYOUT_OFFSET_M, 0.066), -1, "front_right"),
    1: ((-X_LAYOUT_OFFSET_M, X_LAYOUT_OFFSET_M, 0.058), -1, "rear_left"),
    2: ((-X_LAYOUT_OFFSET_M, -X_LAYOUT_OFFSET_M, 0.058), 1, "rear_right"),
    3: ((X_LAYOUT_OFFSET_M, X_LAYOUT_OFFSET_M, 0.066), 1, "front_left"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def create_material(
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


def material_color(material: UsdShade.Material) -> Gf.Vec3f:
    shader = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
    return shader.GetInput("diffuseColor").Get()


def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorPrimvar().Set([material_color(material)])


def add_cube(
    stage: Usd.Stage,
    path: Sdf.Path,
    position: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: UsdShade.Material,
    *,
    rotate_z_deg: float = 0.0,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotate_z_deg:
        cube.AddRotateZOp().Set(rotate_z_deg)
    cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    bind(cube.GetPrim(), material)
    return cube


def add_cylinder(
    stage: Usd.Stage,
    path: Sdf.Path,
    position: tuple[float, float, float],
    *,
    axis: str,
    radius: float,
    height: float,
    material: UsdShade.Material,
) -> UsdGeom.Cylinder:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(axis.upper())
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*position))
    bind(cylinder.GetPrim(), material)
    return cylinder


def add_arm(
    stage: Usd.Stage,
    path: Sdf.Path,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    width: float,
    height: float,
    material: UsdShade.Material,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    midpoint = (
        0.5 * (start[0] + end[0]),
        0.5 * (start[1] + end[1]),
        0.5 * (start[2] + end[2]),
    )
    add_cube(
        stage,
        path,
        midpoint,
        (length, width, height),
        material,
        rotate_z_deg=math.degrees(math.atan2(dy, dx)),
    )


def add_tapered_shell(
    stage: Usd.Stage,
    path: Sdf.Path,
    sections: list[tuple[float, float, float, float]],
    material: UsdShade.Material,
) -> UsdGeom.Mesh:
    """Create an angular fuselage from x/half-width/z-bottom/z-top sections."""

    points = []
    for x_value, half_width, z_bottom, z_top in sections:
        points.extend(
            (
                Gf.Vec3f(x_value, -half_width, z_bottom),
                Gf.Vec3f(x_value, half_width, z_bottom),
                Gf.Vec3f(x_value, half_width, z_top),
                Gf.Vec3f(x_value, -half_width, z_top),
            )
        )

    faces = []
    faces.append((0, 3, 2, 1))
    for section_index in range(len(sections) - 1):
        a = 4 * section_index
        b = 4 * (section_index + 1)
        faces.extend(
            (
                (a + 0, b + 0, b + 1, a + 1),
                (a + 1, b + 1, b + 2, a + 2),
                (a + 2, b + 2, b + 3, a + 3),
                (a + 3, b + 3, b + 0, a + 0),
            )
        )
    last = 4 * (len(sections) - 1)
    faces.append((last + 0, last + 1, last + 2, last + 3))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(
                min(point[0] for point in points),
                min(point[1] for point in points),
                min(point[2] for point in points),
            ),
            Gf.Vec3f(
                max(point[0] for point in points),
                max(point[1] for point in points),
                max(point[2] for point in points),
            ),
        ]
    )
    bind(mesh.GetPrim(), material)
    return mesh


def add_propeller(
    stage: Usd.Stage,
    rotor_id: int,
    position: tuple[float, float, float],
    spin_sign: int,
    label: str,
    propeller_material: UsdShade.Material,
) -> None:
    rotor_path = VISUALS_PATH.AppendPath(f"rotors/rotor_{rotor_id}_{label}")
    rotor = UsdGeom.Xform.Define(stage, rotor_path)
    rotor.AddTranslateOp().Set(Gf.Vec3d(*position))
    rotor.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    prim = rotor.GetPrim()
    prim.CreateAttribute("racer:rotorId", Sdf.ValueTypeNames.Int, custom=True).Set(rotor_id)
    prim.CreateAttribute(
        "racer:spinDirectionSign", Sdf.ValueTypeNames.Int, custom=True
    ).Set(spin_sign)
    prim.CreateAttribute(
        "racer:spinDirection", Sdf.ValueTypeNames.String, custom=True
    ).Set("cw" if spin_sign < 0 else "ccw")
    prim.CreateAttribute(
        "racer:visualRotationAxis", Sdf.ValueTypeNames.Token, custom=True
    ).Set("Z")
    prim.CreateAttribute(
        "racer:visualRotationOp", Sdf.ValueTypeNames.Token, custom=True
    ).Set("xformOp:rotateXYZ")
    prim.CreateAttribute(
        "racer:visualPositionBodyM", Sdf.ValueTypeNames.Double3, custom=True
    ).Set(Gf.Vec3d(*position))

    blade_length = PROPELLER_RADIUS_M - 0.010
    blade_center = 0.010 + 0.5 * blade_length
    for name, center, yaw in (
        ("blade_a", blade_center, -7.0),
        ("blade_b", -blade_center, 173.0),
    ):
        add_cube(
            stage,
            rotor_path.AppendChild(name),
            (center, 0.0, 0.0),
            (blade_length, 0.011, 0.0022),
            propeller_material,
            rotate_z_deg=yaw,
        )
    add_cylinder(
        stage,
        rotor_path.AppendChild("hub"),
        (0.0, 0.0, 0.0),
        axis="Z",
        radius=0.009,
        height=0.006,
        material=propeller_material,
    )


def author_mavic_visual(stage: Usd.Stage) -> None:
    graphite = create_material(
        stage, "MavicGraphite", Gf.Vec3f(0.145, 0.155, 0.165), metallic=0.35, roughness=0.30
    )
    upper_graphite = create_material(
        stage, "MavicUpper", Gf.Vec3f(0.205, 0.215, 0.225), metallic=0.42, roughness=0.25
    )
    black = create_material(
        stage, "MavicBlack", Gf.Vec3f(0.010, 0.012, 0.015), metallic=0.16, roughness=0.31
    )
    motor_metal = create_material(
        stage, "MotorMetal", Gf.Vec3f(0.22, 0.24, 0.255), metallic=0.82, roughness=0.22
    )
    lens = create_material(
        stage, "LensGlass", Gf.Vec3f(0.004, 0.012, 0.022), metallic=0.05, roughness=0.06
    )
    accent = create_material(
        stage, "MavicAccent", Gf.Vec3f(0.58, 0.18, 0.055), metallic=0.15, roughness=0.30
    )

    visuals = UsdGeom.Xform.Define(stage, VISUALS_PATH)
    vprim = visuals.GetPrim()
    vprim.CreateAttribute(
        "racer:visualStyle", Sdf.ValueTypeNames.String, custom=True
    ).Set("Mavic 3 Pro-inspired unfolded shell")
    vprim.CreateAttribute(
        "racer:visualOnlyGeometry", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    vprim.CreateAttribute(
        "racer:visualRotorLayout", Sdf.ValueTypeNames.Token, custom=True
    ).Set("X")
    vprim.CreateAttribute(
        "racer:physicsRotorLayout", Sdf.ValueTypeNames.Token, custom=True
    ).Set("plus_unchanged")
    vprim.CreateAttribute(
        "racer:officialReferenceDimensionsUnfoldedM",
        Sdf.ValueTypeNames.Double3,
        custom=True,
    ).Set(Gf.Vec3d(0.3475, 0.2908, 0.1077))

    body_path = VISUALS_PATH.AppendChild("body")
    UsdGeom.Xform.Define(stage, body_path)
    add_tapered_shell(
        stage,
        body_path.AppendChild("lower_shell"),
        [
            (-0.125, 0.036, -0.027, 0.018),
            (-0.105, 0.053, -0.031, 0.030),
            (0.035, 0.060, -0.033, 0.039),
            (0.105, 0.050, -0.029, 0.030),
            (0.137, 0.033, -0.021, 0.014),
        ],
        graphite,
    )
    add_tapered_shell(
        stage,
        body_path.AppendChild("top_battery"),
        [
            (-0.100, 0.034, 0.030, 0.041),
            (-0.085, 0.044, 0.032, 0.050),
            (0.045, 0.046, 0.038, 0.054),
            (0.078, 0.035, 0.033, 0.045),
        ],
        upper_graphite,
    )
    add_cube(stage, body_path.AppendChild("battery_latch"), (-0.080, 0.0, 0.052), (0.025, 0.028, 0.006), black)
    for side, y_value in (("left", 0.049), ("right", -0.049)):
        add_cube(
            stage,
            body_path.AppendChild(f"side_vent_{side}"),
            (-0.015, y_value, 0.002),
            (0.070, 0.004, 0.018),
            black,
        )

    arms_path = VISUALS_PATH.AppendChild("folding_arms")
    UsdGeom.Xform.Define(stage, arms_path)
    arm_specs = {
        "front_right": ((0.066, -0.043, 0.017), (X_LAYOUT_OFFSET_M, -X_LAYOUT_OFFSET_M, 0.038), 0.030),
        "front_left": ((0.066, 0.043, 0.017), (X_LAYOUT_OFFSET_M, X_LAYOUT_OFFSET_M, 0.038), 0.030),
        "rear_right": ((-0.064, -0.043, 0.012), (-X_LAYOUT_OFFSET_M, -X_LAYOUT_OFFSET_M, 0.030), 0.026),
        "rear_left": ((-0.064, 0.043, 0.012), (-X_LAYOUT_OFFSET_M, X_LAYOUT_OFFSET_M, 0.030), 0.026),
    }
    for name, (start, end, width) in arm_specs.items():
        add_arm(stage, arms_path.AppendChild(name), start, end, width, 0.020, graphite)
        add_cylinder(
            stage,
            arms_path.AppendChild(f"hinge_{name}"),
            start,
            axis="Z",
            radius=0.018,
            height=0.025,
            material=upper_graphite,
        )

    rotors_path = VISUALS_PATH.AppendChild("rotors")
    UsdGeom.Xform.Define(stage, rotors_path)
    for rotor_id, (position, spin_sign, label) in VISUAL_ROTORS.items():
        motor_z = position[2] - 0.020
        add_cylinder(
            stage,
            rotors_path.AppendChild(f"motor_{rotor_id}_{label}"),
            (position[0], position[1], motor_z),
            axis="Z",
            radius=0.023,
            height=0.040,
            material=graphite,
        )
        add_cylinder(
            stage,
            rotors_path.AppendChild(f"motor_cap_{rotor_id}"),
            (position[0], position[1], position[2] - 0.006),
            axis="Z",
            radius=0.016,
            height=0.014,
            material=motor_metal,
        )
        add_propeller(stage, rotor_id, position, spin_sign, label, black)

    # Short integrated landing feet, characteristic of the unfolded Mavic silhouette.
    for name, position in (
        ("front_right", (0.158, -0.158, -0.012)),
        ("front_left", (0.158, 0.158, -0.012)),
        ("rear_right", (-0.126, -0.126, -0.016)),
        ("rear_left", (-0.126, 0.126, -0.016)),
    ):
        add_cube(
            stage,
            VISUALS_PATH.AppendPath(f"landing_feet/{name}"),
            position,
            (0.018, 0.018, 0.050 if name.startswith("front") else 0.035),
            graphite,
        )

    # Front gimbal and asymmetrical triple-camera cluster.
    camera_path = VISUALS_PATH.AppendChild("triple_camera_gimbal")
    UsdGeom.Xform.Define(stage, camera_path)
    add_cylinder(
        stage, camera_path.AppendChild("yaw_ring"), (0.108, 0.0, -0.030),
        axis="Z", radius=0.025, height=0.014, material=motor_metal,
    )
    add_cube(stage, camera_path.AppendChild("gimbal_arm"), (0.127, 0.0, -0.041), (0.032, 0.010, 0.044), motor_metal)
    add_cube(stage, camera_path.AppendChild("camera_housing"), (0.147, 0.0, -0.047), (0.038, 0.060, 0.058), black)
    lens_specs = (
        ("hasselblad_main", -0.012, -0.050, 0.017),
        ("medium_tele", 0.017, -0.031, 0.009),
        ("tele", 0.017, -0.062, 0.009),
    )
    for name, y_value, z_value, radius in lens_specs:
        add_cylinder(
            stage, camera_path.AppendChild(f"{name}_barrel"),
            (0.169, y_value, z_value), axis="X", radius=radius,
            height=0.010, material=motor_metal,
        )
        add_cylinder(
            stage, camera_path.AppendChild(f"{name}_glass"),
            (0.175, y_value, z_value), axis="X", radius=0.72 * radius,
            height=0.003, material=lens,
        )
    add_cube(stage, camera_path.AppendChild("camera_accent"), (0.175, -0.012, -0.073), (0.003, 0.021, 0.003), accent)

    sensors_path = VISUALS_PATH.AppendChild("obstacle_sensors")
    UsdGeom.Xform.Define(stage, sensors_path)
    for side, y_value in (("left", 0.029), ("right", -0.029)):
        add_cylinder(
            stage, sensors_path.AppendChild(f"front_{side}"),
            (0.141, y_value, 0.002), axis="X", radius=0.007,
            height=0.006, material=lens,
        )
        add_cylinder(
            stage, sensors_path.AppendChild(f"rear_{side}"),
            (-0.128, y_value, 0.003), axis="X", radius=0.006,
            height=0.006, material=lens,
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"could not open source USD: {source}")
    if not source_stage.Export(str(output)):
        raise RuntimeError(f"could not export source stage: {output}")

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"could not reopen output USD: {output}")
    stage.RemovePrim(VISUALS_PATH)
    author_mavic_visual(stage)

    root = stage.GetDefaultPrim()
    root.CreateAttribute(
        "racer:modelVariant", Sdf.ValueTypeNames.String, custom=True
    ).Set("Mavic 3 Pro-inspired appearance with unchanged RACER SO3 dynamics")
    root.CreateAttribute(
        "racer:appearanceSource", Sdf.ValueTypeNames.Asset, custom=True
    ).Set(Sdf.AssetPath(str(source)))
    stage.GetRootLayer().customLayerData = {
        **dict(stage.GetRootLayer().customLayerData),
        "appearanceVariant": "mavic3pro_inspired_unfolded",
        "appearanceGenerator": str(Path(__file__).resolve()),
        "dynamicsSourceUsd": str(source),
        "physicsInvariant": "rigid body, mass, inertia, collisions, frames, and racer dynamics are unchanged",
    }

    visual_root = stage.GetPrimAtPath(VISUALS_PATH)
    unexpected_physics = []
    for prim in Usd.PrimRange(visual_root):
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
        raise RuntimeError(f"visual shell unexpectedly contains physics APIs: {unexpected_physics}")

    stage.GetRootLayer().Save()
    print(f"created: {output}")
    print("visual style: Mavic 3 Pro-inspired unfolded X-layout")
    print("physics: copied unchanged from crazyflie_with_racer_dynamics.usd")
    print(f"visual rotor count: {len(VISUAL_ROTORS)}")


if __name__ == "__main__":
    main()
