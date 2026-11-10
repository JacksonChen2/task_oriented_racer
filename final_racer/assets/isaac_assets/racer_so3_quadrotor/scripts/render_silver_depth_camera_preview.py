#!/usr/bin/env python3
"""Render studio previews of the silver depth-camera UAV in Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "usd" / "crazyflie_silver_depth_camera.usd"
DEFAULT_OUTPUT_DIR = ROOT / "preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


def create_material(stage, path: str, color: Gf.Vec3f, roughness: float):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def build_studio(asset_path: Path) -> None:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    drone = UsdGeom.Xform.Define(stage, "/World/Drone")
    drone.GetPrim().GetReferences().AddReference(str(asset_path))
    drone.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.31))

    floor_material = create_material(
        stage, "/World/Looks/Floor", Gf.Vec3f(0.055, 0.065, 0.080), 0.34
    )
    floor = UsdGeom.Cylinder.Define(stage, "/World/StudioFloor")
    floor.CreateAxisAttr(UsdGeom.Tokens.z)
    floor.CreateRadiusAttr(1.3)
    floor.CreateHeightAttr(0.035)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_material)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(550.0)
    dome.CreateColorAttr(Gf.Vec3f(0.72, 0.80, 1.0))

    key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(8500.0)
    key.CreateRadiusAttr(0.20)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.80))
    key.AddTranslateOp().Set(Gf.Vec3d(0.70, -0.65, 1.15))

    fill = UsdLux.SphereLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(5200.0)
    fill.CreateRadiusAttr(0.18)
    fill.CreateColorAttr(Gf.Vec3f(0.58, 0.72, 1.0))
    fill.AddTranslateOp().Set(Gf.Vec3d(-0.65, -0.30, 0.65))

    rim = UsdLux.SphereLight.Define(stage, "/World/RimLight")
    rim.CreateIntensityAttr(6500.0)
    rim.CreateRadiusAttr(0.16)
    rim.CreateColorAttr(Gf.Vec3f(0.78, 0.88, 1.0))
    rim.AddTranslateOp().Set(Gf.Vec3d(-0.25, 0.75, 0.90))


def save_rgb(data: np.ndarray, path: Path) -> None:
    image = Image.fromarray(data).convert("RGB")
    image.save(path, quality=95)


def make_contact_sheet(first: Path, second: Path, output: Path) -> None:
    image_a = Image.open(first).convert("RGB")
    image_b = Image.open(second).convert("RGB")
    gutter = 18
    canvas = Image.new(
        "RGB", (image_a.width + image_b.width + gutter, max(image_a.height, image_b.height)),
        (15, 18, 24),
    )
    canvas.paste(image_a, (0, 0))
    canvas.paste(image_b, (image_a.width + gutter, 0))
    canvas.save(output, quality=95)


def main() -> None:
    asset_path = ARGS.usd.resolve()
    output_dir = ARGS.output_dir.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_studio(asset_path)
    settings = carb.settings.get_settings()
    settings.set("/omni/replicator/captureOnPlay", False)
    settings.set("rtx/post/dlss/execMode", 2)
    settings.set("rtx/post/aa/op", 3)

    cameras = (
        rep.create.camera(
            position=(0.90, 0.78, 0.62),
            look_at=(0.04, 0.0, 0.28),
            focal_length=58.0,
            clipping_range=(0.03, 100.0),
        ),
        rep.create.camera(
            position=(1.00, 0.40, 0.52),
            look_at=(0.02, 0.0, 0.34),
            focal_length=48.0,
            clipping_range=(0.03, 100.0),
        ),
    )
    products = [
        rep.create.render_product(camera, (ARGS.width, ARGS.height))
        for camera in cameras
    ]
    annotators = []
    for product in products:
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        annotators.append(annotator)

    rep.orchestrator.set_capture_on_play(False)
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0)

    hero_path = output_dir / "crazyflie_silver_depth_camera_hero.png"
    front_path = output_dir / "crazyflie_silver_depth_camera_front.png"
    save_rgb(annotators[0].get_data(), hero_path)
    save_rgb(annotators[1].get_data(), front_path)
    contact_path = output_dir / "crazyflie_silver_depth_camera_preview.png"
    make_contact_sheet(hero_path, front_path, contact_path)

    rep.orchestrator.wait_until_complete()
    print(f"hero preview: {hero_path}")
    print(f"front preview: {front_path}")
    print(f"contact sheet: {contact_path}")


try:
    main()
finally:
    APP.close()
