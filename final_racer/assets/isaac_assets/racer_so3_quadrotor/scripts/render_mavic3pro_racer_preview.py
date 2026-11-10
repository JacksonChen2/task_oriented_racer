#!/usr/bin/env python3
"""Render studio views of the Mavic-styled RACER dynamics asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "usd" / "mavic3pro_visual_with_racer_dynamics.usd"
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
    drone.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.30))

    floor_material = create_material(
        stage, "/World/Looks/Floor", Gf.Vec3f(0.035, 0.043, 0.055), 0.30
    )
    floor = UsdGeom.Cylinder.Define(stage, "/World/StudioFloor")
    floor.CreateAxisAttr(UsdGeom.Tokens.z)
    floor.CreateRadiusAttr(1.25)
    floor.CreateHeightAttr(0.035)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_material)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(430.0)
    dome.CreateColorAttr(Gf.Vec3f(0.70, 0.78, 0.94))
    for name, position, intensity, color, radius in (
        ("Key", (0.75, -0.65, 1.05), 9500.0, (1.0, 0.88, 0.72), 0.20),
        ("Fill", (-0.55, -0.35, 0.62), 5200.0, (0.55, 0.72, 1.0), 0.18),
        ("Rim", (-0.30, 0.70, 0.92), 7200.0, (0.72, 0.84, 1.0), 0.16),
    ):
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}Light")
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(radius)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.AddTranslateOp().Set(Gf.Vec3d(*position))


def save_rgb(data: np.ndarray, path: Path) -> None:
    Image.fromarray(data).convert("RGB").save(path, quality=95)


def make_contact_sheet(first: Path, second: Path, output: Path) -> None:
    image_a = Image.open(first).convert("RGB")
    image_b = Image.open(second).convert("RGB")
    gutter = 18
    canvas = Image.new(
        "RGB",
        (image_a.width + image_b.width + gutter, max(image_a.height, image_b.height)),
        (12, 15, 21),
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
            position=(1.08, -1.00, 0.73),
            look_at=(0.025, 0.0, 0.285),
            focal_length=48.0,
            clipping_range=(0.03, 100.0),
        ),
        rep.create.camera(
            position=(0.76, -0.68, 1.16),
            look_at=(0.0, 0.0, 0.30),
            focal_length=47.0,
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

    hero = output_dir / "mavic3pro_racer_hero.png"
    top = output_dir / "mavic3pro_racer_top.png"
    preview = output_dir / "mavic3pro_racer_preview.png"
    save_rgb(annotators[0].get_data(), hero)
    save_rgb(annotators[1].get_data(), top)
    make_contact_sheet(hero, top, preview)
    rep.orchestrator.wait_until_complete()
    print(f"hero preview: {hero}")
    print(f"top preview: {top}")
    print(f"contact sheet: {preview}")


try:
    main()
finally:
    APP.close()
