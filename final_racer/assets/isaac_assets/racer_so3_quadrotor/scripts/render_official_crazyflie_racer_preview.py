#!/usr/bin/env python3
"""Render hero and top views of the official-visual RACER Crazyflie USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "usd" / "crazyflie_official_visual_with_racer_dynamics.usd"
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
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


def material(stage, path: str, color: Gf.Vec3f, roughness: float):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def build_stage(asset_path: Path) -> None:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    drone = UsdGeom.Xform.Define(stage, "/World/Drone")
    drone.GetPrim().GetReferences().AddReference(str(asset_path))
    drone.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.31))

    floor_mat = material(
        stage, "/World/Looks/Floor", Gf.Vec3f(0.025, 0.032, 0.045), 0.27
    )
    floor = UsdGeom.Cylinder.Define(stage, "/World/StudioFloor")
    floor.CreateAxisAttr(UsdGeom.Tokens.z)
    floor.CreateRadiusAttr(1.12)
    floor.CreateHeightAttr(0.035)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_mat)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(520.0)
    dome.CreateColorAttr(Gf.Vec3f(0.76, 0.83, 1.0))
    for name, position, intensity, color, radius in (
        ("Key", (0.72, -0.68, 1.00), 9000.0, (1.0, 0.88, 0.72), 0.20),
        ("Fill", (-0.58, -0.38, 0.68), 5600.0, (0.56, 0.73, 1.0), 0.18),
        ("Rim", (-0.28, 0.68, 0.90), 7000.0, (0.74, 0.84, 1.0), 0.16),
    ):
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}Light")
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(radius)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.AddTranslateOp().Set(Gf.Vec3d(*position))


def save_rgb(data: np.ndarray, path: Path, label: str) -> None:
    image = Image.fromarray(data).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((22, 20, 212, 58), radius=8, fill=(7, 10, 16))
    draw.text((38, 31), label, fill=(230, 238, 250))
    image.save(path, quality=95)


def contact_sheet(first: Path, second: Path, output: Path) -> None:
    a = Image.open(first).convert("RGB")
    b = Image.open(second).convert("RGB")
    gutter = 18
    canvas = Image.new("RGB", (a.width + b.width + gutter, max(a.height, b.height)), (10, 13, 19))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gutter, 0))
    canvas.save(output, quality=95)


def main() -> None:
    asset_path = ARGS.usd.resolve()
    output_dir = ARGS.output_dir.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_stage(asset_path)

    settings = carb.settings.get_settings()
    settings.set("/omni/replicator/captureOnPlay", False)
    settings.set("rtx/post/dlss/execMode", 2)
    settings.set("rtx/post/aa/op", 3)

    cameras = (
        rep.create.camera(
            position=(1.38, -1.32, 0.91),
            look_at=(0.0, 0.0, 0.34),
            focal_length=48.0,
            clipping_range=(0.03, 100.0),
        ),
        rep.create.camera(
            position=(0.02, -0.02, 2.18),
            look_at=(0.0, 0.0, 0.32),
            focal_length=46.0,
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

    hero = output_dir / "crazyflie_official_racer_hero.png"
    top = output_dir / "crazyflie_official_racer_top.png"
    preview = output_dir / "crazyflie_official_racer_preview.png"
    save_rgb(annotators[0].get_data(), hero, "3/4 VIEW")
    save_rgb(annotators[1].get_data(), top, "TOP VIEW")
    contact_sheet(hero, top, preview)
    rep.orchestrator.wait_until_complete()
    print(f"hero preview: {hero}")
    print(f"top preview: {top}")
    print(f"contact sheet: {preview}")


try:
    main()
finally:
    APP.close()
