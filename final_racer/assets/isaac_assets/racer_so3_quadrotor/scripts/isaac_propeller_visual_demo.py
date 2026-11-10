#!/usr/bin/env python3
"""Interactive, ROS-free viewer for the Crazyflie/RACER propeller animation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "usd" / "crazyflie_with_racer_dynamics.usd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument(
        "--rpm",
        type=float,
        default=180.0,
        help="slow visual-demo RPM; does not affect vehicle physics",
    )
    return parser.parse_args()


ARGS = parse_args()
if ARGS.rpm <= 0.0:
    raise SystemExit("--rpm must be positive")
ARGS.usd = ARGS.usd.expanduser().resolve()
if not ARGS.usd.is_file():
    raise SystemExit(f"vehicle USD does not exist: {ARGS.usd}")

APP = SimulationApp(
    {
        "headless": False,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 800,
    }
)

import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402


def build_stage():
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    vehicle = UsdGeom.Xform.Define(stage, "/World/Drone")
    vehicle.GetPrim().GetReferences().AddReference(str(ARGS.usd))
    vehicle.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.28))

    platform = UsdGeom.Cylinder.Define(stage, "/World/DisplayPlatform")
    platform.CreateAxisAttr("Z")
    platform.CreateRadiusAttr(0.48)
    platform.CreateHeightAttr(0.04)
    platform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    platform.CreateDisplayColorAttr([Gf.Vec3f(0.055, 0.065, 0.085)])

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(750.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.84, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(2.5)
    key.AddRotateXYZOp().Set(Gf.Vec3f(45.0, -30.0, 25.0))

    set_camera_view(
        eye=(1.05, -1.05, 0.78),
        target=(0.0, 0.0, 0.27),
        camera_prim_path="/OmniverseKit_Persp",
    )
    APP.update()
    return stage


def find_propellers(stage) -> list[dict]:
    visuals = stage.GetPrimAtPath("/World/Drone/base_link/visuals")
    if not visuals.IsValid():
        raise RuntimeError("referenced vehicle has no base_link/visuals")
    propellers = []
    for prim in Usd.PrimRange(visuals):
        rotor_id_attr = prim.GetAttribute("racer:rotorId")
        if not rotor_id_attr.IsValid() or not rotor_id_attr.HasAuthoredValueOpinion():
            continue
        rotor_id = int(rotor_id_attr.Get())
        sign = int(prim.GetAttribute("racer:spinDirectionSign").Get())
        axis = str(prim.GetAttribute("racer:visualRotationAxis").Get()).upper()
        op_name = str(prim.GetAttribute("racer:visualRotationOp").Get())
        rotation_attr = prim.GetAttribute(op_name)
        base = rotation_attr.Get()
        if axis not in {"X", "Y", "Z"} or sign not in (-1, 1) or base is None:
            raise RuntimeError(f"invalid propeller metadata at {prim.GetPath()}")
        propellers.append(
            {
                "rotor_id": rotor_id,
                "path": str(prim.GetPath()),
                "sign": sign,
                "axis_index": {"X": 0, "Y": 1, "Z": 2}[axis],
                "attribute": rotation_attr,
                "base": tuple(float(value) for value in base),
                "vector_type": type(base),
                "angle": 0.0,
            }
        )
    propellers.sort(key=lambda item: item["rotor_id"])
    if [item["rotor_id"] for item in propellers] != [0, 1, 2, 3]:
        raise RuntimeError(
            f"expected visual rotor ids 0..3, found "
            f"{[item['rotor_id'] for item in propellers]}"
        )
    return propellers


def create_overlay(propellers: list[dict]):
    window = ui.Window(
        "RACER Crazyflie Propeller Demo",
        width=390,
        height=150,
        position_x=18,
        position_y=48,
    )
    with window.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("Crazyflie mesh + RACER visual rotors", height=26)
            ui.Label(
                f"Visual-only speed: {ARGS.rpm:.0f} RPM (slow demonstration)",
                height=22,
            )
            ui.Label("Rotor 0/1: CW    Rotor 2/3: CCW", height=22)
            ui.Label("No ROS2, no RACER, no aerodynamic force", height=22)
            ui.Label("Close the Isaac Sim window to stop", height=22)
    print(
        "RACER_PROPELLER_DEMO_READY "
        + json.dumps(
            {
                "usd": str(ARGS.usd),
                "visual_rpm": ARGS.rpm,
                "propellers": [item["path"] for item in propellers],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return window


def main() -> None:
    stage = build_stage()
    propellers = find_propellers(stage)
    overlay = create_overlay(propellers)
    last = time.monotonic()
    try:
        while APP.is_running():
            now = time.monotonic()
            dt = min(max(now - last, 0.0), 0.1)
            last = now
            for item in propellers:
                item["angle"] = (
                    item["angle"] + item["sign"] * ARGS.rpm * 6.0 * dt
                ) % 360.0
                values = list(item["base"])
                values[item["axis_index"]] += item["angle"]
                item["attribute"].Set(item["vector_type"](*values))
            APP.update()
    finally:
        overlay.visible = False
        APP.close()


if __name__ == "__main__":
    main()
