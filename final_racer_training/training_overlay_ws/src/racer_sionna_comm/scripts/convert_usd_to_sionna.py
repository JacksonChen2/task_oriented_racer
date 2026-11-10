#!/usr/bin/env python3
"""Convert a composed USD scene to a Sionna RT Mitsuba XML/PLY bundle.

Run this script with Isaac Sim's ``python.sh`` so that Omniverse HTTP/Nucleus
asset references and payloads can be resolved before the geometry is baked.
All USD transforms and instances are applied to the exported, meter-scale,
Z-up triangle meshes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import struct
import sys
import time
import xml.etree.ElementTree as ET


ITU_MATERIALS = (
    "concrete",
    "metal",
    "wood",
    "chipboard",
    "glass",
    "ceiling_board",
    "plywood",
)

# Rules are ordered: the first matching keyword determines the RF material.
MATERIAL_RULES = (
    ("ceiling_board", ("ceiling", "roof", "acoustic_tile")),
    ("glass", ("glass", "window", "windscreen")),
    (
        "metal",
        (
            "metal",
            "steel",
            "iron",
            "aluminium",
            "aluminum",
            "chrome",
            "rack",
            "shelf",
            "beam",
            "frame",
            "duct",
            "pipe",
            "fence",
            "forklift",
            "conveyor",
            "roller",
        ),
    ),
    ("wood", ("wood", "timber", "pallet", "palette", "crate_wood")),
    (
        "chipboard",
        ("cardboard", "card_box", "cardbox", "carton", "paper", "box"),
    ),
    (
        "concrete",
        (
            "concrete",
            "cement",
            "floor",
            "ground",
            "wall",
            "column",
            "pillar",
            "building",
            "slab",
            "curb",
        ),
    ),
    (
        "plywood",
        (
            "plastic",
            "polymer",
            "rubber",
            "bottle",
            "barrel",
            "traffic",
            "cone",
            "bin",
        ),
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake a USD scene into a Sionna RT XML and PLY meshes."
    )
    parser.add_argument("input_usd", type=Path)
    parser.add_argument(
        "--base-usd",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional composed scene layer loaded before input_usd; repeat "
            "for overlays whose relative base references are unavailable"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output bundle directory (default: <input-stem>_sionna).",
    )
    parser.add_argument(
        "--default-material",
        choices=ITU_MATERIALS,
        default="concrete",
        help="RF material for geometry without a recognizable semantic name.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("oriented_bbox_proxy", "triangle_mesh"),
        default="oriented_bbox_proxy",
        help=(
            "RF geometry representation. The proxy mode preserves each mesh's "
            "world transform and local bounds while keeping Sionna tractable."
        ),
    )
    parser.add_argument(
        "--load-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for remote USD dependencies.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Load and summarize the USD without writing an output bundle.",
    )
    return parser.parse_args()


ARGS = parse_arguments()
INPUT_USD = ARGS.input_usd.expanduser().resolve()
if not INPUT_USD.is_file():
    raise SystemExit(f"USD file does not exist: {INPUT_USD}")
BASE_USDS = [path.expanduser().resolve() for path in ARGS.base_usd]
for base_usd in BASE_USDS:
    if not base_usd.is_file():
        raise SystemExit(f"base USD file does not exist: {base_usd}")
if ARGS.load_timeout <= 0.0:
    raise SystemExit("--load-timeout must be positive")

OUTPUT_DIR = (
    ARGS.output_dir.expanduser().resolve()
    if ARGS.output_dir
    else INPUT_USD.with_name(INPUT_USD.stem + "_sionna")
)

# Isaac/Kit must be initialized before importing omni.usd or pxr.
from isaacsim import SimulationApp  # noqa: E402


simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdShade, UsdUtils  # noqa: E402


def load_composed_stage():
    """Reference the input in an Isaac stage and wait for every payload."""
    metadata_stage = Usd.Stage.Open(str(INPUT_USD), load=Usd.Stage.LoadNone)
    if metadata_stage is None:
        raise RuntimeError(f"could not open USD metadata: {INPUT_USD}")
    source_up_axis = UsdGeom.GetStageUpAxis(metadata_stage)
    source_meters_per_unit = UsdGeom.GetStageMetersPerUnit(metadata_stage)
    if not math.isfinite(source_meters_per_unit) or source_meters_per_unit <= 0.0:
        raise RuntimeError(
            f"invalid USD metersPerUnit: {source_meters_per_unit!r}"
        )

    context = omni.usd.get_context()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, source_up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, source_meters_per_unit)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for index, base_usd in enumerate(BASE_USDS):
        base = UsdGeom.Xform.Define(stage, f"/World/SionnaBase_{index}")
        if not base.GetPrim().GetReferences().AddReference(str(base_usd)):
            raise RuntimeError(f"failed to reference base USD: {base_usd}")
    source = UsdGeom.Xform.Define(stage, "/World/SionnaSource")
    if not source.GetPrim().GetReferences().AddReference(str(INPUT_USD)):
        raise RuntimeError(f"failed to reference USD: {INPUT_USD}")
    stage.Load()

    deadline = time.monotonic() + ARGS.load_timeout
    stable_frames = 0
    while True:
        simulation_app.update()
        _, _, loading = context.get_stage_loading_status()
        if loading == 0:
            stable_frames += 1
            if stable_frames >= 5:
                break
        else:
            stable_frames = 0
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"USD dependencies did not finish loading in "
                f"{ARGS.load_timeout:.1f}s"
            )
    return (
        stage,
        world.GetPrim(),
        str(source_up_axis),
        float(source_meters_per_unit),
    )


def z_up_conversion(up_axis: str) -> np.ndarray:
    """Return a row-vector rotation from the USD up axis to right-handed Z-up."""
    if up_axis.upper() == "Z":
        return np.eye(4, dtype=np.float64)
    if up_axis.upper() == "Y":
        # +90 degrees about X: (x, y, z) -> (x, -z, y)
        return np.asarray(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
    if up_axis.upper() == "X":
        # -90 degrees about Y: (x, y, z) -> (-z, y, x)
        return np.asarray(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
    raise RuntimeError(f"unsupported USD up axis: {up_axis!r}")


def bound_material_path(prim) -> str:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material and material.GetPrim().IsValid():
        return str(material.GetPrim().GetPath())
    return ""


def classify_material(prim_path: str, usd_material_path: str) -> str:
    text = (prim_path + " " + usd_material_path).lower()
    for material, keywords in MATERIAL_RULES:
        if any(keyword in text for keyword in keywords):
            return material
    return ARGS.default_material


def visible_render_mesh(prim) -> bool:
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return False
    if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return False
    return imageable.ComputePurpose() != UsdGeom.Tokens.guide


def matrix_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.asarray(
        [[matrix[row][column] for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def face_materials(mesh, face_count: int, fallback_path: str) -> list[str]:
    result = [fallback_path] * face_count
    for subset in UsdGeom.Subset.GetGeomSubsets(mesh):
        subset_material = bound_material_path(subset.GetPrim()) or fallback_path
        indices = subset.GetIndicesAttr().Get(Usd.TimeCode.Default()) or ()
        for face_index in indices:
            face_index = int(face_index)
            if 0 <= face_index < face_count:
                result[face_index] = subset_material
    return result


def append_mesh(
    groups,
    mesh,
    xform_cache,
    axis_transform: np.ndarray,
    meters_per_unit: float,
    stats,
    material_name_counts,
):
    prim = mesh.GetPrim()
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if not points or not counts or not indices:
        stats["empty_meshes"] += 1
        return

    points_np = np.asarray(points, dtype=np.float64)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        stats["invalid_meshes"] += 1
        return
    counts_np = np.asarray(counts, dtype=np.int64)
    indices_np = np.asarray(indices, dtype=np.int64)
    if int(counts_np.sum()) != len(indices_np):
        stats["invalid_meshes"] += 1
        return

    world_matrix = matrix_numpy(xform_cache.GetLocalToWorldTransform(prim))
    homogeneous = np.ones((len(points_np), 4), dtype=np.float64)
    homogeneous[:, :3] = points_np
    world_points = homogeneous @ world_matrix
    world_points = world_points @ axis_transform
    world_points = np.asarray(
        world_points[:, :3] * meters_per_unit, dtype="<f4"
    )

    transform_flips = np.linalg.det(world_matrix[:3, :3]) < 0.0
    orientation = mesh.GetOrientationAttr().Get(Usd.TimeCode.Default())
    left_handed = orientation == UsdGeom.Tokens.leftHanded
    reverse_winding = bool(transform_flips) ^ bool(left_handed)

    fallback_path = bound_material_path(prim)
    per_face_material = face_materials(mesh, len(counts_np), fallback_path)
    for material_path in set(per_face_material):
        material_name_counts[
            Path(material_path).name if material_path else "<unbound>"
        ] += 1

    holes = {
        int(value)
        for value in (
            mesh.GetHoleIndicesAttr().Get(Usd.TimeCode.Default()) or ()
        )
    }
    triangles = defaultdict(list)
    cursor = 0
    for face_index, count_value in enumerate(counts_np):
        count = int(count_value)
        face = indices_np[cursor:cursor + count]
        cursor += count
        if face_index in holes or count < 3:
            continue
        if np.any(face < 0) or np.any(face >= len(points_np)):
            stats["invalid_faces"] += 1
            continue
        rf_material = classify_material(
            str(prim.GetPath()), per_face_material[face_index]
        )
        for offset in range(1, count - 1):
            triangle = (int(face[0]), int(face[offset]), int(face[offset + 1]))
            if reverse_winding:
                triangle = (triangle[0], triangle[2], triangle[1])
            triangles[rf_material].append(triangle)

    for rf_material, local_triangles in triangles.items():
        if not local_triangles:
            continue
        group = groups[rf_material]
        base_vertex = group["vertex_count"]
        group["vertices"].extend(world_points.tobytes(order="C"))
        group["vertex_count"] += len(world_points)

        triangle_np = np.asarray(local_triangles, dtype=np.uint32)
        triangle_np += np.uint32(base_vertex)
        records = np.empty(
            len(triangle_np),
            dtype=np.dtype(
                [("count", "u1"), ("vertices", "<u4", (3,))],
                align=False,
            ),
        )
        records["count"] = 3
        records["vertices"] = triangle_np
        group["faces"].extend(records.tobytes(order="C"))
        group["face_count"] += len(triangle_np)
        group["source_meshes"] += 1
        local_min = world_points.min(axis=0)
        local_max = world_points.max(axis=0)
        group["bounds_min"] = np.minimum(group["bounds_min"], local_min)
        group["bounds_max"] = np.maximum(group["bounds_max"], local_max)

    stats["source_vertices"] += len(points_np)
    stats["source_faces"] += len(counts_np)
    stats["triangles"] += sum(len(value) for value in triangles.values())


def append_oriented_bbox_proxy(
    groups,
    mesh,
    xform_cache,
    axis_transform: np.ndarray,
    meters_per_unit: float,
    stats,
    material_name_counts,
):
    """Append a 12-triangle proxy using the mesh's transformed local extent."""
    prim = mesh.GetPrim()
    extent = mesh.GetExtentAttr().Get(Usd.TimeCode.Default())
    if extent is not None and len(extent) == 2:
        local_min = np.asarray(extent[0], dtype=np.float64)
        local_max = np.asarray(extent[1], dtype=np.float64)
    else:
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        if not points:
            stats["empty_meshes"] += 1
            return
        points_np = np.asarray(points, dtype=np.float64)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            stats["invalid_meshes"] += 1
            return
        local_min = points_np.min(axis=0)
        local_max = points_np.max(axis=0)
        stats["proxy_extent_fallbacks"] += 1
    if not np.all(np.isfinite(local_min)) or not np.all(np.isfinite(local_max)):
        stats["invalid_meshes"] += 1
        return

    x0, y0, z0 = local_min
    x1, y1, z1 = local_max
    corners = np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    world_matrix = matrix_numpy(xform_cache.GetLocalToWorldTransform(prim))
    homogeneous = np.ones((8, 4), dtype=np.float64)
    homogeneous[:, :3] = corners
    world_points = homogeneous @ world_matrix
    world_points = world_points @ axis_transform
    world_points = np.asarray(
        world_points[:, :3] * meters_per_unit, dtype="<f4"
    )

    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    orientation = mesh.GetOrientationAttr().Get(Usd.TimeCode.Default())
    if (np.linalg.det(world_matrix[:3, :3]) < 0.0) ^ (
        orientation == UsdGeom.Tokens.leftHanded
    ):
        triangles[:, [1, 2]] = triangles[:, [2, 1]]

    usd_material_path = bound_material_path(prim)
    material_name_counts[
        Path(usd_material_path).name if usd_material_path else "<unbound>"
    ] += 1
    rf_material = classify_material(str(prim.GetPath()), usd_material_path)
    group = groups[rf_material]
    base_vertex = group["vertex_count"]
    group["vertices"].extend(world_points.tobytes(order="C"))
    group["vertex_count"] += 8
    triangles += np.uint32(base_vertex)
    records = np.empty(
        12,
        dtype=np.dtype(
            [("count", "u1"), ("vertices", "<u4", (3,))], align=False
        ),
    )
    records["count"] = 3
    records["vertices"] = triangles
    group["faces"].extend(records.tobytes(order="C"))
    group["face_count"] += 12
    group["source_meshes"] += 1
    group["bounds_min"] = np.minimum(
        group["bounds_min"], world_points.min(axis=0)
    )
    group["bounds_max"] = np.maximum(
        group["bounds_max"], world_points.max(axis=0)
    )
    stats["triangles"] += 12


def new_group():
    return {
        "vertices": bytearray(),
        "faces": bytearray(),
        "vertex_count": 0,
        "face_count": 0,
        "source_meshes": 0,
        "bounds_min": np.full(3, np.inf, dtype=np.float64),
        "bounds_max": np.full(3, -np.inf, dtype=np.float64),
    }


def write_binary_ply(path: Path, group) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated from USD for Sionna RT\n"
        f"element vertex {group['vertex_count']}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {group['face_count']}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(group["vertices"])
        stream.write(group["faces"])


def write_xml(path: Path, materials: list[str]) -> None:
    root = ET.Element("scene", {"version": "2.1.0"})
    root.append(
        ET.Comment(
            " Geometry baked from Isaac USD; RF materials are semantic "
            "approximations. "
        )
    )
    for material in materials:
        bsdf = ET.SubElement(
            root, "bsdf", {"type": "itu-radio-material", "id": material}
        )
        ET.SubElement(
            bsdf, "string", {"name": "type", "value": material}
        )
        ET.SubElement(bsdf, "float", {"name": "thickness", "value": "0.1"})
    for material in materials:
        shape_id = f"mesh-warehouse-{material}"
        shape = ET.SubElement(
            root,
            "shape",
            {"type": "ply", "id": shape_id, "name": shape_id},
        )
        ET.SubElement(
            shape,
            "string",
            {
                "name": "filename",
                "value": f"meshes/warehouse_{material}.ply",
            },
        )
        ET.SubElement(
            shape, "boolean", {"name": "face_normals", "value": "true"}
        )
        ET.SubElement(shape, "ref", {"id": material, "name": "bsdf"})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        path, encoding="utf-8", xml_declaration=True
    )


def main() -> None:
    stage, source_prim, up_axis, meters_per_unit = load_composed_stage()
    axis_transform = z_up_conversion(up_axis)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    groups = defaultdict(new_group)
    stats = Counter()
    prim_type_counts = Counter()
    material_name_counts = Counter()
    point_instancers = []

    predicate = Usd.TraverseInstanceProxies()
    for prim in Usd.PrimRange(source_prim, predicate):
        prim_type_counts[prim.GetTypeName() or "<untyped>"] += 1
        if prim.IsA(UsdGeom.PointInstancer):
            point_instancers.append(str(prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        stats["mesh_prims"] += 1
        if not visible_render_mesh(prim):
            stats["skipped_invisible_or_guide"] += 1
            continue
        mesh = UsdGeom.Mesh(prim)
        if ARGS.geometry_mode == "triangle_mesh":
            append_mesh(
                groups,
                mesh,
                xform_cache,
                axis_transform,
                meters_per_unit,
                stats,
                material_name_counts,
            )
        else:
            append_oriented_bbox_proxy(
                groups,
                mesh,
                xform_cache,
                axis_transform,
                meters_per_unit,
                stats,
                material_name_counts,
            )
        stats["exported_mesh_prims"] += 1

    if point_instancers:
        raise RuntimeError(
            "PointInstancer expansion is not implemented; found: "
            + ", ".join(point_instancers[:10])
        )
    if not groups:
        raise RuntimeError("no visible USD triangle meshes were found")

    overall_min = np.min(
        [group["bounds_min"] for group in groups.values()], axis=0
    )
    overall_max = np.max(
        [group["bounds_max"] for group in groups.values()], axis=0
    )
    materials = [
        material for material in ITU_MATERIALS if material in groups
    ]
    group_report = {}
    for material in materials:
        group = groups[material]
        group_report[material] = {
            "source_meshes": group["source_meshes"],
            "vertices": group["vertex_count"],
            "triangles": group["face_count"],
            "bounds_min_m": group["bounds_min"].tolist(),
            "bounds_max_m": group["bounds_max"].tolist(),
        }

    _, dependencies, unresolved = UsdUtils.ComputeAllDependencies(
        str(INPUT_USD)
    )
    report = {
        "input_usd": str(INPUT_USD),
        "base_usds": [str(path) for path in BASE_USDS],
        "geometry_mode": ARGS.geometry_mode,
        "output_xml": str(OUTPUT_DIR / "warehouse.xml"),
        "source_up_axis": up_axis,
        "source_meters_per_unit": meters_per_unit,
        "output_up_axis": "Z",
        "output_units": "meter",
        "bounds_min_m": overall_min.tolist(),
        "bounds_max_m": overall_max.tolist(),
        "prim_count": int(sum(prim_type_counts.values())),
        "mesh_prim_count": int(stats["mesh_prims"]),
        "exported_mesh_prim_count": int(stats["exported_mesh_prims"]),
        "skipped_invisible_or_guide": int(
            stats["skipped_invisible_or_guide"]
        ),
        "source_vertex_count": int(stats["source_vertices"]),
        "source_face_count": int(stats["source_faces"]),
        "triangle_count": int(stats["triangles"]),
        "asset_dependency_count": len(dependencies),
        "unresolved_dependencies": [str(item) for item in unresolved],
        "rf_material_groups": group_report,
        "top_usd_material_names": material_name_counts.most_common(40),
        "rf_material_note": (
            "USD visual materials were heuristically mapped to Sionna ITU "
            "radio materials. Plastic/rubber is approximated as plywood; "
            "unrecognized geometry uses " + ARGS.default_material + "."
        ),
    }

    if ARGS.inspect_only:
        print("SIONNA_USD_INSPECTION " + json.dumps(report, sort_keys=True))
        return

    meshes_dir = OUTPUT_DIR / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    for material in materials:
        path = meshes_dir / f"warehouse_{material}.ply"
        write_binary_ply(path, groups[material])
        group_report[material]["file"] = str(path)
        group_report[material]["file_size_bytes"] = path.stat().st_size
    write_xml(OUTPUT_DIR / "warehouse.xml", materials)
    with (OUTPUT_DIR / "conversion_report.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print("SIONNA_USD_CONVERSION " + json.dumps(report, sort_keys=True))


try:
    main()
finally:
    simulation_app.close()
