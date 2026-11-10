"""GPU-batched pinhole ray camera for the RACER Isaac Sim bridge.

The implementation follows the same static-mesh approach used by Isaac Lab's
RayCasterCamera: renderable USD geometry is baked into one world-frame Warp
mesh and every camera ray is queried by one CUDA kernel.  Dynamic RACER
vehicles are deliberately excluded, matching the headless RTX setup where the
vehicle visuals are hidden and preventing self returns.

This module must be imported after ``SimulationApp`` has started and the
``omni.warp.core`` extension has been enabled.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
from pxr import Usd, UsdGeom
import warp as wp


wp.init()


@wp.kernel
def _raycast_pinhole_kernel(
    mesh_id: wp.uint64,
    camera_origins: wp.array(dtype=wp.vec3),
    camera_rotations: wp.array(dtype=wp.mat33),
    local_directions: wp.array(dtype=wp.vec3),
    forward_components: wp.array(dtype=wp.float32),
    ray_count: wp.int32,
    near_depth: wp.float32,
    map_depth: wp.float32,
    render_depth: wp.float32,
    points_world: wp.array(dtype=wp.vec3),
    hit_mask: wp.array(dtype=wp.uint8),
    hit_distances: wp.array(dtype=wp.float32),
):
    """Cast all UAV/pixel pairs without a Python loop over rays.

    Depth limits are optical-axis depths, as returned by the old RTX
    ``distance_to_image_plane`` annotator. Warp returns Euclidean ray distance,
    so the per-ray forward component converts between the two conventions.
    """

    index = wp.tid()
    camera_index = index // ray_count
    ray_index = index - camera_index * ray_count
    origin = camera_origins[camera_index]
    direction = wp.normalize(
        camera_rotations[camera_index] * local_directions[ray_index]
    )
    forward = forward_components[ray_index]
    # Begin at the original camera near plane.  This reproduces RTX clipping
    # semantics and prevents geometry inside the blind zone from blocking a
    # valid surface farther along the same ray.
    start_distance = near_depth / forward
    query_origin = origin + direction * start_distance
    query_limit = (render_depth - near_depth) / forward
    query = wp.mesh_query_ray(
        mesh_id, query_origin, direction, query_limit
    )
    hit_distance = render_depth / forward
    if query.result:
        hit_distance = start_distance + query.t

    if (
        query.result
        and hit_distance * forward <= map_depth
    ):
        points_world[index] = origin + direction * hit_distance
        hit_mask[index] = wp.uint8(1)
        hit_distances[index] = hit_distance
    else:
        # RACER expects a max-range endpoint even for a missed ray so its
        # inverse sensor model can clear FREE space along the complete ray.
        miss_distance = map_depth / forward
        points_world[index] = origin + direction * miss_distance
        hit_mask[index] = wp.uint8(0)
        hit_distances[index] = -1.0


@wp.kernel
def _raycast_safety_kernel(
    mesh_id: wp.uint64,
    sensor_origins: wp.array(dtype=wp.vec3),
    sensor_rotations: wp.array(dtype=wp.mat33),
    local_directions: wp.array(dtype=wp.vec3),
    ray_count: wp.int32,
    near_range: wp.float32,
    max_range: wp.float32,
    voxel_origin: wp.vec3,
    voxel_size: wp.float32,
    voxel_dims: wp.vec3i,
    voxels_per_scene: wp.int32,
    miss_key: wp.int32,
    points_world: wp.array(dtype=wp.vec3),
    hit_distances: wp.array(dtype=wp.float32),
    voxel_keys: wp.array(dtype=wp.int32),
    ray_indices: wp.array(dtype=wp.int32),
):
    """Cast all low-density 360-degree safety rays and assign voxel keys.

    Range filtering happens before a point can enter the sort/compaction
    pipeline.  The key includes the UAV index, so one batched radix sort can
    downsample every vehicle without ever mixing their safety observations.
    """

    index = wp.tid()
    sensor_index = index // ray_count
    ray_index = index - sensor_index * ray_count
    origin = sensor_origins[sensor_index]
    direction = wp.normalize(
        sensor_rotations[sensor_index] * local_directions[ray_index]
    )
    query_origin = origin + direction * near_range
    query = wp.mesh_query_ray(
        mesh_id, query_origin, direction, max_range - near_range
    )
    ray_indices[index] = wp.int32(index)
    voxel_keys[index] = miss_key
    hit_distances[index] = -1.0
    if not query.result:
        return

    distance = near_range + query.t
    if distance > max_range:
        return
    point = origin + direction * distance
    ix = wp.int32(wp.floor((point[0] - voxel_origin[0]) / voxel_size))
    iy = wp.int32(wp.floor((point[1] - voxel_origin[1]) / voxel_size))
    iz = wp.int32(wp.floor((point[2] - voxel_origin[2]) / voxel_size))
    if (
        ix < 0
        or iy < 0
        or iz < 0
        or ix >= voxel_dims[0]
        or iy >= voxel_dims[1]
        or iz >= voxel_dims[2]
    ):
        return

    local_key = (ix * voxel_dims[1] + iy) * voxel_dims[2] + iz
    voxel_keys[index] = sensor_index * voxels_per_scene + local_key
    points_world[index] = point
    hit_distances[index] = distance


@wp.kernel
def _compact_safety_voxels_kernel(
    run_count: wp.array(dtype=wp.int32),
    unique_keys: wp.array(dtype=wp.int32),
    run_offsets: wp.array(dtype=wp.int32),
    sorted_ray_indices: wp.array(dtype=wp.int32),
    raw_points: wp.array(dtype=wp.vec3),
    raw_distances: wp.array(dtype=wp.float32),
    ray_count: wp.int32,
    miss_key: wp.int32,
    max_points_per_sensor: wp.int32,
    output_counts: wp.array(dtype=wp.int32),
    output_points: wp.array(dtype=wp.vec3),
    output_distances: wp.array(dtype=wp.float32),
):
    """Keep one representative point from every occupied safety voxel."""

    run_index = wp.tid()
    if run_index >= run_count[0]:
        return
    if unique_keys[run_index] == miss_key:
        return
    source_index = sorted_ray_indices[run_offsets[run_index]]
    sensor_index = source_index // ray_count
    output_index = wp.atomic_add(output_counts, sensor_index, 1)
    if output_index >= max_points_per_sensor:
        return
    destination = sensor_index * max_points_per_sensor + output_index
    output_points[destination] = raw_points[source_index]
    output_distances[destination] = raw_distances[source_index]


def _stage_prims(stage) -> Iterable:
    """Traverse loaded payloads and instance proxies when USD supports it."""

    try:
        return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    except Exception:  # pragma: no cover - compatibility with older USD
        return stage.Traverse()


def _world_points(points, matrix) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    homogeneous = np.ones((len(values), 4), dtype=np.float64)
    homogeneous[:, :3] = values
    # Gf matrices use row-vector transform convention; translation occupies
    # the final row, so this multiplication matches Gf.Matrix4d.Transform().
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64)
    non_unit = np.abs(transformed[:, 3]) > 1.0e-12
    transformed[non_unit, :3] /= transformed[non_unit, 3:4]
    return np.ascontiguousarray(transformed[:, :3], dtype=np.float32)


def _triangulate_mesh(prim, xform_cache):
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if (
        points is None
        or counts is None
        or indices is None
        or len(points) == 0
        or len(counts) == 0
        or len(indices) == 0
    ):
        return None

    world = _world_points(
        points, xform_cache.GetLocalToWorldTransform(prim)
    )
    face_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    face_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    triangles = []
    cursor = 0
    for count in face_counts:
        count = int(count)
        face = face_indices[cursor:cursor + count]
        cursor += count
        if count < 3:
            continue
        anchor = int(face[0])
        for offset in range(1, count - 1):
            triangles.extend(
                (anchor, int(face[offset]), int(face[offset + 1]))
            )
    if not triangles:
        return None
    return world, np.asarray(triangles, dtype=np.int32)


_CUBE_TRIANGLES = np.asarray(
    [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7,
    ],
    dtype=np.int32,
)


def _triangulate_cube(prim, xform_cache):
    cube = UsdGeom.Cube(prim)
    size = cube.GetSizeAttr().Get(Usd.TimeCode.Default())
    half = 0.5 * float(size if size is not None else 2.0)
    points = np.asarray(
        [
            (-half, -half, -half),
            (half, -half, -half),
            (half, half, -half),
            (-half, half, -half),
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        ],
        dtype=np.float32,
    )
    return (
        _world_points(points, xform_cache.GetLocalToWorldTransform(prim)),
        _CUBE_TRIANGLES.copy(),
    )


def _is_renderable(prim) -> bool:
    imageable = UsdGeom.Imageable(prim)
    if not imageable or not imageable.GetPrim().IsValid():
        return True
    try:
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            return False
        # Hydra depth products render default/render purpose geometry, not
        # collision/proxy/guide representations. Match that selection so the
        # ray caster does not introduce obstacles absent from the RTX image.
        return imageable.ComputePurpose() in (
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
        )
    except Exception:
        return True


class WarpRayCasterCameraBatch:
    """Static-scene, multi-camera Warp ray caster with one launch per frame."""

    def __init__(
        self,
        stage,
        *,
        camera_count: int,
        image_width: int,
        image_height: int,
        sample_rows: np.ndarray,
        sample_cols: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near_depth: float,
        map_depth: float,
        render_depth: float,
        mount_translation: Sequence[float],
        safety_horizontal_resolution_deg: float | None = None,
        safety_vertical_resolution_deg: float | None = None,
        safety_near_range: float | None = None,
        safety_max_range: float | None = None,
        safety_voxel_size: float | None = None,
        safety_mount_translation: Sequence[float] | None = None,
        include_prefixes: Sequence[str] = (
            "/World/ExternalScene",
            "/World/Obstacles",
        ),
        exclude_prefixes: Sequence[str] = ("/World/Drones",),
        device: str = "cuda:0",
    ) -> None:
        if camera_count <= 0:
            raise ValueError("camera_count must be positive")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("pinhole focal lengths must be positive")
        if near_depth <= 0.0 or not near_depth < map_depth <= render_depth:
            raise ValueError("expected 0 < near_depth < map_depth <= render_depth")
        self.camera_count = int(camera_count)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.device = wp.get_device(device)
        self.near_depth = float(near_depth)
        self.map_depth = float(map_depth)
        self.render_depth = float(render_depth)
        self.mount_translation = np.asarray(
            mount_translation, dtype=np.float32
        ).reshape(3)

        rows = np.asarray(sample_rows, dtype=np.float32).reshape(-1)
        cols = np.asarray(sample_cols, dtype=np.float32).reshape(-1)
        if len(rows) == 0 or len(rows) != len(cols):
            raise ValueError("sample row/column arrays must be non-empty and equal")
        self.sample_pixel_bounds = {
            "row_min": float(np.min(rows)),
            "row_max": float(np.max(rows)),
            "column_min": float(np.min(cols)),
            "column_max": float(np.max(cols)),
        }
        self.horizontal_fov_rad = float(
            np.arctan2(self.cx, self.fx)
            + np.arctan2(self.image_width - 1.0 - self.cx, self.fx)
        )
        self.vertical_fov_rad = float(
            np.arctan2(self.cy, self.fy)
            + np.arctan2(self.image_height - 1.0 - self.cy, self.fy)
        )
        self.sampled_horizontal_fov_rad = float(
            np.arctan2(
                self.cx - self.sample_pixel_bounds["column_min"], self.fx
            )
            + np.arctan2(
                self.sample_pixel_bounds["column_max"] - self.cx, self.fx
            )
        )
        self.sampled_vertical_fov_rad = float(
            np.arctan2(
                self.cy - self.sample_pixel_bounds["row_min"], self.fy
            )
            + np.arctan2(
                self.sample_pixel_bounds["row_max"] - self.cy, self.fy
            )
        )
        optical_x = (cols - float(cx)) / float(fx)
        optical_y = (rows - float(cy)) / float(fy)
        # Optical [+X right,+Y down,+Z forward] to body FLU [Z,-X,-Y].
        local = np.column_stack(
            (np.ones_like(optical_x), -optical_x, -optical_y)
        ).astype(np.float32)
        norms = np.linalg.norm(local, axis=1, keepdims=True)
        local /= norms
        forward = np.ascontiguousarray(local[:, 0], dtype=np.float32)
        self.ray_count = len(local)
        self.local_directions = wp.array(
            np.ascontiguousarray(local), dtype=wp.vec3, device=self.device
        )
        self.forward_components = wp.array(
            forward, dtype=wp.float32, device=self.device
        )

        build_started = time.perf_counter()
        vertices, indices, prim_count = self._build_static_mesh(
            stage, include_prefixes, exclude_prefixes
        )
        self.mesh_points = wp.array(
            vertices, dtype=wp.vec3, device=self.device
        )
        self.mesh_indices = wp.array(
            indices, dtype=wp.int32, device=self.device
        )
        self.mesh = wp.Mesh(
            points=self.mesh_points,
            indices=self.mesh_indices,
            support_winding_number=False,
        )
        wp.synchronize_device(self.device)
        self.mesh_report = {
            "geometry_prims": prim_count,
            "vertices": int(len(vertices)),
            "triangles": int(len(indices) // 3),
            "build_ms": 1000.0 * (time.perf_counter() - build_started),
            "device": str(self.device),
        }

        total_rays = self.camera_count * self.ray_count
        self.points_world = wp.zeros(
            total_rays, dtype=wp.vec3, device=self.device
        )
        self.hit_mask = wp.zeros(
            total_rays, dtype=wp.uint8, device=self.device
        )
        self.hit_distances = wp.zeros(
            total_rays, dtype=wp.float32, device=self.device
        )
        # Pose buffers are reused every frame. Only camera_count origins and
        # rotations cross PCIe; all per-ray data remain resident on the GPU.
        self.camera_origins = wp.zeros(
            self.camera_count, dtype=wp.vec3, device=self.device
        )
        self.camera_rotations = wp.zeros(
            self.camera_count, dtype=wp.mat33, device=self.device
        )

        safety_parameters = (
            safety_horizontal_resolution_deg,
            safety_vertical_resolution_deg,
            safety_near_range,
            safety_max_range,
            safety_voxel_size,
            safety_mount_translation,
        )
        self.safety_enabled = any(value is not None for value in safety_parameters)
        if self.safety_enabled and not all(
            value is not None for value in safety_parameters
        ):
            raise ValueError(
                "all safety ray parameters must be provided together"
            )
        self.safety_report = None
        if self.safety_enabled:
            self._initialize_safety_rays(
                vertices,
                horizontal_resolution_deg=float(
                    safety_horizontal_resolution_deg
                ),
                vertical_resolution_deg=float(
                    safety_vertical_resolution_deg
                ),
                near_range=float(safety_near_range),
                max_range=float(safety_max_range),
                voxel_size=float(safety_voxel_size),
                mount_translation=safety_mount_translation,
            )

    def _initialize_safety_rays(
        self,
        scene_vertices: np.ndarray,
        *,
        horizontal_resolution_deg: float,
        vertical_resolution_deg: float,
        near_range: float,
        max_range: float,
        voxel_size: float,
        mount_translation: Sequence[float],
    ) -> None:
        if (
            horizontal_resolution_deg <= 0.0
            or vertical_resolution_deg <= 0.0
            or near_range <= 0.0
            or max_range <= near_range
            or voxel_size <= 0.0
        ):
            raise ValueError("invalid 360-degree safety ray configuration")

        azimuth = np.deg2rad(
            np.arange(0.0, 360.0, horizontal_resolution_deg)
        )
        # Pixel/ray centres avoid repeating every azimuth at the two poles.
        elevation = np.deg2rad(
            np.arange(
                -90.0 + 0.5 * vertical_resolution_deg,
                90.0,
                vertical_resolution_deg,
            )
        )
        azimuth_grid, elevation_grid = np.meshgrid(
            azimuth, elevation, indexing="xy"
        )
        cos_elevation = np.cos(elevation_grid)
        directions = np.column_stack(
            (
                (cos_elevation * np.cos(azimuth_grid)).reshape(-1),
                (cos_elevation * np.sin(azimuth_grid)).reshape(-1),
                np.sin(elevation_grid).reshape(-1),
            )
        ).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        safety_origin = (
            np.floor(
                np.min(scene_vertices, axis=0).astype(np.float64)
                / voxel_size
            )
            * voxel_size
            - voxel_size
        ).astype(np.float32)
        safety_maximum = (
            np.ceil(
                np.max(scene_vertices, axis=0).astype(np.float64)
                / voxel_size
            )
            * voxel_size
            + voxel_size
        )
        safety_dims = np.ceil(
            (safety_maximum - safety_origin) / voxel_size
        ).astype(np.int64) + 1
        voxels_per_scene = int(np.prod(safety_dims, dtype=np.int64))
        miss_key = np.iinfo(np.int32).max
        if voxels_per_scene * self.camera_count >= miss_key:
            raise ValueError(
                "safety voxel address space exceeds signed 32-bit radix key "
                f"capacity: dims={safety_dims.tolist()} cameras={self.camera_count}"
            )

        self.safety_horizontal_resolution_deg = horizontal_resolution_deg
        self.safety_vertical_resolution_deg = vertical_resolution_deg
        self.safety_near_range = near_range
        self.safety_max_range = max_range
        self.safety_voxel_size = voxel_size
        self.safety_mount_translation = np.asarray(
            mount_translation, dtype=np.float32
        ).reshape(3)
        self.safety_ray_count = int(len(directions))
        self.safety_total_rays = self.camera_count * self.safety_ray_count
        self.safety_voxel_origin = safety_origin
        self.safety_voxel_dims = safety_dims.astype(np.int32)
        self.safety_voxels_per_scene = voxels_per_scene
        self.safety_miss_key = int(miss_key)
        self.safety_local_directions = wp.array(
            np.ascontiguousarray(directions),
            dtype=wp.vec3,
            device=self.device,
        )
        self.safety_sensor_origins = wp.zeros(
            self.camera_count, dtype=wp.vec3, device=self.device
        )
        self.safety_raw_points = wp.zeros(
            self.safety_total_rays, dtype=wp.vec3, device=self.device
        )
        self.safety_raw_distances = wp.zeros(
            self.safety_total_rays, dtype=wp.float32, device=self.device
        )
        # Warp's radix sorter uses the second half of each allocation as
        # temporary storage. Only the first safety_total_rays entries are
        # populated by the ray kernel and consumed by run-length encoding.
        self.safety_voxel_keys = wp.empty(
            2 * self.safety_total_rays,
            dtype=wp.int32,
            device=self.device,
        )
        self.safety_ray_indices = wp.empty(
            2 * self.safety_total_rays,
            dtype=wp.int32,
            device=self.device,
        )
        self.safety_unique_keys = wp.empty(
            self.safety_total_rays, dtype=wp.int32, device=self.device
        )
        self.safety_run_lengths = wp.zeros(
            self.safety_total_rays, dtype=wp.int32, device=self.device
        )
        self.safety_run_offsets = wp.zeros(
            self.safety_total_rays, dtype=wp.int32, device=self.device
        )
        self.safety_run_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        # Voxelization can never produce more outputs than input rays. The
        # compact GPU buffer is small (~0.6 MiB for ten UAVs) and lets the CPU
        # receive only low-density safety data, never the dense camera hits.
        self.safety_compact_counts = wp.zeros(
            self.camera_count, dtype=wp.int32, device=self.device
        )
        self.safety_compact_points = wp.zeros(
            self.safety_total_rays, dtype=wp.vec3, device=self.device
        )
        self.safety_compact_distances = wp.zeros(
            self.safety_total_rays, dtype=wp.float32, device=self.device
        )
        self.safety_report = {
            "backend": "warp_static_mesh_gpu_batched_360",
            "horizontal_resolution_deg": horizontal_resolution_deg,
            "vertical_resolution_deg": vertical_resolution_deg,
            "rays_per_uav": self.safety_ray_count,
            "total_rays_per_launch": self.safety_total_rays,
            "near_range_m": near_range,
            "max_range_m": max_range,
            "voxel_size_m": voxel_size,
            "voxel_grid_origin": safety_origin.tolist(),
            "voxel_grid_dims": self.safety_voxel_dims.tolist(),
            "mount_translation_body_m": self.safety_mount_translation.tolist(),
        }

    @staticmethod
    def _build_static_mesh(stage, include_prefixes, exclude_prefixes):
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        all_vertices = []
        all_indices = []
        vertex_offset = 0
        prim_count = 0
        for prim in _stage_prims(stage):
            path = str(prim.GetPath())
            if include_prefixes and not any(
                path.startswith(prefix) for prefix in include_prefixes
            ):
                continue
            if any(path.startswith(prefix) for prefix in exclude_prefixes):
                continue
            if (
                not prim.IsActive()
                or not prim.IsLoaded()
                or not _is_renderable(prim)
            ):
                continue
            geometry = None
            if prim.IsA(UsdGeom.Mesh):
                geometry = _triangulate_mesh(prim, xform_cache)
            elif prim.IsA(UsdGeom.Cube):
                geometry = _triangulate_cube(prim, xform_cache)
            if geometry is None:
                continue
            vertices, indices = geometry
            if not len(vertices) or not len(indices):
                continue
            all_vertices.append(vertices)
            all_indices.append(indices + vertex_offset)
            vertex_offset += len(vertices)
            prim_count += 1
        if not all_vertices:
            raise RuntimeError(
                "Warp ray caster found no renderable Mesh/Cube geometry under "
                f"{tuple(include_prefixes)}"
            )
        return (
            np.ascontiguousarray(np.concatenate(all_vertices), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(all_indices), dtype=np.int32),
            prim_count,
        )

    def cast(
        self,
        positions,
        orientations,
        quaternion_matrix,
        *,
        return_distances: bool = False,
    ):
        """Return endpoints, hit flags, optional ranges, and elapsed time.

        Every range is always written to the persistent GPU output buffer.
        The ROS bridge leaves ``return_distances`` disabled because RACER only
        consumes XYZ and the hit flag; this avoids an unnecessary host copy.
        """

        if (
            len(positions) != self.camera_count
            or len(orientations) != self.camera_count
        ):
            raise ValueError("pose count does not match configured camera_count")
        origins = np.empty((self.camera_count, 3), dtype=np.float32)
        rotations = np.empty((self.camera_count, 3, 3), dtype=np.float32)
        for index, (position, orientation) in enumerate(
            zip(positions, orientations)
        ):
            rotation = np.asarray(
                quaternion_matrix(orientation), dtype=np.float32
            ).reshape((3, 3))
            rotations[index] = rotation
            origins[index] = (
                np.asarray(position, dtype=np.float32)
                + rotation @ self.mount_translation
            )

        started = time.perf_counter()
        self.camera_origins.assign(origins)
        self.camera_rotations.assign(np.ascontiguousarray(rotations))
        wp.launch(
            kernel=_raycast_pinhole_kernel,
            dim=self.camera_count * self.ray_count,
            inputs=[
                self.mesh.id,
                self.camera_origins,
                self.camera_rotations,
                self.local_directions,
                self.forward_components,
                self.ray_count,
                self.near_depth,
                self.map_depth,
                self.render_depth,
            ],
            outputs=[
                self.points_world,
                self.hit_mask,
                self.hit_distances,
            ],
            device=self.device,
        )
        # Host PointCloud2 publication is the first CPU consumer. Copy once per
        # batch after the single GPU launch; no per-ray Python work occurs.
        points = self.points_world.numpy().reshape(
            (self.camera_count, self.ray_count, 3)
        )
        hit = self.hit_mask.numpy().reshape(
            (self.camera_count, self.ray_count)
        ).astype(bool, copy=False)
        distances = None
        if return_distances:
            distances = self.hit_distances.numpy().reshape(
                (self.camera_count, self.ray_count)
            )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        return points, hit, distances, elapsed_ms

    def cast_safety(self, positions, orientations, quaternion_matrix):
        """Return GPU range-filtered and voxel-downsampled 360-degree hits.

        The returned lists contain one compact point/distance array per UAV.
        No dense mapping-camera point is consumed by this path.  Timings split
        the mesh queries from GPU voxel sort/compaction and its small host copy.
        """

        if not self.safety_enabled:
            raise RuntimeError("safety rays were not configured")
        if (
            len(positions) != self.camera_count
            or len(orientations) != self.camera_count
        ):
            raise ValueError("pose count does not match configured camera_count")
        origins = np.empty((self.camera_count, 3), dtype=np.float32)
        rotations = np.empty((self.camera_count, 3, 3), dtype=np.float32)
        for index, (position, orientation) in enumerate(
            zip(positions, orientations)
        ):
            rotation = np.asarray(
                quaternion_matrix(orientation), dtype=np.float32
            ).reshape((3, 3))
            rotations[index] = rotation
            origins[index] = (
                np.asarray(position, dtype=np.float32)
                + rotation @ self.safety_mount_translation
            )

        self.safety_sensor_origins.assign(origins)
        self.camera_rotations.assign(np.ascontiguousarray(rotations))
        raycast_started = time.perf_counter()
        wp.launch(
            kernel=_raycast_safety_kernel,
            dim=self.safety_total_rays,
            inputs=[
                self.mesh.id,
                self.safety_sensor_origins,
                self.camera_rotations,
                self.safety_local_directions,
                self.safety_ray_count,
                self.safety_near_range,
                self.safety_max_range,
                wp.vec3(*self.safety_voxel_origin),
                self.safety_voxel_size,
                wp.vec3i(*self.safety_voxel_dims),
                self.safety_voxels_per_scene,
                self.safety_miss_key,
            ],
            outputs=[
                self.safety_raw_points,
                self.safety_raw_distances,
                self.safety_voxel_keys,
                self.safety_ray_indices,
            ],
            device=self.device,
        )
        # The explicit synchronization makes the two requested profiling
        # categories independently meaningful rather than reporting deferred
        # CUDA work in whichever host copy happens to run next.
        wp.synchronize_device(self.device)
        raycasting_ms = 1000.0 * (time.perf_counter() - raycast_started)

        process_started = time.perf_counter()
        self.safety_run_lengths.zero_()
        self.safety_compact_counts.zero_()
        wp.utils.radix_sort_pairs(
            self.safety_voxel_keys,
            self.safety_ray_indices,
            self.safety_total_rays,
        )
        wp.utils.runlength_encode(
            self.safety_voxel_keys,
            self.safety_unique_keys,
            self.safety_run_lengths,
            run_count=self.safety_run_count,
            value_count=self.safety_total_rays,
        )
        wp.utils.array_scan(
            self.safety_run_lengths,
            self.safety_run_offsets,
            inclusive=False,
        )
        wp.launch(
            kernel=_compact_safety_voxels_kernel,
            dim=self.safety_total_rays,
            inputs=[
                self.safety_run_count,
                self.safety_unique_keys,
                self.safety_run_offsets,
                self.safety_ray_indices,
                self.safety_raw_points,
                self.safety_raw_distances,
                self.safety_ray_count,
                self.safety_miss_key,
                self.safety_ray_count,
            ],
            outputs=[
                self.safety_compact_counts,
                self.safety_compact_points,
                self.safety_compact_distances,
            ],
            device=self.device,
        )
        counts = np.clip(
            self.safety_compact_counts.numpy(), 0, self.safety_ray_count
        ).astype(np.int32, copy=False)
        compact_points = self.safety_compact_points.numpy().reshape(
            (self.camera_count, self.safety_ray_count, 3)
        )
        compact_distances = self.safety_compact_distances.numpy().reshape(
            (self.camera_count, self.safety_ray_count)
        )
        points = [
            compact_points[index, : int(count)].copy()
            for index, count in enumerate(counts)
        ]
        distances = [
            compact_distances[index, : int(count)].copy()
            for index, count in enumerate(counts)
        ]
        point_process_ms = 1000.0 * (
            time.perf_counter() - process_started
        )
        return points, distances, raycasting_ms, point_process_ms

    def report(self) -> dict:
        return {
            **self.mesh_report,
            "backend": "warp_static_mesh_gpu_batched",
            "camera_count": self.camera_count,
            "rays_per_camera": self.ray_count,
            "total_rays_per_launch": self.camera_count * self.ray_count,
            "near_depth_m": self.near_depth,
            "map_depth_m": self.map_depth,
            "render_depth_m": self.render_depth,
            "image_width_px": self.image_width,
            "image_height_px": self.image_height,
            "fx_px": self.fx,
            "fy_px": self.fy,
            "cx_px": self.cx,
            "cy_px": self.cy,
            "horizontal_fov_rad": self.horizontal_fov_rad,
            "vertical_fov_rad": self.vertical_fov_rad,
            "sampled_horizontal_fov_rad": self.sampled_horizontal_fov_rad,
            "sampled_vertical_fov_rad": self.sampled_vertical_fov_rad,
            "sample_pixel_bounds": self.sample_pixel_bounds,
            "safety_rays": self.safety_report,
        }
