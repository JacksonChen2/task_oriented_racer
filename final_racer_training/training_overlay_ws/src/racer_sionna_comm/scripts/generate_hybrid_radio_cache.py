#!/usr/bin/env python3
"""Generate mobile-TX anchor radio maps for the warehouse hybrid channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def values(start: float, stop: float, spacing: float) -> np.ndarray:
    count = int(np.floor((stop - start) / spacing)) + 1
    axis = start + spacing * np.arange(max(1, count), dtype=np.float64)
    if axis[-1] < stop - 0.25 * spacing:
        axis = np.append(axis, stop)
    return axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=-26.8)
    parser.add_argument("--x-max", type=float, default=5.8)
    parser.add_argument("--y-min", type=float, default=7.2)
    parser.add_argument("--y-max", type=float, default=26.2)
    parser.add_argument("--tx-spacing", type=float, default=4.0)
    parser.add_argument("--rx-cell", type=float, default=0.5)
    parser.add_argument("--tx-heights", type=float, nargs="+", default=[1.0, 2.5, 4.0])
    parser.add_argument(
        "--rx-heights", type=float, nargs="+", default=[0.75, 1.5, 2.5, 3.5, 4.5]
    )
    parser.add_argument("--frequency-hz", type=float, default=2.4e9)
    parser.add_argument("--bandwidth-hz", type=float, default=20.0e6)
    parser.add_argument("--tx-power-dbm", type=float, default=20.0)
    parser.add_argument("--samples-per-tx", type=int, default=200000)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def as_numpy(value):
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    if not args.scene_xml.is_file():
        raise FileNotFoundError(args.scene_xml)
    if args.tx_spacing <= 0.0 or args.rx_cell <= 0.0:
        raise ValueError("grid spacing must be positive")

    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

    scene = load_scene(str(args.scene_xml.resolve()), merge_shapes=True)
    scene.frequency = args.frequency_hz
    scene.bandwidth = args.bandwidth_hz
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    transmitter = Transmitter(
        name="cache_tx", position=[0.0, 0.0, 1.0], power_dbm=args.tx_power_dbm
    )
    scene.add(transmitter)
    solver = RadioMapSolver()

    tx_x = values(args.x_min, args.x_max, args.tx_spacing)
    tx_y = values(args.y_min, args.y_max, args.tx_spacing)
    tx_positions = np.asarray(
        [
            [x, y, z]
            for z in args.tx_heights
            for y in tx_y
            for x in tx_x
        ],
        dtype=np.float64,
    )
    rx_z = np.asarray(sorted(args.rx_heights), dtype=np.float64)
    center_xy = [0.5 * (args.x_min + args.x_max), 0.5 * (args.y_min + args.y_max)]
    size_xy = [args.x_max - args.x_min, args.y_max - args.y_min]
    maps = None
    rx_x = None
    rx_y = None
    started = time.monotonic()

    for anchor_index, tx_position in enumerate(tx_positions):
        transmitter.position = tx_position.tolist()
        for height_index, height in enumerate(rx_z):
            radio_map = solver(
                scene=scene,
                center=[center_xy[0], center_xy[1], float(height)],
                orientation=[0.0, 0.0, 0.0],
                size=size_xy,
                cell_size=[args.rx_cell, args.rx_cell],
                samples_per_tx=args.samples_per_tx,
                max_depth=args.max_depth,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=True,
                diffraction=False,
                seed=args.seed,
            )
            gain = np.asarray(as_numpy(radio_map.path_gain), dtype=np.float64)[0]
            centers = np.asarray(as_numpy(radio_map.cell_centers), dtype=np.float64)
            if maps is None:
                rx_x = centers[0, :, 0]
                rx_y = centers[:, 0, 1]
                maps = np.full(
                    (tx_positions.shape[0], rx_z.size, gain.shape[0], gain.shape[1]),
                    -300.0,
                    dtype=np.float32,
                )
            if gain.shape != maps.shape[2:]:
                raise RuntimeError("RadioMap grid shape changed between solves")
            maps[anchor_index, height_index] = (
                10.0 * np.log10(np.maximum(gain, 1.0e-30))
            ).astype(np.float32)
        elapsed = time.monotonic() - started
        print(
            f"anchor {anchor_index+1}/{tx_positions.shape[0]} "
            f"elapsed={elapsed:.1f}s position={tx_position.tolist()}",
            flush=True,
        )

    metadata = {
        "format": "racer_3d_hybrid_radio_cache_v1",
        "scene_xml": str(args.scene_xml.resolve()),
        "frequency_hz": args.frequency_hz,
        "bandwidth_hz": args.bandwidth_hz,
        "tx_power_dbm": args.tx_power_dbm,
        "samples_per_tx": args.samples_per_tx,
        "max_depth": args.max_depth,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tx_positions=tx_positions,
        rx_x=rx_x,
        rx_y=rx_y,
        rx_z=rx_z,
        path_gain_db=maps,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
