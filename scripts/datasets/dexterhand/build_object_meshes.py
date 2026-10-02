#!/usr/bin/env python3
"""Build DexterHand object meshes from the official per-session shape metadata.

The released NPZ files contain object class and dimensions, not OBJ files.
Geometry and local axes follow PKU-MoCCA/dextercap Dataset/visualize.py and
ObjectReconstruction/rubikscube.py. This does not infer physical material or
convert the Rubik's Cube face-angle trace into simulator joint states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def y_cylinder(radius: float, height: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=128)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def ring(outer_radius: float, inner_radius: float, height: float) -> trimesh.Trimesh:
    if not 0 < inner_radius < outer_radius:
        raise ValueError("Ring must have 0 < inner radius < outer radius")
    n = 128
    angle = 2 * np.pi * np.arange(n) / n
    xz = np.column_stack((np.cos(angle), np.sin(angle)))
    vertices = np.asarray(
        [[radius * x, y, radius * z]
         for x, z in xz
         for radius, y in ((outer_radius, height / 2), (inner_radius, height / 2),
                           (outer_radius, -height / 2), (inner_radius, -height / 2))],
        dtype=np.float64,
    )
    faces = []
    for i in range(n):
        j = (i + 1) % n
        for a, b, c, d in ((0, 0, 2, 2), (1, 1, 3, 3), (0, 0, 1, 1), (2, 2, 3, 3)):
            faces.extend(((4*i+a, 4*j+b, 4*i+c), (4*j+b, 4*j+d, 4*i+c)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def triangular_prism(base_length: float, height: float) -> trimesh.Trimesh:
    # Official visualizer uses an equilateral triangle in XZ and height on Y.
    radius = base_length / np.sqrt(3)
    angles = 2 * np.pi * np.arange(3) / 3 + np.pi / 6 + np.pi
    xz = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    vertices = np.asarray([[x, y, z] for x, z in xz for y in (height/2, -height/2)])
    faces = []
    for i in range(3):
        j = (i + 1) % 3
        faces.extend(((2*i, 2*j, 2*i+1), (2*j, 2*j+1, 2*i+1)))
    faces.extend(((0, 2, 4), (5, 3, 1)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def make_meshes(kind: str, size: list[float]) -> dict[str, trimesh.Trimesh]:
    if kind == "Cuboid":
        return {"visual": trimesh.creation.box(extents=size)}
    if kind in {"Cylinder", "Plate"}:
        return {"visual": y_cylinder(size[0] / 2, size[1])}
    if kind == "Ring":
        return {"visual": ring(size[0] / 2, size[1] / 2, size[2])}
    if kind == "Prism":
        return {"visual": triangular_prism(size[0], size[1])}
    if kind == "RubiksCube":
        # The official viewer is a 2x2x2 cube with 8 independently moving cubes.
        cubelet_size = size[0] / 2
        pieces = {}
        for ix in (-1, 1):
            for iy in (-1, 1):
                for iz in (-1, 1):
                    name = f"cubelet_{ix:+d}_{iy:+d}_{iz:+d}"
                    mesh = trimesh.creation.box(extents=(cubelet_size,) * 3)
                    mesh.apply_translation(np.array([ix, iy, iz]) * cubelet_size / 2)
                    pieces[name] = mesh
        # A solid rest-pose preview avoids coincident internal faces; the eight
        # separate cubelets above are the parts required for face articulation.
        return {"visual_rest": trimesh.creation.box(extents=(size[0],) * 3), **pieces}
    raise ValueError(f"Unsupported official object class: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/dexterhand_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for source in sorted((root / "raw" / "sessions").glob("*-fps_20.npz")):
        with np.load(source, allow_pickle=True) as data:
            metadata = data["metadata"].item()
        kind = str(metadata["object_class"])
        size = [float(x) for x in metadata["object_size"]]
        if not size or not np.isfinite(size).all() or min(size) <= 0:
            raise ValueError(f"Invalid dimensions in {source}")
        name = str(metadata["mocap_session_name"])
        meshes = make_meshes(kind, size)
        output = root / "assets" / "objects" / name
        output.mkdir(parents=True, exist_ok=True)
        geometry = {}
        for part_name, mesh in meshes.items():
            if not mesh.is_watertight or mesh.volume <= 0:
                raise ValueError(f"Invalid mesh: {name}/{part_name}")
            path = output / f"{part_name}.obj"
            mesh.export(path)
            geometry[part_name] = {
                "path": str(path.relative_to(root)),
                "bounds_m": mesh.bounds.tolist(),
                "volume_m3": float(mesh.volume),
                "watertight": bool(mesh.is_watertight),
            }
        record = {
            "session": name, "object_class": kind, "object_size_m": size,
            "source": str(source.relative_to(root)), "mesh_local_frame": "official object pose frame",
            "geometry": geometry,
            "articulation_status": (
                "8 cubelet meshes; incremental face rotations require separate state reconstruction"
                if kind == "RubiksCube" else "rigid"
            ),
        }
        (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(f"DEXTERHAND_MESH_READY {name} {kind}", flush=True)
    (root / "assets" / "objects" / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
