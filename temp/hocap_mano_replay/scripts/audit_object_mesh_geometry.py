#!/usr/bin/env python3
"""Audit HO-Cap object meshes and quantify collision-hull mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def mesh_stats(path: Path) -> tuple[trimesh.Trimesh, dict[str, object]]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected one mesh in {path}, got {type(loaded).__name__}")
    mesh = loaded.copy()
    components = mesh.split(only_watertight=False)
    nondegenerate = mesh.nondegenerate_faces()
    unique = mesh.unique_faces()
    stats = {
        "path": str(path.resolve()),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_m": mesh.bounds.tolist(),
        "extents_m": mesh.extents.tolist(),
        "body_diagonal_m": float(np.linalg.norm(mesh.extents)),
        "connected_components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "euler_number": int(mesh.euler_number),
        "degenerate_faces": int(len(mesh.faces) - int(nondegenerate.sum())),
        "duplicate_faces": int(len(mesh.faces) - int(unique.sum())),
        "volume_m3": float(abs(mesh.volume)),
        "surface_area_m2": float(mesh.area),
    }
    return mesh, stats


def sampled_surface_distance(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    samples: int,
    seed: int,
) -> dict[str, float]:
    np.random.seed(seed)
    source_points, _ = trimesh.sample.sample_surface(source, samples)
    np.random.seed(seed + 1)
    target_points, _ = trimesh.sample.sample_surface(target, samples)
    distances = cKDTree(target_points).query(source_points, workers=-1)[0]
    return {
        "mean_m": float(distances.mean()),
        "p95_m": float(np.percentile(distances, 95)),
        "p99_m": float(np.percentile(distances, 99)),
        "max_m": float(distances.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()

    meshes: dict[str, trimesh.Trimesh] = {}
    stats: dict[str, dict[str, object]] = {}
    for name in ("cleaned_mesh_2000.obj", "cleaned_mesh_10000.obj", "textured_mesh.obj"):
        meshes[name], stats[name] = mesh_stats(args.mesh_dir / name)

    collision_source = meshes["cleaned_mesh_2000.obj"]
    hull = collision_source.convex_hull
    hull_volume = float(abs(hull.volume))
    source_volume = float(abs(collision_source.volume))
    hull_stats = {
        "vertices": int(len(hull.vertices)),
        "faces": int(len(hull.faces)),
        "bounds_m": hull.bounds.tolist(),
        "extents_m": hull.extents.tolist(),
        "watertight": bool(hull.is_watertight),
        "is_volume": bool(hull.is_volume),
        "volume_m3": hull_volume,
        "visual_volume_m3": source_volume,
        "hull_to_visual_volume_ratio": (
            hull_volume / source_volume if source_volume > 0.0 else None
        ),
        "visual_to_hull_volume_fraction": (
            source_volume / hull_volume if hull_volume > 0.0 else None
        ),
    }

    comparisons: dict[str, object] = {}
    for name in ("cleaned_mesh_10000.obj", "textured_mesh.obj"):
        comparisons[f"2000_to_{name}"] = sampled_surface_distance(
            collision_source, meshes[name], args.samples, seed=11
        )
        comparisons[f"{name}_to_2000"] = sampled_surface_distance(
            meshes[name], collision_source, args.samples, seed=23
        )
    comparisons["visual_2000_to_convex_hull"] = sampled_surface_distance(
        collision_source, hull, args.samples, seed=31
    )
    comparisons["convex_hull_to_visual_2000"] = sampled_surface_distance(
        hull, collision_source, args.samples, seed=47
    )

    failures = []
    for name, item in stats.items():
        if item["vertices"] == 0 or item["faces"] == 0:
            failures.append(f"{name}: empty")
        if item["degenerate_faces"] or item["duplicate_faces"]:
            failures.append(f"{name}: degenerate or duplicate faces")
        if not item["watertight"] or not item["winding_consistent"]:
            failures.append(f"{name}: not a closed consistently wound surface")

    report = {
        "meshes": stats,
        "isaac_collision": {
            "approximation": "convexHull",
            **hull_stats,
        },
        "sampled_surface_comparisons": comparisons,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
