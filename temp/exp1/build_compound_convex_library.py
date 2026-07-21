#!/usr/bin/env python3
"""Build an ordered fixed-slot, medium-complexity convex-cage hand library.

SPIDER-style decomposition initializes the cages, but the saved neural
representation is ordered in each joint-local frame and includes a fixed
42-direction support descriptor per slot. It is therefore not an unordered
raw V-HACD output.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SOURCE_LIBRARY = OUTPUTS / "mesh_library.json"
COMPOUND_ROOT = OUTPUTS / "compound_convex_library"
COMPOUND_LIBRARY = OUTPUTS / "compound_convex_library.json"
MAX_INPUT_FACES = 50_000
MAX_VERTICES_PER_HULL = 42
SUPPORT_DIRECTIONS = np.asarray(trimesh.creation.icosphere(subdivisions=1).vertices, dtype=float)


def hull_budget(node: dict) -> int:
    role = node.get("role", "other")
    faces = int(node.get("faces", 0))
    if role == "palm":
        return 8 if faces > 20_000 else 6 if faces > 2_000 else 3
    if role == "base":
        return 6 if faces > 10_000 else 4 if faces > 1_000 else 3
    if role in {"thumb", "index", "middle", "ring", "pinky"}:
        return 4 if faces > 20_000 else 3 if faces > 1_000 else 2
    return 4 if faces > 20_000 else 3


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    covariance = (points - center).T @ (points - center) / max(len(points), 1)
    _, vectors = np.linalg.eigh(covariance)
    rotation = vectors[:, ::-1]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return center, rotation


def ordered_parts(parts: list[trimesh.Trimesh], source: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    center, rotation = pca_frame(np.asarray(source.vertices, dtype=float))
    return sorted(
        parts,
        key=lambda part: tuple(np.round((np.asarray(part.centroid) - center) @ rotation, 6)),
    )


def support_descriptor(part: trimesh.Trimesh) -> dict:
    center = np.asarray(part.centroid, dtype=float)
    relative = np.asarray(part.vertices, dtype=float) - center
    projections = relative @ SUPPORT_DIRECTIONS.T
    indices = np.argmax(projections, axis=0)
    support_points = relative[indices]
    support_radii = projections[indices, np.arange(len(SUPPORT_DIRECTIONS))]
    return {
        "center": center.round(8).tolist(),
        "support_radii_42": support_radii.round(8).tolist(),
        "support_points_42x3": support_points.round(8).tolist(),
    }


def decomposition_input(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=9)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    if len(mesh.faces) > MAX_INPUT_FACES:
        try:
            reduced = mesh.simplify_quadric_decimation(face_count=MAX_INPUT_FACES, aggression=3)
            if reduced is not None and not reduced.is_empty:
                mesh = reduced
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    return mesh


def split_points(points: np.ndarray, budget: int) -> list[np.ndarray]:
    """Deterministically partition the complete surface without dropping it."""
    clusters = [points]
    while len(clusters) < budget:
        candidates = [(float(np.linalg.norm(np.ptp(cluster, axis=0))), index) for index, cluster in enumerate(clusters) if len(cluster) >= 16]
        if not candidates:
            break
        _, index = max(candidates)
        cluster = clusters.pop(index)
        center, rotation = pca_frame(cluster)
        projection = (cluster - center) @ rotation[:, 0]
        threshold = float(np.median(projection))
        left, right = cluster[projection <= threshold], cluster[projection > threshold]
        if min(len(left), len(right)) < 4:
            clusters.append(cluster)
            break
        clusters.extend((left, right))
    return clusters


def support_cage(points: np.ndarray) -> trimesh.Trimesh:
    """Reduce one spatial slot to a bounded, fixed-direction convex cage."""
    center = np.median(points, axis=0)
    projections = (points - center) @ SUPPORT_DIRECTIONS.T
    support = points[np.argmax(projections, axis=0)]
    support = np.unique(np.round(support, 9), axis=0)
    if len(support) < 4:
        return trimesh.points.PointCloud(points).convex_hull
    return trimesh.points.PointCloud(support).convex_hull


def convex_parts(mesh: trimesh.Trimesh, budget: int, seed: int) -> list[trimesh.Trimesh]:
    # Uniform surface samples make the fitting independent of source triangle
    # density. Global support points guarantee that thin extrema are retained.
    if len(mesh.faces) <= 200:
        return [mesh.convex_hull]
    count = max(2400, budget * 600)
    sampled, _ = trimesh.sample.sample_surface(mesh, count=count, seed=seed)
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = np.median(vertices, axis=0)
    extrema = vertices[np.argmax((vertices - center) @ SUPPORT_DIRECTIONS.T, axis=0)]
    points = np.vstack((sampled, extrema))
    return [support_cage(cluster) for cluster in split_points(points, budget)]


def sampled_error(source: trimesh.Trimesh, approximation: trimesh.Trimesh, seed: int) -> tuple[float, float]:
    count = 1200
    source_points, _ = trimesh.sample.sample_surface(source, count=count, seed=seed)
    approx_points, _ = trimesh.sample.sample_surface(approximation, count=count, seed=seed + 1)
    forward = cKDTree(approx_points).query(source_points, workers=-1)[0]
    backward = cKDTree(source_points).query(approx_points, workers=-1)[0]
    distances = np.concatenate((forward, backward))
    diagonal = max(float(np.linalg.norm(source.extents)), 1.0e-6)
    return float(np.quantile(distances, 0.95) / diagonal), float(np.quantile(distances, 0.99) / diagonal)


def main() -> int:
    source = json.loads(SOURCE_LIBRARY.read_text(encoding="utf-8"))
    if COMPOUND_ROOT.exists():
        shutil.rmtree(COMPOUND_ROOT)
    COMPOUND_ROOT.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "purpose": "ordered fixed-slot convex-cage morphology representation",
        "initialization": "SPIDER-inspired bounded compound convex cages; coverage-preserving deterministic surface partition",
        "max_vertices_per_hull": MAX_VERTICES_PER_HULL,
        "support_directions_42x3": SUPPORT_DIRECTIONS.round(8).tolist(),
        "slot_capacities": {"palm": 8, "base": 6, "digit_link": 4, "other": 4},
        "hands": {},
    }
    total_hulls = total_faces = 0
    all_p95, all_p99 = [], []
    seed = 100
    for hand_id, hand in source["hands"].items():
        hand_dir = COMPOUND_ROOT / hand_id
        hand_dir.mkdir(parents=True)
        records = []
        hand_hulls = hand_faces = 0
        hand_errors = []
        for node in hand["nodes"]:
            record = {
                "index": node["index"], "name": node["name"], "role": node["role"],
                "parent": node["parent"], "relative_pos": node["relative_pos"],
                "pivot": node["pivot"], "joint_type": node["joint_type"],
                "hulls": [], "mesh": None, "faces": 0,
            }
            if node.get("mesh"):
                original = trimesh.load(OUTPUTS / node["mesh"], force="mesh", process=False)
                working = decomposition_input(original)
                slot_capacity = hull_budget(node)
                parts = ordered_parts(convex_parts(working, slot_capacity, seed), working)
                combined = trimesh.util.concatenate(parts)
                mesh_path = hand_dir / f"node_{node['index']:02d}.obj"
                combined.export(mesh_path, file_type="obj", include_normals=True, include_color=False)
                p95, p99 = sampled_error(original, combined, seed)
                seed += 2
                record.update(
                    {
                        "mesh": str(mesh_path.relative_to(OUTPUTS)),
                        "faces": int(len(combined.faces)),
                        "p95_surface_error_over_diagonal": p95,
                        "p99_surface_error_over_diagonal": p99,
                        "slot_capacity": slot_capacity,
                        "active_slots": len(parts),
                        "hulls": [
                            {
                                "slot": slot,
                                **support_descriptor(part),
                                "vertices": np.asarray(part.vertices, dtype=float).round(8).tolist(),
                                "faces": np.asarray(part.faces, dtype=int).tolist(),
                            }
                            for slot, part in enumerate(parts)
                        ],
                    }
                )
                hand_hulls += len(parts)
                hand_faces += len(combined.faces)
                hand_errors.append(p95)
                all_p95.append(p95)
                all_p99.append(p99)
            records.append(record)
        payload["hands"][hand_id] = {
            "display_name": hand["display_name"],
            "digit_count": hand["digit_count"],
            "scalar_dofs": hand["scalar_dofs"],
            "nodes": records,
            "convex_hulls": hand_hulls,
            "triangle_faces": hand_faces,
            "median_link_p95_error": float(np.median(hand_errors)) if hand_errors else 0.0,
        }
        total_hulls += hand_hulls
        total_faces += hand_faces
        print(
            f"{hand_id:20s}: hulls={hand_hulls:3d} faces={hand_faces:5d} "
            f"median_p95={np.median(hand_errors) if hand_errors else 0.0:.3%}",
            flush=True,
        )
    payload["summary"] = {
        "hands": len(payload["hands"]),
        "convex_hulls": total_hulls,
        "triangle_faces": total_faces,
        "median_link_p95_error": float(np.median(all_p95)),
        "median_link_p99_error": float(np.median(all_p99)),
    }
    COMPOUND_LIBRARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {COMPOUND_LIBRARY}: hulls={total_hulls}, faces={total_faces}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
