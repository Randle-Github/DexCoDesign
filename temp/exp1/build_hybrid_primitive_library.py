#!/usr/bin/env python3
"""Build a compact hybrid standard-primitive hand geometry library.

Each URDF rigid body is fitted as one standard primitive whenever possible.
Only bodies whose full surface cannot be represented within a role-dependent
error tolerance are split into a small number of ordered components.  A
42-direction convex cage is the final fallback, not the default.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
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
SUPPORT_DIRECTIONS = np.asarray(trimesh.creation.icosphere(subdivisions=1).vertices, dtype=float)
DIGIT_ROLES = {"thumb", "index", "middle", "ring", "pinky"}


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    covariance = (points - center).T @ (points - center) / max(len(points), 1)
    _, vectors = np.linalg.eigh(covariance)
    rotation = vectors[:, ::-1]
    # Resolve the arbitrary PCA sign in the link-local coordinate system.
    for column in range(3):
        dominant = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[dominant, column] < 0:
            rotation[:, column] *= -1
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return center, rotation


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def axis_frame(rotation: np.ndarray, axis: int) -> np.ndarray:
    z_axis = rotation[:, axis]
    x_axis = rotation[:, (axis + 1) % 3]
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-9)
    y_axis = np.cross(z_axis, x_axis)
    frame = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1
    return frame


def primitive_candidates(points: np.ndarray) -> list[tuple[trimesh.Trimesh, dict]]:
    pca_center, rotation = pca_frame(points)
    local = (points - pca_center) @ rotation
    lower, upper = np.quantile(local, [0.005, 0.995], axis=0)
    midpoint = (lower + upper) / 2.0
    extents = np.maximum(upper - lower, 1.0e-6)
    world_center = pca_center + midpoint @ rotation.T
    candidates: list[tuple[trimesh.Trimesh, dict]] = []

    box = trimesh.creation.box(extents=extents, transform=rigid_transform(rotation, world_center))
    candidates.append((box, {"type": "box", "center": world_center.tolist(), "rotation": rotation.tolist(), "size": extents.tolist()}))

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere.apply_scale(extents / 2.0)
    sphere.apply_transform(rigid_transform(rotation, world_center))
    candidates.append((sphere, {"type": "ellipsoid", "center": world_center.tolist(), "rotation": rotation.tolist(), "radii": (extents / 2.0).tolist()}))

    for axis in range(3):
        frame = axis_frame(rotation, axis)
        axial = (points - pca_center) @ frame[:, 2]
        radial_xy = np.column_stack(((points - pca_center) @ frame[:, 0], (points - pca_center) @ frame[:, 1]))
        radial_center = np.median(radial_xy, axis=0)
        radii = np.linalg.norm(radial_xy - radial_center, axis=1)
        radius = max(float(np.quantile(radii, 0.985)), 1.0e-6)
        z0, z1 = np.quantile(axial, [0.005, 0.995])
        height = max(float(z1 - z0), 1.0e-6)
        center = pca_center + frame[:, 0] * radial_center[0] + frame[:, 1] * radial_center[1] + frame[:, 2] * ((z0 + z1) / 2.0)

        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=20, transform=rigid_transform(frame, center))
        candidates.append((cylinder, {"type": "cylinder", "center": center.tolist(), "rotation": frame.tolist(), "radius": radius, "height": height}))

        capsule_radius = min(radius, height * 0.45)
        cylinder_height = max(height - 2.0 * capsule_radius, height * 0.08)
        capsule = trimesh.creation.capsule(height=cylinder_height, radius=capsule_radius, count=[12, 12])
        capsule.apply_translation(-capsule.centroid)
        capsule.apply_transform(rigid_transform(frame, center))
        candidates.append((capsule, {"type": "capsule", "center": center.tolist(), "rotation": frame.tolist(), "radius": capsule_radius, "cylinder_height": cylinder_height}))
    return candidates


def point_surface_error(points: np.ndarray, mesh: trimesh.Trimesh, seed: int, diagonal: float | None = None) -> tuple[float, float]:
    count = min(3000, max(1600, len(points)))
    surface, _ = trimesh.sample.sample_surface(mesh, count=count, seed=seed)
    forward = cKDTree(surface).query(points, workers=-1)[0]
    backward = cKDTree(points).query(surface, workers=-1)[0]
    distances = np.concatenate((forward, backward))
    scale = diagonal if diagonal is not None else float(np.linalg.norm(np.ptp(points, axis=0)))
    scale = max(scale, 1.0e-6)
    return float(np.quantile(distances, 0.95) / scale), float(np.quantile(distances, 0.99) / scale)


def primitive_penalty(role: str, primitive_type: str, organic: bool) -> float:
    if organic:
        return {"capsule": 0.0, "ellipsoid": 0.002, "cylinder": 0.006, "box": 0.012}.get(primitive_type, 0.006)
    if role == "base":
        return {"cylinder": 0.0, "box": 0.004, "capsule": 0.008}.get(primitive_type, 0.030)
    if role in DIGIT_ROLES:
        return {"capsule": 0.0, "cylinder": 0.003, "box": 0.006}.get(primitive_type, 0.030)
    if role == "palm":
        return {"box": 0.0, "cylinder": 0.015, "capsule": 0.020}.get(primitive_type, 0.035)
    return {"box": 0.0, "cylinder": 0.004, "capsule": 0.006}.get(primitive_type, 0.030)


def best_primitive(points: np.ndarray, role: str, seed: int, diagonal: float | None = None, organic: bool = False) -> tuple[trimesh.Trimesh, dict, float]:
    ranked = []
    for offset, (mesh, metadata) in enumerate(primitive_candidates(points)):
        # Ellipsoids are useful for MANO-like organic tissue but erase the
        # planar housings and sharp link boundaries of mechanical hands.
        if metadata["type"] == "ellipsoid" and not organic:
            continue
        p95, p99 = point_surface_error(points, mesh, seed + offset, diagonal=diagonal)
        ranked.append((p95 + primitive_penalty(role, metadata["type"], organic), p95, p99, mesh, metadata))
    _, p95, p99, mesh, metadata = min(ranked, key=lambda item: item[0])
    metadata = {**metadata, "fit_p95": p95, "fit_p99": p99}
    return mesh, metadata, p95


def split_points(points: np.ndarray, count: int) -> list[np.ndarray]:
    clusters = [points]
    while len(clusters) < count:
        candidates = [(float(np.linalg.norm(np.ptp(cluster, axis=0))), index) for index, cluster in enumerate(clusters) if len(cluster) >= 24]
        if not candidates:
            break
        _, index = max(candidates)
        cluster = clusters.pop(index)
        center, rotation = pca_frame(cluster)
        coordinate = (cluster - center) @ rotation[:, 0]
        median = float(np.median(coordinate))
        left, right = cluster[coordinate <= median], cluster[coordinate > median]
        if min(len(left), len(right)) < 6:
            clusters.append(cluster)
            break
        clusters.extend((left, right))
    return clusters


def support_cage(points: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    center = np.median(points, axis=0)
    relative = points - center
    support = points[np.argmax(relative @ SUPPORT_DIRECTIONS.T, axis=0)]
    support = np.unique(np.round(support, 9), axis=0)
    cage = trimesh.points.PointCloud(support).convex_hull
    metadata = {
        "type": "convex_cage",
        "center": np.asarray(cage.centroid).tolist(),
        "support_directions": 42,
        "support_points": np.asarray(cage.vertices).tolist(),
    }
    return cage, metadata


def role_budget(role: str) -> int:
    if role == "palm":
        return 4
    if role == "base":
        return 2
    if role in DIGIT_ROLES:
        return 2
    return 3


def role_tolerance(role: str) -> float:
    if role == "palm":
        return 0.078
    if role == "base":
        return 0.082
    if role in DIGIT_ROLES:
        return 0.078
    return 0.085


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


def fit_components(mesh: trimesh.Trimesh, role: str, seed: int, organic: bool = False) -> tuple[list[trimesh.Trimesh], list[dict], float, float]:
    budget = role_budget(role)
    sample_count = max(3600, budget * 1600)
    points, _ = trimesh.sample.sample_surface(mesh, count=sample_count, seed=seed)
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = np.median(vertices, axis=0)
    extrema = vertices[np.argmax((vertices - center) @ SUPPORT_DIRECTIONS.T, axis=0)]
    points = np.vstack((points, extrema))
    diagonal = max(float(np.linalg.norm(mesh.extents)), 1.0e-6)
    tolerance = role_tolerance(role)

    best_mesh, best_meta, _ = best_primitive(points, role, seed, diagonal=diagonal, organic=organic)
    p95, p99 = point_surface_error(points, best_mesh, seed + 20, diagonal=diagonal)
    if p95 <= tolerance:
        return [best_mesh], [best_meta], p95, p99

    # Before increasing component count, retain a single source-shaped convex
    # part when that is already accurate.  This preserves a crisp mechanical
    # link without introducing artificial subdivisions inside one DOF.
    one_cage, one_cage_meta = support_cage(points)
    cage_p95, cage_p99 = point_surface_error(points, one_cage, seed + 30, diagonal=diagonal)
    if cage_p95 <= tolerance * 1.12:
        return [one_cage], [one_cage_meta], cage_p95, cage_p99

    last: tuple[list[trimesh.Trimesh], list[dict], float, float] | None = None
    for component_count in range(2, budget + 1):
        clusters = split_points(points, component_count)
        meshes, metadata = [], []
        for index, cluster in enumerate(clusters):
            primitive, info, local_p95 = best_primitive(cluster, role, seed + 100 + index * 10, organic=organic)
            # A badly fitted local primitive is replaced by one compact cage.
            if local_p95 > tolerance * 1.35:
                primitive, info = support_cage(cluster)
            meshes.append(primitive)
            metadata.append(info)
        combined = trimesh.util.concatenate(meshes)
        p95, p99 = point_surface_error(points, combined, seed + 200 + component_count, diagonal=diagonal)
        last = meshes, metadata, p95, p99
        if p95 <= tolerance * 1.12:
            return last
    assert last is not None
    # If the small primitive mixture still misses too much geometry, use the
    # same number of ordered convex cages rather than accepting a giant box or
    # capsule merely because it is a standard primitive.
    clusters = split_points(points, budget)
    cage_meshes, cage_metadata = [], []
    for cluster in clusters:
        cage, info = support_cage(cluster)
        cage_meshes.append(cage)
        cage_metadata.append(info)
    cage_combined = trimesh.util.concatenate(cage_meshes)
    cage_p95, cage_p99 = point_surface_error(points, cage_combined, seed + 280, diagonal=diagonal)
    if cage_p95 < last[2]:
        return cage_meshes, cage_metadata, cage_p95, cage_p99
    return last


def main() -> int:
    source = json.loads(SOURCE_LIBRARY.read_text(encoding="utf-8"))
    if COMPOUND_ROOT.exists():
        shutil.rmtree(COMPOUND_ROOT)
    COMPOUND_ROOT.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "purpose": "small ordered standard-primitive sets per URDF rigid body",
        "representation": "box/cylinder/capsule/ellipsoid first; 42-direction convex cage fallback",
        "component_capacities": {"palm": 4, "base": 2, "digit_link": 2, "other": 3},
        "hands": {},
    }
    totals = Counter()
    all_p95, all_p99 = [], []
    seed = 100
    for hand_id, hand in source["hands"].items():
        hand_dir = COMPOUND_ROOT / hand_id
        hand_dir.mkdir(parents=True)
        records, hand_errors = [], []
        hand_types = Counter()
        hand_faces = 0
        for node in hand["nodes"]:
            record = {
                "index": node["index"], "name": node["name"], "role": node["role"],
                "parent": node["parent"], "relative_pos": node["relative_pos"],
                "pivot": node["pivot"], "joint_type": node["joint_type"],
                "components": [], "hulls": [], "mesh": None, "faces": 0,
            }
            if node.get("mesh"):
                original = trimesh.load(OUTPUTS / node["mesh"], force="mesh", process=False)
                working = decomposition_input(original)
                meshes, components, p95, p99 = fit_components(working, node["role"], seed, organic=(hand_id == "mano"))
                seed += 300
                order_center, order_rotation = pca_frame(np.asarray(working.vertices, dtype=float))
                order = sorted(range(len(meshes)), key=lambda index: tuple(np.round((np.asarray(meshes[index].centroid) - order_center) @ order_rotation, 6)))
                meshes = [meshes[index] for index in order]
                components = [{"slot": slot, **components[index]} for slot, index in enumerate(order)]
                combined = trimesh.util.concatenate(meshes)
                mesh_path = hand_dir / f"node_{node['index']:02d}.obj"
                combined.export(mesh_path, file_type="obj", include_normals=True, include_color=False)
                record.update({
                    "mesh": str(mesh_path.relative_to(OUTPUTS)),
                    "faces": int(len(combined.faces)),
                    "p95_surface_error_over_diagonal": p95,
                    "p99_surface_error_over_diagonal": p99,
                    "slot_capacity": role_budget(node["role"]),
                    "active_slots": len(components),
                    "dof_part_id": node["index"],
                    "components": components,
                })
                for component in components:
                    hand_types[component["type"]] += 1
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
            "components": int(sum(hand_types.values())),
            "component_types": dict(sorted(hand_types.items())),
            "triangle_faces": int(hand_faces),
            "median_link_p95_error": float(np.median(hand_errors)) if hand_errors else 0.0,
        }
        totals.update(hand_types)
        print(f"{hand_id:20s}: components={sum(hand_types.values()):3d} faces={hand_faces:6d} types={dict(hand_types)} median_p95={np.median(hand_errors):.3%}", flush=True)
    payload["summary"] = {
        "hands": len(payload["hands"]),
        "components": int(sum(totals.values())),
        "component_types": dict(sorted(totals.items())),
        "triangle_faces": int(sum(hand["triangle_faces"] for hand in payload["hands"].values())),
        "median_link_p95_error": float(np.median(all_p95)),
        "median_link_p99_error": float(np.median(all_p99)),
    }
    COMPOUND_LIBRARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {COMPOUND_LIBRARY}: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
