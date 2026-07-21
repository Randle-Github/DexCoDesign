#!/usr/bin/env python3
"""Extract canonical link-level visual meshes from every right-hand asset.

Visual meshes remain recognizably derived from the source assets. Only a
moderate per-hand face budget is applied for a 100-instance MacBook gallery.
Collision proxies are not produced here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMP = ROOT / "temp"
sys.path.insert(0, str(TEMP))

from visualize_right_hands import (  # noqa: E402
    PALETTE,
    load_all_hands,
    load_mjcf,
    load_urdf,
    load_usd,
    source_landmark_positions,
)


OUTPUTS = HERE / "outputs"
MESH_ROOT = OUTPUTS / "mesh_library"
LIBRARY = OUTPUTS / "mesh_library.json"
FACE_BUDGET_PER_HAND = 72_000  # soft target; geometric error takes priority
MIN_FACES_PER_NONEMPTY_LINK = 600
MIN_RETAIN_RATIO = 0.25
MAX_VERTEX_CHAMFER_RATIO = 0.004


def raw_scene(hand):
    rgba = PALETTE[0]
    if hand.source_format == "urdf":
        return load_urdf(hand.source_path, rgba)
    if hand.source_format == "mjcf":
        return load_mjcf(hand.source_path, rgba)
    if hand.source_format == "usd":
        return load_usd(hand.source_path, rgba)
    raise ValueError(hand.source_format)


def canonical_transform(hand) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = hand.canonical_basis.T
    transform[:3, 3] = -hand.canonical_basis.T @ hand.root_origin
    return transform


def graph_parents(adjacency: np.ndarray, count: int) -> list[int | None]:
    parents: list[int | None] = [None]
    for node in range(1, count):
        earlier = np.flatnonzero(adjacency[node, :node] > 0.5)
        parents.append(int(earlier[-1]) if len(earlier) else 0)
    return parents


def exact_component_owner(scene_node: str, node_names: list[str]) -> int | None:
    matches = []
    lowered = scene_node.lower()
    for index, name in enumerate(node_names):
        token = name.lower()
        if lowered == token or lowered.startswith(token + "_") or f"/{token}/" in lowered:
            matches.append((len(token), index))
    return max(matches)[1] if matches else None


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Conservative repair: topology cleanup and only trivially small holes."""
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=9)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    # trimesh.fill_holes only fills single-triangle/quad holes. Larger unknown
    # openings are preserved rather than inventing a surface.
    mesh.fill_holes()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def approximate_vertex_error(source: trimesh.Trimesh, result: trimesh.Trimesh) -> float:
    source_vertices = np.asarray(source.vertices, dtype=float)
    result_vertices = np.asarray(result.vertices, dtype=float)
    if len(source_vertices) > 5000:
        source_vertices = source_vertices[np.linspace(0, len(source_vertices) - 1, 5000, dtype=np.int64)]
    if len(result_vertices) > 5000:
        result_vertices = result_vertices[np.linspace(0, len(result_vertices) - 1, 5000, dtype=np.int64)]
    forward = cKDTree(result_vertices).query(source_vertices, workers=-1)[0]
    backward = cKDTree(source_vertices).query(result_vertices, workers=-1)[0]
    return float(max(np.quantile(forward, 0.995), np.quantile(backward, 0.995)))


def simplify_component(component: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    if len(component.faces) <= target or target < 8:
        return component
    simplified = component.simplify_quadric_decimation(face_count=target, aggression=3)
    return clean_mesh(simplified) if simplified is not None and not simplified.is_empty else component


def decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    mesh = clean_mesh(mesh)
    if len(mesh.faces) <= target:
        return mesh
    try:
        components = list(mesh.split(only_watertight=False))
        if len(components) <= 1:
            simplified = simplify_component(mesh, target)
        else:
            total = sum(len(component.faces) for component in components)
            simplified_parts = []
            for component in components:
                component_target = max(8, round(target * len(component.faces) / total))
                simplified_parts.append(simplify_component(component, component_target))
            simplified = clean_mesh(trimesh.util.concatenate(simplified_parts))
        if simplified is not None and not simplified.is_empty:
            source_bounds = np.asarray(mesh.bounds, dtype=float)
            result_bounds = np.asarray(simplified.bounds, dtype=float)
            source_extent = np.ptp(source_bounds, axis=0)
            result_extent = np.ptp(result_bounds, axis=0)
            diagonal = max(float(np.linalg.norm(source_extent)), 1.0e-6)
            tolerance = 0.004 * diagonal
            bounds_ok = np.all(np.abs(result_bounds - source_bounds) <= tolerance)
            extent_ok = np.all(np.abs(result_extent - source_extent) <= 2.0 * tolerance)
            finite_ok = np.isfinite(simplified.vertices).all()
            vertex_error = approximate_vertex_error(mesh, simplified)
            error_ok = vertex_error <= MAX_VERTEX_CHAMFER_RATIO * diagonal
            if bounds_ok and extent_ok and finite_ok and error_ok:
                return simplified
            print(
                f"      rejected QEM: faces={len(mesh.faces):,}->{len(simplified.faces):,} "
                f"bounds_error={float(np.max(np.abs(result_bounds - source_bounds))):.4g} "
                f"vertex_error={vertex_error / diagonal:.3%}",
                flush=True,
            )
    except Exception as error:
        print(f"      decimation fallback: {type(error).__name__}: {error}", flush=True)
    # Never sample disconnected faces: retain the source mesh if a
    # connectivity-preserving simplifier cannot process it.
    return mesh


def main() -> int:
    dataset = json.loads((OUTPUTS / "dataset.json").read_text(encoding="utf-8"))
    records = {record["hand_id"]: record for record in dataset["records"]}
    hands = load_all_hands("right")
    if MESH_ROOT.exists():
        shutil.rmtree(MESH_ROOT)
    MESH_ROOT.mkdir(parents=True)
    library = {"schema_version": 1, "face_budget_per_hand": FACE_BUDGET_PER_HAND, "hands": {}}

    for hand in hands:
        record = records[hand.hand_id]
        count = int(record["node_count"])
        node_names = list(record["node_names"][:count])
        roles = list(record["node_roles"][:count])
        adjacency = np.asarray(record["adjacency"], dtype=float)[:count, :count]
        parents = graph_parents(adjacency, count)

        # Reconstruct the exact scale/translation used by load_all_hands.
        raw = raw_scene(hand)
        bounds = np.asarray(raw.bounds, dtype=float)
        center = bounds.mean(axis=0)
        scale = 2.0 / float(np.ptp(bounds, axis=0).max())
        raw_positions = source_landmark_positions(hand.source_path, hand.source_format)
        pivots = np.full((count, 3), np.nan, dtype=float)
        for index, name in enumerate(node_names):
            candidates = [value for key, value in raw_positions.items() if key == name or key.rsplit("/", 1)[-1] == name]
            if candidates:
                normalized = (np.asarray(candidates[0], dtype=float) - center) * scale
                pivots[index] = (normalized - hand.root_origin) @ hand.canonical_basis

        transform = canonical_transform(hand)
        components: list[tuple[str, trimesh.Trimesh, np.ndarray]] = []
        for scene_node in hand.scene.graph.nodes_geometry:
            node_transform, geometry_name = hand.scene.graph.get(scene_node)
            mesh = hand.scene.geometry[geometry_name].copy()
            mesh.apply_transform(node_transform)
            mesh.apply_transform(transform)
            if mesh.is_empty:
                continue
            components.append((scene_node, mesh, np.asarray(mesh.centroid, dtype=float)))

        finite = np.isfinite(pivots).all(axis=1)
        if not finite.any():
            raise ValueError(f"No link pivots for {hand.hand_id}")
        owned: list[list[trimesh.Trimesh]] = [[] for _ in range(count)]
        for scene_node, mesh, centroid in components:
            owner = exact_component_owner(scene_node, node_names)
            if owner is None or not finite[owner]:
                distances = np.linalg.norm(pivots[finite] - centroid[None, :], axis=1)
                owner = int(np.flatnonzero(finite)[int(np.argmin(distances))])
            owned[owner].append(mesh)

        # Missing pivots inherit a geometry centroid, then their graph parent.
        for index in range(count):
            if np.isfinite(pivots[index]).all():
                continue
            if owned[index]:
                pivots[index] = np.mean([mesh.centroid for mesh in owned[index]], axis=0)
            else:
                parent = parents[index]
                pivots[index] = pivots[parent] if parent is not None and np.isfinite(pivots[parent]).all() else np.zeros(3)

        combined = [trimesh.util.concatenate(parts) if parts else None for parts in owned]
        original_faces = sum(len(mesh.faces) for mesh in combined if mesh is not None)
        hand_dir = MESH_ROOT / hand.hand_id
        hand_dir.mkdir()
        nodes = []
        exported_faces = 0
        for index, mesh in enumerate(combined):
            mesh_path = None
            faces = 0
            if mesh is not None and not mesh.is_empty:
                share = max(
                    MIN_FACES_PER_NONEMPTY_LINK,
                    round(FACE_BUDGET_PER_HAND * len(mesh.faces) / original_faces),
                    round(MIN_RETAIN_RATIO * len(mesh.faces)),
                )
                mesh = decimate(mesh, min(share, len(mesh.faces)))
                mesh.apply_translation(-pivots[index])
                mesh = clean_mesh(mesh)
                mesh_path = f"mesh_library/{hand.hand_id}/node_{index:02d}.obj"
                mesh.export(OUTPUTS / mesh_path, file_type="obj", include_normals=True, include_color=False)
                faces = len(mesh.faces)
                exported_faces += faces
            parent = parents[index]
            relative = pivots[index] - (pivots[parent] if parent is not None else np.zeros(3))
            joint_offset = len(dataset["feature_names"]) - 9  # not used; retained for schema readability
            del joint_offset
            feature = np.asarray(record["x"][index], dtype=float)
            joint_slice = feature[9:15]
            joint_name = ["fixed", "hinge", "slide", "ball", "free", "other"][int(np.argmax(joint_slice))]
            nodes.append(
                {
                    "index": index,
                    "name": node_names[index],
                    "role": roles[index],
                    "parent": parent,
                    "relative_pos": relative.tolist(),
                    "pivot": pivots[index].tolist(),
                    "joint_type": joint_name,
                    "mesh": mesh_path,
                    "faces": faces,
                }
            )
        library["hands"][hand.hand_id] = {
            "display_name": hand.display_name,
            "source_format": hand.source_format,
            "source_faces": int(hand.faces),
            "library_faces": int(exported_faces),
            "target": record["target"],
            "digit_count": record["digit_count"],
            "scalar_dofs": record["scalar_dofs"],
            "nodes": nodes,
        }
        print(
            f"{hand.hand_id:20s}: source={hand.faces:8,d} library={exported_faces:6,d} "
            f"meshed_links={sum(node['mesh'] is not None for node in nodes):2d}/{count:2d}",
            flush=True,
        )

    LIBRARY.write_text(json.dumps(library, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {LIBRARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
