"""Deterministic source-bound candidate mesh compiler."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from .grammar import validate_hand
from .schema import HandIR, ModuleDatabase


@lru_cache(maxsize=256)
def _load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"empty source candidate mesh: {path}")
    return mesh


def _world_positions(hand: HandIR) -> dict[int, np.ndarray]:
    children: dict[int, list] = {}
    child_ids = set()
    for joint in hand.joints:
        children.setdefault(joint.parent_node, []).append(joint)
        child_ids.add(joint.child_node)
    root = next(node.node_id for node in hand.nodes if node.node_id not in child_ids)
    result = {root: np.zeros(3)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for joint in children.get(parent, []):
            result[joint.child_node] = result[parent] + np.asarray(joint.origin_translation, dtype=float)
            stack.append(joint.child_node)
    return result


def _longitudinal_axis(hand: HandIR, node_id: int) -> np.ndarray:
    nodes = {node.node_id: node for node in hand.nodes}
    children = [
        joint for joint in hand.joints
        if joint.parent_node == node_id and nodes[joint.child_node].semantic_role == nodes[node_id].semantic_role
    ]
    if children:
        axis = max(children, key=lambda joint: np.linalg.norm(joint.origin_translation)).origin_translation
    else:
        parent_joint = next((joint for joint in hand.joints if joint.child_node == node_id), None)
        axis = (0.0, 0.0, 1.0) if parent_joint is None else parent_joint.origin_translation
    value = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(value))
    return np.asarray([0.0, 0.0, 1.0]) if norm < 1.0e-8 else value / norm


def compile_hand(hand: HandIR, database: ModuleDatabase, repo_root: Path, output_dir: Path) -> dict:
    audit = validate_hand(hand, database, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = _world_positions(hand)
    parts = []
    for node in hand.nodes:
        candidate = None if node.candidate_id is None else database.candidates[node.candidate_id]
        mesh_record = None
        if candidate is not None and candidate.visual_path is not None:
            mesh = _load_mesh(str(repo_root / candidate.visual_path)).copy()
            vertices = np.asarray(mesh.vertices, dtype=float)
            if node.semantic_role in {"thumb", "index", "middle", "ring", "pinky"}:
                axis = _longitudinal_axis(hand, node.node_id)
                matrix = node.radial_scale * np.eye(3) + (node.length_scale - node.radial_scale) * np.outer(axis, axis)
            else:
                matrix = np.diag(np.asarray(node.palm_scale, dtype=float))
            mesh.vertices = vertices @ matrix.T
            mesh.remove_unreferenced_vertices()
            path = output_dir / f"part_{node.node_id:02d}.obj"
            mesh.export(path, file_type="obj", include_normals=True, include_color=False)
            mesh_record = {
                "path": str(path),
                "candidate_id": candidate.candidate_id,
                "bundle_id": candidate.bundle_id,
                "source_part_id": candidate.source_part_id,
                "faces": int(len(mesh.faces)),
                "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
            }
        parts.append(
            {
                "node_id": node.node_id,
                "role": node.semantic_role,
                "world_pos": world[node.node_id].tolist(),
                "bundle_id": node.bundle_id,
                "mesh": mesh_record,
            }
        )
    return {"hand": hand.to_dict(), "audit": audit, "parts": parts}
