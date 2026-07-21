#!/usr/bin/env python3
"""Build an aggressively simplified, searchable atomic hand library.

Each ordinary rigid link is represented by one fitted primitive. Palm/base
links may use up to three primitives when a single atom cannot cover their
compound geometry. The exported meshes are deliberately low-poly diagnostics;
the searchable representation is the primitive type and parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from decompose_geometry import fit_component, split_clusters


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SOURCE_LIBRARY = OUTPUTS / "mesh_library.json"
ATOMIC_ROOT = OUTPUTS / "atomic_library"
ATOMIC_LIBRARY = OUTPUTS / "atomic_library.json"
POINT_BUDGET = 12_000


def component_count(node: dict) -> int:
    faces = int(node.get("faces", 0))
    role = node.get("role", "other")
    if role == "palm":
        return 3 if faces > 80_000 else 2 if faces > 8_000 else 1
    if role == "base":
        return 2 if faces > 50_000 else 1
    return 1


def rotation_transform(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform


def atom_mesh(component: dict) -> trimesh.Trimesh:
    center = np.asarray(component["center"], dtype=float)
    rotation = np.asarray(component["rotation"], dtype=float)
    if component["type"] == "box":
        mesh = trimesh.creation.box(extents=2.0 * np.asarray(component["half_extents"], dtype=float))
        mesh.apply_transform(rotation_transform(rotation, center))
    elif component["type"] == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
        mesh.vertices *= np.asarray(component["radii"], dtype=float)
        mesh.apply_transform(rotation_transform(rotation, center))
    elif component["type"] == "capsule":
        mesh = trimesh.creation.capsule(
            radius=float(component["radius"]),
            height=2.0 * float(component["half_length"]),
            count=[6, 4],
        )
        # fit_component stores its major axis in column 0, while trimesh's
        # capsule is axial in local Z. A cyclic permutation preserves handedness.
        capsule_rotation = rotation[:, [1, 2, 0]]
        mesh.apply_transform(rotation_transform(capsule_rotation, center))
    else:
        raise ValueError(component["type"])
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def main() -> int:
    source = json.loads(SOURCE_LIBRARY.read_text(encoding="utf-8"))
    rng = np.random.default_rng(17)
    ATOMIC_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "extreme morphology-search proxy: 1 atom/link, up to 3 for palm/base",
        "primitive_vocabulary": ["box", "capsule", "ellipsoid"],
        "hands": {},
    }
    total_atoms = total_faces = 0
    for hand_id, hand in source["hands"].items():
        hand_dir = ATOMIC_ROOT / hand_id
        hand_dir.mkdir(parents=True, exist_ok=True)
        output_nodes = []
        hand_atoms = hand_faces = 0
        for node in hand["nodes"]:
            record = {
                "index": node["index"],
                "name": node["name"],
                "role": node["role"],
                "parent": node["parent"],
                "relative_pos": node["relative_pos"],
                "pivot": node["pivot"],
                "joint_type": node["joint_type"],
                "atoms": [],
                "mesh": None,
                "faces": 0,
            }
            if node.get("mesh"):
                source_mesh = trimesh.load(OUTPUTS / node["mesh"], force="mesh", process=False)
                points = np.asarray(source_mesh.vertices, dtype=float)
                if len(points) > POINT_BUDGET:
                    points = points[rng.choice(len(points), POINT_BUDGET, replace=False)]
                clusters = split_clusters(points, component_count(node))
                components = [fit_component(cluster) for cluster in clusters]
                meshes = [atom_mesh(component) for component in components]
                atomic_mesh = trimesh.util.concatenate(meshes)
                mesh_path = hand_dir / f"node_{node['index']:02d}.obj"
                atomic_mesh.export(mesh_path, file_type="obj", include_normals=True, include_color=False)
                record["atoms"] = components
                record["mesh"] = str(mesh_path.relative_to(OUTPUTS))
                record["faces"] = int(len(atomic_mesh.faces))
                hand_atoms += len(components)
                hand_faces += len(atomic_mesh.faces)
            output_nodes.append(record)
        payload["hands"][hand_id] = {
            "display_name": hand["display_name"],
            "digit_count": hand["digit_count"],
            "scalar_dofs": hand["scalar_dofs"],
            "nodes": output_nodes,
            "atom_count": hand_atoms,
            "triangle_faces": hand_faces,
        }
        total_atoms += hand_atoms
        total_faces += hand_faces
        print(f"{hand_id:20s}: atoms={hand_atoms:2d} lowpoly_faces={hand_faces:4d}", flush=True)
    payload["summary"] = {
        "hands": len(payload["hands"]),
        "atoms": total_atoms,
        "lowpoly_triangle_faces": total_faces,
    }
    ATOMIC_LIBRARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ATOMIC_LIBRARY}: atoms={total_atoms}, faces={total_faces}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
