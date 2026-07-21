#!/usr/bin/env python3
"""Compile grammar-v1 HandIR parts into transformed candidate meshes."""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
STRICT_OUTPUTS = HERE.parent / "strict_v2" / "outputs"
INPUT = HERE / "outputs" / "generated_hand_ir.json"
OUTPUT = HERE / "outputs" / "compiled_hands.json"
MESH_ROOT = HERE / "outputs" / "meshes"


@lru_cache(maxsize=512)
def load_source(relative: str) -> trimesh.Trimesh:
    mesh = trimesh.load(STRICT_OUTPUTS / relative, force="mesh", process=False)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"empty source mesh: {relative}")
    return mesh


def main() -> int:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    if MESH_ROOT.exists():
        shutil.rmtree(MESH_ROOT)
    total_faces = meshed = geometryless = 0
    output_hands = []
    for hand in payload["hands"]:
        output = dict(hand)
        output_parts = []
        hand_faces = hand_meshes = 0
        for node in hand["parts"]:
            result = dict(node)
            source_mesh = node.get("source_mesh")
            if source_mesh is None:
                result["compiled_mesh"] = None
                geometryless += 1
                output_parts.append(result)
                continue
            mesh = load_source(source_mesh["file"]).copy()
            linear = np.asarray(node["mesh_linear"], dtype=float)
            mesh.vertices = np.asarray(mesh.vertices, dtype=float) @ linear.T
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals(multibody=True)
            path = MESH_ROOT / hand["hand_id"] / f"part_{int(node['id']):02d}.obj"
            path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(path, file_type="obj", include_normals=True, include_color=False)
            result["compiled_mesh"] = {
                "file": str(path.relative_to(HERE / "outputs")),
                "source_file": source_mesh["file"],
                "faces": int(len(mesh.faces)),
                "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
                "linear_transform": linear.tolist(),
                "candidate_id": node["candidate_id"],
                "mechanism_bundle_id": node["mechanism_bundle_id"],
            }
            hand_faces += int(len(mesh.faces))
            hand_meshes += 1
            total_faces += int(len(mesh.faces))
            meshed += 1
            output_parts.append(result)
        output["parts"] = output_parts
        output["mesh_summary"] = {"meshed_parts": hand_meshes, "faces": hand_faces}
        output_hands.append(output)
    result = {
        "schema_version": 1,
        "method": "grammar-valid complete mechanism bundles + connector-aware palm/slot transforms",
        "hands": output_hands,
        "summary": {
            "hands": len(output_hands),
            "parts": sum(len(hand["parts"]) for hand in output_hands),
            "meshed_parts": meshed,
            "geometryless_zero_length_frames": geometryless,
            "faces": total_faces,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

