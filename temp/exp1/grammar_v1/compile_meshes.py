#!/usr/bin/env python3
"""Compile grammar-v1 HandIR parts into transformed candidate meshes."""

from __future__ import annotations

import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from palm_generator import (
    deform_template_palm,
    deform_template_to_house_palm,
    generate_house_hull_palm,
    generate_palm_mesh,
    infer_palm_params,
    patches_from_hand_ir,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--palm-generation-mode",
        choices=(
            "fixed_template", "attachment_hull", "parametric_2_5d",
            "template_deform", "house_hull", "hybrid_house",
            "source_topology_house", "hybrid_source_topology",
        ),
        default="hybrid_source_topology",
        help="fixed_template preserves the legacy palm; other modes consume graph attachment frames",
    )
    parser.add_argument("--transverse-arch", type=float, default=0.0)
    parser.add_argument("--longitudinal-arch", type=float, default=0.0)
    parser.add_argument("--central-cup", type=float, default=0.0)
    return parser.parse_args()


def compile_palm(hand: dict, node: dict, fixed_mesh: trimesh.Trimesh, args: argparse.Namespace):
    params = infer_palm_params(
        fixed_mesh,
        transverse_arch=args.transverse_arch,
        longitudinal_arch=args.longitudinal_arch,
        central_cup=args.central_cup,
    )
    patches, wrist = patches_from_hand_ir(hand, fixed_mesh, params)
    linear = np.asarray(node["mesh_linear"], dtype=float)
    scales = np.linalg.norm(linear, axis=0)
    palm_rotation = linear @ np.diag(1.0 / scales)
    layout_mode = hand.get("palm_layout", {}).get("mode", "anthropomorphic")
    if args.palm_generation_mode == "source_topology_house" or (
        args.palm_generation_mode == "hybrid_source_topology"
        and layout_mode not in {"anthropomorphic", "source_fixed"}
    ):
        result = deform_template_to_house_palm(
            hand, patches, wrist, fixed_mesh, palm_rotation
        )
    elif args.palm_generation_mode == "house_hull" or (
        args.palm_generation_mode == "hybrid_house" and layout_mode != "anthropomorphic"
    ):
        result = generate_house_hull_palm(
            hand, patches, wrist, fixed_mesh, palm_rotation
        )
    elif args.palm_generation_mode in {
        "template_deform", "hybrid_house", "hybrid_source_topology",
    }:
        result = deform_template_palm(hand, patches, wrist, fixed_mesh, palm_rotation)
    else:
        result = generate_palm_mesh(
            patches,
            wrist,
            params,
            mode=args.palm_generation_mode,
            fixed_template_mesh=fixed_mesh,
        )
    # Exact equality is intentional: the generator may copy graph frames for
    # validation, but it may not rescale, deform, or re-estimate them.
    expected = {patch.name: patch.transform for patch in [*patches, wrist]}
    for name, transform in expected.items():
        if not np.array_equal(result.attachment_frames[name], transform):
            raise ValueError(f"palm generator modified graph attachment frame {name}")
    return result


def audit_attachment_overlap(hand: dict, parts: list[dict]) -> dict:
    """Conservative visual check that every meshed finger root reaches palm."""
    palm_mesh = parts[0].get("compiled_mesh")
    if palm_mesh is None:
        return {"checked": 0, "unmeshed": len(hand["finger_slots"]), "nonoverlapping": []}
    palm_bounds = np.asarray(palm_mesh["bounds"], dtype=float)
    checked = unmeshed = 0
    nonoverlapping = []
    minimum_axis_overlap = float("inf")
    for slot in hand["finger_slots"]:
        root = parts[int(slot["root_node_id"])]
        compiled = root.get("compiled_mesh")
        if compiled is None:
            unmeshed += 1
            continue
        root_bounds = np.asarray(compiled["bounds"], dtype=float)
        root_bounds += np.asarray(root["world_pos"], dtype=float)
        overlap = np.minimum(palm_bounds[1], root_bounds[1]) - np.maximum(palm_bounds[0], root_bounds[0])
        checked += 1
        minimum_axis_overlap = min(minimum_axis_overlap, float(overlap.min()))
        if np.any(overlap < -1.0e-6):
            nonoverlapping.append({
                "slot_id": int(slot["slot_id"]),
                "role": slot["role"],
                "axis_overlap": overlap.tolist(),
            })
    return {
        "checked": checked,
        "unmeshed": unmeshed,
        "nonoverlapping": nonoverlapping,
        "minimum_axis_overlap": None if checked == 0 else minimum_axis_overlap,
    }


def main() -> int:
    args = parse_args()
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
            palm_result = None
            if int(node["id"]) == 0 and node.get("role") == "palm":
                try:
                    palm_result = compile_palm(hand, node, mesh, args)
                except ValueError as error:
                    raise ValueError(f"{hand['hand_id']}: {error}") from error
                mesh = palm_result.visual_mesh
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
            if palm_result is not None:
                collision_path = MESH_ROOT / hand["hand_id"] / "palm_collision.obj"
                palm_result.collision_mesh.export(
                    collision_path, file_type="obj", include_normals=True, include_color=False
                )
                result["compiled_mesh"]["collision_file"] = str(
                    collision_path.relative_to(HERE / "outputs")
                )
                result["compiled_mesh"]["collision_faces"] = int(
                    len(palm_result.collision_mesh.faces)
                )
                result["palm_generation"] = {
                    **palm_result.metadata,
                    "attachment_frames": {
                        name: transform.tolist()
                        for name, transform in palm_result.attachment_frames.items()
                    },
                    "joint_frame_invariance": True,
                }
            hand_faces += int(len(mesh.faces))
            hand_meshes += 1
            total_faces += int(len(mesh.faces))
            meshed += 1
            output_parts.append(result)
        output["parts"] = output_parts
        output["mesh_summary"] = {"meshed_parts": hand_meshes, "faces": hand_faces}
        output["palm_attachment_audit"] = audit_attachment_overlap(output, output_parts)
        if args.palm_generation_mode != "fixed_template" and output["palm_attachment_audit"]["nonoverlapping"]:
            raise ValueError(
                f"{hand['hand_id']} has finger-root visuals that do not overlap the generated palm: "
                f"{output['palm_attachment_audit']['nonoverlapping']}"
            )
        output_hands.append(output)
    result = {
        "schema_version": 1,
        "method": "grammar-valid complete mechanism bundles + attachment-conditioned palm compiler",
        "palm_generation_mode": args.palm_generation_mode,
        "hands": output_hands,
        "summary": {
            "hands": len(output_hands),
            "parts": sum(len(hand["parts"]) for hand in output_hands),
            "meshed_parts": meshed,
            "geometryless_zero_length_frames": geometryless,
            "faces": total_faces,
            "attachment_roots_checked": sum(
                hand["palm_attachment_audit"]["checked"] for hand in output_hands
            ),
            "nonoverlapping_attachment_roots": sum(
                len(hand["palm_attachment_audit"]["nonoverlapping"]) for hand in output_hands
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
