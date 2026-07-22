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
    patches, wrist = patches_from_hand_ir(
        hand, fixed_mesh, params, root_mesh_loader=load_source
    )
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
    """Check that every meshed finger root actually enters the palm volume."""
    palm_mesh = parts[0].get("compiled_mesh")
    if palm_mesh is None:
        return {
            "checked": 0,
            "unmeshed": len(hand["finger_slots"]),
            "nonoverlapping": [],
            "surface_checked": 0,
            "surface_nonintersecting": [],
            "centering_violations": [],
            "sampled_root_vertices_inside_palm": 0,
            "minimum_tangential_coverage": None,
            "maximum_tangential_center_error": 0.0,
            "minimum_axis_overlap": None,
        }
    palm_bounds = np.asarray(palm_mesh["bounds"], dtype=float)
    collision_file = palm_mesh.get("collision_file")
    collision_mesh = None
    if collision_file is not None:
        collision_mesh = trimesh.load(
            HERE / "outputs" / collision_file, force="mesh", process=False
        )
        collision_mesh.vertices = (
            np.asarray(collision_mesh.vertices, dtype=float)
            + np.asarray(parts[0]["world_pos"], dtype=float)
        )
    checked = unmeshed = 0
    nonoverlapping = []
    surface_nonintersecting = []
    centering_violations = []
    root_vertices_inside = 0
    minimum_axis_overlap = float("inf")
    minimum_tangential_coverage = float("inf")
    maximum_tangential_center_error = 0.0
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
        if collision_mesh is not None:
            root_mesh = trimesh.load(
                HERE / "outputs" / compiled["file"], force="mesh", process=False
            )
            vertices = (
                np.asarray(root_mesh.vertices, dtype=float)
                + np.asarray(root["world_pos"], dtype=float)
            )
            # A deterministic bounded sample keeps this strict check affordable
            # even for source CAD links with hundreds of thousands of vertices.
            if len(vertices) > 1500:
                vertices = vertices[
                    np.linspace(0, len(vertices) - 1, 1500, dtype=int)
                ]
            signed_distance = trimesh.proximity.signed_distance(
                collision_mesh, vertices
            )
            inside = int(np.count_nonzero(signed_distance >= -1.0e-6))
            root_vertices_inside += inside
            if inside == 0:
                surface_nonintersecting.append({
                    "slot_id": int(slot["slot_id"]),
                    "role": slot["role"],
                    "maximum_signed_distance": float(np.max(signed_distance)),
                })
            else:
                inside_vertices = vertices[signed_distance >= -1.0e-6]
                outward = np.asarray(slot["attachment_rotation"], dtype=float)[:, 2][[0, 2]]
                outward /= max(float(np.linalg.norm(outward)), 1.0e-10)
                tangent = np.asarray([-outward[1], outward[0]], dtype=float)
                root_tangent = vertices[:, [0, 2]] @ tangent
                inside_tangent = inside_vertices[:, [0, 2]] @ tangent
                root_width = max(float(np.ptp(root_tangent)), 1.0e-10)
                coverage = float(np.ptp(inside_tangent)) / root_width
                center_error = abs(
                    0.5 * float(inside_tangent.min() + inside_tangent.max())
                    - 0.5 * float(root_tangent.min() + root_tangent.max())
                ) / root_width
                minimum_tangential_coverage = min(minimum_tangential_coverage, coverage)
                maximum_tangential_center_error = max(
                    maximum_tangential_center_error, center_error
                )
                if coverage < 0.45 or center_error > 0.28:
                    centering_violations.append({
                        "slot_id": int(slot["slot_id"]),
                        "role": slot["role"],
                        "tangential_coverage": coverage,
                        "tangential_center_error": center_error,
                    })
    return {
        "checked": checked,
        "unmeshed": unmeshed,
        "nonoverlapping": nonoverlapping,
        "surface_checked": 0 if collision_mesh is None else checked,
        "surface_nonintersecting": surface_nonintersecting,
        "centering_violations": centering_violations,
        "sampled_root_vertices_inside_palm": root_vertices_inside,
        "minimum_tangential_coverage": (
            None if minimum_tangential_coverage == float("inf")
            else minimum_tangential_coverage
        ),
        "maximum_tangential_center_error": maximum_tangential_center_error,
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
                if args.palm_generation_mode in {
                    "template_deform",
                    "source_topology_house",
                    "hybrid_source_topology",
                }:
                    interfaces = palm_result.metadata.get("joint_interface_patches", [])
                    if len(interfaces) != len(hand["finger_slots"]):
                        raise ValueError(
                            f"{hand['hand_id']} extracted {len(interfaces)} palm/finger "
                            f"interfaces for {len(hand['finger_slots'])} finger roots"
                        )
                    invalid_interfaces = [
                        record for record in interfaces
                        if record["maximum_free_interface_frame_error"] > 1.0e-10
                        or record["maximum_locked_interface_conflict"] > 1.0e-10
                    ]
                    if invalid_interfaces:
                        raise ValueError(
                            f"{hand['hand_id']} has palm/finger interface frames that do not "
                            f"follow their graph joints: {invalid_interfaces}"
                        )
            hand_faces += int(len(mesh.faces))
            hand_meshes += 1
            total_faces += int(len(mesh.faces))
            meshed += 1
            output_parts.append(result)
        output["parts"] = output_parts
        output["mesh_summary"] = {"meshed_parts": hand_meshes, "faces": hand_faces}
        output["palm_attachment_audit"] = audit_attachment_overlap(output, output_parts)
        attachment_audit = output["palm_attachment_audit"]
        if args.palm_generation_mode != "fixed_template" and (
            attachment_audit["nonoverlapping"]
            or attachment_audit["surface_nonintersecting"]
            or attachment_audit["centering_violations"]
        ):
            raise ValueError(
                f"{hand['hand_id']} has finger-root visuals that do not intersect the generated palm: "
                f"bounds={attachment_audit['nonoverlapping']}, "
                f"surface={attachment_audit['surface_nonintersecting']}, "
                f"centering={attachment_audit['centering_violations']}"
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
            "surface_nonintersecting_attachment_roots": sum(
                len(hand["palm_attachment_audit"]["surface_nonintersecting"])
                for hand in output_hands
            ),
            "off_center_attachment_roots": sum(
                len(hand["palm_attachment_audit"]["centering_violations"])
                for hand in output_hands
            ),
            "minimum_tangential_root_coverage": min(
                hand["palm_attachment_audit"]["minimum_tangential_coverage"]
                for hand in output_hands
                if hand["palm_attachment_audit"]["minimum_tangential_coverage"] is not None
            ),
            "maximum_tangential_root_center_error": max(
                hand["palm_attachment_audit"]["maximum_tangential_center_error"]
                for hand in output_hands
            ),
            "sampled_root_vertices_inside_palms": sum(
                hand["palm_attachment_audit"]["sampled_root_vertices_inside_palm"]
                for hand in output_hands
            ),
            "semantic_palm_finger_interfaces": sum(
                len(hand["parts"][0]["palm_generation"].get(
                    "joint_interface_patches", []
                ))
                for hand in output_hands
            ),
            "maximum_palm_interface_frame_error": max(
                record["maximum_free_interface_frame_error"]
                for hand in output_hands
                for record in hand["parts"][0]["palm_generation"].get(
                    "joint_interface_patches", []
                )
            ),
            "palm_interface_mount_lock_conflicts": sum(
                record["maximum_locked_interface_conflict"] > 1.0e-10
                for hand in output_hands
                for record in hand["parts"][0]["palm_generation"].get(
                    "joint_interface_patches", []
                )
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
