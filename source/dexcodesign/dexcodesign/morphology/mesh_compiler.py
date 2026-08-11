#!/usr/bin/env python3
"""Compile grammar-v1 HandIR parts into transformed candidate meshes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from .palm_geometry import (
    deform_template_palm,
    deform_template_to_house_palm,
    generate_house_hull_palm,
    generate_palm_mesh,
    infer_palm_params,
    patches_from_hand_ir,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACT_ROOT = ROOT / "artifacts" / "hand_morphology"
GENERATED_ROOT = Path(
    os.environ.get(
        "HAND_GENERATION_ROOT", str(ARTIFACT_ROOT / "generated_100")
    )
)
INPUT = GENERATED_ROOT / "hand_ir.json"
OUTPUT = GENERATED_ROOT / "compiled_hands.json"
MESH_ROOT = GENERATED_ROOT / "meshes"


@lru_cache(maxsize=64)
def load_source(relative: str) -> trimesh.Trimesh:
    mesh = trimesh.load(ARTIFACT_ROOT / relative, force="mesh", process=False)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"empty source mesh: {relative}")
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-id",
        action="append",
        default=[],
        help="compile only the selected generated hand ID (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="compiled HandIR JSON path",
    )
    parser.add_argument(
        "--preserve-existing-meshes",
        action="store_true",
        help="do not clear the shared compiled-mesh directory before a subset compile",
    )
    parser.add_argument(
        "--palm-only",
        action="store_true",
        help="compile only the candidate-specific palm mesh for parametric USD",
    )
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


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def apply_midas_axis_deformation(
    mesh: trimesh.Trimesh, specification: dict
) -> trimesh.Trimesh:
    """Stretch a MiDas link between rigid proximal/distal connector caps."""
    # Own the vertex buffer: callers may cache and reuse the undeformed source
    # mesh across many morphology candidates.
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    longitudinal = np.asarray(specification["longitudinal_axis"], dtype=np.float64)
    longitudinal /= np.linalg.norm(longitudinal)
    width = np.asarray(specification["width_axis"], dtype=np.float64)
    width -= float(np.dot(width, longitudinal)) * longitudinal
    width /= np.linalg.norm(width)
    source_length = float(specification["source_length_canonical"])
    target_length = float(specification["target_length_canonical"])
    proximal = float(specification["proximal_cap_fraction"]) * source_length
    distal = (1.0 - float(specification["distal_cap_fraction"])) * source_length
    if not 0.0 <= proximal < distal <= source_length:
        raise ValueError("invalid MiDas connector-cap fractions")
    target_middle = target_length - proximal - (source_length - distal)
    if target_middle <= 0.0:
        raise ValueError("MiDas target length would invert the protected connector caps")

    coordinate = vertices @ longitudinal
    mapped = coordinate.copy()
    middle = (coordinate > proximal) & (coordinate < distal)
    mapped[middle] = proximal + (coordinate[middle] - proximal) * (
        target_middle / (distal - proximal)
    )
    mapped[coordinate >= distal] = coordinate[coordinate >= distal] + (
        target_length - source_length
    )

    # Width changes only in the body span. Smooth blends keep both mechanical
    # interfaces source-exact; the orthogonal thickness coordinate is untouched.
    ramp_up = _smoothstep((coordinate - proximal) / max(0.20 * source_length, 1.0e-12))
    if bool(specification.get("distal_connector_fixed", True)):
        ramp_down = _smoothstep(
            (distal - coordinate) / max(0.20 * source_length, 1.0e-12)
        )
        body_weight = np.minimum(ramp_up, ramp_down)
    else:
        # The free fingertip end is part of the deformable body rather than a
        # nonexistent distal connector. Its full cross-section therefore
        # follows the requested width all the way to the terminal surface.
        body_weight = ramp_up
    width_multiplier = 1.0 + body_weight * (float(specification["width_scale"]) - 1.0)
    # Scale about the source link's own geometric centreline, not the mesh
    # coordinate origin.  MiDas rigid-part meshes are intentionally authored
    # with their motor frame at the origin, so the visual body is generally
    # offset from zero (most visibly for the distal fingertip links).  Scaling
    # the raw coordinate would therefore hold one side fixed and move only the
    # other side, producing a laterally crooked/asymmetric tip.  The projected
    # bounds midpoint is deterministic, identical for the three normal
    # fingers, and leaves both protected connector caps source-exact because
    # body_weight is zero there.
    width_projection = vertices @ width
    width_center = 0.5 * (
        float(np.min(width_projection)) + float(np.max(width_projection))
    )
    width_coordinate = width_projection - width_center
    vertices += (mapped - coordinate)[:, None] * longitudinal
    vertices += ((width_multiplier - 1.0) * width_coordinate)[:, None] * width
    result = mesh.copy()
    result.vertices = vertices
    return result


def apply_general_axis_deformation(
    mesh: trimesh.Trimesh, specification: dict
) -> trimesh.Trimesh:
    """Stretch one generic phalanx while keeping its motor interfaces rigid.

    Unlike the MiDas implementation, heterogeneous source hands do not expose
    dimensions in millimetres.  The source joint-to-joint distance is used when
    available; terminal links infer their longitudinal extent from the mesh.
    Width is an isotropic radial scale, shared according to the general grammar.
    """
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    axis = np.asarray(specification["longitudinal_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    coordinate = vertices @ axis
    source_min, source_max = float(np.min(coordinate)), float(np.max(coordinate))
    source_length = source_max - source_min
    if source_length <= 1.0e-8:
        raise ValueError("generic phalanx has no longitudinal extent")
    length_scale = float(specification["length_scale"])
    # Generic source CAD has no certified connector-cap locations.  Scaling
    # the complete joint-local rigid segment about its proximal joint is the
    # only source-independent transform that exactly matches the independently
    # scaled child-joint offset.  Guessing a cap from mesh bounds leaves some
    # multi-piece links behind and creates the visible floating seams.
    vertices += ((length_scale - 1.0) * coordinate)[:, None] * axis

    normalized = (coordinate - source_min) / source_length
    ramp_up = _smoothstep(
        normalized / 0.20
    )
    if bool(specification.get("distal_connector_fixed", True)):
        ramp_down = _smoothstep((1.0 - normalized) / 0.20)
        body_weight = np.minimum(ramp_up, ramp_down)
    else:
        body_weight = ramp_up

    # Scale around the phalanx's geometric centreline.  This avoids the hooked
    # fingertip artifact caused by scaling an offset CAD body about frame zero.
    bounds_center = 0.5 * (
        np.asarray(mesh.bounds[0], dtype=np.float64)
        + np.asarray(mesh.bounds[1], dtype=np.float64)
    )
    centreline_offset = bounds_center - float(np.dot(bounds_center, axis)) * axis
    axial = (vertices @ axis)[:, None] * axis
    radial = vertices - axial - centreline_offset
    radius_scale = float(specification["radius_scale"])
    vertices += ((radius_scale - 1.0) * body_weight)[:, None] * radial

    result = mesh.copy()
    result.vertices = vertices
    return result


def apply_midas_palm_deformation(
    mesh: trimesh.Trimesh, specification: dict
) -> trimesh.Trimesh:
    """Continuously resize the palm shell while preserving its wrist cap."""
    # Do not mutate a cached source palm while compiling another candidate.
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])
    height = z_max - z_min
    if height <= 0.0:
        raise ValueError("MiDas palm has no longitudinal extent")
    source_width = float(specification["source_width_mm"])
    target_width = float(specification["target_width_mm"])
    source_height = float(specification["source_height_mm"])
    target_height = float(specification["target_height_mm"])
    width_scale = target_width / source_width
    height_scale = target_height / source_height
    wrist_end = z_min + float(specification["wrist_cap_fraction"]) * height
    finger_start = z_max - float(specification["finger_cap_fraction"]) * height
    target_middle = height_scale * height - (wrist_end - z_min) - (z_max - finger_start)
    if target_middle <= 0.0:
        raise ValueError("MiDas palm height would invert protected end regions")

    z = vertices[:, 2].copy()
    mapped_z = z.copy()
    middle = (z > wrist_end) & (z < finger_start)
    mapped_z[middle] = wrist_end + (z[middle] - wrist_end) * (
        target_middle / (finger_start - wrist_end)
    )
    mapped_z[z >= finger_start] = z[z >= finger_start] + height * (height_scale - 1.0)
    width_weight = _smoothstep((z - wrist_end) / max(0.35 * height, 1.0e-12))
    width_multiplier = 1.0 + width_weight * (width_scale - 1.0)
    x_center = 0.5 * float(bounds[0, 0] + bounds[1, 0])
    vertices[:, 0] = x_center + width_multiplier * (vertices[:, 0] - x_center)
    vertices[:, 2] = mapped_z
    result = mesh.copy()
    result.vertices = vertices
    return result


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
    midas_specification = node.get("midas_palm_deformation")
    if midas_specification is not None:
        height_ratio = float(midas_specification["target_height_mm"]) / float(
            midas_specification["source_height_mm"]
        )
        # A shortened shell retains each rigid motor interface as a collar
        # extending inward from the graph attachment frame. This is an
        # analytic part of the grammar, not a connectivity repair: its size is
        # determined solely by palm-height contraction and it vanishes exactly
        # for the source/expanded palm.
        if height_ratio < 1.0 - 1.0e-12:
            collars = []
            for patch in patches:
                depth = max(
                    0.25 * float(patch.depth),
                    (1.0 - height_ratio) * 0.12,
                )
                transform = patch.transform.copy()
                transform[:3, 3] -= 0.5 * depth * transform[:3, 2]
                collars.append(
                    trimesh.creation.box(
                        extents=[patch.width, patch.thickness, depth],
                        transform=transform,
                    )
                )
            result.visual_mesh = trimesh.util.concatenate(
                [result.visual_mesh, *collars]
            )
            result.collision_mesh = trimesh.util.concatenate(
                [result.collision_mesh, *[collar.copy() for collar in collars]]
            )
            result.metadata["midas_short_palm_interface_collars"] = {
                "count": len(collars),
                "height_ratio": height_ratio,
                "source_exact_when_inactive": True,
            }
    # Exact equality is intentional: the generator may copy graph frames for
    # validation, but it may not rescale, deform, or re-estimate them.
    expected = {patch.name: patch.transform for patch in [*patches, wrist]}
    for name, transform in expected.items():
        if not np.array_equal(result.attachment_frames[name], transform):
            raise ValueError(f"palm generator modified graph attachment frame {name}")
    return result


def check_attachment_connection(hand: dict, parts: list[dict]) -> dict:
    """Fail fast when a meshed finger root is outside the generated palm."""
    palm_mesh = parts[0].get("compiled_mesh")
    if palm_mesh is None:
        return {
            "checked": 0,
            "unmeshed": len(hand["finger_slots"]),
            "nonoverlapping": [],
            "minimum_axis_overlap": None,
        }
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
    selected_ids = set(args.hand_id)
    input_hands = [
        hand for hand in payload["hands"]
        if not selected_ids or hand["hand_id"] in selected_ids
    ]
    missing = selected_ids - {hand["hand_id"] for hand in input_hands}
    if missing:
        raise ValueError(f"unknown generated hand IDs: {sorted(missing)}")
    if MESH_ROOT.exists() and not args.preserve_existing_meshes:
        shutil.rmtree(MESH_ROOT)
    total_faces = meshed = geometryless = 0
    output_hands = []
    for hand in input_hands:
        manifest = MESH_ROOT / hand["hand_id"] / "compiled.json"
        if args.preserve_existing_meshes and manifest.is_file():
            output = json.loads(manifest.read_text(encoding="utf-8"))
            output_hands.append(output)
            total_faces += int(output["mesh_summary"]["faces"])
            meshed += int(output["mesh_summary"]["meshed_parts"])
            geometryless += len(output["parts"]) - int(
                output["mesh_summary"]["meshed_parts"]
            )
            continue
        output = dict(hand)
        output_parts = []
        hand_faces = hand_meshes = 0
        binary_mesh_output = hand.get("grammar_id") == "midas-manufacturing-constraints-v1"
        for node in hand["parts"]:
            result = dict(node)
            source_mesh = node.get("source_mesh")
            if args.palm_only and not (
                int(node["id"]) == 0 and node.get("role") == "palm"
            ):
                result["compiled_mesh"] = None
                geometryless += 1
                output_parts.append(result)
                continue
            if source_mesh is None:
                result["compiled_mesh"] = None
                geometryless += 1
                output_parts.append(result)
                continue
            mesh = load_source(source_mesh["file"]).copy()
            linear = np.asarray(node["mesh_linear"], dtype=float)
            mesh.vertices = np.asarray(mesh.vertices, dtype=float) @ linear.T
            if "general_axis_deformation" in node:
                mesh = apply_general_axis_deformation(
                    mesh, node["general_axis_deformation"]
                )
            if "midas_axis_deformation" in node:
                mesh = apply_midas_axis_deformation(
                    mesh, node["midas_axis_deformation"]
                )
            if "midas_palm_deformation" in node:
                mesh = apply_midas_palm_deformation(
                    mesh, node["midas_palm_deformation"]
                )
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals(multibody=True)
            palm_result = None
            if int(node["id"]) == 0 and node.get("role") == "palm":
                try:
                    palm_result = compile_palm(hand, node, mesh, args)
                except ValueError as error:
                    raise ValueError(f"{hand['hand_id']}: {error}") from error
                mesh = palm_result.visual_mesh
            suffix = ".ply" if binary_mesh_output else ".obj"
            path = MESH_ROOT / hand["hand_id"] / f"part_{int(node['id']):02d}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if binary_mesh_output:
                mesh.export(path, file_type="ply", encoding="binary_little_endian")
            else:
                mesh.export(path, file_type="obj", include_normals=True, include_color=False)
            result["compiled_mesh"] = {
                "file": str(path.relative_to(GENERATED_ROOT)),
                "source_file": source_mesh["file"],
                "faces": int(len(mesh.faces)),
                "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
                "linear_transform": linear.tolist(),
                "candidate_id": node["candidate_id"],
                "mechanism_bundle_id": node["mechanism_bundle_id"],
            }
            if palm_result is not None:
                collision_path = MESH_ROOT / hand["hand_id"] / f"palm_collision{suffix}"
                if binary_mesh_output:
                    palm_result.collision_mesh.export(
                        collision_path,
                        file_type="ply",
                        encoding="binary_little_endian",
                    )
                else:
                    palm_result.collision_mesh.export(
                        collision_path,
                        file_type="obj",
                        include_normals=True,
                        include_color=False,
                    )
                result["compiled_mesh"]["collision_file"] = str(
                    collision_path.relative_to(GENERATED_ROOT)
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
                    invalid_interfaces = [
                        record for record in interfaces
                        if record["maximum_free_interface_frame_error"] > 1.0e-10
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
        output["palm_connection"] = check_attachment_connection(
            output, output_parts
        )
        connection = output["palm_connection"]
        if args.palm_generation_mode != "fixed_template" and (
            connection["nonoverlapping"]
        ):
            raise ValueError(
                f"{hand['hand_id']} has a disconnected finger root: "
                f"{connection['nonoverlapping']}"
            )
        manifest.write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        output_hands.append(output)
        # Some source CAD parts are very large. Keeping every source mesh from
        # all 100 designs in the process-wide cache can exceed laptop memory,
        # even though each hand compiles independently.
        load_source.cache_clear()
        gc.collect()
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
                hand.get("palm_connection", hand.get("palm_attachment_audit"))["checked"]
                for hand in output_hands
            ),
            "nonoverlapping_attachment_roots": sum(
                len(
                    hand.get(
                        "palm_connection", hand.get("palm_attachment_audit")
                    )["nonoverlapping"]
                )
                for hand in output_hands
            ),
            "semantic_palm_finger_interfaces": sum(
                len(hand["parts"][0]["palm_generation"].get(
                    "joint_interface_patches", []
                ))
                for hand in output_hands
            ),
            "maximum_palm_interface_frame_error": max(
                (
                    record["maximum_free_interface_frame_error"]
                    for hand in output_hands
                    for record in hand["parts"][0]["palm_generation"].get(
                        "joint_interface_patches", []
                    )
                ),
                default=0.0,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
