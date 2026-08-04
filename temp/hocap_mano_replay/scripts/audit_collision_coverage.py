#!/usr/bin/env python3
"""Audit visible-part collision coverage for MANO and all robot hands."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
REGISTRY = DIRECT_ROOT / "registry.json"
DEFAULT_PREPARED_ROOT = (
    REPO_ROOT / "artifacts" / "isaaclab_all_hands_residual" / "prepared"
)
DEFAULT_OBJECT_MESH = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "models"
    / "G04_1"
    / "cleaned_mesh_2000.obj"
)


def obj_counts(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            vertices += line.startswith("v ")
            faces += line.startswith("f ")
    return vertices, faces


def audit_urdf(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    visual_elements = 0
    collision_elements = 0
    physical_links = 0
    collision_links = 0
    solid_primitive_collision_links: list[str] = []
    missing_collision_visuals: list[dict[str, object]] = []
    invalid_mesh_files: list[str] = []

    for link in root.findall("link"):
        visuals = link.findall("visual")
        collisions = link.findall("collision")
        visual_elements += len(visuals)
        collision_elements += len(collisions)
        physical_links += bool(visuals)
        collision_links += bool(collisions)
        if any(
            collision.find("geometry/box") is not None
            or collision.find("geometry/sphere") is not None
            or collision.find("geometry/cylinder") is not None
            or collision.find("geometry/capsule") is not None
            for collision in collisions
        ):
            solid_primitive_collision_links.append(str(link.get("name")))
        deficit = max(0, len(visuals) - len(collisions))
        if deficit:
            missing_collision_visuals.append(
                {
                    "link": link.get("name"),
                    "visual_elements": len(visuals),
                    "collision_elements": len(collisions),
                    "missing": deficit,
                }
            )
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                invalid_mesh_files.append(f"{link.get('name')}:<missing filename>")
                continue
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = (path.parent / mesh_path).resolve()
            if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
                invalid_mesh_files.append(str(mesh_path))
                continue
            if mesh_path.suffix.lower() == ".obj":
                vertices, faces = obj_counts(mesh_path)
                if vertices == 0 or faces == 0:
                    invalid_mesh_files.append(str(mesh_path))

    return {
        "urdf": str(path),
        "physical_links": physical_links,
        "visual_elements": visual_elements,
        "collision_links": collision_links,
        "collision_elements": collision_elements,
        "solid_primitive_collision_links": sorted(
            solid_primitive_collision_links
        ),
        "missing_collision_count": sum(
            int(item["missing"]) for item in missing_collision_visuals
        ),
        "missing_collision_visuals": missing_collision_visuals,
        "invalid_mesh_files": sorted(set(invalid_mesh_files)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--object-mesh", type=Path, default=DEFAULT_OBJECT_MESH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete-prepared", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hands: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for hand_id, hand in registry["hands"].items():
        if hand_id == "supr_female_foot":
            continue
        source_path = REPO_ROOT / "assets" / "robot_hands" / hand["entries"]["left"]["path"]
        source = audit_urdf(source_path)
        result: dict[str, object] = {"source": source}
        if hand_id == "mano":
            result["prepared"] = source
        else:
            prepared_path = args.prepared_root / hand_id / "hand_rl.urdf"
            prepared = audit_urdf(prepared_path)
            result["prepared"] = prepared
            if args.require_complete_prepared and prepared["missing_collision_count"]:
                failures.append(
                    f"{hand_id}: {prepared['missing_collision_count']} visible parts "
                    "still lack collision"
                )
            if prepared["invalid_mesh_files"]:
                failures.append(f"{hand_id}: invalid prepared mesh files")
            if hand_id == "tesollo_dg5f" and (
                "hand__ll_dg_palm"
                not in prepared["solid_primitive_collision_links"]
            ):
                failures.append(
                    "tesollo_dg5f: palm lacks a closed primitive collision core"
                )
        hands[hand_id] = result

    object_mesh = args.object_mesh.resolve()
    object_vertices, object_faces = obj_counts(object_mesh)
    object_result = {
        "mesh": str(object_mesh),
        "vertices": object_vertices,
        "faces": object_faces,
        "collision_conversion": "convexDecomposition",
    }
    if object_vertices == 0 or object_faces == 0:
        failures.append("object mesh is empty")

    report = {
        "criterion": (
            "every visible physical-part element has collision geometry; "
            "zero-length joint-frame links are excluded"
        ),
        "hands": hands,
        "object": object_result,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"HAND_COLLISION_AUDIT output={args.output} failures={len(failures)}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
