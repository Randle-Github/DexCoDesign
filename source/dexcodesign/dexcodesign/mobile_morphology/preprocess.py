"""Precompute connector-span link variants for fast morphology search.

Each eligible URDF link is the structural span between movable connectors.
The proximal mesh cap is locked, the distal cap follows the rewritten graph
edge, and only their middle span deforms. Discrete variants are
content-addressed and shared by every generated robot.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from . import SCHEMA_VERSION
from .common import (
    GRAMMAR,
    PRECOMPUTED_ROOT,
    PRECOMPUTED_VARIANTS,
    ROOT,
    SPEC_BY_ID,
    longitudinal_linear,
    numbers,
    rpy_matrix,
)


CACHE_VERSION = "connector-span-manual-interface-ownership-v5"


def variant_key(
    source_id: str,
    production: str,
    link: str,
    factor: float,
) -> str:
    return (
        f"{source_id}|{production}|{link}|{float(factor):.10g}"
    )


@lru_cache(maxsize=None)
def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owner_mesh_variant(
    *,
    source_id: str,
    link_name: str,
    owner_kind: str,
    owner_index: int,
    owner: ET.Element,
    source_dir: Path,
    linear: np.ndarray,
    factor: float,
    production: str,
    connector_rig: dict | None,
) -> dict | None:
    mesh_element = owner.find("./geometry/mesh")
    if mesh_element is None or not mesh_element.get("filename"):
        return None
    source_path = Path(mesh_element.get("filename", ""))
    if not source_path.is_absolute():
        source_path = (source_dir / source_path).resolve()
    scale = numbers(mesh_element.get("scale"), (1, 1, 1))
    origin = owner.find("origin")
    xyz = numbers(None if origin is None else origin.get("xyz"))
    rotation = rpy_matrix(numbers(None if origin is None else origin.get("rpy")))
    fingerprint_payload = {
        "cache_version": CACHE_VERSION,
        "source_sha256": file_digest(str(source_path)),
        "source_id": source_id,
        "production": production,
        "link": link_name,
        "owner_kind": owner_kind,
        "owner_index": owner_index,
        "factor": float(factor),
        "mesh_scale": scale.tolist(),
        "origin_xyz": xyz.tolist(),
        "origin_rotation": rotation.tolist(),
        "linear": linear.tolist(),
        "connector_rig": connector_rig,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    safe_link = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in link_name
    )
    output = (
        PRECOMPUTED_ROOT
        / source_id
        / safe_link
        / f"{owner_kind}_{owner_index:02d}_{fingerprint[:16]}.obj"
    )
    cache_hit = output.is_file()
    if not cache_hit:
        loaded = trimesh.load(source_path, force="mesh", process=False)
        if loaded.is_empty or not len(loaded.faces):
            raise ValueError(f"empty source mesh: {source_path}")
        mesh = loaded.copy()
        vertices = np.asarray(mesh.vertices, dtype=float) * scale
        vertices = vertices @ rotation.T + xyz
        if connector_rig is None:
            mesh.vertices = vertices @ linear.T
        else:
            direction = np.asarray(
                connector_rig["direction"], dtype=float
            )
            coordinate = vertices @ direction
            start = float(connector_rig["middle_start"])
            end = float(connector_rig["middle_end"])
            alpha = np.clip(
                (coordinate - start) / max(end - start, 1.0e-9),
                0.0,
                1.0,
            )
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            displacement = np.asarray(
                connector_rig["distal_displacement"], dtype=float
            )
            # The proximal connector cap is exactly rigid, the distal cap
            # follows the same graph-edge displacement as the child frame,
            # and only the structural middle changes length.
            mesh.vertices = vertices + alpha[:, None] * displacement
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals(multibody=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(
            output,
            file_type="obj",
            include_normals=True,
            include_color=False,
        )
    return {
        "owner_kind": owner_kind,
        "owner_index": owner_index,
        "file": str(output.relative_to(ROOT)),
        "fingerprint": fingerprint,
        "cache_hit": cache_hit,
    }


def owner_link_vertices(owner: ET.Element, source_dir: Path) -> np.ndarray | None:
    mesh_element = owner.find("./geometry/mesh")
    if mesh_element is None or not mesh_element.get("filename"):
        return None
    source_path = Path(mesh_element.get("filename", ""))
    if not source_path.is_absolute():
        source_path = (source_dir / source_path).resolve()
    mesh = trimesh.load(source_path, force="mesh", process=False)
    if mesh.is_empty:
        return None
    scale = numbers(mesh_element.get("scale"), (1, 1, 1))
    origin = owner.find("origin")
    xyz = numbers(None if origin is None else origin.get("xyz"))
    rotation = rpy_matrix(numbers(None if origin is None else origin.get("rpy")))
    return (
        np.asarray(mesh.vertices, dtype=float) * scale
    ) @ rotation.T + xyz


def build_connector_rig(
    link: ET.Element,
    source_dir: Path,
    member: dict,
    factor: float,
) -> dict:
    """Precompute a parent-locked/middle/distal span rig for one graph node."""
    axis = np.asarray(member["deformation_axis"], dtype=float)
    axis /= np.linalg.norm(axis)
    connector = np.asarray(member["connector_vector"], dtype=float)
    sign = 1.0 if float(connector @ axis) >= 0.0 else -1.0
    direction = sign * axis
    visual_vertices = [
        vertices
        for owner in link.findall("visual")
        if (vertices := owner_link_vertices(owner, source_dir)) is not None
    ]
    owners = visual_vertices or [
        vertices
        for owner in link.findall("collision")
        if (vertices := owner_link_vertices(owner, source_dir)) is not None
    ]
    if not owners:
        raise ValueError(
            f"{link.get('name')}: length node has no mesh geometry"
        )
    coordinate = np.concatenate(
        [vertices @ direction for vertices in owners]
    )
    lower, upper = float(coordinate.min()), float(coordinate.max())
    # The link origin is the graph's proximal joint frame. If source CAD does
    # not reach that origin (RBY1 wrist/arm links), lock the closest actual
    # mesh surface instead of scaling about an empty point in space.
    proximal = float(np.clip(0.0, lower, upper))
    connector_coordinate = float(connector @ direction)
    distal = float(np.clip(connector_coordinate, lower, upper))
    if distal <= proximal + 1.0e-5:
        raise ValueError(
            f"{link.get('name')}: invalid connector span rig "
            f"proximal={proximal} distal={distal}"
        )
    effective_span = distal - proximal
    cap = min(0.025, 0.15 * effective_span)
    linear = longitudinal_linear(axis, factor)
    return {
        "direction": direction.tolist(),
        "source_projection_bounds": [lower, upper],
        "proximal_connector_projection": proximal,
        "distal_connector_projection": distal,
        "middle_start": proximal + cap,
        "middle_end": distal - cap,
        "distal_displacement": (
            linear @ connector - connector
        ).tolist(),
        "proximal_cap_locked": True,
        "distal_cap_follows_graph_edge": True,
        "deforms_middle_only": True,
    }


def main() -> int:
    grammar_payload = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    variants = {}
    cache_hits = cache_misses = 0
    for grammar in grammar_payload["robots"]:
        source_id = grammar["source_id"]
        spec = SPEC_BY_ID[source_id]
        robot = ET.parse(spec.urdf).getroot()
        links = {link.get("name", ""): link for link in robot.findall("link")}
        for production_name in (
            "vertical_link_length",
            "shoulder_width",
        ):
            production = grammar["productions"][production_name]
            for group in production["groups"]:
                for member in group["members"]:
                    link_name = member["link"]
                    link = links[link_name]
                    for factor in group.get(
                        "factors", production["factors"]
                    ):
                        linear = longitudinal_linear(
                            member["deformation_axis"], factor
                        )
                        connector_rig = (
                            build_connector_rig(
                                link,
                                spec.urdf.parent,
                                member,
                                factor,
                            )
                            if production_name == "vertical_link_length"
                            else None
                        )
                        records = []
                        for owner_kind in ("visual", "collision"):
                            for owner_index, owner in enumerate(
                                link.findall(owner_kind)
                            ):
                                record = owner_mesh_variant(
                                    source_id=source_id,
                                    link_name=link_name,
                                    owner_kind=owner_kind,
                                    owner_index=owner_index,
                                    owner=owner,
                                    source_dir=spec.urdf.parent,
                                    linear=linear,
                                    factor=factor,
                                    production=production_name,
                                    connector_rig=connector_rig,
                                )
                                if record is None:
                                    continue
                                records.append(record)
                                if record["cache_hit"]:
                                    cache_hits += 1
                                else:
                                    cache_misses += 1
                        key = variant_key(
                            source_id,
                            production_name,
                            link_name,
                            factor,
                        )
                        variants[key] = {
                            "source_id": source_id,
                            "production": production_name,
                            "link": link_name,
                            "factor": float(factor),
                            "deformation_axis": member[
                                "deformation_axis"
                            ],
                            "linear": linear.tolist(),
                            "connector_rig": connector_rig,
                            "mesh_owners": records,
                        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "method": (
            "root-outward connector rigs: length locks the proximal cap and "
            "moves the distal cap; joint orientation edits are excluded"
        ),
        "variants": variants,
        "summary": {
            "variants": len(variants),
            "mesh_owner_records": sum(
                len(variant["mesh_owners"]) for variant in variants.values()
            ),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "runtime_mesh_deformation_required": False,
        },
    }
    PRECOMPUTED_VARIANTS.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
