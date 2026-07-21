#!/usr/bin/env python3
"""Fit a searchable compound-primitive representation to each visual link.

The source-derived visual meshes are not modified.  This file produces the
low-dimensional geometry/collision layer used by morphology search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
LIBRARY = OUTPUTS / "mesh_library.json"
RESULT = OUTPUTS / "geometry_decomposition.json"
MAX_COMPONENTS = 6
POINT_BUDGET = 18_000
FACES_PER_COMPONENT = 4_000


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    rotation = vectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    local = centered @ rotation
    low, high = np.quantile(local, [0.005, 0.995], axis=0)
    local_center = 0.5 * (low + high)
    world_center = center + local_center @ rotation.T
    half_extents = np.maximum(0.5 * (high - low), 1.0e-4)
    return world_center, rotation, half_extents


def split_clusters(points: np.ndarray, count: int) -> list[np.ndarray]:
    clusters = [points]
    while len(clusters) < count:
        index = max(range(len(clusters)), key=lambda i: np.ptp(clusters[i], axis=0).max())
        selected = clusters.pop(index)
        if len(selected) < 40:
            clusters.append(selected)
            break
        center, rotation, _ = pca_frame(selected)
        coordinate = (selected - center) @ rotation[:, 0]
        pivot = np.median(coordinate)
        left, right = selected[coordinate <= pivot], selected[coordinate > pivot]
        if len(left) < 20 or len(right) < 20:
            clusters.append(selected)
            break
        clusters.extend((left, right))
    return clusters


def fit_component(points: np.ndarray) -> dict[str, object]:
    center, rotation, half_extents = pca_frame(points)
    ordered = np.sort(half_extents)[::-1]
    long_ratio = ordered[0] / max(ordered[1], 1.0e-6)
    round_ratio = ordered[0] / max(ordered[2], 1.0e-6)
    if long_ratio > 2.2:
        primitive = "capsule"
        radius = float(0.5 * (ordered[1] + ordered[2]))
        parameters = {"radius": radius, "half_length": float(max(ordered[0] - radius, 1.0e-4))}
    elif round_ratio < 1.45:
        primitive = "ellipsoid"
        parameters = {"radii": half_extents.tolist()}
    else:
        primitive = "box"
        parameters = {"half_extents": half_extents.tolist()}
    return {
        "type": primitive,
        "center": center.tolist(),
        "rotation": rotation.tolist(),
        **parameters,
        "support_points": int(len(points)),
    }


def main() -> int:
    library = json.loads(LIBRARY.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "purpose": "searchable geometry and collision proxy; visual mesh remains source-derived",
        "max_components_per_link": MAX_COMPONENTS,
        "hands": {},
    }
    rng = np.random.default_rng(11)
    total_links = total_components = 0
    for hand_id, hand in library["hands"].items():
        hand_records = []
        for node in hand["nodes"]:
            if not node.get("mesh"):
                hand_records.append({"node": node["index"], "name": node["name"], "components": []})
                continue
            mesh = trimesh.load(OUTPUTS / node["mesh"], force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=float)
            if len(vertices) > POINT_BUDGET:
                vertices = vertices[rng.choice(len(vertices), POINT_BUDGET, replace=False)]
            desired = int(np.clip(np.ceil(node["faces"] / FACES_PER_COMPONENT), 1, MAX_COMPONENTS))
            clusters = split_clusters(vertices, desired)
            components = [fit_component(cluster) for cluster in clusters]
            hand_records.append(
                {
                    "node": node["index"],
                    "name": node["name"],
                    "role": node["role"],
                    "visual_faces": node["faces"],
                    "components": components,
                }
            )
            total_links += 1
            total_components += len(components)
        result["hands"][hand_id] = hand_records
        print(
            f"{hand_id:20s}: meshed_links={sum(bool(x['components']) for x in hand_records):2d} "
            f"compound_parts={sum(len(x['components']) for x in hand_records):3d}",
            flush=True,
        )
    result["summary"] = {"meshed_links": total_links, "compound_components": total_components}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT}: {total_components} components over {total_links} visual links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
