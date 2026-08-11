"""Source-local grammar parameters for the established hand deformation path.

This module only groups optimizer parameters.  Geometry continues to use the
existing source-bundle affine transforms and palm compiler; it does not define
or implement a second mesh deformation algorithm.
"""

from __future__ import annotations

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NORMAL_FINGERS = ("index", "middle", "ring", "pinky")
PALM_EXPANSION_LEVELS = 32
PALM_EXPANSION_MAX = 0.70


def build_schema(
    source_hand: str,
    part_ids_by_finger: dict[str, tuple[int, ...]],
    *,
    palm_affine_editable: bool,
    protected_finger_root: bool,
) -> dict:
    """Build an unpadded vector from the source's actual rigid finger parts."""
    editable = {}
    for finger in FINGERS:
        ids = list(part_ids_by_finger.get(finger, ()))
        if protected_finger_root and ids:
            ids = ids[1:]
        editable[finger] = ids

    parameters = [{
        "name": "palm_expansion",
        "bounds": [0.0, 1.0],
        "source": 0.0,
        "kind": "ordered_32_prototype_selector",
    }]
    if palm_affine_editable:
        parameters.extend((
            {"name": "palm_scale_x", "bounds": [-1.0, 1.0], "source": 0.0},
            {"name": "palm_scale_z", "bounds": [-1.0, 1.0], "source": 0.0},
            {"name": "palm_yaw", "bounds": [-1.0, 1.0], "source": 0.0},
        ))
    for finger in FINGERS:
        for source_part_id in editable[finger]:
            parameters.append({
                "name": f"{finger}_part_{source_part_id}_length",
                "bounds": [-1.0, 1.0],
                "source": 0.0,
                "source_part_id": source_part_id,
            })

    normal_ids = [source_id for finger in NORMAL_FINGERS for source_id in editable[finger]]
    thumb_ids = editable["thumb"]
    if len(normal_ids) > 1:
        parameters.append({"name": "normal_body_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    if normal_ids:
        parameters.append({"name": "normal_distal_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    if len(thumb_ids) > 1:
        parameters.append({"name": "thumb_body_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    if thumb_ids:
        parameters.append({"name": "thumb_distal_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    return {
        "grammar_id": "general-simulation-hand-v2",
        "source_hand": source_hand,
        "vector_dimension": len(parameters),
        "parameters": parameters,
        "vector_names": [record["name"] for record in parameters],
        "source_part_ids_by_finger": {
            finger: list(part_ids_by_finger.get(finger, ())) for finger in FINGERS
        },
        "editable_part_ids_by_finger": editable,
        "protected_finger_root": protected_finger_root,
        "palm_prototype_count": PALM_EXPANSION_LEVELS,
        "zero_vector_is_source": True,
    }


def _scale(value: float, lower: float, upper: float) -> float:
    return 1.0 + value * ((1.0 - lower) if value < 0.0 else (upper - 1.0))


def decode_vector(vector: np.ndarray | list[float], schema: dict) -> dict:
    latent = np.asarray(vector, dtype=np.float64)
    if latent.shape != (int(schema["vector_dimension"]),):
        raise ValueError(f"{schema['source_hand']} vector has the wrong dimension")
    bounds = np.asarray([record["bounds"] for record in schema["parameters"]], dtype=float)
    if not np.all(np.isfinite(latent)) or np.any(latent < bounds[:, 0]) or np.any(latent > bounds[:, 1]):
        raise ValueError(f"{schema['source_hand']} vector exceeds its bounds")
    raw = dict(zip(schema["vector_names"], latent, strict=True))
    prototype = int(np.rint(raw["palm_expansion"] * (PALM_EXPANSION_LEVELS - 1)))
    result = {
        "palm": {
            "prototype_index": prototype,
            "expansion": prototype * PALM_EXPANSION_MAX / (PALM_EXPANSION_LEVELS - 1),
            # Preserve the proven general-preview ranges. Wider explicit graph
            # requests remain supported by generate.py, but the normalized
            # optimizer vector must not silently change their prior.
            "scale_x": _scale(raw.get("palm_scale_x", 0.0), 0.93, 1.10),
            "scale_z": _scale(raw.get("palm_scale_z", 0.0), 0.92, 1.13),
            "yaw": raw.get("palm_yaw", 0.0) * 0.16,
        },
        "fingers": {},
    }
    for finger in FINGERS:
        values = {
            str(source_id): _scale(raw[f"{finger}_part_{source_id}_length"], 0.87, 1.17)
            for source_id in schema["editable_part_ids_by_finger"][finger]
        }
        if values:
            result["fingers"][finger] = {"length_scales_by_source_part": values}
    result["width_scales"] = {
        "normal_body": _scale(raw.get("normal_body_radius", 0.0), 0.88, 1.14),
        "normal_distal": _scale(raw.get("normal_distal_radius", 0.0), 0.88, 1.14),
        "thumb_body": _scale(raw.get("thumb_body_radius", 0.0), 0.88, 1.14),
        "thumb_distal": _scale(raw.get("thumb_distal_radius", 0.0), 0.88, 1.14),
    }
    return result


def latin_hypercube_vectors(count: int, schema: dict, seed: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    vectors = np.zeros((count, int(schema["vector_dimension"])), dtype=np.float64)
    if count == 1:
        return vectors
    rng = np.random.default_rng(seed)
    samples = count - 1
    bounds = np.asarray([record["bounds"] for record in schema["parameters"]], dtype=float)
    for column in range(vectors.shape[1]):
        strata = (np.arange(samples) + rng.random(samples)) / samples
        rng.shuffle(strata)
        vectors[1:, column] = bounds[column, 0] + strata * (bounds[column, 1] - bounds[column, 0])
    return vectors
