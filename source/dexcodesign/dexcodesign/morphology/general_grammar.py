"""Source-local general hand grammar built on rigid-cap segment deformation.

The module only defines optimizer parameters.  Geometry is implemented by the
same connector-cap-preserving deformation used by the MiDas manufacturing
grammar, while legacy whole-finger graph requests remain untouched.
"""

from __future__ import annotations

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NORMAL_FINGERS = ("index", "middle", "ring", "pinky")
PALM_PROTOTYPES = 32
PALM_EXPANSION_MAX = 0.70


def build_schema(
    source_hand: str,
    segment_ids_by_finger: dict[str, tuple[int, ...]],
    *,
    palm_affine_editable: bool,
) -> dict:
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
        for source_part_id in segment_ids_by_finger.get(finger, ()):
            parameters.append({
                "name": f"{finger}_segment_{source_part_id}_length",
                "bounds": [-1.0, 1.0],
                "source": 0.0,
                "source_part_id": int(source_part_id),
            })

    normal_segments = [
        source_id
        for finger in NORMAL_FINGERS
        for source_id in segment_ids_by_finger.get(finger, ())
    ]
    thumb_segments = list(segment_ids_by_finger.get("thumb", ()))
    if normal_segments:
        parameters.extend((
            {"name": "normal_body_width", "bounds": [-1.0, 1.0], "source": 0.0},
            {"name": "normal_distal_width", "bounds": [-1.0, 1.0], "source": 0.0},
        ))
    if thumb_segments:
        parameters.extend((
            {"name": "thumb_body_width", "bounds": [-1.0, 1.0], "source": 0.0},
            {"name": "thumb_distal_width", "bounds": [-1.0, 1.0], "source": 0.0},
        ))
    return {
        "grammar_id": "general-simulation-hand-v2",
        "source_hand": source_hand,
        "vector_dimension": len(parameters),
        "vector_names": [record["name"] for record in parameters],
        "parameters": parameters,
        "segment_ids_by_finger": {
            finger: list(segment_ids_by_finger.get(finger, ()))
            for finger in FINGERS
        },
        "palm_prototype_count": PALM_PROTOTYPES,
        "zero_vector_is_source": True,
        "auxiliary_linkages_are_not_phalanx_parameters": True,
    }


def _scale(value: float, lower: float, upper: float) -> float:
    return 1.0 + value * ((1.0 - lower) if value < 0.0 else (upper - 1.0))


def decode_vector(vector: np.ndarray | list[float], schema: dict) -> dict:
    latent = np.asarray(vector, dtype=np.float64)
    if latent.shape != (int(schema["vector_dimension"]),):
        raise ValueError(f"{schema['source_hand']} vector has the wrong dimension")
    if not np.all(np.isfinite(latent)):
        raise ValueError("general morphology vector must be finite")
    bounds = np.asarray([record["bounds"] for record in schema["parameters"]])
    if np.any(latent < bounds[:, 0]) or np.any(latent > bounds[:, 1]):
        raise ValueError("general morphology vector exceeds its bounds")
    raw = dict(zip(schema["vector_names"], latent, strict=True))
    prototype = int(np.rint(raw["palm_expansion"] * (PALM_PROTOTYPES - 1)))
    result = {
        "palm": {
            "prototype_index": prototype,
            "expansion": prototype * PALM_EXPANSION_MAX / (PALM_PROTOTYPES - 1),
            "scale_x": _scale(raw.get("palm_scale_x", 0.0), 0.93, 1.10),
            "scale_z": _scale(raw.get("palm_scale_z", 0.0), 0.92, 1.13),
            "yaw": raw.get("palm_yaw", 0.0) * 0.16,
        },
        "fingers": {},
        "width_scales": {
            "normal_body": _scale(raw.get("normal_body_width", 0.0), 0.60, 1.45),
            "normal_distal": _scale(raw.get("normal_distal_width", 0.0), 0.60, 1.45),
            "thumb_body": _scale(raw.get("thumb_body_width", 0.0), 0.60, 1.45),
            "thumb_distal": _scale(raw.get("thumb_distal_width", 0.0), 0.60, 1.45),
        },
    }
    for finger in FINGERS:
        segment_ids = schema["segment_ids_by_finger"][finger]
        if segment_ids:
            result["fingers"][finger] = {
                "length_scales_by_source_part": {
                    str(source_id): _scale(
                        raw[f"{finger}_segment_{source_id}_length"], 0.55, 1.60
                    )
                    for source_id in segment_ids
                }
            }
    return result


def latin_hypercube_vectors(count: int, schema: dict, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    vectors = np.zeros((count, int(schema["vector_dimension"])), dtype=np.float64)
    if count == 1:
        return vectors
    rng = np.random.default_rng(seed)
    samples = count - 1
    bounds = np.asarray([record["bounds"] for record in schema["parameters"]])
    for column in range(vectors.shape[1]):
        values = (np.arange(samples) + rng.random(samples)) / samples
        rng.shuffle(values)
        vectors[1:, column] = bounds[column, 0] + values * (
            bounds[column, 1] - bounds[column, 0]
        )
    return vectors
