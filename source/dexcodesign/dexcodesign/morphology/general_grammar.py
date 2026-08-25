"""The single source of truth for simulation-hand morphology.

MiDas manufacturing constraints intentionally live in ``midas_grammar.py``.
Every other robot hand, including WUJI morphology SAC, consumes this schema and
decoder. Geometry is implemented by the shared connector-cap-preserving
compiler; algorithms may choose different optimizers, but may not redefine the
morphology vector.
"""

from __future__ import annotations

import numpy as np


GRAMMAR_ID = "general-simulation-hand-v3"


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NORMAL_FINGERS = ("index", "middle", "ring", "pinky")
PALM_PROTOTYPES = 32
# The 32 ordered prototypes span the complete source-to-star fusion bank used
# by morphology search.  Earlier all-hand experiments accidentally stopped at
# 0.35, which made neighbouring prototypes visually almost indistinguishable.
# 0.70 is the already validated WUJI search range: the source motor mounts stay
# fixed-size while the palm surface and complete finger-root frames move
# together toward the radial target.
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
        "kind": "ordered_32_source_star_fusion_selector",
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
        "grammar_id": GRAMMAR_ID,
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


def vector_bounds(schema: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return optimizer bounds in the exact canonical vector order."""
    bounds = np.asarray(
        [record["bounds"] for record in schema["parameters"]],
        dtype=np.float64,
    )
    return bounds[:, 0], bounds[:, 1]


def source_vector(schema: dict) -> np.ndarray:
    """Return the exact source-hand vector (all source values are zero)."""
    return np.asarray(
        [record["source"] for record in schema["parameters"]],
        dtype=np.float64,
    )


def palm_prototype_index(value: float) -> int:
    """Quantize the ordered palm coordinate to one of 32 prototypes."""
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("palm prototype coordinate must lie in [0, 1]")
    return int(np.rint(value * (PALM_PROTOTYPES - 1)))


def palm_prototype_coordinate(index: int) -> float:
    """Return the canonical vector coordinate of an ordered prototype."""
    if not 0 <= int(index) < PALM_PROTOTYPES:
        raise ValueError(f"palm prototype index must lie in [0, {PALM_PROTOTYPES - 1}]")
    return int(index) / (PALM_PROTOTYPES - 1)


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
    prototype = palm_prototype_index(raw["palm_expansion"])
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


def graph_spec_from_vector(
    vector: np.ndarray | list[float],
    schema: dict,
    *,
    hand_id: str,
) -> dict:
    """Translate one canonical vector into the shared graph-compiler request."""
    decoded = decode_vector(vector, schema)
    prototype = int(decoded["palm"]["prototype_index"])
    return {
        "hand_id": hand_id,
        "source_hand": schema["source_hand"],
        "palm": {
            "layout_mode": (
                "source_fixed" if prototype == 0 else "source_star_fusion"
            ),
            "prototype_bank_id": f"{schema['source_hand']}:palm32",
            **decoded["palm"],
        },
        "fingers": {
            **decoded["fingers"],
            "width_scales": decoded["width_scales"],
        },
        "general_morphology_vector": np.asarray(vector, dtype=float).tolist(),
        "general_morphology_vector_names": list(schema["vector_names"]),
        "grammar_id": schema["grammar_id"],
    }


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
