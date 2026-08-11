"""Source-local simulation morphology schemas shared by all robot hands.

Different hands deliberately have different vector dimensions.  A schema only
contains parameters that affect geometry actually present in that source hand;
there are no padded/masked digit slots.  Every source keeps an ordered bank of
32 palm prototypes which is reused by all candidates derived from that source.
"""

from __future__ import annotations

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NORMAL_FINGERS = ("index", "middle", "ring", "pinky")
PHALANGES = ("proximal", "middle", "distal")
PALM_EXPANSION_LEVELS = 32
PALM_EXPANSION_MAX = 0.70


def phalanx_stages(part_count: int, *, protected_root: bool) -> tuple[str, ...]:
    """Return physical phalanx slots represented by a source finger bundle."""
    editable = max(0, int(part_count) - int(protected_root))
    if editable >= 3:
        return PHALANGES
    if editable == 2:
        return ("proximal", "distal")
    if editable == 1:
        return ("distal",)
    return ()


def build_schema(
    source_hand: str,
    stages_by_finger: dict[str, tuple[str, ...]],
    *,
    palm_affine_editable: bool,
) -> dict:
    """Build the exact, unpadded vector schema for one source hand."""
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
        for stage in stages_by_finger.get(finger, ()):
            parameters.append({
                "name": f"{finger}_{stage}_length",
                "bounds": [-1.0, 1.0],
                "source": 0.0,
            })

    normal_stages = {
        stage
        for finger in NORMAL_FINGERS
        for stage in stages_by_finger.get(finger, ())
    }
    thumb_stages = set(stages_by_finger.get("thumb", ()))
    if normal_stages & {"proximal", "middle"}:
        parameters.append({"name": "normal_body_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    if "distal" in normal_stages:
        parameters.append({"name": "normal_distal_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    if thumb_stages & {"proximal", "middle"}:
        parameters.append({"name": "thumb_body_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    if "distal" in thumb_stages:
        parameters.append({"name": "thumb_distal_radius", "bounds": [-1.0, 1.0], "source": 0.0})
    return {
        "grammar_id": "general-simulation-hand-v2",
        "source_hand": source_hand,
        "vector_dimension": len(parameters),
        "parameters": parameters,
        "vector_names": [record["name"] for record in parameters],
        "stages_by_finger": {
            finger: list(stages_by_finger.get(finger, ())) for finger in FINGERS
        },
        "palm_prototype_count": PALM_EXPANSION_LEVELS,
        "palm_prototype_reuse": "one source-local bank shared by every candidate",
        "zero_vector_is_source": True,
    }


def _asymmetric_scale(value: float, lower: float, upper: float) -> float:
    return 1.0 + value * ((1.0 - lower) if value < 0.0 else (upper - 1.0))


def decode_vector(vector: np.ndarray | list[float], schema: dict) -> dict:
    """Decode one source-local vector into generic graph controls."""
    latent = np.asarray(vector, dtype=np.float64)
    expected = int(schema["vector_dimension"])
    if latent.shape != (expected,):
        raise ValueError(f"{schema['source_hand']} vector needs {expected} values")
    bounds = np.asarray([record["bounds"] for record in schema["parameters"]], dtype=float)
    if not np.all(np.isfinite(latent)) or np.any(latent < bounds[:, 0]) or np.any(latent > bounds[:, 1]):
        raise ValueError(f"{schema['source_hand']} vector exceeds its source-local bounds")
    raw = dict(zip(schema["vector_names"], latent, strict=True))
    expansion_level = int(np.rint(raw["palm_expansion"] * (PALM_EXPANSION_LEVELS - 1)))
    palm = {
        "prototype_index": expansion_level,
        "expansion": expansion_level * PALM_EXPANSION_MAX / (PALM_EXPANSION_LEVELS - 1),
        "scale_x": _asymmetric_scale(raw.get("palm_scale_x", 0.0), 0.70, 1.45),
        "scale_z": _asymmetric_scale(raw.get("palm_scale_z", 0.0), 0.70, 1.45),
        "yaw": raw.get("palm_yaw", 0.0) * 0.35,
    }
    fingers = {}
    for finger in FINGERS:
        stages = schema["stages_by_finger"].get(finger, [])
        if not stages:
            continue
        values = {stage: 1.0 for stage in PHALANGES}
        for stage in stages:
            values[stage] = _asymmetric_scale(
                raw[f"{finger}_{stage}_length"], 0.55, 1.60
            )
        fingers[finger] = {"length_scales": [values[stage] for stage in PHALANGES]}
    width = {
        "normal_body": _asymmetric_scale(raw.get("normal_body_radius", 0.0), 0.60, 1.45),
        "normal_distal": _asymmetric_scale(raw.get("normal_distal_radius", 0.0), 0.60, 1.45),
        "thumb_body": _asymmetric_scale(raw.get("thumb_body_radius", 0.0), 0.60, 1.45),
        "thumb_distal": _asymmetric_scale(raw.get("thumb_distal_radius", 0.0), 0.60, 1.45),
    }
    return {"palm": palm, "fingers": fingers, "width_scales": width}


def latin_hypercube_vectors(count: int, schema: dict, seed: int) -> np.ndarray:
    """Sample a source-local schema, keeping row zero exactly at the source."""
    if count <= 0:
        raise ValueError("count must be positive")
    dimension = int(schema["vector_dimension"])
    vectors = np.zeros((count, dimension), dtype=np.float64)
    if count == 1:
        return vectors
    rng = np.random.default_rng(seed)
    samples = count - 1
    bounds = np.asarray([record["bounds"] for record in schema["parameters"]], dtype=float)
    for column in range(dimension):
        strata = (np.arange(samples) + rng.random(samples)) / samples
        rng.shuffle(strata)
        vectors[1:, column] = bounds[column, 0] + strata * (
            bounds[column, 1] - bounds[column, 0]
        )
    return vectors
