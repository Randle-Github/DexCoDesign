#!/usr/bin/env python3
"""WUJI optimizer encoding for the canonical general hand grammar.

This module is an adapter, not a third grammar. It binds ``wuji_hand_2`` to
``dexcodesign.morphology.general_grammar`` and exposes the mixed search encoding
used by the existing SAC/CEM orchestration: column zero is an integer palm
prototype index and every remaining column is the exact canonical latent
coordinate. Graph compilation always goes back through the general decoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
from dexcodesign.morphology.general_grammar import (
    FINGERS,
    PALM_EXPANSION_MAX,
    PALM_PROTOTYPES,
    build_schema,
    decode_vector,
    graph_spec_from_vector,
    palm_prototype_coordinate,
    palm_prototype_index,
    source_vector,
    vector_bounds,
)
from dexcodesign.morphology.generate import editable_finger_segments


SOURCE_HAND_ID = "wuji_hand_2"
MORPHOLOGY_ROOT = REPO_ROOT / "artifacts" / "hand_morphology"


def _build_wuji_schema() -> dict:
    source_payload = json.loads(
        (MORPHOLOGY_ROOT / "reference_graphs.json").read_text(encoding="utf-8")
    )
    library = json.loads(
        (MORPHOLOGY_ROOT / "mechanism_bundles.json").read_text(encoding="utf-8")
    )
    hand = next(
        record for record in source_payload["hands"]
        if record["hand_id"] == SOURCE_HAND_ID
    )
    bundles = {
        record["bundle_id"]: record for record in library["bundles"]
    }
    segments = {
        finger: tuple(editable_finger_segments(
            hand, bundles[f"{SOURCE_HAND_ID}:{finger}"]
        ))
        for finger in FINGERS
    }
    return build_schema(
        SOURCE_HAND_ID,
        segments,
        palm_affine_editable=True,
    )


SCHEMA = _build_wuji_schema()
PALM_EXPANSION_LEVELS = PALM_PROTOTYPES
PALM_EXPANSION_MIN = 0.0
VECTOR_NAMES = (
    "palm_prototype_index",
    *SCHEMA["vector_names"][1:],
)
_LATENT_LOWER, _LATENT_UPPER = vector_bounds(SCHEMA)
LOWER_BOUNDS = np.concatenate(([0.0], _LATENT_LOWER[1:]))
UPPER_BOUNDS = np.concatenate(
    ([float(PALM_EXPANSION_LEVELS - 1)], _LATENT_UPPER[1:])
)
SOURCE_VECTOR = np.concatenate(([0.0], source_vector(SCHEMA)[1:]))
CONTINUOUS_LOWER_BOUNDS = LOWER_BOUNDS[1:]
CONTINUOUS_UPPER_BOUNDS = UPPER_BOUNDS[1:]
CONTINUOUS_SOURCE_VECTOR = SOURCE_VECTOR[1:]
PARAMETER_INDEX = {name: index for index, name in enumerate(VECTOR_NAMES)}
FINGER_LENGTH_INDICES = {
    finger: tuple(
        PARAMETER_INDEX[f"{finger}_segment_{source_id}_length"]
        for source_id in SCHEMA["segment_ids_by_finger"][finger]
    )
    for finger in FINGERS
}
NORMAL_DISTAL_WIDTH_INDEX = PARAMETER_INDEX["normal_distal_width"]
THUMB_DISTAL_WIDTH_INDEX = PARAMETER_INDEX["thumb_distal_width"]


def palm_expansion_from_index(index: np.ndarray | float | int) -> np.ndarray:
    value = np.asarray(index, dtype=np.float64)
    return np.rint(value) * (
        PALM_EXPANSION_MAX / (PALM_EXPANSION_LEVELS - 1)
    )


def validate_design_vectors(vectors: np.ndarray) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float64)
    if value.shape[-1] != len(VECTOR_NAMES):
        raise ValueError(
            f"WUJI general-grammar vectors need {len(VECTOR_NAMES)} values, "
            f"got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("morphology vectors must be finite")
    if np.any(value < LOWER_BOUNDS - 1.0e-6) or np.any(
        value > UPPER_BOUNDS + 1.0e-6
    ):
        raise ValueError("morphology vectors exceed general-grammar bounds")
    if np.any(np.abs(value[..., 0] - np.rint(value[..., 0])) > 1.0e-6):
        raise ValueError("palm_prototype_index must be an integer")
    result = np.clip(value, LOWER_BOUNDS, UPPER_BOUNDS).copy()
    result[..., 0] = np.rint(result[..., 0])
    return result


def search_to_latent(vectors: np.ndarray) -> np.ndarray:
    result = validate_design_vectors(vectors).copy()
    result[..., 0] = np.vectorize(palm_prototype_coordinate)(
        result[..., 0].astype(np.int64)
    )
    return result


def latent_to_search(vectors: np.ndarray) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float64)
    if value.shape[-1] != int(SCHEMA["vector_dimension"]):
        raise ValueError("canonical general-hand vector has the wrong dimension")
    result = value.copy()
    result[..., 0] = np.vectorize(palm_prototype_index)(result[..., 0])
    return validate_design_vectors(result)


def _physical_record(latent: np.ndarray) -> np.ndarray:
    decoded = decode_vector(latent, SCHEMA)
    physical: dict[str, float] = {
        "palm_expansion": float(decoded["palm"]["expansion"]),
        "palm_scale_x": float(decoded["palm"]["scale_x"]),
        "palm_scale_z": float(decoded["palm"]["scale_z"]),
        "palm_yaw": float(decoded["palm"]["yaw"]),
        "normal_body_width": float(decoded["width_scales"]["normal_body"]),
        "normal_distal_width": float(decoded["width_scales"]["normal_distal"]),
        "thumb_body_width": float(decoded["width_scales"]["thumb_body"]),
        "thumb_distal_width": float(decoded["width_scales"]["thumb_distal"]),
    }
    for finger, record in decoded["fingers"].items():
        for source_id, value in record["length_scales_by_source_part"].items():
            physical[f"{finger}_segment_{source_id}_length"] = float(value)
    return np.asarray(
        [physical["palm_expansion"]]
        + [physical[name] for name in VECTOR_NAMES[1:]],
        dtype=np.float64,
    )


def resolve_design_vectors(vectors: np.ndarray) -> np.ndarray:
    latent = search_to_latent(vectors)
    flat = latent.reshape(-1, latent.shape[-1])
    resolved = np.stack([_physical_record(row) for row in flat], axis=0)
    return resolved.reshape(latent.shape)


def graph_from_search_vector(vector: np.ndarray, hand_id: str) -> dict:
    latent = search_to_latent(np.asarray(vector, dtype=np.float64))
    return graph_spec_from_vector(latent, SCHEMA, hand_id=hand_id)


def sample_mixed_vectors(
    rng: np.random.Generator,
    count: int,
    palm_probabilities: np.ndarray,
    continuous_mean_normalized: np.ndarray,
    continuous_sigma_normalized: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(palm_probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    palm = rng.choice(PALM_EXPANSION_LEVELS, size=count, p=probabilities)
    normalized = np.clip(
        rng.normal(
            continuous_mean_normalized,
            continuous_sigma_normalized,
            size=(count, len(CONTINUOUS_SOURCE_VECTOR)),
        ),
        0.0,
        1.0,
    )
    continuous = CONTINUOUS_LOWER_BOUNDS + normalized * (
        CONTINUOUS_UPPER_BOUNDS - CONTINUOUS_LOWER_BOUNDS
    )
    return np.concatenate((palm[:, None], continuous), axis=1)
