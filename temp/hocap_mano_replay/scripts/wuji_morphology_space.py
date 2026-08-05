#!/usr/bin/env python3
"""Canonical mixed discrete/continuous WUJI morphology design space."""

from __future__ import annotations

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PALM_EXPANSION_LEVELS = 32
PALM_EXPANSION_MIN = 0.0
PALM_EXPANSION_MAX = 0.35
VECTOR_NAMES = (
    "palm_expansion_index",
    "palm_scale_x",
    "palm_scale_z",
    "palm_yaw",
    *(f"{finger}_length" for finger in FINGERS),
    *(f"{finger}_radius" for finger in FINGERS),
)
LOWER_BOUNDS = np.asarray(
    [0.0, 0.90, 0.90, -0.12, *([0.82] * 5), *([0.85] * 5)],
    dtype=np.float64,
)
UPPER_BOUNDS = np.asarray(
    [PALM_EXPANSION_LEVELS - 1, 1.12, 1.12, 0.12, *([1.20] * 5), *([1.15] * 5)],
    dtype=np.float64,
)
SOURCE_VECTOR = np.asarray(
    [0.0, 1.0, 1.0, 0.0, *([1.0] * 10)], dtype=np.float64
)
CONTINUOUS_LOWER_BOUNDS = LOWER_BOUNDS[1:]
CONTINUOUS_UPPER_BOUNDS = UPPER_BOUNDS[1:]
CONTINUOUS_SOURCE_VECTOR = SOURCE_VECTOR[1:]


def palm_expansion_from_index(index: np.ndarray | float | int) -> np.ndarray:
    """Resolve a prototype index to its certified physical expansion value."""
    value = np.asarray(index, dtype=np.float64)
    return PALM_EXPANSION_MIN + np.rint(value) * (
        (PALM_EXPANSION_MAX - PALM_EXPANSION_MIN) / (PALM_EXPANSION_LEVELS - 1)
    )


def validate_design_vectors(vectors: np.ndarray) -> np.ndarray:
    """Validate raw vectors while preserving the discrete prototype index."""
    value = np.asarray(vectors)
    if value.shape[-1] != len(VECTOR_NAMES):
        raise ValueError(
            f"morphology vectors need {len(VECTOR_NAMES)} values, got {value.shape}"
        )
    if np.any(value < LOWER_BOUNDS - 1.0e-6) or np.any(
        value > UPPER_BOUNDS + 1.0e-6
    ):
        raise ValueError("morphology vectors exceed certified bounds")
    if np.any(np.abs(value[..., 0] - np.rint(value[..., 0])) > 1.0e-6):
        raise ValueError("palm_expansion_index must be an integer prototype index")
    result = np.clip(value, LOWER_BOUNDS, UPPER_BOUNDS).copy()
    result[..., 0] = np.rint(result[..., 0])
    return result


def resolve_design_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return physical vectors, replacing the palm index by expansion [0, 0.35]."""
    result = validate_design_vectors(vectors).copy()
    result[..., 0] = palm_expansion_from_index(result[..., 0])
    return result


def sample_mixed_vectors(
    rng: np.random.Generator,
    count: int,
    palm_probabilities: np.ndarray,
    continuous_mean_normalized: np.ndarray,
    continuous_sigma_normalized: np.ndarray,
) -> np.ndarray:
    palm_probabilities = np.asarray(palm_probabilities, dtype=np.float64)
    palm_probabilities = palm_probabilities / palm_probabilities.sum()
    indices = rng.choice(PALM_EXPANSION_LEVELS, size=count, p=palm_probabilities)
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
    return np.concatenate((indices[:, None], continuous), axis=1)

