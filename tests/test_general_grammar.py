from __future__ import annotations

import numpy as np

from dexcodesign.morphology.general_grammar import (
    FINGERS,
    build_schema,
    decode_vector,
    latin_hypercube_vectors,
)


def test_source_local_dimensions_are_not_padded() -> None:
    five = build_schema(
        "five",
        {finger: (10 * index + 1, 10 * index + 2, 10 * index + 3)
         for index, finger in enumerate(FINGERS)},
        palm_affine_editable=True,
        protected_finger_root=False,
    )
    four_short = build_schema(
        "four_short",
        {finger: (index + 1,) for index, finger in enumerate(FINGERS[:4])},
        palm_affine_editable=False,
        protected_finger_root=False,
    )
    assert five["vector_dimension"] == 23
    assert four_short["vector_dimension"] == 8


def test_zero_vector_is_exact_source_and_prototype_zero() -> None:
    schema = build_schema(
        "source",
        {"thumb": (1, 2, 3, 4), "index": (5,)},
        palm_affine_editable=True,
        protected_finger_root=True,
    )
    decoded = decode_vector(np.zeros(schema["vector_dimension"]), schema)
    assert decoded["palm"]["prototype_index"] == 0
    assert decoded["palm"]["scale_x"] == 1.0
    assert decoded["palm"]["scale_z"] == 1.0
    assert decoded["palm"]["yaw"] == 0.0
    assert all(value == 1.0 for value in decoded["width_scales"].values())
    assert all(
        scale == 1.0
        for finger in decoded["fingers"].values()
        for scale in finger["length_scales_by_source_part"].values()
    )


def test_samples_respect_source_local_schema_and_32_levels() -> None:
    schema = build_schema(
        "source",
        {finger: (10 * index + 1, 10 * index + 2, 10 * index + 3)
         for index, finger in enumerate(FINGERS[:4])},
        palm_affine_editable=False,
        protected_finger_root=False,
    )
    vectors = latin_hypercube_vectors(100, schema, seed=11)
    levels = [decode_vector(vector, schema)["palm"]["prototype_index"] for vector in vectors]
    assert vectors.shape == (100, schema["vector_dimension"])
    assert min(levels) == 0
    assert max(levels) <= 31
