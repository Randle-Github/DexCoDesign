from __future__ import annotations

import numpy as np

from dexcodesign.morphology.general_grammar import (
    FINGERS,
    build_schema,
    decode_vector,
    latin_hypercube_vectors,
    phalanx_stages,
)


def test_source_local_dimensions_are_not_padded() -> None:
    five = build_schema(
        "five",
        {finger: ("proximal", "middle", "distal") for finger in FINGERS},
        palm_affine_editable=True,
    )
    four_short = build_schema(
        "four_short",
        {finger: ("distal",) for finger in FINGERS[:4]},
        palm_affine_editable=False,
    )
    assert five["vector_dimension"] == 23
    assert four_short["vector_dimension"] == 7


def test_zero_vector_is_exact_source_and_prototype_zero() -> None:
    schema = build_schema(
        "source",
        {"thumb": phalanx_stages(4, protected_root=True), "index": ("distal",)},
        palm_affine_editable=True,
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
        for scale in finger["length_scales"]
    )


def test_samples_respect_source_local_schema_and_32_levels() -> None:
    schema = build_schema(
        "source",
        {finger: ("proximal", "middle", "distal") for finger in FINGERS[:4]},
        palm_affine_editable=False,
    )
    vectors = latin_hypercube_vectors(100, schema, seed=11)
    levels = [decode_vector(vector, schema)["palm"]["prototype_index"] for vector in vectors]
    assert vectors.shape == (100, schema["vector_dimension"])
    assert min(levels) == 0
    assert max(levels) <= 31
