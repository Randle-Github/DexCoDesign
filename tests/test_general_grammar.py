from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexcodesign.morphology.general_grammar import (
    GRAMMAR_ID,
    build_schema,
    decode_vector,
    graph_spec_from_vector,
    palm_prototype_coordinate,
    palm_prototype_index,
)
from dexcodesign.morphology.generate import editable_finger_segments, main_finger_path


ROOT = Path(__file__).resolve().parents[1]


def test_midas_main_paths_exclude_linkage_branches() -> None:
    sources = json.loads(
        (ROOT / "artifacts/hand_morphology/reference_graphs.json").read_text()
    )
    hand = next(record for record in sources["hands"] if record["hand_id"] == "midas_hand")
    bundles = json.loads(
        (ROOT / "artifacts/hand_morphology/mechanism_bundles.json").read_text()
    )["bundles"]
    by_id = {record["bundle_id"]: record for record in bundles}
    assert main_finger_path(hand, by_id["midas_hand:index"]) == [2, 6, 11, 17]
    assert main_finger_path(hand, by_id["midas_hand:middle"]) == [3, 7, 13, 18]
    assert main_finger_path(hand, by_id["midas_hand:ring"]) == [4, 8, 15, 19]
    assert main_finger_path(hand, by_id["midas_hand:thumb"]) == [1, 5, 9, 16]
    assert editable_finger_segments(hand, by_id["midas_hand:index"]) == [6, 11, 17]
    assert editable_finger_segments(hand, by_id["midas_hand:thumb"]) == [5, 9, 16]


def test_midas_general_schema_is_17d_and_zero_is_source() -> None:
    segment_ids = {
        "thumb": (5, 9, 16),
        "index": (6, 11, 17),
        "middle": (7, 13, 18),
        "ring": (8, 15, 19),
    }
    schema = build_schema("midas_hand", segment_ids, palm_affine_editable=False)
    assert schema["vector_dimension"] == 17
    decoded = decode_vector(np.zeros(17), schema)
    assert decoded["palm"]["prototype_index"] == 0
    assert decoded["palm"]["expansion"] == 0.0
    assert all(value == 1.0 for value in decoded["width_scales"].values())
    assert all(
        value == 1.0
        for finger in decoded["fingers"].values()
        for value in finger["length_scales_by_source_part"].values()
    )


def test_general_extremes_restore_the_original_broad_search_range() -> None:
    segment_ids = {
        "thumb": (5, 9, 16),
        "index": (6, 11, 17),
        "middle": (7, 13, 18),
        "ring": (8, 15, 19),
    }
    schema = build_schema("midas_hand", segment_ids, palm_affine_editable=False)
    lower = decode_vector(np.asarray([0.0, *([-1.0] * 16)]), schema)
    upper = decode_vector(np.ones(17), schema)
    assert lower["palm"]["prototype_index"] == 0
    assert upper["palm"]["prototype_index"] == 31
    assert upper["palm"]["expansion"] == 0.70
    assert set(lower["width_scales"].values()) == {0.60}
    assert set(upper["width_scales"].values()) == {1.45}
    assert {
        value
        for finger in lower["fingers"].values()
        for value in finger["length_scales_by_source_part"].values()
    } == {0.55}
    assert {
        value
        for finger in upper["fingers"].values()
        for value in finger["length_scales_by_source_part"].values()
    } == {1.60}


def test_ordered_palm_bank_and_graph_are_canonical() -> None:
    schema = build_schema(
        "wuji_hand_2",
        {
            "thumb": (6, 11, 16),
            "index": (7, 12, 17),
            "middle": (8, 13, 18),
            "ring": (9, 14, 19),
            "pinky": (10, 15, 20),
        },
        palm_affine_editable=True,
    )
    assert schema["vector_dimension"] == 23
    for index in range(32):
        coordinate = palm_prototype_coordinate(index)
        assert palm_prototype_index(coordinate) == index
    source = graph_spec_from_vector(
        np.zeros(schema["vector_dimension"]), schema, hand_id="source"
    )
    maximum = np.zeros(schema["vector_dimension"])
    maximum[0] = 1.0
    star = graph_spec_from_vector(maximum, schema, hand_id="star")
    assert source["grammar_id"] == GRAMMAR_ID
    assert source["palm"]["layout_mode"] == "source_fixed"
    assert star["palm"]["layout_mode"] == "source_star_fusion"
    assert star["palm"]["prototype_index"] == 31
    assert star["palm"]["expansion"] == 0.70
