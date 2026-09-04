from __future__ import annotations

import numpy as np
import pytest

from dexcodesign.morphology.midas_grammar import (
    FINGER_MP_DP_RATIO,
    SOURCE_MM,
    VECTOR_NAMES,
    decode_vector,
    latin_hypercube_vectors,
)
def test_link_width_deformation_is_centred_on_the_mesh_body() -> None:
    trimesh = pytest.importorskip("trimesh")
    from dexcodesign.morphology.mesh_compiler import apply_midas_axis_deformation

    # Deliberately offset the mesh from the motor-frame origin.  This is the
    # important MiDas fingertip case: symmetric widening must not translate
    # the body sideways just because its authored local origin is asymmetric.
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 10.0)).subdivide()
    mesh.apply_translation((4.0, 0.0, 5.0))
    source_vertices = np.asarray(mesh.vertices).copy()
    middle = np.isclose(np.asarray(mesh.vertices)[:, 2], 5.0)
    middle_before = np.asarray(mesh.vertices)[middle, 0]
    before_center = 0.5 * (float(np.min(middle_before)) + float(np.max(middle_before)))
    result = apply_midas_axis_deformation(
        mesh,
        {
            "longitudinal_axis": [0.0, 0.0, 1.0],
            "width_axis": [1.0, 0.0, 0.0],
            "source_length_canonical": 10.0,
            "target_length_canonical": 10.0,
            "width_scale": 1.8,
            "thickness_scale": 1.0,
            "proximal_cap_fraction": 0.12,
            "distal_cap_fraction": 0.12,
        },
    )
    middle_result = np.asarray(result.vertices)[middle, 0]
    after_center = 0.5 * (float(np.min(middle_result)) + float(np.max(middle_result)))
    assert np.isclose(after_center, before_center)
    assert np.ptp(middle_result) > np.ptp(middle_before)
    assert np.array_equal(np.asarray(mesh.vertices), source_vertices)


def test_terminal_fingertip_has_no_fictitious_rigid_distal_cap() -> None:
    trimesh = pytest.importorskip("trimesh")
    from dexcodesign.morphology.mesh_compiler import apply_midas_axis_deformation

    mesh = trimesh.creation.box(extents=(2.0, 1.0, 10.0)).subdivide()
    mesh.apply_translation((4.0, 0.0, 5.0))
    source = np.asarray(mesh.vertices).copy()
    terminal = apply_midas_axis_deformation(
        mesh,
        {
            "longitudinal_axis": [0.0, 0.0, 1.0],
            "width_axis": [1.0, 0.0, 0.0],
            "source_length_canonical": 10.0,
            "target_length_canonical": 10.0,
            "width_scale": 0.6,
            "thickness_scale": 1.0,
            "proximal_cap_fraction": 0.12,
            "distal_cap_fraction": 0.0,
            "distal_connector_fixed": False,
        },
    )
    distal = np.isclose(source[:, 2], 10.0)
    assert np.isclose(
        np.ptp(np.asarray(terminal.vertices)[distal, 0]),
        0.6 * np.ptp(source[distal, 0]),
    )


def test_zero_vector_is_exact_source() -> None:
    decoded = decode_vector(np.zeros(len(VECTOR_NAMES)))
    assert decoded == SOURCE_MM


def test_sampled_vectors_are_feasible() -> None:
    for vector in latin_hypercube_vectors(500, seed=7):
        decoded = decode_vector(vector)
        assert decoded["finger_base_spacing"] > decoded["finger_mp_width"]
        assert decoded["finger_base_spacing"] > decoded["finger_pp_width"]
        assert decoded["finger_dp_width"] < decoded["finger_mp_width"]
        assert decoded["finger_dp_width"] < decoded["finger_pp_width"]
        assert np.isclose(
            decoded["finger_mp_length"] / decoded["finger_dp_length"],
            FINGER_MP_DP_RATIO,
        )


def test_latin_hypercube_reserves_source_row() -> None:
    vectors = latin_hypercube_vectors(50, seed=42)
    assert vectors.shape == (50, len(VECTOR_NAMES))
    assert np.array_equal(vectors[0], np.zeros(len(VECTOR_NAMES)))
    assert np.all(vectors >= -1.0)
    assert np.all(vectors <= 1.0)
