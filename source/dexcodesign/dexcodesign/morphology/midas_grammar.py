"""MiDas-specific, manufacturing-constrained morphology grammar.

This module intentionally does not extend or reinterpret the generic hand
grammar.  It consumes source-topology MiDas HandIR records and applies a
source-local 15-dimensional grammar expressed in physical millimetres.

The latent vector lies in ``[-1, 1]`` and the all-zero vector is exactly the
source MiDas hand.  Bounds are asymmetric about the source dimensions.  Some
dimensions are decoded conditionally so invalid combinations are impossible
by construction (for example a finger body can never be wider than the
available finger-base spacing).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


GRAMMAR_ID = "midas-manufacturing-constraints-v1"
SOURCE_HAND_ID = "midas_hand"
VECTOR_NAMES = (
    "palm_width",
    "palm_height",
    "finger_base_spacing",
    "thumb_base_distance",
    "finger_dp_length",
    "finger_pp_length",
    "finger_dp_width",
    "finger_mp_width",
    "finger_pp_width",
    "thumb_dp_length",
    "thumb_mp_length",
    "thumb_pp_length",
    "thumb_dp_width",
    "thumb_mp_width",
    "thumb_pp_width",
)

# Dimensions supplied by the MiDas mechanical-design constraint set, in mm.
SOURCE_MM = {
    "palm_width": 88.6,
    "palm_height": 70.0,
    "finger_base_spacing": 30.6,
    "thumb_base_distance": 51.7,
    "finger_dp_length": 33.2,
    "finger_mp_length": 33.0,
    "finger_pp_length": 35.5,
    "finger_dp_width": 17.0,
    "finger_mp_width": 28.5,
    "finger_pp_width": 28.5,
    "thumb_dp_length": 42.5,
    "thumb_mp_length": 35.5,
    "thumb_pp_length": 43.9,
    "thumb_dp_width": 28.5,
    "thumb_mp_width": 28.5,
    "thumb_pp_width": 28.5,
}

# Finite research bounds are required by an optimizer where the ME document
# states X/TBD.  The documented lower/upper limits remain exact; only open
# directions receive conservative finite caps.  Polymer is the v1 material
# profile, hence the 27.5 mm rather than 24.4 mm MP/PP lower bound.
STATIC_BOUNDS_MM = {
    "palm_width": (70.0, 125.0),
    "palm_height": (56.0, 95.0),
    "finger_base_spacing": (28.0, 42.0),
    "thumb_base_distance": (44.7, 70.0),
    "finger_dp_length": (25.2, 45.0),
    "finger_pp_length": (35.5, 50.0),
    "finger_dp_width": (10.0, 25.0),
    "finger_mp_width": (27.5, 36.0),
    "finger_pp_width": (27.5, 36.0),
    "thumb_dp_length": (25.0, 55.0),
    "thumb_mp_length": (35.5, 50.0),
    "thumb_pp_length": (25.5, 51.0),
    "thumb_dp_width": (27.5, 38.0),
    "thumb_mp_width": (27.5, 38.0),
    "thumb_pp_width": (27.5, 38.0),
}

FINGER_MP_DP_RATIO = SOURCE_MM["finger_mp_length"] / SOURCE_MM["finger_dp_length"]
FINGER_CLEARANCE_MM = 0.5
PALM_EDGE_OVERHANG_MM = 2.0

# source_graph.py maps source metres into a normalized canonical frame.  The
# stored audit scale is 9.09609740980112 for MiDas, so one physical mm is this
# many canonical units.
CANONICAL_UNITS_PER_MM = 9.09609740980112 / 1000.0

NORMAL_SEGMENTS = {
    6: ("finger_pp_length", "finger_pp_width"),
    7: ("finger_pp_length", "finger_pp_width"),
    8: ("finger_pp_length", "finger_pp_width"),
    11: ("finger_mp_length", "finger_mp_width"),
    13: ("finger_mp_length", "finger_mp_width"),
    15: ("finger_mp_length", "finger_mp_width"),
    17: ("finger_dp_length", "finger_dp_width"),
    18: ("finger_dp_length", "finger_dp_width"),
    19: ("finger_dp_length", "finger_dp_width"),
}
THUMB_SEGMENTS = {
    5: ("thumb_pp_length", "thumb_pp_width"),
    9: ("thumb_mp_length", "thumb_mp_width"),
    16: ("thumb_dp_length", "thumb_dp_width"),
}
NORMAL_ROOT_PARTS = {"index": 2, "middle": 3, "ring": 4}


def _asymmetric_decode(value: float, source: float, lower: float, upper: float) -> float:
    """Map one latent value onto asymmetric physical bounds."""
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"MiDas latent value {value} is outside [-1, 1]")
    if value < 0.0:
        return source + (-value) * (lower - source)
    return source + value * (upper - source)


def decode_vector(vector: np.ndarray | list[float]) -> dict[str, float]:
    """Decode a 15D latent vector into a feasible physical design in mm."""
    latent = np.asarray(vector, dtype=np.float64)
    if latent.shape != (len(VECTOR_NAMES),):
        raise ValueError(f"MiDas vector must have shape ({len(VECTOR_NAMES)},), got {latent.shape}")
    if not np.all(np.isfinite(latent)) or np.any(np.abs(latent) > 1.0 + 1.0e-12):
        raise ValueError("MiDas vector must contain finite values in [-1, 1]")
    latent = np.clip(latent, -1.0, 1.0)
    raw = dict(zip(VECTOR_NAMES, latent, strict=True))
    result: dict[str, float] = {}

    for name in (
        "finger_dp_length",
        "finger_pp_length",
        "thumb_dp_length",
        "thumb_mp_length",
        "thumb_pp_length",
        "thumb_dp_width",
        "thumb_mp_width",
        "thumb_pp_width",
    ):
        lo, hi = STATIC_BOUNDS_MM[name]
        result[name] = _asymmetric_decode(raw[name], SOURCE_MM[name], lo, hi)

    # PIP/DIP linkage geometry requires this ratio to remain fixed.  It is a
    # resolved graph dimension, not an optimizer dimension.
    result["finger_mp_length"] = result["finger_dp_length"] * FINGER_MP_DP_RATIO

    # Palm height and the unconstrained outer-width request are decoded first.
    # Finger roots live on the palm deformation cage: changing palm dimensions
    # must move their reference attachment frames before the explicit f-DB and
    # t-RB variables add their own spread.  This is the continuous MiDas
    # analogue of the generic grammar's star/radial attachment expansion.
    lo, hi = STATIC_BOUNDS_MM["palm_height"]
    result["palm_height"] = _asymmetric_decode(
        raw["palm_height"], SOURCE_MM["palm_height"], lo, hi
    )
    lo, hi = STATIC_BOUNDS_MM["palm_width"]
    requested_palm_width = _asymmetric_decode(
        raw["palm_width"], SOURCE_MM["palm_width"], lo, hi
    )
    palm_width = requested_palm_width

    # Palm width, attachment spread, motor-body width and the minimum shell
    # width are mutually coupled. A short deterministic fixed-point solve keeps
    # every sampled vector feasible without rejection or post-hoc repair.
    for _ in range(12):
        palm_width_scale = palm_width / SOURCE_MM["palm_width"]
        palm_height_scale = result["palm_height"] / SOURCE_MM["palm_height"]

        lo, hi = STATIC_BOUNDS_MM["finger_base_spacing"]
        palm_warped_spacing = float(np.clip(
            SOURCE_MM["finger_base_spacing"] * palm_width_scale, lo, hi
        ))
        result["finger_base_spacing"] = _asymmetric_decode(
            raw["finger_base_spacing"], palm_warped_spacing, lo, hi
        )

        lo, hi = STATIC_BOUNDS_MM["thumb_base_distance"]
        radial_scale = float(np.sqrt(palm_width_scale * palm_height_scale))
        palm_warped_thumb_distance = float(np.clip(
            SOURCE_MM["thumb_base_distance"] * radial_scale, lo, hi
        ))
        result["thumb_base_distance"] = _asymmetric_decode(
            raw["thumb_base_distance"], palm_warped_thumb_distance, lo, hi
        )

        body_width_upper = result["finger_base_spacing"] - FINGER_CLEARANCE_MM
        if body_width_upper < STATIC_BOUNDS_MM["finger_mp_width"][0] - 1.0e-10:
            raise ValueError("finger-base spacing cannot fit the minimum polymer finger width")
        for name in ("finger_mp_width", "finger_pp_width"):
            width_lo, static_hi = STATIC_BOUNDS_MM[name]
            width_hi = min(static_hi, body_width_upper)
            conditional_source = min(SOURCE_MM[name], width_hi)
            result[name] = _asymmetric_decode(
                raw[name], conditional_source, width_lo, width_hi
            )

        dp_width_upper = min(
            STATIC_BOUNDS_MM["finger_dp_width"][1],
            result["finger_mp_width"] - FINGER_CLEARANCE_MM,
            result["finger_pp_width"] - FINGER_CLEARANCE_MM,
        )
        width_lo, _ = STATIC_BOUNDS_MM["finger_dp_width"]
        result["finger_dp_width"] = _asymmetric_decode(
            raw["finger_dp_width"],
            min(SOURCE_MM["finger_dp_width"], dp_width_upper),
            width_lo,
            dp_width_upper,
        )

        minimum_palm_width = max(
            STATIC_BOUNDS_MM["palm_width"][0],
            2.0 * result["finger_base_spacing"]
            + max(result["finger_mp_width"], result["finger_pp_width"])
            - PALM_EDGE_OVERHANG_MM,
            result["thumb_base_distance"]
            + 0.5 * max(result["thumb_mp_width"], result["thumb_pp_width"])
            + 8.0,
        )
        updated = max(requested_palm_width, minimum_palm_width)
        if updated > STATIC_BOUNDS_MM["palm_width"][1] + 1.0e-9:
            raise ValueError("resolved attachment layout exceeds maximum palm width")
        if abs(updated - palm_width) <= 1.0e-10:
            palm_width = updated
            break
        palm_width = updated
    result["palm_width"] = palm_width

    validate_dimensions(result)
    return result


def validate_dimensions(dimensions: dict[str, float]) -> None:
    """Fail fast if a resolved design violates the MiDas v1 contract."""
    required = set(SOURCE_MM)
    if set(dimensions) != required:
        raise ValueError(f"resolved MiDas dimensions differ from schema: {set(dimensions) ^ required}")
    if dimensions["finger_base_spacing"] < 24.4:
        raise ValueError("f-DB is below the documented 24.4 mm limit")
    if dimensions["thumb_base_distance"] < 44.7:
        raise ValueError("t-RB is below the documented 44.7 mm limit")
    if dimensions["finger_base_spacing"] <= max(
        dimensions["finger_mp_width"], dimensions["finger_pp_width"]
    ):
        raise ValueError("f-DB must exceed both MP and PP widths")
    if dimensions["finger_dp_width"] >= min(
        dimensions["finger_mp_width"], dimensions["finger_pp_width"]
    ):
        raise ValueError("finger DP width must remain below MP and PP widths")
    if not np.isclose(
        dimensions["finger_mp_length"] / dimensions["finger_dp_length"],
        FINGER_MP_DP_RATIO,
        atol=1.0e-12,
    ):
        raise ValueError("finger MP/DP coupling ratio changed")
    for name, (lower, upper) in STATIC_BOUNDS_MM.items():
        # Palm width has a dynamic lower bound and may therefore exceed its
        # static source-centred decode point, but never the hard outer range.
        value = dimensions[name]
        if value < lower - 1.0e-9 or value > upper + 1.0e-9:
            raise ValueError(f"{name}={value} is outside [{lower}, {upper}] mm")
    if dimensions["finger_mp_length"] < 25.0:
        raise ValueError("derived finger MP length is below 25 mm")


def grammar_payload() -> dict:
    """Return a serializable description of the dedicated grammar."""
    return {
        "schema_version": 1,
        "grammar_id": GRAMMAR_ID,
        "source_hand": SOURCE_HAND_ID,
        "latent_domain": [-1.0, 1.0],
        "zero_vector_is_source": True,
        "vector_names": list(VECTOR_NAMES),
        "vector_dimension": len(VECTOR_NAMES),
        "source_dimensions_mm": SOURCE_MM,
        "static_bounds_mm": {name: list(bounds) for name, bounds in STATIC_BOUNDS_MM.items()},
        "derived_dimensions": {
            "finger_mp_length": {
                "expression": "finger_dp_length * source_finger_mp_dp_ratio",
                "ratio": FINGER_MP_DP_RATIO,
            }
        },
        "hard_constraints": {
            "material_profile": "polymer",
            "minimum_polymer_mp_pp_width_mm": 27.5,
            "fixed_thickness": True,
            "identical_digits": ["index", "middle", "ring"],
            "base_and_transmission_fixed": True,
            "motor_housings_and_connector_caps_fixed": True,
            "palm_parameterization": "continuous width/height; no template bank",
            "finger_base_spacing_is_explicit": True,
            "thumb_base_distance_is_explicit": True,
            "joint_types_axes_limits_and_dof_fixed": True,
        },
    }


def latin_hypercube_vectors(count: int, seed: int) -> np.ndarray:
    """Sample deterministic stratified vectors, reserving row zero for source."""
    if count < 1:
        raise ValueError("count must be positive")
    vectors = np.zeros((count, len(VECTOR_NAMES)), dtype=np.float64)
    if count == 1:
        return vectors
    rng = np.random.default_rng(seed)
    samples = count - 1
    for column in range(len(VECTOR_NAMES)):
        strata = (np.arange(samples, dtype=np.float64) + rng.random(samples)) / samples
        rng.shuffle(strata)
        vectors[1:, column] = 2.0 * strata - 1.0
    return vectors


def _normalized(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        value = fallback
        norm = float(np.linalg.norm(value))
    return value / norm


def _scale_parallel(vector: list[float], axis: np.ndarray, factor: float) -> list[float]:
    value = np.asarray(vector, dtype=np.float64)
    parallel = float(np.dot(value, axis)) * axis
    return (value + (factor - 1.0) * parallel).tolist()


def _smoothstep_scalar(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _map_palm_attachment(
    point: np.ndarray,
    bounds: np.ndarray,
    width_scale: float,
    height_scale: float,
    wrist_cap_fraction: float = 0.20,
    finger_cap_fraction: float = 0.12,
) -> np.ndarray:
    """Apply the exact palm-mesh cage map to one attachment frame origin."""
    mapped = np.asarray(point, dtype=np.float64).copy()
    z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])
    height = z_max - z_min
    wrist_end = z_min + wrist_cap_fraction * height
    finger_start = z_max - finger_cap_fraction * height
    target_middle = (
        height_scale * height
        - (wrist_end - z_min)
        - (z_max - finger_start)
    )
    z = float(point[2])
    if z <= wrist_end:
        mapped_z = z
    elif z < finger_start:
        mapped_z = wrist_end + (z - wrist_end) * (
            target_middle / (finger_start - wrist_end)
        )
    else:
        mapped_z = z + height * (height_scale - 1.0)
    width_weight = _smoothstep_scalar(
        (z - wrist_end) / max(0.35 * height, 1.0e-12)
    )
    width_multiplier = 1.0 + width_weight * (width_scale - 1.0)
    x_center = 0.5 * float(bounds[0, 0] + bounds[1, 0])
    mapped[0] = x_center + width_multiplier * (float(point[0]) - x_center)
    mapped[2] = mapped_z
    return mapped


def _recompute_world(parts: list[dict]) -> None:
    world: list[np.ndarray] = []
    children: dict[int, list[int]] = {int(node["id"]): [] for node in parts}
    for node in parts:
        parent = node["parent"]
        if parent is None:
            position = np.zeros(3, dtype=np.float64)
        else:
            position = world[int(parent)] + np.asarray(node["relative_pos"], dtype=np.float64)
            children[int(parent)].append(int(node["id"]))
        world.append(position)
    for node, position in zip(parts, world, strict=True):
        node["world_pos"] = position.tolist()
        node["child_count"] = len(children[int(node["id"])])


def apply_to_hand(hand: dict, vector: np.ndarray | list[float]) -> dict:
    """Apply one decoded design to a source-fixed generic MiDas HandIR hand."""
    result = deepcopy(hand)
    latent = np.asarray(vector, dtype=np.float64)
    dimensions = decode_vector(latent)
    if result.get("seed_source") != SOURCE_HAND_ID:
        raise ValueError("MiDas grammar can only consume a midas_hand source graph")
    parts = result["parts"]
    by_source = {int(node["source_part_id"]): node for node in parts}
    source_root_positions = {
        role: np.asarray(by_source[source_id]["relative_pos"], dtype=np.float64)
        for role, source_id in NORMAL_ROOT_PARTS.items()
    }
    source_thumb = np.asarray(by_source[1]["relative_pos"], dtype=np.float64)

    # First move every motor attachment through the exact same continuous cage
    # used by the palm mesh. Explicit MiDas placement variables are applied on
    # top of these star/affine-warped reference frames, never independently of
    # the palm surface.
    palm = by_source[0]
    palm_bounds = np.asarray(palm["source_mesh"]["bounds"], dtype=np.float64)
    palm_width_scale = dimensions["palm_width"] / SOURCE_MM["palm_width"]
    palm_height_scale = dimensions["palm_height"] / SOURCE_MM["palm_height"]
    warped_roots = {
        role: _map_palm_attachment(
            position, palm_bounds, palm_width_scale, palm_height_scale
        )
        for role, position in source_root_positions.items()
    }
    warped_thumb = _map_palm_attachment(
        source_thumb, palm_bounds, palm_width_scale, palm_height_scale
    )

    # Explicit, symmetric f-DB preserves the source middle-finger stagger.
    center_x = float(warped_roots["middle"][0])
    spacing = dimensions["finger_base_spacing"] * CANONICAL_UNITS_PER_MM
    target_roots: dict[str, np.ndarray] = {}
    for role, sign in (("index", 1.0), ("middle", 0.0), ("ring", -1.0)):
        target = warped_roots[role].copy()
        target[0] = center_x + sign * spacing
        target_roots[role] = target
        by_source[NORMAL_ROOT_PARTS[role]]["relative_pos"] = target.tolist()

    # t-RB follows the already palm-warped thumb-to-index ray. The decoded
    # dimension contains the palm radial expansion plus its explicit residual.
    warped_delta = warped_thumb - warped_roots["index"]
    radial_scale = float(np.sqrt(palm_width_scale * palm_height_scale))
    warped_semantic_distance = float(np.clip(
        SOURCE_MM["thumb_base_distance"] * radial_scale,
        *STATIC_BOUNDS_MM["thumb_base_distance"],
    ))
    thumb_residual_scale = dimensions["thumb_base_distance"] / warped_semantic_distance
    thumb_target = target_roots["index"] + thumb_residual_scale * warped_delta
    by_source[1]["relative_pos"] = thumb_target.tolist()

    # Joint-centre propagation. Only the longitudinal component changes;
    # transverse motor offsets and every joint axis/range remain source-exact.
    normal_axis = np.asarray([0.0, 0.0, 1.0])
    pp_scale = dimensions["finger_pp_length"] / SOURCE_MM["finger_pp_length"]
    mp_scale = dimensions["finger_mp_length"] / SOURCE_MM["finger_mp_length"]
    for source_id in (11, 13, 15):
        by_source[source_id]["relative_pos"] = _scale_parallel(
            by_source[source_id]["relative_pos"], normal_axis, pp_scale
        )
    for source_id in (17, 18, 19):
        by_source[source_id]["relative_pos"] = _scale_parallel(
            by_source[source_id]["relative_pos"], normal_axis, mp_scale
        )
    # The linkage branch is recalibrated with PP length while preserving all
    # transverse pivots and the fixed source MP/DP coupling ratio.
    for source_id in (10, 12, 14):
        by_source[source_id]["relative_pos"] = _scale_parallel(
            by_source[source_id]["relative_pos"], normal_axis, pp_scale
        )

    # Thumb segment directions are source-derived rather than assumed to align
    # with a global axis.
    thumb_axes: dict[int, np.ndarray] = {}
    for source_id, child_source_id in ((5, 9), (9, 16)):
        thumb_axes[source_id] = _normalized(
            np.asarray(by_source[child_source_id]["relative_pos"], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0]),
        )
    thumb_axes[16] = thumb_axes[9].copy()
    for source_id, length_name in ((9, "thumb_pp_length"), (16, "thumb_mp_length")):
        parent_source_id = 5 if source_id == 9 else 9
        axis = thumb_axes[parent_source_id]
        scale = dimensions[length_name] / SOURCE_MM[length_name]
        by_source[source_id]["relative_pos"] = _scale_parallel(
            by_source[source_id]["relative_pos"], axis, scale
        )

    # Mesh deformation is compiled later. Connector caps stay rigid, the
    # middle span changes length/width, and the thickness direction is locked.
    for source_id, (length_name, width_name) in {**NORMAL_SEGMENTS, **THUMB_SEGMENTS}.items():
        node = by_source[source_id]
        terminal_fingertip = source_id in {16, 17, 18, 19}
        if source_id in NORMAL_SEGMENTS:
            axis = normal_axis
            width_axis = np.asarray([1.0, 0.0, 0.0])
        else:
            axis = thumb_axes[source_id]
            joint_axis = np.asarray(node["joint_axis"], dtype=np.float64)
            width_axis = joint_axis - float(np.dot(joint_axis, axis)) * axis
            width_axis = _normalized(width_axis, np.asarray([1.0, 0.0, 0.0]))
        source_length = SOURCE_MM[length_name]
        source_width = SOURCE_MM[width_name]
        node["mesh_linear"] = np.eye(3).tolist()
        node["midas_axis_deformation"] = {
            "longitudinal_axis": axis.tolist(),
            "width_axis": width_axis.tolist(),
            "source_length_canonical": source_length * CANONICAL_UNITS_PER_MM,
            "target_length_canonical": dimensions[length_name] * CANONICAL_UNITS_PER_MM,
            "width_scale": dimensions[width_name] / source_width,
            "thickness_scale": 1.0,
            "proximal_cap_fraction": 0.12,
            # A terminal DP has no distal motor/connector to preserve. Keeping
            # a fictitious rigid distal cap makes a narrowed fingertip flare
            # back to source width and creates a crescent/hook silhouette.
            "distal_cap_fraction": 0.0 if terminal_fingertip else 0.12,
            "distal_connector_fixed": not terminal_fingertip,
            "deform_middle_span_only": True,
        }
        node["midas_segment"] = length_name.removesuffix("_length")

    palm["mesh_linear"] = np.eye(3).tolist()
    palm["midas_palm_deformation"] = {
        "source_width_mm": SOURCE_MM["palm_width"],
        "target_width_mm": dimensions["palm_width"],
        "source_height_mm": SOURCE_MM["palm_height"],
        "target_height_mm": dimensions["palm_height"],
        "thickness_scale": 1.0,
        "wrist_cap_fraction": 0.20,
        "finger_cap_fraction": 0.12,
        "continuous_no_prototype": True,
    }

    _recompute_world(parts)
    for slot in result["finger_slots"]:
        root = parts[int(slot["root_node_id"])]
        slot["attachment_translation"] = root["world_pos"]
        slot["reference_attachment_translation"] = by_source[int(root["source_part_id"])].get(
            "source_world_pos", slot.get("reference_attachment_translation", root["world_pos"])
        )
    result.update(
        {
            "grammar_id": GRAMMAR_ID,
            "grammar_version": 1,
            "midas_latent_vector": latent.tolist(),
            "midas_vector_names": list(VECTOR_NAMES),
            "midas_dimensions_mm": dimensions,
            "palm_parameterization": "continuous_width_height",
            "palm_prototype_used": False,
            "thickness_scale_locked": 1.0,
            "identical_normal_finger_parameters": True,
            "fixed_finger_mp_dp_ratio": FINGER_MP_DP_RATIO,
            "edit_mode": "midas_manufacturing_constraints",
            "grammar_actions": [
                {
                    "operation": "APPLY_MIDAS_MANUFACTURING_CONSTRAINTS",
                    "latent_domain": [-1.0, 1.0],
                    "zero_vector_is_source": True,
                    "physical_units": "millimetres",
                },
                {
                    "operation": "DEFORM_LINK_MIDDLE_SPANS",
                    "connector_caps_fixed": True,
                    "thickness_fixed": True,
                    "distal_graph_propagated": True,
                },
                {
                    "operation": "DEFORM_CONTINUOUS_PALM_WIDTH_HEIGHT",
                    "prototype_bank": False,
                    "transmission_base_fixed": True,
                },
            ],
        }
    )
    return result
