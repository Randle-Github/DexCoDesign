#!/usr/bin/env python3
"""Attachment-conditioned, graph-frame-preserving palm geometry.

The kinematic graph owns every attachment transform.  This module consumes
those transforms as immutable inputs and generates geometry around them.  It
never estimates a joint frame from a mesh and never writes a modified frame
back to the graph.

The canonical hands used by ``grammar_v1`` are upright: X is palm width, Z is
palm length, and Y is palm thickness.  The plane/thickness axes are explicit
parameters so the geometry code does not silently depend on that convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union


PalmGenerationMode = Literal[
    "fixed_template", "attachment_hull", "parametric_2_5d", "template_deform"
]


def _as_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"attachment transform must be 4x4, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError("attachment transform has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("attachment rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError("attachment rotation must be proper")
    return transform.copy()


@dataclass(frozen=True)
class AttachmentPatch:
    name: str
    transform: np.ndarray
    width: float
    depth: float
    thickness: float
    locked: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", _as_transform(self.transform))
        for name in ("width", "depth", "thickness"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{self.name}.{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PalmGeometryParams:
    thickness: float
    boundary_margin: float
    edge_rounding_radius: float
    wrist_width: float
    transverse_arch: float = 0.0
    longitudinal_arch: float = 0.0
    central_cup: float = 0.0
    plane_axes: tuple[int, int] = (0, 2)
    thickness_axis: int = 1
    lock_falloff: float | None = None
    deformation_resolution: float | None = None

    def __post_init__(self) -> None:
        for name in ("thickness", "wrist_width"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("boundary_margin", "edge_rounding_radius"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        axes = (*self.plane_axes, self.thickness_axis)
        if sorted(axes) != [0, 1, 2]:
            raise ValueError(f"plane_axes and thickness_axis must partition XYZ, got {axes}")


@dataclass
class PalmMeshResult:
    visual_mesh: trimesh.Trimesh
    collision_mesh: trimesh.Trimesh
    attachment_frames: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)


def transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float)
    result[:3, 3] = np.asarray(translation, dtype=float)
    return _as_transform(result)


def _normalize_2d(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > 1.0e-10:
        return value / norm
    return np.asarray([0.0, 1.0] if fallback is None else fallback, dtype=float)


def patch_corners_2d(patch: AttachmentPatch, params: PalmGeometryParams) -> np.ndarray:
    """Project one rigid patch to the palm plane without changing its frame."""
    plane = np.asarray(params.plane_axes, dtype=int)
    center = patch.transform[plane, 3]
    # The graph frame Z direction is the outgoing finger-chain direction.
    # Its projection defines patch depth; the perpendicular direction defines
    # patch width.  Degenerate projections use graph-frame X, then +V.
    depth_axis = patch.transform[plane, 2]
    if np.linalg.norm(depth_axis) < 1.0e-8:
        depth_axis = patch.transform[plane, 0]
    depth_axis = _normalize_2d(depth_axis)
    width_axis = np.asarray([-depth_axis[1], depth_axis[0]], dtype=float)
    return np.asarray([
        center - 0.5 * patch.width * width_axis - 0.5 * patch.depth * depth_axis,
        center + 0.5 * patch.width * width_axis - 0.5 * patch.depth * depth_axis,
        center + 0.5 * patch.width * width_axis + 0.5 * patch.depth * depth_axis,
        center - 0.5 * patch.width * width_axis + 0.5 * patch.depth * depth_axis,
    ])


def _patch_polygon(patch: AttachmentPatch, params: PalmGeometryParams) -> Polygon:
    polygon = Polygon(patch_corners_2d(patch, params))
    if not polygon.is_valid or polygon.area <= 1.0e-12:
        raise ValueError(f"attachment patch {patch.name!r} has a degenerate projection")
    return polygon


def build_attachment_outline(
    patches: Iterable[AttachmentPatch],
    wrist_patch: AttachmentPatch,
    params: PalmGeometryParams,
    *,
    rounded: bool,
) -> tuple[Polygon, dict[str, Polygon]]:
    """Convex-hull baseline with an isolated outline-generation interface.

    This function is intentionally the only place that chooses the outline
    algorithm.  A future alpha-shape implementation can replace the convex
    hull without touching extrusion, deformation, graph frames, or exporters.
    """
    all_patches = [*patches, wrist_patch]
    polygons = {patch.name: _patch_polygon(patch, params) for patch in all_patches}
    points = [tuple(point) for polygon in polygons.values() for point in polygon.exterior.coords[:-1]]
    outline = MultiPoint(points).convex_hull
    if not isinstance(outline, Polygon) or outline.area <= 1.0e-12:
        raise ValueError("attachment layout cannot produce a 2D palm outline")
    if params.boundary_margin > 0.0:
        outline = outline.buffer(params.boundary_margin, join_style="mitre")
    if rounded and params.edge_rounding_radius > 0.0:
        radius = min(params.edge_rounding_radius, 0.2 * np.sqrt(outline.area))
        rounded_outline = outline.buffer(-radius, join_style="round")
        if not rounded_outline.is_empty:
            rounded_outline = rounded_outline.buffer(radius, join_style="round")
            # Locked patch footprints win over cosmetic corner rounding.
            outline = rounded_outline.union(unary_union(list(polygons.values())))
            if outline.geom_type != "Polygon":
                outline = outline.convex_hull
    if not outline.is_valid:
        outline = outline.buffer(0.0)
    for name, polygon in polygons.items():
        if not outline.buffer(1.0e-9).covers(polygon):
            raise ValueError(f"outline does not cover locked attachment patch {name!r}")
    return outline, polygons


def _normal_span(
    patches: list[AttachmentPatch], wrist_patch: AttachmentPatch, params: PalmGeometryParams
) -> tuple[float, float]:
    all_patches = [*patches, wrist_patch]
    axis = params.thickness_axis
    lower = min(patch.transform[axis, 3] - 0.5 * patch.thickness for patch in all_patches)
    upper = max(patch.transform[axis, 3] + 0.5 * patch.thickness for patch in all_patches)
    requested = float(params.thickness)
    center = 0.5 * (lower + upper)
    effective = max(requested, upper - lower)
    return center, effective


def _extrude_outline(
    outline: Polygon,
    center_normal: float,
    thickness: float,
    params: PalmGeometryParams,
) -> trimesh.Trimesh:
    # trimesh extrudes a 2D XY polygon along local Z.  Reorder the result into
    # the explicit graph/palm coordinate convention (X,Z plane; Y thickness in
    # the current dataset).  No joint or attachment transform is involved.
    raw = trimesh.creation.extrude_polygon(outline, height=thickness)
    source = np.asarray(raw.vertices, dtype=float)
    target = np.zeros_like(source)
    target[:, params.plane_axes[0]] = source[:, 0]
    target[:, params.plane_axes[1]] = source[:, 1]
    target[:, params.thickness_axis] = source[:, 2] - 0.5 * thickness + center_normal
    mesh = trimesh.Trimesh(vertices=target, faces=raw.faces.copy(), process=False)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _apply_masked_deformation(
    mesh: trimesh.Trimesh,
    outline: Polygon,
    locked_polygons: list[Polygon],
    params: PalmGeometryParams,
) -> tuple[trimesh.Trimesh, dict]:
    amplitudes = np.asarray([
        params.transverse_arch, params.longitudinal_arch, params.central_cup
    ], dtype=float)
    if np.allclose(amplitudes, 0.0):
        return mesh, {"enabled": False, "maximum_displacement": 0.0}

    bounds = np.asarray(outline.bounds, dtype=float)
    planar_extent = np.maximum(bounds[2:] - bounds[:2], 1.0e-6)
    max_edge = params.deformation_resolution
    if max_edge is None:
        max_edge = max(float(planar_extent.min()) / 9.0, float(params.thickness) / 2.5, 1.0e-3)
    # ``subdivide_to_size`` may split only one side of a shared edge and can
    # therefore produce T-junctions on an otherwise watertight extrusion.
    # Uniform subdivision preserves the closed topology exactly.
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    longest = float(np.max(mesh.edges_unique_length))
    iterations = int(np.clip(np.ceil(np.log2(max(longest / float(max_edge), 1.0))), 0, 3))
    for _ in range(iterations):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    plane = np.asarray(params.plane_axes, dtype=int)
    uv = np.asarray(result.vertices[:, plane], dtype=float)
    center = 0.5 * (bounds[:2] + bounds[2:])
    normalized = 2.0 * (uv - center) / planar_extent
    transverse = float(params.transverse_arch) * np.exp(-2.2 * normalized[:, 0] ** 2)
    longitudinal = float(params.longitudinal_arch) * np.exp(-2.2 * normalized[:, 1] ** 2)
    cup = -float(params.central_cup) * np.exp(-2.8 * np.sum(normalized ** 2, axis=1))
    requested_delta = transverse + longitudinal + cup

    locked_union = unary_union(locked_polygons)
    falloff = params.lock_falloff
    if falloff is None:
        falloff = max(0.12 * float(planar_extent.min()), 1.0e-4)
    distance = np.asarray([locked_union.distance(Point(point)) for point in uv], dtype=float)
    mask = _smoothstep(distance / float(falloff))
    delta = mask * requested_delta
    result.vertices[:, params.thickness_axis] += delta
    result.remove_unreferenced_vertices()
    result.fix_normals(multibody=True)
    return result, {
        "enabled": True,
        "maximum_displacement": float(np.max(np.abs(delta))),
        "locked_vertex_count": int(np.count_nonzero(mask == 0.0)),
        "falloff": float(falloff),
        "resolution": float(max_edge),
        "subdivision_iterations": iterations,
    }


def _mesh_quality(mesh: trimesh.Trimesh) -> dict:
    triangles = np.asarray(mesh.triangles, dtype=float)
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "degenerate_faces": int(np.count_nonzero(doubled_area <= 1.0e-12)),
        "body_count": int(mesh.body_count),
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
    }


def generate_palm_mesh(
    attachment_patches: list[AttachmentPatch],
    wrist_patch: AttachmentPatch,
    params: PalmGeometryParams,
    *,
    mode: PalmGenerationMode = "parametric_2_5d",
    fixed_template_mesh: trimesh.Trimesh | None = None,
) -> PalmMeshResult:
    """Generate palm geometry while returning bitwise-identical graph frames."""
    if mode not in {"fixed_template", "attachment_hull", "parametric_2_5d"}:
        raise ValueError(f"unknown palm generation mode: {mode}")
    names = [patch.name for patch in attachment_patches] + [wrist_patch.name]
    if len(set(names)) != len(names):
        raise ValueError("attachment patch names must be unique")
    frames = {
        patch.name: patch.transform.copy()
        for patch in [*attachment_patches, wrist_patch]
    }

    if mode == "fixed_template":
        if fixed_template_mesh is None:
            raise ValueError("fixed_template mode requires fixed_template_mesh")
        visual = fixed_template_mesh.copy()
        visual.remove_unreferenced_vertices()
        visual.fix_normals(multibody=True)
        collision = visual.convex_hull
        return PalmMeshResult(
            visual_mesh=visual,
            collision_mesh=collision,
            attachment_frames=frames,
            metadata={
                "mode": mode,
                "frame_source": "graph_passthrough",
                "visual_quality": _mesh_quality(visual),
                "collision_quality": _mesh_quality(collision),
            },
        )

    outline, polygons = build_attachment_outline(
        attachment_patches, wrist_patch, params, rounded=mode == "parametric_2_5d"
    )
    center_normal, effective_thickness = _normal_span(attachment_patches, wrist_patch, params)
    visual = _extrude_outline(outline, center_normal, effective_thickness, params)
    deformation = {"enabled": False, "maximum_displacement": 0.0}
    if mode == "parametric_2_5d":
        locked = [
            polygons[patch.name]
            for patch in [*attachment_patches, wrist_patch]
            if patch.locked
        ]
        visual, deformation = _apply_masked_deformation(visual, outline, locked, params)
    if not visual.is_watertight or not visual.is_winding_consistent:
        raise ValueError("generated palm visual mesh is not a valid watertight solid")
    if _mesh_quality(visual)["degenerate_faces"]:
        raise ValueError("generated palm visual mesh contains degenerate faces")
    # Collision v1 reuses the exact 2D footprint and the deformed visual's
    # outer normal span.  It is a low-face watertight prism, not an oversized
    # axis-aligned bounding box.  VHACD can later replace this independently.
    normal_bounds = visual.bounds[:, params.thickness_axis]
    collision = _extrude_outline(
        outline,
        center_normal=float(normal_bounds.mean()),
        thickness=float(normal_bounds[1] - normal_bounds[0]),
        params=params,
    )
    quality = _mesh_quality(collision)
    if not quality["watertight"] or not quality["winding_consistent"]:
        raise ValueError("generated palm collision mesh is invalid")

    coverage = {
        name: bool(outline.buffer(1.0e-9).covers(polygon))
        for name, polygon in polygons.items()
    }
    metadata = {
        "mode": mode,
        "frame_source": "graph_passthrough",
        "plane_axes": list(params.plane_axes),
        "thickness_axis": int(params.thickness_axis),
        "requested_thickness": float(params.thickness),
        "effective_thickness": float(effective_thickness),
        "normal_center": float(center_normal),
        "outline_area": float(outline.area),
        "outline_coordinates": np.asarray(outline.exterior.coords, dtype=float).tolist(),
        "attachment_coverage": coverage,
        "deformation": deformation,
        "visual_quality": _mesh_quality(visual),
        "collision_quality": quality,
    }
    return PalmMeshResult(visual, collision, frames, metadata)


def _ray_boundary_point(outline: Polygon, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = _normalize_2d(direction)
    radius = max(float(np.linalg.norm(np.asarray(outline.bounds)[2:] - np.asarray(outline.bounds)[:2])), 1.0)
    ray = LineString([origin, origin + 4.0 * radius * direction])
    intersection = outline.boundary.intersection(ray)
    points = []
    if intersection.geom_type == "Point":
        points = [np.asarray(intersection.coords[0], dtype=float)]
    elif intersection.geom_type == "MultiPoint":
        points = [np.asarray(point.coords[0], dtype=float) for point in intersection.geoms]
    elif intersection.geom_type in {"LineString", "MultiLineString"}:
        geometries = [intersection] if intersection.geom_type == "LineString" else intersection.geoms
        for geometry in geometries:
            if geometry.is_empty or len(geometry.coords) == 0:
                continue
            points.extend(np.asarray([geometry.coords[0], geometry.coords[-1]], dtype=float))
    forward = [point for point in points if float(np.dot(point - origin, direction)) >= -1.0e-8]
    if forward:
        return min(forward, key=lambda point: float(np.linalg.norm(point - origin)))
    boundary_points = np.asarray(outline.exterior.coords[:-1], dtype=float)
    return boundary_points[int(np.argmin(np.linalg.norm(boundary_points - origin, axis=1)))]


def generate_house_hull_palm(
    hand: dict,
    attachment_patches: list[AttachmentPatch],
    wrist_patch: AttachmentPatch,
    fixed_template_mesh: trimesh.Trimesh,
    palm_rotation: np.ndarray,
) -> PalmMeshResult:
    """House of Dextra style graph-to-palm compiler.

    Finger/motor frames come from HandIR.  Two tangential grip points and a
    small rectangular mounting footprint are emitted around every motor base;
    an original-palm-sized center grid and wrist footprint guarantee central
    coverage.  Their 2D convex hull is extruded at the *source palm thickness*.
    No source surface vertex is stretched, so radial layouts cannot create RBF
    spikes or folded CAD shells.
    """
    rotation = np.asarray(palm_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("palm_rotation must be a proper 3x3 rotation")

    template_local = np.asarray(fixed_template_mesh.vertices, dtype=float) @ rotation
    template_uv = template_local[:, [0, 2]]
    template_bounds = np.asarray([template_uv.min(axis=0), template_uv.max(axis=0)])
    template_extent = np.maximum(template_bounds[1] - template_bounds[0], 1.0e-4)
    template_center = 0.5 * (template_bounds[0] + template_bounds[1])
    thickness_bounds = template_local[:, 1].min(), template_local[:, 1].max()
    thickness = float(thickness_bounds[1] - thickness_bounds[0])

    grip_points: list[np.ndarray] = []
    footprint_records = []
    for slot, patch in zip(hand["finger_slots"], attachment_patches):
        center3 = np.asarray(slot["attachment_translation"], dtype=float) @ rotation
        frame = rotation.T @ np.asarray(slot["attachment_rotation"], dtype=float)
        center = center3[[0, 2]]
        outward = _normalize_2d(frame[[0, 2], 2])
        tangent = np.asarray([-outward[1], outward[0]], dtype=float)
        half_width = 0.58 * float(patch.width)
        half_depth = 0.34 * float(patch.depth)
        # The official generator uses two tangential grip points per motor.
        # We additionally include the inner pair so the actual source-specific
        # motor housing, not only an abstract centerline, is supported.
        points = np.asarray([
            center + half_width * tangent + half_depth * outward,
            center - half_width * tangent + half_depth * outward,
            center + half_width * tangent - half_depth * outward,
            center - half_width * tangent - half_depth * outward,
        ])
        grip_points.extend(points)
        footprint_records.append({
            "slot_id": int(slot["slot_id"]),
            "role": slot["role"],
            "center_local": center.tolist(),
            "outward_local": outward.tolist(),
            "grip_points_local": points.tolist(),
        })

    # Equivalent to House of Dextra's minimum center grid, but its size is
    # calibrated from the selected source palm instead of a single hardware
    # family.  It prevents a convex hull made only from motors from collapsing
    # into a thin spoke or triangle.
    center = np.asarray(
        hand.get("palm_layout", {}).get("center_local_xz", template_center), dtype=float
    )
    if center.shape != (2,):
        center = template_center
    half_center = np.maximum(0.23 * template_extent, 0.08 * template_extent.max())
    for u in np.linspace(-half_center[0], half_center[0], 5):
        for v in np.linspace(-half_center[1], half_center[1], 5):
            grip_points.append(center + np.asarray([u, v]))

    wrist_center3 = wrist_patch.transform[:3, 3] @ rotation
    wrist_frame = rotation.T @ wrist_patch.transform[:3, :3]
    wrist_outward = _normalize_2d(wrist_frame[[0, 2], 2])
    wrist_tangent = np.asarray([-wrist_outward[1], wrist_outward[0]], dtype=float)
    wrist_center = wrist_center3[[0, 2]]
    for tangent_sign in (-1.0, 1.0):
        for depth_sign in (-1.0, 1.0):
            grip_points.append(
                wrist_center
                + tangent_sign * 0.5 * wrist_patch.width * wrist_tangent
                + depth_sign * 0.5 * wrist_patch.depth * wrist_outward
            )

    points = np.asarray(grip_points, dtype=float)
    hull = ConvexHull(points, qhull_options="Qx")
    outline = Polygon(points[np.asarray(hull.vertices, dtype=np.int64)])
    if not outline.is_valid or outline.area <= 1.0e-10:
        raise ValueError("graph motor footprints cannot form a valid palm hull")

    local_params = PalmGeometryParams(
        thickness=thickness,
        boundary_margin=0.0,
        edge_rounding_radius=0.0,
        wrist_width=wrist_patch.width,
    )
    local_mesh = _extrude_outline(
        outline,
        center_normal=0.5 * float(thickness_bounds[0] + thickness_bounds[1]),
        thickness=thickness,
        params=local_params,
    )
    local_mesh.vertices = np.asarray(local_mesh.vertices, dtype=float) @ rotation.T
    local_mesh.fix_normals(multibody=True)
    visual = local_mesh.copy()
    collision = local_mesh.copy()

    coverage = []
    for record in footprint_records:
        covered = outline.buffer(1.0e-9).covers(Point(record["center_local"]))
        record["center_covered"] = bool(covered)
        coverage.append(bool(covered))
    frames = {
        patch.name: patch.transform.copy()
        for patch in [*attachment_patches, wrist_patch]
    }
    quality = _mesh_quality(visual)
    metadata = {
        "mode": "house_hull",
        "algorithm": "motor grip footprints + center grid + 2D convex hull + fixed-thickness extrusion",
        "frame_source": "graph_passthrough",
        "layout_mode": hand.get("palm_layout", {}).get("mode"),
        "source_palm_thickness": thickness,
        "generated_palm_thickness": float(np.ptp((np.asarray(visual.vertices) @ rotation)[:, 1])),
        "thickness_error": float(abs(np.ptp((np.asarray(visual.vertices) @ rotation)[:, 1]) - thickness)),
        "outline_coordinates": np.asarray(outline.exterior.coords, dtype=float).tolist(),
        "outline_area": float(outline.area),
        "motor_footprints": footprint_records,
        "all_motor_centers_covered": all(coverage),
        "center_grid_half_extent": half_center.tolist(),
        "visual_quality": quality,
        "collision_quality": _mesh_quality(collision),
    }
    if not metadata["all_motor_centers_covered"]:
        raise ValueError("generated House palm does not cover every motor center")
    if metadata["thickness_error"] > 1.0e-10:
        raise ValueError("generated House palm changed source palm thickness")
    if not visual.is_watertight or not visual.is_winding_consistent:
        raise ValueError("generated House palm is not a valid watertight solid")
    return PalmMeshResult(visual, collision, frames, metadata)


def deform_template_to_house_palm(
    hand: dict,
    attachment_patches: list[AttachmentPatch],
    wrist_patch: AttachmentPatch,
    fixed_template_mesh: trimesh.Trimesh,
    palm_rotation: np.ndarray,
) -> PalmMeshResult:
    """Preserve source palm topology while following a House-style graph hull.

    The official convex palm is used as a smooth target *cage*, not as the
    final visual mesh.  Every original vertex keeps its local thickness
    coordinate.  A radial convex-to-convex map changes only the palm plane,
    while a root-mount disk is exactly locked and surrounded by a smooth
    transition.  This avoids the isolated control-point spikes produced by
    the removed large-displacement RBF implementation.
    """
    rotation = np.asarray(palm_rotation, dtype=float)
    guide = generate_house_hull_palm(
        hand, attachment_patches, wrist_patch, fixed_template_mesh, rotation
    )
    target_outline = Polygon(guide.metadata["outline_coordinates"])

    visual = fixed_template_mesh.copy()
    local_vertices = np.asarray(visual.vertices, dtype=float) @ rotation
    original_local = local_vertices.copy()
    uv = local_vertices[:, [0, 2]]
    source_scale = max(float(np.max(np.ptp(uv, axis=0))), 1.0e-5)
    # A finite source boundary cannot reproduce every corner of a target cage
    # exactly.  Offset the guide slightly so the deformed *source* surface,
    # not merely the mathematical hull, still reaches each motor interface.
    target_outline = target_outline.buffer(0.045 * source_scale, join_style=1)
    source_hull = ConvexHull(uv, qhull_options="Qx")
    source_outline = Polygon(uv[np.asarray(source_hull.vertices, dtype=np.int64)])
    if not source_outline.is_valid or source_outline.area <= 1.0e-12:
        raise ValueError("source palm cannot form a valid topology deformation cage")

    shared = source_outline.intersection(target_outline)
    if shared.is_empty:
        raise ValueError("source palm and graph target palm have no common deformation region")
    layout_center = np.asarray(
        hand.get("palm_layout", {}).get("center_local_xz", [np.nan, np.nan]),
        dtype=float,
    )
    layout_center_point = Point(layout_center) if np.all(np.isfinite(layout_center)) else None
    # The graph layout center is the natural polar origin and keeps the warp
    # stable as its target cage grows.  Fall back only for legacy inputs whose
    # center is not inside both source and target footprints.
    if (
        layout_center_point is not None
        and source_outline.buffer(1.0e-9).covers(layout_center_point)
        and target_outline.buffer(1.0e-9).covers(layout_center_point)
    ):
        center_point = layout_center_point
    else:
        center_point = shared.representative_point()
    center = np.asarray(center_point.coords[0], dtype=float)

    # Sample smooth periodic support functions for both convex cages.  Every
    # source vertex keeps its polar angle and normalized radius, producing a
    # global coherent deformation rather than local tentacles.
    sample_count = 720
    sample_angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    source_radius = np.empty(sample_count, dtype=float)
    target_radius = np.empty(sample_count, dtype=float)
    for index, angle in enumerate(sample_angles):
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        source_point = _ray_boundary_point(source_outline, center, direction)
        target_point = _ray_boundary_point(target_outline, center, direction)
        source_radius[index] = max(float(np.dot(source_point - center, direction)), 1.0e-8)
        target_radius[index] = max(float(np.dot(target_point - center, direction)), 1.0e-8)

    vector = uv - center
    radius = np.linalg.norm(vector, axis=1)
    angle = np.mod(np.arctan2(vector[:, 1], vector[:, 0]), 2.0 * np.pi)
    xp = np.r_[sample_angles, 2.0 * np.pi]
    source_interp = np.interp(angle, xp, np.r_[source_radius, source_radius[0]])
    target_interp = np.interp(angle, xp, np.r_[target_radius, target_radius[0]])
    direction = vector / np.maximum(radius[:, None], 1.0e-10)
    normalized_radius = np.clip(radius / source_interp, 0.0, 1.05)
    mapped_uv = center + direction * (normalized_radius * target_interp)[:, None]
    mapped_uv[radius <= 1.0e-10] = center

    wrist_center3 = wrist_patch.transform[:3, 3] @ rotation
    wrist_center = wrist_center3[[0, 2]]
    lock_inner = max(
        0.62 * float(wrist_patch.width),
        0.62 * float(wrist_patch.depth),
        0.11 * source_scale,
    )
    lock_outer = 1.85 * lock_inner
    mount_distance = np.linalg.norm(uv - wrist_center, axis=1)
    blend = _smoothstep((mount_distance - lock_inner) / max(lock_outer - lock_inner, 1.0e-8))
    protected_base = np.zeros(len(uv), dtype=bool)
    if hand.get("protected_transmission_source", False):
        finger_centers = np.asarray([
            (np.asarray(slot["attachment_translation"], dtype=float) @ rotation)[[0, 2]]
            for slot in hand["finger_slots"]
        ])
        forward = _normalize_2d(np.mean(finger_centers, axis=0) - wrist_center)
        longitudinal = (uv - wrist_center) @ forward
        base_lock_end = 0.10 * source_scale
        base_fade_end = 0.28 * source_scale
        base_blend = _smoothstep(
            (longitudinal - base_lock_end) / max(base_fade_end - base_lock_end, 1.0e-8)
        )
        blend *= base_blend
        protected_base = longitudinal <= base_lock_end
    deformed_uv = uv + blend[:, None] * (mapped_uv - uv)
    locked = (mount_distance <= lock_inner) | protected_base
    deformed_uv[locked] = uv[locked]

    local_vertices[:, 0] = deformed_uv[:, 0]
    local_vertices[:, 2] = deformed_uv[:, 1]
    local_vertices[:, 1] = original_local[:, 1]
    visual.vertices = local_vertices @ rotation.T
    visual.remove_unreferenced_vertices()
    visual.fix_normals(multibody=True)

    deformed_hull = ConvexHull(deformed_uv, qhull_options="Qx")
    deformed_outline = Polygon(deformed_uv[np.asarray(deformed_hull.vertices, dtype=np.int64)])
    thickness_bounds = original_local[:, 1].min(), original_local[:, 1].max()
    thickness = float(thickness_bounds[1] - thickness_bounds[0])
    collision_params = PalmGeometryParams(
        thickness=thickness,
        boundary_margin=0.0,
        edge_rounding_radius=0.0,
        wrist_width=wrist_patch.width,
    )
    collision = _extrude_outline(
        deformed_outline,
        center_normal=0.5 * float(thickness_bounds[0] + thickness_bounds[1]),
        thickness=thickness,
        params=collision_params,
    )
    collision.vertices = np.asarray(collision.vertices, dtype=float) @ rotation.T
    collision.fix_normals(multibody=True)

    motor_records = []
    for slot, patch in zip(hand["finger_slots"], attachment_patches):
        center3 = np.asarray(slot["attachment_translation"], dtype=float) @ rotation
        motor_center = center3[[0, 2]]
        center_distance = float(deformed_outline.distance(Point(motor_center)))
        covered = bool(deformed_outline.buffer(1.0e-9).covers(Point(motor_center)))
        # House places a motor center on the target boundary.  With an
        # unchanged finite source topology, mapped boundary vertices form
        # chords and need not contain that mathematical center exactly.  The
        # mechanical condition is that the motor's interface footprint
        # intersects the palm, so allow at most half of its smaller footprint
        # dimension.  The compiler separately checks the actual root mesh.
        interface_reach = 0.5 * min(float(patch.width), float(patch.depth))
        interface_reached = center_distance <= interface_reach + 1.0e-9
        motor_records.append({
            "slot_id": int(slot["slot_id"]),
            "role": slot["role"],
            "center_local": motor_center.tolist(),
            "covered_by_deformed_palm": covered,
            "distance_to_deformed_outline": center_distance,
            "interface_reach": interface_reach,
            "interface_reached": interface_reached,
        })

    frames = {
        patch.name: patch.transform.copy()
        for patch in [*attachment_patches, wrist_patch]
    }
    thickness_error = float(np.max(np.abs(local_vertices[:, 1] - original_local[:, 1])))
    locked_error = float(np.max(
        np.linalg.norm(deformed_uv[locked] - uv[locked], axis=1)
    )) if np.any(locked) else 0.0
    metadata = {
        "mode": "source_topology_house",
        "algorithm": "House motor-footprint hull as target cage + global radial map + locked root mount",
        "frame_source": "graph_passthrough",
        "layout_mode": hand.get("palm_layout", {}).get("mode"),
        "template_vertex_count_preserved": len(visual.vertices) == len(fixed_template_mesh.vertices),
        "template_face_count_preserved": len(visual.faces) == len(fixed_template_mesh.faces),
        "local_thickness_coordinate_max_error": thickness_error,
        "original_local_thickness": thickness,
        "deformed_local_thickness": float(np.ptp(local_vertices[:, 1])),
        "root_mount_center_local": wrist_center.tolist(),
        "root_mount_lock_inner": lock_inner,
        "root_mount_lock_outer": lock_outer,
        "root_mount_locked_vertex_count": int(np.count_nonzero(locked)),
        "root_mount_locked_max_error": locked_error,
        "protected_transmission_base_vertex_count": int(np.count_nonzero(protected_base)),
        "protected_transmission_base_locked": bool(
            not hand.get("protected_transmission_source", False) or np.any(protected_base)
        ),
        "maximum_planar_vertex_displacement": float(
            np.max(np.linalg.norm(deformed_uv - uv, axis=1))
        ),
        "maximum_planar_displacement_fraction": float(
            np.max(np.linalg.norm(deformed_uv - uv, axis=1)) / source_scale
        ),
        "source_outline_coordinates": np.asarray(source_outline.exterior.coords, dtype=float).tolist(),
        "target_cage_outline_coordinates": np.asarray(target_outline.exterior.coords, dtype=float).tolist(),
        "deformed_outline_coordinates": np.asarray(deformed_outline.exterior.coords, dtype=float).tolist(),
        "motor_centers": motor_records,
        "all_motor_centers_covered": all(record["covered_by_deformed_palm"] for record in motor_records),
        "all_motor_interfaces_reached": all(record["interface_reached"] for record in motor_records),
        "visual_quality": _mesh_quality(visual),
        "collision_quality": _mesh_quality(collision),
    }
    if thickness_error != 0.0:
        raise ValueError("source-topology palm deformation changed a thickness coordinate")
    if locked_error != 0.0:
        raise ValueError("source-topology palm deformation moved the root mount region")
    if not metadata["template_vertex_count_preserved"] or not metadata["template_face_count_preserved"]:
        raise ValueError("source-topology palm deformation changed visual topology")
    if metadata["maximum_planar_displacement_fraction"] > 0.30:
        raise ValueError(
            "source-topology palm exceeds the 30% manufacturing deformation limit"
        )
    # Center-to-outline distance is diagnostic only: the actual candidate
    # motor/root mesh can be substantially larger than this generic patch.
    # compile_meshes performs the authoritative post-fill overlap audit using
    # the concrete palm and finger-root visuals.
    if not collision.is_watertight or not collision.is_winding_consistent:
        raise ValueError("source-topology palm collision mesh is invalid")
    return PalmMeshResult(visual, collision, frames, metadata)


def deform_template_palm(
    hand: dict,
    attachment_patches: list[AttachmentPatch],
    wrist_patch: AttachmentPatch,
    fixed_template_mesh: trimesh.Trimesh,
    palm_rotation: np.ndarray,
) -> PalmMeshResult:
    """Deform the original palm only in its local palm plane.

    This adapts the grip-point idea used by House of Dextra to an existing palm
    asset: finger motor/joint frames select boundary grip regions, but the
    original vertices, faces, surface detail, and local thickness coordinates
    remain the template.  It does not replace the palm with a new prism.
    """
    rotation = np.asarray(palm_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("palm_rotation must be a proper 3x3 rotation")
    visual = fixed_template_mesh.copy()
    world_vertices = np.asarray(visual.vertices, dtype=float)
    local_vertices = world_vertices @ rotation
    original_local_vertices = local_vertices.copy()
    uv = local_vertices[:, [0, 2]]
    # Do not joggle dense CAD projections: QJ can retain nearly duplicate
    # boundary points and return a self-touching Shapely polygon.
    hull = ConvexHull(uv, qhull_options="Qx")
    hull_indices = np.asarray(hull.vertices, dtype=np.int64)
    source_outline = Polygon(uv[hull_indices])
    if not source_outline.is_valid or source_outline.area <= 1.0e-12:
        raise ValueError("original palm cannot produce a valid planar footprint")

    patch_by_slot = {
        int(slot["slot_id"]): patch
        for slot, patch in zip(hand["finger_slots"], attachment_patches)
    }
    control_delta: dict[int, list[np.ndarray]] = {}
    mounting_records = []
    used: set[int] = set()
    palm_scale = max(float(np.linalg.norm(np.asarray(source_outline.bounds)[2:] - np.asarray(source_outline.bounds)[:2])), 1.0e-6)

    for slot in hand["finger_slots"]:
        slot_id = int(slot["slot_id"])
        patch = patch_by_slot[slot_id]
        current_translation = np.asarray(slot["attachment_translation"], dtype=float) @ rotation
        current_rotation = rotation.T @ np.asarray(slot["attachment_rotation"], dtype=float)
        reference_translation = np.asarray(
            slot.get("reference_attachment_translation", slot["attachment_translation"]), dtype=float
        ) @ rotation
        reference_rotation = rotation.T @ np.asarray(
            slot.get("reference_attachment_rotation", slot["attachment_rotation"]), dtype=float
        )
        source_center = _ray_boundary_point(
            source_outline,
            reference_translation[[0, 2]],
            reference_rotation[[0, 2], 2],
        )
        planar_delta = current_translation[[0, 2]] - reference_translation[[0, 2]]
        outward = _normalize_2d(reference_rotation[[0, 2], 2])
        tangent = np.asarray([-outward[1], outward[0]], dtype=float)
        selected = []
        for offset in (-0.42 * patch.width, 0.0, 0.42 * patch.width):
            query = source_center + offset * tangent
            order = np.argsort(np.linalg.norm(uv[hull_indices] - query, axis=1))
            vertex_index = None
            for candidate in hull_indices[order]:
                if int(candidate) not in used:
                    vertex_index = int(candidate)
                    break
            if vertex_index is None:
                vertex_index = int(hull_indices[order[0]])
            used.add(vertex_index)
            control_delta.setdefault(vertex_index, []).append(planar_delta)
            selected.append(vertex_index)
        center_vertex = selected[1]
        mounting_records.append({
            "slot_id": slot_id,
            "role": slot["role"],
            "joint_origin_local": current_translation.tolist(),
            "joint_outward_local": current_rotation[:, 2].tolist(),
            "reference_boundary_point": uv[center_vertex].tolist(),
            "control_target_local": (uv[center_vertex] + planar_delta).tolist(),
            "layout_delta": planar_delta.tolist(),
            "control_vertex_ids": selected,
        })

    # Keep several source boundary landmarks fixed, especially around the
    # wrist, so local finger-layout edits do not turn into a global palm warp.
    hull_uv = uv[hull_indices]
    z_threshold = float(np.quantile(hull_uv[:, 1], 0.42))
    fixed_candidates = [int(index) for index in hull_indices if uv[int(index), 1] <= z_threshold]
    stride = max(1, len(fixed_candidates) // 8)
    for index in fixed_candidates[::stride][:8]:
        if index not in used:
            control_delta.setdefault(index, []).append(np.zeros(2))

    control_indices = np.asarray(sorted(control_delta), dtype=np.int64)
    deltas = np.asarray([
        np.mean(control_delta[int(index)], axis=0) for index in control_indices
    ])
    maximum_allowed = 0.14 * palm_scale
    lengths = np.linalg.norm(deltas, axis=1)
    too_large = lengths > maximum_allowed
    if np.any(too_large):
        deltas[too_large] *= (maximum_allowed / lengths[too_large])[:, None]
    controls = uv[control_indices]
    sigma = max(0.22 * palm_scale, 1.0e-5)
    distance_squared = np.sum((uv[:, None, :] - controls[None, :, :]) ** 2, axis=2)
    weights = np.exp(-0.5 * distance_squared / (sigma * sigma))
    displacement = (weights @ deltas) / (1.25 + weights.sum(axis=1, keepdims=True))
    deformed_uv = uv + displacement
    # Exact control interpolation prevents the mounting region from drifting
    # away from the graph-selected edge location.
    deformed_uv[control_indices] = uv[control_indices] + deltas

    # The wrist/mount is a mechanical interface, not a morphology variable.
    # Keep an inner disk bit-identical to the source mesh and fade palm edits
    # in only outside a surrounding transition annulus.
    wrist_center3 = wrist_patch.transform[:3, 3] @ rotation
    wrist_center = wrist_center3[[0, 2]]
    lock_inner = max(
        0.62 * float(wrist_patch.width),
        0.62 * float(wrist_patch.depth),
        0.11 * palm_scale,
    )
    lock_outer = 1.85 * lock_inner
    mount_distance = np.linalg.norm(uv - wrist_center, axis=1)
    mount_blend = _smoothstep(
        (mount_distance - lock_inner) / max(lock_outer - lock_inner, 1.0e-8)
    )
    protected_base = np.zeros(len(uv), dtype=bool)
    if hand.get("protected_transmission_source", False):
        finger_centers = np.asarray([
            (np.asarray(slot["attachment_translation"], dtype=float) @ rotation)[[0, 2]]
            for slot in hand["finger_slots"]
        ])
        forward = _normalize_2d(np.mean(finger_centers, axis=0) - wrist_center)
        longitudinal = (uv - wrist_center) @ forward
        base_lock_end = 0.10 * palm_scale
        base_fade_end = 0.28 * palm_scale
        base_blend = _smoothstep(
            (longitudinal - base_lock_end) / max(base_fade_end - base_lock_end, 1.0e-8)
        )
        mount_blend *= base_blend
        protected_base = longitudinal <= base_lock_end
    deformed_uv = uv + mount_blend[:, None] * (deformed_uv - uv)
    locked = (mount_distance <= lock_inner) | protected_base
    deformed_uv[locked] = uv[locked]
    local_vertices[:, 0] = deformed_uv[:, 0]
    local_vertices[:, 2] = deformed_uv[:, 1]
    # local Y is copied verbatim: no thickness scaling and no arch/cup warp.
    local_vertices[:, 1] = original_local_vertices[:, 1]
    visual.vertices = local_vertices @ rotation.T
    visual.remove_unreferenced_vertices()
    visual.fix_normals(multibody=True)

    deformed_hull = ConvexHull(deformed_uv, qhull_options="Qx")
    target_outline = Polygon(deformed_uv[np.asarray(deformed_hull.vertices, dtype=np.int64)])
    thickness_bounds = original_local_vertices[:, 1].min(), original_local_vertices[:, 1].max()
    collision_params = PalmGeometryParams(
        thickness=float(thickness_bounds[1] - thickness_bounds[0]),
        boundary_margin=0.0,
        edge_rounding_radius=0.0,
        wrist_width=wrist_patch.width,
    )
    collision_local = _extrude_outline(
        target_outline,
        center_normal=0.5 * float(thickness_bounds[0] + thickness_bounds[1]),
        thickness=float(thickness_bounds[1] - thickness_bounds[0]),
        params=collision_params,
    )
    collision_local.vertices = np.asarray(collision_local.vertices) @ rotation.T
    collision_local.fix_normals(multibody=True)

    target_boundary = target_outline.boundary
    for record in mounting_records:
        origin = np.asarray(record["joint_origin_local"], dtype=float)[[0, 2]]
        outward = np.asarray(record["joint_outward_local"], dtype=float)[[0, 2]]
        mounting_point = _ray_boundary_point(target_outline, origin, outward)
        record["mounting_point_local"] = mounting_point.tolist()
        record["mounting_shift_from_control_target"] = float(np.linalg.norm(
            mounting_point - np.asarray(record["control_target_local"], dtype=float)
        ))
        point = Point(record["mounting_point_local"])
        record["target_boundary_distance"] = float(target_boundary.distance(point))
        record["on_target_boundary"] = record["target_boundary_distance"] <= 0.02 * palm_scale

    frames = {
        patch.name: patch.transform.copy()
        for patch in [*attachment_patches, wrist_patch]
    }
    thickness_error = float(np.max(np.abs(local_vertices[:, 1] - original_local_vertices[:, 1])))
    locked_error = float(np.max(
        np.linalg.norm(deformed_uv[locked] - uv[locked], axis=1)
    )) if np.any(locked) else 0.0
    metadata = {
        "mode": "template_deform",
        "frame_source": "graph_passthrough",
        "template_vertex_count_preserved": len(visual.vertices) == len(fixed_template_mesh.vertices),
        "template_face_count_preserved": len(visual.faces) == len(fixed_template_mesh.faces),
        "local_thickness_coordinate_max_error": thickness_error,
        "original_local_thickness": float(np.ptp(original_local_vertices[:, 1])),
        "deformed_local_thickness": float(np.ptp(local_vertices[:, 1])),
        "maximum_planar_vertex_displacement": float(
            np.linalg.norm(deformed_uv - uv, axis=1).max()
        ),
        "maximum_planar_displacement_fraction": float(
            np.linalg.norm(deformed_uv - uv, axis=1).max() / palm_scale
        ),
        "root_mount_center_local": wrist_center.tolist(),
        "root_mount_lock_inner": lock_inner,
        "root_mount_lock_outer": lock_outer,
        "root_mount_locked_vertex_count": int(np.count_nonzero(locked)),
        "root_mount_locked_max_error": locked_error,
        "protected_transmission_base_vertex_count": int(np.count_nonzero(protected_base)),
        "protected_transmission_base_locked": bool(
            not hand.get("protected_transmission_source", False) or np.any(protected_base)
        ),
        "mounting_points": mounting_records,
        "all_mounting_points_on_boundary": all(record["on_target_boundary"] for record in mounting_records),
        "source_outline_coordinates": np.asarray(source_outline.exterior.coords, dtype=float).tolist(),
        "target_outline_coordinates": np.asarray(target_outline.exterior.coords, dtype=float).tolist(),
        "visual_quality": _mesh_quality(visual),
        "collision_quality": _mesh_quality(collision_local),
    }
    if thickness_error != 0.0:
        raise ValueError(f"template palm deformation changed local thickness coordinates: {thickness_error}")
    if locked_error != 0.0:
        raise ValueError("template palm deformation moved the root mount region")
    if not metadata["all_mounting_points_on_boundary"]:
        raise ValueError("one or more finger mounting regions moved into the palm interior")
    if not collision_local.is_watertight or not collision_local.is_winding_consistent:
        raise ValueError("template-deformed palm collision mesh is invalid")
    return PalmMeshResult(visual, collision_local, frames, metadata)


def infer_palm_params(
    fixed_palm_mesh: trimesh.Trimesh,
    *,
    transverse_arch: float = 0.0,
    longitudinal_arch: float = 0.0,
    central_cup: float = 0.0,
) -> PalmGeometryParams:
    size = np.maximum(np.asarray(fixed_palm_mesh.extents, dtype=float), 1.0e-4)
    planar_scale = float(max(size[0], size[2]))
    return PalmGeometryParams(
        thickness=float(size[1]),
        boundary_margin=0.045 * planar_scale,
        edge_rounding_radius=0.035 * planar_scale,
        wrist_width=max(0.42 * float(size[0]), 0.08 * planar_scale),
        transverse_arch=float(transverse_arch),
        longitudinal_arch=float(longitudinal_arch),
        central_cup=float(central_cup),
    )


def patches_from_hand_ir(
    hand: dict,
    fixed_palm_mesh: trimesh.Trimesh,
    params: PalmGeometryParams,
) -> tuple[list[AttachmentPatch], AttachmentPatch]:
    """Convert existing HandIR slots to patches without changing the graph."""
    extents = np.maximum(np.asarray(fixed_palm_mesh.extents, dtype=float), 1.0e-4)
    planar_width, planar_length = float(extents[0]), float(extents[2])
    patch_depth = max(0.10 * planar_length, 0.035 * max(planar_width, planar_length))
    patch_thickness = max(0.22 * float(params.thickness), 1.0e-4)
    patches: list[AttachmentPatch] = []
    for slot in hand["finger_slots"]:
        transform = transform_from_rotation_translation(
            np.asarray(slot["attachment_rotation"], dtype=float),
            np.asarray(slot["attachment_translation"], dtype=float),
        )
        role_scale = 1.10 if slot["role"] == "thumb" else 1.0
        patches.append(AttachmentPatch(
            name=f"finger_slot_{int(slot['slot_id'])}_{slot['role']}",
            transform=transform,
            width=max(role_scale * 0.16 * planar_width, 0.045 * planar_length),
            depth=patch_depth,
            thickness=patch_thickness,
            locked=True,
        ))

    plane_axes = params.plane_axes
    normal_slots = [slot for slot in hand["finger_slots"] if slot["role"] != "thumb"]
    source = normal_slots if normal_slots else hand["finger_slots"]
    target = np.mean([np.asarray(slot["attachment_translation"], dtype=float) for slot in source], axis=0)
    root = np.asarray(hand["parts"][0].get("world_pos", [0.0, 0.0, 0.0]), dtype=float)
    direction = target - root
    direction[params.thickness_axis] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8:
        direction = np.zeros(3)
        direction[plane_axes[1]] = 1.0
    else:
        direction /= norm
    normal = np.zeros(3)
    normal[params.thickness_axis] = 1.0
    width_axis = np.cross(normal, direction)
    width_axis /= max(float(np.linalg.norm(width_axis)), 1.0e-8)
    rotation = np.column_stack([width_axis, normal, direction])
    wrist = AttachmentPatch(
        name="wrist_root",
        transform=transform_from_rotation_translation(rotation, root),
        width=float(params.wrist_width),
        depth=max(0.14 * planar_length, patch_depth),
        thickness=patch_thickness,
        locked=True,
    )
    return patches, wrist
