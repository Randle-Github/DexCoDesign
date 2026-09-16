"""Connector-preserving mesh overlays for prototype-based USD evaluation.

The affine overlay alone cannot represent a stretched middle span with rigid
connector caps. These helpers bake that missing, non-affine deformation into
candidate-local mesh points, using the same operation as the full compiler.
Prototype USDs and the training/retargeting algorithms are not modified.
"""

from __future__ import annotations

import json

import numpy as np


def _is_identity_deformation(specification: dict) -> bool:
    return bool(
        np.isclose(
            specification["source_length_canonical"],
            specification["target_length_canonical"],
            rtol=0.0,
            atol=1.0e-12,
        )
        and np.isclose(specification["width_scale"], 1.0, rtol=0.0, atol=1.0e-12)
    )


def connector_deformation_overlay(
    part: dict,
    prototype_part: dict,
    polar_linear: np.ndarray,
    source_scale: float,
) -> dict | None:
    """Describe the compiler deformation in the template's exported frame.

    Bank fingers must be undeformed: an affine inverse cannot undo an already
    piecewise-deformed prototype. Reject that case rather than silently producing
    another incorrect contact mesh.
    """
    specification = part.get("connector_cap_deformation")
    if specification is None:
        return None
    baseline_specification = prototype_part.get("connector_cap_deformation")
    if baseline_specification and not _is_identity_deformation(baseline_specification):
        raise ValueError(f"part {part['id']}: connector overlays require undeformed prototype fingers")
    if _is_identity_deformation(specification):
        return None
    if not np.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError("source similarity scale must be positive and finite")
    baseline_export = np.asarray(prototype_part["mesh_linear"], dtype=np.float64).T @ polar_linear
    current_linear = np.asarray(part["mesh_linear"], dtype=np.float64)
    # Template vertices -> rotated canonical vertices before cap deformation.
    # The inverse mapping puts the baked points back before the separate USD
    # affine op, so that op is applied exactly once to both visuals/collisions.
    canonical_from_template = np.linalg.solve(baseline_export, current_linear.T) * source_scale
    return {
        "connector_cap_deformation": specification,
        "canonical_from_template": canonical_from_template.tolist(),
    }


def deform_template_points(points: np.ndarray, overlay: dict) -> np.ndarray:
    """Apply the full compiler's non-affine operation without changing frames."""
    import trimesh

    from .mesh_compiler import apply_midas_axis_deformation

    mapping = np.asarray(overlay["canonical_from_template"], dtype=np.float64)
    canonical = np.asarray(points, dtype=np.float64) @ mapping
    mesh = trimesh.Trimesh(vertices=canonical, process=False)
    deformed = apply_midas_axis_deformation(mesh, overlay["connector_cap_deformation"])
    return np.asarray(deformed.vertices) @ np.linalg.inv(mapping)


def deform_link_meshes(stage, link_path, overlay: dict, affine: np.ndarray) -> int:
    """Bake points in both scopes after their affine ops have been authored.

    All geometry is read in the link frame, including nested mesh transforms, and
    then returned to each mesh's own frame. This preserves its topology, collision
    APIs, materials and references. Edits are opinions in the candidate/scene layer,
    never in the referenced prototype layer. Reapplying the same overlay is a no-op.
    """
    from pxr import Sdf, Usd, UsdGeom, Vt

    link = stage.GetPrimAtPath(link_path)
    if not link.IsValid():
        raise ValueError(f"missing morphology link: {link_path}")
    signature = json.dumps([overlay, np.asarray(affine).tolist()], sort_keys=True)
    marker = link.GetAttribute("dexcodesign:connectorDeformation")
    if marker and marker.HasAuthoredValueOpinion():
        if marker.Get() == signature:
            return 0
        raise ValueError(f"{link_path}: rebuild from the prototype before changing baked deformation")

    # Referenced geometry scopes can be instance roots, even when the link is
    # not. Deinstance in the current edit layer before overriding mesh points.
    pending = [link]
    while pending:
        prim = pending.pop()
        if prim.IsInstance() or prim.IsInstanceable():
            prim.SetInstanceable(False)
        pending.extend(prim.GetChildren())

    cache = UsdGeom.XformCache()
    world_to_link = np.asarray(cache.GetLocalToWorldTransform(link).GetInverse())
    inverse_affine = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    meshes = []
    template_points = []
    collision_mesh_count = 0
    for scope_name in ("collisions", "visuals"):
        scope = stage.GetPrimAtPath(link.GetPath().AppendChild(scope_name))
        if not scope.IsValid():
            continue
        for prim in Usd.PrimRange(scope):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                raise ValueError(f"empty or invalid morphology mesh: {prim.GetPath()}")
            mesh_to_template = np.asarray(cache.GetLocalToWorldTransform(prim)) @ world_to_link @ inverse_affine
            homogeneous = np.column_stack((points, np.ones(len(points))))
            template_points.append((homogeneous @ mesh_to_template)[:, :3])
            meshes.append((mesh, mesh_to_template, len(points)))
            collision_mesh_count += scope_name == "collisions"
    if not collision_mesh_count:
        raise ValueError(f"{link_path}: connector deformation found no collision mesh")

    # One body-wide width centre, not a different centre for each mesh chunk.
    baked = deform_template_points(np.concatenate(template_points), overlay)
    begin = 0
    for mesh, mesh_to_template, count in meshes:
        points = baked[begin : begin + count]
        homogeneous = np.column_stack((points, np.ones(count)))
        local = (homogeneous @ np.linalg.inv(mesh_to_template))[:, :3]
        if not np.isfinite(local).all():
            raise ValueError(f"non-finite deformed mesh: {mesh.GetPath()}")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(local, dtype=np.float32)))
        mesh.GetExtentAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.asarray([local.min(axis=0), local.max(axis=0)], dtype=np.float32))
        )
        # Normals and explicitly stored cooked blobs refer to the old points.
        # Let rendering/PhysX rebuild them from the corrected candidate mesh.
        for attribute in mesh.GetPrim().GetAttributes():
            name = attribute.GetName()
            if name in ("normals", "primvars:normals") or "cookedData" in name:
                attribute.Block()
        begin += count
    link.CreateAttribute("dexcodesign:connectorDeformation", Sdf.ValueTypeNames.String).Set(signature)
    return len(meshes)
