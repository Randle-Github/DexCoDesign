"""Fast WUJI contact meshes must match the connector-preserving compiler."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from dexcodesign.morphology.mesh_compiler import apply_midas_axis_deformation
from dexcodesign.morphology.parametric_mesh import (
    connector_deformation_overlay,
    deform_link_meshes,
    deform_template_points,
)


ROOT = Path(__file__).resolve().parents[1]


def example(length=1.6, width=1.25):
    source = np.array(
        [[x, y, z] for z in (-0.1, 0.05, 0.12, 0.3, 0.7, 0.88, 1.0, 1.1) for x in (-0.2, 0.3) for y in (-0.1, 0.1)]
    )
    current = Rotation.from_euler("xyz", [0.15, -0.25, 0.2]).as_matrix()
    baseline = Rotation.from_euler("xyz", [-0.1, 0.05, -0.3]).as_matrix()
    polar = Rotation.from_euler("xyz", [0.4, 0.2, -0.2]).as_matrix() @ np.diag([-1, 1, 1])
    scale = 9.5
    specification = {
        "longitudinal_axis": (current @ [0, 0, 1]).tolist(),
        "width_axis": (current @ [1, 0, 0]).tolist(),
        "source_length_canonical": 1.0,
        "target_length_canonical": length,
        "width_scale": width,
        "proximal_cap_fraction": 0.12,
        "distal_cap_fraction": 0.12,
        "distal_connector_fixed": True,
    }
    part = {"id": 6, "mesh_linear": current.tolist(), "connector_cap_deformation": specification}
    prototype = {"id": 6, "mesh_linear": baseline.tolist()}
    baseline_export = baseline.T @ polar
    template = source @ baseline_export / scale
    affine = np.eye(4)
    affine[:3, :3] = np.linalg.solve(baseline_export, current.T @ polar)
    expected = (
        apply_midas_axis_deformation(
            trimesh.Trimesh(vertices=source @ current.T, process=False), specification
        ).vertices
        @ polar
        / scale
    )
    return template, affine, expected, connector_deformation_overlay(part, prototype, polar, scale)


@pytest.mark.parametrize("length,width", [(1.6, 1.25), (0.55, 0.6), (1.0, 1.4)])
def test_baked_points_match_full_compiler_with_rotated_frames(length, width):
    template, affine, expected, overlay = example(length, width)
    original = template.copy()
    actual = deform_template_points(template, overlay) @ affine[:3, :3]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1.0e-14)
    np.testing.assert_array_equal(template, original)


def test_caps_remain_rigid_and_distal_connector_moves_to_new_joint():
    template, _, _, overlay = example()
    mapping = np.asarray(overlay["canonical_from_template"])
    before = template @ mapping
    after = deform_template_points(template, overlay) @ mapping
    axis = np.asarray(overlay["connector_cap_deformation"]["longitudinal_axis"])
    coordinate = before @ axis
    np.testing.assert_allclose(after[coordinate < 0.12], before[coordinate < 0.12], atol=1e-14)
    np.testing.assert_allclose(
        after[coordinate > 0.88] - before[coordinate > 0.88],
        np.broadcast_to(0.6 * axis, after[coordinate > 0.88].shape),
        atol=1e-14,
    )


def test_identity_has_no_point_override_and_deformed_prototype_is_rejected():
    _, _, _, overlay = example()
    part = {
        "id": 6,
        "mesh_linear": np.eye(3).tolist(),
        "connector_cap_deformation": copy.deepcopy(overlay["connector_cap_deformation"]),
    }
    prototype = {"mesh_linear": np.eye(3).tolist()}
    part["connector_cap_deformation"]["target_length_canonical"] = 1.0
    part["connector_cap_deformation"]["width_scale"] = 1.0
    assert connector_deformation_overlay(part, prototype, np.eye(3), 1.0) is None
    prototype["connector_cap_deformation"] = overlay["connector_cap_deformation"]
    with pytest.raises(ValueError, match="undeformed prototype"):
        connector_deformation_overlay(part, prototype, np.eye(3), 1.0)


def usd_modules():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom, Vt

    return Gf, Usd, UsdGeom, Vt


def make_template(path, template_points):
    Gf, Usd, UsdGeom, Vt = usd_modules()
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Hand").GetPrim()
    stage.SetDefaultPrim(root)
    link = UsdGeom.Xform.Define(stage, "/Hand/part_06_index")
    link.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.3))
    for scope_name in ("collisions", "visuals"):
        scope = UsdGeom.Xform.Define(stage, f"{link.GetPath()}/{scope_name}")
        # Exercise geometry reference instances and non-identity nested xforms.
        geometry = UsdGeom.Xform.Define(stage, f"/Geometry_{scope_name}")
        nested = np.eye(4)
        nested[:3, :3] = Rotation.from_euler("xyz", [0.15, 0.2, -0.1]).as_matrix()
        nested[3, :3] = [0.01, -0.02, 0.03]
        child = UsdGeom.Xform.Define(stage, f"{geometry.GetPath()}/nested")
        child.AddTransformOp().Set(Gf.Matrix4d(*nested.reshape(-1)))
        local = (np.column_stack((template_points, np.ones(len(template_points)))) @ np.linalg.inv(nested))[:, :3]
        mesh = UsdGeom.Mesh.Define(stage, f"{child.GetPath()}/mesh")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray([(0, 0, 1)]))
        scope.GetPrim().GetReferences().AddInternalReference(geometry.GetPath())
        scope.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().Save()
    return stage


def mesh_points_in_link(stage, link_path, scope_name):
    _, Usd, UsdGeom, _ = usd_modules()
    link = stage.GetPrimAtPath(link_path)
    cache = UsdGeom.XformCache()
    world_to_link = np.asarray(cache.GetLocalToWorldTransform(link).GetInverse())
    meshes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{link_path}/{scope_name}"), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            transform = np.asarray(cache.GetLocalToWorldTransform(prim)) @ world_to_link
            meshes.append((np.column_stack((points, np.ones(len(points)))) @ transform)[:, :3])
    return np.concatenate(meshes)


def test_runtime_points_and_materialized_assets_agree_and_prototype_is_unchanged(tmp_path):
    Gf, Usd, UsdGeom, _ = usd_modules()
    template, affine, expected, overlay = example()
    template_path = tmp_path / "template.usda"
    make_template(template_path, template)
    original_bytes = template_path.read_bytes()

    stage = Usd.Stage.CreateInMemory()
    hand = stage.DefinePrim("/World/envs/env_0/Hand", "Xform")
    hand.GetReferences().AddReference(str(template_path))
    link_path = "/World/envs/env_0/Hand/part_06_index"
    for scope_name in ("collisions", "visuals"):
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(f"{link_path}/{scope_name}"))
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(*affine.reshape(-1)))
    assert deform_link_meshes(stage, link_path, overlay, affine) == 2
    assert deform_link_meshes(stage, link_path, overlay, affine) == 0
    for scope_name in ("collisions", "visuals"):
        np.testing.assert_allclose(mesh_points_in_link(stage, link_path, scope_name), expected, atol=2e-8, rtol=0)

    utility_path = ROOT / "temp/hocap_mano_replay/isaaclab/wuji_parametric_usd.py"
    spec = importlib.util.spec_from_file_location("wuji_parametric_usd_test", utility_path)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    candidate_path = tmp_path / "candidate.usda"
    candidate_path.write_bytes(original_bytes)
    utility.attach_parametric_collisions(
        candidate_path,
        template_path,
        ["part_06_index"],
        [affine.tolist()],
        [[0.4, 0.5, 0.6]],
        [],
        [],
        [overlay],
    )
    # Attaching an already-prepared candidate again must not stretch it twice.
    utility.attach_parametric_collisions(
        candidate_path,
        template_path,
        ["part_06_index"],
        [affine.tolist()],
        [[0.4, 0.5, 0.6]],
        [],
        [],
        [overlay],
    )
    materialized = Usd.Stage.Open(str(candidate_path))
    for scope_name in ("collisions", "visuals"):
        np.testing.assert_allclose(
            mesh_points_in_link(materialized, "/Hand/part_06_index", scope_name), expected, atol=2e-8, rtol=0
        )
    assert template_path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="prototype USD"):
        utility.attach_parametric_collisions(template_path, template_path, [], [], [], [], [])


def test_saved_best_candidate_all_editable_segments_match_full_compiler():
    """Regression for the thumb/index gaps observed in generation 89, proposal 31."""
    run = (
        ROOT
        / "artifacts/wuji_sac/wuji_hand_2_g04_1_refined_palm0_retarget_rollouts4_pop64_physxbs4_gen100_isolated_20260912_155810"
    )
    bank = ROOT / "artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70"
    paths = [
        run / "generation_089/hand_ir/hand_ir.json",
        bank / "prepared/compiled/compiled_hands.json",
        ROOT / "artifacts/hand_morphology/reference_graphs.json",
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("downloaded WUJI regression artifacts are unavailable")
    hand = json.loads(paths[0].read_text())["hands"][31]
    prototype = json.loads(paths[1].read_text())["hands"][0]
    source = next(h for h in json.loads(paths[2].read_text())["hands"] if h["hand_id"] == "wuji_hand_2")
    audit = source.get("canonicalization", source.get("direct_geometry_audit"))
    polar = np.asarray(audit["similarity_rotation"]).T @ np.diag([-1, 1, 1])
    scale = audit["similarity_scale"]
    checked = 0
    for part, baseline in zip(hand["parts"], prototype["parts"], strict=True):
        overlay = connector_deformation_overlay(part, baseline, polar, scale)
        if overlay is None:
            continue
        mesh_path = ROOT / "artifacts/hand_morphology" / part["source_mesh"]["file"]
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        original = np.asarray(mesh.vertices).copy()
        template = original @ np.asarray(baseline["mesh_linear"]).T @ polar / scale
        mesh.vertices = original @ np.asarray(part["mesh_linear"]).T
        expected = apply_midas_axis_deformation(mesh, part["connector_cap_deformation"]).vertices @ polar / scale
        affine = np.linalg.solve(
            np.asarray(baseline["mesh_linear"]).T @ polar, np.asarray(part["mesh_linear"]).T @ polar
        )
        np.testing.assert_allclose(deform_template_points(template, overlay) @ affine, expected, atol=1e-14, rtol=0)
        checked += 1
    assert checked == 15
