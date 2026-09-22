"""Keep WUJI's source mounting base separate from its editable palm shell.

The canonical palm concatenates two source URDF links. Use that provenance,
never a height cut or connected components (the STL triangles are unwelded).
"""
from functools import lru_cache
import json

import numpy as np
import trimesh

from .source_graph import (
    ARTIFACT_ROOT, DIRECT_ROOT, OUTPUT_GRAPHS, load_direct_urdf, load_visual_meshes,
)


@lru_cache(maxsize=1)
def source_palm_partition():
    graph = next(h for h in json.loads(OUTPUT_GRAPHS.read_text())["hands"]
                 if h["hand_id"] == "wuji_hand_2")
    part = graph["parts"][0]
    if part["member_links"] != ["r_base_link", "r_wrist"]:
        raise ValueError("WUJI palm source membership changed; rebuild its collision partition")
    audit = graph["canonicalization"]
    direct = load_direct_urdf(DIRECT_ROOT / "wuji_hand_2/right/hand.urdf")
    pieces = [load_visual_meshes(
        {"member_links": [name]}, direct, audit["similarity_scale"],
        np.asarray(audit["similarity_rotation"]), np.asarray(audit["similarity_translation"]),
        np.asarray(part["world_pos"]),
    ) for name in part["member_links"]]
    source = trimesh.load(ARTIFACT_ROOT / part["mesh"]["file"], force="mesh", process=False)
    rebuilt = trimesh.util.concatenate(pieces)
    if (source.vertices.shape != rebuilt.vertices.shape
            or not np.allclose(source.vertices, rebuilt.vertices, atol=1e-7, rtol=0)
            or not np.array_equal(source.faces, rebuilt.faces)):
        raise ValueError("WUJI reference palm lost source-link topology/provenance")
    return source, len(pieces[0].vertices), len(pieces[0].faces)


def split_wuji_palm(mesh):
    """Return fixed base and deformed palm in canonical coordinates.

    Requires a topology-preserving palm generator; fail rather than silently
    convexifying both bodies together when a remeshing generator is selected.
    """
    source, base_vertices, base_faces = source_palm_partition()
    if (mesh.vertices.shape != source.vertices.shape
            or mesh.faces.shape != source.faces.shape
            or not np.array_equal(np.sort(mesh.faces, axis=1), np.sort(source.faces, axis=1))):
        raise ValueError("WUJI split palm collisions require source-preserving mesh topology")
    fixed = mesh.copy()
    fixed.vertices[:base_vertices] = source.vertices[:base_vertices]
    return {
        "base": fixed.submesh([np.arange(base_faces)], append=True, repair=False),
        "palm": fixed.submesh([np.arange(base_faces, len(mesh.faces))], append=True, repair=False),
    }


def lock_wuji_base(mesh):
    split_wuji_palm(mesh)  # Validate provenance before restoring vertices.
    source, base_vertices, _ = source_palm_partition()
    result = mesh.copy()
    result.vertices[:base_vertices] = source.vertices[:base_vertices]
    return result


def preserve_fixed_base(stage, geometry_path, transform):
    """Cancel a parent morphology affine on the independently authored base."""
    from pxr import Gf, Usd, UsdGeom
    root = stage.GetPrimAtPath(geometry_path)
    if not root:
        return
    if root.IsInstance():
        root.SetInstanceable(False)
    for prim in Usd.PrimRange(root):
        # Explicit metadata avoids accidental matches on unrelated link names.
        if (prim.GetCustomDataByKey("dexcodesign:fixedBase")
                or prim.GetName() == "fixed_base"
                or prim.GetName().endswith("__fixed_base")):
            xf = UsdGeom.Xformable(prim)
            operations = [op for op in xf.GetOrderedXformOps()
                          if op.GetOpName() != "xformOp:transform:fixedBase"]
            attr = prim.GetAttribute("xformOp:transform:fixedBase")
            op = UsdGeom.XformOp(attr) if attr else xf.AddTransformOp(opSuffix="fixedBase")
            op.Set(Gf.Matrix4d(*np.linalg.inv(transform).reshape(-1).tolist()))
            xf.SetXformOpOrder([op, *operations])
