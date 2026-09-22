#!/usr/bin/env python3
"""Create a corrected copy of a WUJI prototype bank; never edit the source bank.

Run with a Python environment providing pxr, numpy, trimesh and dexcodesign.
Preserves reference motions, finger meshes, joints and inertial parameters.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from dexcodesign.morphology.wuji_palm_collision import split_wuji_palm
from prepare_wuji_parametric_training_assets import _inverse_source_linear


PARTITION = "source_base_and_palm_v1"


def write_asset(source_usd, source_urdf, destination, pieces):
    destination.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.Open(str(source_usd))
    stage.SetEditTarget(stage.GetSessionLayer())
    # Instance proxies must be editable; only the temporary session is changed.
    for prim in list(stage.Traverse()):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    layer = stage.Flatten()
    usd = destination / "hand.usd"
    layer.Export(str(usd))
    stage = Usd.Stage.Open(str(usd))
    palms = [p for p in stage.Traverse() if p.GetName().endswith("__part_00_palm") or p.GetName() == "part_00_palm"]
    if len(palms) != 1:
        raise ValueError(f"Expected one palm in {source_usd}, got {len(palms)}")
    palm = palms[0]
    old = stage.GetPrimAtPath(palm.GetPath().AppendChild("collisions"))
    properties = [(a.GetName(), a.GetTypeName(), a.Get()) for a in old.GetAttributes()
                  if a.GetName().startswith(("physics:", "physx")) and a.Get() is not None]
    old_apis = [a for a in old.GetAppliedSchemas() if "Collision" in a]
    for scope in ("collisions", "visuals"):
        path = palm.GetPath().AppendChild(scope)
        if scope == "visuals" and not stage.GetPrimAtPath(path):
            continue
        stage.RemovePrim(path)
        UsdGeom.Xform.Define(stage, path)
        for label, mesh in pieces.items():
            name = "fixed_base" if label == "base" else "palm_shell"
            geometry = UsdGeom.Mesh.Define(stage, path.AppendChild(name))
            geometry.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
            geometry.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            geometry.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).reshape(-1).tolist())
            geometry.CreateSubdivisionSchemeAttr("none")
            geometry.CreateExtentAttr([Gf.Vec3f(*mesh.bounds[0]), Gf.Vec3f(*mesh.bounds[1])])
            geometry.CreateDisplayColorAttr([Gf.Vec3f(.68, .47, .74) if label == "base" else Gf.Vec3f(.23, .55, .79)])
            prim = geometry.GetPrim()
            if label == "base":
                prim.SetCustomDataByKey("dexcodesign:fixedBase", True)
            if scope == "collisions":
                prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(old_apis))
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                for name, typ, value in properties:
                    prim.CreateAttribute(name, typ).Set(value)
    stage.GetRootLayer().Save()
    robot = ET.parse(source_urdf)
    for element in robot.findall(".//mesh"):
        filename = Path(element.get("filename"))
        if not filename.is_absolute():
            element.set("filename", str((Path(source_urdf).parent / filename).resolve()))
    link = next(x for x in robot.findall("link") if x.get("name").endswith("part_00_palm"))
    has_visual = bool(link.findall("visual"))
    for kind in ("collision", "visual"):
        for element in link.findall(kind):
            link.remove(element)
    for label, mesh in pieces.items():
        meshfile = destination / f"{label}.obj"
        mesh.export(meshfile)
        for kind in (("collision", "visual") if has_visual else ("collision",)):
            element = ET.SubElement(link, kind, name="fixed_base" if label == "base" else "palm_shell")
            ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(element, "geometry"), "mesh", filename=str(meshfile))
    urdf = destination / "hand_rl.urdf"
    ET.indent(robot, space="  ")
    robot.write(urdf, encoding="utf-8", xml_declaration=True)
    return str(usd), str(urdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    bank, output = args.bank.resolve(), args.output_root.resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")
    manifest = json.loads((bank / "prepared/physx_batch_manifest.json").read_text())
    compiled_root = bank / "prepared/compiled"
    compiled = json.loads((compiled_root / "compiled_hands.json").read_text())
    hands = {h["hand_id"]: h for h in compiled["hands"]}
    scale, rotation = _inverse_source_linear("wuji_hand_2")
    polar = rotation.T @ np.diag([-1., 1., 1.])
    result = copy.deepcopy(manifest)
    output.mkdir(parents=True)
    for i, name in enumerate(manifest["candidate_ids"]):
        hand = hands[name]
        if hand["seed_source"] != "wuji_hand_2":
            raise ValueError("This migration is specific to WUJI")
        mesh = trimesh.load(compiled_root / hand["parts"][0]["compiled_mesh"]["file"], force="mesh", process=False)
        pieces = split_wuji_palm(mesh)
        for surface in pieces.values():
            surface.vertices = np.asarray(surface.vertices) @ polar / scale
            surface.faces = np.asarray(surface.faces)[:, ::-1]
        usd, urdf = write_asset(manifest["hand_usd_paths"][i], manifest["hand_urdf_paths"][i],
                                output / "prepared/candidates" / name / "asset", pieces)
        result["hand_usd_paths"][i], result["hand_urdf_paths"][i] = usd, urdf
        print(f"SPLIT_PALM {i+1}/{len(manifest['candidate_ids'])} {name}", flush=True)
    result["palm_collision_partition"] = PARTITION
    result["fixed_source_base"] = True
    result["source_bank"] = str(bank)
    (output / "prepared/physx_batch_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    # Immutable compiled graphs/trajectories can be shared with the source bank.
    (output / "prepared/compiled").symlink_to(compiled_root, target_is_directory=True)
    schema = json.loads((bank / "vectors.schema.json").read_text())
    schema["palm_collision_partition"] = PARTITION
    schema["fixed_source_base"] = True
    (output / "vectors.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    shutil.copy2(bank / "vectors.npy", output / "vectors.npy")
    print(f"CORRECTED_BANK={output}")


if __name__ == "__main__":
    main()
