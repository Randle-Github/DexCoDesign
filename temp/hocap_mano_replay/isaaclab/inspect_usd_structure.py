#!/usr/bin/env python3
"""Print the physics-relevant structure of an imported hand USD."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("usd")
parser.add_argument("--flatten-output")
parser.add_argument("--json-output")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return [_plain(item) for item in value]
    except TypeError:
        return str(value)


def main() -> int:
    stage = Usd.Stage.Open(args.usd)
    if args.flatten_output:
        flattened = stage.Flatten()
        if not flattened.Export(args.flatten_output):
            raise RuntimeError(f"failed to export {args.flatten_output}")
        print(f"FLATTENED {args.flatten_output}", flush=True)
    print(f"DEFAULT {stage.GetDefaultPrim().GetPath()}", flush=True)
    records = []
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    interesting = (
        "localPos",
        "localRot",
        "axis",
        "points",
        "indices",
        "approximation",
        "body0",
        "body1",
    )
    for prim in stage.Traverse():
        attributes = [
            attribute.GetName()
            for attribute in prim.GetAttributes()
            if any(token in attribute.GetName() for token in interesting)
        ]
        relationships = [
            relationship.GetName() for relationship in prim.GetRelationships()
        ]
        if attributes or relationships or prim.GetTypeName() in {
            "Mesh",
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
            "PhysicsFixedJoint",
        }:
            print(
                f"PRIM {prim.GetPath()} type={prim.GetTypeName()} "
                f"attrs={attributes} rels={relationships}",
                flush=True,
            )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or "Joint" in prim.GetTypeName():
            record = {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
            for name in (
                "xformOp:translate",
                "xformOp:orient",
                "physics:localPos0",
                "physics:localPos1",
                "physics:localRot0",
                "physics:localRot1",
                "physics:axis",
            ):
                attribute = prim.GetAttribute(name)
                if attribute:
                    record[name] = _plain(attribute.Get())
            for name in ("physics:body0", "physics:body1"):
                relationship = prim.GetRelationship(name)
                if relationship:
                    record[name] = [str(item) for item in relationship.GetTargets()]
            records.append(record)
        elif prim.GetName() == "collisions":
            bounds = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
            records.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "local_bounds_min": _plain(bounds.GetMin()),
                    "local_bounds_max": _plain(bounds.GetMax()),
                }
            )
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as stream:
            json.dump(records, stream, indent=2)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
