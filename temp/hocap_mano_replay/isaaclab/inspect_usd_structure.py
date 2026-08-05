#!/usr/bin/env python3
"""Print the physics-relevant structure of an imported hand USD."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd  # noqa: E402


def main() -> int:
    stage = Usd.Stage.Open(args.usd)
    print(f"DEFAULT {stage.GetDefaultPrim().GetPath()}", flush=True)
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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
