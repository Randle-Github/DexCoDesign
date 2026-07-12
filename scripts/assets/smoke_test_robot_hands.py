#!/usr/bin/env python3
"""Convert/open every registered hand with Isaac Lab and inspect the USD stage.

Run this on a supported NVIDIA Isaac Sim machine:

    ./isaaclab.sh -p scripts/assets/smoke_test_robot_hands.py --headless
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Isaac Lab robot-hand asset smoke test")
parser.add_argument("--hand", action="append", default=[], help="Only test the selected registry hand ID")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from pxr import Usd, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import (  # noqa: E402
    MjcfConverter,
    MjcfConverterCfg,
    UrdfConverter,
    UrdfConverterCfg,
)


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "robot_hands"
CACHE = Path("/tmp/dexcodesign_robot_hand_smoke")


def converted_usd(hand_id: str, side: str, entry: dict[str, object]) -> Path:
    source = (ASSETS / str(entry["path"])).resolve()
    output_dir = CACHE / hand_id / side
    output_dir.mkdir(parents=True, exist_ok=True)
    if entry["format"] == "usd":
        return source
    if entry["format"] == "urdf":
        config = UrdfConverterCfg(
            asset_path=str(source),
            usd_dir=str(output_dir),
            usd_file_name="hand.usd",
            force_usd_conversion=True,
            fix_base=True,
            merge_fixed_joints=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
        return Path(UrdfConverter(config).usd_path)
    if entry["format"] == "mjcf":
        config = MjcfConverterCfg(
            asset_path=str(source),
            usd_dir=str(output_dir),
            usd_file_name="hand.usd",
            force_usd_conversion=True,
            fix_base=True,
        )
        return Path(MjcfConverter(config).usd_path)
    raise ValueError(f"Unsupported format: {entry['format']}")


def inspect_stage(path: Path) -> tuple[int, int, int]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {path}")
    prims = list(stage.Traverse())
    if not prims:
        raise RuntimeError(f"USD stage contains no prims: {path}")
    rigid_bodies = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims)
    joints = sum(prim.IsA(UsdPhysics.Joint) for prim in prims)
    articulations = sum(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in prims)
    if rigid_bodies == 0:
        raise RuntimeError(f"USD stage contains no rigid bodies: {path}")
    if joints == 0:
        raise RuntimeError(f"USD stage contains no physics joints: {path}")
    if articulations == 0:
        raise RuntimeError(f"USD stage contains no articulation root: {path}")
    return rigid_bodies, joints, articulations


def main() -> int:
    if CACHE.exists():
        shutil.rmtree(CACHE)
    registry = json.loads((ASSETS / "registry.json").read_text(encoding="utf-8"))
    selected = set(args.hand)
    failures: list[str] = []
    tested = 0
    for hand_id, metadata in registry["hands"].items():
        if selected and hand_id not in selected:
            continue
        for side in ("left", "right"):
            try:
                usd_path = converted_usd(hand_id, side, metadata["entries"][side])
                rigid_bodies, joints, articulations = inspect_stage(usd_path)
                print(
                    f"PASS {hand_id:22} {side:5} "
                    f"rigid_bodies={rigid_bodies} joints={joints} articulations={articulations}"
                )
                tested += 1
            except Exception as error:  # Isaac importers expose several backend-specific exception types.
                failures.append(f"{hand_id}/{side}: {type(error).__name__}: {error}")
    if failures:
        print("\nFAILURES", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"\nIsaac Lab opened {tested} left/right hand assets successfully.")
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
