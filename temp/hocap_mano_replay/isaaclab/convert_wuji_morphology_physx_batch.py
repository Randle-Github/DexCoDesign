#!/usr/bin/env python3
"""Convert a morphology batch to USD inside one Isaac Sim process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.manifest = args_cli.manifest.expanduser().resolve()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main() -> None:
    manifest = json.loads(args_cli.manifest.read_text(encoding="utf-8"))
    urdfs = [Path(value) for value in manifest["hand_urdf_paths"]]
    usds = [Path(value) for value in manifest["hand_usd_paths"]]
    if len(urdfs) != len(usds):
        raise ValueError("hand_urdf_paths and hand_usd_paths have different sizes")
    start = time.perf_counter()
    for index, (urdf, usd) in enumerate(zip(urdfs, usds, strict=True)):
        usd.parent.mkdir(parents=True, exist_ok=True)
        cfg = UrdfConverterCfg(
            asset_path=str(urdf.resolve()),
            usd_dir=str(usd.parent.resolve()),
            usd_file_name=usd.name,
            fix_base=True,
            merge_fixed_joints=False,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=300.0,
                    damping=34.6410162,
                ),
                target_type="position",
            ),
        )
        converted = Path(UrdfConverter(cfg).usd_path)
        if not converted.is_file():
            raise RuntimeError(f"URDF conversion did not create {converted}")
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(urdfs):
            print(
                f"WUJI_PHYSX_USD_PROGRESS completed={index + 1}/{len(urdfs)}",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    manifest["usd_conversion_seconds"] = elapsed
    manifest["usd_candidates_per_second"] = len(urdfs) / max(elapsed, 1.0e-9)
    args_cli.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"WUJI_PHYSX_USD_COMPLETE candidates={len(urdfs)} "
        f"seconds={elapsed:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
