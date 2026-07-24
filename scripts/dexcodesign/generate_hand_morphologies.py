#!/usr/bin/env python3
"""Run the production hand-morphology pipeline.

Stages:
1. normalized direct/mimic URDF assets (optional rebuild);
2. canonical rigid-link reference graphs (cached preprocessing);
3. source-local motor/link grammar;
4. 100 graph variants;
5. source-topology palm and rigid-link mesh compilation;
6. optional MuJoCo contact sheet.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
ARTIFACTS = ROOT / "artifacts" / "hand_morphology"


def run_module(module: str, *arguments: str, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--rebuild-direct-motor",
        action="store_true",
        help="recreate assets/robot_hands/direct_motor from the source registry",
    )
    parser.add_argument(
        "--rebuild-reference",
        action="store_true",
        help="re-merge source rigid links instead of using cached preprocessing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-hand mesh manifests after an interrupted run",
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SOURCE)
        if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["HAND_GENERATION_SEED"] = str(args.seed)

    if args.rebuild_direct_motor:
        subprocess.run(
            [sys.executable, "scripts/assets/build_direct_motor_hands.py"],
            cwd=ROOT,
            env=env,
            check=True,
        )

    reference_graph = ARTIFACTS / "reference_graphs.json"
    if args.rebuild_reference or not reference_graph.is_file():
        run_module("dexcodesign.morphology.source_graph", env=env)
    run_module("dexcodesign.morphology.bundles", env=env)
    run_module("dexcodesign.morphology.generate", env=env)
    compiler_args = ("--preserve-existing-meshes",) if args.resume else ()
    run_module("dexcodesign.morphology.mesh_compiler", *compiler_args, env=env)
    if args.render:
        run_module("dexcodesign.morphology.render", env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
