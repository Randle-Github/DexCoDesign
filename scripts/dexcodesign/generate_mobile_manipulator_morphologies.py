#!/usr/bin/env python3
"""Run the production mobile-manipulator morphology pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"


def run_module(module: str, *arguments: str, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--rebuild-preprocess",
        action="store_true",
        help="rebuild source graphs, grammar, and content-addressed mesh variants",
    )
    parser.add_argument(
        "--fast-search",
        action="store_true",
        help="reuse the validated cache and skip mesh-heavy independent validation",
    )
    args = parser.parse_args()
    if args.fast_search and args.render:
        raise ValueError("--fast-search cannot be combined with --render")
    env = os.environ.copy()
    env["MOBILE_MORPHOLOGY_COUNT"] = str(args.count)
    env["PYTHONPATH"] = (
        str(SOURCE)
        if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )
    cache_files = (
        ROOT / "artifacts/mobile_manipulator_morphology/reference_graphs.json",
        ROOT / "artifacts/mobile_manipulator_morphology/grammar.json",
        ROOT
        / "artifacts/mobile_manipulator_morphology/precomputed_link_variants.json",
    )
    if args.rebuild_preprocess or not all(path.is_file() for path in cache_files):
        run_module("dexcodesign.mobile_morphology.source_graph", env=env)
        run_module("dexcodesign.mobile_morphology.grammar", env=env)
        run_module("dexcodesign.mobile_morphology.preprocess", env=env)
    else:
        print("reusing precomputed mobile morphology graph/grammar/mesh cache")
    run_module(
        "dexcodesign.mobile_morphology.generate",
        "--seed",
        str(args.seed),
        "--count",
        str(args.count),
        env=env,
    )
    run_module("dexcodesign.mobile_morphology.mesh_compiler", env=env)
    if not args.fast_search:
        run_module("dexcodesign.mobile_morphology.validate", env=env)
    if args.render:
        if args.count != 32:
            raise ValueError("--render currently requires --count 32")
        run_module("dexcodesign.mobile_morphology.render", env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
