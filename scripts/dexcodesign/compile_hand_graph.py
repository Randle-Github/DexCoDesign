#!/usr/bin/env python3
"""Deterministically compile one source-bound hand graph without an LLM.

The first invocation compiles meshes into a content-addressed artifact. An
identical graph/source request reuses that artifact without rerunning geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
ARTIFACTS = ROOT / "artifacts" / "hand_morphology"
REFERENCE_GRAPH = ARTIFACTS / "reference_graphs.json"
BUNDLES = ARTIFACTS / "mechanism_bundles.json"
GENERATOR_SOURCE = (
    SOURCE / "dexcodesign" / "morphology" / "generate.py"
)
COMPILER_SOURCE = (
    SOURCE / "dexcodesign" / "morphology" / "mesh_compiler.py"
)


def canonical_payload(path: Path) -> tuple[dict, bytes]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "hands" in payload:
        hands = payload["hands"]
        if not isinstance(hands, list) or len(hands) != 1:
            raise ValueError("the fast compiler accepts exactly one hand per graph file")
        hand = hands[0]
    else:
        hand = payload
    if not isinstance(hand, dict):
        raise ValueError("the graph specification must be a JSON object")
    for key in ("hand_id", "source_hand"):
        if not isinstance(hand.get(key), str) or not hand[key]:
            raise ValueError(f"graph specification requires non-empty {key!r}")
    encoded = json.dumps(hand, sort_keys=True, separators=(",", ":")).encode()
    return hand, encoded


def run_module(module: str, *arguments: str, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-preprocess", action="store_true")
    parser.add_argument(
        "--palm-generation-mode",
        default="hybrid_source_topology",
        choices=(
            "fixed_template",
            "attachment_hull",
            "parametric_2_5d",
            "template_deform",
            "house_hull",
            "hybrid_house",
            "source_topology_house",
            "hybrid_source_topology",
        ),
    )
    args = parser.parse_args()

    hand, encoded = canonical_payload(args.graph)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SOURCE)
        if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )

    if args.rebuild_preprocess or not REFERENCE_GRAPH.is_file():
        run_module("dexcodesign.morphology.source_graph", env=env)
    if args.rebuild_preprocess or not BUNDLES.is_file():
        run_module("dexcodesign.morphology.bundles", env=env)

    digest = hashlib.sha256()
    digest.update(encoded)
    digest.update(str(args.seed).encode())
    digest.update(args.palm_generation_mode.encode())
    for path in (REFERENCE_GRAPH, BUNDLES, GENERATOR_SOURCE, COMPILER_SOURCE):
        digest.update(path.read_bytes())
    cache_key = digest.hexdigest()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else ARTIFACTS
        / "generated_fast"
        / f"{hand['hand_id']}-{cache_key[:12]}"
    ).resolve()
    compiled = output_dir / "compiled_hands.json"
    request = output_dir / "generation_request.json"
    if compiled.is_file() and request.is_file() and not args.force:
        previous = json.loads(request.read_text(encoding="utf-8"))
        if previous.get("cache_key") == cache_key:
            print(f"HAND_GRAPH_CACHE_HIT hand_id={hand['hand_id']} output={output_dir}")
            return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_graph = output_dir / "graph.json"
    canonical_graph.write_text(
        json.dumps(hand, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env["HAND_GRAPH_SPEC_PATH"] = str(canonical_graph)
    env["HAND_GENERATION_ROOT"] = str(output_dir)
    env["HAND_GENERATION_SEED"] = str(args.seed)
    run_module("dexcodesign.morphology.generate", env=env)
    run_module(
        "dexcodesign.morphology.mesh_compiler",
        "--palm-generation-mode",
        args.palm_generation_mode,
        env=env,
    )
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_key": cache_key,
                "hand_id": hand["hand_id"],
                "source_hand": hand["source_hand"],
                "graph": str(canonical_graph),
                "compiled": str(compiled),
                "llm_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"HAND_GRAPH_COMPILED hand_id={hand['hand_id']} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
