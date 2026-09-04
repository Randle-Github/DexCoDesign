#!/usr/bin/env python3
"""Compile many source-bound hand graphs in one preprocessed batch.

The single-hand fast compiler is ideal for interactive generation, but a
morphology optimizer must not start Python, reload source meshes, and rebuild
the grammar cache once per candidate.  This entry point accepts a ``hands``
array, invokes the graph generator once, then invokes the mesh compiler once.
No LLM is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import concurrent.futures
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
ARTIFACTS = ROOT / "artifacts" / "hand_morphology"
REFERENCE_GRAPH = ARTIFACTS / "reference_graphs.json"
BUNDLES = ARTIFACTS / "mechanism_bundles.json"


def run_module(module: str, *arguments: str, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel per-hand mesh compiler processes",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--palm-only",
        action="store_true",
        help="compile only candidate palms; finger collision uses USD prototypes",
    )
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

    payload = json.loads(args.graphs.read_text(encoding="utf-8"))
    hands = payload.get("hands")
    if not isinstance(hands, list) or not hands:
        raise ValueError("batch graph must contain a non-empty 'hands' array")
    hand_ids = []
    for hand in hands:
        if not isinstance(hand, dict):
            raise ValueError("every batch hand must be an object")
        for key in ("hand_id", "source_hand"):
            if not isinstance(hand.get(key), str) or not hand[key]:
                raise ValueError(f"every batch hand requires non-empty {key!r}")
        hand_ids.append(hand["hand_id"])
    if len(set(hand_ids)) != len(hand_ids):
        raise ValueError("batch hand_id values must be unique")

    output_dir = args.output_dir.resolve()
    canonical = {
        "schema_version": int(payload.get("schema_version", 1)),
        "hands": hands,
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256()
    digest.update(encoded)
    digest.update(str(args.seed).encode())
    digest.update(args.palm_generation_mode.encode())
    digest.update(str(bool(args.palm_only)).encode())
    for path in (REFERENCE_GRAPH, BUNDLES):
        if path.is_file():
            digest.update(path.read_bytes())
    cache_key = digest.hexdigest()
    compiled = output_dir / "compiled_hands.json"
    request = output_dir / "generation_request.json"
    if compiled.is_file() and request.is_file() and not args.force:
        previous = json.loads(request.read_text(encoding="utf-8"))
        if previous.get("cache_key") == cache_key:
            print(
                f"HAND_GRAPH_BATCH_CACHE_HIT count={len(hands)} "
                f"output={output_dir}"
            )
            return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "graphs.json"
    canonical_path.write_text(
        json.dumps(canonical, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    env["HAND_GRAPH_SPEC_PATH"] = str(canonical_path)
    env["HAND_GENERATION_ROOT"] = str(output_dir)
    env["HAND_GENERATION_SEED"] = str(args.seed)
    run_module("dexcodesign.morphology.generate", env=env)
    if args.workers <= 1 or len(hand_ids) == 1:
        run_module(
            "dexcodesign.morphology.mesh_compiler",
            *(["--palm-only"] if args.palm_only else []),
            "--palm-generation-mode",
            args.palm_generation_mode,
            env=env,
        )
    else:
        mesh_root = output_dir / "meshes"
        if mesh_root.is_dir():
            shutil.rmtree(mesh_root)
        partial_root = output_dir / "compiled_parts"
        partial_root.mkdir(parents=True, exist_ok=True)

        def compile_one(hand_id: str) -> Path:
            output = partial_root / f"{hand_id}.json"
            run_module(
                "dexcodesign.morphology.mesh_compiler",
                "--hand-id",
                hand_id,
                "--output",
                str(output),
                "--preserve-existing-meshes",
                *(["--palm-only"] if args.palm_only else []),
                "--palm-generation-mode",
                args.palm_generation_mode,
                env=env,
            )
            return output

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(hand_ids))
        ) as pool:
            partials = list(pool.map(compile_one, hand_ids))
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in partials]
        hands = [payload["hands"][0] for payload in payloads]
        sum_fields = (
            "parts",
            "meshed_parts",
            "geometryless_zero_length_frames",
            "faces",
            "attachment_roots_checked",
            "nonoverlapping_attachment_roots",
            "semantic_palm_finger_interfaces",
            "palm_interface_mount_lock_conflicts",
        )
        summary = {
            "hands": len(hands),
            **{
                field: sum(int(payload["summary"][field]) for payload in payloads)
                for field in sum_fields
            },
            "maximum_palm_interface_frame_error": max(
                float(payload["summary"]["maximum_palm_interface_frame_error"])
                for payload in payloads
            ),
        }
        compiled.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": payloads[0]["method"],
                    "palm_generation_mode": args.palm_generation_mode,
                    "hands": hands,
                    "summary": summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_key": cache_key,
                "hand_count": len(hands),
                "hand_ids": hand_ids,
                "graphs": str(canonical_path),
                "compiled": str(compiled),
                "llm_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"HAND_GRAPH_BATCH_COMPILED count={len(hands)} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
