#!/usr/bin/env python3
"""Generate and compile dedicated MiDas manufacturing-constrained hands."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
DEFAULT_OUTPUT = ROOT / "artifacts" / "hand_morphology" / "midas_constraints_50"

sys.path.insert(0, str(SOURCE))

from dexcodesign.morphology.midas_grammar import (  # noqa: E402
    GRAMMAR_ID,
    SOURCE_HAND_ID,
    VECTOR_NAMES,
    apply_to_hand,
    decode_vector,
    grammar_payload,
    latin_hypercube_vectors,
)


def run_module(module: str, env: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-meshes", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    vectors = latin_hypercube_vectors(args.count, args.seed)

    # First let the established source grammar copy MiDas topology, motor
    # bindings, limits and candidate ownership. Every generic edit is identity.
    base_specs = {
        "schema_version": 1,
        "hands": [
            {
                "hand_id": f"midas_constraint_{index:03d}",
                "source_hand": SOURCE_HAND_ID,
                "palm": {
                    "layout_mode": "source_fixed",
                    "expansion": 0.0,
                    "scale_x": 1.0,
                    "scale_z": 1.0,
                    "yaw": 0.0,
                },
                "fingers": {
                    "default_length_scale": 1.0,
                    "default_radius_scale": 1.0,
                },
            }
            for index in range(args.count)
        ],
    }
    base_path = output / "base_source_graphs.json"
    base_path.write_text(json.dumps(base_specs, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) if not env.get("PYTHONPATH") else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    env["HAND_GRAPH_SPEC_PATH"] = str(base_path)
    env["HAND_GENERATION_ROOT"] = str(output)
    env["HAND_GENERATION_SEED"] = str(args.seed)
    run_module("dexcodesign.morphology.generate", env)

    hand_ir_path = output / "hand_ir.json"
    hand_ir = json.loads(hand_ir_path.read_text(encoding="utf-8"))
    if len(hand_ir["hands"]) != args.count:
        raise RuntimeError("generic source-topology pass returned the wrong hand count")
    resolved_hands = []
    records = []
    for index, (base_hand, vector) in enumerate(zip(hand_ir["hands"], vectors, strict=True)):
        hand = apply_to_hand(base_hand, vector)
        resolved_hands.append(hand)
        records.append(
            {
                "index": index,
                "hand_id": hand["hand_id"],
                "latent_vector": vector.tolist(),
                "resolved_dimensions_mm": decode_vector(vector),
            }
        )
    hand_ir["hands"] = resolved_hands
    hand_ir["grammar_id"] = GRAMMAR_ID
    hand_ir["method"] = "dedicated MiDas constrained physical-dimension grammar"
    hand_ir_path.write_text(json.dumps(hand_ir, indent=2) + "\n", encoding="utf-8")
    (output / "grammar.json").write_text(
        json.dumps(grammar_payload(), indent=2) + "\n", encoding="utf-8"
    )
    (output / "designs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "grammar_id": GRAMMAR_ID,
                "seed": args.seed,
                "count": args.count,
                "vector_names": list(VECTOR_NAMES),
                "designs": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(output / "design_vectors.npy", vectors)
    (output / "generation_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "grammar_id": GRAMMAR_ID,
                "generated_hands": args.count,
                "vector_dimension": len(VECTOR_NAMES),
                "zero_vector_source_hand_id": resolved_hands[0]["hand_id"],
                "all_zero_source_exact": bool(np.array_equal(vectors[0], np.zeros(len(VECTOR_NAMES)))),
                "existing_generic_grammar_modified": False,
                "meshes_compiled": not args.skip_meshes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.skip_meshes:
        if args.workers <= 1:
            run_module(
                "dexcodesign.morphology.mesh_compiler",
                env,
                "--palm-generation-mode",
                "hybrid_source_topology",
            )
        else:
            partial_root = output / "compiled_parts"
            # Subset workers use --preserve-existing-meshes so they cannot
            # delete one another's output. Clear the previous generation once
            # in the parent process first; otherwise compiled.json silently
            # reuses stale meshes after the grammar/deformation code changes.
            mesh_root = output / "meshes"
            if mesh_root.exists():
                shutil.rmtree(mesh_root)
            if partial_root.exists():
                shutil.rmtree(partial_root)
            partial_root.mkdir(parents=True, exist_ok=True)

            def compile_one(hand_id: str) -> Path:
                partial = partial_root / f"{hand_id}.json"
                run_module(
                    "dexcodesign.morphology.mesh_compiler",
                    env,
                    "--hand-id",
                    hand_id,
                    "--output",
                    str(partial),
                    "--preserve-existing-meshes",
                    "--palm-generation-mode",
                    "hybrid_source_topology",
                )
                return partial

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.workers, args.count)
            ) as pool:
                partials = list(
                    pool.map(
                        compile_one,
                        [hand["hand_id"] for hand in resolved_hands],
                    )
                )
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in partials]
            compiled_hands = [payload["hands"][0] for payload in payloads]
            summaries = [payload["summary"] for payload in payloads]
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
            combined = {
                "schema_version": 1,
                "method": "dedicated MiDas constrained physical-dimension grammar",
                "palm_generation_mode": "hybrid_source_topology",
                "hands": compiled_hands,
                "summary": {
                    "hands": len(compiled_hands),
                    **{
                        field: sum(int(summary[field]) for summary in summaries)
                        for field in sum_fields
                    },
                    "maximum_palm_interface_frame_error": max(
                        float(summary["maximum_palm_interface_frame_error"])
                        for summary in summaries
                    ),
                },
            }
            (output / "compiled_hands.json").write_text(
                json.dumps(combined, indent=2) + "\n", encoding="utf-8"
            )
    print(
        json.dumps(
            {
                "grammar_id": GRAMMAR_ID,
                "count": args.count,
                "dimension": len(VECTOR_NAMES),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
