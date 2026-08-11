#!/usr/bin/env python3
"""Generate source-local general hands without altering the legacy path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
ARTIFACTS = ROOT / "artifacts" / "hand_morphology"
DEFAULT_OUTPUT = ARTIFACTS / "general_simulation"
sys.path.insert(0, str(SOURCE))

from dexcodesign.morphology.general_grammar import (  # noqa: E402
    FINGERS,
    build_schema,
    decode_vector,
    latin_hypercube_vectors,
)
from dexcodesign.morphology.generate import (  # noqa: E402
    PROTECTED_TRANSMISSION_SOURCES,
    editable_finger_segments,
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
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--source-hand", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-meshes", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SOURCE)
        if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["HAND_GENERATION_ROOT"] = str(output)
    env["HAND_GENERATION_SEED"] = str(args.seed)
    if not (ARTIFACTS / "reference_graphs.json").is_file():
        run_module("dexcodesign.morphology.source_graph", env)
    run_module("dexcodesign.morphology.bundles", env)

    library = json.loads((ARTIFACTS / "mechanism_bundles.json").read_text())
    source_payload = json.loads((ARTIFACTS / "reference_graphs.json").read_text())
    source_hands = {hand["hand_id"]: hand for hand in source_payload["hands"]}
    bundles = {record["bundle_id"]: record for record in library["bundles"]}
    requested = set(args.source_hand)
    sources = []
    for source_id in library["palm_sources"]:
        if source_id == "mano" or (requested and source_id not in requested):
            continue
        roles = [role for role in FINGERS if f"{source_id}:{role}" in bundles]
        if len(roles) >= 4:
            sources.append((source_id, roles))
    if requested - {source_id for source_id, _ in sources}:
        raise ValueError(f"unknown/ineligible sources: {sorted(requested - {s[0] for s in sources})}")
    if not sources:
        raise ValueError("no eligible robot-hand sources")

    quotient, remainder = divmod(args.count, len(sources))
    counts = [quotient + int(index < remainder) for index in range(len(sources))]
    schemas = {}
    designs = []
    serial = 0
    for source_index, ((source_id, roles), count) in enumerate(
        zip(sources, counts, strict=True)
    ):
        protected = source_id in PROTECTED_TRANSMISSION_SOURCES
        segment_ids = {}
        for role in roles:
            bundle = bundles[f"{source_id}:{role}"]
            segment_ids[role] = tuple(editable_finger_segments(
                source_hands[source_id], bundle
            ))
        schema = build_schema(
            source_id,
            segment_ids,
            palm_affine_editable=not protected,
        )
        schemas[source_id] = schema
        vectors = latin_hypercube_vectors(
            count, schema, seed=args.seed + 97 * source_index
        )
        for local_index, vector in enumerate(vectors):
            decoded = decode_vector(vector, schema)
            if local_index == 0:
                layout = "source_fixed"
            else:
                layout = "anthropomorphic"
            palm = {
                "layout_mode": layout,
                "prototype_bank_id": f"{source_id}:palm32",
                **decoded["palm"],
            }
            designs.append({
                "hand_id": f"general_v2_{serial:03d}_{source_id}",
                "source_hand": source_id,
                "palm": palm,
                "fingers": {
                    **decoded["fingers"],
                    "width_scales": decoded["width_scales"],
                },
                "general_morphology_vector": vector.tolist(),
                "general_morphology_vector_names": schema["vector_names"],
            })
            serial += 1

    graph_path = output / "graph_spec.json"
    graph_path.write_text(json.dumps({"hands": designs}, indent=2) + "\n")
    (output / "source_local_grammars.json").write_text(
        json.dumps({"schemas": schemas}, indent=2) + "\n"
    )
    env["HAND_GRAPH_SPEC_PATH"] = str(graph_path)
    run_module("dexcodesign.morphology.generate", env)
    if not args.skip_meshes:
        run_module("dexcodesign.morphology.mesh_compiler", env)
    print(json.dumps({
        "generated_hands": len(designs),
        "dimensions_by_source": {
            source_id: schema["vector_dimension"]
            for source_id, schema in schemas.items()
        },
        "meshes_compiled": not args.skip_meshes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
