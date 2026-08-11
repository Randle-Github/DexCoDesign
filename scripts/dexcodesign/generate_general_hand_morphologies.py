#!/usr/bin/env python3
"""Generate an all-source contact sheet with source-local general grammars."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
ARTIFACTS = ROOT / "artifacts" / "hand_morphology"
DEFAULT_OUTPUT = ARTIFACTS / "general_simulation_30"
sys.path.insert(0, str(SOURCE))

from dexcodesign.morphology.general_grammar import (  # noqa: E402
    FINGERS,
    build_schema,
    decode_vector,
    latin_hypercube_vectors,
)
from dexcodesign.morphology.generate import (  # noqa: E402
    BOUNDED_PALM_LAYOUT_SOURCES,
    PROTECTED_TRANSMISSION_SOURCES,
)


def run_module(module: str, env: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments], cwd=ROOT, env=env, check=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-meshes", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SOURCE) if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["HAND_GENERATION_ROOT"] = str(output)
    env["HAND_GENERATION_SEED"] = str(args.seed)
    if not (ARTIFACTS / "reference_graphs.json").is_file():
        run_module("dexcodesign.morphology.source_graph", env)
    run_module("dexcodesign.morphology.bundles", env)

    library = json.loads(
        (ARTIFACTS / "mechanism_bundles.json").read_text(encoding="utf-8")
    )
    bundles = {record["bundle_id"]: record for record in library["bundles"]}
    sources = []
    for source_id in library["palm_sources"]:
        # MANO is the human reference/retarget source, not a robot morphology
        # design seed. It remains available unchanged elsewhere in the project.
        if source_id == "mano":
            continue
        roles = [role for role in FINGERS if f"{source_id}:{role}" in bundles]
        if len(roles) >= 4:
            sources.append((source_id, roles))
    if not sources:
        raise ValueError("general grammar found no eligible source hands")

    quotient, remainder = divmod(args.count, len(sources))
    counts = [quotient + int(index < remainder) for index in range(len(sources))]
    design_records = []
    schemas = {}
    serial = 0
    for source_index, ((source_id, roles), count) in enumerate(zip(sources, counts, strict=True)):
        if count == 0:
            continue
        protected = source_id in PROTECTED_TRANSMISSION_SOURCES
        part_ids_by_finger = {
            role: tuple(bundles[f"{source_id}:{role}"]["source_part_ids"])
            for role in roles
        }
        schema = build_schema(
            source_id,
            part_ids_by_finger,
            palm_affine_editable=not protected,
            protected_finger_root=protected,
        )
        schemas[source_id] = schema
        vectors = latin_hypercube_vectors(
            count, schema, seed=args.seed + 97 * source_index
        )
        for local_index, vector in enumerate(vectors):
            decoded = decode_vector(vector, schema)
            if local_index == 0:
                layout_mode = "source_fixed"
            elif source_id in BOUNDED_PALM_LAYOUT_SOURCES:
                layout_mode = "anthropomorphic"
            else:
                layout_mode = ("anthropomorphic", "symmetric", "asymmetric")[
                    (source_index + local_index - 1) % 3
                ]
            palm = {
                "layout_mode": layout_mode,
                "prototype_bank_id": f"{source_id}:palm32",
                **decoded["palm"],
            }
            # House layouts retain their established automatic clearance
            # resolver. The ordered expansion prototype is the continuous
            # anthropomorphic backend and must not bypass that resolver.
            if layout_mode in {"symmetric", "asymmetric"}:
                palm.pop("expansion", None)
            design_records.append({
                "hand_id": f"general_v2_{serial:03d}_{source_id}",
                "source_hand": source_id,
                "palm": palm,
                "fingers": {**decoded["fingers"], "width_scales": decoded["width_scales"]},
                "general_morphology_vector": vector.tolist(),
                "general_morphology_vector_names": schema["vector_names"],
            })
            serial += 1

    graph_spec = output / "graph_spec.json"
    graph_spec.write_text(
        json.dumps({"hands": design_records}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_local_grammars.json").write_text(
        json.dumps({"schemas": schemas}, indent=2) + "\n", encoding="utf-8"
    )
    env["HAND_GRAPH_SPEC_PATH"] = str(graph_spec)
    run_module("dexcodesign.morphology.generate", env)
    try:
        run_module("dexcodesign.morphology.mesh_compiler", env)
        run_module("dexcodesign.morphology.render", env)
    finally:
        if not args.keep_meshes:
            for path in (output / "meshes", output / "render_tiles"):
                if path.exists():
                    shutil.rmtree(path)
            compiled = output / "compiled_hands.json"
            if compiled.exists():
                compiled.unlink()
    print(json.dumps({
        "generated_hands": len(design_records),
        "source_hands": len(schemas),
        "dimensions_by_source": {
            source_id: schema["vector_dimension"] for source_id, schema in schemas.items()
        },
        "persistent_mesh_cache_removed": not args.keep_meshes,
        "image": str(output / f"hands_{len(design_records)}.png"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
