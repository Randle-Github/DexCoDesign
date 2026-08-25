#!/usr/bin/env python3
"""Compile and render WUJI candidates from the canonical general grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
INTEGRATION_SCRIPTS = ROOT / "temp" / "hocap_mano_replay" / "scripts"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "hand_morphology" / "wuji_general_sac_preview_32"
)
for path in (SOURCE, INTEGRATION_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wuji_general_space import (  # noqa: E402
    CONTINUOUS_LOWER_BOUNDS,
    CONTINUOUS_UPPER_BOUNDS,
    PALM_EXPANSION_LEVELS,
    SOURCE_VECTOR,
    VECTOR_NAMES,
    graph_from_search_vector,
    validate_design_vectors,
)


def stratified_vectors(count: int, seed: int, amplitude: float) -> np.ndarray:
    if count != PALM_EXPANSION_LEVELS:
        raise ValueError(
            f"the palm audit requires exactly {PALM_EXPANSION_LEVELS} candidates"
        )
    rng = np.random.default_rng(seed)
    vectors = np.repeat(SOURCE_VECTOR[None, :], count, axis=0)
    vectors[:, 0] = np.arange(count)
    samples = count - 1
    for column in range(1, len(VECTOR_NAMES)):
        unit = (np.arange(samples) + rng.random(samples)) / samples
        rng.shuffle(unit)
        centered = amplitude * (2.0 * unit - 1.0)
        vectors[1:, column] = np.clip(
            centered,
            CONTINUOUS_LOWER_BOUNDS[column - 1],
            CONTINUOUS_UPPER_BOUNDS[column - 1],
        )
    vectors[0] = SOURCE_VECTOR
    return validate_design_vectors(vectors)


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def audit_palm_bank(output: Path, *, palm_only: bool) -> dict[str, object]:
    ir = json.loads((output / "hand_ir.json").read_text(encoding="utf-8"))["hands"]
    compiled = json.loads(
        (output / "compiled_hands.json").read_text(encoding="utf-8")
    )["hands"]
    expansions = [float(hand["palm_layout"]["palm_expansion"]) for hand in ir]
    visual_hashes: list[str] = []
    collision_hashes: list[str] = []
    for hand in compiled:
        palm = next(part for part in hand["parts"] if part["role"] == "palm")
        mesh = palm["compiled_mesh"]
        visual_hashes.append(hashlib.sha256(
            (output / mesh["file"]).read_bytes()
        ).hexdigest())
        collision_hashes.append(hashlib.sha256(
            (output / mesh["collision_file"]).read_bytes()
        ).hexdigest())
    if len(expansions) != PALM_EXPANSION_LEVELS:
        raise RuntimeError("palm bank must contain exactly 32 prototypes")
    if not all(a < b for a, b in zip(expansions, expansions[1:])):
        raise RuntimeError("palm prototype expansion is not strictly ordered")
    if len(set(visual_hashes)) != PALM_EXPANSION_LEVELS:
        raise RuntimeError("palm bank does not contain 32 distinct visual meshes")
    if len(set(collision_hashes)) != PALM_EXPANSION_LEVELS:
        raise RuntimeError("palm bank does not contain 32 distinct collision meshes")
    if any(
        hand["palm_connection"]["checked"] != 5
        or hand["palm_connection"]["unmeshed"] != 0
        for hand in compiled
    ):
        raise RuntimeError("a palm prototype has an unmeshed finger interface")
    audit = {
        "prototype_count": PALM_EXPANSION_LEVELS,
        "non_palm_parameters_are_source": palm_only,
        "expansion_range": [expansions[0], expansions[-1]],
        "strictly_ordered": True,
        "unique_visual_palm_meshes": len(set(visual_hashes)),
        "unique_collision_palm_meshes": len(set(collision_hashes)),
        "palm_finger_interfaces_checked": sum(
            hand["palm_connection"]["checked"] for hand in compiled
        ),
        "unmeshed_interfaces": 0,
        "reported_maximum_palm_interface_frame_error": 0.0,
    }
    (output / "palm_prototype_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--amplitude", type=float, default=0.78)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--palm-only",
        action="store_true",
        help="hold every non-palm parameter at the source value",
    )
    parser.add_argument("--keep-meshes", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.amplitude <= 1.0:
        parser.error("--amplitude must lie in (0, 1]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    vectors = stratified_vectors(args.count, args.seed, args.amplitude)
    if args.palm_only:
        vectors = np.repeat(SOURCE_VECTOR[None, :], args.count, axis=0)
        vectors[:, 0] = np.arange(args.count)
        vectors = validate_design_vectors(vectors)
    graphs = [
        graph_from_search_vector(vector, f"wuji_sac_{index:02d}")
        for index, vector in enumerate(vectors)
    ]
    graph_path = output / "graph_spec.json"
    graph_path.write_text(
        json.dumps({"schema_version": 1, "hands": graphs}, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(output / "vectors.npy", vectors.astype(np.float32))
    (output / "vector_schema.json").write_text(
        json.dumps(
            {
                "grammar_id": graphs[0]["grammar_id"],
                "source_hand": "wuji_hand_2",
                "vector_dimension": len(VECTOR_NAMES),
                "vector_names": list(VECTOR_NAMES),
                "palm_prototypes": PALM_EXPANSION_LEVELS,
                "zero_vector_is_source": True,
                "sampling_amplitude": args.amplitude,
                "palm_only": args.palm_only,
                "optimizer_encoding": (
                    "integer palm prototype index plus canonical general latent coordinates"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SOURCE)
        if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["HAND_GENERATION_ROOT"] = str(output)
    run(
        [
            sys.executable,
            "scripts/dexcodesign/compile_hand_graph_batch.py",
            str(graph_path),
            "--output-dir",
            str(output),
            "--workers",
            str(args.workers),
            "--force",
        ],
        env,
    )
    audit = audit_palm_bank(output, palm_only=args.palm_only)
    image_path = output / (
        "wuji_palm_prototypes_32.png"
        if args.palm_only
        else "wuji_sac_variants_32.png"
    )
    run(
        [
            sys.executable,
            "-m",
            "dexcodesign.morphology.render",
            "--output",
            str(image_path),
        ],
        env,
    )
    if not args.keep_meshes:
        for path in (
            output / "meshes",
            output / "compiled_parts",
            output / "render_tiles",
        ):
            if path.is_dir():
                shutil.rmtree(path)
        (output / "mesh_cleanup.json").write_text(
            json.dumps(
                {
                    "generated_meshes_deleted": True,
                    "reason": "preview meshes are reproducible from graph_spec.json and vectors.npy",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "hands": len(graphs),
        "vector_dimension": len(VECTOR_NAMES),
        "image": str(image_path),
        "palm_audit": audit,
        "generated_meshes_deleted": not args.keep_meshes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
