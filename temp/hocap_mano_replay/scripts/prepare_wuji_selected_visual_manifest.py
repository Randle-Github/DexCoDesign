#!/usr/bin/env python3
"""Export one searched WUJI morphology with complete visual geometry.

The batched search stores an exact parametric collision manifest and a fixed
retarget reference, but intentionally does not materialize 4096 visual meshes.
This utility reconstructs only the selected vector after search and reuses the
stored fixed reference without performing retargeting again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from evaluate_wuji_morphology import vector_to_graph  # noqa: E402


def run(command: list[str], log: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("search_manifest", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.search_manifest.expanduser().resolve()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    count = len(manifest["candidate_ids"])
    if not 0 <= args.index < count:
        raise IndexError(f"index {args.index} outside [0, {count})")

    hand_id = str(manifest["candidate_ids"][args.index])
    vector = manifest["vectors"][args.index]
    graph = vector_to_graph(vector, hand_id)
    graphs_path = root / "graphs.json"
    graphs_path.write_text(
        json.dumps({"schema_version": 1, "hands": [graph]}, indent=2) + "\n",
        encoding="utf-8",
    )
    compiled_root = root / "compiled"
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/dexcodesign/compile_hand_graph_batch.py"),
            str(graphs_path),
            "--output-dir",
            str(compiled_root),
            "--seed",
            "0",
            "--workers",
            "1",
        ],
        root / "compile.log",
    )

    runtime = root / "runtime"
    run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "export_compiled_hand_urdf.py"),
            str(compiled_root / "compiled_hands.json"),
            "--compiled-hand-id",
            hand_id,
            "--output-root",
            str(runtime),
            "--hand-id",
            hand_id,
            "--display-name",
            hand_id,
        ],
        root / "export.log",
    )
    urdf = runtime / hand_id / "left" / "hand.urdf"
    usd = root / "asset" / "hand.usd"
    fixed_reference = Path(
        manifest.get("fixed_reference")
        or manifest["reference_paths"][args.index]
    ).expanduser().resolve()
    output = {
        "schema_version": 1,
        "candidate_ids": [hand_id],
        "vectors": [vector],
        "hand_urdf_paths": [str(urdf)],
        "hand_usd_paths": [str(usd)],
        "reference_paths": [str(fixed_reference)],
        "all_candidates_require_physical_rollout": True,
        "runtime_parametric_overlays": False,
        "fixed_reference": str(fixed_reference),
        "retarget_performed": False,
        "proxy_used": False,
        "complete_visual_geometry": True,
        "selected_search_index": args.index,
    }
    output_path = root / "physx_batch_manifest.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        "WUJI_SELECTED_VISUAL_MANIFEST_PREPARED "
        f"candidate={hand_id} index={args.index} manifest={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
