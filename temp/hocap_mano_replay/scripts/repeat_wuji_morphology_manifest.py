#!/usr/bin/env python3
"""Repeat one exact morphology-batch row for parallel PhysX replay."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1024)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.expanduser().resolve().read_text())
    count = len(manifest["candidate_ids"])
    if not 0 <= args.index < count:
        raise IndexError(f"index {args.index} outside [0, {count})")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    repeated = {}
    for key, value in manifest.items():
        if isinstance(value, list) and len(value) == count:
            repeated[key] = [deepcopy(value[args.index]) for _ in range(args.repeats)]
        else:
            repeated[key] = deepcopy(value)
    source_id = manifest["candidate_ids"][args.index]
    repeated["candidate_ids"] = [
        f"{source_id}_replica_{replica:04d}" for replica in range(args.repeats)
    ]
    repeated["capture_source_candidate_index"] = args.index
    repeated["capture_replica_count"] = args.repeats

    output = args.output_manifest.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(repeated, indent=2) + "\n", encoding="utf-8")
    print(
        f"WUJI_MORPHOLOGY_MANIFEST_REPEATED source={source_id} "
        f"index={args.index} repeats={args.repeats} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
