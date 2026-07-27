#!/usr/bin/env python3
"""Write one compact status file for an all-hand training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metadata(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    data = np.load(path)
    return json.loads(str(data["metadata_json"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-id", required=True)
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    success = metadata(args.success)
    best = metadata(args.best)
    selected_path = args.success if success is not None else args.best
    selected = success if success is not None else best
    if selected is None:
        raise FileNotFoundError(
            f"Neither success nor best rollout exists for {args.hand_id}"
        )
    result = {
        "hand_id": args.hand_id,
        "success": success is not None,
        "iterations_budget": args.iterations,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "selected_rollout": str(selected_path),
        "selected_rollout_metadata": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
