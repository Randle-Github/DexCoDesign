#!/usr/bin/env python3
"""Audit canonical trajectories across datasets and emit one JSON report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from trajectory_schema import audit_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for root in args.roots:
        records.extend(audit_trajectory(path) for path in sorted(root.rglob("trajectory.npz")))
    datasets = Counter(record.get("dataset", "unknown") for record in records if record["ok"])
    report = {
        "schema": "dexcodesign.pose_dataset_audit.v1",
        "files": len(records), "valid": sum(record["ok"] for record in records),
        "invalid": sum(not record["ok"] for record in records),
        "datasets": dict(sorted(datasets.items())), "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("files", "valid", "invalid", "datasets")}))
    if report["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
