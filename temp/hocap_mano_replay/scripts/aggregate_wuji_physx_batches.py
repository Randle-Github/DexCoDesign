#!/usr/bin/env python3
"""Aggregate disjoint all-physical PhysX batches into one CEM result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    rows = []
    conversion = initialization = rollout = 0.0
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend") != "isaaclab_physx_gpu":
            raise ValueError(f"{path}: not an Isaac/PhysX result")
        if not payload.get("all_candidates_physically_evaluated", False):
            raise ValueError(f"{path}: batch is not fully physically evaluated")
        if payload.get("proxy_used", True) or payload.get(
            "top_k_prefilter_used", True
        ):
            raise ValueError(f"{path}: proxy/top-k results are forbidden")
        if payload["completed"] != payload["candidate_count"]:
            raise ValueError(f"{path}: incomplete batch")
        rows.extend(payload["results"])
        conversion += float(payload["usd_conversion_seconds"])
        initialization += float(payload["initialization_seconds"])
        rollout += float(payload["rollout_seconds"])

    by_id = {row["candidate_id"]: row for row in rows}
    if len(rows) != args.expected or len(by_id) != args.expected:
        raise ValueError(
            f"expected {args.expected} unique physical results, got "
            f"{len(rows)} rows/{len(by_id)} ids"
        )
    rows.sort(key=lambda row: float(row["total_reward"]), reverse=True)
    result = {
        "schema_version": 2,
        "backend": "isaaclab_physx_gpu_chunked",
        "all_candidates_physically_evaluated": True,
        "proxy_used": False,
        "top_k_prefilter_used": False,
        "reward": "RL C-error + binary pinch contact",
        "candidate_count": args.expected,
        "completed": args.expected,
        "batch_count": len(args.inputs),
        "usd_conversion_seconds": conversion,
        "initialization_seconds": initialization,
        "rollout_seconds": rollout,
        "best": rows[0],
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "WUJI_PHYSX_ALL_BATCHES_COMPLETE "
        f"completed={args.expected}/{args.expected} "
        f"batches={len(args.inputs)} "
        f"best_reward={rows[0]['total_reward']:.9f} "
        f"best_phase={rows[0]['phase']}/445",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
