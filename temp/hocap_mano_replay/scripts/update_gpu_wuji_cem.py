#!/usr/bin/env python3
"""Sample/update GPU WUJI CEM using only exact physical rollout rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
VECTOR_NAMES = (
    "palm_expansion",
    "palm_scale_x",
    "palm_scale_z",
    "palm_yaw",
    *(f"{finger}_length" for finger in FINGERS),
    *(f"{finger}_radius" for finger in FINGERS),
)
LOWER_BOUNDS = np.asarray(
    [0.0, 0.90, 0.90, -0.12, *([0.82] * 5), *([0.85] * 5)],
    dtype=np.float64,
)
UPPER_BOUNDS = np.asarray(
    [0.35, 1.12, 1.12, 0.12, *([1.20] * 5), *([1.15] * 5)],
    dtype=np.float64,
)
SOURCE_VECTOR = np.asarray(
    [0.0, 1.0, 1.0, 0.0, *([1.0] * 10)], dtype=np.float64
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-vectors", type=Path, required=True)
    parser.add_argument("--exact-summary", type=Path)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.population < 2:
        parser.error("--population must be at least 2")

    span = UPPER_BOUNDS - LOWER_BOUNDS
    if args.state.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        mean = np.asarray(state["mean_normalized"], dtype=np.float64)
        sigma = np.asarray(state["sigma_normalized"], dtype=np.float64)
        generation = int(state["generation"])
    else:
        mean = (SOURCE_VECTOR - LOWER_BOUNDS) / span
        sigma = np.full(len(mean), 0.18, dtype=np.float64)
        generation = 0

    update = None
    if args.exact_summary is not None:
        summary = json.loads(args.exact_summary.read_text(encoding="utf-8"))
        completed = [
            row for row in summary["results"] if row.get("status") == "completed"
        ]
        completed.sort(key=lambda row: float(row["objective"]), reverse=True)
        if len(completed) < 2:
            raise ValueError("exact summary needs at least two completed candidates")
        elite_count = max(2, int(np.ceil(0.25 * len(completed))))
        elite = np.asarray(
            [row["vector"] for row in completed[:elite_count]], dtype=np.float64
        )
        elite_normalized = (elite - LOWER_BOUNDS) / span
        measured_mean = elite_normalized.mean(axis=0)
        measured_sigma = np.maximum(elite_normalized.std(axis=0), 0.04)
        mean = 0.25 * mean + 0.75 * measured_mean
        sigma = 0.25 * sigma + 0.75 * measured_sigma
        update = {
            "exact_summary": str(args.exact_summary.resolve()),
            "elite_count": elite_count,
            "best_exact_objective": float(completed[0]["objective"]),
            "best_exact_vector": completed[0]["vector"],
        }
        generation += 1

    rng = np.random.default_rng(args.seed + generation)
    normalized = np.clip(
        rng.normal(mean, sigma, size=(args.population, len(mean))), 0.0, 1.0
    )
    vectors = (LOWER_BOUNDS + normalized * span).astype(np.float32)
    vectors[0] = SOURCE_VECTOR.astype(np.float32)
    args.output_vectors.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_vectors, vectors)
    state = {
        "schema_version": 1,
        "generation": generation,
        "vector_names": list(VECTOR_NAMES),
        "mean_normalized": mean.tolist(),
        "sigma_normalized": sigma.tolist(),
        "population": args.population,
        "vectors": str(args.output_vectors.resolve()),
        "update_from_exact_physics": update,
        "proxy_reward_updates_cem": False,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(
        f"WUJI_GPU_CEM_SAMPLED generation={generation} "
        f"population={args.population} exact_update={update is not None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
