#!/usr/bin/env python3
"""Sample/update GPU WUJI CEM using only exact physical rollout rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wuji_morphology_space import (
    CONTINUOUS_LOWER_BOUNDS,
    CONTINUOUS_SOURCE_VECTOR,
    CONTINUOUS_UPPER_BOUNDS,
    PALM_EXPANSION_LEVELS,
    PALM_EXPANSION_MAX,
    PALM_EXPANSION_MIN,
    SOURCE_VECTOR,
    VECTOR_NAMES,
    sample_mixed_vectors,
    validate_design_vectors,
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

    continuous_span = CONTINUOUS_UPPER_BOUNDS - CONTINUOUS_LOWER_BOUNDS
    if args.state.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if int(state.get("schema_version", 0)) != 2:
            raise ValueError("legacy continuous-palm CEM state cannot be resumed")
        palm_probabilities = np.asarray(
            state["palm_expansion_probabilities"], dtype=np.float64
        )
        mean = np.asarray(state["continuous_mean_normalized"], dtype=np.float64)
        sigma = np.asarray(state["continuous_sigma_normalized"], dtype=np.float64)
        generation = int(state["generation"])
    else:
        palm_probabilities = np.full(
            PALM_EXPANSION_LEVELS, 1.0 / PALM_EXPANSION_LEVELS, dtype=np.float64
        )
        mean = (CONTINUOUS_SOURCE_VECTOR - CONTINUOUS_LOWER_BOUNDS) / continuous_span
        sigma = np.full(len(mean), 0.18, dtype=np.float64)
        generation = 0

    update = None
    if args.exact_summary is not None:
        summary = json.loads(args.exact_summary.read_text(encoding="utf-8"))
        if not summary.get("all_candidates_physically_evaluated", False):
            raise ValueError(
                "CEM updates require all_candidates_physically_evaluated=true"
            )
        if summary.get("proxy_used", True):
            raise ValueError("proxy scores must never update morphology CEM")
        if summary.get("top_k_prefilter_used", True):
            raise ValueError("top-k prefiltering is forbidden for morphology CEM")
        completed = list(summary["results"])
        expected = int(summary["candidate_count"])
        if len(completed) != expected or int(summary["completed"]) != expected:
            raise ValueError(
                f"physical evaluation incomplete: {len(completed)}/{expected}"
            )
        completed.sort(key=lambda row: float(row["total_reward"]), reverse=True)
        if len(completed) < 2:
            raise ValueError("exact summary needs at least two completed candidates")
        elite_count = max(2, int(np.ceil(0.25 * len(completed))))
        elite = validate_design_vectors(np.asarray(
            [row["vector"] for row in completed[:elite_count]], dtype=np.float64
        ))
        elite_normalized = (
            elite[:, 1:] - CONTINUOUS_LOWER_BOUNDS
        ) / continuous_span
        measured_mean = elite_normalized.mean(axis=0)
        measured_sigma = np.maximum(elite_normalized.std(axis=0), 0.04)
        mean = 0.25 * mean + 0.75 * measured_mean
        sigma = 0.25 * sigma + 0.75 * measured_sigma
        counts = np.bincount(
            elite[:, 0].astype(np.int64), minlength=PALM_EXPANSION_LEVELS
        ).astype(np.float64)
        # A small Dirichlet floor prevents an unobserved prototype from being
        # irreversibly removed after one generation.
        measured_probabilities = (counts + 0.5) / (
            counts.sum() + 0.5 * PALM_EXPANSION_LEVELS
        )
        palm_probabilities = (
            0.25 * palm_probabilities + 0.75 * measured_probabilities
        )
        update = {
            "exact_summary": str(args.exact_summary.resolve()),
            "elite_count": elite_count,
            "best_exact_objective": float(completed[0]["total_reward"]),
            "best_exact_vector": completed[0]["vector"],
        }
        generation += 1

    rng = np.random.default_rng(args.seed + generation)
    vectors = sample_mixed_vectors(
        rng, args.population, palm_probabilities, mean, sigma
    ).astype(np.float32)
    vectors[0] = SOURCE_VECTOR.astype(np.float32)
    args.output_vectors.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_vectors, vectors)
    state = {
        "schema_version": 2,
        "generation": generation,
        "vector_names": list(VECTOR_NAMES),
        "palm_expansion_levels": PALM_EXPANSION_LEVELS,
        "palm_expansion_minimum": PALM_EXPANSION_MIN,
        "palm_expansion_maximum": PALM_EXPANSION_MAX,
        "palm_expansion_probabilities": palm_probabilities.tolist(),
        "continuous_mean_normalized": mean.tolist(),
        "continuous_sigma_normalized": sigma.tolist(),
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
