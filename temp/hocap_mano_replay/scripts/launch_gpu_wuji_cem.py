#!/usr/bin/env python3
"""Submit a dependency-linked multi-generation GPU/exact WUJI CEM search."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SBATCH = "/opt/slurm/Ubuntu-20.04/24.11.0/bin/sbatch"
ISAACLAB = REPO_ROOT / "temp/hocap_mano_replay/isaaclab"


def submit(script: Path, exports: dict[str, str], dependency: str | None) -> str:
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    command = [SBATCH, "--parsable", f"--export={export_arg}"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip().split(";")[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = root / "cem_state.json"
    previous_exact_job = None
    jobs = []
    previous_summary = None
    for generation in range(args.generations):
        generation_root = root / f"generation_{generation:02d}"
        vectors = generation_root / "vectors.npy"
        proposal = generation_root / "gpu_proposal.npz"
        exact_root = generation_root / "exact_topk"
        sample_exports = {
            "STATE": str(state), "VECTORS": str(vectors),
            "POPULATION": str(args.population),
            "GENERATION_SEED": str(20260805),
        }
        if previous_summary is not None:
            sample_exports["EXACT_SUMMARY"] = str(previous_summary)
        sample_job = submit(
            ISAACLAB / "sample_gpu_wuji_cem.sbatch",
            sample_exports,
            previous_exact_job,
        )
        gpu_job = submit(
            ISAACLAB / "benchmark_gpu_wuji_retarget.sbatch",
            {
                "CANDIDATES": str(args.population), "ITERATIONS": "4",
                "CANDIDATE_CHUNK": "128", "TOP_K": str(args.top_k),
                "VECTORS": str(vectors), "OUTPUT": str(proposal),
            },
            sample_job,
        )
        exact_job = submit(
            ISAACLAB / "evaluate_gpu_wuji_topk.sbatch",
            {
                "PROPOSAL": str(proposal), "OUTPUT_ROOT": str(exact_root),
                "TOP_K": str(args.top_k), "WORKERS": str(args.workers),
                "EXACT_IK_ITERATIONS": "0",
            },
            gpu_job,
        )
        previous_exact_job = exact_job
        previous_summary = exact_root / "exact_topk_summary.json"
        jobs.append(
            {"generation": generation, "sample": sample_job,
             "gpu": gpu_job, "exact": exact_job,
             "summary": str(previous_summary)}
        )
    manifest = {
        "schema_version": 1,
        "generations": args.generations,
        "population": args.population,
        "top_k_exact": args.top_k,
        "objective": "exact MuJoCo C-error + binary pinch contact",
        "jobs": jobs,
    }
    (root / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
