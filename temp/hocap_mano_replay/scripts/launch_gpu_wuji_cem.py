#!/usr/bin/env python3
"""Launch one persistent, all-candidate Isaac/PhysX morphology search job."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SBATCH = "/opt/slurm/Ubuntu-20.04/24.11.0/bin/sbatch"
SEARCH_JOB = (
    REPO_ROOT
    / "temp/hocap_mano_replay/isaaclab/search_wuji_morphology_physx.sbatch"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--physics-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be positive")
    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.physics_batch_size < 1:
        parser.error("--physics-batch-size must be positive")

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    exports = {
        "OUTPUT_ROOT": str(root),
        "GENERATIONS": str(args.generations),
        "POPULATION": str(args.population),
        "PHYSICS_BATCH_SIZE": str(args.physics_batch_size),
        "WORKERS": str(args.workers),
        "GENERATION_SEED": str(args.seed),
    }
    export_arg = "ALL," + ",".join(
        f"{key}={value}" for key, value in exports.items()
    )
    job_id = subprocess.check_output(
        [SBATCH, "--parsable", f"--export={export_arg}", str(SEARCH_JOB)],
        cwd=REPO_ROOT,
        text=True,
    ).strip().split(";")[0]
    manifest = {
        "schema_version": 2,
        "job_id": job_id,
        "single_persistent_gpu_allocation": True,
        "generations": args.generations,
        "population": args.population,
        "physics_batch_size": args.physics_batch_size,
        "all_candidates_physically_evaluated": True,
        "proxy_used": False,
        "top_k_prefilter_used": False,
        "objective": "Isaac Lab C-error + binary pinch contact",
    }
    (root / "job.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
