#!/usr/bin/env python3
"""Summarize MANO Isaac Lab single- and multi-GPU scaling artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


LABELS = (
    "single_e512_i1200",
    "single_e1024_i600",
    "single_e2048_i300",
    "single_e4096_i150",
    "ddp2_e512_i600",
    "ddp4_e256_i600",
    "ddp2_e1024_i300",
    "ddp4_e1024_i150",
)


def gpu_summary(path: Path) -> tuple[float, float, int]:
    utilities: list[float] = []
    memory: list[float] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                utilization = float(row[" utilization.gpu [%]"].replace("%", ""))
                used_mib = float(row[" memory.used [MiB]"].replace("MiB", ""))
            except (KeyError, ValueError):
                continue
            if used_mib >= 1000 and utilization > 0:
                utilities.append(utilization)
                memory.append(used_mib)
    return statistics.mean(utilities), statistics.median(utilities), int(max(memory))


def training_seconds(slurm_dir: Path, label: str) -> float | None:
    values: list[float] = []
    for path in slurm_dir.glob("*.out"):
        text = path.read_text(errors="ignore")
        if label not in text:
            continue
        values.extend(float(item) for item in re.findall(r"Training time: ([0-9.]+) seco", text))
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label in LABELS:
        result_dir = args.raw / label
        timing = json.loads((result_dir / "timing.json").read_text())
        config = json.loads((result_dir / "benchmark_config.json").read_text())
        best = json.loads(
            (args.evaluations / f"scaling_{label}_best" / "evaluation_diagnostics.json").read_text()
        )
        final = json.loads(
            (args.evaluations / f"scaling_{label}_final" / "evaluation_diagnostics.json").read_text()
        )
        util_mean, util_median, memory_peak = gpu_summary(result_dir / "gpu_samples.csv")
        train_seconds = training_seconds(args.raw / "slurm", label)
        rows.append(
            {
                "label": label,
                "gpus": config["world_size"],
                "envs_per_gpu": config["envs_per_rank"],
                "global_envs": config["global_envs"],
                "iterations": config["iterations"],
                "transitions": timing["transitions"],
                "wall_seconds": timing["elapsed_seconds"],
                "wall_samples_per_second": round(timing["samples_per_second"], 1),
                "training_seconds": train_seconds,
                "training_samples_per_second": (
                    round(timing["transitions"] / train_seconds, 1) if train_seconds else None
                ),
                "gpu_util_mean_percent": round(util_mean, 1),
                "gpu_util_median_percent": round(util_median, 1),
                "gpu_memory_peak_mib": memory_peak,
                "best_phase": best["final_phase"],
                "best_position_error_m": best["position_error_m"],
                "best_orientation_error_rad": best["rotation_error_rad"],
                "final_phase": final["final_phase"],
                "success": bool(best["success"] or final["success"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
