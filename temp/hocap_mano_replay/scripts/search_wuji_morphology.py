#!/usr/bin/env python3
"""CEM search over WUJI reshape vectors using physical rollout return."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
EVALUATOR = SCRIPT_ROOT / "evaluate_wuji_morphology.py"
sys.path.insert(0, str(SCRIPT_ROOT))
from evaluate_wuji_morphology import (  # noqa: E402
    LOWER_BOUNDS,
    SOURCE_VECTOR,
    UPPER_BOUNDS,
    VECTOR_NAMES,
)


def run_candidate(
    output_dir: str,
    vector: list[float],
    iterations: int,
) -> dict[str, object]:
    path = Path(output_dir)
    evaluation = path / "evaluation.json"
    if not evaluation.is_file():
        command = [
            sys.executable,
            str(EVALUATOR),
            "--output-dir",
            str(path),
            "--iterations",
            str(iterations),
            "--vector",
            *(f"{value:.12g}" for value in vector),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (path / "evaluation.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0:
            return {
                "status": "failed",
                "failure_reason": f"evaluator exit {completed.returncode}",
                "objective": -1.0e9,
                "vector": vector,
                "output_dir": str(path),
            }
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    rollout = payload["rollout"]
    return {
        "status": "completed",
        "objective": float(rollout["pose_tracking_return"]),
        "phase": int(rollout["strict_final_phase"]),
        "success": bool(rollout["strict_success"]),
        "contact_frame_fraction": float(rollout["contact_frame_fraction"]),
        "vector": vector,
        "output_dir": str(path),
    }


def draw_curve(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 1100, 700
    margin = (90, 55, 45, 85)
    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    values = [float(row["best_so_far"]) for row in rows]
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1.0
    left, top = margin[0], margin[1]
    right, bottom = width - margin[2], height - margin[3]
    draw.rectangle((left, top, right, bottom), outline=(60, 65, 75), width=2)
    points = []
    for index, value in enumerate(values):
        x = left + (right - left) * index / max(len(values) - 1, 1)
        y = bottom - (bottom - top) * (value - low) / (high - low)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(26, 112, 220), width=4)
    for point in points:
        draw.ellipse(
            (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
            fill=(26, 112, 220),
        )
    draw.text((left, 18), "WUJI morphology search: best pose-only C-error return", fill=(20, 25, 35), font=font)
    draw.text((left, bottom + 28), "evaluated morphology", fill=(20, 25, 35), font=font)
    draw.text((8, top), f"{high:.3f}", fill=(20, 25, 35), font=font)
    draw.text((8, bottom - 10), f"{low:.3f}", fill=(20, 25, 35), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def compose(source_video: Path, best_video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source_video), "-i", str(best_video),
            "-filter_complex",
            (
                "[0:v]drawtext=text='WUJI v2 SOURCE':x=24:y=22:fontsize=28:"
                "fontcolor=white:box=1:boxcolor=black@0.65[left];"
                "[1:v]drawtext=text='BEST MORPHOLOGY':x=24:y=22:fontsize=28:"
                "fontcolor=white:box=1:boxcolor=black@0.65[right];"
                "[left][right]hstack=inputs=2[out]"
            ),
            "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "19",
            "-pix_fmt", "yuv420p", str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.population < 4:
        parser.error("--population must be at least 4")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    span = UPPER_BOUNDS - LOWER_BOUNDS
    mean = (SOURCE_VECTOR - LOWER_BOUNDS) / span
    sigma = np.full(len(mean), 0.18, dtype=np.float64)
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    evaluation_index = 0

    # Evaluate the exact source-shaped generated graph as a stable anchor.
    populations = [mean[None, :]]
    for generation in range(args.generations):
        if generation == 0:
            normalized = populations[0]
        else:
            normalized = np.clip(
                rng.normal(mean, sigma, size=(args.population, len(mean))),
                0.0,
                1.0,
            )
        vectors = LOWER_BOUNDS + normalized * span
        jobs = []
        for local_index, vector in enumerate(vectors):
            candidate_dir = root / "candidates" / f"candidate_{evaluation_index:04d}"
            jobs.append((str(candidate_dir), vector.tolist(), args.iterations))
            evaluation_index += 1
        # Each evaluation is already isolated in its own subprocess. Threads
        # only orchestrate those subprocesses and avoid an unnecessary second
        # multiprocessing layer (and semaphore limits on managed runners).
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(run_candidate, *job) for job in jobs
            ]
            results = [future.result() for future in futures]
        if not results:
            raise RuntimeError("no morphology evaluations completed")
        for result in results:
            result["generation"] = generation
            if best is None or float(result["objective"]) > float(best["objective"]):
                best = result
            result["best_so_far"] = float(best["objective"])
            history.append(result)
            print(
                f"MORPH_EVAL generation={generation} "
                f"objective={result['objective']:.6f} "
                f"phase={result.get('phase', -1)}/445 "
                f"best={best['objective']:.6f}",
                flush=True,
            )
        completed = [row for row in results if row["status"] == "completed"]
        # Generation zero is the exact identity anchor, not a sampled
        # population; do not collapse the search variance around one point.
        if completed and generation > 0:
            completed.sort(key=lambda row: float(row["objective"]), reverse=True)
            elite_count = max(2, int(np.ceil(0.25 * len(completed))))
            elite_vectors = np.asarray([row["vector"] for row in completed[:elite_count]])
            elite_normalized = (elite_vectors - LOWER_BOUNDS) / span
            new_mean = elite_normalized.mean(axis=0)
            new_sigma = np.maximum(elite_normalized.std(axis=0), 0.04)
            mean = 0.25 * mean + 0.75 * new_mean
            sigma = 0.25 * sigma + 0.75 * new_sigma

    assert best is not None
    history_path = root / "search_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    curve_rows = [
        {
            "evaluation": index,
            "objective": row["objective"],
            "best_so_far": row["best_so_far"],
            "phase": row.get("phase", -1),
        }
        for index, row in enumerate(history)
    ]
    with (root / "best_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)
    draw_curve(curve_rows, root / "best_curve.png")

    source_dir = root / "source"
    subprocess.run(
        [sys.executable, str(EVALUATOR), "--source", "--output-dir", str(source_dir),
         "--iterations", str(args.iterations), "--render-video", "--reuse-ik"],
        cwd=REPO_ROOT,
        check=True,
    )
    best_dir = Path(str(best["output_dir"]))
    subprocess.run(
        [sys.executable, str(EVALUATOR), "--output-dir", str(best_dir),
         "--iterations", str(args.iterations), "--render-video", "--reuse-ik",
         "--vector", *(f"{value:.12g}" for value in best["vector"])],
        cwd=REPO_ROOT,
        check=True,
    )
    source_video = source_dir / "rollout" / "wuji_source_physical_rollout.mp4"
    best_hand_id = best_dir.name.replace("-", "_")
    best_video = best_dir / "rollout" / f"{best_hand_id}_physical_rollout.mp4"
    combined = root / "wuji_source_vs_best.mp4"
    compose(source_video, best_video, combined)
    source_evaluation = json.loads(
        (source_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    best_evaluation = json.loads(
        (best_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    summary = {
        "algorithm": "cross_entropy_method",
        "rl_used": False,
        "generations": args.generations,
        "population": args.population,
        "evaluations": len(history),
        "reshape_vector_names": list(VECTOR_NAMES),
        "best": best,
        "source_rollout": source_evaluation["rollout"],
        "best_rollout": best_evaluation["rollout"],
        "source_video": str(source_video),
        "best_video": str(best_video),
        "combined_video": str(combined),
        "best_curve": str(root / "best_curve.png"),
    }
    (root / "search_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
