#!/usr/bin/env python3
"""Batch-compile and exactly rescore top GPU WUJI morphology proposals."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from evaluate_wuji_morphology import vector_to_graph  # noqa: E402


def run(command: list[str], log: Path | None = None) -> int:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE if log is not None else None,
        stderr=subprocess.STDOUT if log is not None else None,
    )
    if log is not None:
        log.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--exact-ik-iterations", type=int, default=4)
    args = parser.parse_args()

    with np.load(args.proposal) as data:
        available = len(data["top_indices"])
        count = min(args.top_k, available)
        indices = data["top_indices"][:count].astype(np.int64)
        vectors = data["top_vectors"][:count].astype(np.float64)
        qpos = data["top_qpos"][:count].astype(np.float64)
        wrist_position = data["top_wrist_position"][:count].astype(np.float64)
        wrist_quaternion = data["top_wrist_quaternion_xyzw"][:count].astype(
            np.float64
        )
        frame_ids = data["frame_ids"].astype(np.int64)

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    graphs = []
    for rank, sample_index in enumerate(indices):
        hand_id = f"wuji_gpu_{rank:04d}_{sample_index:06d}"
        candidate = root / "candidates" / hand_id
        candidate.mkdir(parents=True, exist_ok=True)
        graph = vector_to_graph(vectors[rank], hand_id)
        (candidate / "reshape_graph.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8"
        )
        warm_start = candidate / "gpu_warm_start.npz"
        np.savez_compressed(
            warm_start,
            frame_ids=frame_ids,
            qpos=qpos[rank],
            wrist_position=wrist_position[rank],
            wrist_quaternion_xyzw=wrist_quaternion[rank],
        )
        rows.append(
            {
                "rank": rank,
                "sample_index": int(sample_index),
                "hand_id": hand_id,
                "vector": vectors[rank].tolist(),
                "candidate": str(candidate),
                "warm_start": str(warm_start),
            }
        )
        graphs.append(graph)

    batch_graph = root / "topk_graphs.json"
    batch_graph.write_text(
        json.dumps({"schema_version": 1, "hands": graphs}, indent=2) + "\n",
        encoding="utf-8",
    )
    compiled = root / "compiled_topk"
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/dexcodesign/compile_hand_graph_batch.py"),
            str(batch_graph),
            "--output-dir",
            str(compiled),
            "--seed",
            "0",
        ]
    )

    def export(row: dict[str, object]) -> dict[str, object]:
        candidate = Path(str(row["candidate"]))
        runtime = candidate / "runtime"
        code = run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "export_compiled_hand_urdf.py"),
                str(compiled / "compiled_hands.json"),
                "--compiled-hand-id",
                str(row["hand_id"]),
                "--output-root",
                str(runtime),
                "--hand-id",
                str(row["hand_id"]),
                "--display-name",
                f"WUJI GPU proposal {row['rank']}",
            ],
            candidate / "export.log",
        )
        row["runtime"] = str(runtime)
        row["export_returncode"] = code
        return row

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(export, rows))

    def evaluate(row: dict[str, object]) -> dict[str, object]:
        candidate = Path(str(row["candidate"]))
        if int(row["export_returncode"]) != 0:
            row["status"] = "export_failed"
            return row
        code = run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "evaluate_wuji_morphology.py"),
                "--output-dir",
                str(candidate),
                "--prepared-runtime",
                str(row["runtime"]),
                "--initial-trajectory",
                str(row["warm_start"]),
                "--iterations",
                str(args.exact_ik_iterations),
                "--vector",
                *(f"{value:.12g}" for value in row["vector"]),
            ],
            candidate / "evaluation.log",
        )
        evaluation = candidate / "evaluation.json"
        if code or not evaluation.is_file():
            row["status"] = "evaluation_failed"
            row["evaluation_returncode"] = code
            return row
        rollout = json.loads(evaluation.read_text(encoding="utf-8"))["rollout"]
        row.update(
            status="completed",
            objective=float(rollout["total_reward_return"]),
            pose_tracking_return=float(rollout["pose_tracking_return"]),
            contact_return=float(rollout["contact_return"]),
            phase=int(rollout["strict_final_phase"]),
            success=bool(rollout["strict_success"]),
        )
        return row

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(evaluate, rows))
    completed = [row for row in rows if row.get("status") == "completed"]
    completed.sort(key=lambda row: float(row["objective"]), reverse=True)
    summary = {
        "proposal": str(args.proposal.resolve()),
        "exact_top_k": count,
        "workers": args.workers,
        "exact_ik_iterations": args.exact_ik_iterations,
        "completed": len(completed),
        "best": completed[0] if completed else None,
        "results": rows,
    }
    (root / "exact_topk_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"WUJI_GPU_EXACT_TOPK_COMPLETE completed={len(completed)}/{count} "
        f"best={None if not completed else completed[0]['objective']}"
    )
    return 0 if len(completed) == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
