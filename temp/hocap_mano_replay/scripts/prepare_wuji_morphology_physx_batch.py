#!/usr/bin/env python3
"""Compile every GPU-retargeted WUJI candidate for batched PhysX rollout."""

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


def run(command: list[str], log: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("retarget_batch", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    source = args.retarget_batch.expanduser().resolve()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with np.load(source) as data:
        required = (
            "vectors",
            "qpos",
            "wrist_position_all",
            "wrist_quaternion_xyzw_all",
            "frame_ids",
            "joint_names",
            "qpos_ids",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"{source} misses {missing}; rerun gpu_wuji_retarget.py with "
                "--save-trajectories --skip-proxy"
            )
        available = len(data["vectors"])
        if args.offset < 0 or args.offset >= available:
            raise ValueError(
                f"--offset must be in [0, {available - 1}], got {args.offset}"
            )
        count = available - args.offset
        if args.limit is not None:
            count = min(count, args.limit)
        selection = slice(args.offset, args.offset + count)
        vectors = data["vectors"][selection].astype(np.float64)
        qpos = data["qpos"][selection].astype(np.float32)
        wrist_position = data["wrist_position_all"][selection].astype(np.float32)
        wrist_quaternion = data["wrist_quaternion_xyzw_all"][selection].astype(
            np.float32
        )
        frame_ids = data["frame_ids"].astype(np.int64)
        joint_names = data["joint_names"].astype(str)
        qpos_ids = data["qpos_ids"].astype(np.int64)

    candidate_ids = [
        f"wuji_physx_{index:06d}"
        for index in range(args.offset, args.offset + count)
    ]
    graphs = [
        vector_to_graph(vectors[index], candidate_ids[index])
        for index in range(count)
    ]
    graphs_path = root / "graphs.json"
    graphs_path.write_text(
        json.dumps({"schema_version": 1, "hands": graphs}, indent=2) + "\n",
        encoding="utf-8",
    )
    compiled_root = root / "compiled"
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/dexcodesign/compile_hand_graph_batch.py"),
            str(graphs_path),
            "--output-dir",
            str(compiled_root),
            "--seed",
            "0",
            "--workers",
            str(args.workers),
        ],
        root / "compile.log",
    )

    rows = []
    for index, hand_id in enumerate(candidate_ids):
        candidate = root / "candidates" / hand_id
        trajectory = root / "trajectories" / f"{hand_id}.npz"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory,
            frame_ids=frame_ids,
            qpos=qpos[index],
            wrist_position=wrist_position[index],
            wrist_quaternion_xyzw=wrist_quaternion[index],
            source_joint_names=joint_names,
            qpos_ids=qpos_ids,
        )
        rows.append(
            {
                "index": index,
                "hand_id": hand_id,
                "vector": vectors[index].tolist(),
                "candidate": candidate,
                "runtime": candidate / "runtime",
                "trajectory": trajectory,
                "prepared": candidate / "prepared",
                "usd": candidate / "asset" / "hand.usd",
            }
        )

    def export_and_prepare(row: dict[str, object]) -> None:
        candidate = Path(row["candidate"])
        run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "export_compiled_hand_urdf.py"),
                str(compiled_root / "compiled_hands.json"),
                "--compiled-hand-id",
                str(row["hand_id"]),
                "--output-root",
                str(row["runtime"]),
                "--hand-id",
                str(row["hand_id"]),
                "--display-name",
                str(row["hand_id"]),
                "--physics-only",
            ],
            candidate / "export.log",
        )
        run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "prepare_generated_hand_rl_reference.py"),
                str(row["runtime"]),
                str(row["trajectory"]),
                "--output-dir",
                str(row["prepared"]),
            ],
            candidate / "reference.log",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(export_and_prepare, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    manifest = {
        "schema_version": 1,
        "candidate_ids": candidate_ids,
        "vectors": vectors.tolist(),
        "hand_urdf_paths": [
            str(Path(row["prepared"]) / str(row["hand_id"]) / "hand_rl.urdf")
            for row in rows
        ],
        "hand_usd_paths": [str(row["usd"]) for row in rows],
        "reference_paths": [
            str(Path(row["prepared"]) / str(row["hand_id"]) / "reference.npz")
            for row in rows
        ],
        "all_candidates_require_physical_rollout": True,
        "proxy_used": False,
    }
    manifest_path = root / "physx_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"WUJI_PHYSX_BATCH_PREPARED candidates={count} manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
