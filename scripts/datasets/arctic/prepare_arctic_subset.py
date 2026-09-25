#!/usr/bin/env python3
"""Create a compact, balanced ARCTIC benchmark without image data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def axis_angle_to_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1.0e-8
    scale[~small] = np.sin(half[~small]) / angle[~small]
    scale[small] = 0.5 - angle[small] ** 2 / 48.0
    return np.concatenate([np.cos(half), axis_angle * scale], axis=-1).astype(np.float32)


def stable_key(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.as_posix()}".encode()).hexdigest()


def select(paths: list[Path], count: int, seed: int) -> list[Path]:
    by_object: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_object[path.name.split("_", 1)[0]].append(path)
    for name in by_object:
        by_object[name].sort(key=lambda path: ("_use_" not in path.name, stable_key(path, seed)))

    chosen = []
    names = sorted(by_object)
    cursor = 0
    while len(chosen) < count and any(by_object.values()):
        name = names[cursor % len(names)]
        if by_object[name]:
            chosen.append(by_object[name].pop(0))
        cursor += 1
    return chosen


def hand_payload(data: dict, side: str, indices: np.ndarray) -> dict[str, np.ndarray]:
    hand = data[side]
    shape = np.asarray(hand["shape"], dtype=np.float32)
    return {
        f"{side}_global_orient_axis_angle": np.asarray(hand["rot"], dtype=np.float32)[indices],
        f"{side}_pose_axis_angle": np.asarray(hand["pose"], dtype=np.float32)[indices],
        f"{side}_translation_m": np.asarray(hand["trans"], dtype=np.float32)[indices],
        f"{side}_shape": shape,
        f"{side}_fitting_error": np.asarray(hand["fitting_err"], dtype=np.float32)[indices],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--stride", type=int, default=3, help="ARCTIC is 30 Hz; stride 3 gives 10 Hz.")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    raw = args.root / "raw" / "raw_seqs"
    mano_paths = sorted(raw.glob("*/*.mano.npy"))
    paired = [path for path in mano_paths if Path(str(path).replace(".mano.npy", ".object.npy")).is_file()]
    if not paired:
        raise SystemExit(
            f"No paired MANO/object trajectories found under {raw}. "
            "Run download_arctic_minimal.py first."
        )
    selected = select(paired, min(args.count, len(paired)), args.seed)
    output_root = args.root / "benchmark_100"
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, mano_path in enumerate(selected):
        object_path = Path(str(mano_path).replace(".mano.npy", ".object.npy"))
        mano = np.load(mano_path, allow_pickle=True).item()
        object_state = np.asarray(np.load(object_path, allow_pickle=True), dtype=np.float32)
        frame_count = min(len(object_state), len(mano["left"]["rot"]), len(mano["right"]["rot"]))
        indices = np.arange(0, frame_count, args.stride, dtype=np.int32)
        if len(indices) < 2:
            continue
        name = object_path.name.split("_", 1)[0]
        sequence_id = f"{mano_path.parent.name}/{mano_path.name.removesuffix('.mano.npy')}"
        payload = {
            "frame_indices": indices,
            "object_joint_position_rad": object_state[indices, 0],
            "object_root_quaternion_wxyz": axis_angle_to_wxyz(object_state[indices, 1:4]),
            # ARCTIC object/Vicon translation is millimetres. MANO translation is metres.
            "object_root_position_m": object_state[indices, 4:7] * 0.001,
        }
        payload.update(hand_payload(mano, "left", indices))
        payload.update(hand_payload(mano, "right", indices))
        metadata = {
            "schema": "dexcodesign.arctic_sequence.v1",
            "sequence_id": sequence_id,
            "object_id": name,
            "source_rate_hz": 30,
            "output_rate_hz": 30.0 / args.stride,
            "frames": int(len(indices)),
            "articulation": {
                "joint_name": "articulation",
                "type": "revolute",
                "axis_parent": [0.0, 0.0, -1.0],
                "parent": "bottom",
                "child": "top",
            },
        }
        payload["metadata_json"] = np.asarray(json.dumps(metadata))
        destination = output_root / f"{index:03d}_{name}_{mano_path.parent.name}.npz"
        np.savez_compressed(destination, **payload)
        records.append({**metadata, "path": destination.relative_to(args.root).as_posix()})

    manifest = {
        "schema": "dexcodesign.arctic_benchmark.v1",
        "requested_count": args.count,
        "actual_count": len(records),
        "selection": "object-balanced round robin; use sequences precede grab sequences; deterministic hash within group",
        "contains_images": False,
        "records": records,
    }
    manifest_path = args.root / "manifests" / "benchmark_100.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ARCTIC_SUBSET_READY sequences={len(records)} manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
