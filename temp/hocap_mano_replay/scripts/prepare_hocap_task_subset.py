#!/usr/bin/env python3
"""Extract one manifest-defined HO-Cap task from the official small archives."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = EXPERIMENT_ROOT / "data" / "raw"
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "data" / "benchmark_tasks.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data" / "tasks"


def quaternion_step_angle(quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def automatic_window(object_pose: np.ndarray, frames: int) -> int:
    """Choose a manipulation-rich window with approach/release context."""
    if len(object_pose) <= frames:
        return 0
    translation_step = np.linalg.norm(
        np.diff(object_pose[:, 4:7].astype(np.float64), axis=0), axis=1
    )
    rotation_step = quaternion_step_angle(object_pose[:, :4])
    step_score = translation_step + 0.03 * rotation_step
    context = min(45, max(0, frames // 10))
    core_steps = max(1, frames - 2 * context - 1)
    rolling = np.convolve(step_score, np.ones(core_steps), mode="valid")
    core_start = int(np.argmax(rolling))
    return int(np.clip(core_start - context, 0, len(object_pose) - frames))


def extract_member(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        with zip_file.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels-full", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task = next(
        (entry for entry in manifest["tasks"] if entry["task_id"] == args.task_id),
        None,
    )
    if task is None:
        raise KeyError(f"Unknown task_id {args.task_id!r}")
    target_frames = int(manifest["target_frames"])
    sequence = str(task["sequence"])
    hand_slot = int(task["hand_slot"])
    object_slot = int(task["object_slot"])
    object_id = str(task["object_id"])
    task_root = args.output_root / args.task_id
    task_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(RAW_ROOT / "poses.zip") as archive:
        mano_all = np.load(io.BytesIO(archive.read(f"{sequence}/poses_m.npy")))
        object_all = np.load(io.BytesIO(archive.read(f"{sequence}/poses_o.npy")))
    mano_pose = np.asarray(mano_all[hand_slot], dtype=np.float32)
    object_pose = np.asarray(object_all[object_slot], dtype=np.float32)
    if len(mano_pose) != len(object_pose):
        raise RuntimeError(f"{args.task_id}: MANO/object frame mismatch")
    configured_start = task.get("window_start", "auto")
    start = (
        automatic_window(object_pose, target_frames)
        if configured_start == "auto"
        else int(configured_start)
    )
    stop = min(start + target_frames, len(object_pose))
    if stop - start != target_frames:
        raise RuntimeError(
            f"{args.task_id}: requested {target_frames} frames, got {stop - start}"
        )
    mano_pose = mano_pose[start:stop]
    object_pose = object_pose[start:stop]
    np.save(task_root / "mano_pose_left.npy", mano_pose)
    np.save(task_root / f"object_pose_{object_id}.npy", object_pose)

    model_root = args.output_root / "models" / object_id
    for filename in (
        "cleaned_mesh_2000.obj",
        "cleaned_mesh_10000.obj",
        "textured_mesh.obj",
        "textured_mesh.mtl",
        "textured_mesh_0.png",
    ):
        extract_member(
            RAW_ROOT / "models.zip",
            f"models/{object_id}/{filename}",
            model_root / filename,
        )
    subject = sequence.split("/")[0]
    extract_member(
        RAW_ROOT / "calibration.zip",
        f"calibration/mano/{subject}.yaml",
        args.output_root / "calibration" / "mano" / f"{subject}.yaml",
    )
    extract_member(
        RAW_ROOT / "calibration.zip",
        "calibration/extrinsics/extrinsics_20231014.yaml",
        args.output_root
        / "calibration"
        / "extrinsics"
        / "extrinsics_20231014.yaml",
    )

    if args.labels_full is not None:
        labels = np.load(args.labels_full)
        if len(labels) != len(mano_all[hand_slot]):
            raise RuntimeError(
                f"{args.task_id}: labels have {len(labels)} frames, expected "
                f"{len(mano_all[hand_slot])}"
            )
        np.save(task_root / "hand_joints_3d_left.npy", labels[start:stop])

    translation = object_pose[:, 4:7].astype(np.float64)
    metadata = {
        **task,
        "source": "HO-Cap official models.zip, poses.zip, calibration.zip, labels.zip",
        "source_frames": int(len(object_all[object_slot])),
        "window_start": start,
        "window_stop": stop,
        "num_frames": int(stop - start),
        "max_object_translation_m": float(
            np.max(np.linalg.norm(translation - translation[0], axis=1))
        ),
        "max_object_rotation_rad": float(
            np.max(
                2.0
                * np.arccos(
                    np.clip(
                        np.abs(
                            object_pose[:, :4].astype(np.float64)
                            @ object_pose[0, :4].astype(np.float64)
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        ),
        "labels_ready": bool(args.labels_full is not None),
    }
    (task_root / "subset.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(task_root)


if __name__ == "__main__":
    main()
