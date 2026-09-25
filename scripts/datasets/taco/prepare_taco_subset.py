#!/usr/bin/env python3
"""Build a compact, quality-filtered TACO pose subset without videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
SOURCE = {
    "dataset": "TACO v1",
    "official_repo": "https://github.com/leolyliu/TACO-Instructions",
    "mirror": "https://huggingface.co/datasets/mzhobro/taco_dataset",
    "license": "CC BY 4.0",
}


@dataclass
class Metrics:
    frames: int
    hand_finite_fraction: float
    hand_jump_p999_m: float
    object_jump_p995_m: float
    object_excursion_m: float
    rotation_jump_p995_rad: float
    length_mismatch_frames: int
    score: float


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_angle(rotations: np.ndarray) -> np.ndarray:
    if len(rotations) < 2:
        return np.zeros(0, dtype=np.float32)
    relative = np.einsum("tji,tjk->tik", rotations[:-1], rotations[1:])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _pose_files_by_sequence(archive: zipfile.ZipFile, root: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == root and name.endswith(".npy"):
            result[f"{parts[1]}/{parts[2]}"].append(name)
    return result


def _load_npy(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    return np.load(io.BytesIO(archive.read(member)), allow_pickle=False)


def audit_sequence(
    row: dict[str, str],
    hand_zip: zipfile.ZipFile,
    object_zip: zipfile.ZipFile,
    object_members: dict[str, list[str]],
) -> tuple[Metrics | None, list[str], dict[str, str]]:
    sequence_id = row["sequence_id"]
    hand_member = f"Hand_Poses_3D/{sequence_id}/hand_joints.npy"
    pose_members = sorted(object_members.get(sequence_id, []))
    reasons: list[str] = []
    role_to_object: dict[str, str] = {}

    if hand_member not in hand_zip.NameToInfo:
        return None, ["missing_hand_pose"], role_to_object
    if len(pose_members) < 2:
        return None, ["missing_tool_or_target_pose"], role_to_object

    hand = _load_npy(hand_zip, hand_member)
    object_poses = [_load_npy(object_zip, member) for member in pose_members]
    for member in pose_members:
        role, object_id = Path(member).stem.split("_", 1)
        role_to_object[role] = object_id

    if hand.ndim != 4 or hand.shape[1:] != (2, 21, 3):
        return None, ["bad_hand_shape"], role_to_object
    if any(p.ndim != 3 or p.shape[1:] != (4, 4) for p in object_poses):
        return None, ["bad_object_pose_shape"], role_to_object

    lengths = [len(hand), *(len(p) for p in object_poses)]
    frames = min(lengths)
    mismatch = max(lengths) - frames
    hand = hand[:frames]
    object_poses = [p[:frames] for p in object_poses]

    hand_finite = float(np.isfinite(hand).mean())
    hand_steps = np.linalg.norm(np.diff(hand, axis=0), axis=-1)
    hand_jump = float(np.nanpercentile(hand_steps, 99.9)) if hand_steps.size else math.inf

    object_jumps: list[np.ndarray] = []
    object_excursions: list[float] = []
    rotation_jumps: list[np.ndarray] = []
    for pose in object_poses:
        translation = pose[:, :3, 3]
        object_jumps.append(np.linalg.norm(np.diff(translation, axis=0), axis=-1))
        object_excursions.append(float(np.linalg.norm(translation - translation[0], axis=-1).max()))
        rotation_jumps.append(_rotation_angle(pose[:, :3, :3]))
        bottom_error = np.abs(pose[:, 3] - np.array([0, 0, 0, 1], dtype=pose.dtype)).max()
        ortho_error = np.abs(
            np.einsum("tji,tjk->tik", pose[:, :3, :3], pose[:, :3, :3]) - np.eye(3)
        ).max()
        determinant_error = np.abs(np.linalg.det(pose[:, :3, :3]) - 1.0).max()
        if bottom_error > 1e-3 or ortho_error > 2e-2 or determinant_error > 2e-2:
            reasons.append("invalid_object_transform")

    object_jump = max(float(np.nanpercentile(values, 99.5)) for values in object_jumps)
    object_excursion = max(object_excursions)
    rotation_jump = max(float(np.nanpercentile(values, 99.5)) for values in rotation_jumps)

    if not (_bool(row["has_hand_poses"]) and _bool(row["has_object_poses"])):
        reasons.append("metadata_pose_missing")
    if row["calib_status"] in {"bad_hand_pose", "incomplete"}:
        reasons.append(f"metadata_{row['calib_status']}")
    if frames < 90:
        reasons.append("too_short")
    if frames > 360:
        reasons.append("too_long_for_atomic_demo")
    if mismatch > 2:
        reasons.append("modality_length_mismatch")
    if hand_finite < 1.0:
        reasons.append("nonfinite_hand_pose")
    if hand_jump > 0.080:
        reasons.append("hand_pose_jump")
    if object_jump > 0.045:
        reasons.append("object_pose_jump")
    if rotation_jump > 0.45:
        reasons.append("object_rotation_jump")
    if object_excursion < 0.075:
        reasons.append("insufficient_object_motion")

    duration_term = math.exp(-abs(frames - 155) / 170.0)
    motion_term = min(object_excursion / 0.25, 1.0)
    smoothness = math.exp(-object_jump / 0.020) * math.exp(-hand_jump / 0.045)
    calibration = 1.0 if row["calib_status"] == "good" else 0.85
    score = 0.30 * duration_term + 0.30 * motion_term + 0.30 * smoothness + 0.10 * calibration
    metrics = Metrics(
        frames=frames,
        hand_finite_fraction=hand_finite,
        hand_jump_p999_m=hand_jump,
        object_jump_p995_m=object_jump,
        object_excursion_m=object_excursion,
        rotation_jump_p995_rad=rotation_jump,
        length_mismatch_frames=mismatch,
        score=score,
    )
    return metrics, sorted(set(reasons)), role_to_object


def _copy_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def _model_member_map(model_zip: zipfile.ZipFile) -> dict[str, str]:
    result = {}
    for name in model_zip.namelist():
        if name.endswith("_cm.obj"):
            result[Path(name).stem.removesuffix("_cm")] = name
    return result


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/taco_v1"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-per-triplet", type=int, default=1)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        help="JSON list of sequence IDs rejected by downstream physical validation",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    downloads = root / "_downloads"
    metadata_path = root / "taco_info.csv"
    hand_archive_path = downloads / "Hand_Poses_3D.zip"
    object_archive_path = downloads / "Object_Poses.zip"
    model_archive_path = downloads / "Object_Models.zip"
    for path in (metadata_path, hand_archive_path, object_archive_path, model_archive_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        metadata_rows = list(csv.DictReader(stream))

    exclusions_path = args.exclusions or (root / "manifests" / "settling_exclusions.json")
    excluded_sequences = set()
    if exclusions_path.is_file():
        excluded_sequences = set(json.loads(exclusions_path.read_text(encoding="utf-8")))

    accepted_pool: list[dict] = []
    rejected: list[dict] = []
    with zipfile.ZipFile(hand_archive_path) as hand_zip, zipfile.ZipFile(object_archive_path) as object_zip:
        object_members = _pose_files_by_sequence(object_zip, "Object_Poses")
        for row in metadata_rows:
            metrics, reasons, role_to_object = audit_sequence(row, hand_zip, object_zip, object_members)
            record = {
                "sequence_id": row["sequence_id"],
                "triplet": row["triplet"],
                "action": row["action"],
                "tool_category": row["tool"],
                "target_category": row["object"],
                "object_ids": role_to_object,
                "metrics": asdict(metrics) if metrics else None,
            }
            if row["sequence_id"] in excluded_sequences:
                reasons.append("failed_mujoco_settling")
            if reasons:
                record["reasons"] = reasons
                rejected.append(record)
            else:
                accepted_pool.append(record)

        accepted_pool.sort(key=lambda record: (-record["metrics"]["score"], record["sequence_id"]))
        selected: list[dict] = []
        triplet_counts: Counter[str] = Counter()
        for record in accepted_pool:
            if triplet_counts[record["triplet"]] >= args.max_per_triplet:
                continue
            selected.append(record)
            triplet_counts[record["triplet"]] += 1
            if len(selected) == args.count:
                break
        if len(selected) < args.count:
            raise RuntimeError(
                f"Only {len(selected)} diverse sequences passed; requested {args.count}. "
                "Increase --max-per-triplet only after reviewing the rejection manifest."
            )

        selected_ids = {record["sequence_id"] for record in selected}
        for record in accepted_pool:
            if record["sequence_id"] not in selected_ids:
                omitted = dict(record)
                omitted["reasons"] = ["not_selected_after_diversity_ranking"]
                rejected.append(omitted)

        raw_root = root / "raw"
        if raw_root.exists():
            shutil.rmtree(raw_root)
        for record in selected:
            sequence_id = record["sequence_id"]
            _copy_member(
                hand_zip,
                f"Hand_Poses_3D/{sequence_id}/hand_joints.npy",
                raw_root / "hand_poses_3d" / sequence_id / "hand_joints.npy",
            )
            for member in object_members[sequence_id]:
                _copy_member(
                    object_zip,
                    member,
                    raw_root / "object_poses" / sequence_id / Path(member).name,
                )

    selected_object_ids = sorted(
        {object_id for record in selected for object_id in record["object_ids"].values()}
    )
    with zipfile.ZipFile(model_archive_path) as model_zip:
        model_members = _model_member_map(model_zip)
        missing_models = [object_id for object_id in selected_object_ids if object_id not in model_members]
        if missing_models:
            raise RuntimeError(f"Missing object models: {missing_models}")
        for object_id in selected_object_ids:
            member = model_members[object_id]
            _copy_member(model_zip, member, root / "raw" / "object_models" / Path(member).name)

    manifests = root / "manifests"
    _write_jsonl(manifests / "selected_100.jsonl", selected)
    _write_jsonl(manifests / "rejected.jsonl", sorted(rejected, key=lambda record: record["sequence_id"]))
    rejection_counts = Counter(reason for record in rejected for reason in record["reasons"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "selection": {
            "requested": args.count,
            "selected": len(selected),
            "distinct_triplets": len({record["triplet"] for record in selected}),
            "distinct_actions": len({record["action"] for record in selected}),
            "distinct_object_models": len(selected_object_ids),
            "max_per_triplet": args.max_per_triplet,
        },
        "inventory": {
            "metadata_sequences": len(metadata_rows),
            "pose_quality_pool": len(accepted_pool),
            "rejected_or_not_selected": len(rejected),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "archives": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (hand_archive_path, object_archive_path, model_archive_path)
        },
        "notes": [
            "No RGB, depth, segmentation, or camera video data is downloaded.",
            "hand_joints.npy has shape (T, 2, 21, 3), ordered left then right.",
            "Object meshes are in centimeters and must be scaled by 0.01 for simulation meters.",
            "Physical table-stability validation is a separate simulator gate; pose-only checks cannot prove stable contact.",
        ],
    }
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.keep_archives:
        shutil.rmtree(downloads)

    print(json.dumps(summary["selection"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
