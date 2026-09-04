#!/usr/bin/env python3
"""Catalog HO-Cap pose sequences and their official object identities.

Only one tiny official label record is fetched per sequence.  Full trajectories
come from the already-downloaded ``poses.zip`` archive, so this is suitable for
selecting diverse single-task validation sequences without downloading RGB-D.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from download_label_subset import HTTPRangeReader, LABELS_URL


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSES = EXPERIMENT_ROOT / "data" / "raw" / "poses.zip"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data" / "hocap_sequence_catalog.json"


def quaternion_excursion(quaternion_xyzw: np.ndarray) -> float:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norms, 1.0e-12)
    dots = np.abs(q @ q[0])
    return float(np.max(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))))


def motion_class(translation_m: float, rotation_rad: float, z_range_m: float) -> str:
    if rotation_rad >= 1.2 and translation_m <= 0.16:
        return "rotation_dominant"
    if z_range_m >= 0.12:
        return "vertical_manipulation"
    if translation_m >= 0.22 and z_range_m <= 0.08:
        return "planar_or_transfer"
    if rotation_rad >= 0.8:
        return "mixed_reorientation"
    return "local_manipulation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label-chunk-mb", type=int, default=1)
    args = parser.parse_args()

    label_reader = HTTPRangeReader(
        LABELS_URL, chunk_size=args.label_chunk_mb * 1024 * 1024
    )
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.poses) as poses_zip, zipfile.ZipFile(label_reader) as labels_zip:
        pose_sequences = sorted(
            name.removesuffix("/poses_m.npy")
            for name in poses_zip.namelist()
            if name.endswith("/poses_m.npy")
        )
        label_members: dict[str, list[str]] = defaultdict(list)
        for name in labels_zip.namelist():
            if name.endswith("/label_000000.npz"):
                parts = name.split("/")
                if len(parts) == 4:
                    label_members[f"{parts[0]}/{parts[1]}"].append(name)

        for sequence in pose_sequences:
            members = sorted(label_members.get(sequence, []))
            if not members:
                print(f"skip {sequence}: no official frame-zero label", flush=True)
                continue
            label_member = members[0]
            camera = label_member.split("/")[2]
            with np.load(io.BytesIO(labels_zip.read(label_member)), allow_pickle=False) as label:
                class_names = [str(value) for value in label["obj_class_names"]]
            object_names = [name for name in class_names if not name.endswith("_HAND")]

            mano = np.load(io.BytesIO(poses_zip.read(f"{sequence}/poses_m.npy")))
            objects = np.load(io.BytesIO(poses_zip.read(f"{sequence}/poses_o.npy")))
            if len(object_names) > objects.shape[0]:
                raise RuntimeError(
                    f"{sequence}: {objects.shape[0]} pose slots but "
                    f"{len(object_names)} label objects {object_names}"
                )

            per_object = []
            # poses_o.npy uses a fixed four-slot tensor even when a sequence
            # contains fewer objects. Official label names retain slot order;
            # trailing pose slots are padding and intentionally ignored.
            for slot, name in enumerate(object_names):
                pose = np.asarray(objects[slot], dtype=np.float64)
                translation = pose[:, 4:7]
                relative = translation - translation[0]
                translation_excursion = float(np.max(np.linalg.norm(relative, axis=1)))
                z_range = float(np.ptp(translation[:, 2]))
                xy_excursion = float(np.max(np.linalg.norm(relative[:, :2], axis=1)))
                rotation_excursion = quaternion_excursion(pose[:, :4])
                score = translation_excursion + 0.08 * rotation_excursion
                per_object.append(
                    {
                        "slot": slot,
                        "object_id": name,
                        "translation_excursion_m": translation_excursion,
                        "rotation_excursion_rad": rotation_excursion,
                        "z_range_m": z_range,
                        "xy_excursion_m": xy_excursion,
                        "activity_score": score,
                    }
                )
            active = max(per_object, key=lambda value: float(value["activity_score"]))

            per_hand = []
            for slot, side in enumerate(("right", "left")):
                pose = np.asarray(mano[slot], dtype=np.float64)
                translation = pose[:, -3:]
                excursion = float(
                    np.max(np.linalg.norm(translation - translation[0], axis=1))
                )
                nonzero = float(np.mean(np.linalg.norm(pose, axis=1) > 1.0e-8))
                per_hand.append(
                    {
                        "slot": slot,
                        "side": side,
                        "translation_excursion_m": excursion,
                        "valid_fraction": nonzero,
                    }
                )
            active_hand = max(
                per_hand,
                key=lambda value: float(value["translation_excursion_m"])
                * float(value["valid_fraction"]),
            )

            row = {
                "sequence": sequence,
                "camera": camera,
                "frames": int(objects.shape[1]),
                "object_group": str(active["object_id"]).split("_")[0],
                "active_object": active,
                "objects": per_object,
                "active_hand": active_hand,
                "hands": per_hand,
                "motion_class": motion_class(
                    float(active["translation_excursion_m"]),
                    float(active["rotation_excursion_rad"]),
                    float(active["z_range_m"]),
                ),
            }
            rows.append(row)
            print(
                f"{sequence} {active['object_id']} {row['motion_class']} "
                f"T={active['translation_excursion_m']:.3f}m "
                f"R={active['rotation_excursion_rad']:.2f}rad "
                f"hand={active_hand['side']} frames={row['frames']}",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "source": "HO-Cap official poses.zip and one labels.zip record per sequence",
        "sequence_count": len(rows),
        "motion_class_counts": {
            name: sum(row["motion_class"] == name for row in rows)
            for name in sorted({str(row["motion_class"]) for row in rows})
        },
        "sequences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
