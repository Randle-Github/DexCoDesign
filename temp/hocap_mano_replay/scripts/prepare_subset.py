#!/usr/bin/env python3
"""Extract a tiny HO-Cap hand-object subset from the official archives."""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = EXPERIMENT_ROOT / "data" / "raw"
SUBSET_ROOT = EXPERIMENT_ROOT / "data" / "subset"

SEQUENCE = "subject_7/20231022_192832"
HAND_SIDE = "left"
HAND_SLOT = 1  # poses_m.npy order is [right, left].
OBJECT_SLOT = 0
OBJECT_ID = "G04_1"


def extract_member(archive: Path, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        with zip_file.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def main() -> None:
    sequence_root = SUBSET_ROOT / SEQUENCE
    sequence_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(RAW_ROOT / "poses.zip") as zip_file:
        mano_all = np.load(
            io.BytesIO(zip_file.read(f"{SEQUENCE}/poses_m.npy"))
        ).astype(np.float32)
        object_all = np.load(
            io.BytesIO(zip_file.read(f"{SEQUENCE}/poses_o.npy"))
        ).astype(np.float32)

    mano_pose = mano_all[HAND_SLOT]
    object_pose = object_all[OBJECT_SLOT]
    if mano_pose.shape[0] != object_pose.shape[0]:
        raise RuntimeError("MANO and object trajectories have different lengths")

    np.save(sequence_root / "mano_pose_left.npy", mano_pose)
    np.save(sequence_root / f"object_pose_{OBJECT_ID}.npy", object_pose)

    object_root = SUBSET_ROOT / "models" / OBJECT_ID
    for filename in (
        "cleaned_mesh_2000.obj",
        "cleaned_mesh_10000.obj",
        "textured_mesh.obj",
        "textured_mesh.mtl",
        "textured_mesh_0.png",
    ):
        extract_member(
            RAW_ROOT / "models.zip",
            f"models/{OBJECT_ID}/{filename}",
            object_root / filename,
        )

    extract_member(
        RAW_ROOT / "calibration.zip",
        "calibration/mano/subject_7.yaml",
        SUBSET_ROOT / "calibration" / "mano" / "subject_7.yaml",
    )

    object_motion = np.linalg.norm(
        object_pose[:, 4:] - object_pose[0, 4:], axis=1
    )
    metadata = {
        "source": "HO-Cap official models.zip, poses.zip, calibration.zip",
        "sequence": SEQUENCE,
        "num_frames": int(mano_pose.shape[0]),
        "hand_side": HAND_SIDE,
        "mano_pose_layout": "[global_rotvec(3), hand_PCA(45), translation_m(3)]",
        "object_id": OBJECT_ID,
        "object_slot": OBJECT_SLOT,
        "object_pose_layout": "[qx, qy, qz, qw, tx, ty, tz]",
        "max_object_translation_m": float(object_motion.max()),
        "object_id_note": (
            "Verified from the official labels.zip frame annotation: pose slot 0 "
            "is G04_1. Trajectory values are unmodified."
        ),
    }
    (sequence_root / "subset.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
