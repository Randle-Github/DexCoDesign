#!/usr/bin/env python3
"""Prepare the canonical DexCoDesign subset from an extracted HO-Cap dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOCAP_ROOT = Path.home() / "Desktop" / "HO-Cap" / "datasets"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "data" / "subset"
SEQUENCE = Path("subject_7") / "20231022_192832"
CAMERA = "043422252387"
HAND_SLOT = 1  # HO-Cap pose/label order is [right, left].
OBJECT_SLOT = 0
OBJECT_ID = "G04_1"
MODEL_FILES = (
    "cleaned_mesh_2000.obj",
    "cleaned_mesh_10000.obj",
    "textured_mesh.obj",
    "textured_mesh.mtl",
    "textured_mesh_0.png",
)
JOINT_ORDER = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


def homogeneous_matrix(values: list[float]) -> np.ndarray:
    """Convert HO-Cap's flattened 3x4 extrinsic into a 4x4 matrix."""
    if len(values) != 12:
        raise ValueError(f"Expected 12 extrinsic values, received {len(values)}")
    return np.asarray(
        [values[0:4], values[4:8], values[8:12], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def numbered_label_files(camera_root: Path) -> list[Path]:
    labels = sorted(
        camera_root.glob("label_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", maxsplit=1)[1]),
    )
    frame_ids = [int(path.stem.rsplit("_", maxsplit=1)[1]) for path in labels]
    if frame_ids != list(range(len(labels))):
        raise RuntimeError(
            "Label frames must be contiguous and start at zero; "
            f"found first={frame_ids[:5]} and last={frame_ids[-5:]}"
        )
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hocap-root",
        type=Path,
        default=DEFAULT_HOCAP_ROOT,
        help="Root of the already-extracted HO-Cap datasets directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for the compact DexCoDesign subset.",
    )
    args = parser.parse_args()

    hocap_root = args.hocap_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_sequence = hocap_root / SEQUENCE
    output_sequence = output_root / SEQUENCE

    mano_all = np.load(source_sequence / "poses_m.npy")
    object_all = np.load(source_sequence / "poses_o.npy")
    if mano_all.ndim != 3 or mano_all.shape[0] <= HAND_SLOT:
        raise RuntimeError(f"Unexpected poses_m.npy shape: {mano_all.shape}")
    if object_all.ndim != 3 or object_all.shape[0] <= OBJECT_SLOT:
        raise RuntimeError(f"Unexpected poses_o.npy shape: {object_all.shape}")

    mano_pose = np.asarray(mano_all[HAND_SLOT], dtype=np.float32)
    object_pose = np.asarray(object_all[OBJECT_SLOT], dtype=np.float32)
    if len(mano_pose) != len(object_pose):
        raise RuntimeError("MANO and object trajectories have different lengths")

    extrinsics_source = (
        hocap_root / "calibration" / "extrinsics" / "extrinsics_20231014.yaml"
    )
    calibration = yaml.safe_load(extrinsics_source.read_text(encoding="utf-8"))[
        "extrinsics"
    ]
    camera_to_world = (
        np.linalg.inv(homogeneous_matrix(calibration["tag_1"]))
        @ homogeneous_matrix(calibration[CAMERA])
    )

    label_files = numbered_label_files(source_sequence / CAMERA)
    if len(label_files) != len(mano_pose):
        raise RuntimeError(
            f"Found {len(label_files)} label frames but {len(mano_pose)} pose frames"
        )
    joints_camera = []
    for label_path in label_files:
        with np.load(label_path) as label:
            joints = np.asarray(label["hand_joints_3d"], dtype=np.float32)
        if joints.shape != (2, 21, 3):
            raise RuntimeError(
                f"Unexpected hand_joints_3d shape in {label_path}: {joints.shape}"
            )
        joints_camera.append(joints[HAND_SLOT])
    joints_camera_array = np.stack(joints_camera)
    joints_world = (
        joints_camera_array @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    ).astype(np.float32)

    output_sequence.mkdir(parents=True, exist_ok=True)
    np.save(output_sequence / "mano_pose_left.npy", mano_pose)
    np.save(output_sequence / f"object_pose_{OBJECT_ID}.npy", object_pose)
    np.save(output_sequence / "hand_joints_3d_left.npy", joints_world)

    object_output = output_root / "models" / OBJECT_ID
    object_output.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        shutil.copy2(hocap_root / "models" / OBJECT_ID / filename, object_output)

    mano_calibration_output = output_root / "calibration" / "mano"
    extrinsics_output = output_root / "calibration" / "extrinsics"
    mano_calibration_output.mkdir(parents=True, exist_ok=True)
    extrinsics_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        hocap_root / "calibration" / "mano" / "subject_7.yaml",
        mano_calibration_output / "subject_7.yaml",
    )
    shutil.copy2(
        extrinsics_source,
        extrinsics_output / "extrinsics_20231014.yaml",
    )

    object_motion = np.linalg.norm(object_pose[:, 4:7] - object_pose[0, 4:7], axis=1)
    metadata = {
        "source": str(hocap_root),
        "sequence": str(SEQUENCE),
        "num_frames": int(len(mano_pose)),
        "hand_side": "left",
        "hand_slot": HAND_SLOT,
        "mano_pose_layout": "[global_rotvec(3), hand_PCA(45), translation_m(3)]",
        "object_id": OBJECT_ID,
        "object_slot": OBJECT_SLOT,
        "object_pose_layout": "[qx, qy, qz, qw, tx, ty, tz]",
        "label_camera": CAMERA,
        "joint_coordinate_frame": "HO-Cap world",
        "joint_order": list(JOINT_ORDER),
        "camera_to_world": camera_to_world.tolist(),
        "max_object_translation_m": float(object_motion.max()),
    }
    (output_sequence / "subset.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Prepared {len(mano_pose)} frames in {output_sequence}")
    print(f"Copied object assets to {object_output}")
    print("isaaclab_reference.npz is generated separately by prepare_isaaclab_reference.py")


if __name__ == "__main__":
    main()
