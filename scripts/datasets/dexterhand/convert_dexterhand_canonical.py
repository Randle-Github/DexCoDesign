#!/usr/bin/env python3
"""Restore canonical DexterHand MANO/object trajectories from official NPZ sessions."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Compatibility for the official MANO pickle with recent NumPy/Python.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for name, value in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("unicode", str), ("str", str)):
    if name not in np.__dict__:
        setattr(np, name, value)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "datasets" / "common"))
from trajectory_schema import save_trajectory  # noqa: E402

TIP_VERTICES = {"index": 320, "middle": 443, "ring": 554, "pinky": 671, "thumb": 744}
MANO_BLOCKS = {"thumb": (13, 14, 15), "index": (1, 2, 3), "middle": (4, 5, 6), "ring": (10, 11, 12), "pinky": (7, 8, 9)}
FACE_NAMES = ("U", "D", "L", "R", "F", "B")


@torch.inference_mode()
def evaluate_joints(layer, source, start: int, stop: int, batch_size: int) -> np.ndarray:
    batches = []
    for lo in range(start, stop, batch_size):
        hi = min(stop, lo + batch_size)
        count = hi - lo
        output = layer(
            betas=torch.from_numpy(np.asarray(source["hand_shapes"][lo:hi], dtype=np.float32)),
            global_orient=torch.zeros((count, 3), dtype=torch.float32),
            hand_pose=torch.from_numpy(np.asarray(source["hand_poses"][lo:hi], dtype=np.float32)),
            transl=torch.zeros((count, 3), dtype=torch.float32),
        )
        joints = output.joints.detach().cpu().numpy()
        vertices = output.vertices.detach().cpu().numpy()
        ordered = [joints[:, 0]]
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            ordered.extend(joints[:, list(MANO_BLOCKS[finger])].transpose(1, 0, 2))
            ordered.append(vertices[:, TIP_VERTICES[finger]])
        local = np.stack(ordered, axis=1)
        rotation = Rotation.from_rotvec(np.asarray(source["hand_orientations_axis_angle"][lo:hi], dtype=np.float32)).as_matrix()
        world = np.einsum("bij,bkj->bki", rotation, local)
        world += np.asarray(source["hand_translations"][lo:hi], dtype=np.float32)[:, None, :]
        batches.append(world.astype(np.float32))
    return np.concatenate(batches)


def spans(valid: np.ndarray, starts: set[int]):
    cuts = sorted({0, len(valid), *starts, *(np.flatnonzero(valid[1:] != valid[:-1]) + 1).tolist()})
    for left, right in zip(cuts, cuts[1:]):
        if valid[left] and right - left >= 10:
            yield left, right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/dexterhand_v1"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "canonical").resolve()
    import smplx
    layer = smplx.MANO(str(args.model_root.resolve() / "MANO_LEFT.pkl"), is_rhand=False, use_pca=False, flat_hand_mean=True).eval()
    records = []
    for source_path in sorted((root / "raw" / "sessions").glob("*-fps_20.npz")):
        source = np.load(source_path, allow_pickle=True)
        meta = source["metadata"].item()
        session = str(meta["mocap_session_name"])
        count = len(source["hand_translations"])
        invalid = np.asarray(meta.get("invalid_point_value", [-1000, -1000, -1000]), dtype=np.float64)
        translations = np.asarray(source["object_translations"], dtype=np.float64)
        roots_xyzw = np.asarray(source["object_orientations_quat_xyzw"], dtype=np.float64)
        valid = np.isfinite(translations).all(axis=1) & (np.linalg.norm(translations - invalid, axis=1) > 1e-5)
        valid &= np.isfinite(roots_xyzw).all(axis=1) & (np.linalg.norm(roots_xyzw, axis=1) > 1e-8)
        starts = {int(x) for x in meta.get("index_init_frame", [0]) if 0 <= int(x) < count}
        for segment_index, (left, right) in enumerate(spans(valid, starts)):
            hand = np.full((right-left, 2, 21, 3), np.nan, dtype=np.float32)
            hand[:, 0] = evaluate_joints(layer, source, left, right, args.batch_size)
            hand_valid = np.zeros((right-left, 2), dtype=bool)
            hand_valid[:, 0] = True
            object_xyzw = roots_xyzw[left:right].astype(np.float32)
            object_root = np.concatenate((translations[left:right].astype(np.float32), object_xyzw[:, [3, 0, 1, 2]]), axis=1)[:, None]
            is_rubik = session.startswith("RubiksCube")
            if is_rubik:
                face = np.asarray(source["object_face_designators"][left:right]).astype(str)
                angle = np.asarray(source["object_rotation_angles"][left:right], dtype=np.float32)
                object_q = np.zeros((right-left, 1, len(FACE_NAMES)), dtype=np.float32)
                for index, name in enumerate(FACE_NAMES):
                    object_q[face == name, 0, index] = angle[face == name]
                joint_names = [f"active_face_{name}" for name in FACE_NAMES]
            else:
                object_q = np.empty((right-left, 1, 0), dtype=np.float32)
                joint_names = []
            sequence_id = f"{session}__seg_{segment_index:02d}"
            destination = output / sequence_id / "trajectory.npz"
            object_info = {
                "object_id": session.split("_")[0], "role": "manipulated_object",
                "nominal_dimensions_m": [float(x) for x in meta["object_size"]],
                "mesh_asset_dir": str(Path("assets/objects") / session),
                "joint_names": joint_names, "joint_types": ["revolute"] * len(joint_names),
                "articulation_status": (
                    "incremental per-frame face turns; not cumulative simulator joint state"
                    if is_rubik else "rigid"
                ),
            }
            save_trajectory(
                destination, hand_joints_m=hand, hand_valid=hand_valid,
                object_root_pose_wxyz=object_root, object_joint_positions_rad=object_q,
                frame_indices=np.arange(left, right, dtype=np.int64), fps=float(meta.get("fps", 20)),
                metadata={
                    "dataset": "DexterCap/DexterHand", "sequence_id": sequence_id,
                    "source_session": session, "source_file": str(source_path.relative_to(root)),
                    "source_frame_range_inclusive_exclusive": [left, right],
                    "hand_representation": "official MANO_LEFT forward kinematics in world space",
                    "objects": [object_info], "simulator_ready": False,
                },
            )
            if is_rubik:
                np.savez_compressed(
                    destination.parent / "rubik_source_trace.npz",
                    active_face_designator=face, active_face_angle_rad=angle,
                    source_frame_indices=np.arange(left, right, dtype=np.int64),
                )
            records.append({"sequence_id": sequence_id, "path": str(destination.relative_to(output)), "frames": right-left, "fps": float(meta.get("fps", 20))})
            print(f"DEXTERHAND_CANONICAL_READY {sequence_id}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (root / "raw/conversion_manifest.json").write_text(json.dumps({"schema": "dexcodesign.pose_dataset.v1", "dataset": "DexterCap/DexterHand", "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
