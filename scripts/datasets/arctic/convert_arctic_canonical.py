#!/usr/bin/env python3
"""Convert the selected ARCTIC benchmark to canonical exact-MANO trajectories."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

# MANO v1.2 pickles reference chumpy.  Chumpy 0.70 predates Python 3.11 and
# NumPy 1.24, but only needs these names while unpickling the model arrays.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _value in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("unicode", str), ("str", str)):
    if _name not in np.__dict__:
        setattr(np, _name, _value)


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "datasets" / "common"))
from trajectory_schema import save_trajectory  # noqa: E402


def mano_layer(model_root: Path, side: str):
    import smplx

    return smplx.MANO(
        str(model_root / f"MANO_{side.upper()}.pkl"),
        is_rhand=side == "right", use_pca=False, flat_hand_mean=False,
    ).eval()


@torch.inference_mode()
def mano_joints(layer, source: np.lib.npyio.NpzFile, side: str, batch_size: int) -> np.ndarray:
    orient = np.asarray(source[f"{side}_global_orient_axis_angle"], dtype=np.float32)
    pose = np.asarray(source[f"{side}_pose_axis_angle"], dtype=np.float32)
    translation = np.asarray(source[f"{side}_translation_m"], dtype=np.float32)
    shape = np.asarray(source[f"{side}_shape"], dtype=np.float32).reshape(-1, 10)
    result = []
    for start in range(0, len(pose), batch_size):
        stop = min(len(pose), start + batch_size)
        betas = shape if len(shape) == stop - start else np.repeat(shape[:1], stop - start, axis=0)
        output = layer(
            betas=torch.from_numpy(betas),
            global_orient=torch.from_numpy(orient[start:stop]),
            hand_pose=torch.from_numpy(pose[start:stop]),
            transl=torch.from_numpy(translation[start:stop]),
            return_verts=True,
        )
        base = output.joints.detach().cpu().numpy()
        vertices = output.vertices.detach().cpu().numpy()
        # MANO native order is wrist, index, middle, pinky, ring, thumb with
        # three skeletal joints per finger.  DexCoDesign uses the common
        # wrist, thumb, index, middle, ring, pinky order and adds mesh tips.
        blocks = {
            "thumb": (13, 14, 15, 744),
            "index": (1, 2, 3, 320),
            "middle": (4, 5, 6, 443),
            "ring": (10, 11, 12, 554),
            "pinky": (7, 8, 9, 671),
        }
        ordered = [base[:, 0]]
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            first, second, third, tip = blocks[finger]
            ordered.extend((base[:, first], base[:, second], base[:, third], vertices[:, tip]))
        result.append(np.stack(ordered, axis=1))
    return np.concatenate(result).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "canonical_100")).resolve()
    left_layer = mano_layer(args.model_root.resolve(), "left")
    right_layer = mano_layer(args.model_root.resolve(), "right")
    manifest = json.loads((root / "manifests" / "benchmark_100.json").read_text())
    converted = []

    for record in manifest["records"]:
        source_path = root / record["path"]
        source = np.load(source_path, allow_pickle=False)
        frames = np.asarray(source["frame_indices"], dtype=np.int64)
        hands = np.stack(
            (
                mano_joints(left_layer, source, "left", args.batch_size),
                mano_joints(right_layer, source, "right", args.batch_size),
            ),
            axis=1,
        )
        object_root = np.concatenate(
            (source["object_root_position_m"], source["object_root_quaternion_wxyz"]), axis=1
        )[:, None, :]
        object_q = np.asarray(source["object_joint_position_rad"], dtype=np.float32)[:, None, None]
        sequence_id = record["sequence_id"]
        destination = output / f"{source_path.stem}" / "trajectory.npz"
        save_trajectory(
            destination,
            hand_joints_m=hands,
            hand_valid=np.ones((len(frames), 2), dtype=bool),
            object_root_pose_wxyz=object_root,
            object_joint_positions_rad=object_q,
            frame_indices=frames,
            fps=float(record["output_rate_hz"]),
            metadata={
                "dataset": "ARCTIC",
                "sequence_id": sequence_id,
                "source_file": record["path"],
                "hand_representation": "official MANO forward kinematics, 21 world-space joints",
                "objects": [{
                    "object_id": record["object_id"], "role": "articulated_object",
                    "joint_names": [record["articulation"]["joint_name"]],
                    "joint_types": [record["articulation"]["type"]],
                    "joint_axes_parent": [record["articulation"]["axis_parent"]],
                }],
            },
        )
        converted.append({"sequence_id": sequence_id, "path": str(destination.relative_to(root))})
        print(f"ARCTIC_CANONICAL_READY {len(converted):03d}/{len(manifest['records'])} {sequence_id}", flush=True)

    (output / "manifest.json").write_text(
        json.dumps({"schema": "dexcodesign.pose_dataset.v1", "dataset": "ARCTIC", "records": converted}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
