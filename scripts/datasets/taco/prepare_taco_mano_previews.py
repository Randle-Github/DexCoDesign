#!/usr/bin/env python3
"""Retarget five selected TACO trajectories to the existing bimanual MANO hand."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TACO_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
DEFAULT_ACTIONS = ("cut", "pour in some", "hit", "stir", "put in")
PREFERRED_SEQUENCES = {
    "hit": "(hit, hammer, box)/20231031_178",
    "put in": "(put in, spatula, plate)/20231015_117",
}


def select_sequences(manifest: Path, actions: tuple[str, ...]) -> list[dict]:
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    selected = []
    for action in actions:
        candidates = [record for record in records if record["action"] == action]
        if not candidates:
            raise RuntimeError(f"No selected TACO sequence for action {action!r}")
        preferred = PREFERRED_SEQUENCES.get(action)
        selected.append(
            next(record for record in candidates if record["sequence_id"] == preferred)
            if preferred
            else max(candidates, key=lambda record: record["metrics"]["score"])
        )
    return selected


def fingertip_offset(mesh_root: Path, side: str, finger: str) -> np.ndarray:
    mesh = trimesh.load(
        mesh_root / f"{side}_{finger}3_visual.obj", process=False, force="mesh"
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    # The mirrored assets have opposite link-local distal axes: left extends
    # along +X while right extends along -X.  Using +X for the right hand
    # selects the joint-side cap instead of the physical fingertip.
    distal_sign = 1.0 if side == "left" else -1.0
    projection = distal_sign * vertices[:, 0]
    distal_projection = float(projection.max())
    cap = vertices[projection >= distal_projection - 5e-4]
    offset = cap.mean(axis=0)
    offset[0] = distal_sign * distal_projection
    return offset


def build_point_specs(model: mujoco.MjModel, asset_root: Path, side: str):
    specs = [(0, model.body(f"{side}_palm").id, np.zeros(3), 1.0)]
    mesh_root = asset_root / side / "meshes"
    for finger in FINGERS:
        mcp, pip, dip, tip = TACO_INDICES[finger]
        first = f"{side}_{finger}1z"
        second = f"{side}_{finger}2z" if finger == "thumb" else f"{side}_{finger}2"
        third = f"{side}_{finger}3"
        specs.extend(
            [
                (mcp, model.body(first).id, np.zeros(3), 0.8),
                (pip, model.body(second).id, np.zeros(3), 0.9),
                (dip, model.body(third).id, np.zeros(3), 1.0),
                (tip, model.body(third).id, fingertip_offset(mesh_root, side, finger), 1.5),
            ]
        )
    return specs


def point_position(data: mujoco.MjData, body_id: int, offset: np.ndarray) -> np.ndarray:
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    return np.asarray(data.xpos[body_id]) + rotation @ offset


def solve_hand(
    urdf: Path,
    targets: np.ndarray,
    side: str,
    stride: int,
    iterations: int,
    priority_target_ids: set[int] | None = None,
    priority_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model = mujoco.MjModel.from_xml_path(str(urdf))
    data = mujoco.MjData(model)
    specs = build_point_specs(model, urdf.parents[1], side)
    frame_ids = np.arange(0, len(targets), stride, dtype=np.int32)
    q_trajectory = []
    mean_errors = []
    q_previous = np.zeros(model.nq, dtype=np.float64)
    limited = np.asarray(model.jnt_limited, dtype=bool)
    ranges = np.asarray(model.jnt_range, dtype=np.float64)

    for trajectory_index, frame_id in enumerate(frame_ids):
        target = targets[frame_id].astype(np.float64)
        data.qpos[:] = q_previous
        frame_start = q_previous.copy()
        frame_iterations = iterations * 6 if trajectory_index == 0 else iterations
        for _ in range(frame_iterations):
            mujoco.mj_forward(model, data)
            jacobians, errors = [], []
            for target_id, body_id, offset, weight in specs:
                if priority_target_ids is not None and target_id in priority_target_ids:
                    weight *= priority_weight
                point = point_position(data, body_id, offset)
                jac_pos = np.zeros((3, model.nv), dtype=np.float64)
                jac_rot = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jac(model, data, jac_pos, jac_rot, point, body_id)
                jacobians.append(weight * jac_pos)
                errors.append(weight * (target[target_id] - point))
            if trajectory_index:
                temporal_weight = 0.012
                jacobians.append(temporal_weight * np.eye(model.nv))
                errors.append(temporal_weight * (frame_start - data.qpos))
            jacobian = np.vstack(jacobians)
            error = np.concatenate(errors)
            damping = 0.018
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(len(error)), error
            )
            maximum = float(np.max(np.abs(delta)))
            if maximum > 0.12:
                delta *= 0.12 / maximum
            data.qpos[:] += delta
            data.qpos[limited] = np.clip(
                data.qpos[limited], ranges[limited, 0], ranges[limited, 1]
            )

        mujoco.mj_forward(model, data)
        errors = [
            np.linalg.norm(target[target_id] - point_position(data, body_id, offset))
            for target_id, body_id, offset, _ in specs
        ]
        mean_errors.append(float(np.mean(errors)))
        q_previous = data.qpos.copy()
        q_trajectory.append(q_previous)

    joint_names = np.asarray(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    )
    diagnostics = {
        "side": side,
        "frames": len(frame_ids),
        "mean_keypoint_error_mm": float(np.mean(mean_errors) * 1000.0),
        "max_frame_mean_keypoint_error_mm": float(np.max(mean_errors) * 1000.0),
    }
    return np.asarray(q_trajectory, dtype=np.float32), joint_names, diagnostics


def matrix_to_wxyz(poses: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    return np.column_stack((poses[:, :3, 3], quaternion_xyzw[:, [3, 0, 1, 2]])).astype(
        np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/taco_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/taco_mano_preview"))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=18)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    mano_root = repo_root / "assets" / "robot_hands" / "direct_motor" / "mano"
    selected = select_sequences(root / "manifests" / "selected_100.jsonl", DEFAULT_ACTIONS)
    index = []

    for preview_id, record in enumerate(selected):
        sequence_id = record["sequence_id"]
        joints = np.load(root / "raw" / "hand_poses_3d" / sequence_id / "hand_joints.npy")
        preview_dir = output / f"{preview_id:02d}_{record['action'].replace(' ', '_')}"
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        payload = {}
        diagnostics = {}
        for hand_index, side in enumerate(("left", "right")):
            q, names, report = solve_hand(
                mano_root / side / "hand.urdf",
                joints[:, hand_index],
                side,
                args.stride,
                args.iterations,
            )
            payload[f"{side}_q"] = q
            payload[f"{side}_joint_names"] = names
            diagnostics[side] = report

        object_files = sorted((root / "raw" / "object_poses" / sequence_id).glob("*.npy"))
        support_heights = []
        scene_positions = []
        for pose_path in object_files:
            poses = np.load(pose_path)
            payload[f"{pose_path.stem}_pose_wxyz"] = matrix_to_wxyz(poses[:: args.stride])
            object_id = pose_path.stem.split("_", 1)[1]
            source_mesh = trimesh.load(
                root / "raw" / "object_models" / f"{object_id}_cm.obj",
                process=False,
                force="mesh",
            )
            vertices_m = np.asarray(source_mesh.vertices, dtype=np.float64) * 0.01
            first_pose = poses[0]
            support_heights.append(
                float((vertices_m @ first_pose[:3, :3].T + first_pose[:3, 3])[:, 2].min())
            )
            scene_positions.append(poses[:, :3, 3])
        table_z = float(min(support_heights))
        scene_center = np.concatenate(scene_positions, axis=0).mean(axis=0)
        scene_info = {
            "table_z": table_z,
            "camera_target": [float(scene_center[0]), float(scene_center[1]), table_z + 0.10],
            "camera_eye": [
                float(scene_center[0] + 0.48),
                float(scene_center[1] + 0.48),
                table_z + 0.38,
            ],
        }
        payload["metadata_json"] = np.asarray(
            json.dumps(
                {
                    **record,
                    "stride": args.stride,
                    "diagnostics": diagnostics,
                    "scene": scene_info,
                }
            )
        )
        np.savez_compressed(preview_dir / "trajectory.npz", **payload)
        (preview_dir / "metadata.json").write_text(
            json.dumps({**record, "diagnostics": diagnostics, "scene": scene_info}, indent=2)
            + "\n"
        )
        for object_id in record["object_ids"].values():
            source = root / "raw" / "object_models" / f"{object_id}_cm.obj"
            destination = preview_dir / f"{object_id}_m.obj"
            object_mesh = trimesh.load(source, process=False, force="mesh")
            object_mesh.apply_scale(0.01)
            object_mesh.export(destination)
        index.append(
            {
                "preview_id": preview_id,
                "directory": preview_dir.name,
                "sequence_id": sequence_id,
                "action": record["action"],
                "diagnostics": diagnostics,
            }
        )
        print(json.dumps(index[-1]), flush=True)

    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
