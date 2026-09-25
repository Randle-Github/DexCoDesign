#!/usr/bin/env python3
"""Canonical, dataset-independent pose trajectory schema for DexCoDesign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCHEMA = "dexcodesign.pose_trajectory.v1"
HAND_SIDES = np.asarray(("left", "right"))


def save_trajectory(
    path: Path,
    *,
    hand_joints_m: np.ndarray,
    hand_valid: np.ndarray,
    object_root_pose_wxyz: np.ndarray,
    object_joint_positions_rad: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    metadata: dict,
) -> None:
    """Write one canonical trajectory after strict shape/unit validation."""
    hand = np.asarray(hand_joints_m, dtype=np.float32)
    valid = np.asarray(hand_valid, dtype=bool)
    roots = np.asarray(object_root_pose_wxyz, dtype=np.float32)
    object_q = np.asarray(object_joint_positions_rad, dtype=np.float32)
    frames = np.asarray(frame_indices, dtype=np.int64)
    count = len(frames)
    if hand.shape != (count, 2, 21, 3):
        raise ValueError(f"hand_joints_m must be [T,2,21,3], got {hand.shape}")
    if valid.shape != (count, 2):
        raise ValueError(f"hand_valid must be [T,2], got {valid.shape}")
    if roots.ndim != 3 or roots.shape[0] != count or roots.shape[2] != 7:
        raise ValueError(f"object_root_pose_wxyz must be [T,O,7], got {roots.shape}")
    if object_q.ndim != 3 or object_q.shape[:2] != roots.shape[:2]:
        raise ValueError(f"object_joint_positions_rad must be [T,O,J], got {object_q.shape}")
    if not np.isfinite(hand[valid]).all():
        raise ValueError("valid hand samples contain non-finite values")
    if not np.isfinite(roots).all() or not np.isfinite(object_q).all():
        raise ValueError("object samples contain non-finite values")
    norms = np.linalg.norm(roots[..., 3:7], axis=-1)
    if roots.shape[1] and np.max(np.abs(norms - 1.0)) > 2e-3:
        raise ValueError("object quaternions are not normalized wxyz")
    doc = {
        "schema": SCHEMA,
        "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
        "hand_order": ["left", "right"],
        **metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hand_joints_m=hand,
        hand_valid=valid,
        hand_sides=HAND_SIDES,
        object_root_pose_wxyz=roots,
        object_joint_positions_rad=object_q,
        frame_indices=frames,
        fps=np.asarray(fps, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(doc, ensure_ascii=False)),
    )


def audit_trajectory(path: Path) -> dict:
    required = {
        "hand_joints_m",
        "hand_valid",
        "hand_sides",
        "object_root_pose_wxyz",
        "object_joint_positions_rad",
        "frame_indices",
        "fps",
        "metadata_json",
    }
    try:
        data = np.load(path, allow_pickle=False)
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing fields: {missing}")
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"schema={metadata.get('schema')!r}")
        hand = data["hand_joints_m"]
        valid = data["hand_valid"]
        roots = data["object_root_pose_wxyz"]
        object_q = data["object_joint_positions_rad"]
        frames = data["frame_indices"]
        count = len(frames)
        if hand.shape != (count, 2, 21, 3) or valid.shape != (count, 2):
            raise ValueError(f"bad hand shapes {hand.shape}, {valid.shape}")
        if roots.ndim != 3 or roots.shape[0] != count or roots.shape[2] != 7:
            raise ValueError(f"bad object root shape {roots.shape}")
        if object_q.ndim != 3 or object_q.shape[:2] != roots.shape[:2]:
            raise ValueError(f"bad object joint shape {object_q.shape}")
        if not np.isfinite(hand[valid]).all():
            raise ValueError("non-finite valid hand samples")
        if not np.isfinite(roots).all() or not np.isfinite(object_q).all():
            raise ValueError("non-finite object samples")
        quaternion_error = float(np.max(np.abs(np.linalg.norm(roots[..., 3:], axis=-1) - 1)))
        if roots.shape[1] and quaternion_error > 2e-3:
            raise ValueError(f"quaternion norm error {quaternion_error}")
        return {
            "path": str(path),
            "ok": True,
            "dataset": metadata.get("dataset"),
            "sequence_id": metadata.get("sequence_id"),
            "frames": count,
            "objects": roots.shape[1],
            "object_joint_capacity": object_q.shape[2],
        }
    except Exception as error:
        return {"path": str(path), "ok": False, "error": str(error)}
