"""One captured MANO motion supplies both candidate IK and RL task targets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def targets_in_wrist_frame(points_w, directions_w, wrist_position, wrist_quaternion_xyzw):
    """Express world-space MANO targets in the candidate IK wrist frame."""
    points_w, directions_w = np.asarray(points_w), np.asarray(directions_w)
    rotation = Rotation.from_quat(wrist_quaternion_xyzw).as_matrix()
    if points_w.shape != (len(rotation), 5, 3) or directions_w.shape != points_w.shape:
        raise ValueError("MANO targets and seed wrist must have matching (frames, 5, 3) shapes")
    inverse = rotation.transpose(0, 2, 1)
    return (
        np.einsum("tij,tfj->tfi", inverse, points_w - np.asarray(wrist_position)[:, None]),
        np.einsum("tij,tfj->tfi", inverse, directions_w),
    )


def build_mano_task_targets(capture_path: Path) -> dict[str, np.ndarray]:
    # Reuse the established MANO FK and terminal-surface direction convention.
    from retarget_captured_success_all_hands import captured_mano_targets

    capture_path = Path(capture_path).expanduser().resolve()
    with np.load(capture_path, allow_pickle=False) as capture:
        count = len(capture["hand_q"])
        object_pose = np.array(capture["object_pose_wxyz"], copy=True)
        if object_pose.shape != (count, 7) or not np.isfinite(object_pose).all():
            raise ValueError("Captured object trajectory must be finite (frames, 7)")
        wrist_p, wrist_q, tip_p, tip_q = captured_mano_targets(capture, count)
    wrist_r = Rotation.from_quat(wrist_q)
    points, directions, poses = [], [], []
    for finger in FINGERS:
        world_p = wrist_p + wrist_r.apply(tip_p[finger])
        world_r = wrist_r * Rotation.from_quat(tip_q[finger])
        points.append(world_p)
        directions.append(world_r.apply(np.tile([0., 0., 1.], (count, 1))))
        poses.append(np.concatenate((world_p, world_r.as_quat()[:, [3, 0, 1, 2]]), axis=1))
    points = np.stack(points, axis=1)
    directions = np.stack(directions, axis=1)
    return dict(
        mano_wrist_position=wrist_p.astype(np.float32),
        mano_wrist_quaternion_xyzw=wrist_q.astype(np.float32),
        mano_tip_positions_local=np.stack([tip_p[f] for f in FINGERS], axis=1).astype(np.float32),
        target_points_world=points.astype(np.float32),
        target_directions_world=directions.astype(np.float32),
        object_pose_wxyz=object_pose,
        fingertip_pose_wxyz=np.stack(poses[:2], axis=1).astype(np.float32),
        frame_ids=np.arange(count, dtype=np.int64),
        source_mano_rollout=np.asarray(str(capture_path)),
        source_mano_rollout_sha256=np.asarray(hashlib.sha256(capture_path.read_bytes()).hexdigest()),
    )
