#!/usr/bin/env python3
"""Retarget a captured MANO left hand to the 5-DOF SUPR female left foot.

The five MANO digits directly drive the five toes.  Root orientation is not
copied as a raw wrist quaternion: a palm/sole frame is fitted from the 21
captured keypoints, so hand-forward maps to toe-forward and the palm normal
maps to the sole normal.  The fitted wrist-to-sole transform contains the
anatomical approximately 90 degree wrist/ankle bend.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
SEQUENCE_ROOT = (
    EXPERIMENT_ROOT / "data" / "subset" / "subject_7" / "20231022_192832"
)
OUTPUT_ROOT = EXPERIMENT_ROOT / "artifacts" / "supr_female_foot_retarget"

sys.path.insert(0, str(SCRIPT_DIR))
from replay_mujoco import LOCAL_HAND_TO_MANO  # noqa: E402
from retarget_all_hands import (  # noqa: E402
    FINGERS,
    annotate,
    assign_wrist_pose,
    build_hand,
    configure_colors,
    reference_trajectory,
    relative_tip_poses,
)


# MANO digit -> direct-drive foot joint. Positive joint motion bends the toe
# around +Y from toe-forward (+X) toward the sole (-Z).
TOE_JOINTS = {
    "thumb": "left_big_toe",
    "index": "left_toe_2",
    "middle": "left_toe_3",
    "ring": "left_toe_4",
    "pinky": "left_toe_5",
}
HOCAP_DIGIT_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TOE_OPEN_RAD = {
    "thumb": 0.04,
    "index": 0.04,
    "middle": 0.04,
    "ring": 0.04,
    "pinky": 0.04,
}
TOE_CLOSED_RAD = {
    "thumb": 0.64,
    "index": 0.72,
    "middle": 0.72,
    "ring": 0.70,
    "pinky": 0.66,
}
# The MANO wrist body's local back normal and the SUPR foot-base dorsum normal
# differ by one anatomical quarter-turn about their already-correct common
# forward axis. Keep this explicit: it preserves toe/finger forward while
# making the hand back and foot dorsum point the same way.
DORSUM_ALIGNMENT = Rotation.from_euler("x", -np.pi / 2.0)
# Orientation is expressed as an equivalent positional residual. At 2 mm/rad
# it is deliberately much weaker than centimeter-scale fingertip position.
POSITION_WEIGHT = 1.0
ORIENTATION_EQUIVALENT_LENGTH_M = 0.002
TEMPORAL_EQUIVALENT_LENGTH_M = 0.001


def palm_sole_frames(points: np.ndarray) -> list[Rotation]:
    """Fit frames with +X forward, +Y toward pinky, and -Z toward the palm."""
    rotations: list[Rotation] = []
    for frame in points:
        forward = frame[[5, 9, 13, 17]].mean(axis=0) - frame[0]
        forward /= np.linalg.norm(forward) + 1e-12
        little_side = frame[17] - frame[5]
        little_side -= forward * np.dot(little_side, forward)
        little_side /= np.linalg.norm(little_side) + 1e-12
        dorsum = np.cross(forward, little_side)
        dorsum /= np.linalg.norm(dorsum) + 1e-12
        little_side = np.cross(dorsum, forward)
        rotations.append(
            Rotation.from_matrix(np.column_stack((forward, little_side, dorsum)))
        )
    return rotations


def digit_bend(points: np.ndarray, joint_ids: tuple[int, ...]) -> np.ndarray:
    """Sum PIP/DIP direction changes; zero means a straight digit."""
    segments = np.diff(points[:, joint_ids], axis=1)
    segments /= np.linalg.norm(segments, axis=2, keepdims=True) + 1e-12
    cosines = np.sum(segments[:, :-1] * segments[:, 1:], axis=2)
    return np.arccos(np.clip(cosines, -1.0, 1.0)).sum(axis=1)


def direct_toe_trajectory(
    points: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Map each human digit's bend monotonically into its toe joint range."""
    trajectory: dict[str, np.ndarray] = {}
    calibration: dict[str, dict[str, float]] = {}
    for finger in FINGERS:
        raw = digit_bend(points, HOCAP_DIGIT_CHAINS[finger])
        open_bend, closed_bend = np.percentile(raw, (5.0, 95.0))
        denominator = max(float(closed_bend - open_bend), 1e-4)
        activation = np.clip((raw - open_bend) / denominator, 0.0, 1.0)
        activation = gaussian_filter1d(activation, sigma=1.5, mode="nearest")
        q = TOE_OPEN_RAD[finger] + activation * (
            TOE_CLOSED_RAD[finger] - TOE_OPEN_RAD[finger]
        )
        trajectory[finger] = q
        calibration[finger] = {
            "source_open_bend_rad_p05": float(open_bend),
            "source_closed_bend_rad_p95": float(closed_bend),
            "target_open_rad": TOE_OPEN_RAD[finger],
            "target_closed_rad": TOE_CLOSED_RAD[finger],
        }
    return trajectory, calibration


def position_dominant_tip_ik(
    foot,
    points: np.ndarray,
    sole_frames: list[Rotation],
    orientation_seed: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
]:
    """Solve five scalar toe joints against physical distal mesh positions."""
    assign_wrist_pose(foot, np.zeros(3), Rotation.identity())
    for finger, joint_name in TOE_JOINTS.items():
        set_named_joint(
            foot.model, foot.data, joint_name, TOE_OPEN_RAD[finger]
        )
    mujoco.mj_forward(foot.model, foot.data)

    result = {
        finger: np.empty(len(points), dtype=np.float64)
        for finger in FINGERS
    }
    target_positions = {
        finger: np.empty((len(points), 3), dtype=np.float64)
        for finger in FINGERS
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for finger in FINGERS:
        joint_name = TOE_JOINTS[finger]
        joint_id = foot.model.joint(joint_name).id
        lower, upper = foot.model.jnt_range[joint_id]
        chain = HOCAP_DIGIT_CHAINS[finger]
        # A one-DOF toe can only follow the human fingertip after projecting
        # its MCP-to-tip vector into the shared forward/sole sagittal plane.
        # This is a Cartesian endpoint measurement, unlike the separate
        # low-weight terminal-orientation seed below.
        endpoint_pitch = []
        for frame_id, frame in enumerate(points):
            endpoint = sole_frames[frame_id].inv().apply(
                frame[chain[-1]] - frame[chain[0]]
            )
            endpoint_pitch.append(np.arctan2(-endpoint[2], endpoint[0]))
        endpoint_pitch = np.asarray(endpoint_pitch)
        open_pitch, closed_pitch = np.percentile(
            endpoint_pitch, (5.0, 95.0)
        )
        pitch_activation = np.clip(
            (endpoint_pitch - open_pitch)
            / max(float(closed_pitch - open_pitch), 1e-4),
            0.0,
            1.0,
        )
        pitch_activation = gaussian_filter1d(
            pitch_activation, sigma=1.5, mode="nearest"
        )
        position_q = TOE_OPEN_RAD[finger] + pitch_activation * (
            TOE_CLOSED_RAD[finger] - TOE_OPEN_RAD[finger]
        )
        targets = np.empty((len(points), 3), dtype=np.float64)
        for frame_id, q_value in enumerate(position_q):
            set_named_joint(
                foot.model, foot.data, joint_name, float(q_value)
            )
            mujoco.mj_forward(foot.model, foot.data)
            current, _ = relative_tip_poses(
                foot.model,
                foot.data,
                foot.wrist_id,
                {finger: foot.tip_ids[finger]},
                {finger: foot.tip_offsets[finger]},
            )
            targets[frame_id] = current[finger]
        target_positions[finger][:] = targets

        previous_q = TOE_OPEN_RAD[finger]
        errors = []
        for frame_id, target in enumerate(targets):
            orientation_q = float(orientation_seed[finger][frame_id])

            def objective(q_value: float) -> float:
                set_named_joint(
                    foot.model, foot.data, joint_name, q_value
                )
                mujoco.mj_forward(foot.model, foot.data)
                current, _ = relative_tip_poses(
                    foot.model,
                    foot.data,
                    foot.wrist_id,
                    {finger: foot.tip_ids[finger]},
                    {finger: foot.tip_offsets[finger]},
                )
                position_residual = current[finger] - target
                return (
                    POSITION_WEIGHT
                    * float(position_residual @ position_residual)
                    + ORIENTATION_EQUIVALENT_LENGTH_M**2
                    * (q_value - orientation_q) ** 2
                    + TEMPORAL_EQUIVALENT_LENGTH_M**2
                    * (q_value - previous_q) ** 2
                )

            solved = minimize_scalar(
                objective,
                bounds=(float(lower), float(upper)),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 32},
            )
            q_value = float(solved.x)
            result[finger][frame_id] = q_value
            previous_q = q_value
            set_named_joint(foot.model, foot.data, joint_name, q_value)
            mujoco.mj_forward(foot.model, foot.data)
            current, _ = relative_tip_poses(
                foot.model,
                foot.data,
                foot.wrist_id,
                {finger: foot.tip_ids[finger]},
                {finger: foot.tip_offsets[finger]},
            )
            errors.append(float(np.linalg.norm(current[finger] - target)))

        diagnostics[finger] = {
            "source_open_endpoint_pitch_rad_p05": float(open_pitch),
            "source_closed_endpoint_pitch_rad_p95": float(closed_pitch),
            "mean_tip_position_error_mm": 1000.0 * float(np.mean(errors)),
            "max_tip_position_error_mm": 1000.0 * float(np.max(errors)),
        }
    return result, target_positions, diagnostics


def hide_environment(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        if name in {"hocap_object_geom", "table"} or "collision" in name.lower():
            model.geom_rgba[geom_id, 3] = 0.0


def set_named_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    value: float,
) -> None:
    joint_id = model.joint(name).id
    data.qpos[int(model.jnt_qposadr[joint_id])] = value


def camera_for_pair() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Fixed wide camera: do not follow the wrist, otherwise the captured
    # translation would again look artificially stationary.
    camera.lookat[:] = (0.075, 0.03, 0.135)
    camera.distance = 0.56
    camera.azimuth = 90
    camera.elevation = -72
    return camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=540)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Render only this many output frames; zero renders all.",
    )
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    points = np.load(SEQUENCE_ROOT / "hand_joints_3d_left.npy").astype(
        np.float64
    )
    mano_pose = np.load(SEQUENCE_ROOT / "mano_pose_left.npy").astype(np.float64)
    reference = np.load(SEQUENCE_ROOT / "isaaclab_reference.npz")
    wrist_position, wrist_quaternion, ref_pos, ref_rot = reference_trajectory(
        mano_pose, points
    )
    root_positions = wrist_position - wrist_position[0]

    wrist_rotations = [
        Rotation.from_quat(quaternion) for quaternion in wrist_quaternion
    ]
    sole_frames = palm_sole_frames(points)
    wrist_to_sole = Rotation.from_quat(
        [
            (wrist.inv() * sole).as_quat()
            for wrist, sole in zip(wrist_rotations, sole_frames)
        ]
    ).mean()
    # Normalize global capture motion at frame zero. The human wrist keeps its
    # native local frame; the foot retains the anatomical 90-degree roll that
    # aligns its dorsum with the back of the MANO mesh.
    hand_root_rotations = [
        wrist * wrist_rotations[0].inv() for wrist in wrist_rotations
    ]
    foot_root_rotations = [
        rotation * DORSUM_ALIGNMENT for rotation in hand_root_rotations
    ]
    orientation_seed, bend_calibration = direct_toe_trajectory(points)

    first_ref_pos = {finger: ref_pos[finger][0] for finger in FINGERS}
    first_ref_rot = {finger: ref_rot[finger][0] for finger in FINGERS}
    mano = build_hand(
        "mano", "MANO Human Hand", first_ref_pos, first_ref_rot
    )
    foot = build_hand(
        "supr_female_foot",
        "SUPR Female Foot · 5-DOF Direct Drive",
        first_ref_pos,
        first_ref_rot,
    )
    toe_q, tip_targets, tip_ik_diagnostics = position_dominant_tip_ik(
        foot, points, sole_frames, orientation_seed
    )
    del tip_targets
    configure_colors(mano.model, (0.20, 0.60, 0.86, 1.0))
    configure_colors(foot.model, (0.86, 0.48, 0.38, 1.0))
    hide_environment(mano.model)
    hide_environment(foot.model)

    source_names = [str(name) for name in reference["joint_names"]]
    source_q = np.asarray(reference["hand_q"], dtype=np.float64)
    source_finger_columns = [
        index for index, name in enumerate(source_names)
        if name in {
            mano.model.joint(joint_id).name
            for joint_id in range(mano.model.njnt)
        }
        and name not in {
            "left_pos_x", "left_pos_y", "left_pos_z",
            "left_rot_x", "left_rot_y", "left_rot_z",
        }
    ]
    if len(source_finger_columns) != 22:
        raise RuntimeError(
            f"Expected 22 MANO finger channels, got {len(source_finger_columns)}"
        )

    frame_ids = np.arange(0, len(points), args.stride, dtype=int)
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not len(frame_ids):
        raise ValueError("No frames selected")

    renderers = [
        mujoco.Renderer(
            mano.model, height=args.tile_height, width=args.tile_width
        ),
        mujoco.Renderer(
            foot.model, height=args.tile_height, width=args.tile_width
        ),
    ]
    camera = camera_for_pair()
    output_video = OUTPUT_ROOT / "mano_left_foot_right.mp4"
    preview_path = OUTPUT_ROOT / "mano_left_foot_right_preview.png"
    trajectory_path = OUTPUT_ROOT / "supr_foot_retarget.npz"
    diagnostics_path = OUTPUT_ROOT / "retarget_diagnostics.json"
    width, height = 2 * args.tile_width, args.tile_height
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p", str(output_video),
        ],
        stdin=subprocess.PIPE,
    )

    foot_q_rows = []
    for output_frame, frame_id in enumerate(frame_ids):
        for column in source_finger_columns:
            set_named_joint(
                mano.model,
                mano.data,
                source_names[column],
                source_q[frame_id, column],
            )
        foot_row = []
        for finger, joint_name in TOE_JOINTS.items():
            value = float(toe_q[finger][frame_id])
            set_named_joint(foot.model, foot.data, joint_name, value)
            foot_row.append(value)
        foot_q_rows.append(foot_row)

        root_position = root_positions[frame_id]
        assign_wrist_pose(
            mano, root_position, hand_root_rotations[frame_id]
        )
        assign_wrist_pose(
            foot, root_position, foot_root_rotations[frame_id]
        )
        mujoco.mj_forward(mano.model, mano.data)
        mujoco.mj_forward(foot.model, foot.data)

        renderers[0].update_scene(mano.data, camera=camera)
        hand_tile = annotate(
            renderers[0].render(), "Human hand · MANO source"
        )
        renderers[1].update_scene(foot.data, camera=camera)
        foot_tile = annotate(
            renderers[1].render(),
            "SUPR foot · wrist position + orientation retarget",
        )
        tiles = [hand_tile, foot_tile]
        canvas = np.concatenate(tiles, axis=1)
        if output_frame == len(frame_ids) // 2:
            Image.fromarray(canvas).save(preview_path)
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.write(canvas.tobytes())
        if output_frame % 40 == 0:
            print(
                f"render {output_frame + 1}/{len(frame_ids)}",
                flush=True,
            )

    assert ffmpeg.stdin is not None
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    for renderer in renderers:
        renderer.close()

    foot_q_array = np.asarray(foot_q_rows, dtype=np.float32)
    np.savez_compressed(
        trajectory_path,
        frame_ids=frame_ids,
        joint_names=np.asarray(list(TOE_JOINTS.values())),
        qpos=foot_q_array,
        root_quaternion_xyzw=np.asarray(
            [foot_root_rotations[index].as_quat() for index in frame_ids],
            dtype=np.float32,
        ),
        root_position=np.asarray(root_positions[frame_ids], dtype=np.float32),
        fps=np.float32(args.fps),
    )
    relation_error = np.asarray(
        [
            (
                wrist_to_sole.inv()
                * wrist.inv()
                * sole
            ).magnitude()
            for wrist, sole in zip(wrist_rotations, sole_frames)
        ]
    )
    diagnostics = {
        "source": "HO-Cap official 21-point MANO left-hand capture",
        "target": "SUPR female left foot, direct-motor registry entry",
        "mapping": dict(zip(FINGERS, TOE_JOINTS.values())),
        "root_axes": {
            "+X": "human finger forward -> foot toe forward",
            "+Y": "human pinky side -> foot little-toe side",
            "-Z": "human palm direction -> foot sole direction",
        },
        "wrist_to_sole_xyz_degrees": wrist_to_sole.as_euler(
            "xyz", degrees=True
        ).tolist(),
        "applied_dorsum_alignment_xyz_degrees": DORSUM_ALIGNMENT.as_euler(
            "xyz", degrees=True
        ).tolist(),
        "nominal_anatomical_bend": "-90 degrees around wrist-local X",
        "fixed_transform_fit_error_degrees": {
            "median": float(np.rad2deg(np.median(relation_error))),
            "max": float(np.rad2deg(relation_error.max())),
        },
        "bend_calibration": bend_calibration,
        "tip_ik_weights": {
            "position": POSITION_WEIGHT,
            "orientation_equivalent_length_m_per_rad": (
                ORIENTATION_EQUIVALENT_LENGTH_M
            ),
            "temporal_equivalent_length_m_per_rad": (
                TEMPORAL_EQUIVALENT_LENGTH_M
            ),
        },
        "tip_position_ik": tip_ik_diagnostics,
        "rendered_frames": int(len(frame_ids)),
        "fps": args.fps,
        "duration_seconds": len(frame_ids) / args.fps,
        "wrist_translation": {
            "source": "MANO wrist position, first frame subtracted",
            "minimum_m": root_positions.min(axis=0).tolist(),
            "maximum_m": root_positions.max(axis=0).tolist(),
            "maximum_displacement_m": float(
                np.linalg.norm(root_positions, axis=1).max()
            ),
            "applied_identically_to": ["MANO hand", "SUPR foot"],
        },
        "toe_q_range_rad": {
            joint_name: [
                float(foot_q_array[:, index].min()),
                float(foot_q_array[:, index].max()),
            ]
            for index, joint_name in enumerate(TOE_JOINTS.values())
        },
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(output_video)
    print(preview_path)
    print(trajectory_path)
    print(diagnostics_path)


if __name__ == "__main__":
    main()
