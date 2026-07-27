#!/usr/bin/env python3
"""Retarget a captured successful MANO rollout to every robot hand.

The source targets are obtained by forward kinematics of the saved MANO
simulator states. Each target hand is solved independently for at most the
requested number of trajectory frames. Only finite, geometrically valid IK
solutions are rendered.
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
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
REGISTRY = DIRECT_ROOT / "registry.json"
DEFAULT_CAPTURE = (
    REPO_ROOT
    / "artifacts"
    / "isaaclab_mano_residual"
    / "success_capture_fixed_clamp_reset0"
    / "successful_rollout.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "all_hands_success_action_retarget"
)

sys.path.insert(0, str(SCRIPT_ROOT))
from retarget_all_hands import (  # noqa: E402
    FINGERS,
    MANO_TIPS,
    TIP_LINKS,
    assign_wrist_pose,
    build_hand,
    configure_colors,
    lowest_common_ancestor,
    make_scene,
    mano_fingertip_offsets,
    relative_tip_poses,
    rotation_matrix,
    solve_frame,
)


def captured_mano_targets(
    captured: np.lib.npyio.NpzFile,
    frame_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    metadata = json.loads(str(captured["metadata_json"]))
    scene = make_scene("mano")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos_ids = [model.joint(name).qposadr[0] for name in metadata["joint_names"]]
    tip_ids = {finger: model.body(link).id for finger, link in MANO_TIPS.items()}
    tip_offsets = mano_fingertip_offsets()
    wrist_id = lowest_common_ancestor(model, list(tip_ids.values()))

    wrist_position = np.empty((frame_count, 3), dtype=np.float64)
    wrist_quaternion = np.empty((frame_count, 4), dtype=np.float64)
    tip_position = {
        finger: np.empty((frame_count, 3), dtype=np.float64)
        for finger in FINGERS
    }
    tip_quaternion = {
        finger: np.empty((frame_count, 4), dtype=np.float64)
        for finger in FINGERS
    }
    for frame_index, q in enumerate(captured["hand_q"][:frame_count]):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        wrist_position[frame_index] = data.xpos[wrist_id]
        wrist_rotation = Rotation.from_matrix(rotation_matrix(data, wrist_id))
        wrist_quaternion[frame_index] = wrist_rotation.as_quat()
        positions, rotations = relative_tip_poses(
            model, data, wrist_id, tip_ids, tip_offsets
        )
        for finger in FINGERS:
            tip_position[finger][frame_index] = positions[finger]
            tip_quaternion[finger][frame_index] = rotations[finger].as_quat()
    return wrist_position, wrist_quaternion, tip_position, tip_quaternion


def render_hand_video(
    hand,
    object_pose: np.ndarray,
    qpos: np.ndarray,
    wrist_position: np.ndarray,
    wrist_quaternion: np.ndarray,
    output_path: Path,
    preview_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    configure_colors(hand.model, (0.16, 0.58, 0.82, 1.0))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.72
    camera.azimuth = 132
    camera.elevation = -25
    renderer = mujoco.Renderer(hand.model, height=height, width=width)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    preview_index = len(qpos) // 2
    for frame_index in range(len(qpos)):
        hand.data.qpos[hand.qpos_ids] = qpos[frame_index]
        assign_wrist_pose(
            hand,
            wrist_position[frame_index],
            Rotation.from_quat(wrist_quaternion[frame_index]),
        )
        hand.data.mocap_pos[hand.object_mocap_id] = object_pose[frame_index, :3]
        hand.data.mocap_quat[hand.object_mocap_id] = object_pose[frame_index, 3:7]
        mujoco.mj_forward(hand.model, hand.data)
        renderer.update_scene(hand.data, camera=camera)
        pixels = renderer.render()
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.write(pixels.tobytes())
        if frame_index == preview_index:
            Image.fromarray(pixels).save(preview_path)
    assert ffmpeg.stdin is not None
    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--ik-iterations", type=int, default=14)
    parser.add_argument("--max-mean-tip-error-m", type=float, default=0.03)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--solve-only",
        action="store_true",
        help="Write retargeted joint trajectories without rendering direct-state videos.",
    )
    args = parser.parse_args()

    captured = np.load(args.capture)
    source_metadata = json.loads(str(captured["metadata_json"]))
    frame_count = min(args.max_frames, len(captured["hand_q"]))
    object_pose = captured["object_pose_wxyz"][:frame_count].astype(np.float64)
    (
        source_wrist_position,
        source_wrist_quaternion,
        source_tip_position,
        source_tip_quaternion,
    ) = captured_mano_targets(captured, frame_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hand_ids = [hand_id for hand_id in registry["hands"] if hand_id != "mano"]
    report = {
        "source_capture": str(args.capture),
        "source_success_metadata": source_metadata,
        "frames_attempted": frame_count,
        "ik_iterations_per_frame": args.ik_iterations,
        "success_threshold_mean_tip_position_m": args.max_mean_tip_error_m,
        "hands": {},
    }

    for hand_index, hand_id in enumerate(hand_ids, 1):
        hand_dir = args.output_dir / hand_id
        hand_dir.mkdir(parents=True, exist_ok=True)
        try:
            hand = build_hand(
                hand_id,
                registry["hands"][hand_id]["display_name"],
                {finger: source_tip_position[finger][0] for finger in FINGERS},
                {finger: source_tip_quaternion[finger][0] for finger in FINGERS},
            )
            q_trajectory = []
            solved_wrist_position = []
            solved_wrist_quaternion = []
            position_errors = []
            orientation_errors = []
            wrist_errors = []
            wrist_position = source_wrist_position[0].copy()
            wrist_rotation = (
                Rotation.from_quat(source_wrist_quaternion[0])
                * hand.map_rotation.inv()
            )
            for frame_index in range(frame_count):
                (
                    position_error,
                    orientation_error,
                    wrist_error,
                    wrist_position,
                    wrist_rotation,
                ) = solve_frame(
                    hand,
                    source_wrist_position[frame_index],
                    Rotation.from_quat(source_wrist_quaternion[frame_index]),
                    {
                        finger: source_tip_position[finger][frame_index]
                        for finger in FINGERS
                    },
                    {
                        finger: source_tip_quaternion[finger][frame_index]
                        for finger in FINGERS
                    },
                    args.ik_iterations,
                    wrist_position,
                    wrist_rotation,
                )
                values = np.concatenate(
                    (
                        hand.data.qpos[hand.qpos_ids],
                        wrist_position,
                        wrist_rotation.as_quat(),
                        [position_error, orientation_error, wrist_error],
                    )
                )
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"non-finite IK result at frame {frame_index}"
                    )
                q_trajectory.append(hand.data.qpos[hand.qpos_ids].copy())
                solved_wrist_position.append(wrist_position.copy())
                solved_wrist_quaternion.append(wrist_rotation.as_quat())
                position_errors.append(position_error)
                orientation_errors.append(orientation_error)
                wrist_errors.append(wrist_error)

            q_trajectory = np.asarray(q_trajectory)
            solved_wrist_position = np.asarray(solved_wrist_position)
            solved_wrist_quaternion = np.asarray(solved_wrist_quaternion)
            mean_position_error = float(np.mean(position_errors))
            success = mean_position_error <= args.max_mean_tip_error_m
            result = {
                "status": "success" if success else "failed",
                "frames_solved": frame_count,
                "fingers": list(hand.tip_ids),
                "ik_dofs": int(len(hand.dof_ids)),
                "mean_tip_position_error_mm": 1000.0 * mean_position_error,
                "max_tip_position_error_mm": 1000.0
                * float(np.max(position_errors)),
                "mean_tip_orientation_error_deg": float(
                    np.rad2deg(np.mean(orientation_errors))
                ),
                "mean_wrist_orientation_error_deg": float(
                    np.rad2deg(np.mean(wrist_errors))
                ),
            }
            if not success:
                result["failure_reason"] = (
                    f"mean tip error {mean_position_error:.6f} m exceeds "
                    f"{args.max_mean_tip_error_m:.6f} m"
                )
                report["hands"][hand_id] = result
                print(
                    f"[{hand_index:02d}/{len(hand_ids)}] {hand_id}: failed "
                    f"({result['failure_reason']})",
                    flush=True,
                )
                continue

            cache_path = hand_dir / "retargeted_trajectory.npz"
            video_path = hand_dir / f"{hand_id}_successful_retarget.mp4"
            preview_path = hand_dir / f"{hand_id}_preview.png"
            np.savez_compressed(
                cache_path,
                frame_ids=np.arange(frame_count),
                qpos_ids=hand.qpos_ids,
                qpos=q_trajectory,
                wrist_position=solved_wrist_position,
                wrist_quaternion_xyzw=solved_wrist_quaternion,
                object_pose_wxyz=object_pose,
                mean_tip_position_error_m=np.asarray(position_errors),
                mean_tip_orientation_error_rad=np.asarray(orientation_errors),
            )
            result["trajectory"] = str(cache_path)
            if not args.solve_only:
                render_hand_video(
                    hand,
                    object_pose,
                    q_trajectory,
                    solved_wrist_position,
                    solved_wrist_quaternion,
                    video_path,
                    preview_path,
                    args.fps,
                    args.width,
                    args.height,
                )
                result.update(
                    {
                        "video": str(video_path),
                        "preview": str(preview_path),
                    }
                )
            report["hands"][hand_id] = result
            print(
                f"[{hand_index:02d}/{len(hand_ids)}] {hand_id}: success, "
                f"{result['mean_tip_position_error_mm']:.1f} mm",
                flush=True,
            )
        except Exception as error:
            report["hands"][hand_id] = {
                "status": "failed",
                "frames_solved": 0,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
            print(
                f"[{hand_index:02d}/{len(hand_ids)}] {hand_id}: failed "
                f"({type(error).__name__}: {error})",
                flush=True,
            )

    successful = [
        hand_id
        for hand_id, result in report["hands"].items()
        if result["status"] == "success"
    ]
    failed = [
        hand_id
        for hand_id, result in report["hands"].items()
        if result["status"] == "failed"
    ]
    report["summary"] = {
        "success_count": len(successful),
        "failure_count": len(failed),
        "successful_hands": successful,
        "failed_hands": failed,
    }
    report_path = args.output_dir / "retarget_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
