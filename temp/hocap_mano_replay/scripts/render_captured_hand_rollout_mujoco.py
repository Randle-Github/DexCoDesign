#!/usr/bin/env python3
"""Render an exact captured Isaac rollout with the stable MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"

sys.path.insert(0, str(SCRIPT_ROOT))
from retarget_all_hands import configure_colors, make_scene  # noqa: E402


ROOT_JOINT_NAMES = (
    "root_pos_x",
    "root_pos_y",
    "root_pos_z",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
)


def sanitize(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not result or result[0].isdigit():
        result = f"n_{result}"
    return result


def unique_mapping(names: list[str], prefix: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for original in names:
        base = f"{prefix}{sanitize(original)}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        mapping[original] = candidate
        used.add(candidate)
    return mapping


def render(
    hand_id: str,
    trajectory_path: Path,
    output_dir: Path,
    fps: int,
    width: int,
    height: int,
) -> Path:
    capture = np.load(trajectory_path)
    hand_q = capture["hand_q"].astype(np.float64)
    object_pose = capture["object_pose_wxyz"].astype(np.float64)
    metadata = json.loads(str(capture["metadata_json"]))
    captured_joint_names = list(metadata["joint_names"])
    if object_pose.shape != (len(hand_q), 7):
        raise ValueError(
            f"Invalid captured rollout: q={hand_q.shape}, object={object_pose.shape}"
        )

    source_urdf = DIRECT_ROOT / hand_id / "left" / "hand.urdf"
    source_root = ET.parse(source_urdf).getroot()
    source_joint_names = [
        str(joint.get("name"))
        for joint in source_root.findall("joint")
        if joint.get("name")
    ]
    wrapped_to_source = {
        wrapped: original
        for original, wrapped in unique_mapping(
            source_joint_names, "finger__"
        ).items()
    }

    model = mujoco.MjModel.from_xml_path(str(make_scene(hand_id)))
    data = mujoco.MjData(model)
    configure_colors(model, (0.16, 0.58, 0.82, 1.0))
    model.geom_rgba[model.geom("hocap_object_geom").id] = (0.96, 0.48, 0.14, 1.0)
    model.geom_rgba[model.geom("table").id] = (0.72, 0.74, 0.78, 1.0)

    root_columns = [captured_joint_names.index(name) for name in ROOT_JOINT_NAMES]
    finger_columns: list[int] = []
    finger_qpos_ids: list[int] = []
    for column, wrapped_name in enumerate(captured_joint_names):
        if wrapped_name in ROOT_JOINT_NAMES:
            continue
        source_name = wrapped_to_source.get(wrapped_name)
        if source_name is None:
            raise KeyError(
                f"{hand_id}: cannot map captured joint {wrapped_name!r} to source URDF"
            )
        finger_columns.append(column)
        finger_qpos_ids.append(int(model.joint(source_name).qposadr[0]))

    wrapper_mocap_id = int(
        model.body_mocapid[model.body("retarget_wrapper").id]
    )
    object_mocap_id = int(
        model.body_mocapid[model.body("hocap_object_body").id]
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.72
    camera.azimuth = 132
    camera.elevation = -25
    renderer = mujoco.Renderer(model, height=height, width=width)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "farthest_rollout.mp4"
    temporary_path = output_dir / ".farthest_rollout.tmp.mp4"
    preview_path = output_dir / "farthest_rollout_preview.png"
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
            str(temporary_path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        mujoco.mj_resetData(model, data)
        for frame_index, q in enumerate(hand_q):
            root_q = q[root_columns]
            data.mocap_pos[wrapper_mocap_id] = root_q[:3]
            root_rotation = Rotation.from_euler("XYZ", root_q[3:6])
            data.mocap_quat[wrapper_mocap_id] = np.roll(
                root_rotation.as_quat(), 1
            )
            data.qpos[finger_qpos_ids] = q[finger_columns]
            data.mocap_pos[object_mocap_id] = object_pose[frame_index, :3]
            data.mocap_quat[object_mocap_id] = object_pose[frame_index, 3:7]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(pixels.tobytes())
            if frame_index == len(hand_q) // 2:
                Image.fromarray(pixels).save(preview_path)
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
    os.replace(temporary_path, output_path)
    summary = {
        **metadata,
        "frames": len(hand_q),
        "video": str(output_path),
        "trajectory": str(trajectory_path),
        "render_semantics": "exact captured hand/object states; MuJoCo viewer only",
    }
    (output_dir / "rollout_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"HAND_MUJOCO_ROLLOUT_RENDERED hand_id={hand_id} "
        f"frames={len(hand_q)} video={output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-id", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render(
        args.hand_id,
        args.trajectory,
        args.output_dir,
        args.fps,
        args.width,
        args.height,
    )


if __name__ == "__main__":
    main()
