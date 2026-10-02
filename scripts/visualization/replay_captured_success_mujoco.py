#!/usr/bin/env python3
"""Render saved successful simulator states without evaluating a policy."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from replay_mujoco import configure_colors, make_scene_xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    captured = np.load(args.trajectory)
    metadata = json.loads(str(captured["metadata_json"]))
    hand_q = captured["hand_q"]
    object_pose = captured["object_pose_wxyz"]
    if hand_q.shape != (446, 28) or object_pose.shape != (446, 7):
        raise ValueError(
            f"Expected 446 captured states, got {hand_q.shape}, {object_pose.shape}"
        )
    if metadata["final_phase"] != 445:
        raise ValueError(f"Captured trajectory is not successful: {metadata}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene_xml = args.output.with_suffix(".scene.xml")
    make_scene_xml(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    configure_colors(model)
    qpos_addresses = [
        model.joint(name).qposadr[0] for name in metadata["joint_names"]
    ]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.72
    camera.azimuth = 132
    camera.elevation = -25

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
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
            f"{args.width}x{args.height}",
            "-framerate",
            str(args.fps),
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
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    preview_index = len(hand_q) // 2
    for frame_index, (q, pose) in enumerate(zip(hand_q, object_pose)):
        data.qpos[qpos_addresses] = q
        data.mocap_pos[0] = pose[:3]
        data.mocap_quat[0] = pose[3:7]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.write(pixels.tobytes())
        if frame_index == preview_index:
            Image.fromarray(pixels).save(args.preview)

    assert ffmpeg.stdin is not None
    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    print(
        json.dumps(
            {
                "frames": len(hand_q),
                "fps": args.fps,
                "video": str(args.output),
                "render_semantics": (
                    "direct q/object-pose replay of captured successful env; "
                    "no policy evaluation"
                ),
                **metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
