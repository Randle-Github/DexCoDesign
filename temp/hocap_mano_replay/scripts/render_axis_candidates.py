#!/usr/bin/env python3
"""Render all six signed 90-degree corrections from the first replay mapping."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "replay_mujoco", SCRIPT_DIR / "replay_mujoco.py"
)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

TILE_WIDTH = 480
TILE_HEIGHT = 360
FPS = 15
STRIDE = 2

FINGER_NAMES = (
    "left_j_index1y", "left_j_index1z", "left_j_index2", "left_j_index3",
    "left_j_middle1y", "left_j_middle1z", "left_j_middle2", "left_j_middle3",
    "left_j_pinky1y", "left_j_pinky1z", "left_j_pinky2", "left_j_pinky3",
    "left_j_ring1y", "left_j_ring1z", "left_j_ring2", "left_j_ring3",
    "left_j_thumb1x", "left_j_thumb1y", "left_j_thumb1z",
    "left_j_thumb2y", "left_j_thumb2z", "left_j_thumb3",
)


def main() -> None:
    replay.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    scene_xml = replay.ARTIFACT_ROOT / "hocap_mano_axis_candidates_scene.xml"
    output = replay.ARTIFACT_ROOT / "hocap_mano_axis_90_candidates.mp4"
    preview = replay.ARTIFACT_ROOT / "hocap_mano_axis_90_candidates_preview.png"
    replay.make_scene_xml(scene_xml)

    mano_pose = np.load(replay.SEQUENCE_ROOT / "mano_pose_left.npy")
    object_pose = np.load(replay.SEQUENCE_ROOT / "object_pose_G04_1.npy")
    finger_q, _ = replay.reduced_hand_trajectory(mano_pose, object_pose)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    replay.configure_colors(model)
    finger_qpos = [model.joint(name).qposadr[0] for name in FINGER_NAMES]

    axes = np.eye(3)
    candidates = []
    for axis_index, axis_name in enumerate(("X", "Y", "Z")):
        for sign, sign_name in ((1.0, "+"), (-1.0, "-")):
            local_delta = Rotation.from_rotvec(
                sign * (np.pi / 2.0) * axes[axis_index]
            )
            candidates.append(
                (
                    f"{chr(65 + len(candidates))}: {sign_name}90 local {axis_name}",
                    replay.FIRST_VERSION_LOCAL_HAND_TO_MANO * local_delta,
                )
            )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.82
    camera.azimuth = 132
    camera.elevation = -28
    renderer = mujoco.Renderer(
        model, height=TILE_HEIGHT, width=TILE_WIDTH
    )

    canvas_width = TILE_WIDTH * 3
    canvas_height = TILE_HEIGHT * 2
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{canvas_width}x{canvas_height}",
            "-framerate", str(FPS), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p", str(output),
        ],
        stdin=subprocess.PIPE,
    )

    frame_ids = list(range(0, len(mano_pose), STRIDE))
    preview_index = len(frame_ids) // 2
    for output_index, frame_id in enumerate(frame_ids):
        pose = mano_pose[frame_id]
        root_mano = Rotation.from_rotvec(pose[:3])
        obj = object_pose[frame_id]
        tiles = []
        for label, mapping in candidates:
            replay.set_root_pose(
                model,
                data,
                pose[48:51] + replay.HOCAP_MANO_ROOT_OFFSET_WORLD,
                root_mano * mapping,
            )
            data.qpos[finger_qpos] = finger_q[frame_id]
            data.mocap_pos[0] = obj[4:]
            data.mocap_quat[0] = (obj[3], obj[0], obj[1], obj[2])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            tile = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 180, 25), fill=(18, 22, 28))
            draw.text((7, 6), label, fill=(255, 255, 255))
            tiles.append(np.asarray(tile))

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, 3)
            canvas[
                row * TILE_HEIGHT : (row + 1) * TILE_HEIGHT,
                column * TILE_WIDTH : (column + 1) * TILE_WIDTH,
            ] = tile
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.write(canvas.tobytes())
        if output_index == preview_index:
            Image.fromarray(canvas).save(preview)

    assert ffmpeg.stdin is not None
    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    print(output)


if __name__ == "__main__":
    main()
