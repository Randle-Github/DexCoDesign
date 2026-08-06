#!/usr/bin/env python3
"""Render captured WUJI morphology states without rerunning physics."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    capture = np.load(args.trajectory)
    metadata = json.loads(str(capture["metadata_json"]))
    hand_q = capture["hand_q"]
    object_pose = capture["object_pose_wxyz"]
    if not metadata["success"] or metadata["final_phase"] != 445:
        raise ValueError("trajectory is not a captured 445/445 success")

    candidate = args.candidate_root.resolve()
    source_xml = candidate / "generated_ik_scene_cache" / "wuji_physx_003527.xml"
    tree = ET.parse(source_xml)
    root = tree.getroot()
    mesh_root = candidate / "runtime/wuji_physx_003527/left"
    repaired_meshes = mesh_root / "repaired_meshes"
    root.find("compiler").set("meshdir", str(mesh_root))
    if repaired_meshes.is_dir():
        for mesh in root.findall("asset/mesh"):
            file_name = mesh.get("file", "")
            if file_name.startswith("meshes/part_") and file_name.endswith("_visual.obj"):
                mesh.set("file", file_name.replace("meshes/", "repaired_meshes/", 1))
    object_mesh = root.find("asset/mesh[@name='hocap_object']")
    object_mesh.set(
        "file",
        str(
            Path(__file__).resolve().parents[1]
            / "data/subset/models/G04_1/cleaned_mesh_2000.obj"
        ),
    )
    scene_xml = args.output.with_suffix(".scene.xml")
    tree.write(scene_xml, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    finger_names = [name.removeprefix("finger__") for name in metadata["joint_names"][6:]]
    qpos_addresses = [model.joint(name).qposadr[0] for name in finger_names]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.22, -0.10, 0.10)
    camera.distance = 0.62
    camera.azimuth = 132
    camera.elevation = -25
    width, height, fps = 960, 720, 30
    renderer = mujoco.Renderer(model, height=height, width=width)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    for q, pose in zip(hand_q, object_pose):
        data.qpos[qpos_addresses] = q[6:]
        data.mocap_pos[0] = q[:3]
        quat_xyzw = Rotation.from_euler("XYZ", q[3:6]).as_quat()
        data.mocap_quat[0] = np.roll(quat_xyzw, 1)
        data.mocap_pos[1] = pose[:3]
        data.mocap_quat[1] = pose[3:7]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        ffmpeg.stdin.write(renderer.render().tobytes())
    ffmpeg.stdin.close()
    code = ffmpeg.wait()
    renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    print(f"WUJI_CAPTURED_SUCCESS_MUJOCO_RENDERED frames={len(hand_q)} output={args.output}")


if __name__ == "__main__":
    main()
