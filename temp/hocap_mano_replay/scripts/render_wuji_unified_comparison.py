#!/usr/bin/env python3
"""Render WUJI rollout states with one MuJoCo camera and visual style."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--source-diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trajectory = np.load(args.trajectory)
    if "hand_q" in trajectory:
        hand_q = trajectory["hand_q"].astype(np.float64)
        object_pose = trajectory["object_pose_wxyz"].astype(np.float64)
        metadata = json.loads(str(trajectory["metadata_json"]))
        all_joint_names = [name.removeprefix("finger__") for name in metadata["joint_names"]]
        source_format = False
    else:
        diagnostics = np.load(args.source_diagnostics)
        object_pose = diagnostics["object_pose_wxyz"].astype(np.float64)
        source_format = True

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    wrapper_id = model.body("retarget_wrapper").id
    wrapper_mocap = int(model.body_mocapid[wrapper_id])
    object_id = model.body("hocap_object_body").id
    object_mocap = int(model.body_mocapid[object_id])
    object_qadr = None
    if object_mocap < 0:
        object_qadr = int(model.jnt_qposadr[model.joint("hocap_object_free").id])

    if source_format:
        qpos = trajectory["qpos"].astype(np.float64)
        qpos_ids = trajectory["qpos_ids"].astype(int)
        wrist_position = trajectory["wrist_position"].astype(np.float64)
        wrist_quaternion = trajectory["wrist_quaternion_xyzw"].astype(np.float64)
        frame_count = len(qpos)
    else:
        root_is_articulated = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
            for name in all_joint_names[:6]
        )
        driven_joint_names = all_joint_names if root_is_articulated else all_joint_names[6:]
        qpos_ids = np.asarray([model.joint(name).qposadr[0] for name in driven_joint_names])
        frame_count = len(hand_q)

    # Force identical colors across source, morphology, and policy scenes.
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "hocap_object_geom":
            model.geom_rgba[geom_id] = (0.96, 0.48, 0.14, 1.0)
        elif name == "table":
            model.geom_rgba[geom_id] = (0.72, 0.74, 0.78, 1.0)
        elif model.geom_group[geom_id] == 1:
            model.geom_rgba[geom_id] = (0.16, 0.58, 0.82, 1.0)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.72
    camera.azimuth = 132
    camera.elevation = -25
    width, height, fps = 960, 720, 30
    renderer = mujoco.Renderer(model, height=height, width=width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    for frame in range(frame_count):
        if source_format:
            data.qpos[qpos_ids] = qpos[frame]
            data.mocap_pos[wrapper_mocap] = wrist_position[frame]
            data.mocap_quat[wrapper_mocap] = np.roll(wrist_quaternion[frame], 1)
        else:
            q = hand_q[frame]
            if root_is_articulated:
                data.qpos[qpos_ids] = q
            else:
                data.qpos[qpos_ids] = q[6:]
                data.mocap_pos[wrapper_mocap] = q[:3]
                data.mocap_quat[wrapper_mocap] = np.roll(
                    Rotation.from_euler("XYZ", q[3:6]).as_quat(), 1
                )
        pose = object_pose[frame]
        if object_mocap >= 0:
            data.mocap_pos[object_mocap] = pose[:3]
            data.mocap_quat[object_mocap] = pose[3:7]
        else:
            data.qpos[object_qadr : object_qadr + 7] = pose
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        encoder.stdin.write(renderer.render().tobytes())

    encoder.stdin.close()
    code = encoder.wait()
    renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited with status {code}")
    print(f"rendered {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
