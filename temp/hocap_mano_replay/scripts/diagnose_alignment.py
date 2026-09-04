#!/usr/bin/env python3
"""Render all proper signed-axis mappings for MANO-to-local hand alignment."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "replay_mujoco", SCRIPT_ROOT / "replay_mujoco.py"
)
assert spec is not None and spec.loader is not None
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def proper_axis_mappings():
    axes = np.eye(3)
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.column_stack(
                [signs[i] * axes[:, permutation[i]] for i in range(3)]
            )
            if np.linalg.det(matrix) > 0.5:
                yield permutation, signs, matrix


def main() -> None:
    replay.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    scene_xml = replay.ARTIFACT_ROOT / "hocap_mano_replay_scene.xml"
    replay.make_scene_xml(scene_xml)

    mano_pose = np.load(replay.SEQUENCE_ROOT / "mano_pose_left.npy")
    object_pose = np.load(replay.SEQUENCE_ROOT / "object_pose_G04_1.npy")
    finger_q, distances = replay.reduced_hand_trajectory(mano_pose, object_pose)
    frame_id = int(np.argmin(np.abs(distances - 0.025)))

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    replay.configure_colors(model)
    finger_names = (
        "left_j_index1y", "left_j_index1z", "left_j_index2", "left_j_index3",
        "left_j_middle1y", "left_j_middle1z", "left_j_middle2", "left_j_middle3",
        "left_j_pinky1y", "left_j_pinky1z", "left_j_pinky2", "left_j_pinky3",
        "left_j_ring1y", "left_j_ring1z", "left_j_ring2", "left_j_ring3",
        "left_j_thumb1x", "left_j_thumb1y", "left_j_thumb1z",
        "left_j_thumb2y", "left_j_thumb2z", "left_j_thumb3",
    )
    finger_qpos = [model.joint(name).qposadr[0] for name in finger_names]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.68
    camera.azimuth = 132
    camera.elevation = -24
    renderer = mujoco.Renderer(model, height=360, width=480)

    pose = mano_pose[frame_id]
    root_mano = Rotation.from_rotvec(pose[:3])
    obj = object_pose[frame_id]
    tiles = []
    axis_names = ("X", "Y", "Z")
    for permutation, signs, matrix in proper_axis_mappings():
        rotation = root_mano * Rotation.from_matrix(matrix)
        replay.set_root_pose(
            model,
            data,
            pose[48:51] + replay.HOCAP_MANO_ROOT_OFFSET_WORLD,
            rotation,
        )
        data.qpos[finger_qpos] = finger_q[frame_id]
        data.mocap_pos[0] = obj[4:]
        data.mocap_quat[0] = (obj[3], obj[0], obj[1], obj[2])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render())
        label = " ".join(
            f"L{axis_names[i]}→{'+' if signs[i] > 0 else '-'}"
            f"{axis_names[permutation[i]]}"
            for i in range(3)
        )
        ImageDraw.Draw(image).rectangle((0, 0, 480, 25), fill=(20, 25, 32))
        ImageDraw.Draw(image).text((7, 6), label, fill=(255, 255, 255))
        tiles.append(image)
    renderer.close()

    montage = Image.new("RGB", (480 * 4, 360 * 6), (20, 25, 32))
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % 4) * 480, (index // 4) * 360))
    output = replay.ARTIFACT_ROOT / "mano_local_axis_candidates.png"
    montage.save(output)
    print(f"frame={frame_id} distance={distances[frame_id]:.6f}")
    print(output)


if __name__ == "__main__":
    main()
