#!/usr/bin/env python3
"""Render the 21-point-retargeted HO-Cap MANO/object replay in MuJoCo."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SEQUENCE_ROOT = (
    EXPERIMENT_ROOT / "data" / "subset" / "subject_7" / "20231022_192832"
)
OBJECT_MESH = (
    EXPERIMENT_ROOT
    / "data"
    / "subset"
    / "models"
    / "G04_1"
    / "cleaned_mesh_2000.obj"
)
HAND_URDF = (
    REPO_ROOT
    / "assets"
    / "robot_hands"
    / "direct_motor"
    / "mano"
    / "left"
    / "hand.urdf"
)
HAND_MESH_ROOT = HAND_URDF.parent
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_TIP_LINKS = {
    "thumb": "left_thumb3",
    "index": "left_index3",
    "middle": "left_middle3",
    "ring": "left_ring3",
    "pinky": "left_pinky3",
}
FINGER_JOINT_NAMES = (
    "left_j_index1y",
    "left_j_index1z",
    "left_j_index2",
    "left_j_index3",
    "left_j_middle1y",
    "left_j_middle1z",
    "left_j_middle2",
    "left_j_middle3",
    "left_j_pinky1y",
    "left_j_pinky1z",
    "left_j_pinky2",
    "left_j_pinky3",
    "left_j_ring1y",
    "left_j_ring1z",
    "left_j_ring2",
    "left_j_ring3",
    "left_j_thumb1x",
    "left_j_thumb1y",
    "left_j_thumb1z",
    "left_j_thumb2y",
    "left_j_thumb2z",
    "left_j_thumb3",
)

# The current direct-motor URDF canonicalizes the complete hand with this
# fixed transform before its six scalar root joints.
URDF_CANONICAL_ROTATION = Rotation.from_euler("y", -np.pi / 2.0)

# Original replay mapping retained for the signed-axis review.
FIRST_VERSION_LOCAL_HAND_TO_MANO = Rotation.from_matrix(
    np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
)
# User-reviewed candidate B: -90 degrees around the hand-local X axis.
LOCAL_HAND_TO_MANO = FIRST_VERSION_LOCAL_HAND_TO_MANO * Rotation.from_rotvec(
    np.array([-np.pi / 2.0, 0.0, 0.0])
)

# HO-Cap/manopth returns wrist = pose_translation + shaped MANO J0. This
# subject-specific constant was measured from official frame-0 and frame-222
# 3D wrist labels; both agree within 4e-8 m.
HOCAP_MANO_ROOT_OFFSET_WORLD = np.array(
    [-0.096753545, 0.006290510, 0.006127380], dtype=np.float64
)


def make_scene_xml(output_path: Path) -> None:
    # Import the current generated direct-motor URDF through MuJoCo, then add
    # only the temporary HO-Cap object and visualization environment.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hand_model = mujoco.MjModel.from_xml_path(str(HAND_URDF))
    mujoco.mj_saveLastXML(str(output_path), hand_model)
    tree = ET.parse(output_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(HAND_MESH_ROOT))

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    assert asset is not None and worldbody is not None

    # This replay is visual/kinematic only. The generated URDF contains a
    # second collision mesh for every visible link; importing both caused the
    # small joint-adjacent blobs and also enabled unwanted contacts.
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if "collision" in geom.get("name", "").lower():
                body.remove(geom)

    asset.append(
        ET.Element(
            "mesh",
            {
                "name": "hocap_object",
                "file": str(OBJECT_MESH),
            },
        )
    )
    object_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "hocap_object_body", "mocap": "true"},
    )
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": "hocap_object_geom",
            "type": "mesh",
            "mesh": "hocap_object",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "rgba": "0.96 0.48 0.14 1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table",
            "type": "plane",
            "size": "0.7 0.55 0.02",
            "pos": "0 0 -0.006",
            "rgba": "0.72 0.74 0.78 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {"offwidth": "1280", "offheight": "960"},
    )
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.75 0.75 0.75", "ambient": "0.45 0.45 0.45"},
    )
    ET.SubElement(
        visual,
        "rgba",
        {"haze": "0.92 0.94 0.97 1"},
    )

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def set_root_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    world_translation: np.ndarray,
    world_rotation: Rotation,
) -> None:
    """Set the URDF's six scalar root joints to an exact world SE(3) pose."""
    local_translation = URDF_CANONICAL_ROTATION.inv().apply(world_translation)
    local_euler_xyz = (
        URDF_CANONICAL_ROTATION.inv() * world_rotation
    ).as_euler("XYZ")
    for name, value in zip(
        (
            "left_pos_x",
            "left_pos_y",
            "left_pos_z",
            "left_rot_x",
            "left_rot_y",
            "left_rot_z",
        ),
        np.concatenate((local_translation, local_euler_xyz)),
    ):
        data.qpos[model.joint(name).qposadr[0]] = value


def surface_distance(
    wrist_positions: np.ndarray,
    object_poses: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(
        trimesh.load(OBJECT_MESH, process=False).vertices, dtype=np.float32
    )
    distances = np.empty(len(wrist_positions), dtype=np.float32)
    for frame_id, (wrist, pose) in enumerate(zip(wrist_positions, object_poses)):
        world_vertices = Rotation.from_quat(pose[:4]).apply(vertices) + pose[4:]
        distances[frame_id] = np.linalg.norm(
            world_vertices - wrist[None, :], axis=1
        ).min()
    return distances


def reduced_hand_trajectory(
    mano_pose: np.ndarray,
    object_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create 22 reduced hand joints from interaction and MANO PCA signals."""
    pca = mano_pose[:, 3:48]
    med = np.median(pca, axis=0)
    scale = np.percentile(np.abs(pca - med), 80, axis=0)
    normalized = np.tanh((pca - med) / np.maximum(scale, 1e-4))

    distances = surface_distance(
        mano_pose[:, 48:51] + HOCAP_MANO_ROOT_OFFSET_WORLD,
        object_pose,
    )
    proximity = np.clip((0.115 - distances) / 0.10, 0.0, 1.0)
    proximity = gaussian_filter1d(proximity, sigma=3.0)

    q = np.zeros((len(mano_pose), 22), dtype=np.float64)
    # index, middle, pinky, ring: abduction + MCP/PIP/DIP flexion.
    for finger_id, start in enumerate((0, 4, 8, 12)):
        variation = 0.10 * normalized[:, finger_id]
        curl = np.clip(0.08 + 0.82 * proximity + variation, 0.02, 0.98)
        q[:, start] = 0.12 * normalized[:, 5 + finger_id]
        q[:, start + 1] = 1.25 * curl
        q[:, start + 2] = 1.45 * curl
        q[:, start + 3] = 1.10 * curl

    thumb = np.clip(
        0.10 + 0.78 * proximity + 0.10 * normalized[:, 4], 0.02, 0.95
    )
    q[:, 16] = 0.72 * thumb
    q[:, 17] = 0.48 * thumb
    q[:, 18] = -0.55 * thumb
    q[:, 19] = 0.10 * normalized[:, 10]
    q[:, 20] = 1.20 * thumb
    q[:, 21] = 1.00 * thumb
    return q, distances


def mano_fingertip_offsets() -> dict[str, np.ndarray]:
    """Return the physical distal point of each terminal MANO mesh.

    The terminal meshes extend along their local +X axes.  Averaging the
    vertices on the final 0.5 mm cap avoids treating the distal-link body
    origin (the DIP joint center) as the fingertip.
    """
    offsets = {}
    mesh_root = HAND_URDF.parent / "meshes"
    for finger in FINGERS:
        mesh = trimesh.load(
            mesh_root / f"left_{finger}3_visual.obj",
            process=False,
            force="mesh",
        )
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        distal_x = float(vertices[:, 0].max())
        distal_cap = vertices[vertices[:, 0] >= distal_x - 5e-4]
        offset = distal_cap.mean(axis=0)
        offset[0] = distal_x
        offsets[finger] = offset
    return offsets


def mano_fingertip_world_positions(
    data: mujoco.MjData,
    tip_ids: dict[str, int],
    tip_offsets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    positions = {}
    for finger, body_id in tip_ids.items():
        body_rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(
            3, 3
        )
        positions[finger] = (
            np.asarray(data.xpos[body_id], dtype=np.float64)
            + body_rotation @ tip_offsets[finger]
        )
    return positions


def configure_colors(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "collision" in name.lower():
            model.geom_rgba[geom_id, 3] = 0.0
        elif name not in {"hocap_object_geom", "table"}:
            model.geom_rgba[geom_id] = (0.16, 0.58, 0.82, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    scene_xml = ARTIFACT_ROOT / "hocap_mano_replay_scene.xml"
    output_video = ARTIFACT_ROOT / "hocap_mano_replay.mp4"
    preview_path = ARTIFACT_ROOT / "hocap_mano_replay_preview.png"
    diagnostics_path = ARTIFACT_ROOT / "replay_diagnostics.json"
    make_scene_xml(scene_xml)

    reference = np.load(SEQUENCE_ROOT / "isaaclab_reference.npz")
    hand_q = reference["hand_q"]
    hand_joints_world = np.load(
        SEQUENCE_ROOT / "hand_joints_3d_left.npy"
    )
    object_pose = np.load(SEQUENCE_ROOT / "object_pose_G04_1.npy")
    distances = surface_distance(
        hand_joints_world[:, 0],
        object_pose,
    )

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    configure_colors(model)

    joint_qpos = [
        model.joint(name).qposadr[0]
        for name in (
            "left_pos_x",
            "left_pos_y",
            "left_pos_z",
            "left_rot_x",
            "left_rot_y",
            "left_rot_z",
            "left_j_index1y",
            "left_j_index1z",
            "left_j_index2",
            "left_j_index3",
            "left_j_middle1y",
            "left_j_middle1z",
            "left_j_middle2",
            "left_j_middle3",
            "left_j_pinky1y",
            "left_j_pinky1z",
            "left_j_pinky2",
            "left_j_pinky3",
            "left_j_ring1y",
            "left_j_ring1z",
            "left_j_ring2",
            "left_j_ring3",
            "left_j_thumb1x",
            "left_j_thumb1y",
            "left_j_thumb1z",
            "left_j_thumb2y",
            "left_j_thumb2z",
            "left_j_thumb3",
        )
    ]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.82
    camera.azimuth = 132
    camera.elevation = -28

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
            str(output_video),
        ],
        stdin=subprocess.PIPE,
    )

    rendered = 0
    preview_frame = len(hand_q) // (2 * args.stride)
    for output_frame, frame_id in enumerate(range(0, len(hand_q), args.stride)):
        data.qpos[joint_qpos] = hand_q[frame_id]

        object_frame = object_pose[frame_id]
        data.mocap_pos[0] = object_frame[4:]
        data.mocap_quat[0] = (
            object_frame[3],
            object_frame[0],
            object_frame[1],
            object_frame[2],
        )
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.write(pixels.tobytes())
        if output_frame == preview_frame:
            from PIL import Image

            Image.fromarray(pixels).save(preview_path)
        rendered += 1

    assert ffmpeg.stdin is not None
    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    diagnostics = {
        "source_frames": int(len(hand_q)),
        "rendered_frames": rendered,
        "fps": args.fps,
        "stride": args.stride,
        "duration_seconds": rendered / args.fps,
        "wrist_to_object_surface_distance_m": {
            "min": float(distances.min()),
            "median": float(np.median(distances)),
            "max": float(distances.max()),
        },
        "exact_channels": [
            "HO-Cap official 21-point hand trajectory",
            "MANO wrist global orientation (with fixed local-axis convention transform)",
            "object translation",
            "object orientation",
        ],
        "selected_orientation_candidate": "B: -90 degrees around hand-local X",
        "object_id": "G04_1",
        "retargeting": "EgoEngine-style wrist and fingertip pose IK",
        "surface_projection": False,
        "hand_model": str(HAND_URDF.relative_to(REPO_ROOT)),
        "hand_model_nq": int(model.nq),
        "hand_model_nv": int(model.nv),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))
    print(output_video)


if __name__ == "__main__":
    main()
