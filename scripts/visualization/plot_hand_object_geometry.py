"""CPU-only diagnostic: canonical MANO joints and source object mesh in world XYZ.

No IK, retargeting, object/table alignment, or simulator transforms are applied.
This is deliberately a temporary debugging tool, not a reference renderer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {
    "taco": (
        "datasets/taco_v1/canonical_100/065/trajectory.npz",
        ["datasets/taco_v1/raw/object_models/087_cm.obj", "datasets/taco_v1/raw/object_models/110_cm.obj"],
        0.01,
        [0, 20],
    ),
    "arctic": (
        "datasets/arctic_v1/canonical_100_tabletop_semantic_v3/009_scissors_s08/trajectory.npz",
        ["datasets/arctic_v1/assets/object_vtemplates/scissors/bottom.obj", "datasets/arctic_v1/assets/object_vtemplates/scissors/top.obj"],
        0.001,
        [10, 80],
    ),
    "hocap": (
        "artifacts/datasets/hocap_v1/canonical/g04_pick_place/trajectory.npz",
        ["temp/hocap_mano_replay/data/tasks/models/G04_1/cleaned_mesh_10000.obj"],
        1.0,
        [0, 100],
    ),
    "dexterhand": (
        "datasets/dexterhand_v1/canonical/Cuboid_02__seg_00/trajectory.npz",
        ["datasets/dexterhand_v1/assets/objects/Cuboid_02/visual.obj"],
        1.0,
        [380, 440],
    ),
    "gigahands": (
        "datasets/gigahands_v1/canonical_candidates_v2/GIGA_0207_p048-sandwich_0013/trajectory.npz",
        ["datasets/gigahands_v1/assets/objects/3_sandwich_sandwich_spam_can/sandwich-spam-can.obj"],
        1.0,
        [0, 40],
    ),
}


def load_mesh(path: str, scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(ROOT / path, force="mesh", process=False)
    mesh.vertices *= scale
    return mesh


def world_mesh(mesh: trimesh.Trimesh, pose: np.ndarray, articulation: float = 0.0) -> np.ndarray:
    # ARCTIC scissors URDF: top revolves about its local -Z axis at the root.
    local = np.asarray(mesh.vertices)
    if articulation:
        local = Rotation.from_rotvec([0.0, 0.0, -articulation]).apply(local)
    rot = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
    return rot.apply(local) + pose[:3]


def draw_skeleton(ax, joints: np.ndarray, color: str, label: str) -> None:
    ax.scatter(*joints[0], s=14, c=color, depthshade=False)
    for start in (1, 5, 9, 13, 17):
        chain = joints[[0, start, start + 1, start + 2, start + 3]]
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=color, lw=1.7)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=5, c=color, depthshade=False, label=label)


def draw_mesh(ax, mesh: trimesh.Trimesh, vertices: np.ndarray, color: str, label: str) -> None:
    faces = np.asarray(mesh.faces)
    if len(faces) > 6000:
        faces = faces[np.linspace(0, len(faces) - 1, 6000, dtype=int)]
    triangles = vertices[faces]
    surface = Poly3DCollection(triangles, facecolor=color, edgecolor="none", alpha=0.75)
    ax.add_collection3d(surface)
    # One marker makes the legend readable without a dense mesh outline.
    ax.scatter([], [], [], c=color, marker="s", s=32, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=[*SAMPLES, "all"], default="all")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = SAMPLES if args.dataset == "all" else {args.dataset: SAMPLES[args.dataset]}

    for name, (trajectory_path, mesh_paths, scale, frames) in selected.items():
        data = np.load(ROOT / trajectory_path, allow_pickle=True)
        meshes = [load_mesh(path, scale) for path in mesh_paths]
        hands = data["hand_joints_m"]
        valid = data["hand_valid"]
        poses = data["object_root_pose_wxyz"]
        articulation = data["object_joint_positions_rad"]
        fps = float(data["fps"])
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        axes = [fig.add_subplot(1, 2, n + 1, projection="3d") for n in range(2)]
        all_points = []
        for ax, frame in zip(axes, frames):
            frame = min(frame, len(hands) - 1)
            for side, color, label in [(0, "#265bb4", "left MANO skeleton"), (1, "#c43b3b", "right MANO skeleton")]:
                if valid[frame, side] and np.isfinite(hands[frame, side]).all():
                    joints = hands[frame, side]
                    draw_skeleton(ax, joints, color, label)
                    all_points.append(joints)
            for index, mesh in enumerate(meshes):
                object_index = index if name == "taco" else 0
                joint_angle = float(articulation[frame, 0, 0]) if name == "arctic" and index == 1 else 0.0
                vertices = world_mesh(mesh, poses[frame, object_index], joint_angle)
                draw_mesh(ax, mesh, vertices, ["#d9a623", "#4e9e77"][index % 2], f"object mesh {index + 1}")
                all_points.append(vertices)
            raw_frame = int(data["frame_indices"][frame]) if "frame_indices" in data else frame
            ax.set_title(f"{name} | canonical {frame}, source {raw_frame}, {frame / fps:.2f} s")
            ax.set_xlabel("world X (m)")
            ax.set_ylabel("world Y (m)")
            ax.set_zlabel("world Z (m)")
            ax.view_init(elev=23, azim=-60)
            ax.legend(loc="upper left", fontsize=8)
        points = np.concatenate(all_points)
        mid = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(float(np.max(np.ptp(points, axis=0))) * 0.57, 0.06)
        for ax in axes:
            ax.set_xlim(mid[0] - radius, mid[0] + radius)
            ax.set_ylim(mid[1] - radius, mid[1] + radius)
            ax.set_zlim(mid[2] - radius, mid[2] + radius)
            ax.set_box_aspect((1, 1, 1))
        destination = output / f"{name}_mano_skeleton_object_mesh.png"
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
