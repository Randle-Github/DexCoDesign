"""Render an exact-geometry kinematic MANO fit for Shadow cube reorientation.

This is a reference trajectory check, not a physics rollout. The colored cube
is transformed from Bi-DexHands' official ``cube_multicolor.obj`` at URDF scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_preview_style import draw_hand_mesh, scene_limits, style_axes


def read_cube(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if str(path) == "-":
        # Visualization-only fallback when the original Bi-DexHands OBJ is
        # unavailable locally. The official URDF scales a unit cube by 0.05.
        vertices = np.array([
            [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
            [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
        ])
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ])
        return vertices * 0.05, faces
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(token.split("/")[0]) - 1 for token in parts[1:4]])
    return np.asarray(vertices) * 0.05, np.asarray(faces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retarget", type=Path)
    parser.add_argument("cube_obj", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    data = np.load(args.retarget)
    cube_vertices, cube_faces = read_cube(args.cube_obj)
    hand_vertices = data["mano_mesh_vertices_world"]
    hand_faces = data["mano_mesh_faces"]
    cube_pose = data["object_pose_xyzw"]
    target = data["source_fingertips_world"]
    all_points = np.concatenate((hand_vertices.reshape(-1, 3), cube_pose[:, :3]))
    limits = scene_limits(all_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.3), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=1200,
                          extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(args.output), dpi=100):
        for frame in range(len(hand_vertices)):
            ax.cla()
            draw_hand_mesh(ax, hand_vertices[frame], hand_faces, "right")
            cube_world = Rotation.from_quat(cube_pose[frame, 3:7]).apply(cube_vertices)
            cube_world += cube_pose[frame, :3]
            colors = ["#db5360", "#48bb78", "#4186cf", "#efd36d", "#f4f4f4", "#e99a55"]
            cube = Poly3DCollection(
                cube_world[cube_faces],
                facecolors=[colors[index // 2] for index in range(len(cube_faces))],
                edgecolor="#252d38",
                linewidth=.7,
            )
            ax.add_collection3d(cube)
            ax.scatter(*target[frame].T, c="#8b2020", s=12, depthshade=False)
            style_axes(ax, limits)
            ax.set_title(
                f"Shadow cube → MANO (kinematic reference) | {frame + 1}/{len(hand_vertices)}"
                f"{' | cube proxy' if str(args.cube_obj) == '-' else ''}\n"
                f"tip fit {data['fingertip_error_m'][frame].mean() * 1000:.1f} mm"
            )
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main()
