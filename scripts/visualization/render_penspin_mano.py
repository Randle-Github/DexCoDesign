"""Render a PenSpin oracle trajectory retargeted to MANO (kinematic preview).

Pen sizes come from the four official ``assets/cylinder/pencil-5-7/*.urdf``
assets and the actor scale recorded by ``collect_official_rollouts.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_preview_style import draw_hand_mesh, scene_limits, style_axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retarget", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    data = np.load(args.retarget)
    vertices = data["mano_mesh_vertices_world"]
    faces = data["mano_mesh_faces"]
    object_pose = data["object_pose_xyzw"]
    if "object_scale" not in data or "object_asset_id" not in data:
        raise ValueError("Exact pen dimensions require object_scale and object_asset_id")
    scale = data["object_scale"]
    asset_id = data["object_asset_id"]
    if not (np.all(asset_id == asset_id[0]) and np.allclose(scale, scale[0])):
        raise ValueError("Pen asset or scale changed inside one episode")
    radius = .04 * float(scale[0])
    length = (.4 + .04 * int(asset_id[0])) * float(scale[0])
    angle = np.linspace(0, 2 * np.pi, 24)
    height = np.linspace(-length / 2, length / 2, 8)
    x = radius * np.outer(np.cos(angle), np.ones_like(height))
    y = radius * np.outer(np.sin(angle), np.ones_like(height))
    z = np.outer(np.ones_like(angle), height)
    cylinder = np.stack((x, y, z), axis=-1)
    all_points = np.concatenate((vertices.reshape(-1, 3), object_pose[:, :3]))
    limits = scene_limits(all_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.3), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=1200,
                          extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(args.output), dpi=100):
        for frame in range(len(vertices)):
            ax.cla()
            draw_hand_mesh(ax, vertices[frame], faces, "right")
            pen_world = Rotation.from_quat(object_pose[frame, 3:7]).apply(
                cylinder.reshape(-1, 3)
            ).reshape(cylinder.shape) + object_pose[frame, :3]
            ax.plot_surface(
                pen_world[..., 0], pen_world[..., 1], pen_world[..., 2],
                color="#d9a623", alpha=.90, linewidth=0,
            )
            ax.scatter(*data["source_fingertips_world"][frame].T,
                       c="#8b2020", s=12, depthshade=False)
            style_axes(ax, limits)
            ax.set_title(
                f"PenSpin Allegro → MANO (kinematic reference) | {frame + 1}/{len(vertices)}\n"
                f"tip fit {data['fingertip_error_m'][frame].mean() * 1000:.1f} mm"
            )
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main()
