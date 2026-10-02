"""CPU video of canonical MANO skeletons and source object meshes.

Draw a horizontal grid for the separately tabletop-aligned ARCTIC clip,
DexterHand's documented Z-down-to-Z-up view, and explicitly labeled support
proxies for TACO/HO-Cap. GigaHands remains in source XYZ with no invented floor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_hand_object_geometry import ROOT, SAMPLES, draw_skeleton, load_mesh, world_mesh


CLIPS = {
    "taco": (0, None, 1),
    "arctic": (10, 210, 1),
    "hocap": (0, None, 3),
    "dexterhand": (380, 500, 2),
    "gigahands": (0, None, 3),
}


def light_mesh(mesh: trimesh.Trimesh, max_faces: int = 2000) -> trimesh.Trimesh:
    """Take a fixed face sample for video speed; never alter the source asset."""
    faces = np.asarray(mesh.faces)
    if len(faces) <= max_faces:
        return mesh
    faces = faces[np.linspace(0, len(faces) - 1, max_faces, dtype=int)]
    used, inverse = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(vertices=mesh.vertices[used], faces=inverse.reshape(-1, 3), process=False)


def view_points(points: np.ndarray, name: str, dexter_z_offset: float) -> np.ndarray:
    """A shared view-only world transform; never transform hand and object separately."""
    if name != "dexterhand":
        return points
    result = np.array(points, copy=True)
    # DexterCap's official viewer declares RIGHT_HAND_Z_DOWN.  Rx(pi) maps
    # that to Isaac's right-handed Z-up convention; a Z-only reflection would
    # change handedness and cannot be represented by a valid quaternion.
    result[..., 1] *= -1.0
    result[..., 2] = dexter_z_offset - result[..., 2]
    return result


def scene_limits(data, meshes, name: str, frames: range, ground_z: float | None, dexter_z_offset: float):
    hands = data["hand_joints_m"]
    valid = data["hand_valid"]
    roots = data["object_root_pose_wxyz"]
    angles = data["object_joint_positions_rad"]
    samples = list(frames)[:: max(1, len(frames) // 25)]
    points = []
    for frame in samples:
        for side in range(2):
            if valid[frame, side]:
                points.append(view_points(hands[frame, side], name, dexter_z_offset))
        for index, mesh in enumerate(meshes):
            object_index = index if name == "taco" else 0
            angle = float(angles[frame, 0, 0]) if name == "arctic" and index == 1 else 0.0
            vertices = view_points(world_mesh(mesh, roots[frame, object_index], angle), name, dexter_z_offset)
            points.append(vertices.min(axis=0))
            points.append(vertices.max(axis=0))
    merged = np.concatenate([np.atleast_2d(p) for p in points])
    center_xy = (merged[:, :2].min(axis=0) + merged[:, :2].max(axis=0)) / 2
    half_xy = max(float(np.max(np.ptp(merged[:, :2], axis=0))) * 0.58, 0.14)
    zmin = float(merged[:, 2].min()) - 0.03
    zmax = float(merged[:, 2].max()) + 0.03
    if ground_z is not None:
        zmin = min(ground_z, zmin)
        zmax = max(ground_z + 0.12, zmax)
    return (center_xy[0] - half_xy, center_xy[0] + half_xy), (center_xy[1] - half_xy, center_xy[1] + half_xy), (zmin, zmax)


def draw_ground(ax, xlim, ylim, ground_z: float, label: str) -> None:
    # Sparse 10 cm grid at an explicitly stated world-space height.
    step = 0.1
    xs = np.arange(np.ceil(xlim[0] / step) * step, xlim[1] + 1e-6, step)
    ys = np.arange(np.ceil(ylim[0] / step) * step, ylim[1] + 1e-6, step)
    for x in xs:
        ax.plot([x, x], [ylim[0], ylim[1]], [ground_z, ground_z], color="#8d9399", lw=0.65, alpha=0.65)
    for y in ys:
        ax.plot([xlim[0], xlim[1]], [y, y], [ground_z, ground_z], color="#8d9399", lw=0.65, alpha=0.65)
    ax.text(xlim[0], ylim[0], ground_z, label, color="#42484d", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=[*SAMPLES, "all"], default="all")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = SAMPLES if args.dataset == "all" else {args.dataset: SAMPLES[args.dataset]}

    for name, (trajectory_path, mesh_paths, scale, _) in selected.items():
        data = np.load(ROOT / trajectory_path, allow_pickle=False)
        source_meshes = [load_mesh(path, scale) for path in mesh_paths]
        meshes = [light_mesh(mesh) for mesh in source_meshes]
        hands = data["hand_joints_m"]
        valid = data["hand_valid"]
        roots = data["object_root_pose_wxyz"]
        angles = data["object_joint_positions_rad"]
        fps = float(data["fps"])
        start, stop, stride = CLIPS[name]
        frames = range(start, min(stop or len(hands), len(hands)), stride)
        dexter_z_offset = 0.0
        if name == "dexterhand":
            # Rotate BOTH world-space hand joints and posed object mesh by
            # Rx(pi).  A single shared translation places the first clip
            # frame's mesh bottom at z=0 without changing relative geometry.
            dexter_z_offset = float(world_mesh(source_meshes[0], roots[start, 0])[:, 2].max())
        if name == "taco":
            ground_z = min(float(world_mesh(mesh, roots[start, index])[:, 2].min()) for index, mesh in enumerate(source_meshes))
            ground_label = f"first-frame support proxy z={ground_z:.2f} m; not calibrated"
        elif name == "gigahands":
            ground_z, ground_label = None, "source XYZ; gravity/table not calibrated"
        else:
            ground_z = 0.0
            ground_label = {
                "arctic": "aligned tabletop z=0",
                "dexterhand": "first-frame support z=0",
                "hocap": "world z=0 support proxy; not calibrated",
            }[name]
        xlim, ylim, zlim = scene_limits(data, meshes, name, frames, ground_z, dexter_z_offset)
        fig = plt.figure(figsize=(8, 7), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        destination = output / f"{name}_mano_skeleton_object_mesh.mp4"
        writer = FFMpegWriter(fps=fps / stride, codec="libx264", bitrate=2200, extra_args=["-pix_fmt", "yuv420p"])
        with writer.saving(fig, str(destination), dpi=110):
            for number, frame in enumerate(frames):
                ax.cla()
                if ground_z is not None:
                    draw_ground(ax, xlim, ylim, ground_z, ground_label)
                for side, color, label in ((0, "#265bb4", "left"), (1, "#c43b3b", "right")):
                    if valid[frame, side] and np.isfinite(hands[frame, side]).all():
                        draw_skeleton(ax, view_points(hands[frame, side], name, dexter_z_offset), color, label)
                for index, mesh in enumerate(meshes):
                    object_index = index if name == "taco" else 0
                    angle = float(angles[frame, 0, 0]) if name == "arctic" and index == 1 else 0.0
                    vertices = view_points(world_mesh(mesh, roots[frame, object_index], angle), name, dexter_z_offset)
                    triangles = vertices[mesh.faces]
                    ax.add_collection3d(Poly3DCollection(triangles, facecolor=("#d9a623", "#4e9e77")[index % 2], edgecolor="none", alpha=0.86))
                ax.set(xlim=xlim, ylim=ylim, zlim=zlim, xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
                ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
                ax.view_init(elev=23, azim=-60)
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                source_frame = int(data["frame_indices"][frame])
                suffix = " | DexterCap Z-down to Z-up" if name == "dexterhand" else ""
                if ground_z is None:
                    suffix = " | source XYZ; up unverified"
                ax.set_title(f"{name} | frame {frame} (source {source_frame}) | {frame / fps:.2f} s{suffix}", fontsize=11)
                writer.grab_frame()
                if number % 25 == 0:
                    print(f"{name}: {number + 1}/{len(frames)}", flush=True)
        plt.close(fig)
        print(f"VIDEO_READY {destination} frames={len(frames)} fps={fps / stride:g}", flush=True)


if __name__ == "__main__":
    main()
