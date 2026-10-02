"""Shared plotting style for reference-only hand/object geometry previews.

Matches the CPU dataset geometry gallery: white canvas, fixed camera,
unfilled axis panes, and a sparse grid only when a support plane is known.
"""

from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


HAND_COLORS = {"left": "#265bb4", "right": "#c43b3b"}


def scene_limits(*point_arrays: np.ndarray) -> tuple[tuple[float, float], ...]:
    points = np.concatenate([np.asarray(array).reshape(-1, 3) for array in point_arrays])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Cannot frame a scene without finite geometry")
    center_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
    half_xy = max(float(np.ptp(points[:, :2], axis=0).max()) * 0.58, 0.14)
    zmin, zmax = float(points[:, 2].min()) - 0.03, float(points[:, 2].max()) + 0.03
    return (
        (float(center_xy[0] - half_xy), float(center_xy[0] + half_xy)),
        (float(center_xy[1] - half_xy), float(center_xy[1] + half_xy)),
        (zmin, zmax),
    )


def draw_ground(ax, xlim, ylim, ground_z: float, label: str) -> None:
    """The same 10 cm sparse grid used by the earlier geometry gallery."""
    step = 0.1
    xs = np.arange(np.ceil(xlim[0] / step) * step, xlim[1] + 1e-6, step)
    ys = np.arange(np.ceil(ylim[0] / step) * step, ylim[1] + 1e-6, step)
    for x in xs:
        ax.plot([x, x], [ylim[0], ylim[1]], [ground_z, ground_z], color="#8d9399", lw=0.65, alpha=0.65)
    for y in ys:
        ax.plot([xlim[0], xlim[1]], [ground_z, ground_z], color="#8d9399", lw=0.65, alpha=0.65)
    ax.text(xlim[0], ylim[0], ground_z, label, color="#42484d", fontsize=9)


def draw_hand_mesh(ax, vertices: np.ndarray, faces: np.ndarray, side: str) -> None:
    ax.add_collection3d(
        Poly3DCollection(
            vertices[faces], facecolor=HAND_COLORS[side], edgecolor="none", alpha=0.78
        )
    )


def style_axes(ax, limits: tuple[tuple[float, float], ...], ground_z: float | None = None,
               ground_label: str = "") -> None:
    xlim, ylim, zlim = limits
    if ground_z is not None:
        draw_ground(ax, xlim, ylim, ground_z, ground_label)
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim, xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.view_init(elev=23, azim=-60)
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
