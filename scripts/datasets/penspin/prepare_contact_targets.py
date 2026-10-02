"""Convert Allegro fingertip body centers to approximate contact-surface targets.

The official round-tip collision mesh extends about 12 mm from each recorded
fingertip rigid-body origin. Its pen assets are cylinders with 40 mm base
radius and 400/440/480/520 mm base lengths, scaled per environment. This
changes only retargeting targets; original trajectories stay untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def closest_cylinder(points: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    xy = points[..., :2]
    radial = np.linalg.norm(xy, axis=-1, keepdims=True)
    direction = xy / np.maximum(radial, 1e-9)
    side = np.concatenate((direction * radius,
                           np.clip(points[..., 2:3], -half_length, half_length)), axis=-1)
    cap_xy = direction * np.minimum(radial, radius)
    cap = np.concatenate((cap_xy,
                          np.where(points[..., 2:3] >= 0, half_length, -half_length)), axis=-1)
    choose_side = np.linalg.norm(points - side, axis=-1) <= np.linalg.norm(points - cap, axis=-1)
    return np.where(choose_side[..., None], side, cap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = np.load(args.source)
    pose = source["object_pose"]
    center = source["fingertips_world"]
    scales = source["object_scale"]
    ids = source["object_asset_id"]
    if not (np.allclose(scales, scales[0]) and np.all(ids == ids[0])):
        raise ValueError("One episode has multiple pen scales or assets")
    radius = .04 * float(scales[0])
    half_length = (.4 + .04 * int(ids[0])) * float(scales[0]) / 2
    rotations = Rotation.from_quat(np.repeat(pose[:, 3:7], 4, axis=0))
    local = rotations.inv().apply((center - pose[:, None, :3]).reshape(-1, 3))
    closest = closest_cylinder(local, radius, half_length)
    closest_world = rotations.apply(closest).reshape(center.shape) + pose[:, None, :3]
    direction = closest_world - center
    gap = np.linalg.norm(direction, axis=-1)
    # Source mesh radius is ~12 mm; only fingers plausibly near the pen are
    # projected to their skin contact point. Distant fingers keep their centers.
    shift = np.where(gap <= .022, np.minimum(gap, .012), 0)
    target = center + direction * (shift / np.maximum(gap, 1e-9))[..., None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fingertips_contact_world=target,
        fingertips_body_centers_world=center,
        fingertip_center_to_pen_surface_m=gap,
        object_pose=pose,
        object_scale=scales,
        object_asset_id=ids,
        hand_qpos=source["hand_qpos"],
        hand_root_pose=source["hand_root_pose"],
    )
    print(
        f"{len(center)} frames; median nearest fingertip-center gap "
        f"{np.median(gap.min(axis=1))*1000:.1f} mm; "
        f"{np.mean(gap.min(axis=1) < .022)*100:.1f}% frames have a source "
        "12-mm-radius fingertip near the pen"
    )


if __name__ == "__main__":
    main()
