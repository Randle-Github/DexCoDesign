"""Contact-keypoint retargeting from a robot's five fingertips to the project MANO hand.

This is an initial *kinematic* fit, not a physics replay or an inverse of the
source robot joint angles.  It preserves the measured world fingertip targets
and the object trajectory and reports residuals for every frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


FINGER_NAMES = ("index", "middle", "ring", "pinky", "thumb")


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    fix = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ fix @ vt
    return rotation, target_center - source_center @ rotation


def fit_trajectory(
    source_path: Path,
    model_path: Path,
    output_path: Path,
    side: str,
    key: str,
    stride: int = 1,
    fingers: tuple[str, ...] = FINGER_NAMES,
    object_key: str = "object_pose_xyzw",
    root_mode: str = "free",
) -> dict:
    source = np.load(source_path)
    if stride < 1:
        raise ValueError("stride must be at least 1")
    raw = np.asarray(source[key][::stride], dtype=np.float64)
    target = raw[:, :, :3] if raw.ndim == 3 else raw
    if target.ndim != 3 or target.shape[1:] != (len(fingers), 3):
        raise ValueError(f"Expected T x {len(fingers)} x 3 fingertips, got {target.shape}")
    if len(set(fingers)) != len(fingers) or not set(fingers).issubset(FINGER_NAMES):
        raise ValueError(f"Unknown or repeated finger names: {fingers}")
    if root_mode not in ("free", "anchored", "fixed"):
        raise ValueError(f"Unknown root mode {root_mode}")
    if not np.isfinite(target).all():
        raise ValueError("Source fingertips contain non-finite values")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    site_ids = [model.site(f"{side}_{name}_tip").id for name in fingers]
    mesh_geoms = [
        index
        for index in range(model.ngeom)
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[index] == 1
    ]
    mesh_local_vertices = []
    mesh_faces = []
    vertex_offset = 0
    for geom_id in mesh_geoms:
        mesh_id = model.geom_dataid[geom_id]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
        ].copy()
        faces = model.mesh_face[
            model.mesh_faceadr[mesh_id] : model.mesh_faceadr[mesh_id] + model.mesh_facenum[mesh_id]
        ].copy()
        mesh_local_vertices.append(vertices)
        mesh_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)
    mesh_faces = np.concatenate(mesh_faces)
    lower = model.jnt_range[:, 0].copy()
    upper = model.jnt_range[:, 1].copy()
    neutral = np.clip(np.zeros(model.nq), lower + 1e-5, upper - 1e-5)
    data.qpos[:] = neutral
    mujoco.mj_forward(model, data)
    initial_sites = np.asarray([data.site_xpos[site] for site in site_ids]).copy()
    rotation, translation = _kabsch(initial_sites, target[0])
    initial = neutral.copy()
    initial[:3] = translation
    initial[3:6] = Rotation.from_matrix(rotation.T).as_euler("xyz")
    initial = np.clip(initial, lower + 1e-5, upper - 1e-5)

    qpos = np.empty((len(target), model.nq), dtype=np.float64)
    fitted_tips = np.empty_like(target)
    mesh_vertices = np.empty((len(target), vertex_offset, 3), dtype=np.float32)
    previous = None
    root_anchor = None
    for frame, desired in enumerate(target):
        seed = initial if previous is None else previous

        def residual(q: np.ndarray) -> np.ndarray:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            actual = np.asarray([data.site_xpos[site] for site in site_ids])
            position = ((actual - desired) / 0.01).ravel()
            posture = (q[6:] - neutral[6:]) / 0.65
            if previous is None:
                return np.r_[position, posture]
            smooth_fingers = (q[6:] - previous[6:]) / 0.18
            if root_mode == "fixed":
                root_residual = np.r_[
                    (q[:3] - root_anchor[:3]) / .001,
                    (q[3:6] - root_anchor[3:6]) / .02,
                ]
            elif root_mode == "anchored":
                root_residual = np.r_[
                    (q[:3] - root_anchor[:3]) / .012,
                    (q[3:6] - root_anchor[3:6]) / .15,
                    (q[:3] - previous[:3]) / .01,
                    (q[3:6] - previous[3:6]) / .25,
                ]
            else:
                root_residual = np.r_[
                    (q[:3] - previous[:3]) / .01,
                    (q[3:6] - previous[3:6]) / .25,
                ]
            return np.r_[position, posture, smooth_fingers, root_residual]

        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=35 if previous is not None else 150,
            ftol=1e-5,
            xtol=1e-5,
        )
        previous = solution.x
        if root_anchor is None:
            root_anchor = solution.x[:6].copy()
        qpos[frame] = solution.x
        data.qpos[:] = solution.x
        mujoco.mj_forward(model, data)
        fitted_tips[frame] = np.asarray([data.site_xpos[site] for site in site_ids])
        mesh_vertices[frame] = np.concatenate(
            [
                vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
                for geom_id, vertices in zip(mesh_geoms, mesh_local_vertices)
            ]
        )
    error = np.linalg.norm(fitted_tips - target, axis=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    object_metadata = {
        name: source[name][::stride]
        for name in ("object_scale", "object_asset_id")
        if name in source
    }
    np.savez_compressed(
        output_path,
        mano_qpos=qpos,
        source_fingertips_world=target,
        fitted_fingertips_world=fitted_tips,
        fingertip_error_m=error,
        mano_mesh_vertices_world=mesh_vertices,
        mano_mesh_faces=mesh_faces,
        object_pose_xyzw=source[object_key][::stride],
        side=np.asarray(side),
        source_path=np.asarray(str(source_path)),
        source_stride=np.asarray(stride),
        fitted_fingers=np.asarray(fingers),
        root_mode=np.asarray(root_mode),
        **object_metadata,
    )
    metrics = {
        "side": side,
        "frames": int(len(qpos)),
        "mean_fingertip_error_m": float(np.mean(error)),
        "p95_fingertip_error_m": float(np.quantile(error, 0.95)),
        "max_fingertip_error_m": float(np.max(error)),
        "kinematic_only": True,
        "root_mode": root_mode,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--key", default="right_fingertip_state_world")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fingers", default=",".join(FINGER_NAMES))
    parser.add_argument("--object-key", default="object_pose_xyzw")
    parser.add_argument("--root-mode", choices=("free", "anchored", "fixed"), default="free")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.model is None:
        root = Path(__file__).resolve().parents[3]
        args.model = root / "assets/robot_hands/mano/source" / f"{args.side}.xml"
    fingers = tuple(name.strip() for name in args.fingers.split(","))
    print(json.dumps(fit_trajectory(args.source, args.model, args.output, args.side, args.key, args.stride, fingers, args.object_key, args.root_mode), indent=2))


if __name__ == "__main__":
    main()
