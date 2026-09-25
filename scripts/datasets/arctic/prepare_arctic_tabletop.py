#!/usr/bin/env python3
"""Create and validate tabletop-aligned ARCTIC references without changing source data.

The object is placed in a mechanically stable pose found from its original
bottom mesh.  One rigid transform aligns the first recorded object pose with
that stable pose and is applied to the complete object and hand trajectories.
The recorded root motion, articulation, and hand/object relative geometry are
therefore preserved exactly.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def load_metadata(source: np.lib.npyio.NpzFile) -> dict:
    value = source["metadata_json"]
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


def object_id(source: np.lib.npyio.NpzFile) -> str:
    metadata = load_metadata(source)
    if "object_id" in metadata:
        return str(metadata["object_id"])
    objects = metadata.get("objects", [])
    if len(objects) != 1:
        raise ValueError(f"Expected one ARCTIC object, got {objects}")
    return str(objects[0]["object_id"])


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.apply_scale(0.001)
    return mesh


def stable_table_poses(bottom_path: Path, top_path: Path, joint_q: float, maximum: int = 16) -> list[tuple[np.ndarray, np.ndarray, float]]:
    bottom = load_mesh(bottom_path)
    top = load_mesh(top_path)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([0.0, 0.0, -joint_q]).as_matrix()
    top.apply_transform(transform)
    mesh = trimesh.util.concatenate((bottom, top))
    bottom_hull = bottom.convex_hull
    top_hull = top.convex_hull
    total_volume = bottom_hull.volume + top_hull.volume
    center_mass = (
        bottom_hull.volume * bottom_hull.center_mass + top_hull.volume * top_hull.center_mass
    ) / total_volume
    transforms, probabilities = trimesh.poses.compute_stable_poses(mesh.convex_hull, center_mass=center_mass)
    candidates = []
    for index in np.argsort(-probabilities)[:maximum]:
        rotation = transforms[index, :3, :3]
        vertices = (rotation @ mesh.vertices.T).T
        position = np.array([0.0, 0.0, -float(vertices[:, 2].min()) + 5.0e-4])
        candidates.append((rotation, position, float(probabilities[index])))
    if not candidates:
        raise RuntimeError(f"No stable pose found for {bottom_path}")
    return candidates


def quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]]


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def free_release_audit(
    bottom_path: Path,
    top_path: Path,
    rotation: np.ndarray,
    position: np.ndarray,
    joint_q: float,
    seconds: float,
) -> dict:
    top_quat = quat_wxyz(Rotation.from_rotvec([0.0, 0.0, -joint_q]).as_matrix())
    top_quat_text = " ".join(str(float(value)) for value in top_quat)
    with tempfile.TemporaryDirectory(prefix="arctic_table_") as directory:
        xml_path = Path(directory) / "audit.xml"
        xml_path.write_text(
            f"""<mujoco model="arctic_table_audit">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <asset>
    <mesh name="bottom" file="{bottom_path}" scale="0.001 0.001 0.001"/>
    <mesh name="top" file="{top_path}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.02" friction="1.0 0.02 0.001" contype="1" conaffinity="2"/>
    <body name="object">
      <freejoint name="root"/>
      <geom mesh="bottom" density="567" friction="1.0 0.02 0.001" contype="2" conaffinity="1"/>
      <geom mesh="top" quat="{top_quat_text}" density="567" friction="1.0 0.02 0.001" contype="2" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>\n""",
            encoding="utf-8",
        )
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        initial_q = quat_wxyz(rotation)
        data.qpos[:3] = position
        data.qpos[3:7] = initial_q
        mujoco.mj_forward(model, data)
        for _ in range(int(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        drift = float(np.linalg.norm(data.qpos[:2] - position[:2]))
        dz = float(abs(data.qpos[2] - position[2]))
        angle = quat_angle(initial_q, np.asarray(data.qpos[3:7]))
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        passed = bool(finite and drift <= 0.03 and dz <= 0.03 and angle <= np.deg2rad(15.0))
        return {
            "pass": passed,
            "horizontal_drift_m": drift,
            "vertical_change_m": dz,
            "rotation_change_deg": float(np.rad2deg(angle)),
            "simulated_seconds": seconds,
        }


def transform_points(points: np.ndarray, roots: np.ndarray, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    root_r = Rotation.from_quat(roots[:, [4, 5, 6, 3]]).as_matrix()
    local = np.einsum("tji,thj->thi", root_r, points - roots[:, None, :3])
    return np.einsum("ij,thj->thi", rotation, local) + position[None, None, :]


def transform_orientations(axis_angle: np.ndarray, roots: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    root_r = Rotation.from_quat(roots[:, [4, 5, 6, 3]]).as_matrix()
    hand_r = Rotation.from_rotvec(axis_angle).as_matrix()
    result = np.einsum("ij,tjk,tkl->til", rotation, np.swapaxes(root_r, 1, 2), hand_r)
    return Rotation.from_matrix(result).as_rotvec().astype(np.float32)


def trajectory_alignment(roots: np.ndarray, target_rotation: np.ndarray, target_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rigid world transform that maps frame-zero root to the table pose."""
    initial_rotation = Rotation.from_quat(roots[0, [4, 5, 6, 3]]).as_matrix()
    world_rotation = target_rotation @ initial_rotation.T
    world_translation = target_position - world_rotation @ roots[0, :3]
    return world_rotation, world_translation


def apply_world_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, points) + translation


def transform_root_trajectory(roots: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    root_rotations = Rotation.from_quat(roots[:, [4, 5, 6, 3]]).as_matrix()
    transformed_rotations = np.einsum("ij,tjk->tik", rotation, root_rotations)
    transformed_quaternions = Rotation.from_matrix(transformed_rotations).as_quat()[:, [3, 0, 1, 2]]
    transformed_positions = apply_world_transform(roots[:, :3], rotation, translation)
    return np.c_[transformed_positions, transformed_quaternions]


def write_table_dynamic(
    root: Path,
    poses: dict[str, dict],
    object_ids: set[str] | None = None,
) -> dict:
    """Rigidly align full trajectories to stable table poses without freezing roots."""
    canonical_out = root / "canonical_100_tabletop"
    benchmark_out = root / "benchmark_100_tabletop"
    canonical_records = []
    raw_records = []
    max_relative_error = 0.0

    for source_path in sorted((root / "canonical_100").glob("*/trajectory.npz")):
        with np.load(source_path, allow_pickle=False) as source:
            oid = object_id(source)
            if object_ids is not None and oid not in object_ids:
                continue
            pose = poses[oid]
            target_rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            target_position = np.asarray(pose["position_m"], dtype=np.float64)
            roots = np.asarray(source["object_root_pose_wxyz"][:, 0], dtype=np.float64)
            hands = np.asarray(source["hand_joints_m"], dtype=np.float64)
            world_rotation, world_translation = trajectory_alignment(
                roots, target_rotation, target_position
            )
            transformed_hands = apply_world_transform(hands, world_rotation, world_translation)
            transformed_roots = transform_root_trajectory(roots, world_rotation, world_translation)[:, None, :]
            original_relative = np.linalg.norm(hands - roots[:, None, None, :3], axis=-1)
            transformed_relative = np.linalg.norm(
                transformed_hands - transformed_roots[:, None, :, :3], axis=-1
            )
            max_relative_error = max(
                max_relative_error,
                float(np.max(np.abs(original_relative - transformed_relative))),
            )
            metadata = load_metadata(source)
            metadata.update({
                "support_mode": "dynamic_root_tabletop",
                "table_pose_source": "stable pose of original bottom mesh + MuJoCo free-release audit",
                "source_trajectory": str(source_path.relative_to(root)),
            })
            destination = canonical_out / source_path.parent.name / "trajectory.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                hand_joints_m=transformed_hands.astype(np.float32),
                hand_valid=source["hand_valid"], hand_sides=source["hand_sides"],
                object_root_pose_wxyz=transformed_roots.astype(np.float32),
                object_joint_positions_rad=source["object_joint_positions_rad"],
                frame_indices=source["frame_indices"], fps=source["fps"],
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            canonical_records.append({"sequence_id": source_path.parent.name, "path": str(destination.relative_to(root))})

    for source_path in sorted((root / "benchmark_100").glob("*.npz")):
        with np.load(source_path, allow_pickle=False) as source:
            oid = object_id(source)
            if object_ids is not None and oid not in object_ids:
                continue
            pose = poses[oid]
            target_rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            target_position = np.asarray(pose["position_m"], dtype=np.float64)
            roots = np.c_[source["object_root_position_m"], source["object_root_quaternion_wxyz"]]
            world_rotation, world_translation = trajectory_alignment(
                roots, target_rotation, target_position
            )
            transformed_roots = transform_root_trajectory(roots, world_rotation, world_translation)
            output = {key: source[key] for key in source.files}
            for side in ("left", "right"):
                output[f"{side}_translation_m"] = apply_world_transform(
                    np.asarray(source[f"{side}_translation_m"], dtype=np.float64),
                    world_rotation,
                    world_translation,
                ).astype(np.float32)
                hand_rotation = Rotation.from_rotvec(
                    np.asarray(source[f"{side}_global_orient_axis_angle"])
                ).as_matrix()
                output[f"{side}_global_orient_axis_angle"] = Rotation.from_matrix(
                    np.einsum("ij,tjk->tik", world_rotation, hand_rotation)
                ).as_rotvec().astype(np.float32)
            output["object_root_position_m"] = transformed_roots[:, :3].astype(np.float32)
            output["object_root_quaternion_wxyz"] = transformed_roots[:, 3:].astype(np.float32)
            metadata = load_metadata(source)
            metadata.update({
                "support_mode": "dynamic_root_tabletop",
                "source_trajectory": str(source_path.relative_to(root)),
            })
            output["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            destination = benchmark_out / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **output)
            raw_records.append({"sequence_id": source_path.stem, "path": str(destination.relative_to(root))})

    (canonical_out / "manifest.json").write_text(json.dumps({"schema": "dexcodesign.pose_dataset.v1", "dataset": "ARCTIC_TABLETOP", "records": canonical_records}, indent=2) + "\n")
    (benchmark_out / "manifest.json").write_text(json.dumps({"schema": "dexcodesign.arctic_subset.v1", "dataset": "ARCTIC_TABLETOP", "records": raw_records}, indent=2) + "\n")
    return {"canonical_count": len(canonical_records), "benchmark_count": len(raw_records), "max_relative_distance_error_m": max_relative_error}


def write_table_fixed(
    root: Path,
    poses: dict[str, dict],
    object_ids: set[str] | None = None,
) -> dict:
    canonical_out = root / "canonical_100_table_fixed"
    benchmark_out = root / "benchmark_100_table_fixed"
    canonical_records = []
    raw_records = []
    max_relative_error = 0.0

    for source_path in sorted((root / "canonical_100").glob("*/trajectory.npz")):
        with np.load(source_path, allow_pickle=False) as source:
            oid = object_id(source)
            if object_ids is not None and oid not in object_ids:
                continue
            pose = poses[oid]
            rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            position = np.asarray(pose["position_m"], dtype=np.float64)
            roots = np.asarray(source["object_root_pose_wxyz"][:, 0], dtype=np.float64)
            hands = np.asarray(source["hand_joints_m"], dtype=np.float64)
            transformed = np.stack(
                [transform_points(hands[:, side], roots, rotation, position) for side in range(hands.shape[1])],
                axis=1,
            )
            fixed_root = np.tile(np.r_[position, quat_wxyz(rotation)], (len(roots), 1))[:, None, :]
            original_dist = np.linalg.norm(hands - roots[:, None, None, :3], axis=-1)
            fixed_dist = np.linalg.norm(transformed - position[None, None, None, :], axis=-1)
            max_relative_error = max(max_relative_error, float(np.max(np.abs(original_dist - fixed_dist))))
            metadata = load_metadata(source)
            metadata.update({
                "support_mode": "fixed_base_table",
                "table_pose_source": "stable pose of original bottom mesh + MuJoCo free-release audit",
                "source_trajectory": str(source_path.relative_to(root)),
            })
            destination = canonical_out / source_path.parent.name / "trajectory.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                hand_joints_m=transformed.astype(np.float32),
                hand_valid=source["hand_valid"], hand_sides=source["hand_sides"],
                object_root_pose_wxyz=fixed_root.astype(np.float32),
                object_joint_positions_rad=source["object_joint_positions_rad"],
                frame_indices=source["frame_indices"], fps=source["fps"],
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            canonical_records.append({"sequence_id": source_path.parent.name, "path": str(destination.relative_to(root))})

    for source_path in sorted((root / "benchmark_100").glob("*.npz")):
        with np.load(source_path, allow_pickle=False) as source:
            oid = object_id(source)
            if object_ids is not None and oid not in object_ids:
                continue
            pose = poses[oid]
            rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            position = np.asarray(pose["position_m"], dtype=np.float64)
            roots = np.c_[source["object_root_position_m"], source["object_root_quaternion_wxyz"]]
            output = {key: source[key] for key in source.files}
            for side in ("left", "right"):
                translation = np.asarray(source[f"{side}_translation_m"], dtype=np.float64)
                output[f"{side}_translation_m"] = transform_points(
                    translation[:, None, :], roots, rotation, position
                )[:, 0].astype(np.float32)
                output[f"{side}_global_orient_axis_angle"] = transform_orientations(
                    np.asarray(source[f"{side}_global_orient_axis_angle"]), roots, rotation
                )
            output["object_root_position_m"] = np.tile(position, (len(roots), 1)).astype(np.float32)
            output["object_root_quaternion_wxyz"] = np.tile(quat_wxyz(rotation), (len(roots), 1)).astype(np.float32)
            metadata = load_metadata(source)
            metadata.update({"support_mode": "fixed_base_table", "source_trajectory": str(source_path.relative_to(root))})
            output["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            destination = benchmark_out / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **output)
            raw_records.append({"sequence_id": source_path.stem, "path": str(destination.relative_to(root))})

    (canonical_out / "manifest.json").write_text(json.dumps({"schema": "dexcodesign.pose_dataset.v1", "dataset": "ARCTIC_TABLE_FIXED", "records": canonical_records}, indent=2) + "\n")
    (benchmark_out / "manifest.json").write_text(json.dumps({"schema": "dexcodesign.arctic_subset.v1", "dataset": "ARCTIC_TABLE_FIXED", "records": raw_records}, indent=2) + "\n")
    return {"canonical_count": len(canonical_records), "benchmark_count": len(raw_records), "max_relative_distance_error_m": max_relative_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument("--write-trajectories", action="store_true")
    parser.add_argument(
        "--allow-unstable",
        action="store_true",
        help="Write best-candidate tabletop trajectories even when the release audit fails.",
    )
    parser.add_argument("--seconds", type=float, default=1.5)
    args = parser.parse_args()
    root = args.root.resolve()
    q_by_object: dict[str, list[float]] = {}
    for path in sorted((root / "canonical_100").glob("*/trajectory.npz")):
        with np.load(path, allow_pickle=False) as source:
            q_by_object.setdefault(object_id(source), []).extend(np.asarray(source["object_joint_positions_rad"]).reshape(-1).tolist())

    poses = {}
    for oid in sorted(q_by_object):
        asset = root / "assets" / "object_vtemplates" / oid
        q = float(np.median(q_by_object[oid]))
        tested = []
        selected = None
        for rank, (rotation, position, probability) in enumerate(
            stable_table_poses(asset / "bottom.obj", asset / "top.obj", q)
        ):
            audit = free_release_audit(asset / "bottom.obj", asset / "top.obj", rotation, position, q, args.seconds)
            tested.append({"rank": rank, "probability": probability, **audit})
            if audit["pass"]:
                selected = (rank, rotation, position, probability, audit)
                break
        if selected is None:
            best_index = min(
                range(len(tested)),
                key=lambda index: tested[index]["horizontal_drift_m"] + tested[index]["vertical_change_m"] + np.deg2rad(tested[index]["rotation_change_deg"]),
            )
            rotation, position, probability = stable_table_poses(asset / "bottom.obj", asset / "top.obj", q)[best_index]
            audit = tested[best_index]
            rank = best_index
        else:
            rank, rotation, position, probability, audit = selected
        poses[oid] = {
            "rotation_matrix": rotation.tolist(), "quaternion_wxyz": quat_wxyz(rotation).tolist(),
            "position_m": position.tolist(), "stable_pose_probability": probability,
            "stable_pose_rank": rank, "audit_joint_position_rad": q,
            "free_release": audit, "tested_candidates": tested,
        }
        print(f"TABLE_AUDIT {oid:16s} pass={audit['pass']} drift={audit['horizontal_drift_m']:.4f}m rot={audit['rotation_change_deg']:.2f}deg")

    failed = [name for name, pose in poses.items() if not pose["free_release"]["pass"]]
    manifest = {"schema": "dexcodesign.arctic_tabletop.v1", "method": "stable bottom-mesh pose, free root release with internal articulation held", "objects": poses, "failed_objects": failed}
    output = root / "manifests" / "tabletop_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.write_trajectories:
        if failed and not args.allow_unstable:
            raise RuntimeError(f"Refusing to rewrite trajectories; unstable objects: {failed}")
        manifest["trajectory_conversion"] = write_table_dynamic(root, poses)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ARCTIC_TABLETOP_READY objects={len(poses)} failed={len(failed)} manifest={output}")


if __name__ == "__main__":
    main()
