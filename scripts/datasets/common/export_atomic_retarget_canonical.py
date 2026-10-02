#!/usr/bin/env python3
"""Export the existing atomic-task MANO fits to the standard pose trajectory schema.

This adapter changes storage format, not the fit or coordinate frame.  It also
writes simple object geometry where the source task defines a primitive but no
standalone local mesh was retained.  The resulting selection can be rendered
by scripts/visualization/render_geometry_gallery.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import trimesh

from trajectory_schema import audit_trajectory, save_trajectory


ROOT = Path(__file__).resolve().parents[3]
PREVIEWS = ROOT / "artifacts/atomic_tasks/previews"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PHALANX_BODIES = {
    "thumb": ("thumb1x", "thumb2y", "thumb3"),
    "index": ("index1y", "index2", "index3"),
    "middle": ("middle1y", "middle2", "middle3"),
    "ring": ("ring1y", "ring2", "ring3"),
    "pinky": ("pinky1y", "pinky2", "pinky3"),
}


def mano_joints(fit_path: Path, side: str) -> np.ndarray:
    fit = np.load(fit_path, allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(ROOT / f"assets/robot_hands/mano/source/{side}.xml"))
    state = mujoco.MjData(model)
    qpos = fit["mano_qpos"]
    if qpos.shape[1] != model.nq:
        raise ValueError(f"{fit_path}: qpos shape {qpos.shape} does not match model nq={model.nq}")
    palm_site = model.site(f"{side}_palm").id
    body_ids = {
        finger: [model.body(f"{side}_{name}").id for name in PHALANX_BODIES[finger]]
        for finger in FINGERS
    }
    tip_ids = {finger: model.site(f"{side}_{finger}_tip").id for finger in FINGERS}
    result = np.empty((len(qpos), 21, 3), dtype=np.float32)
    fitted_fingers = tuple(str(name) for name in fit["fitted_fingers"]) if "fitted_fingers" in fit else (
        "index", "middle", "ring", "pinky", "thumb"
    )
    max_tip_disagreement = 0.0
    for frame, pose in enumerate(qpos):
        state.qpos[:] = pose
        mujoco.mj_forward(model, state)
        result[frame, 0] = state.site_xpos[palm_site]
        for finger_number, finger in enumerate(FINGERS):
            start = 1 + 4 * finger_number
            result[frame, start:start + 3] = state.xpos[body_ids[finger]]
            result[frame, start + 3] = state.site_xpos[tip_ids[finger]]
        for source_index, finger in enumerate(fitted_fingers):
            tip = result[frame, 1 + 4 * FINGERS.index(finger) + 3]
            error = float(np.linalg.norm(tip - fit["fitted_fingertips_world"][frame, source_index]))
            max_tip_disagreement = max(max_tip_disagreement, error)
    if max_tip_disagreement > 1e-4:
        raise ValueError(f"{fit_path}: MANO FK differs from saved fitted tips by {max_tip_disagreement:g} m")
    return result


def object_asset(kind: str, fit, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "pen":
        scales = fit["object_scale"]
        asset_ids = fit["object_asset_id"]
        if not (np.allclose(scales, scales[0]) and np.all(asset_ids == asset_ids[0])):
            raise ValueError("Pen mesh scale/asset changes inside the sample")
        scale, asset_id = float(scales[0]), int(asset_ids[0])
        mesh = trimesh.creation.cylinder(radius=0.04 * scale,
                                         height=(0.4 + 0.04 * asset_id) * scale, sections=32)
        provenance = "primitive dimensions from source PenSpin URDF and recorded scale"
    elif kind == "egg":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.vertices *= np.array([0.03, 0.03, 0.04])
        provenance = "source Bi-DexHands egg.xml ellipsoid dimensions"
    elif kind == "cube":
        source = ROOT / "datasets/bidexhands_v1/assets/objects/source/cube_multicolor.obj"
        mesh = trimesh.load(source, force="mesh")
        mesh.apply_scale(0.05)
        provenance = "official Bi-DexHands cube_multicolor.obj with URDF scale 0.05"
    else:
        raise ValueError(kind)
    mesh.export(destination)
    return provenance


def export_sample(spec: dict) -> dict:
    paths = {side: PREVIEWS / file_name for side, file_name in spec["fits"].items()}
    fit = np.load(next(iter(paths.values())), allow_pickle=False)
    count = len(fit["mano_qpos"])
    hands = np.zeros((count, 2, 21, 3), dtype=np.float32)
    valid = np.zeros((count, 2), dtype=bool)
    for side, fit_path in paths.items():
        side_fit = np.load(fit_path, allow_pickle=False)
        if len(side_fit["mano_qpos"]) != count or not np.allclose(
            side_fit["object_pose_xyzw"], fit["object_pose_xyzw"]
        ):
            raise ValueError(f"{spec['sample_id']}: hands do not share a time/object trajectory")
        side_index = 0 if side == "left" else 1
        hands[:, side_index] = mano_joints(fit_path, side)
        valid[:, side_index] = True

    xyzw = np.asarray(fit["object_pose_xyzw"], dtype=np.float32)
    wxyz = np.concatenate((xyzw[:, :3], xyzw[:, 6:7], xyzw[:, 3:6]), axis=1)
    wxyz[:, 3:] /= np.linalg.norm(wxyz[:, 3:], axis=1, keepdims=True)
    stride = int(fit["source_stride"]) if "source_stride" in fit else 1
    frames = np.arange(count, dtype=np.int64) * stride
    source_fps = spec.get("source_fps_from_manifest")
    for fit_path in paths.values():
        side_fit = np.load(fit_path, allow_pickle=False)
        source = Path(str(side_fit["source_path"]))
        if not source.exists():
            continue
        raw = np.load(source, allow_pickle=False)
        if "step_num" in raw:
            frames = np.asarray(raw["step_num"][::stride][:count], dtype=np.int64)
        if "fps" in raw:
            source_fps = float(raw["fps"])
    if len(frames) != count:
        raise ValueError(f"{spec['sample_id']}: source frame count does not match MANO fit")
    fps = source_fps / stride if source_fps is not None else spec["fallback_fps"]
    fps_provenance = "source rollout/manifest fps divided by retarget stride" if source_fps is not None else (
        "previous preview playback rate; source rollout omitted timing"
    )

    dataset_root = ROOT / "datasets" / "supp_v1"
    trajectory_path = dataset_root / "canonical" / spec["sample_id"] / "trajectory.npz"
    mesh_path = dataset_root / "assets" / "objects" / f"{spec['sample_id']}.obj"
    fit_copies = {}
    for side, source_fit in paths.items():
        destination_fit = dataset_root / "retarget_fits" / source_fit.name
        destination_fit.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_fit, destination_fit)
        fit_copies[side] = str(destination_fit.relative_to(ROOT))
    mesh_provenance = object_asset(spec["object_kind"], fit, mesh_path)
    save_trajectory(
        trajectory_path,
        hand_joints_m=hands,
        hand_valid=valid,
        object_root_pose_wxyz=wxyz[:, None, :],
        object_joint_positions_rad=np.zeros((count, 1, 0), dtype=np.float32),
        frame_indices=frames,
        fps=fps,
        metadata={
            "dataset": "supp",
            "sequence_id": spec["sample_id"],
            "source_dataset": spec["dataset"],
            "hand_representation": "retargeted MANO forward-kinematics skeleton",
            "reference_only": True,
            "source_fits": fit_copies,
            "fps_provenance": fps_provenance,
            "object_geometry_provenance": mesh_provenance,
            "object_source_mesh_exact": False,
        },
    )
    audit = audit_trajectory(trajectory_path)
    if not audit["ok"]:
        raise ValueError(audit)
    return {
        "dataset": "supp",
        "sample_id": spec["sample_id"],
        "source_dataset": spec["dataset"],
        "source_fits": fit_copies,
        "trajectory": str(trajectory_path.relative_to(ROOT)),
        "frames": count,
        "fps": fps,
        "objects": [{"path": str(mesh_path.relative_to(ROOT)), "scale": 1.0,
                     "object_index": 0, "part": "root"}],
        "clip_start": 0,
        "clip_stop": count,
        "ground": "not_calibrated",
        "object_geometry_provenance": mesh_provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "artifacts/atomic_tasks/canonical_selection.json")
    args = parser.parse_args()
    specs = [
        {"dataset": "penspin", "sample_id": f"penspin_{index:03d}", "object_kind": "pen",
         "fits": {"right": f"penspin_{index:03d}_mano_right.npz"},
         "source_fps_from_manifest": 20.0, "fallback_fps": 10.0}
        for index in range(7)
    ] + [
        {"dataset": "bidexhands", "sample_id": f"reorientation_{index:03d}",
         "object_kind": "cube",
         "fits": {"right": f"bidex_reorientation_{index:03d}_mano_right.npz"},
         "selection_status": "diagnostic_only_source_distal_centers_not_contact_points",
         "fallback_fps": 10.0}
        for index in range(5)
    ]
    groups: dict[str, list[dict]] = {}
    diagnostic_groups: dict[str, list[dict]] = {}
    for spec in specs:
        record = export_sample(spec)
        if "selection_status" in spec:
            record["selection_status"] = spec["selection_status"]
            diagnostic_groups.setdefault(record["dataset"], []).append(record)
            label = "CANONICAL_DIAGNOSTIC"
        else:
            groups.setdefault(record["dataset"], []).append(record)
            label = "CANONICAL_READY"
        print(f"{label} {record['trajectory']} frames={record['frames']} fps={record['fps']:g}", flush=True)
    selection = {"schema": "dexcodesign.geometry_gallery.v1",
                 "groups": groups, "diagnostic_groups": diagnostic_groups}
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps(selection, indent=2) + "\n")
    manifest = ROOT / "datasets/supp_v1/manifest.json"
    manifest.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"SELECTION_READY {args.selection}", flush=True)
    print(f"SUPP_MANIFEST_READY {manifest}", flush=True)


if __name__ == "__main__":
    main()
