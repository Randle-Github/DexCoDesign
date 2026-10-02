#!/usr/bin/env python3
"""Build a small, no-video GigaHands pose/mesh candidate set in common schema.

Only clips with object tracking manually marked successful by the dataset authors
are admitted. The exported GigaHands hand slots are stored left/right by their
published two-hand convention; ``source_hand_order_verified`` remains false until
visual spot-checking because the release JSON itself only names the 42-point axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "datasets" / "common"))
from trajectory_schema import save_trajectory  # noqa: E402


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def action_group(text: str) -> str:
    text = text.lower()
    groups = {
        "tool_use": ("tool", "knife", "scissor", "screwdriver", "hammer", "brush", "spoon", "fork", "pen", "pencil"),
        "open_close": ("open", "close", "unscrew", "unfasten", "fasten", "zip", "unwrap"),
        "rotate_twist": ("rotate", "turn", "twist", "spin", "screw"),
        "press_contact": ("press", "push", "poke", "tap", "click", "button"),
        "transfer_place": ("pick", "lift", "place", "put", "move", "transfer", "drop"),
        "pour_scoop": ("pour", "scoop", "stir", "squeeze", "extract"),
        "pinch_fine": ("pinch", "thread", "string", "insert", "grip", "grasp", "hold"),
        "wipe_fold": ("wipe", "fold", "tear", "clean", "wrap", "roll"),
    }
    for name, words in groups.items():
        if any(word in text for word in words):
            return name
    return "other"


def object_score(action: str, path: str) -> int:
    stop = {"the", "a", "an", "on", "in", "with", "from", "to", "of", "and", "your", "it"}
    words = lambda s: {x for x in re.findall(r"[a-z0-9]+", s.lower().replace("_", " ").replace("-", " ")) if len(x) > 2 and x not in stop}
    return len(words(action) & words(path))


def read_success_objects(root: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for csv_path in sorted((root / "raw" / "object_meta" / "scene_wise_round1").glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                scene, seq = row[0], row[1]
                try:
                    seq_int = int(seq)
                except ValueError:
                    continue
                session = f"{scene}_{seq_int:04d}"
                for idx in range(2, min(len(header), len(row)) - 1, 2):
                    source_obj, present, success = header[idx], row[idx], row[idx + 1]
                    if present != "T" or success != "T" or not source_obj:
                        continue
                    parts = source_obj.split("/")
                    if len(parts) < 3:
                        continue
                    category = parts[0]
                    # Annotation labels sometimes use '-' where published folders use '_'.
                    obj_folder = parts[1].replace("-", "_")
                    result[session].append({
                        "scene_category": category,
                        "object_folder": obj_folder,
                        "mesh_folder": parts[1],
                        "annotation_object_path": source_obj,
                    })
    pose_root = root / "raw" / "object_poses"
    pose_lookup = {}
    for pose_path in pose_root.glob("**/optimized_pose.json"):
        rel = pose_path.relative_to(pose_root)
        category, object_folder, session = rel.parts[:3]
        pose_lookup[(session, category, object_folder)] = pose_path
    for session, entries in list(result.items()):
        matched = []
        for item in entries:
            pose = pose_lookup.get((session, item["scene_category"], item["object_folder"]))
            if pose is not None:
                matched.append({**item, "pose_path": str(pose.relative_to(root))})
        result[session] = matched
    return {k: v for k, v in result.items() if v}


def read_action_annotations(root: Path, success: dict[str, list[dict]]) -> list[dict]:
    result = []
    path = root / "_downloads" / "annotations_v2.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = str(item.get("clarify_annotation", "")).strip()
            if not text or text.lower() in {"none", "buggy"}:
                continue
            scene = str(item.get("scene", ""))
            seq_value = item.get("sequence", "0")
            if isinstance(seq_value, list):
                seq_value = seq_value[0]
            sequence_source = str(seq_value)
            try:
                seq_int = int(seq_value)
            except (TypeError, ValueError):
                continue
            session = f"{scene}_{seq_int:04d}"
            if session not in success:
                continue
            candidates = success[session]
            # Prefer the tracked object whose name occurs in the action text; otherwise
            # preserve the first author-ordered successful object for this session.
            chosen = max(candidates, key=lambda x: (object_score(text, x["annotation_object_path"]), -candidates.index(x)))
            result.append({
                "scene": scene,
                "sequence": sequence_source,
                "sequence_index": seq_int,
                "session": session,
                "text": text,
                "description": str(item.get("description", "")),
                "start_frame_id": int(item.get("start_frame_id", 0)),
                "end_frame_id": int(item.get("end_frame_id", -1)),
                "action_group": action_group(text),
                **chosen,
            })
    # One atomic annotation per capture sequence; if duplicates exist, keep the longest.
    unique = {}
    for item in result:
        old = unique.get(item["session"])
        if old is None or len(item["text"]) > len(old["text"]):
            unique[item["session"]] = item
    return list(unique.values())


def balanced_select(candidates: list[dict], count: int) -> list[dict]:
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        groups[item["scene_category"]][item["object_folder"]].append(item)
    for category in groups:
        for obj in groups[category]:
            groups[category][obj].sort(key=lambda x: hashlib.sha256(x["session"].encode()).hexdigest())
    selected = []
    categories = sorted(groups)
    while len(selected) < count:
        progressed = False
        for category in categories:
            for obj in sorted(groups[category]):
                if groups[category][obj]:
                    selected.append(groups[category][obj].pop(0))
                    progressed = True
                    if len(selected) == count:
                        break
            if len(selected) == count:
                break
        if not progressed:
            break
    for index, item in enumerate(selected):
        item["candidate_id"] = f"GIGA_{index:04d}_{safe_id(item['session'])}"
    return selected


def convert_one(root: Path, output: Path, item: dict, hand_points: np.ndarray) -> dict | None:
    if hand_points.ndim != 3 or hand_points.shape[1:] != (42, 3):
        return None
    pose_path = root / item["pose_path"]
    pose_json = json.loads(pose_path.read_text())
    source_ids = np.asarray(sorted(int(k) for k in pose_json), dtype=np.int64)
    hand_count = len(hand_points)
    requested_start = max(0, item["start_frame_id"])
    requested_stop = hand_count if item["end_frame_id"] < 0 else min(hand_count, item["end_frame_id"] + 1)
    # GigaHands object poses are sparse tracked keyframes (~10 Hz), while hands
    # are 30 Hz. Interpolate only inside the observed object-pose time interval.
    start = max(requested_start, int(source_ids[0]))
    stop = min(requested_stop, hand_count, int(source_ids[-1]) + 1)
    if stop - start < 12 or len(source_ids) < 2:
        return None
    source_gaps = np.diff(source_ids)
    if len(source_gaps) and int(source_gaps.max()) > 30:
        return None
    frame_ids = np.arange(start, stop, dtype=np.int64)
    hand_points = hand_points[frame_ids]
    hand = hand_points.reshape(len(frame_ids), 2, 21, 3).astype(np.float32)
    valid = np.isfinite(hand).all(axis=(2, 3))
    # Reject absent/zero-filled hand slots, but retain valid source coordinates verbatim.
    spread = np.nanmax(hand, axis=2) - np.nanmin(hand, axis=2)
    valid &= np.linalg.norm(spread, axis=-1) > 0.015
    hand[~valid] = np.nan
    if not valid.any(axis=1).all():
        return None

    source_translations = []
    source_quaternions = []
    for frame in source_ids:
        pose = pose_json[str(int(frame)).zfill(6)]
        source_translations.append(np.asarray(pose["mesh_translation"], dtype=np.float32).reshape(3))
        # Official GigaHands object annotation and renderer use PyTorch3D wxyz.
        source_quaternions.append(np.asarray(pose["mesh_rotation"], dtype=np.float32).reshape(4))
    tracked_trans = np.asarray(source_translations, dtype=np.float32)
    tracked_wxyz = np.asarray(source_quaternions, dtype=np.float32)
    norms = np.linalg.norm(tracked_wxyz, axis=1)
    if not np.isfinite(tracked_trans).all() or np.min(norms) < 1e-5:
        return None
    tracked_wxyz /= norms[:, None]
    # The official render_mesh_video.py transposes the rotation matrix made
    # from mesh_rotation. In quaternion form this is the inverse rotation.
    tracked_wxyz[:, 1:] *= -1
    # Keep quaternion signs continuous for stable interpolation/training.
    for i in range(1, len(tracked_wxyz)):
        if np.dot(tracked_wxyz[i - 1], tracked_wxyz[i]) < 0:
            tracked_wxyz[i] *= -1
    query = frame_ids.astype(np.float64)
    trans = np.column_stack([np.interp(query, source_ids, tracked_trans[:, axis]) for axis in range(3)]).astype(np.float32)
    tracked_xyzw = tracked_wxyz[:, [1, 2, 3, 0]]
    quat_xyzw = Slerp(source_ids.astype(np.float64), Rotation.from_quat(tracked_xyzw))(query).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    root_pose = np.concatenate((trans, quat_wxyz), axis=1)[:, None]
    object_q = np.empty((len(frame_ids), 1, 0), dtype=np.float32)
    mesh_id = f"{item['scene_category']}/{item['object_folder']}"
    mesh_dir = f"assets/objects/{safe_id(mesh_id)}"
    output_path = output / item["candidate_id"] / "trajectory.npz"
    save_trajectory(
        output_path,
        hand_joints_m=hand,
        hand_valid=valid,
        object_root_pose_wxyz=root_pose,
        object_joint_positions_rad=object_q,
        frame_indices=frame_ids,
        fps=30.0,
        metadata={
            "dataset": "GigaHands",
            "sequence_id": item["candidate_id"],
            "source_session": item["session"],
            "source_scene": item["scene_category"],
            "source_object_path": item["annotation_object_path"],
            "action_text": item["text"],
            "action_group": item["action_group"],
            "source_object_pose_keyframes": int(len(source_ids)),
            "object_rotation_convention": "official PyTorch3D mesh_rotation transposed into simulator frame",
            "object_pose_interpolation": "linear translation + quaternion SLERP at 30 Hz; no extrapolation",
            "source_hand_representation": "official MANO-aligned keypoints, [frame,42,3], source values in meters",
            "source_hand_order": "first 21 / second 21 mapped to left / right by release convention; not independently verified",
            "source_hand_order_verified": False,
            "objects": [{
                "object_id": mesh_id,
                "role": "manipulated_object_candidate",
                "mesh_asset_dir": mesh_dir,
                "tracking_status": "authors' manually annotated success",
                "joint_names": [],
                "joint_types": [],
                "articulation_status": "only 6-DoF pose released; no articulation state provided",
            }],
            "simulator_ready": False,
            "fps_source": "GigaHands capture rate 30 Hz",
        },
    )
    return {
        "candidate_id": item["candidate_id"],
        "path": str(output_path.relative_to(output)),
        "source_session": item["session"],
        "source_object_path": item["annotation_object_path"],
        "action_text": item["text"],
        "action_group": item["action_group"],
        "frames": len(frame_ids),
        "frame_start": int(frame_ids[0]),
        "frame_stop_exclusive": int(frame_ids[-1] + 1),
        "source_object_pose_keyframes": int(len(source_ids)),
        "scene_category": item["scene_category"],
        "object_folder": item["object_folder"],
    }


def minimum_hand_to_mesh_box_gap(trajectory: Path, mesh_path: Path) -> float:
    """Necessary (not sufficient) contact test in the object's local frame."""
    with np.load(trajectory, allow_pickle=False) as data:
        roots = np.asarray(data["object_root_pose_wxyz"][:, 0], dtype=np.float64)
        hands = np.asarray(data["hand_joints_m"], dtype=np.float64)
        valid = np.asarray(data["hand_valid"], dtype=bool)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Invalid object mesh: {mesh_path}")
    frame_ids = np.unique(np.linspace(0, len(roots) - 1, min(80, len(roots)), dtype=int))
    rotation = Rotation.from_quat(roots[frame_ids][:, [4, 5, 6, 3]]).as_matrix()
    points = hands[frame_ids].reshape(len(frame_ids), 42, 3)
    active = np.repeat(valid[frame_ids], 21, axis=1)
    local = np.einsum(
        "tij,tkj->tki", rotation.transpose(0, 2, 1),
        points - roots[frame_ids, None, :3],
    )
    outside = np.maximum(np.maximum(mesh.bounds[0] - local, local - mesh.bounds[1]), 0.0)
    distance = np.linalg.norm(outside, axis=-1)
    distance[~active] = np.inf
    return float(np.min(distance))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/gigahands_v1"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "canonical_candidates_v2").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty candidate output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    success = read_success_objects(root)
    candidates = read_action_annotations(root, success)
    selected = balanced_select(candidates, args.count)
    manifests = output.parent / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"candidate_pool_{args.count}.json").write_text(json.dumps({
        "dataset": "GigaHands", "target_count": args.count,
        "manually_successful_tracking_sessions": len(success),
        "eligible_action_sessions": len(candidates),
        "selected": selected,
    }, indent=2) + "\n")

    wanted = {}
    for item in selected:
        item["hand_member"] = f"{item['scene']}/keypoints_3d_mano_align/{item['sequence']}.json"
        wanted[item["hand_member"]] = item
    converted = []
    found_count = 0
    rejected_count = 0
    archive = root / "_downloads" / "keypoints_3d_mano_align.tar.gz"
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            item = wanted.get(name)
            if item is None or not member.isfile():
                continue
            found_count += 1
            stream = tar.extractfile(member)
            if stream is None:
                continue
            try:
                points = np.asarray(json.load(stream), dtype=np.float32)
                record = convert_one(root, output, item, points)
            except (ValueError, KeyError, json.JSONDecodeError):
                record = None
            if record is not None:
                converted.append(record)
                print(f"GIGAHANDS_CANONICAL_READY {len(converted):03d}/{len(selected)} {record['candidate_id']}", flush=True)
            else:
                rejected_count += 1

    # Extract only mesh folders actually referenced by successfully converted records.
    mesh_root = root / "assets" / "objects"
    mesh_root.mkdir(parents=True, exist_ok=True)
    wanted_prefixes = {}
    converted_sessions = {r["source_session"] for r in converted}
    for item in selected:
        if item["session"] in converted_sessions:
            prefix = f"publish/{item['scene_category']}/{item['mesh_folder']}/"
            wanted_prefixes[prefix] = mesh_root / safe_id(f"{item['scene_category']}/{item['object_folder']}")
    mesh_archive = root / "_downloads" / "scans_publish.zip"
    mesh_status = {}
    with zipfile.ZipFile(mesh_archive) as zf:
        for member in zf.infolist():
            parts = member.filename.split("/")
            if len(parts) < 4 or member.is_dir():
                continue
            prefix = "/".join(parts[:3]) + "/"
            destination = wanted_prefixes.get(prefix)
            if destination is None:
                continue
            relative = Path(*parts[3:])
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            if target.suffix.lower() == ".obj":
                mesh_status[prefix] = True

    accepted = []
    rejected = []
    for record in converted:
        item = next(x for x in selected if x["session"] == record["source_session"])
        prefix = f"publish/{item['scene_category']}/{item['mesh_folder']}/"
        mesh_dir = Path("assets/objects") / safe_id(f"{item['scene_category']}/{item['object_folder']}")
        mesh_path = root / mesh_dir / Path(item["annotation_object_path"]).name
        reason = None
        if not mesh_status.get(prefix, False) or not mesh_path.is_file():
            reason = "exact_annotated_mesh_missing"
        else:
            try:
                gap = minimum_hand_to_mesh_box_gap(output / record["path"], mesh_path)
                record["minimum_hand_to_mesh_bbox_gap_m"] = gap
                if not np.isfinite(gap) or gap > 0.04:
                    reason = "no_hand_near_tracked_object"
            except (ValueError, OSError) as error:
                reason = f"mesh_geometry_error: {error}"
        if reason is not None:
            rejected.append({**record, "reason": reason})
            shutil.rmtree((output / record["path"]).parent)
            continue
        record["mesh_file"] = str(mesh_path.relative_to(root))
        record["mesh_asset_dir"] = str(mesh_dir)
        record["mesh_available"] = True
        accepted.append(record)

    (output / "manifest.json").write_text(json.dumps({
        "schema": "dexcodesign.pose_dataset.v1", "dataset": "GigaHands",
        "source_license": "CC BY-NC 4.0", "records": accepted,
    }, indent=2) + "\n")
    (output / "rejected.json").write_text(json.dumps(rejected, indent=2) + "\n")
    summary = {
        "requested": args.count, "eligible_sessions": len(candidates),
        "selected": len(selected), "hand_members_found": found_count,
        "rejected_after_hand_match": rejected_count,
        "missing_hand_members": len(selected) - found_count,
        "converted_before_geometry_filter": len(converted),
        "accepted": len(accepted), "rejected": len(rejected),
        "rejection_reasons": {reason: sum(r["reason"] == reason for r in rejected) for reason in sorted({r["reason"] for r in rejected})},
        "object_meshes_available": len(accepted),
        "source_hand_order_verified": False,
        "videos_downloaded": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("GIGAHANDS_CANONICAL_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
