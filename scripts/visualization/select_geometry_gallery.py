"""Choose diverse canonical hand/object trajectories for CPU geometry previews.

Selection is deterministic and does not modify the datasets.  The resulting
manifest records every source path, mesh, frame window and selection reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def metadata(path: Path) -> tuple[dict, int, float]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"])), len(data["frame_indices"]), float(data["fps"])


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def greedy(records: list[dict], count: int, keys: tuple[str, ...], seeds: list[str] = ()) -> list[dict]:
    selected: list[dict] = []
    remaining = records[:]
    for seed in seeds:
        found = next((r for r in remaining if r["sample_id"] == seed), None)
        if found:
            selected.append(found)
            remaining.remove(found)
    while remaining and len(selected) < count:
        seen = {key: {r.get(key) for r in selected} for key in keys}
        best = max(
            remaining,
            key=lambda r: (
                sum((len(keys) - i) * (r.get(key) not in seen[key]) for i, key in enumerate(keys)),
                r.get("quality", 0.0),
                -abs(r["frames"] - 120),
                r["sample_id"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def taco() -> list[dict]:
    root = ROOT / "datasets/taco_v1"
    records = []
    for index, line in enumerate((root / "manifests/selected_100.jsonl").read_text().splitlines()):
        source = json.loads(line)
        path = root / f"canonical/{index:03d}/trajectory.npz"
        if not path.is_file():
            continue
        meta, frames, fps = metadata(path)
        objects = []
        for object_index, info in enumerate(meta["objects"]):
            mesh = root / f"raw/object_models/{info['object_id']}_cm.obj"
            if mesh.is_file():
                objects.append({"path": rel(mesh), "scale": 0.01, "object_index": object_index, "part": info["role"]})
        if len(objects) != len(meta["objects"]):
            continue
        records.append({
            "dataset": "taco", "sample_id": f"{index:03d}_{source['action'].replace(' ', '_')}",
            "trajectory": rel(path), "frames": frames, "fps": fps,
            "action": source["action"], "tool": source["tool_category"], "target": source["target_category"],
            "quality": float(source.get("metrics", {}).get("score", 0)), "objects": objects,
            "clip_start": 0, "clip_stop": frames, "ground": "unverified_source_world",
        })
    # The previously inspected hammer/box sequence is a control.
    return greedy(records, 10, ("action", "tool", "target"), ["065_hit"])


def arctic() -> list[dict]:
    root = ROOT / "datasets/arctic_v1"
    records = []
    for path in sorted((root / "canonical").glob("*/trajectory.npz")):
        meta, frames, fps = metadata(path)
        object_id = meta["objects"][0]["object_id"]
        objects = [
            {"path": rel(root / f"assets/object_vtemplates/{object_id}/{part}.obj"), "scale": 0.001,
             "object_index": 0, "part": part}
            for part in ("bottom", "top")
        ]
        if not all((ROOT / spec["path"]).is_file() for spec in objects):
            continue
        # Initial/final frames have hands well outside the workspace; keep
        # the complete interaction interval while recording the crop.
        start, stop = min(10, frames - 1), max(min(frames - 10, frames), 11)
        records.append({
            "dataset": "arctic", "sample_id": path.parent.name, "trajectory": rel(path),
            "frames": frames, "fps": fps, "action": object_id, "objects": objects,
            "clip_start": start, "clip_stop": stop, "ground": "world_zero",
            "tabletop_audit_caveat": object_id in {"capsulemachine", "microwave", "notebook"},
        })
    return records[:10]


def hocap() -> list[dict]:
    root = ROOT / "datasets/hocap_v1/canonical"
    records = []
    for path in sorted(root.glob("*/trajectory.npz")):
        meta, frames, fps = metadata(path)
        object_id = meta["objects"][0]["object_id"]
        mesh = ROOT / f"datasets/hocap_v1/assets/objects/{object_id}/cleaned_mesh_10000.obj"
        if not mesh.is_file():
            continue
        records.append({
            "dataset": "hocap", "sample_id": path.parent.name, "trajectory": rel(path),
            "frames": frames, "fps": fps, "action": path.parent.name, "objects": [
                {"path": rel(mesh), "scale": 1.0, "object_index": 0, "part": object_id}
            ], "clip_start": 0, "clip_stop": frames, "ground": "unverified_source_world",
        })
    return records[:10]


def moving_window(path: Path, seconds: float = 6.0) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        pose = data["object_root_pose_wxyz"][:, 0]
        fps = float(data["fps"])
        valid = data["hand_valid"]
        hands = data["hand_joints_m"]
    width = min(len(pose), max(2, round(seconds * fps)))
    if len(pose) == width:
        return 0, len(pose)
    starts = np.arange(0, len(pose) - width, max(1, width // 4))
    displacement = np.linalg.norm(pose[starts + width - 1, :3] - pose[starts, :3], axis=1)
    # Prefer motion while either valid hand is actually near the object.
    # DexterHand sessions include stretches with missing (NaN) joints; feeding
    # those gaps to argmax silently selects a non-contact interval.
    midpoint = starts + width // 2
    gap = np.linalg.norm(hands[midpoint] - pose[midpoint, None, None, :3], axis=-1)
    gap = np.where(valid[midpoint, :, None] & np.isfinite(gap), gap, np.inf)
    nearest = gap.min(axis=(1, 2))
    finite = np.isfinite(nearest)
    if not finite.any():
        raise ValueError(f"No valid hand positions in {path}")
    proximity_penalty = np.maximum(nearest - 0.07, 0.0)
    score = displacement - 2.0 * proximity_penalty
    score[~finite] = -np.inf
    start = int(starts[int(np.argmax(score))])
    return start, start + width


def dexterhand() -> list[dict]:
    root = ROOT / "datasets/dexterhand_v1"
    records = []
    for path in sorted((root / "canonical").glob("*/trajectory.npz")):
        meta, frames, fps = metadata(path)
        session = meta["source_session"]
        object_id = meta["objects"][0]["object_id"]
        mesh_name = "visual_rest.obj" if object_id == "RubiksCube" else "visual.obj"
        mesh = root / "assets/objects" / session / mesh_name
        if not mesh.is_file():
            continue
        start, stop = moving_window(path)
        records.append({
            "dataset": "dexterhand", "sample_id": path.parent.name, "trajectory": rel(path),
            "frames": frames, "fps": fps, "action": object_id, "objects": [
                {"path": rel(mesh), "scale": 1.0, "object_index": 0, "part": object_id}
            ], "clip_start": start, "clip_stop": stop, "ground": "source_z_down_support",
            "articulation_caveat": object_id == "RubiksCube",
        })
    return records[:10]


def gigahands() -> list[dict]:
    root = ROOT / "datasets/gigahands_v1"
    manifest = json.loads((root / "canonical/manifest.json").read_text())
    records = []
    for source in manifest["records"]:
        path = root / "canonical" / source["path"]
        mesh = root / source["mesh_file"]
        if not path.is_file() or not mesh.is_file():
            continue
        meta, frames, fps = metadata(path)
        if frames < 40:
            continue
        records.append({
            "dataset": "gigahands", "sample_id": source["candidate_id"],
            "trajectory": rel(path), "frames": frames, "fps": fps,
            "action": source["action_group"], "scene": source["scene_category"],
            "object": source["object_folder"], "action_text": source["action_text"],
            "quality": -float(source.get("minimum_hand_to_mesh_bbox_gap_m", 1.0)),
            "objects": [{"path": rel(mesh), "scale": 1.0, "object_index": 0, "part": source["object_folder"]}],
            "clip_start": 0, "clip_stop": frames, "ground": "unverified_source_world",
        })
    return greedy(records, 10, ("action", "scene", "object"), ["GIGA_0207_p048-sandwich_0013"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = {name: function() for name, function in (
        ("taco", taco), ("arctic", arctic), ("hocap", hocap),
        ("dexterhand", dexterhand), ("gigahands", gigahands),
    )}
    payload = {"schema": "dexcodesign.geometry_gallery.v1", "groups": grouped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for name, records in grouped.items():
        print(f"{name}: {len(records)} selected", flush=True)
        for record in records:
            print(f"  {record['sample_id']} | {record['action']} | {record['frames']} frames", flush=True)


if __name__ == "__main__":
    main()
