#!/usr/bin/env python3
"""Build the shared sample/asset index for the six active dataset roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_catalog import DATASET_NAMES, dataset_root
from trajectory_schema import audit_trajectory


def _asset(root: Path, path: str, *, scale: float = 1.0, part: str = "root", index: int = 0) -> dict:
    target = root / path
    if not target.is_file():
        raise FileNotFoundError(target)
    return {"path": path, "scale": scale, "part": part, "object_index": index}


def _objects(dataset: str, root: Path, sample_id: str, meta: dict, giga_meshes: dict, supp_objects: dict) -> list[dict]:
    objects = meta.get("objects") or []
    if dataset == "hocap":
        return [_asset(root, f"assets/objects/{objects[0]['object_id']}/cleaned_mesh_10000.obj")]
    if dataset == "arctic":
        object_id = objects[0]["object_id"]
        return [
            _asset(root, f"assets/object_vtemplates/{object_id}/{part}.obj", scale=0.001, part=part)
            for part in ("bottom", "top")
        ]
    if dataset == "taco":
        return [
            _asset(root, f"raw/object_models/{info['object_id']}_cm.obj", scale=0.01,
                   part=info["role"], index=index)
            for index, info in enumerate(objects)
        ]
    if dataset == "gigahands":
        return [_asset(root, giga_meshes[sample_id])]
    if dataset == "dexterhand":
        mesh = "visual_rest.obj" if objects[0]["object_id"] == "RubiksCube" else "visual.obj"
        return [_asset(root, f"assets/objects/{meta['source_session']}/{mesh}")]
    if dataset == "supp":
        return [
            _asset(root, value["path"].removeprefix("datasets/supp_v1/"),
                   scale=float(value.get("scale", 1.0)), part=value.get("part", "root"),
                   index=int(value.get("object_index", 0)))
            for value in supp_objects[sample_id]
        ]
    raise ValueError(dataset)


def build_index(dataset: str) -> dict:
    root = dataset_root(dataset)
    paths = sorted((root / "canonical").glob("*/trajectory.npz"))
    if not paths:
        raise FileNotFoundError(f"No active canonical trajectories in {root}")
    giga_meshes = {}
    if dataset == "gigahands":
        source = json.loads((root / "raw/conversion_manifest.json").read_text())
        giga_meshes = {row["candidate_id"]: row["mesh_file"] for row in source["records"]}
    supp_objects = {}
    if dataset == "supp":
        source = json.loads((root / "raw/source_manifest.json").read_text())
        supp_rows = source["groups"]["supp"] + source.get("diagnostic_groups", {}).get("supp", [])
        supp_objects = {row["sample_id"]: row["objects"] for row in supp_rows}
    samples = []
    for path in paths:
        audit = audit_trajectory(path)
        if not audit["ok"]:
            raise ValueError(f"{path}: {audit['error']}")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            valid = np.asarray(data["hand_valid"])
            sides = [side for index, side in enumerate(("left", "right")) if valid[:, index].any()]
            fps = float(data["fps"])
        sample_id = path.parent.name
        assets = _objects(dataset, root, sample_id, meta, giga_meshes, supp_objects)
        samples.append({
            "sample_id": sample_id,
            "trajectory": str(path.relative_to(root)),
            "frames": audit["frames"],
            "fps": fps,
            "hand_sides": sides,
            "objects": assets,
            "source_dataset": meta.get("source_dataset", meta.get("dataset", dataset)),
            "readiness": (
                "reference_only" if dataset == "supp"
                else "candidate_needs_validation" if dataset in ("gigahands", "dexterhand")
                else "canonical_needs_task_validation"
            ),
        })
    return {"schema": "dexcodesign.dataset_index.v1", "dataset": dataset, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help=f"Subset of {', '.join(DATASET_NAMES)}")
    parser.add_argument("--write", action="store_true", help="Write index.json after strict trajectory/asset checks")
    args = parser.parse_args()
    for dataset in args.datasets or DATASET_NAMES:
        if dataset not in DATASET_NAMES:
            parser.error(f"Unknown dataset {dataset!r}; choose from {DATASET_NAMES}")
        index = build_index(dataset)
        print(f"{dataset}: {len(index['samples'])} valid trajectories with resolvable object meshes")
        if args.write:
            path = dataset_root(dataset) / "index.json"
            path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
