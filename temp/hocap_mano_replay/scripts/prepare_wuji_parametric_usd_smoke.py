#!/usr/bin/env python3
"""Prepare skeleton/reference manifests for parametric WUJI USD validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def save_reference(
    template_path: Path,
    output: Path,
    candidate_id: str,
    qpos: np.ndarray,
    wrist_position: np.ndarray,
    wrist_quaternion: np.ndarray,
) -> None:
    with np.load(template_path) as template:
        values = {key: template[key] for key in template.files}
    hand_q = np.asarray(values["hand_q"], dtype=np.float32).copy()
    if hand_q.shape[1] != 6 + qpos.shape[1]:
        raise ValueError(
            f"template/qpos mismatch: {hand_q.shape} versus {qpos.shape}"
        )
    hand_q[:, :3] = wrist_position
    hand_q[:, 3:6] = np.unwrap(
        Rotation.from_quat(wrist_quaternion).as_euler("XYZ"), axis=0
    ).astype(np.float32)
    hand_q[:, 6:] = qpos
    values["hand_id"] = np.asarray(candidate_id)
    values["display_name"] = np.asarray(candidate_id)
    values["hand_q"] = hand_q
    values["hand_ctrl"] = hand_q.copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("retarget_batch", type=Path)
    parser.add_argument("compiled_hands", type=Path)
    parser.add_argument("--template-usd", type=Path, required=True)
    parser.add_argument("--template-reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    compiled = json.loads(args.compiled_hands.read_text(encoding="utf-8"))
    by_id = {hand["hand_id"]: hand for hand in compiled["hands"]}
    with np.load(args.retarget_batch) as data:
        count = min(args.limit, len(data["vectors"]))
        vectors = data["vectors"][:count].astype(np.float64)
        qpos = data["qpos"][:count].astype(np.float32)
        wrist_position = data["wrist_position_all"][:count].astype(np.float32)
        wrist_quaternion = data["wrist_quaternion_xyzw_all"][:count].astype(
            np.float32
        )

    candidate_ids = [f"wuji_physx_{index:06d}" for index in range(count)]
    urdfs = []
    usds = []
    references = []
    link_names = []
    relative_transforms = []
    baseline = by_id[candidate_ids[0]]
    baseline_linear = {
        int(part["id"]): np.asarray(part["mesh_linear"], dtype=np.float64)
        for part in baseline["parts"]
    }
    for index, candidate_id in enumerate(candidate_ids):
        candidate = by_id[candidate_id]
        candidate_root = root / "candidates" / candidate_id
        runtime = candidate_root / "runtime"
        run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "export_compiled_hand_urdf.py"),
                str(args.compiled_hands.resolve()),
                "--compiled-hand-id",
                candidate_id,
                "--output-root",
                str(runtime),
                "--hand-id",
                candidate_id,
                "--display-name",
                candidate_id,
                "--skeleton-only",
            ]
        )
        metadata = json.loads(
            (runtime / "runtime_metadata.json").read_text(encoding="utf-8")
        )
        reference = candidate_root / "reference.npz"
        save_reference(
            args.template_reference.resolve(),
            reference,
            candidate_id,
            qpos[index],
            wrist_position[index],
            wrist_quaternion[index],
        )
        urdfs.append(
            str(runtime / candidate_id / "left" / "hand.urdf")
        )
        usds.append(str(candidate_root / "asset" / "hand.usd"))
        references.append(str(reference))
        ordered_links = [
            metadata["link_names"][str(part["id"])]
            for part in candidate["parts"]
        ]
        link_names.append(ordered_links)
        transforms = []
        for part in candidate["parts"]:
            part_id = int(part["id"])
            current = np.asarray(part["mesh_linear"], dtype=np.float64)
            # Canonical-frame relative transform. The identity candidate is
            # validated first; exporter-frame conjugation is added after it.
            relative = np.linalg.solve(baseline_linear[part_id], current)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = relative
            transforms.append(matrix.tolist())
        relative_transforms.append(transforms)

    manifest = {
        "schema_version": 2,
        "candidate_ids": candidate_ids,
        "vectors": vectors.tolist(),
        "hand_urdf_paths": urdfs,
        "hand_usd_paths": usds,
        "reference_paths": references,
        "parametric_template_usd": str(args.template_usd.resolve()),
        "parametric_link_names": link_names,
        "parametric_relative_transforms": relative_transforms,
        "all_candidates_require_physical_rollout": True,
        "proxy_used": False,
    }
    path = root / "physx_batch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"WUJI_PARAMETRIC_SMOKE_PREPARED count={count} manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
