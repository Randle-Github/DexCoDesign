#!/usr/bin/env python3
"""Prepare skeleton/reference manifests for parametric WUJI USD validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from export_compiled_hand_urdf import _inverse_source_linear


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
    parser.add_argument("--direct-template", action="store_true")
    parser.add_argument("--direct-candidate-assets", type=Path)
    parser.add_argument("--prototype-bank-manifest", type=Path)
    parser.add_argument("--prototype-bank-compiled", type=Path)
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
    link_translations = []
    joint_names = []
    joint_local_positions = []
    bank_rows: dict[int, dict[str, str]] | None = None
    bank_by_id: dict[str, dict[str, object]] | None = None
    if (args.prototype_bank_manifest is None) != (
        args.prototype_bank_compiled is None
    ):
        raise ValueError("both prototype-bank arguments are required together")
    if args.prototype_bank_manifest is not None:
        bank_manifest = json.loads(
            args.prototype_bank_manifest.read_text(encoding="utf-8")
        )
        bank_compiled = json.loads(
            args.prototype_bank_compiled.read_text(encoding="utf-8")
        )
        bank_rows = {
            int(round(vector[0])): {
                "candidate_id": str(candidate_id),
                "usd": str(usd),
            }
            for vector, candidate_id, usd in zip(
                bank_manifest["vectors"],
                bank_manifest["candidate_ids"],
                bank_manifest["hand_usd_paths"],
                strict=True,
            )
        }
        bank_by_id = {
            str(hand["hand_id"]): hand for hand in bank_compiled["hands"]
        }
        if set(bank_rows) != set(range(32)):
            raise ValueError("prototype bank must contain indices 0 through 31")
    baseline = by_id[candidate_ids[0]]
    source_scale, source_rotation = _inverse_source_linear(
        str(baseline["seed_source"])
    )
    reflection = np.diag((-1.0, 1.0, 1.0))
    polar_linear = source_rotation.T @ reflection
    for index, candidate_id in enumerate(candidate_ids):
        candidate = by_id[candidate_id]
        prototype_row = (
            None if bank_rows is None else bank_rows[int(round(vectors[index, 0]))]
        )
        prototype = (
            baseline
            if prototype_row is None
            else bank_by_id[prototype_row["candidate_id"]]
        )
        prototype_linear = {
            int(part["id"]): np.asarray(part["mesh_linear"], dtype=np.float64)
            for part in prototype["parts"]
        }
        template_usd = (
            args.template_usd.resolve()
            if prototype_row is None
            else Path(prototype_row["usd"]).resolve()
        )
        candidate_root = root / "candidates" / candidate_id
        metadata = {
            "link_names": {
                str(part["id"]): f"part_{int(part['id']):02d}_{part['role']}"
                for part in candidate["parts"]
            }
        }
        reference = candidate_root / "reference.npz"
        save_reference(
            args.template_reference.resolve(),
            reference,
            candidate_id,
            qpos[index],
            wrist_position[index],
            wrist_quaternion[index],
        )
        urdfs.append("")
        candidate_usd = candidate_root / "asset" / "hand.usd"
        if not args.direct_template and args.direct_candidate_assets is None:
            candidate_usd.parent.mkdir(parents=True, exist_ok=True)
            if bank_rows is not None:
                # Sublayer the prototype from its original directory so USD
                # resolves and caches its heavy configuration/mesh layers by
                # one canonical identifier. Copying the top layer made every
                # candidate's relative ``configuration/`` path unique and
                # forced Kit to parse the same 4.6 MB layer thousands of
                # times. Candidate-local transform opinions are authored into
                # this tiny wrapper by ``attach_manifest``.
                prototype_root = f"{prototype_row['candidate_id']}_rl"
                prototype_asset = template_usd.as_posix().replace("@", "@@")
                candidate_usd.write_text(
                    "#usda 1.0\n"
                    "(\n"
                    f'    defaultPrim = "{prototype_root}"\n'
                    f"    subLayers = [@{prototype_asset}@]\n"
                    ")\n",
                    encoding="utf-8",
                )
            else:
                shutil.copy2(template_usd, candidate_usd)
                for child in template_usd.parent.iterdir():
                    if child.name == template_usd.name:
                        continue
                    destination = candidate_usd.parent / child.name
                    if not destination.exists() and not destination.is_symlink():
                        os.symlink(
                            child,
                            destination,
                            target_is_directory=child.is_dir(),
                        )
        usds.append(
            str(
                (
                    args.direct_candidate_assets.resolve()
                    / "candidates"
                    / candidate_id
                    / "asset"
                    / "hand.usd"
                )
                if args.direct_candidate_assets is not None
                else (
                    template_usd
                    if args.direct_template
                    else candidate_usd
                )
            )
        )
        references.append(str(reference))
        ordered_links = [
            metadata["link_names"][str(part["id"])]
            for part in candidate["parts"]
        ]
        link_names.append(ordered_links)
        transforms = []
        candidate_world_positions: dict[int, np.ndarray] = {}
        candidate_joint_names = []
        candidate_joint_positions = []
        for part in candidate["parts"]:
            part_id = int(part["id"])
            current = np.asarray(part["mesh_linear"], dtype=np.float64)
            # The compiler uses row vectors: Vc = V @ L.T.  The runtime URDF
            # exporter then applies P = source_rotation.T @ reflection, so
            # Ve = V @ L.T @ P / scale.  Author the exact relative affine in
            # that exported link frame; the source similarity scale cancels.
            baseline_export = prototype_linear[part_id].T @ polar_linear
            current_export = current.T @ polar_linear
            relative = np.linalg.solve(baseline_export, current_export)
            matrix = np.eye(4, dtype=np.float64)
            # ``relative`` acts on compiler/exporter row vectors. USD/Gf
            # xform matrices act on column vectors, hence the transpose.
            matrix[:3, :3] = relative.T
            transforms.append(matrix.tolist())
            local_position = (
                np.asarray(part["relative_pos"], dtype=np.float64)
                @ polar_linear
                / source_scale
            )
            parent = part["parent"]
            world_position = local_position.copy()
            if parent is not None:
                world_position += candidate_world_positions[int(parent)]
                candidate_joint_names.append(str(part["joint_name"]))
                candidate_joint_positions.append(local_position.tolist())
            candidate_world_positions[part_id] = world_position
        relative_transforms.append(transforms)
        link_translations.append(
            [
                candidate_world_positions[int(part["id"])].tolist()
                for part in candidate["parts"]
            ]
        )
        joint_names.append(candidate_joint_names)
        joint_local_positions.append(candidate_joint_positions)

    manifest = {
        "schema_version": 2,
        "candidate_ids": candidate_ids,
        "vectors": vectors.tolist(),
        "hand_urdf_paths": urdfs,
        "hand_usd_paths": usds,
        "reference_paths": references,
        "parametric_template_usd": (
            None
            if (
                args.direct_template
                or args.direct_candidate_assets is not None
                or bank_rows is not None
            )
            else str(args.template_usd.resolve())
        ),
        "parametric_template_usd_paths": (
            None
            if bank_rows is None
            else [
                str(Path(bank_rows[int(round(vector[0]))]["usd"]).resolve())
                for vector in vectors
            ]
        ),
        "palm_prototype_indices": [int(round(vector[0])) for vector in vectors],
        "parametric_link_names": link_names,
        "parametric_relative_transforms": relative_transforms,
        "parametric_link_translations": link_translations,
        "parametric_joint_names": joint_names,
        "parametric_joint_local_positions": joint_local_positions,
        "all_candidates_require_physical_rollout": True,
        "proxy_used": False,
    }
    path = root / "physx_batch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"WUJI_PARAMETRIC_SMOKE_PREPARED count={count} manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
