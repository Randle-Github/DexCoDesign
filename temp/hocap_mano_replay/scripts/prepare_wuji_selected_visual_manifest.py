#!/usr/bin/env python3
"""Export one searched WUJI morphology with complete visual geometry.

The batched search stores an exact parametric collision manifest and a fixed
retarget reference, but intentionally does not materialize 4096 visual meshes.
This utility reconstructs only the selected vector after search and reuses the
stored fixed reference without performing retargeting again.
"""

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
sys.path.insert(0, str(SCRIPT_ROOT))
from evaluate_wuji_morphology import vector_to_graph  # noqa: E402


def run(command: list[str], log: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("search_manifest", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fixed-reference",
        type=Path,
        help=(
            "local override for a downloaded search manifest whose recorded "
            "reference path belongs to another machine"
        ),
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="export collision-only URDF/USD for the physical capture scene",
    )
    args = parser.parse_args()

    source = args.search_manifest.expanduser().resolve()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    count = len(manifest["candidate_ids"])
    if not 0 <= args.index < count:
        raise IndexError(f"index {args.index} outside [0, {count})")

    source_hand_id = str(manifest["candidate_ids"][args.index])
    # Downstream one-candidate parametric preparation numbers its sole row 0.
    # Use the same canonical id so the compiled graph lookup is unambiguous.
    hand_id = "wuji_physx_000000"
    vector = np.asarray(manifest["vectors"][args.index], dtype=np.float64)
    np.save(root / "vectors.npy", vector[None])
    graph = vector_to_graph(vector, hand_id)
    graphs_path = root / "graphs.json"
    graphs_path.write_text(
        json.dumps({"schema_version": 1, "hands": [graph]}, indent=2) + "\n",
        encoding="utf-8",
    )
    compiled_root = root / "compiled"
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/dexcodesign/compile_hand_graph_batch.py"),
            str(graphs_path),
            "--output-dir",
            str(compiled_root),
            "--seed",
            "0",
            "--workers",
            "1",
        ],
        root / "compile.log",
    )

    runtime = root / "runtime"
    export_command = [
            sys.executable,
            str(SCRIPT_ROOT / "export_compiled_hand_urdf.py"),
            str(compiled_root / "compiled_hands.json"),
            "--compiled-hand-id",
            hand_id,
            "--output-root",
            str(runtime),
            "--hand-id",
            hand_id,
            "--display-name",
            hand_id,
        ]
    if args.physics_only:
        export_command.append("--physics-only")
    run(
        export_command,
        root / "export.log",
    )
    raw_urdf = runtime / hand_id / "left" / "hand.urdf"
    usd = root / "asset" / "hand.usd"
    fixed_reference = (
        args.fixed_reference
        if args.fixed_reference is not None
        else Path(
            manifest.get("fixed_reference")
            or manifest["reference_paths"][args.index]
        )
    ).expanduser().resolve()
    # Repackage the already-fixed command trajectory into the generated-hand
    # wrapper format (6-DoF virtual wrist + contact/body metadata).  This is a
    # representation conversion only: no IK or retarget optimization runs.
    with np.load(fixed_reference) as reference:
        reference_joint_names = reference["joint_names"].astype(str).tolist()
        reference_hand_q = reference["hand_q"].astype(np.float32)
    finger_columns = {
        name.removeprefix("finger__"): index
        for index, name in enumerate(reference_joint_names)
        if name.startswith("finger__")
    }
    import xml.etree.ElementTree as ET

    raw_joint_names = [
        str(joint.get("name"))
        for joint in ET.parse(raw_urdf).getroot().findall("joint")
        if joint.get("type") != "fixed"
    ]
    missing_joints = sorted(set(raw_joint_names) - set(finger_columns))
    if missing_joints:
        raise ValueError(
            f"fixed reference lacks generated finger joints: {missing_joints}"
        )
    root_columns = [reference_joint_names.index(name) for name in (
        "root_pos_x", "root_pos_y", "root_pos_z",
        "root_rot_x", "root_rot_y", "root_rot_z",
    )]
    qpos = np.stack(
        [reference_hand_q[:, finger_columns[name]] for name in raw_joint_names],
        axis=1,
    )
    wrist_position = reference_hand_q[:, root_columns[:3]]
    wrist_quaternion = Rotation.from_euler(
        "XYZ", reference_hand_q[:, root_columns[3:]]
    ).as_quat().astype(np.float32)
    trajectory = root / "fixed_command_trajectory.npz"
    np.savez_compressed(
        trajectory,
        qpos=qpos,
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=wrist_quaternion,
        qpos_ids=np.arange(len(raw_joint_names), dtype=np.int64),
    )
    wrapped_root = root / "wrapped"
    run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "prepare_generated_hand_rl_reference.py"),
            str(runtime),
            str(trajectory),
            "--output-dir",
            str(wrapped_root),
            "--capture",
            str(fixed_reference),
        ],
        root / "wrap.log",
    )
    urdf = wrapped_root / hand_id / "hand_rl.urdf"
    wrapped_reference = wrapped_root / hand_id / "reference.npz"
    output = {
        "schema_version": 1,
        "candidate_ids": [hand_id],
        "vectors": [vector.tolist()],
        "hand_urdf_paths": [str(urdf)],
        "hand_usd_paths": [str(usd)],
        "reference_paths": [str(wrapped_reference)],
        "all_candidates_require_physical_rollout": True,
        "runtime_parametric_overlays": False,
        "fixed_reference": str(wrapped_reference),
        "retarget_performed": False,
        "proxy_used": False,
        "complete_visual_geometry": not args.physics_only,
        "physics_only": args.physics_only,
        "selected_search_index": args.index,
        "selected_search_candidate_id": source_hand_id,
    }
    output_path = root / "physx_batch_manifest.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        "WUJI_SELECTED_VISUAL_MANIFEST_PREPARED "
        f"candidate={hand_id} index={args.index} manifest={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
