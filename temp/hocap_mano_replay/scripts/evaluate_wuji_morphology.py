#!/usr/bin/env python3
"""Compile, MANO-command retarget, and physically score one WUJI morphology."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
SOURCE_HAND_ID = "wuji_hand_2"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
VECTOR_NAMES = (
    "palm_expansion",
    "palm_scale_x",
    "palm_scale_z",
    "palm_yaw",
    *(f"{finger}_length" for finger in FINGERS),
    *(f"{finger}_radius" for finger in FINGERS),
)
LOWER_BOUNDS = np.asarray(
    [0.0, 0.90, 0.90, -0.12, *([0.82] * 5), *([0.85] * 5)],
    dtype=np.float64,
)
UPPER_BOUNDS = np.asarray(
    [0.35, 1.12, 1.12, 0.12, *([1.20] * 5), *([1.15] * 5)],
    dtype=np.float64,
)
SOURCE_VECTOR = np.asarray(
    [0.0, 1.0, 1.0, 0.0, *([1.0] * 5), *([1.0] * 5)],
    dtype=np.float64,
)

sys.path.insert(0, str(SCRIPT_ROOT))
import retarget_all_hands as retarget  # noqa: E402
import run_generated_hand_retarget_physics as generated  # noqa: E402
import simulate_retargeted_all_hands as physics  # noqa: E402


def vector_to_graph(vector: np.ndarray, hand_id: str) -> dict[str, object]:
    if vector.shape != (len(VECTOR_NAMES),):
        raise ValueError(
            f"expected {len(VECTOR_NAMES)} reshape values, got {vector.shape}"
        )
    if np.any(vector < LOWER_BOUNDS - 1e-6) or np.any(vector > UPPER_BOUNDS + 1e-6):
        raise ValueError("reshape vector lies outside the certified search bounds")
    # GPU proposal vectors are float32. Values sampled exactly on a certified
    # boundary can differ from the float64 constant by a few ulps.
    vector = np.clip(vector, LOWER_BOUNDS, UPPER_BOUNDS)
    finger_length = vector[4:9]
    finger_radius = vector[9:14]
    return {
        "schema_version": 1,
        "hand_id": hand_id,
        "source_hand": SOURCE_HAND_ID,
        "palm": {
            "layout_mode": "anthropomorphic",
            "expansion": float(vector[0]),
            "scale_x": float(vector[1]),
            "scale_z": float(vector[2]),
            "yaw": float(vector[3]),
        },
        "fingers": {
            "default_length_scale": 1.0,
            "default_radius_scale": 1.0,
            **{
                finger: {
                    "length_scale": float(finger_length[index]),
                    "radius_scale": float(finger_radius[index]),
                }
                for index, finger in enumerate(FINGERS)
            },
        },
    }


def prepare_source_runtime(output_dir: Path) -> Path:
    runtime_root = output_dir / "runtime"
    hand_id = "wuji_source"
    destination = runtime_root / hand_id / "left"
    if not destination.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(DIRECT_ROOT / SOURCE_HAND_ID / "left", destination)
    registry = json.loads(
        (DIRECT_ROOT / "registry.json").read_text(encoding="utf-8")
    )
    entry = registry["hands"][SOURCE_HAND_ID]["entries"]["left"]
    metadata = {
        "hand_id": hand_id,
        "display_name": "WUJI v2 Source",
        "source_hand": SOURCE_HAND_ID,
        "urdf": str(destination / "hand.urdf"),
        "tips": retarget.TIP_LINKS[SOURCE_HAND_ID],
        "active_dofs": int(entry["active_dofs"]),
        "passive_mimic_dofs": int(entry["passive_mimic_dofs"]),
        "all_parts_have_visual_and_collision": True,
    }
    (runtime_root / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return runtime_root


def prepare_generated_runtime(
    output_dir: Path,
    vector: np.ndarray,
    hand_id: str,
) -> tuple[Path, dict[str, object]]:
    graph = vector_to_graph(vector, hand_id)
    graph_path = output_dir / "reshape_graph.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    compiled_root = output_dir / "compiled"
    reference_graph = (
        REPO_ROOT / "artifacts" / "hand_morphology" / "reference_graphs.json"
    )
    rebuild_preprocess = True
    if reference_graph.is_file():
        payload = json.loads(reference_graph.read_text(encoding="utf-8"))
        source = next(
            (
                hand
                for hand in payload.get("hands", [])
                if hand.get("hand_id") == SOURCE_HAND_ID
            ),
            None,
        )
        rebuild_preprocess = (
            source is None
            or not (
                "canonicalization" in source
                or "direct_geometry_audit" in source
            )
        )
    compile_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dexcodesign" / "compile_hand_graph.py"),
        str(graph_path),
        "--output-dir",
        str(compiled_root),
        "--seed",
        "0",
    ]
    if rebuild_preprocess:
        compile_command.append("--rebuild-preprocess")
    subprocess.run(
        compile_command,
        cwd=REPO_ROOT,
        check=True,
    )
    runtime_root = output_dir / "runtime"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "export_compiled_hand_urdf.py"),
            str(compiled_root / "compiled_hands.json"),
            "--output-root",
            str(runtime_root),
            "--hand-id",
            hand_id,
            "--display-name",
            f"WUJI Morphology {hand_id}",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return runtime_root, graph


def evaluate(
    output_dir: Path,
    vector: np.ndarray,
    *,
    source: bool,
    iterations: int,
    render_video: bool,
    reuse_ik: bool,
    initial_trajectory: Path | None = None,
    prepared_runtime: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_id = "wuji_source" if source else output_dir.name.replace("-", "_")
    if prepared_runtime is not None:
        runtime_root = prepared_runtime.resolve()
        graph_path = output_dir / "reshape_graph.json"
        graph = (
            json.loads(graph_path.read_text(encoding="utf-8"))
            if graph_path.is_file()
            else None
        )
    elif source:
        runtime_root = prepare_source_runtime(output_dir)
        graph = None
    else:
        runtime_root, graph = prepare_generated_runtime(output_dir, vector, hand_id)

    metadata, trajectory = generated.prepare_retarget(
        runtime_root,
        iterations=iterations,
        reuse_ik=reuse_ik,
        initial_trajectory=initial_trajectory,
    )
    physics.PHYSICS_CACHE = output_dir / "physics_cache"
    capture = np.load(physics.DEFAULT_CAPTURE)
    result = physics.simulate_hand(
        str(metadata["hand_id"]),
        str(metadata["display_name"]),
        trajectory,
        capture["object_pose_wxyz"].astype(np.float64),
        output_dir / "rollout",
        30,
        960,
        720,
        render_video=render_video,
    )
    payload = {
        "source": source,
        "source_hand": SOURCE_HAND_ID,
        "reference": {
            "hand": "pre-RL MANO hand_ctrl forward kinematics",
            "object": str(physics.DEFAULT_CAPTURE),
            "rl_used": False,
        },
        "reshape_vector_names": list(VECTOR_NAMES),
        "reshape_vector": vector.tolist(),
        "graph": graph,
        "retargeted_trajectory": str(trajectory),
        "rollout": result,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector", nargs=len(VECTOR_NAMES), type=float)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--reuse-ik", action="store_true")
    parser.add_argument("--initial-trajectory", type=Path)
    parser.add_argument(
        "--prepared-runtime",
        type=Path,
        help="reuse a runtime exported from a shared batch compilation",
    )
    args = parser.parse_args()
    if args.source:
        vector = SOURCE_VECTOR.copy()
    elif args.vector is None:
        parser.error("--vector is required unless --source is set")
    else:
        vector = np.asarray(args.vector, dtype=np.float64)
    payload = evaluate(
        args.output_dir,
        vector,
        source=args.source,
        iterations=args.iterations,
        render_video=args.render_video,
        reuse_ik=args.reuse_ik,
        initial_trajectory=args.initial_trajectory,
        prepared_runtime=args.prepared_runtime,
    )
    print(json.dumps(payload["rollout"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
