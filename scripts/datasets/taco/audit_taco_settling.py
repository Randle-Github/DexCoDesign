#!/usr/bin/env python3
"""Check whether selected TACO objects remain stable on a MuJoCo table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _model(mesh_path: Path) -> mujoco.MjModel:
    xml = f"""
    <mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <asset><mesh name="object" file="{mesh_path.resolve()}" scale="0.01 0.01 0.01"/></asset>
      <worldbody>
        <geom type="plane" size="1 1 0.05" friction="1 0.01 0.001"/>
        <body name="object"><freejoint/>
          <geom type="mesh" mesh="object" density="500" friction="1 0.01 0.001" condim="4"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/taco_v1"))
    parser.add_argument("--steps", type=int, default=750, help="0.002-second MuJoCo steps")
    parser.add_argument("--max-rotation-deg", type=float, default=15.0)
    parser.add_argument("--max-drift-m", type=float, default=0.03)
    args = parser.parse_args()

    root = args.root.resolve()
    selected = _load_jsonl(root / "manifests" / "selected_100.jsonl")
    mesh_root = root / "raw" / "object_models"
    pose_root = root / "raw" / "object_poses"
    model_cache: dict[str, mujoco.MjModel] = {}
    vertices_cache: dict[str, np.ndarray] = {}
    records = []

    for sequence in selected:
        objects = []
        for pose_path in sorted((pose_root / sequence["sequence_id"]).glob("*.npy")):
            role, object_id = pose_path.stem.split("_", 1)
            mesh_path = mesh_root / f"{object_id}_cm.obj"
            try:
                if object_id not in model_cache:
                    model_cache[object_id] = _model(mesh_path)
                    mesh = trimesh.load(mesh_path, force="mesh", process=False)
                    vertices_cache[object_id] = np.asarray(mesh.vertices, dtype=np.float64) * 0.01

                pose = np.load(pose_path, allow_pickle=False)[0]
                rotation = pose[:3, :3].astype(np.float64)
                vertices = vertices_cache[object_id]
                height = -float((vertices @ rotation.T)[:, 2].min()) + 0.001
                quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
                quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]

                model = model_cache[object_id]
                data = mujoco.MjData(model)
                data.qpos[:3] = [0.0, 0.0, height]
                data.qpos[3:7] = quaternion_wxyz
                mujoco.mj_forward(model, data)
                initial_rotation = Rotation.from_quat(quaternion_xyzw)
                for _ in range(args.steps):
                    mujoco.mj_step(model, data)

                final_rotation = Rotation.from_quat(
                    [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]
                )
                rotation_change = float((final_rotation * initial_rotation.inv()).magnitude())
                drift = float(np.linalg.norm(data.qpos[:2]))
                passed = bool(
                    np.isfinite(data.qpos).all()
                    and rotation_change <= np.deg2rad(args.max_rotation_deg)
                    and drift <= args.max_drift_m
                )
                objects.append(
                    {
                        "role": role,
                        "object_id": object_id,
                        "rotation_change_deg": float(np.rad2deg(rotation_change)),
                        "horizontal_drift_m": drift,
                        "passed": passed,
                    }
                )
            except Exception as error:  # preserve the failure in the audit manifest
                objects.append(
                    {"role": role, "object_id": object_id, "passed": False, "error": str(error)}
                )
        records.append(
            {
                "sequence_id": sequence["sequence_id"],
                "passed": bool(objects) and all(item["passed"] for item in objects),
                "objects": objects,
            }
        )

    output = root / "manifests" / "settling_audit.jsonl"
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path = root / "manifests" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed = [record["sequence_id"] for record in records if not record["passed"]]
    summary["settling_audit"] = {
        "simulator": f"MuJoCo {mujoco.__version__}",
        "duration_s": args.steps * 0.002,
        "max_rotation_deg": args.max_rotation_deg,
        "max_drift_m": args.max_drift_m,
        "passed": len(records) - len(failed),
        "failed": len(failed),
        "failed_sequence_ids": failed,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["settling_audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
