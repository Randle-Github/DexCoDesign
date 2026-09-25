#!/usr/bin/env python3
"""Select and validate stable ARCTIC tabletop poses in Isaac Sim/PhysX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
parser.add_argument("--seconds", type=float, default=1.5)
parser.add_argument("--candidates", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def metadata(source) -> dict:
    value = source["metadata_json"]
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.apply_scale(0.001)
    return mesh


def stable_candidates(bottom_path: Path, top_path: Path, q: float, maximum: int):
    bottom, top = load_mesh(bottom_path), load_mesh(top_path)
    top_tf = np.eye(4)
    top_tf[:3, :3] = Rotation.from_rotvec([0.0, 0.0, -q]).as_matrix()
    top.apply_transform(top_tf)
    combined = trimesh.util.concatenate((bottom, top))
    bh, th = bottom.convex_hull, top.convex_hull
    center_mass = (bh.volume * bh.center_mass + th.volume * th.center_mass) / (bh.volume + th.volume)
    transforms, probabilities = trimesh.poses.compute_stable_poses(combined.convex_hull, center_mass=center_mass)
    result = []
    for index in np.argsort(-probabilities)[:maximum]:
        rotation = transforms[index, :3, :3]
        vertices = (rotation @ combined.vertices.T).T
        position = np.array([0.0, 0.0, -vertices[:, 2].min() + 0.002], dtype=np.float64)
        xyzw = Rotation.from_matrix(rotation).as_quat()
        result.append({
            "rotation": rotation, "position": position,
            "quat_wxyz": xyzw[[3, 0, 1, 2]], "probability": float(probabilities[index]),
        })
    return result


def quaternion_angle(q0: torch.Tensor, q1: torch.Tensor) -> float:
    dot = torch.clamp(torch.abs(torch.sum(q0 * q1)), 0.0, 1.0)
    return float((2.0 * torch.acos(dot)).item())


def main() -> None:
    root = args.root.resolve()
    q_by_object: dict[str, list[float]] = {}
    for path in sorted((root / "canonical_100").glob("*/trajectory.npz")):
        with np.load(path, allow_pickle=False) as source:
            oid = metadata(source)["objects"][0]["object_id"]
            q_by_object.setdefault(oid, []).extend(source["object_joint_positions_rad"].reshape(-1).tolist())

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=8, device=args.device))
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0))
    ground.func("/World/Ground", ground)

    candidates = {}
    entries = []
    spacing = 0.65
    for object_index, oid in enumerate(sorted(q_by_object)):
        q = float(np.median(q_by_object[oid]))
        asset = root / "assets" / "object_vtemplates" / oid
        candidates[oid] = stable_candidates(asset / "bottom.obj", asset / "top.obj", q, args.candidates)
        for rank, candidate in enumerate(candidates[oid]):
            origin = np.array([rank * spacing, object_index * spacing, 0.0])
            position = candidate["position"] + origin
            cfg = ArticulationCfg(
                prim_path=f"/World/Objects/{oid}_{rank:02d}",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str((root / "assets" / "object_urdf" / f"{oid}.urdf").resolve()),
                    fix_base=False, merge_fixed_joints=False, self_collision=False,
                    force_usd_conversion=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False, max_depenetration_velocity=1.0,
                        linear_damping=0.02, angular_damping=0.02),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False, solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(position.tolist()), rot=tuple(candidate["quat_wxyz"].tolist()),
                    joint_pos={"articulation": q},
                ),
                actuators={"joint": ImplicitActuatorCfg(
                    joint_names_expr=["articulation"], effort_limit_sim=50.0,
                    stiffness=40.0, damping=2.0)},
            )
            entries.append({
                "object_id": oid, "rank": rank, "q": q, "origin": origin,
                "candidate": candidate, "asset": Articulation(cfg),
            })

    sim.reset()
    for entry in entries:
        asset = entry["asset"]
        target = torch.tensor([[entry["q"]]], dtype=torch.float32, device=sim.device)
        asset.set_joint_position_target(target)
        asset.write_data_to_sim()

    for _ in range(int(args.seconds / sim.get_physics_dt())):
        for entry in entries:
            entry["asset"].write_data_to_sim()
        sim.step(render=False)
        for entry in entries:
            entry["asset"].update(sim.get_physics_dt())

    results = {}
    for oid in sorted(candidates):
        tested = []
        for entry in (item for item in entries if item["object_id"] == oid):
            asset, candidate, origin = entry["asset"], entry["candidate"], entry["origin"]
            final_pos = asset.data.root_pos_w[0].detach().cpu().numpy()
            final_quat = asset.data.root_quat_w[0]
            initial_pos = candidate["position"] + origin
            initial_quat = torch.tensor(candidate["quat_wxyz"], device=final_quat.device, dtype=final_quat.dtype)
            drift = float(np.linalg.norm(final_pos[:2] - initial_pos[:2]))
            dz = float(abs(final_pos[2] - initial_pos[2]))
            angle = quaternion_angle(initial_quat, final_quat)
            passed = bool(np.isfinite(final_pos).all() and drift <= 0.03 and dz <= 0.03 and angle <= np.deg2rad(15.0))
            tested.append({
                "rank": entry["rank"], "pass": passed,
                "horizontal_drift_m": drift, "vertical_change_m": dz,
                "rotation_change_deg": float(np.rad2deg(angle)),
                "stable_pose_probability": candidate["probability"],
            })
        passing = [item for item in tested if item["pass"]]
        selected = max(passing, key=lambda item: item["stable_pose_probability"]) if passing else min(
            tested, key=lambda item: item["horizontal_drift_m"] + item["vertical_change_m"] + np.deg2rad(item["rotation_change_deg"]))
        candidate = candidates[oid][selected["rank"]]
        results[oid] = {
            "pass": bool(selected["pass"]), "selected_rank": selected["rank"],
            "rotation_matrix": candidate["rotation"].tolist(),
            "quaternion_wxyz": candidate["quat_wxyz"].tolist(),
            "position_m": candidate["position"].tolist(),
            "audit_joint_position_rad": float(np.median(q_by_object[oid])),
            "selected_metrics": selected, "tested_candidates": tested,
        }
        print(f"ISAAC_TABLE_AUDIT {oid:16s} pass={selected['pass']} rank={selected['rank']} drift={selected['horizontal_drift_m']:.4f}m rot={selected['rotation_change_deg']:.2f}deg", flush=True)

    output = root / "manifests" / "tabletop_audit_isaac.json"
    output.write_text(json.dumps({
        "schema": "dexcodesign.arctic_tabletop_isaac.v1",
        "simulator": "Isaac Sim / PhysX", "simulated_seconds": args.seconds,
        "objects": results, "failed_objects": [name for name, value in results.items() if not value["pass"]],
    }, indent=2) + "\n")
    print(f"ARCTIC_ISAAC_TABLETOP_READY manifest={output}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
