#!/usr/bin/env python3
"""Evaluate every WUJI morphology in real batched Isaac Lab physics.

Each manifest row maps one independently generated hand USD and its retargeted
reference to one PhysX environment. No geometry/contact proxy or top-k filter
is used: every row executes the same C-error, binary pinch reward, contact
sensors, object dynamics, and termination code used by residual PPO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="DexCoDesign-Hand-Residual-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--convert-urdf",
    action="store_true",
    help="convert every manifest URDF before creating the PhysX batch",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.manifest = args_cli.manifest.expanduser().resolve()
args_cli.output = args_cli.output.expanduser().resolve()
os.environ["DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST"] = str(args_cli.manifest)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def attach_parametric_collisions(
    candidate_usd: Path,
    template_usd: Path,
    link_names: list[str],
    transforms: list[list[list[float]]],
) -> None:
    """Reference shared collision prototypes into a skeleton articulation."""
    candidate_stage = Usd.Stage.Open(str(candidate_usd))
    template_stage = Usd.Stage.Open(str(template_usd))
    candidate_root = candidate_stage.GetDefaultPrim().GetPath()
    template_root = template_stage.GetDefaultPrim().GetPath()
    if len(link_names) != len(transforms):
        raise ValueError("parametric link/transform count mismatch")
    for link_name, transform in zip(link_names, transforms, strict=True):
        candidate_path = candidate_root.AppendChild(link_name).AppendChild(
            "collisions"
        )
        # Isaac's earlier generated-hand exporter names the imported links
        # ``hand__part_*`` while the geometry-free URDF importer keeps the
        # canonical ``part_*`` names. Resolve that importer namespace only on
        # the shared template side; candidate articulation names stay intact.
        template_link = template_root.AppendChild(link_name)
        if not template_stage.GetPrimAtPath(template_link):
            matches = [
                child.GetPath()
                for child in template_stage.GetPrimAtPath(template_root).GetChildren()
                if child.GetName().endswith(f"__{link_name}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot uniquely map template link {link_name}: {matches}"
                )
            template_link = matches[0]
        template_path = template_link.AppendChild("collisions")
        template_collision = template_stage.GetPrimAtPath(template_path)
        if not template_collision:
            raise ValueError(f"template collision prim is missing: {template_path}")
        # A flattened Isaac asset keeps each collision mesh in an internal
        # /Flattened_Prototype_* prim and makes link/collisions instance it.
        # Referencing link/collisions from another layer would preserve that
        # absolute internal reference in the wrong composition namespace and
        # can stall PhysX parsing. Reference the self-contained prototype
        # directly instead.
        target_path = template_path
        reference_list = template_collision.GetMetadata("references")
        if reference_list is not None:
            items = list(reference_list.GetAddedOrExplicitItems())
            if len(items) == 1 and not items[0].assetPath and items[0].primPath:
                target_path = items[0].primPath
        collision = candidate_stage.OverridePrim(candidate_path)
        references = collision.GetReferences()
        references.ClearReferences()
        references.AddReference(str(template_usd), target_path)
        matrix = np.asarray(transform, dtype=np.float64)
        if not np.allclose(matrix, np.eye(4), atol=1.0e-12):
            xform = UsdGeom.Xformable(collision)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(Gf.Matrix4d(*matrix.reshape(-1).tolist()))
    candidate_stage.GetRootLayer().Save()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _experiment_cfg: dict) -> None:
    manifest = json.loads(args_cli.manifest.read_text(encoding="utf-8"))
    count = len(manifest["hand_usd_paths"])
    conversion_seconds = 0.0
    if args_cli.convert_urdf:
        convert_start = time.perf_counter()
        for index, (urdf_value, usd_value) in enumerate(
            zip(
                manifest["hand_urdf_paths"],
                manifest["hand_usd_paths"],
                strict=True,
            )
        ):
            urdf = Path(urdf_value).resolve()
            usd = Path(usd_value).resolve()
            usd.parent.mkdir(parents=True, exist_ok=True)
            converter_cfg = UrdfConverterCfg(
                asset_path=str(urdf),
                usd_dir=str(usd.parent),
                usd_file_name=usd.name,
                fix_base=True,
                merge_fixed_joints=False,
                force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=300.0,
                        damping=34.6410162,
                    ),
                    target_type="position",
                ),
            )
            converted = Path(UrdfConverter(converter_cfg).usd_path)
            if not converted.is_file():
                raise RuntimeError(f"URDF conversion did not create {converted}")
            if index == 0 or (index + 1) % 32 == 0 or index + 1 == count:
                print(
                    f"WUJI_PHYSX_USD_PROGRESS completed={index + 1}/{count}",
                    flush=True,
                )
        conversion_seconds = time.perf_counter() - convert_start
    template_value = manifest.get("parametric_template_usd")
    if template_value:
        attach_start = time.perf_counter()
        template_usd = Path(template_value).resolve()
        for index, usd_value in enumerate(manifest["hand_usd_paths"]):
            attach_parametric_collisions(
                Path(usd_value).resolve(),
                template_usd,
                manifest["parametric_link_names"][index],
                manifest["parametric_relative_transforms"][index],
            )
        print(
            "WUJI_PARAMETRIC_COLLISIONS_ATTACHED "
            f"completed={count}/{count} "
            f"seconds={time.perf_counter() - attach_start:.6f}",
            flush=True,
        )
    env_cfg.scene.num_envs = count
    env_cfg.scene.replicate_physics = False
    env_cfg.scene.clone_in_fabric = False
    env_cfg.randomize_start_phase = False
    env_cfg.episode_length_s = 15.0
    env_cfg.seed = args_cli.seed

    start = time.perf_counter()
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    initialization_seconds = time.perf_counter() - start
    actions = torch.zeros(
        (count, raw.action_dim), dtype=torch.float32, device=raw.device
    )
    active = torch.ones(count, dtype=torch.bool, device=raw.device)
    pose_return = torch.zeros(count, dtype=torch.float32, device=raw.device)
    contact_return = torch.zeros_like(pose_return)
    final_phase = torch.full(
        (count,), -1, dtype=torch.long, device=raw.device
    )
    final_position_error = torch.full_like(pose_return, float("nan"))
    final_orientation_error = torch.full_like(pose_return, float("nan"))
    pinch_steps = torch.zeros(count, dtype=torch.long, device=raw.device)
    max_thumb_force = torch.zeros_like(pose_return)
    max_other_force = torch.zeros_like(pose_return)

    rollout_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(raw._reference_length + 2):
            _, _, terminated, truncated, _ = env.step(actions)
            pose_return[active] += raw._last_pose_tracking_reward[active]
            contact_return[active] += raw._last_contact_reward[active]
            pinch_steps[active] += raw._last_pinch_contact[active].to(torch.long)
            max_thumb_force[active] = torch.maximum(
                max_thumb_force[active], raw._last_thumb_contact_force[active]
            )
            max_other_force[active] = torch.maximum(
                max_other_force[active],
                raw._last_other_finger_contact_force[active],
            )
            finished = active & (
                torch.as_tensor(terminated, device=raw.device)
                | torch.as_tensor(truncated, device=raw.device)
            )
            if finished.any():
                final_phase[finished] = raw._last_evaluated_phase[finished]
                final_position_error[finished] = raw._object_position_error[finished]
                final_orientation_error[finished] = raw._object_rotation_error[finished]
                active[finished] = False
            if not active.any():
                break
    rollout_seconds = time.perf_counter() - rollout_start

    if active.any():
        final_phase[active] = raw._last_evaluated_phase[active]
        final_position_error[active] = raw._object_position_error[active]
        final_orientation_error[active] = raw._object_rotation_error[active]
    total_return = pose_return + contact_return
    vectors = manifest.get("vectors", [None] * count)
    candidate_ids = manifest.get(
        "candidate_ids", [f"candidate_{index:06d}" for index in range(count)]
    )
    rows = []
    for index in range(count):
        rows.append(
            {
                "candidate_index": index,
                "candidate_id": candidate_ids[index],
                "vector": vectors[index],
                "total_reward": float(total_return[index].item()),
                "pose_reward": float(pose_return[index].item()),
                "contact_reward": float(contact_return[index].item()),
                "pinch_contact_steps": int(pinch_steps[index].item()),
                "phase": int(final_phase[index].item()),
                "success": bool(final_phase[index].item() >= raw._reference_length - 1),
                "position_error_m": float(final_position_error[index].item()),
                "orientation_error_rad": float(
                    final_orientation_error[index].item()
                ),
                "max_thumb_contact_force_n": float(max_thumb_force[index].item()),
                "max_other_finger_contact_force_n": float(
                    max_other_force[index].item()
                ),
                "hand_usd": manifest["hand_usd_paths"][index],
                "reference": manifest["reference_paths"][index],
            }
        )
    rows.sort(key=lambda row: row["total_reward"], reverse=True)
    result = {
        "schema_version": 1,
        "backend": "isaaclab_physx_gpu",
        "all_candidates_physically_evaluated": True,
        "proxy_used": False,
        "top_k_prefilter_used": False,
        "parametric_usd": bool(template_value),
        "reward": "RL C-error + binary pinch contact",
        "candidate_count": count,
        "completed": len(rows),
        "usd_conversion_seconds": conversion_seconds,
        "initialization_seconds": initialization_seconds,
        "rollout_seconds": rollout_seconds,
        "environment_steps_per_second": (
            count * raw._reference_length / max(rollout_seconds, 1.0e-9)
        ),
        "best": rows[0],
        "results": rows,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "WUJI_PHYSX_BATCH_COMPLETE "
        f"completed={len(rows)}/{count} best={rows[0]['total_reward']:.9f} "
        f"phase={rows[0]['phase']}/{raw._reference_length - 1} "
        f"rollout_seconds={rollout_seconds:.6f}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
