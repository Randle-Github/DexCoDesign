#!/usr/bin/env python3
"""Render one retargeted bimanual TACO trajectory in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--left-usd", type=Path, required=True)
parser.add_argument("--right-usd", type=Path, required=True)
parser.add_argument("--object-usd-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext


def articulation(path: str, usd_path: Path) -> Articulation:
    return Articulation(
        ArticulationCfg(
            prim_path=path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path.resolve()),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(),
            actuators={},
        )
    )


def rigid_object(path: str, usd_path: Path, color: tuple[float, float, float]) -> RigidObject:
    return RigidObject(
        RigidObjectCfg(
            prim_path=path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path.resolve()),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, roughness=0.48, metallic=0.05
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
    )


def main() -> None:
    trajectory_path = args.trajectory.resolve()
    data = np.load(trajectory_path)
    metadata = json.loads(str(data["metadata_json"]))
    object_keys = sorted(key for key in data.files if key.endswith("_pose_wxyz"))
    if len(object_keys) != 2:
        raise ValueError(f"Expected tool and target poses, found {object_keys}")

    all_object_positions = np.concatenate([data[key][:, :3] for key in object_keys], axis=0)
    scene = metadata.get("scene")
    if scene is None:
        center = all_object_positions.mean(axis=0)
        table_z_guess = float(all_object_positions[:, 2].min() - 0.055)
        scene = {
            "table_z": table_z_guess,
            "camera_target": [float(center[0]), float(center[1]), table_z_guess + 0.10],
            "camera_eye": [float(center[0] + 0.48), float(center[1] + 0.48), table_z_guess + 0.38],
        }

    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args.device)
    )
    table_z = float(scene["table_z"])
    ground_cfg = sim_utils.GroundPlaneCfg(
        color=(0.18, 0.21, 0.27),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
    )
    ground_cfg.func("/World/Table", ground_cfg, translation=(0.0, 0.0, table_z))
    light_cfg = sim_utils.DomeLightCfg(intensity=2300.0, color=(0.88, 0.90, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    left = articulation("/World/LeftMANO", args.left_usd)
    right = articulation("/World/RightMANO", args.right_usd)
    objects = []
    colors = ((0.95, 0.45, 0.12), (0.12, 0.65, 0.92))
    for index, key in enumerate(object_keys):
        object_id = key.removesuffix("_pose_wxyz").split("_", 1)[1]
        usd_path = args.object_usd_dir.resolve() / f"{object_id}.usd"
        objects.append(rigid_object(f"/World/Object{index}", usd_path, colors[index]))

    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            height=720,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=36.0,
                focus_distance=0.55,
                horizontal_aperture=24.0,
                clipping_range=(0.01, 10.0),
            ),
        )
    )
    sim.reset()
    eye = torch.tensor([scene["camera_eye"]], device=sim.device)
    target = torch.tensor([scene["camera_target"]], device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    left_names = data["left_joint_names"].tolist()
    right_names = data["right_joint_names"].tolist()
    left_order = [left_names.index(name) for name in left.joint_names]
    right_order = [right_names.index(name) for name in right.joint_names]
    left_q = torch.as_tensor(data["left_q"][:, left_order], device=sim.device)
    right_q = torch.as_tensor(data["right_q"][:, right_order], device=sim.device)
    object_poses = [torch.as_tensor(data[key], device=sim.device) for key in object_keys]
    frames = len(left_q)
    if len(right_q) != frames or any(len(poses) != frames for poses in object_poses):
        raise ValueError("Hand and object trajectories have different lengths")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = args.output.with_suffix(".png")
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )
    try:
        for frame_index in range(frames):
            left.write_joint_state_to_sim(
                left_q[frame_index].unsqueeze(0), torch.zeros_like(left_q[frame_index]).unsqueeze(0)
            )
            right.write_joint_state_to_sim(
                right_q[frame_index].unsqueeze(0), torch.zeros_like(right_q[frame_index]).unsqueeze(0)
            )
            for object_asset, poses in zip(objects, object_poses):
                object_asset.write_root_pose_to_sim(poses[frame_index].unsqueeze(0))
                object_asset.write_root_velocity_to_sim(
                    torch.zeros((1, 6), device=sim.device, dtype=torch.float32)
                )
            sim.forward()
            simulation_app.update()
            sim.render()
            camera.update(0.0)
            frame = camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
            if frame_index == 0:
                imageio.imwrite(preview_path, frame)
                print(
                    f"TACO_ISAAC_FIRST_FRAME shape={frame.shape} min={frame.min()} max={frame.max()}",
                    flush=True,
                )
            writer.append_data(frame)
    finally:
        writer.close()
    print(
        f"TACO_ISAAC_RENDER_COMPLETE sequence={metadata['sequence_id']} frames={frames} "
        f"video={args.output}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac/RTX shutdown can hang on the headless cluster after successful capture.
    os._exit(0)


if __name__ == "__main__":
    main()
