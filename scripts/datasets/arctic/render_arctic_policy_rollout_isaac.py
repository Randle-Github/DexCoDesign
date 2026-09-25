#!/usr/bin/env python3
"""Render an exactly captured MANO + articulated-object policy rollout in Isaac."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--rollout", type=Path, required=True)
parser.add_argument("--hand-usd", type=Path, required=True)
parser.add_argument("--object-usd", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
print("ARCTIC_POLICY_RENDER_STAGE app_ready", flush=True)

import imageio.v2 as imageio
import numpy as np
import torch
import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext


def articulation(path: str, usd: Path, color: tuple[float, float, float]) -> Articulation:
    return Articulation(
        ArticulationCfg(
            prim_path=path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd.resolve()),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, roughness=0.44
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(),
            actuators={},
        )
    )


def main() -> None:
    print("ARCTIC_POLICY_RENDER_STAGE loading_rollout", flush=True)
    # The standalone full Isaac experience does not inherit Isaac Lab's custom
    # camera-enabled marker even though RTX rendering is active.
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)
    with np.load(args.rollout.resolve()) as source:
        hand_q = source["hand_q"].astype(np.float32)
        object_pose = source["object_pose_wxyz"].astype(np.float32)
        object_q = source["object_joint_position_rad"].astype(np.float32)
        metadata = json.loads(str(source["metadata_json"]))
    object_q = object_q.reshape(len(hand_q), -1)
    if object_pose.shape != (len(hand_q), 7) or object_q.shape[0] != len(hand_q):
        raise ValueError(
            f"Inconsistent rollout shapes: hand={hand_q.shape}, "
            f"object_pose={object_pose.shape}, object_q={object_q.shape}"
        )

    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, render_interval=1, device=args.device)
    )
    print("ARCTIC_POLICY_RENDER_STAGE simulation_ready", flush=True)
    # Use a local primitive instead of the Nucleus-backed GroundPlane asset so
    # rendering also works with Isaac Sim's standalone full experience.
    ground = sim_utils.CuboidCfg(
        size=(2.0, 2.0, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.10, 0.13, 0.19), roughness=0.55
        ),
    )
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.005))
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.80, 0.86, 1.0))
    light.func("/World/Light", light)
    hand = articulation("/World/Hand", args.hand_usd, (0.12, 0.55, 0.96))
    print("ARCTIC_POLICY_RENDER_STAGE hand_spawned", flush=True)
    obj = articulation("/World/Object", args.object_usd, (0.95, 0.45, 0.08))
    print("ARCTIC_POLICY_RENDER_STAGE object_spawned", flush=True)
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            height=720,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=42.0,
                focus_distance=1.0,
                horizontal_aperture=24.0,
                clipping_range=(0.01, 12.0),
            ),
        )
    )
    print("ARCTIC_POLICY_RENDER_STAGE camera_spawned", flush=True)
    sim.reset()
    print("ARCTIC_POLICY_RENDER_STAGE simulation_reset", flush=True)

    saved_names = metadata.get("joint_names")
    if saved_names is None:
        raise ValueError("Captured rollout does not contain joint_names metadata")
    order = [saved_names.index(name) for name in hand.joint_names]
    hand_q = hand_q[:, order]
    center = object_pose[:, :3].mean(axis=0)
    camera.set_world_poses_from_view(
        torch.tensor(
            [center + np.array([0.36, 0.34, 0.28])],
            dtype=torch.float32,
            device=sim.device,
        ),
        torch.tensor(
            [center + np.array([0.0, 0.0, 0.03])],
            dtype=torch.float32,
            device=sim.device,
        ),
    )
    print("ARCTIC_POLICY_RENDER_STAGE camera_positioned", flush=True)

    hand_q_t = torch.as_tensor(hand_q, device=sim.device)
    object_pose_t = torch.as_tensor(object_pose, device=sim.device)
    object_q_t = torch.as_tensor(object_q, device=sim.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        for frame_index in range(len(hand_q)):
            q = hand_q_t[frame_index : frame_index + 1]
            hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            pose = object_pose_t[frame_index : frame_index + 1]
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
            q_obj = object_q_t[frame_index : frame_index + 1]
            obj.write_joint_state_to_sim(q_obj, torch.zeros_like(q_obj))
            sim.forward()
            app.update()
            sim.render()
            camera.update(0.0)
            frame = camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
            if frame_index == 0:
                imageio.imwrite(args.output.with_suffix(".png"), frame)
                print(
                    f"ARCTIC_POLICY_RENDER_FRAME shape={frame.shape} "
                    f"min={int(frame.min())} max={int(frame.max())}",
                    flush=True,
                )
            writer.append_data(frame)
    finally:
        writer.close()
    print(
        f"ARCTIC_POLICY_ISAAC_RENDER_COMPLETE frames={len(hand_q)} "
        f"video={args.output}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
