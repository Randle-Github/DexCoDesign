#!/usr/bin/env python3
"""Render one bimanual ARCTIC trajectory with its articulated object in Isaac Sim."""

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
parser.add_argument("--object-usd", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import imageio.v2 as imageio
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext


def asset(path: str, usd: Path, color: tuple[float, float, float]) -> Articulation:
    return Articulation(ArticulationCfg(
        prim_path=path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd.resolve()),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.44),
        ),
        init_state=ArticulationCfg.InitialStateCfg(), actuators={}))


def main() -> None:
    z = np.load(args.trajectory.resolve())
    meta = json.loads(str(z["metadata_json"]))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, render_interval=1, device=args.device))
    ground = sim_utils.GroundPlaneCfg(color=(0.10, 0.13, 0.19))
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, float(meta["scene"]["ground_z"])))
    light = sim_utils.DomeLightCfg(intensity=1450.0, color=(0.78, 0.84, 1.0))
    light.func("/World/Light", light)
    left = asset("/World/LeftMANO", args.left_usd, (0.08, 0.48, 0.96))
    right = asset("/World/RightMANO", args.right_usd, (0.96, 0.16, 0.10))
    obj = asset("/World/Object", args.object_usd, (0.95, 0.55, 0.04))
    camera = Camera(CameraCfg(
        prim_path="/World/Camera", update_period=0.0, height=720, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=42.0, focus_distance=1.0,
            horizontal_aperture=24.0, clipping_range=(0.01, 12.0))))
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([meta["scene"]["camera_eye"]], device=sim.device),
        torch.tensor([meta["scene"]["camera_target"]], device=sim.device))

    def ordered(side: str, hand: Articulation):
        names = z[f"{side}_joint_names"].tolist()
        order = [names.index(name) for name in hand.joint_names]
        return torch.as_tensor(z[f"{side}_q"][:, order], device=sim.device)
    left_q, right_q = ordered("left", left), ordered("right", right)
    obj_pos = torch.as_tensor(z["object_root_position_m"], device=sim.device)
    obj_quat = torch.as_tensor(z["object_root_quaternion_wxyz"], device=sim.device)
    obj_q = torch.as_tensor(z["object_joint_position_rad"], device=sim.device).reshape(-1, 1)
    frames = len(left_q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
    try:
        for i in range(frames):
            for hand, q in ((left, left_q), (right, right_q)):
                hand.write_joint_state_to_sim(q[i:i+1], torch.zeros_like(q[i:i+1]))
            pose = torch.cat((obj_pos[i:i+1], obj_quat[i:i+1]), dim=1)
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
            obj.write_joint_state_to_sim(obj_q[i:i+1], torch.zeros_like(obj_q[i:i+1]))
            sim.forward(); app.update(); sim.render(); camera.update(0.0)
            frame = camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
            if i == 0:
                imageio.imwrite(args.output.with_suffix(".png"), frame)
            writer.append_data(frame)
    finally:
        writer.close()
    object_id = meta.get("object_id") or meta.get("objects", [{}])[0].get("object_id", "unknown")
    print(f"ARCTIC_ISAAC_RENDER_COMPLETE object={object_id} frames={frames} video={args.output}", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
