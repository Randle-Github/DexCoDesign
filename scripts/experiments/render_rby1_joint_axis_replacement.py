#!/usr/bin/env python3
"""Render an isolated RBY1 joint-axis replacement experiment."""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from scripts.assets import render_mobile_manipulators as base


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "assets"
    / "mobile_manipulators"
    / "rby1"
    / "eef_free"
    / "robot.urdf"
)
OUTPUT = (
    ROOT
    / "artifacts"
    / "mobile_manipulator_joint_axis_experiment"
)
VARIANT = OUTPUT / "rby1_arm3_pitch_to_roll.urdf"
SCENE = OUTPUT / "rby1_arm3_pitch_to_roll_scene.xml"
IMAGE = OUTPUT / "rby1_arm3_pitch_to_roll.png"
METADATA = OUTPUT / "rby1_arm3_pitch_to_roll.json"


def build_variant() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE)
    robot = tree.getroot()
    joints = {
        joint.get("name", ""): joint
        for joint in robot.findall("joint")
    }
    shoulder_before = {
        name: ET.tostring(joints[name])
        for name in ("left_arm_0", "right_arm_0")
    }
    replacements = {
        # A revolute axis is an axial vector. Mirroring across y=0 maps
        # left +X to right -X, so this pair is one symmetric design action.
        "left_arm_3": "1 0 0",
        "right_arm_3": "-1 0 0",
    }
    for name, axis_xyz in replacements.items():
        axis = joints[name].find("axis")
        if axis is None:
            raise ValueError(f"{name}: missing axis")
        axis.set("xyz", axis_xyz)
    for name, expected in shoulder_before.items():
        if ET.tostring(joints[name]) != expected:
            raise ValueError(f"shoulder joint changed: {name}")

    for mesh in robot.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        source_mesh = Path(filename)
        if not source_mesh.is_absolute():
            source_mesh = (SOURCE.parent / source_mesh).resolve()
        mesh.set("filename", os.path.relpath(source_mesh, OUTPUT))

    robot.set("name", "rby1_arm3_pitch_to_roll")
    ET.indent(tree, space="  ")
    tree.write(VARIANT, encoding="utf-8", xml_declaration=True)
    METADATA.write_text(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(ROOT)),
                "variant": str(VARIANT.relative_to(ROOT)),
                "design_action": {
                    "kind": "bilateral_joint_axis_replacement",
                    "joint_pair": ["left_arm_3", "right_arm_3"],
                    "source_axis_local": {
                        "left_arm_3": [0, 1, 0],
                        "right_arm_3": [0, 1, 0],
                    },
                    "target_axis_local": {
                        "left_arm_3": [1, 0, 0],
                        "right_arm_3": [-1, 0, 0],
                    },
                    "motor_centers_changed": False,
                    "joint_limits_changed": False,
                    "shoulder_joints_changed": False,
                    "mesh_changed": False,
                },
                "diagnostic_pose_radians": {
                    "left_arm_3": -math.pi / 3,
                    "right_arm_3": -math.pi / 3,
                },
                "image_order": ["source_pitch_axis", "candidate_roll_axis"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render() -> None:
    diagnostic_pose = (
        ("left_arm_3", -math.pi / 3),
        ("right_arm_3", -math.pi / 3),
    )
    base.ROBOTS = (
        base.Robot(
            "rby1_source_pitch_axis",
            SOURCE,
            ".42 .50 .62 1",
            diagnostic_pose,
            yaw_degrees=-90.0,
        ),
        base.Robot(
            "rby1_candidate_roll_axis",
            VARIANT,
            ".94 .36 .16 1",
            diagnostic_pose,
            yaw_degrees=-90.0,
        ),
    )
    base.OUTPUT_ROOT = OUTPUT
    base.SCENE_PATH = SCENE
    base.IMAGE_PATH = IMAGE
    base.WIDTH, base.HEIGHT = 2000, 1200
    _, scene_size = base.create_scene()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, scene_size[1] * 0.48]
    camera.distance = 3.45
    camera.azimuth = 90.0
    camera.elevation = -7.0
    with mujoco.Renderer(
        model, height=base.HEIGHT, width=base.WIDTH
    ) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert(
        "RGB"
    ).save(IMAGE, quality=96)


def main() -> int:
    build_variant()
    render()
    print(IMAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
