#!/usr/bin/env python3
"""Interactively inspect a palm-deformed Sharpa hand from the 100-hand run."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DEFAULT_INPUT = OUTPUTS / "sharpa_interactive" / "generated_sharpa_compiled.json"
DEFAULT_HAND_ID = "grammar_005"
MJCF = OUTPUTS / "sharpa_interactive" / "generated_sharpa_variant.xml"
METADATA = OUTPUTS / "sharpa_interactive" / "generated_sharpa_variant.json"


def fmt(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def build_mjcf(input_path: Path, hand_id: str) -> tuple[mujoco.MjModel, dict]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    matches = [hand for hand in payload["hands"] if hand["hand_id"] == hand_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {hand_id} in {input_path}, found {len(matches)}")
    hand = matches[0]
    if hand["seed_source"] != "sharpa_wave_01":
        raise ValueError(f"{hand_id} is {hand['seed_source']}, not Sharpa")
    parts = {int(part["id"]): part for part in hand["parts"]}
    children: dict[int, list[int]] = {}
    for part in parts.values():
        if part["parent"] is not None:
            children.setdefault(int(part["parent"]), []).append(int(part["id"]))

    root = ET.Element("mujoco", {"model": f"{hand_id} palm-deformed Sharpa"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {"gravity": "0 0 0", "timestep": "0.002", "integrator": "implicitfast"},
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        {"damping": "0.12", "armature": "0.002", "limited": "true"},
    )
    ET.SubElement(
        default,
        "geom",
        {
            "contype": "0",
            "conaffinity": "0",
            "density": "650",
            "rgba": "0.18 0.72 0.95 1",
        },
    )
    asset = ET.SubElement(root, "asset")
    for part_id, part in parts.items():
        compiled = part.get("compiled_mesh")
        if compiled is None:
            continue
        path = (OUTPUTS / compiled["file"]).resolve()
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"mesh_{part_id}", "file": str(path)},
        )

    world = ET.SubElement(root, "worldbody")
    root_body = ET.SubElement(world, "body", {"name": "palm_root"})
    actuator_records = []

    def add_part(part_id: int, parent_body: ET.Element) -> None:
        part = parts[part_id]
        if part_id == 0:
            body = parent_body
        else:
            body = ET.SubElement(
                parent_body,
                "body",
                {
                    "name": f"part_{part_id}_{part['role']}",
                    "pos": fmt(part["relative_pos"]),
                },
            )
            if part["joint_type"] != "fixed":
                joint_name = str(part["joint_name"])
                lower, upper = part["joint_range"]
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": joint_name,
                        "type": "hinge" if part["joint_type"] == "hinge" else "slide",
                        "axis": fmt(part["joint_axis"]),
                        "range": fmt([lower, upper]),
                    },
                )
                actuator_records.append(
                    {
                        "name": joint_name,
                        "range": [float(lower), float(upper)],
                        "role": part["role"],
                    }
                )
        if part.get("compiled_mesh") is not None:
            ET.SubElement(body, "geom", {"type": "mesh", "mesh": f"mesh_{part_id}"})
        for child_id in sorted(children.get(part_id, [])):
            add_part(child_id, body)

    add_part(0, root_body)
    actuator = ET.SubElement(root, "actuator")
    for record in actuator_records:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"drive_{record['name']}",
                "joint": record["name"],
                "kp": "8",
                "dampratio": "1",
                "ctrllimited": "true",
                "ctrlrange": fmt(record["range"]),
            },
        )
    ET.indent(root, space="  ")
    MJCF.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(MJCF, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    if model.njnt != hand["dof_count"] or model.nu != hand["dof_count"]:
        raise ValueError(
            f"compiled DoF mismatch: HandIR={hand['dof_count']}, "
            f"MuJoCo joints/actuators={model.njnt}/{model.nu}"
        )
    metadata = {
        "generated_hand_id": hand_id,
        "seed_source": hand["seed_source"],
        "palm_layout": hand["palm_layout"],
        "palm_transform": hand["palm_transform"],
        "finger_count": hand["finger_count"],
        "dof_count": hand["dof_count"],
        "joint_count": model.njnt,
        "actuator_count": model.nu,
        "joints": actuator_records,
        "compiled_input": str(input_path),
        "mjcf": str(MJCF),
        "palm_attachment_audit": hand["palm_attachment_audit"],
    }
    return model, metadata


def validate(model: mujoco.MjModel) -> dict:
    data = mujoco.MjData(model)
    finite_poses = 0
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        for target in model.jnt_range[joint_id]:
            data.qpos[:] = 0.0
            data.qpos[address] = target
            mujoco.mj_forward(model, data)
            if not np.all(np.isfinite(data.xpos)):
                raise ValueError(f"joint {joint_id} produced a non-finite limit pose")
            finite_poses += 1
    return {"finite_joint_limit_poses": finite_poses}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hand-id", default=DEFAULT_HAND_ID)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    model, metadata = build_mjcf(args.input, args.hand_id)
    metadata["validation"] = validate(model)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    if args.build_only:
        return 0

    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = max(2.2, 1.55 * model.stat.extent)
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -12.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_JOINT
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - start)
            if remaining > 0.0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
