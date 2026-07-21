#!/usr/bin/env python3
"""Create directly importable hand assets with a shared anatomical root frame.

Convention for both sides:
  +Z: wrist to the centroid of the non-thumb fingertips
  +X: lateral axis pointing toward the thumb side
  +Y: completes a right-handed frame (Z cross X)

Upstream files remain untouched.  The registry is redirected to adjacent
``*.canonical`` assets and records the original path plus baked rotation.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from visualize_right_hands import (
    ASSETS,
    REGISTRY,
    RIGHT_HAND_LANDMARKS,
    canonical_basis,
    origin_transform,
    source_landmark_positions,
)


LEFT_HAND_LANDMARKS = {
    "mano": ("left_palm", ["left_index1y", "left_middle1y", "left_ring1y", "left_pinky1y"], ["left_index2", "left_middle2", "left_ring2", "left_pinky2"], "left_thumb3"),
    "ability_hand": ("base", ["index_L1", "middle_L1", "ring_L1", "pinky_L1"], ["index_L2", "middle_L2", "ring_L2", "pinky_L2"], "thumb_anchor"),
    "schunk_svh": ("left_hand_link", ["left_hand_index_proximal", "left_hand_middle_proximal", "left_hand_ring_proximal", "left_hand_pinky_proximal"], ["left_hand_index_intermediate", "left_hand_middle_intermediate", "left_hand_ring_intermediate", "left_hand_pinky_intermediate"], "left_hand_thumb_distal"),
    "wuji_hand_2": ("l_base_link", ["l_index_finger_proximal", "l_middle_finger_proximal", "l_ring_finger_proximal", "l_pinky_proximal"], ["l_index_finger_middle", "l_middle_finger_middle", "l_ring_finger_middle", "l_pinky_middle"], "l_thumb_tip"),
    "sharpa_wave_01": ("left_hand_C_MC", ["left_index_PP", "left_middle_PP", "left_ring_PP", "left_pinky_PP"], ["left_index_MP", "left_middle_MP", "left_ring_MP", "left_pinky_MP"], "left_thumb_fingertip"),
    "tesollo_dg5f": ("ll_dg_mount", ["ll_dg_2_1", "ll_dg_3_1", "ll_dg_4_1", "ll_dg_5_1"], ["ll_dg_2_2", "ll_dg_3_2", "ll_dg_4_2", "ll_dg_5_2"], "ll_dg_1_tip"),
    "unitree_dex5_1": ("base_link00L", ["Link_21L", "Link_31L", "Link_41L", "Link_51L"], ["Link_22L", "Link_32L", "Link_42L", "Link_52L"], "Link_14L"),
    "robotera_xhand1": ("left_hand_base_link", ["left_hand_index_bend_link", "left_hand_mid_link1", "left_hand_ring_link1", "left_hand_pinky_link1"], ["left_hand_index_rota_link1", "left_hand_mid_link2", "left_hand_ring_link2", "left_hand_pinky_link2"], "left_hand_thumb_rota_tip"),
    "orca_hand_v2": ("ForeArmStructure-Model_e18f2368", ["I-AP-L_57ce92f7", "M-AP_e04a96f2", "M-AP_6ec59111", "P-AP_f5e42b61"], ["I-PP_3df4f91d", "M-PP_08efa608", "M-PP_8660a1eb", "P-PP_1d411b9b"], "T-DP_307db3cc"),
    "shadow_hand_e": ("lh_wrist", ["lh_ffknuckle", "lh_mfknuckle", "lh_rfknuckle", "lh_lfknuckle"], ["lh_ffmiddle", "lh_mfmiddle", "lh_rfmiddle", "lh_lfmiddle"], "lh_thtip"),
    "allegro_hand_v5": ("palm_link", ["link_0_0", "link_4_0", "link_8_0"], ["link_1_0", "link_5_0", "link_9_0"], "link_15_0_tip"),
    "midas_hand": ("palm_base", ["index_mcp_abad_link", "middle_mcp_abad_link", "ring_mcp_abad_link"], ["index_pip_link", "middle_pip_link", "ring_pip_link"], "thumb_dip"),
    "ruka_v2": ("base_new", ["mcp", "mcp_2", "mcp_3", "mcp_4"], ["pip", "pip_2", "pip_3", "pinky___joint_2"], "thumb_actual_tip"),
    "inspire_rh56dfx": ("base", ["index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"], ["index_intermediate", "middle_intermediate", "ring_intermediate", "pinky_intermediate"], "thumb_distal"),
}

LANDMARKS = {
    hand_id: {"left": LEFT_HAND_LANDMARKS[hand_id], "right": RIGHT_HAND_LANDMARKS[hand_id]}
    for hand_id in RIGHT_HAND_LANDMARKS
}

MJCF_ROOTS = {
    "mano": {"left": "left_palm", "right": "right_palm"},
    "schunk_svh": {"left": "left_pos_x_link", "right": "R_forearm_ty_link"},
    "shadow_hand_e": {"left": "lh_forearm", "right": "rh_forearm"},
    "inspire_rh56dfx": {"left": "base", "right": "base"},
}

USD_ROOTS = {
    "sharpa_wave_01": {"left": "left_sharpa_wave", "right": "right_sharpa_wave"},
    "tesollo_dg5f": {"left": "dg5f_left", "right": "dg5f_right"},
}


def canonical_path(source: Path, source_format: str) -> Path:
    suffix = ".usda" if source_format == "usd" else source.suffix
    return source.with_name(f"{source.stem}.canonical{suffix}")


def set_origin(parent: ET.Element, transform: np.ndarray) -> None:
    origin = parent.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        parent.insert(0, origin)
    xyz = transform[:3, 3]
    rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    origin.set("xyz", " ".join(f"{value:.12g}" for value in xyz))
    origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))


def canonicalize_urdf(source: Path, output: Path, rotation: np.ndarray) -> None:
    tree = ET.parse(source)
    robot = tree.getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    child_links = {joint.find("child").get("link") for joint in robot.findall("joint") if joint.find("child") is not None}
    roots = set(links).difference(child_links)
    if len(roots) != 1:
        raise ValueError(f"Expected one root link in {source}, got {sorted(roots)}")
    root_name = next(iter(roots))
    root_link = links[root_name]
    root_rotation = np.eye(4)
    root_rotation[:3, :3] = rotation

    # URDF has no separate default-q field: q=0 is the reference pose.  If an
    # upstream bounded joint excludes zero, reparameterize it equivalently by
    # baking the nearest limit into the joint origin and shifting its limits.
    # This preserves the reachable physical poses while making zero valid.
    for joint in robot.findall("joint"):
        if joint.get("type") in {"fixed", "continuous"}:
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if lower <= 0.0 <= upper:
            continue
        offset = lower if lower > 0.0 else upper
        axis_node = joint.find("axis")
        axis = np.fromstring(axis_node.get("xyz"), sep=" ") if axis_node is not None else np.array([1.0, 0.0, 0.0])
        axis /= np.linalg.norm(axis)
        shift = np.eye(4)
        shift[:3, :3] = Rotation.from_rotvec(axis * offset).as_matrix()
        set_origin(joint, origin_transform(joint.find("origin")) @ shift)
        limit.set("lower", f"{lower - offset:.12g}")
        limit.set("upper", f"{upper - offset:.12g}")

    for tag in ("visual", "collision", "inertial"):
        for element in root_link.findall(tag):
            set_origin(element, root_rotation @ origin_transform(element.find("origin")))
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        if parent is not None and parent.get("link") == root_name:
            set_origin(joint, root_rotation @ origin_transform(joint.find("origin")))

    robot.set("name", f"{robot.get('name', root_name)}_canonical")
    ET.indent(robot, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def set_mjcf_root_rotation(body: ET.Element, rotation: np.ndarray) -> None:
    for attribute in ("euler", "axisangle", "xyaxes", "zaxis"):
        if attribute in body.attrib:
            raise ValueError(f"Unsupported existing MJCF root orientation {attribute} on {body.get('name')}")
    existing = np.eye(3)
    if "quat" in body.attrib:
        w, x, y, z = np.fromstring(body.get("quat"), sep=" ")
        existing = Rotation.from_quat([x, y, z, w]).as_matrix()
    composed = rotation @ existing
    x, y, z, w = Rotation.from_matrix(composed).as_quat()
    body.set("quat", " ".join(f"{value:.12g}" for value in (w, x, y, z)))
    if "pos" in body.attrib:
        position = np.fromstring(body.get("pos"), sep=" ")
        body.set("pos", " ".join(f"{value:.12g}" for value in rotation @ position))


def canonicalize_mjcf(source: Path, output: Path, root_name: str, rotation: np.ndarray) -> None:
    tree = ET.parse(source)
    root_body = tree.find(f".//body[@name='{root_name}']")
    if root_body is not None:
        set_mjcf_root_rotation(root_body, rotation)
    else:
        matched = False
        for include in tree.findall(".//include"):
            included_source = (source.parent / include.get("file")).resolve()
            if not included_source.exists():
                continue
            included_tree = ET.parse(included_source)
            if included_tree.find(f".//body[@name='{root_name}']") is None:
                continue
            included_output = canonical_path(included_source, "mjcf")
            canonicalize_mjcf(included_source, included_output, root_name, rotation)
            include.set("file", included_output.name)
            matched = True
            break
        if not matched:
            raise KeyError(f"Could not find MJCF root body {root_name!r} from {source}")
    ET.indent(tree.getroot(), space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def canonicalize_usd(source: Path, output: Path, root_prim: str, rotation: np.ndarray) -> None:
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    relative = source.name
    output.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "canonical_hand"\n'
        ")\n\n"
        f'def Xform "canonical_hand" (\n    prepend references = @{relative}@</{root_prim}>\n)\n'
        "{\n"
        f"    quatd xformOp:orient = ({w:.17g}, {x:.17g}, {y:.17g}, {z:.17g})\n"
        '    uniform token[] xformOpOrder = ["xformOp:orient"]\n'
        "}\n",
        encoding="utf-8",
    )


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry["canonical_frame"] = {
        "+x": "right-hand thumb side; left-hand little-finger side",
        "+y": "common palm normal for left and right hands",
        "+z": "non-thumb proximal-link direction at q=0",
    }
    registry["default_joint_position"] = 0.0
    for hand_id, metadata in registry["hands"].items():
        for side, entry in metadata["entries"].items():
            source_relative = entry.get("source_path", entry["path"])
            source = (ASSETS / source_relative).resolve()
            source_format = entry["format"]
            landmarks = source_landmark_positions(source, source_format)
            _, basis = canonical_basis(
                landmarks,
                LANDMARKS[hand_id][side],
                thumb_x_sign=1.0 if side == "right" else -1.0,
            )
            rotation = basis.T
            output = canonical_path(source, source_format)

            if source_format == "urdf":
                canonicalize_urdf(source, output, rotation)
            elif source_format == "mjcf":
                canonicalize_mjcf(source, output, MJCF_ROOTS[hand_id][side], rotation)
            elif source_format == "usd":
                canonicalize_usd(source, output, USD_ROOTS[hand_id][side], rotation)
            else:
                raise ValueError(f"Unsupported format: {source_format}")

            quaternion = Rotation.from_matrix(rotation).as_quat()
            entry["source_path"] = source_relative
            entry["path"] = str(output.relative_to(ASSETS))
            entry["canonicalized"] = True
            entry["canonical_rotation_xyzw"] = [round(float(value), 12) for value in quaternion]
            print(f"{hand_id:20s} {side:5s} -> {output.relative_to(ASSETS)}")

    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
