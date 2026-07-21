#!/usr/bin/env python3
"""Compile and view 100 high-fidelity, graph-mutated right hands in MuJoCo."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SCENE = OUTPUTS / "generated_100_hands.xml"
REALIZED = OUTPUTS / "realized_graphs.json"
DIGIT_ROLES = ["index", "middle", "ring", "pinky"]
COLORS = [
    ".95 .55 .22 1", ".25 .65 .98 1", ".30 .82 .58 1", ".93 .35 .47 1",
    ".67 .48 .98 1", ".98 .55 .25 1", ".20 .78 .82 1", ".88 .42 .86 1",
    ".56 .78 .28 1", ".98 .35 .29 1", ".38 .58 .85 1", ".78 .62 .42 1",
    ".35 .76 .52 1", ".92 .69 .08 1",
]


def fmt(values) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def clean(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def descendants(nodes: dict[str, dict], roots: set[str]) -> set[str]:
    result = set(roots)
    changed = True
    while changed:
        changed = False
        for uid, node in nodes.items():
            if node.get("parent_uid") in result and uid not in result:
                result.add(uid)
                changed = True
    return result


def evenly_select(values: list[str], count: int) -> list[str]:
    if count >= len(values):
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in dict.fromkeys(indices)]


def make_graph(template: dict, sample: dict) -> list[dict]:
    """Apply bounded grammar mutations to a real source kinematic tree."""
    nodes: dict[str, dict] = {}
    index_to_uid = {}
    for source in template["nodes"]:
        uid = f"n{source['index']:02d}"
        index_to_uid[source["index"]] = uid
        node = copy.deepcopy(source)
        node["uid"] = uid
        nodes[uid] = node
    for node in nodes.values():
        parent = node.get("parent")
        node["parent_uid"] = index_to_uid.get(parent) if parent is not None else None

    desired_digits = int(sample["graph_mutation"]["desired_digit_count"])
    desired_nonthumb = desired_digits - 1
    available = [role for role in DIGIT_ROLES if any(node["role"] == role for node in nodes.values())]
    keep = evenly_select(available, min(desired_nonthumb, len(available)))
    remove_roles = set(available).difference(keep)
    remove_roots = {uid for uid, node in nodes.items() if node["role"] in remove_roles}
    for uid in descendants(nodes, remove_roots):
        nodes.pop(uid, None)

    # Recount after subtree deletion: anonymous industrial graphs can place
    # one semantic role below another, so deleting a branch may remove more
    # than its role label alone suggests.
    surviving = [role for role in keep if any(node["role"] == role for node in nodes.values())]
    # Duplicate the outermost surviving non-thumb branch for additional digits.
    extra_needed = desired_nonthumb - len(surviving)
    source_role = surviving[-1] if surviving else "thumb"
    source_group = [node for node in nodes.values() if node["role"] == source_role]
    for extra in range(extra_needed):
        mapping = {node["uid"]: f"extra{extra}_{node['uid']}" for node in source_group}
        for source in source_group:
            clone = copy.deepcopy(source)
            clone["uid"] = mapping[source["uid"]]
            clone["name"] = f"extra_digit_{extra}_{source['name']}"
            clone["role"] = f"extra_digit_{extra}"
            parent_uid = source.get("parent_uid")
            clone["parent_uid"] = mapping.get(parent_uid, "n00")
            if parent_uid not in mapping:
                shift = -(extra + 1) * float(sample["parameters"]["palm_width"]) / max(desired_nonthumb, 2)
                relative = np.asarray(clone["relative_pos"], dtype=float)
                relative[0] += shift
                clone["relative_pos"] = relative.tolist()
            nodes[clone["uid"]] = clone

    digit_roles = {"thumb", *surviving, *(f"extra_digit_{i}" for i in range(extra_needed))}
    digit_nodes = [node for node in nodes.values() if node["role"] in digit_roles]

    # Add or remove one terminal link from a deterministic digit branch.
    segment_delta = int(sample["graph_mutation"]["segment_delta"])
    if digit_nodes and segment_delta:
        chosen_role = sorted(digit_roles)[int(sample["sample_id"]) % len(digit_roles)]
        group = [node for node in nodes.values() if node["role"] == chosen_role]
        children = {node["parent_uid"] for node in group}
        leaves = [node for node in group if node["uid"] not in children]
        if leaves:
            leaf = leaves[-1]
            if segment_delta < 0 and len(group) > 2:
                nodes.pop(leaf["uid"], None)
            elif segment_delta > 0:
                clone = copy.deepcopy(leaf)
                clone["uid"] = f"terminal_{leaf['uid']}"
                clone["name"] = f"terminal_{leaf['name']}"
                clone["parent_uid"] = leaf["uid"]
                relative = np.asarray(leaf["relative_pos"], dtype=float)
                if np.linalg.norm(relative) < 0.04:
                    relative = np.asarray([0.0, 0.0, 0.16])
                clone["relative_pos"] = relative.tolist()
                clone["joint_type"] = "hinge"
                nodes[clone["uid"]] = clone

    # Real DOF mutations: these change emitted MuJoCo joints, not only labels.
    fixed_candidates = sorted(
        (node for node in nodes.values() if node["role"] in digit_roles and node["joint_type"] == "fixed"),
        key=lambda node: node["uid"],
    )
    for node in fixed_candidates[: int(sample["graph_mutation"]["fixed_to_hinge"])]:
        node["joint_type"] = "hinge"
    hinge_candidates = sorted(
        (node for node in nodes.values() if node["role"] in digit_roles and node["joint_type"] in {"hinge", "slide", "ball", "free", "other"}),
        key=lambda node: node["uid"],
        reverse=True,
    )
    for node in hinge_candidates[: int(sample["graph_mutation"]["hinge_to_fixed"])]:
        node["joint_type"] = "fixed"
    base_nodes = sorted((node for node in nodes.values() if node["role"] == "base"), key=lambda node: node["uid"])
    base_joint_count = int(sample["graph_mutation"]["base_joint_count"])
    for index, node in enumerate(base_nodes):
        node["joint_type"] = "hinge" if index < base_joint_count else "fixed"

    # Remove any node whose parent disappeared; iterate to a valid rooted tree.
    valid = {"n00"}
    while True:
        added = {uid for uid, node in nodes.items() if node.get("parent_uid") in valid}
        new_valid = valid | added
        if new_valid == valid:
            break
        valid = new_valid
    return [nodes[uid] for uid in sorted(valid, key=lambda value: (value != "n00", value)) if uid in nodes]


def scale_for(node: dict, sample: dict, template: dict, target_names: list[str]) -> np.ndarray:
    p = sample["parameters"]
    base = dict(zip(target_names, template["target"]))
    width = np.clip(p["palm_width"] / base["palm_width"], 0.72, 1.35)
    thickness = np.clip(p["palm_thickness"] / base["palm_thickness"], 0.72, 1.35)
    palm_length = np.clip(p["palm_length"] / base["palm_length"], 0.72, 1.35)
    finger = np.clip(p["finger_length"] / base["finger_length"], 0.72, 1.35)
    thumb = np.clip(p["thumb_length"] / base["thumb_length"], 0.72, 1.35)
    radius = np.clip(p["finger_radius"] / base["finger_radius"], 0.72, 1.35)
    base_length = np.clip(p["base_length"] / base["base_length"], 0.72, 1.35)
    role = node["role"]
    if role == "palm":
        return np.asarray([width, thickness, palm_length])
    if role == "base":
        return np.asarray([width, thickness, base_length])
    if role == "thumb":
        return np.asarray([radius, radius, thumb])
    if role in DIGIT_ROLES or role.startswith("extra_digit"):
        return np.asarray([radius, radius, finger])
    return np.asarray([width, thickness, palm_length])


def scaled_relative(node: dict, parent: dict | None, sample: dict, template: dict, target_names: list[str]) -> np.ndarray:
    relative = np.asarray(node["relative_pos"], dtype=float)
    scale = scale_for(node, sample, template, target_names)
    if parent is not None and node["role"] != parent["role"]:
        # Attachments on the palm follow palm dimensions; intra-digit offsets
        # follow digit length/radius.
        palm_probe = {**node, "role": "palm"}
        scale = scale_for(palm_probe, sample, template, target_names)
    return relative * scale


def add_assets(asset: ET.Element, samples, graphs, library, target_names) -> dict[tuple[int, str], str]:
    mesh_names = {}
    for sample, graph in zip(samples, graphs):
        template = library["hands"][sample["topology_template"]]
        for node in graph:
            if not node.get("mesh"):
                continue
            name = f"mesh_{sample['sample_id']:03d}_{clean(node['uid'])}"
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": name,
                    "file": node["mesh"],
                    "scale": fmt(scale_for(node, sample, template, target_names)),
                },
            )
            mesh_names[(int(sample["sample_id"]), node["uid"])] = name
    return mesh_names


def add_hand(world, sample, graph, template, target_names, mesh_names, position, material) -> dict:
    by_uid = {node["uid"]: node for node in graph}
    children = {uid: [] for uid in by_uid}
    for node in graph:
        parent_uid = node.get("parent_uid")
        if parent_uid in children:
            children[parent_uid].append(node["uid"])
    sample_id = int(sample["sample_id"])

    def emit(node: dict, parent_element: ET.Element, parent_node: dict | None, root: bool = False) -> None:
        attrs = {"name": f"s{sample_id:03d}_{clean(node['uid'])}"}
        if root:
            attrs["pos"] = fmt(position)
        else:
            attrs["pos"] = fmt(scaled_relative(node, parent_node, sample, template, target_names))
        body = ET.SubElement(parent_element, "body", attrs)
        movable = not root and node["joint_type"] != "fixed"
        if movable:
            axis_options = ("0 1 0", "1 0 0", "0 0 1")
            ET.SubElement(
                body,
                "joint",
                {
                    "name": f"j_s{sample_id:03d}_{clean(node['uid'])}",
                    "type": "hinge",
                    "axis": axis_options[(sample_id + len(node["uid"])) % 3],
                    "range": "-0.45 1.45",
                    "damping": ".03",
                    "armature": ".001",
                },
            )
            ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": ".02", "diaginertia": ".00008 .00008 .00008"})
        mesh_name = mesh_names.get((sample_id, node["uid"]))
        if mesh_name:
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"v_s{sample_id:03d}_{clean(node['uid'])}",
                    "type": "mesh",
                    "mesh": mesh_name,
                    "material": material,
                    "mass": "0",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
        for child_uid in sorted(children[node["uid"]]):
            emit(by_uid[child_uid], body, node)
        if root:
            ET.SubElement(
                body,
                "site",
                {
                    "name": f"{sample_id + 1:03d} D{sample['parameters']['digit_count']} {sample['topology_template']}",
                    "pos": "0 0 2.35", "size": ".001", "rgba": "1 1 1 0", "group": "4",
                },
            )

    emit(by_uid["n00"], world, None, root=True)
    dofs = sum(node["joint_type"] != "fixed" for node in graph if node["uid"] != "n00")
    roles = {node["role"] for node in graph}
    digits = int("thumb" in roles) + sum(role in roles for role in DIGIT_ROLES) + sum(role.startswith("extra_digit") for role in roles)
    return {"sample_id": sample_id, "template": sample["topology_template"], "nodes": len(graph), "dofs": dofs, "digits": digits}


def build_scene(
    samples: list[dict],
    scene_path: Path = SCENE,
    realized_path: Path | None = REALIZED,
    columns: int | None = None,
) -> tuple[Path, list[dict]]:
    library = json.loads((OUTPUTS / "mesh_library.json").read_text(encoding="utf-8"))
    dataset = json.loads((OUTPUTS / "dataset.json").read_text(encoding="utf-8"))
    target_names = dataset["target_names"]
    graphs = [make_graph(library["hands"][sample["topology_template"]], sample) for sample in samples]
    columns = columns or int(math.ceil(math.sqrt(len(samples))))
    rows = int(math.ceil(len(samples) / columns))
    spacing_x, spacing_y = 3.05, 3.20

    root = ET.Element("mujoco", {"model": "DexCoDesign high fidelity graph decoded right hands"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true", "meshdir": str(OUTPUTS)})
    ET.SubElement(root, "option", {"timestep": ".01", "gravity": "0 0 0", "integrator": "implicitfast"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"azimuth": "90", "elevation": "-32", "offwidth": "1920", "offheight": "1080"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"ambient": ".38 .38 .38", "diffuse": ".72 .72 .72", "specular": ".22 .22 .22"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": ".10 .14 .20", "rgb2": ".01 .015 .025", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"name": "floor_tex", "type": "2d", "builtin": "checker", "rgb1": ".18 .20 .25", "rgb2": ".08 .09 .12", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "floor", "texture": "floor_tex", "texrepeat": "14 14", "reflectance": ".12"})
    for index, color in enumerate(COLORS):
        ET.SubElement(asset, "material", {"name": f"handmat_{index}", "rgba": color, "roughness": ".48", "metallic": ".06"})
    mesh_names = add_assets(asset, samples, graphs, library, target_names)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 28", "dir": "0 0 -1", "directional": "true", "castshadow": "true"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": f"{columns * spacing_x / 2 + 2:.2f} {rows * spacing_y / 2 + 2:.2f} .1", "pos": "0 0 -.04", "material": "floor", "contype": "0", "conaffinity": "0"})
    x_center = (columns - 1) * spacing_x / 2
    y_center = (rows - 1) * spacing_y / 2
    realized = []
    for index, (sample, graph) in enumerate(zip(samples, graphs)):
        row, column = divmod(index, columns)
        position = [column * spacing_x - x_center, y_center - row * spacing_y, 0.03]
        template = library["hands"][sample["topology_template"]]
        realized.append(add_hand(world, sample, graph, template, target_names, mesh_names, position, f"handmat_{index % len(COLORS)}"))

    ET.indent(root, space="  ")
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)
    if realized_path is not None:
        realized_path.write_text(json.dumps({"samples": realized}, indent=2) + "\n", encoding="utf-8")
    return scene_path, realized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--page", type=int, choices=range(1, 6), help="Interactively show one 20-hand page (1..5)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = json.loads((OUTPUTS / "generated_hands.json").read_text(encoding="utf-8"))
    if args.page is not None:
        start = (args.page - 1) * 20
        selected = generated["samples"][start : start + 20]
        scene_path = OUTPUTS / "render_pages" / f"page_{args.page:02d}.xml"
        scene, realized = build_scene(selected, scene_path=scene_path, realized_path=None, columns=10)
    elif args.check:
        scene, realized = build_scene(generated["samples"])
    else:
        # A 13M-face 100-hand scene compiles but exceeds the stable interactive
        # framebuffer budget on this MacBook. Default to the first 20-hand page.
        selected = generated["samples"][:20]
        scene_path = OUTPUTS / "render_pages" / "page_01.xml"
        scene, realized = build_scene(selected, scene_path=scene_path, realized_path=None, columns=10)
    print(f"Compiling high-fidelity gallery: {scene}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(
        f"MuJoCo OK: hands={len(realized)}, bodies={model.nbody - 1}, joints={model.njnt}, "
        f"meshes={model.nmesh}, mesh_faces={int(model.mesh_facenum.sum()):,}",
        flush=True,
    )
    if args.check:
        return 0
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
        viewer.cam.distance = 39.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -34.0
        viewer.sync()
        print("Viewer running: left-drag rotate, right-drag pan, wheel zoom, Esc close.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
