"""Dependency-light HandIR structure diagrams."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schema import HandIR

COLORS = {
    "palm": "#f1c75b",
    "base": "#d9b44a",
    "other": "#b7a46a",
    "thumb": "#f26b67",
    "index": "#4e9ee8",
    "middle": "#46bd85",
    "ring": "#a979e8",
    "pinky": "#ee83be",
}
DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def draw_structure_graph(hand: HandIR, path: Path) -> None:
    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "#101722")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    title = ImageFont.load_default(size=28)
    draw.text((55, 35), f"{hand.metadata.get('display_name', hand.hand_id)} — HandIR rigid-link graph", fill="white", font=title)
    nodes = {node.node_id: node for node in hand.nodes}
    parent = {joint.child_node: joint.parent_node for joint in hand.joints}
    role_x = {role: 330 + index * 280 for index, role in enumerate(DIGITS)}
    positions = {}
    for node in hand.nodes:
        if node.semantic_role in DIGITS:
            depth = 1
            cursor = node.node_id
            while cursor in parent and nodes[parent[cursor]].semantic_role == node.semantic_role:
                depth += 1
                cursor = parent[cursor]
            positions[node.node_id] = (role_x[node.semantic_role], 900 - depth * 175)
        else:
            positions[node.node_id] = (900, 1010 - 105 * node.node_id)
    joint_by_child = {joint.child_node: joint for joint in hand.joints}
    for joint in hand.joints:
        start = positions[joint.parent_node]
        end = positions[joint.child_node]
        draw.line((start, end), fill="#aab7c8", width=5)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        label = "R" if joint.active else "F"
        draw.ellipse((mid[0] - 14, mid[1] - 14, mid[0] + 14, mid[1] + 14), fill="#1d2633", outline="#f4f6f8", width=2)
        draw.text((mid[0] - 6, mid[1] - 10), label, fill="white", font=small)
    for node_id, node in nodes.items():
        x, y = positions[node_id]
        color = COLORS.get(node.semantic_role, "#d0d5dc")
        draw.rounded_rectangle((x - 105, y - 48, x + 105, y + 48), radius=18, fill=color, outline="white", width=3)
        draw.text((x - 92, y - 34), f"part {node_id:02d}  {node.semantic_role}", fill="#101722", font=font)
        draw.text((x - 92, y - 8), node.bundle_id.split(":")[-1], fill="#263343", font=small)
        if node_id in joint_by_child:
            draw.text((x - 92, y + 15), joint_by_child[node_id].source_joint_name[-24:], fill="#263343", font=small)
    draw.text((55, 1140), "Node = maximal rigid functional part; edge = source joint; R = active revolute, F = fixed", fill="#dce5ef", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
