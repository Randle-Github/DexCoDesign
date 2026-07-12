#!/usr/bin/env python3
"""Static validation for all registered robot-hand entry points."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "robot_hands"
REGISTRY = ASSETS / "registry.json"


class ValidationError(RuntimeError):
    pass


def _require_file(path: Path, context: str) -> None:
    if not path.is_file():
        raise ValidationError(f"{context}: missing file: {path.relative_to(ROOT)}")
    if path.stat().st_size == 0:
        raise ValidationError(f"{context}: empty file: {path.relative_to(ROOT)}")


def validate_urdf(path: Path) -> tuple[int, int, int]:
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValidationError(f"{path}: expected <robot>, got <{root.tag}>")
    links = [element.get("name") for element in root.findall("link")]
    joints = root.findall("joint")
    if not links or len(links) != len(set(links)):
        raise ValidationError(f"{path}: missing or duplicate link names")
    joint_names = [element.get("name") for element in joints]
    if len(joint_names) != len(set(joint_names)):
        raise ValidationError(f"{path}: duplicate joint names")

    link_set = set(links)
    child_links: set[str] = set()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValidationError(f"{path}: joint {joint.get('name')} has no parent/child")
        parent_name, child_name = parent.get("link"), child.get("link")
        if parent_name not in link_set or child_name not in link_set:
            raise ValidationError(f"{path}: joint {joint.get('name')} references an unknown link")
        if child_name in child_links:
            raise ValidationError(f"{path}: link {child_name} has multiple parents")
        child_links.add(child_name)
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            if float(limit.get("lower")) > float(limit.get("upper")):
                raise ValidationError(f"{path}: reversed limits on joint {joint.get('name')}")

    roots = link_set.difference(child_links)
    if len(roots) != 1:
        raise ValidationError(f"{path}: expected one root link, got {sorted(roots)}")

    mesh_count = 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_count += 1
        if filename.startswith("package://"):
            raise ValidationError(f"{path}: unresolved ROS package URI: {filename}")
        if "://" in filename:
            raise ValidationError(f"{path}: remote mesh URI is not self-contained: {filename}")
        _require_file((path.parent / filename).resolve(), str(path))
    return len(links), len(joints), mesh_count


def validate_mjcf(path: Path) -> tuple[int, int]:
    visited: set[Path] = set()
    mesh_count = 0

    def visit(current: Path, inherited_meshdir: Path | None = None) -> None:
        nonlocal mesh_count
        current = current.resolve()
        if current in visited:
            return
        visited.add(current)
        root = ET.parse(current).getroot()
        if root.tag != "mujoco":
            # MuJoCo include fragments may be rooted at body/default/asset elements.
            pass
        compiler = root.find("compiler")
        meshdir = inherited_meshdir
        if compiler is not None and compiler.get("meshdir"):
            meshdir = (current.parent / compiler.get("meshdir")).resolve()
        for include in root.iter("include"):
            include_file = include.get("file")
            if include_file:
                include_path = (current.parent / include_file).resolve()
                _require_file(include_path, str(current))
                visit(include_path, meshdir)
        for mesh in root.iter("mesh"):
            mesh_file = mesh.get("file")
            if not mesh_file:
                continue
            mesh_count += 1
            base = meshdir if meshdir is not None else current.parent
            _require_file((base / mesh_file).resolve(), str(current))

    visit(path)
    return len(visited), mesh_count


def validate_usd(path: Path) -> int:
    _require_file(path, str(path))
    data = path.read_bytes()
    if not (data.startswith(b"PXR-USDC") or data.lstrip().startswith(b"#usda")):
        raise ValidationError(f"{path}: not a recognized USD crate or USDA layer")
    # Crate string tables expose relative dependency names as plain byte strings.
    dependencies = set(re.findall(rb"[A-Za-z0-9_./-]+\.usd[ac]?", data))
    checked = 0
    for dependency in dependencies:
        relative = dependency.decode("utf-8")
        if relative.startswith("/") or "://" in relative:
            continue
        candidate = (path.parent / relative).resolve()
        if candidate == path.resolve():
            continue
        _require_file(candidate, str(path))
        checked += 1
    return checked


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    failures: list[str] = []
    rows: list[str] = []
    hands = registry.get("hands", {})
    for hand_id, metadata in hands.items():
        entries = metadata.get("entries", {})
        if set(entries) != {"left", "right"}:
            failures.append(f"{hand_id}: registry must contain exactly left and right entries")
            continue
        for side in ("left", "right"):
            entry = entries[side]
            path = ASSETS / entry["path"]
            try:
                _require_file(path, f"{hand_id}/{side}")
                if entry["format"] == "urdf":
                    links, joints, meshes = validate_urdf(path)
                    detail = f"{links} links, {joints} joints, {meshes} mesh refs"
                elif entry["format"] == "mjcf":
                    files, meshes = validate_mjcf(path)
                    detail = f"{files} XML files, {meshes} mesh refs"
                elif entry["format"] == "usd":
                    dependencies = validate_usd(path)
                    detail = f"{dependencies} USD dependencies"
                else:
                    raise ValidationError(f"unsupported format: {entry['format']}")
                rows.append(f"PASS {hand_id:22} {side:5} {entry['format']:4} {detail}")
            except (ET.ParseError, OSError, ValueError, ValidationError) as error:
                failures.append(f"{hand_id}/{side}: {error}")

    print("\n".join(rows))
    if failures:
        print("\nFAILURES", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"\nValidated {len(hands)} hands and {len(rows)} left/right entry points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
