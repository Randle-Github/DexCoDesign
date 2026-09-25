#!/usr/bin/env python3
"""Build valid two-link URDFs and a typed articulation manifest for ARCTIC."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def empirical_limits(raw_root: Path) -> dict[str, tuple[float, float, int]]:
    values: dict[str, list[np.ndarray]] = {}
    for path in sorted(raw_root.glob("raw_seqs/*/*.object.npy")):
        name = path.name.split("_", 1)[0]
        array = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"Expected Nx7 object state in {path}, got {array.shape}")
        finite = array[np.isfinite(array[:, 0]), 0]
        if finite.size:
            values.setdefault(name, []).append(finite)
    result = {}
    for name, chunks in values.items():
        q = np.concatenate(chunks)
        result[name] = (float(q.min()), float(q.max()), int(q.size))
    return result


def inertial(mesh_path: Path, density: float) -> dict[str, object]:
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {mesh_path}")
    array = np.asarray(vertices, dtype=np.float64) * 0.001
    bounds = np.stack([array.min(axis=0), array.max(axis=0)])
    center = bounds.mean(axis=0)
    size = np.maximum(bounds[1] - bounds[0], 1.0e-4)
    # These meshes are not guaranteed watertight. Use a conservative fraction
    # of the AABB volume, then compute a positive diagonal box inertia.
    mass = max(float(np.prod(size) * density * 0.25), 0.01)
    x, y, z = size
    return {
        "mass": mass,
        "center": center.tolist(),
        "inertia": [
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        ],
        "aabb_size_m": size.tolist(),
        "method": "0.25*AABB volume at source density; approximate",
    }


def vector(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def add_link(robot: ET.Element, name: str, mesh_uri: str, properties: dict[str, object]) -> None:
    link = ET.SubElement(robot, "link", name=name)
    for kind in ("visual", "collision"):
        element = ET.SubElement(link, kind)
        ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "mesh", filename=mesh_uri, scale="0.001 0.001 0.001")
    inertia = ET.SubElement(link, "inertial")
    ET.SubElement(inertia, "origin", xyz=vector(properties["center"]), rpy="0 0 0")
    ET.SubElement(inertia, "mass", value=f"{properties['mass']:.9g}")
    ixx, iyy, izz = properties["inertia"]
    ET.SubElement(
        inertia,
        "inertia",
        ixx=f"{ixx:.9g}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.9g}",
        iyz="0",
        izz=f"{izz:.9g}",
    )


def build_urdf(
    name: str,
    template_dir: Path,
    output: Path,
    lower: float,
    upper: float,
    density: float,
) -> dict[str, object]:
    bottom_props = inertial(template_dir / "bottom.obj", density)
    top_props = inertial(template_dir / "top.obj", density)
    robot = ET.Element("robot", name=f"arctic_{name}")
    add_link(robot, "bottom", f"../object_vtemplates/{name}/bottom.obj", bottom_props)
    add_link(robot, "top", f"../object_vtemplates/{name}/top.obj", top_props)
    joint = ET.SubElement(robot, "joint", name="articulation", type="revolute")
    ET.SubElement(joint, "parent", link="bottom")
    ET.SubElement(joint, "child", link="top")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(joint, "axis", xyz="0 0 -1")
    ET.SubElement(
        joint,
        "limit",
        lower=f"{lower:.9g}",
        upper=f"{upper:.9g}",
        effort="50",
        velocity=f"{math.pi:.9g}",
    )
    ET.SubElement(joint, "dynamics", damping="0.1", friction="0.02")
    ET.indent(robot, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)
    return {"bottom": bottom_props, "top": top_props}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument("--density", type=float, default=567.0)
    args = parser.parse_args()

    templates = args.root / "assets" / "object_vtemplates"
    urdf_root = args.root / "assets" / "object_urdf"
    limits = empirical_limits(args.root / "raw")
    objects = []
    for template in sorted(path for path in templates.iterdir() if path.is_dir()):
        name = template.name
        for part in ("bottom.obj", "top.obj"):
            if not (template / part).is_file():
                raise FileNotFoundError(template / part)
        observed = limits.get(name)
        if observed is None:
            lower, upper, samples = 0.0, math.pi, 0
            limit_source = "fallback ARCTIC convention; refresh after raw sequence download"
        else:
            lower, upper, samples = observed
            if upper - lower < 1.0e-4:
                upper = lower + 1.0e-4
            limit_source = "empirical min/max over downloaded ARCTIC trajectories"
        physical = build_urdf(
            name,
            template,
            urdf_root / f"{name}.urdf",
            lower,
            upper,
            args.density,
        )
        objects.append(
            {
                "object_id": name,
                "graph": {
                    "root_link": "bottom",
                    "nodes": [
                        {"id": "bottom", "mesh": f"object_vtemplates/{name}/bottom.obj"},
                        {"id": "top", "mesh": f"object_vtemplates/{name}/top.obj"},
                    ],
                    "edges": [
                        {
                            "id": "articulation",
                            "type": "revolute",
                            "parent": "bottom",
                            "child": "top",
                            "origin_xyz_m": [0.0, 0.0, 0.0],
                            "axis_parent": [0.0, 0.0, -1.0],
                            "limit_rad": [lower, upper],
                            "limit_source": limit_source,
                            "observed_samples": samples,
                        }
                    ],
                },
                "trajectory_state": {
                    "root": "SE3(position_m, quaternion_wxyz)",
                    "joint_position": "one scalar radian, q[articulation]",
                    "source_object_npy": "[q_rad, root_axis_angle_xyz, root_translation_mm]",
                },
                "physical": physical,
                "urdf": f"object_urdf/{name}.urdf",
            }
        )

    manifest = {
        "schema": "dexcodesign.arctic_articulated_objects.v1",
        "source": {
            "dataset": "ARCTIC",
            "official_project": "https://arctic.is.tue.mpg.de/",
            "mesh_urdf_resource": "https://github.com/cypypccpy/ObjDexEnvs",
            "units_in_source_mesh": "millimeter",
            "license_note": "ARCTIC is non-commercial; retain upstream terms.",
        },
        "representation": {
            "root_pose": "floating object pose supplied per frame, not encoded as an internal URDF joint",
            "internal_articulation": "bottom --revolute(-Z)--> top",
            "top_transform": "Rz(-q) in object canonical coordinates, then apply root SE(3)",
        },
        "objects": objects,
    }
    output = args.root / "manifests" / "articulated_objects.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ARCTIC_OBJECTS_READY objects={len(objects)} manifest={output.resolve()}")


if __name__ == "__main__":
    main()
