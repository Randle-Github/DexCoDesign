#!/usr/bin/env python3
"""Convert MANO's per-link visual meshes to collision-free USD overlays."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--output-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
from pathlib import Path
from xml.etree import ElementTree

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


def main() -> None:
    urdf_path = Path(args.urdf).resolve()
    output_dir = Path(args.output_dir).resolve()
    root = ElementTree.parse(urdf_path).getroot()
    manifest: dict[str, str] = {}

    for link in root.findall("link"):
        mesh = link.find("visual/geometry/mesh")
        if mesh is None:
            continue
        link_name = link.attrib["name"]
        mesh_path = (urdf_path.parent / mesh.attrib["filename"]).resolve()
        link_output_dir = output_dir / link_name
        converter = MeshConverter(
            MeshConverterCfg(
                asset_path=str(mesh_path),
                usd_dir=str(link_output_dir),
                usd_file_name="visual.usd",
                force_usd_conversion=True,
                make_instanceable=False,
                collision_props=None,
                mesh_collision_props=None,
                mass_props=None,
                rigid_props=None,
            )
        )
        manifest[link_name] = str(Path(converter.usd_path).relative_to(output_dir.parent))

    manifest_path = output_dir.parent / "mano_visuals.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(manifest)} collision-free visual overlays: {manifest_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
