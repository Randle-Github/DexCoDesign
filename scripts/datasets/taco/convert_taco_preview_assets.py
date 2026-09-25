"""Batch-convert the compact TACO preview meshes to Isaac Sim USD assets."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--mass", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meshes = sorted(args.input_root.glob("*/*_m.obj"))
    if not meshes:
        raise RuntimeError(f"No meter-scaled OBJ files found under {args.input_root}")

    converted: set[str] = set()
    for mesh in meshes:
        object_id = mesh.stem.removesuffix("_m")
        if object_id in converted:
            continue
        cfg = MeshConverterCfg(
            asset_path=str(mesh.resolve()),
            usd_dir=str(args.output_dir.resolve()),
            usd_file_name=f"{object_id}.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            mass_props=schemas_cfg.MassPropertiesCfg(mass=args.mass),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
        )
        converter = MeshConverter(cfg)
        print(f"Converted {mesh.name} -> {converter.usd_path}")
        converted.add(object_id)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac/RTX shutdown can hang on the headless cluster after conversion.
    # The converter outputs are already flushed, so let Slurm reap the process.
    os._exit(0)
