#!/usr/bin/env python3
"""Render the 50 dedicated MiDas variants as one MuJoCo contact sheet."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "dexcodesign"
DEFAULT_ROOT = ROOT / "artifacts" / "hand_morphology" / "midas_constraints_50"


def hand_bounds(hand: dict) -> np.ndarray:
    values = []
    for node in hand["parts"]:
        record = node.get("compiled_mesh")
        if record is None:
            continue
        values.append(
            np.asarray(record["bounds"], dtype=np.float64)
            + np.asarray(node["world_pos"], dtype=np.float64)
        )
    array = np.asarray(values)
    return np.asarray([array[:, 0].min(axis=0), array[:, 1].max(axis=0)])


def render_exact_thumbnail(
    hand: dict,
    generation_root: Path,
    temporary_root: Path,
    tile_size: int,
    camera_distance: float,
) -> Image.Image:
    for old in temporary_root.glob("current_hand_*.stl"):
        old.unlink()
    bounds = hand_bounds(hand)
    center = 0.5 * (bounds[0] + bounds[1])
    mesh_paths = []
    for node in hand["parts"]:
        record = node.get("compiled_mesh")
        if record is None:
            continue
        mesh = trimesh.load(generation_root / record["file"], force="mesh", process=False)
        mesh.apply_translation(np.asarray(node["world_pos"], dtype=np.float64) - center)
        # MuJoCo limits one STL asset to 200k faces. Splitting only partitions
        # the exact face array; it performs no geometric simplification.
        for start in range(0, len(mesh.faces), 190_000):
            faces = np.asarray(mesh.faces[start : start + 190_000], dtype=np.int64)
            chunk = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
                faces=faces.copy(),
                process=False,
            )
            chunk.remove_unreferenced_vertices()
            mesh_path = temporary_root / f"current_hand_{len(mesh_paths):02d}.stl"
            chunk.export(mesh_path, file_type="stl")
            mesh_paths.append(mesh_path)
    xml_path = temporary_root / "current_hand.xml"
    mesh_assets = "\n".join(
        f'    <mesh name="hand_mesh_{index}" file="{path}" smoothnormal="true"/>'
        for index, path in enumerate(mesh_paths)
    )
    mesh_geoms = "\n".join(
        f'    <geom type="mesh" mesh="hand_mesh_{index}" material="hand" mass="0" contype="0" conaffinity="0"/>'
        for index in range(len(mesh_paths))
    )
    xml_path.write_text(
        f"""<mujoco model="{hand['hand_id']}">
  <visual>
    <global offwidth="{tile_size}" offheight="{tile_size}"/>
    <quality shadowsize="2048" offsamples="4"/>
    <headlight ambient=".68 .68 .68" diffuse=".62 .62 .62" specular=".08 .08 .08"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".20 .23 .27" rgb2=".04 .05 .07" width="256" height="1536"/>
    <material name="hand" rgba=".28 .70 .96 1" roughness=".58" metallic=".02" emission=".12"/>
{mesh_assets}
  </asset>
  <worldbody>
    <light pos="-3 -4 8" dir=".2 .2 -1" directional="true"/>
{mesh_geoms}
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.0]
    camera.distance = camera_distance
    camera.azimuth = 90.0
    camera.elevation = -22.0
    with mujoco.Renderer(model, height=tile_size, width=tile_size) as renderer:
        renderer.update_scene(data, camera=camera)
        return Image.fromarray(renderer.render()).convert("RGB")


def render_exact_sheet(
    hands: list[dict], generation_root: Path, output: Path, tile_size: int
) -> None:
    maximum_extent = max(float(np.max(np.ptp(hand_bounds(hand), axis=0))) for hand in hands)
    camera_distance = 2.05 * maximum_extent
    columns = min(10, len(hands))
    rows = math.ceil(len(hands) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size))
    with tempfile.TemporaryDirectory(prefix="midas_exact_render_") as temporary:
        temporary_root = Path(temporary)
        for index, hand in enumerate(hands):
            tile = render_exact_thumbnail(
                hand,
                generation_root,
                temporary_root,
                tile_size,
                camera_distance,
            )
            canvas.paste(
                tile,
                ((index % columns) * tile_size, (index // columns) * tile_size),
            )
            print(f"MIDAS_RENDER_PROGRESS {index + 1}/{len(hands)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_streaming_compiled_sheet(
    source_hands: list[dict],
    generation_root: Path,
    output: Path,
    tile_size: int,
    compiler_python: Path,
    batch_size: int,
) -> None:
    """Compile, render and discard bounded batches of exact morphology meshes."""
    columns = 10
    rows = math.ceil(len(source_hands) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size))
    mesh_root = generation_root / "meshes"
    partial_root = generation_root / "streaming_compiled_parts"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SOURCE)
        if not env.get("PYTHONPATH")
        else f"{SOURCE}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["HAND_GENERATION_ROOT"] = str(generation_root)
    summaries: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="midas_exact_render_") as temporary:
        temporary_root = Path(temporary)
        for batch_start in range(0, len(source_hands), batch_size):
            batch = source_hands[batch_start : batch_start + batch_size]
            if mesh_root.exists():
                shutil.rmtree(mesh_root)
            if partial_root.exists():
                shutil.rmtree(partial_root)
            partial_root.mkdir(parents=True)

            def compile_one(hand: dict) -> Path:
                partial = partial_root / f"{hand['hand_id']}.json"
                subprocess.run(
                    [
                        str(compiler_python),
                        "-m",
                        "dexcodesign.morphology.mesh_compiler",
                        "--hand-id",
                        hand["hand_id"],
                        "--output",
                        str(partial),
                        "--preserve-existing-meshes",
                        "--palm-generation-mode",
                        "hybrid_source_topology",
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                return partial

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                partials = list(pool.map(compile_one, batch))
            payloads = [json.loads(path.read_text()) for path in partials]
            compiled_batch = [payload["hands"][0] for payload in payloads]
            summaries.extend(payload["summary"] for payload in payloads)

            maximum_extent = max(
                float(np.max(np.ptp(hand_bounds(hand), axis=0)))
                for hand in compiled_batch
            )
            camera_distance = 2.05 * maximum_extent
            for offset, hand in enumerate(compiled_batch):
                index = batch_start + offset
                tile = render_exact_thumbnail(
                    hand,
                    generation_root,
                    temporary_root,
                    tile_size,
                    camera_distance,
                )
                canvas.paste(
                    tile,
                    ((index % columns) * tile_size, (index // columns) * tile_size),
                )
                print(f"MIDAS_RENDER_PROGRESS {index + 1}/{len(source_hands)}", flush=True)

    if mesh_root.exists():
        shutil.rmtree(mesh_root)
    if partial_root.exists():
        shutil.rmtree(partial_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    audit = {
        "hands": len(source_hands),
        "parts": sum(int(summary["parts"]) for summary in summaries),
        "meshed_parts": sum(int(summary["meshed_parts"]) for summary in summaries),
        "attachment_roots_checked": sum(
            int(summary["attachment_roots_checked"]) for summary in summaries
        ),
        "disconnected_attachment_roots": sum(
            int(summary["nonoverlapping_attachment_roots"]) for summary in summaries
        ),
        "maximum_palm_interface_frame_error": max(
            float(summary["maximum_palm_interface_frame_error"])
            for summary in summaries
        ),
        "streaming_exact_meshes": True,
        "persistent_mesh_cache_removed": True,
    }
    (generation_root / "streaming_compile_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


def prepare_hand_mesh(hand: dict, generation_root: Path, cache_root: Path, target_faces: int) -> Path:
    output = cache_root / f"{hand['hand_id']}.obj"
    if output.is_file():
        return output
    entries = [node for node in hand["parts"] if node.get("compiled_mesh")]
    total = sum(int(node["compiled_mesh"]["faces"]) for node in entries)
    meshes = []
    for node in entries:
        record = node["compiled_mesh"]
        mesh = trimesh.load(generation_root / record["file"], force="mesh", process=False)
        mesh.remove_unreferenced_vertices()
        quota = max(80, int(round(target_faces * len(mesh.faces) / total)))
        if len(mesh.faces) > quota:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=quota, aggression=7)
            except (IndexError, ValueError):
                # A few source chunks contain duplicate/degenerate faces that
                # fast_simplification accepts as input but indexes incorrectly
                # on output. Repair only this disposable render copy.
                mesh.process(validate=True)
                mesh.remove_unreferenced_vertices()
                mesh = mesh.simplify_quadric_decimation(face_count=quota, aggression=7)
        mesh.apply_translation(np.asarray(node["world_pos"], dtype=np.float64))
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output, file_type="obj", include_normals=True, include_color=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--faces-per-hand", type=int, default=30000)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fast-grid", action="store_true")
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--streaming-compile", action="store_true")
    parser.add_argument("--compile-batch-size", type=int, default=8)
    parser.add_argument(
        "--compiler-python",
        type=Path,
        default=ROOT / ".venv-morphology" / "bin" / "python",
    )
    args = parser.parse_args()
    generation_root = args.generation_root.resolve()
    source_path = generation_root / (
        "hand_ir.json" if args.streaming_compile else "compiled_hands.json"
    )
    hands = json.loads(source_path.read_text())["hands"]
    if not hands:
        raise ValueError("no compiled MiDas variants to render")
    output = args.output or generation_root / f"midas_constraints_{len(hands)}.png"
    if args.streaming_compile:
        render_streaming_compiled_sheet(
            hands,
            generation_root,
            output,
            args.tile_size,
            # Keep the venv launcher path intact. Resolving its symlink points
            # at the base interpreter and silently drops morphology packages.
            args.compiler_python,
            args.compile_batch_size,
        )
        print(f"MIDAS_RENDER_COMPLETE hands={len(hands)} output={output}")
        return 0
    elif not args.fast_grid:
        render_exact_sheet(hands, generation_root, output, args.tile_size)
        print(f"MIDAS_RENDER_COMPLETE hands={len(hands)} output={output}")
        return 0
    cache_root = generation_root / "render_meshes"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(
            pool.map(
                lambda hand: prepare_hand_mesh(
                    hand, generation_root, cache_root, args.faces_per_hand
                ),
                hands,
            )
        )

    columns = 10
    rows = math.ceil(len(hands) / columns)
    spacing_x, spacing_y = 2.35, 2.55
    root = ET.Element("mujoco", {"model": "MiDas constrained morphologies"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(args.width), "offheight": str(args.height)})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": ".78 .78 .78", "diffuse": ".58 .58 .58", "specular": ".08 .08 .08"},
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": ".22 .25 .29",
            "rgb2": ".055 .065 .08",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "hand",
            "rgba": ".30 .72 .96 1",
            "roughness": ".58",
            "metallic": ".02",
            "emission": ".22",
        },
    )
    for index, path in enumerate(paths):
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"hand_mesh_{index}", "file": str(path), "smoothnormal": "true"},
        )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {"pos": "0 -5 28", "dir": "0 .12 -1", "directional": "true", "castshadow": "true"},
    )
    center_x = 0.5 * (columns - 1) * spacing_x
    center_y = 0.5 * (rows - 1) * spacing_y
    for index in range(len(hands)):
        row, column = divmod(index, columns)
        body = ET.SubElement(
            world,
            "body",
            {
                "pos": f"{column * spacing_x - center_x:.7f} {center_y - row * spacing_y:.7f} 0",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": f"hand_mesh_{index}",
                "material": "hand",
                "mass": "0",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    xml_path = generation_root / "midas_constraints_50_render.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.75]
    camera.distance = 18.5
    camera.azimuth = 90.0
    camera.elevation = -22.0
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render()).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"MIDAS_RENDER_COMPLETE hands={len(hands)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
