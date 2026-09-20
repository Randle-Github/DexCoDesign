#!/usr/bin/env python3
"""Compare the original WUJI hand with the highest mean-reward proposal.

This is a geometry-only, zero-joint-position inspection. The prototype's
exported mesh surfaces receive the saved runtime USD morphology transforms;
it is not a new rollout or a reconstruction of a successful simulator state.
Both hands use identical cameras, lighting, and metric scale, in front and
three-quarter views. The original comes from the unmodified source URDF.

From the repository root, with the project's Conda environment active:

    MUJOCO_GL=osmesa python temp/hocap_mano_replay/scripts/render_best_wuji_sac_hand.py RUN_ROOT

For results saved before the connector-mesh fix, add --corrected-geometry.
That preview does not inherit the old geometry's reward or survival results.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[3]
# Allow direct invocation from any working directory, without depending on an
# editable package installation that might still point to a different checkout.
sys.path.insert(0, str(REPO_ROOT / "source/dexcodesign"))

from dexcodesign.morphology.parametric_mesh import (  # noqa: E402
    connector_deformation_overlay,
    deform_template_points,
)

ORIGINAL_URDF = REPO_ROOT / "assets/robot_hands/direct_motor/wuji_hand_2/left/hand.urdf"


def load_json(path: Path):
    return json.loads(path.read_text())


def template_urdf(manifest: dict, index: int) -> Path:
    """Find the bank's exported surfaces, not a materialized candidate overlay."""
    paths = manifest.get("parametric_template_usd_paths")
    usd = Path(
        paths[index]
        if paths is not None
        else manifest.get("parametric_template_usd") or manifest["hand_usd_paths"][index]
    )
    urdfs = list((usd.parent.parent / "runtime").glob("*/left/hand.urdf"))
    if len(urdfs) != 1:
        raise ValueError(f"Cannot locate prototype URDF for {usd}")
    return urdfs[0]


def best_proposal(run_root: Path):
    history = load_json(run_root / "training_history.json")
    entry = max(history, key=lambda row: row["best_reward"])
    generation = int(entry["generation"])
    folder = run_root / f"generation_{generation:03d}"
    results = load_json(folder / "physx_results.json")
    best = max(results["results"], key=lambda row: row["total_reward"])
    if not np.isclose(best["total_reward"], entry["best_reward"]):
        raise ValueError("Training history and generation results disagree")
    rollouts = [row for row in results["rollout_results"] if row["candidate_index"] == best["candidate_index"]]
    return generation, folder, best, rollouts


def corrected_manifest(folder: Path, manifest: dict, index: int) -> dict:
    """Reconstruct intended geometry without changing an old run's manifest.

    This is a new geometry preview, not the geometry used for its old rewards.
    Recover the exact prototype frame rather than merely guessing whether an
    old affine should be transposed.
    """
    result = copy.deepcopy(manifest)
    hand = load_json(folder / "hand_ir/hand_ir.json")["hands"][index]
    prototype_root = template_urdf(manifest, index).parents[3]
    compiled_path = prototype_root.parent.parent / "compiled/compiled_hands.json"
    prototypes = load_json(compiled_path)["hands"]
    prototype = next(p for p in prototypes if p["hand_id"] == prototype_root.name)
    baseline_parts = {int(p["id"]): p for p in prototype["parts"]}
    repo_root = next(
        (p for p in (folder, *folder.parents) if (p / "isaaclab.sh").is_file()),
        Path.cwd(),
    )
    source = next(
        h
        for h in load_json(repo_root / "artifacts/hand_morphology/reference_graphs.json")["hands"]
        if h["hand_id"] == hand["seed_source"]
    )
    audit = source.get("canonicalization", source.get("direct_geometry_audit"))
    polar = np.asarray(audit["similarity_rotation"]).T @ np.diag([-1, 1, 1])
    source_scale = float(audit["similarity_scale"])
    transforms = []
    deformations = []
    translations = []
    positions = {}
    joint_names = []
    joint_positions = []
    for part in hand["parts"]:
        part_id = int(part["id"])
        baseline = baseline_parts[part_id]
        affine = np.eye(4)
        affine[:3, :3] = np.linalg.solve(
            np.asarray(baseline["mesh_linear"]).T @ polar,
            np.asarray(part["mesh_linear"]).T @ polar,
        )
        transforms.append(affine.tolist())
        deformations.append(connector_deformation_overlay(part, baseline, polar, source_scale))
        local = np.asarray(part["relative_pos"]) @ polar / source_scale
        parent = part["parent"]
        positions[part_id] = local if parent is None else local + positions[int(parent)]
        translations.append(positions[part_id].tolist())
        if parent is not None:
            joint_names.append(part["joint_name"])
            joint_positions.append(local.tolist())
    result["parametric_relative_transforms"][index] = transforms
    result["parametric_link_translations"][index] = translations
    result["parametric_joint_names"][index] = joint_names
    result["parametric_joint_local_positions"][index] = joint_positions
    if "parametric_mesh_deformations" not in result:
        result["parametric_mesh_deformations"] = [[] for _ in result["vectors"]]
    result["parametric_mesh_deformations"][index] = deformations
    return result


def exact_meshes(manifest: dict, index: int, output: Path):
    urdf = template_urdf(manifest, index)
    links = {link.attrib["name"]: link for link in ET.parse(urdf).getroot().findall("link")}
    mesh_root = output / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    mesh_records = []
    all_bounds = []
    deformations = manifest.get("parametric_mesh_deformations")
    if deformations is not None and len(deformations[index]) != len(manifest["parametric_link_names"][index]):
        raise ValueError("Mesh-deformation/link count mismatch")
    for part_index, (link_name, transform, translation) in enumerate(
        zip(
            manifest["parametric_link_names"][index],
            manifest["parametric_relative_transforms"][index],
            manifest["parametric_link_translations"][index],
            strict=True,
        )
    ):
        geometry = links[link_name].find("collision")
        if geometry is None:
            raise ValueError(f"No prototype collision surface for {link_name}")
        origin = geometry.find("origin")
        if origin is not None:
            for field in ("xyz", "rpy"):
                if not np.allclose(np.fromstring(origin.get(field, "0 0 0"), sep=" "), 0):
                    raise ValueError("Nonzero geometry origins require additional transforms")
        mesh_element = geometry.find("geometry/mesh")
        source = urdf.parent / mesh_element.attrib["filename"]
        mesh = trimesh.load(source, force="mesh", process=False)
        scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
        matrix = np.asarray(transform, dtype=np.float64)
        # Gf.Matrix4d is row-vector oriented. The saved matrix is passed
        # directly to Gf in ManoResidualEnv._apply_runtime_morphology_overlays.
        points = np.asarray(mesh.vertices, dtype=np.float64) * scale
        overlay = None if deformations is None else deformations[index][part_index]
        if overlay is not None:
            points = deform_template_points(points, overlay)
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix
        mesh.vertices = homogeneous[:, :3] / homogeneous[:, 3, None]
        mesh.apply_translation(translation)
        if np.linalg.det(matrix[:3, :3]) < 0:
            mesh.faces = np.asarray(mesh.faces)[:, ::-1]
        all_bounds.append(mesh.bounds)
        mesh_records.extend(export_mesh_chunks(mesh, link_name, mesh_root))
    bounds = np.asarray(all_bounds)
    return mesh_records, np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)]), urdf


def export_mesh_chunks(mesh: trimesh.Trimesh, name: str, output: Path) -> list:
    """Partition faces for MuJoCo's STL limit without simplifying any surface."""
    records = []
    for start in range(0, len(mesh.faces), 190000):
        chunk = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices).copy(),
            faces=np.asarray(mesh.faces[start : start + 190000]).copy(),
            process=False,
        )
        chunk.remove_unreferenced_vertices()
        path = output / f"{name}_{start // 190000:02d}.stl"
        chunk.export(path)
        records.append((name, path))
    return records


def origin_transform(origin: ET.Element | None) -> np.ndarray:
    """URDF local-to-parent transform, with the standard Rz(yaw) Ry(pitch) Rx(roll)."""
    transform = np.eye(4)
    if origin is not None:
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        if len(xyz) != 3 or len(rpy) != 3:
            raise ValueError("URDF origins require three xyz and rpy values")
        transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        transform[:3, 3] = xyz
    return transform


def zero_pose_link_transforms(root: ET.Element) -> dict[str, np.ndarray]:
    """Accumulate all fixed and movable joint origins at q=0, in the base frame."""
    links = {link.attrib["name"] for link in root.findall("link")}
    parents = {}
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]
        parent = joint.find("parent").attrib["link"]
        if child not in links or parent not in links or child in parents:
            raise ValueError(f"Invalid URDF joint tree at {joint.attrib['name']}")
        parents[child] = (parent, origin_transform(joint.find("origin")))
    bases = links - parents.keys()
    if len(bases) != 1:
        raise ValueError(f"Expected one original hand base link, got {sorted(bases)}")
    transforms = {next(iter(bases)): np.eye(4)}
    pending = dict(parents)
    while pending:
        resolved = [child for child, (parent, _) in pending.items() if parent in transforms]
        if not resolved:
            raise ValueError("Cycle or disconnected links in original hand URDF")
        for child in resolved:
            parent, local = pending.pop(child)
            transforms[child] = transforms[parent] @ local
    return transforms


def original_meshes(urdf: Path, output: Path):
    """Assemble source collision surfaces, including wrist and mesh-local origins."""
    root = ET.parse(urdf).getroot()
    transforms = zero_pose_link_transforms(root)
    mesh_root = output / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    records = []
    all_bounds = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        for number, collision in enumerate(link.findall("collision")):
            element = collision.find("geometry/mesh")
            if element is None:
                raise ValueError(f"Expected a source collision mesh for {name}")
            source = (urdf.parent / element.attrib["filename"]).resolve()
            mesh = trimesh.load(source, force="mesh", process=False)
            scale = np.fromstring(element.get("scale", "1 1 1"), sep=" ")
            if len(scale) != 3 or not np.all(np.isfinite(scale)) or np.any(scale == 0):
                raise ValueError(f"Invalid original mesh scale for {name}")
            mesh.vertices = np.asarray(mesh.vertices) * scale
            if np.prod(scale) < 0:
                mesh.faces = np.asarray(mesh.faces)[:, ::-1]
            mesh.apply_transform(transforms[name] @ origin_transform(collision.find("origin")))
            all_bounds.append(mesh.bounds)
            records.extend(export_mesh_chunks(mesh, f"{name}_{number:02d}", mesh_root))
    if not records:
        raise ValueError(f"No original hand collision meshes in {urdf}")
    bounds = np.asarray(all_bounds)
    return records, np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])


def make_scene(records, output: Path, width: int, height: int) -> Path:
    root = ET.Element("mujoco", model="best WUJI morphology - geometry inspection")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", ambient=".58 .58 .58", diffuse=".62 .62 .62", specular=".12 .12 .12")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".12 .16 .22",
        rgb2=".035 .05 .08",
        width="256",
        height="1536",
    )
    ET.SubElement(asset, "material", name="hand", rgba=".22 .66 .90 1", roughness=".58", metallic=".02", emission=".02")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-1 -2 3", dir=".1 .2 -1", directional="true")
    for number, (_, mesh) in enumerate(records):
        name = f"part_{number:03d}"
        ET.SubElement(asset, "mesh", name=name, file=str(mesh), smoothnormal="true")
        ET.SubElement(world, "geom", type="mesh", mesh=name, material="hand", mass="0", contype="0", conaffinity="0")
    ET.indent(root, space="  ")
    path = output / "inspection_scene.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def render_views(records, output: Path, width: int, height: int, center: np.ndarray, distance: float):
    """Render one hand with the camera settings shared by both comparisons."""
    scene = make_scene(records, output, width, height)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tiles = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for azimuth, elevation, label in [(90, 0, "Front"), (125, -18, "Three-quarter")]:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.lookat[:] = center
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera)
            tile = Image.fromarray(renderer.render())
            tile.save(output / f"{label.lower().replace('-', '_')}.png")
            tiles.append((label, tile))
    return tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--corrected-geometry",
        action="store_true",
        help="preview intended HandIR geometry; old rewards do not describe this corrected mesh",
    )
    parser.add_argument(
        "--output-root", type=Path, help="comparison image, source/candidate views, meshes and selection JSON"
    )
    parser.add_argument(
        "--original-urdf",
        type=Path,
        default=ORIGINAL_URDF,
        help="unmodified WUJI source URDF (default: original left hand)",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    generation, folder, best, rollouts = best_proposal(run_root)
    index = int(best["candidate_index"])
    suffix = "_corrected_geometry" if args.corrected_geometry else ""
    output = (
        args.output_root.resolve()
        if args.output_root is not None
        else run_root / f"best_hand_generation_{generation:03d}_candidate_{index:03d}{suffix}_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(folder / "prepared/physx_batch_manifest.json")
    if args.corrected_geometry:
        manifest = corrected_manifest(folder, manifest, index)
    elif "parametric_mesh_deformations" not in manifest:
        print(
            "Rendering the legacy recorded geometry, not the mesh fix. Use --corrected-geometry for an intended-geometry preview."
        )
    records, bounds, urdf = exact_meshes(manifest, index, output / "generated")
    original_urdf = args.original_urdf.resolve()
    source_records, source_bounds = original_meshes(original_urdf, output / "original")
    width, height, header, footer = 800, 700, 115, 85
    # A shared base-frame target/distance retains true size and positional
    # differences; never independently normalize either hand to fill its panel.
    comparison_bounds = np.array([np.minimum(bounds[0], source_bounds[0]), np.maximum(bounds[1], source_bounds[1])])
    center = comparison_bounds.mean(axis=0)
    distance = 1.75 * float(np.max(comparison_bounds[1] - comparison_bounds[0]))
    canvas = Image.new("RGB", (2 * width, header + 2 * height + footer), (17, 24, 34))
    for column, (hand_records, hand_output, name) in enumerate(
        [
            (source_records, output / "original", "Original WUJI hand 2"),
            (records, output / "generated", "Generated WUJI hand 2"),
        ]
    ):
        for row, (label, tile) in enumerate(render_views(hand_records, hand_output, width, height, center, distance)):
            top = header + row * height
            canvas.paste(tile, (column * width, top))
            ImageDraw.Draw(canvas).text(
                (column * width + 25, top + 20), f"{name} | {label}", font=font(22), fill=(224, 236, 246)
            )
    draw = ImageDraw.Draw(canvas)
    title = (
        "WUJI hand 2 | Corrected geometry preview"
        if args.corrected_geometry
        else "WUJI hand 2 | Best recorded mean-reward proposal"
    )
    detail = (
        f"Generation {generation}  /  Candidate {index}  /  Selected by OLD mean reward {best['total_reward']:.2f}"
        if args.corrected_geometry
        else f"Generation {generation}  /  Candidate {index}  /  Mean reward {best['total_reward']:.2f}  /  {len(rollouts)} rollouts"
    )
    draw.text((28, 17), title, font=font(30), fill=(235, 244, 251))
    draw.text((28, 66), detail, font=font(22), fill=(147, 203, 229))
    phases = ", ".join(str(r["phase"]) for r in rollouts)
    performance = (
        "Geometry corrected; reward and survival must be reevaluated."
        if args.corrected_geometry
        else f"Survival: {phases} / 445   |   Mean: {best['phase_mean']:.2f}/445 ({100 * best['survival_ratio_mean']:.1f}%)"
    )
    draw.text((28, header + 2 * height + 14), performance, font=font(21), fill=(229, 237, 246))
    draw.text(
        (28, header + 2 * height + 48),
        "Both q=0; identical cameras and metric scale. Geometry inspection only, not a new task evaluation.",
        font=font(18),
        fill=(150, 168, 188),
    )
    preview = output / "best_hand.png"
    canvas.save(preview)
    selection = {
        "selection_criterion": "highest recorded proposal mean total_reward across all generations",
        "run_root": str(run_root),
        "generation": generation,
        "best": best,
        "rollouts": rollouts,
        "prototype_urdf": str(urdf),
        "original_urdf": str(original_urdf),
        "original_bounds_m": source_bounds.tolist(),
        "comparison_camera": {"lookat_m": center.tolist(), "distance_m": distance, "identical_metric_scale": True},
        "source_manifest": str(folder / "prepared/physx_batch_manifest.json"),
        "reference_trajectory": manifest["reference_paths"][index],
        "geometry_mode": "corrected_from_hand_ir" if args.corrected_geometry else "recorded_manifest",
        "evaluation_matches_rendered_geometry": not args.corrected_geometry,
        "render_semantics": "static zero-joint-position inspection with prototype surfaces, USD/Gf affine and connector-preserving vertex overlays; no new physics evaluation",
        "preview": str(preview),
        "bounds_m": bounds.tolist(),
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps(
            {
                "generation": generation,
                "candidate_index": index,
                "old_mean_reward" if args.corrected_geometry else "mean_reward": best["total_reward"],
                "evaluation_matches_rendered_geometry": not args.corrected_geometry,
                "phases": [r["phase"] for r in rollouts],
                "preview": str(preview),
                "bounds_m": bounds.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
