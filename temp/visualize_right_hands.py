#!/usr/bin/env python3
"""Geometry loaders shared by the macOS all-right-hands gallery."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "robot_hands"
REGISTRY = ASSETS / "registry.json"

# Anatomical landmarks in the neutral/default pose.  Four non-thumb
# fingertips define the wrist-to-fingers axis and palm width; the thumb fixes
# the lateral sign.  Three-finger hands use all available non-thumb fingers.
RIGHT_HAND_LANDMARKS = {
    "mano": ("right_palm", ["right_index1y", "right_middle1y", "right_ring1y", "right_pinky1y"], ["right_index2", "right_middle2", "right_ring2", "right_pinky2"], "right_thumb3"),
    "ability_hand": ("base", ["index_L1", "middle_L1", "ring_L1", "pinky_L1"], ["index_L2", "middle_L2", "ring_L2", "pinky_L2"], "thumb_anchor"),
    "schunk_svh": ("right_hand_base_link", ["right_hand_j", "right_hand_i", "right_hand_l", "right_hand_k"], ["right_hand_n", "right_hand_m", "right_hand_p", "right_hand_o"], "right_hand_c"),
    "wuji_hand_2": ("r_base_link", ["r_index_finger_proximal", "r_middle_finger_proximal", "r_ring_finger_proximal", "r_pinky_proximal"], ["r_index_finger_middle", "r_middle_finger_middle", "r_ring_finger_middle", "r_pinky_middle"], "r_thumb_tip"),
    "sharpa_wave_01": ("right_hand_C_MC", ["right_index_PP", "right_middle_PP", "right_ring_PP", "right_pinky_PP"], ["right_index_MP", "right_middle_MP", "right_ring_MP", "right_pinky_MP"], "right_thumb_fingertip"),
    "tesollo_dg5f": ("rl_dg_mount", ["rl_dg_2_1", "rl_dg_3_1", "rl_dg_4_1", "rl_dg_5_1"], ["rl_dg_2_2", "rl_dg_3_2", "rl_dg_4_2", "rl_dg_5_2"], "rl_dg_1_tip"),
    "unitree_dex5_1": ("base_link00", ["Link_21R", "Link_31R", "Link_41R", "Link_51R"], ["Link_22R", "Link_32R", "Link_42R", "Link_52R"], "Link_14R"),
    "robotera_xhand1": ("right_hand_base_link", ["right_hand_index_bend_link", "right_hand_mid_link1", "right_hand_ring_link1", "right_hand_pinky_link1"], ["right_hand_index_rota_link1", "right_hand_mid_link2", "right_hand_ring_link2", "right_hand_pinky_link2"], "right_hand_thumb_rota_tip"),
    "orca_hand_v2": ("ForeArmStructure-Model_e18f2368", ["I-AP-R_d95d02d1", "M-AP_e04a96f2", "M-AP_6ec59111", "P-AP_f5e42b61"], ["I-PP_bacbd481", "M-PP_08efa608", "M-PP_8660a1eb", "P-PP_1d411b9b"], "T-DP_b7429e50"),
    "shadow_hand_e": ("rh_wrist", ["rh_ffknuckle", "rh_mfknuckle", "rh_rfknuckle", "rh_lfknuckle"], ["rh_ffmiddle", "rh_mfmiddle", "rh_rfmiddle", "rh_lfmiddle"], "rh_thtip"),
    "allegro_hand_v5": ("palm_link", ["link_0_0", "link_4_0", "link_8_0"], ["link_1_0", "link_5_0", "link_9_0"], "link_15_0_tip"),
    "midas_hand": ("palm_base", ["index_mcp_abad_link", "middle_mcp_abad_link", "ring_mcp_abad_link"], ["index_pip_link", "middle_pip_link", "ring_pip_link"], "thumb_dip"),
    "ruka_v2": ("base_new", ["mcp", "mcp_2", "mcp_3", "mcp_4"], ["pip", "pip_2", "pip_3", "pinky___joint_2"], "thumb_actual_tip"),
    "inspire_rh56dfx": ("base", ["index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"], ["index_intermediate", "middle_intermediate", "ring_intermediate", "pinky_intermediate"], "thumb_distal"),
}

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

PALETTE = np.asarray(
    [
        [239, 177, 67, 255],
        [85, 170, 255, 255],
        [109, 211, 160, 255],
        [232, 113, 132, 255],
        [170, 132, 255, 255],
        [255, 151, 85, 255],
        [76, 201, 210, 255],
        [222, 127, 224, 255],
        [151, 203, 91, 255],
        [255, 111, 97, 255],
        [102, 153, 204, 255],
        [196, 164, 132, 255],
        [120, 190, 140, 255],
        [218, 165, 32, 255],
        [135, 150, 175, 255],
    ],
    dtype=np.uint8,
)


@dataclass
class HandScene:
    hand_id: str
    side: str
    display_name: str
    source_format: str
    source_path: Path
    scene: trimesh.Scene
    root_origin: np.ndarray
    canonical_basis: np.ndarray
    vertices: int
    faces: int


def _numbers(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    return np.fromstring(value, sep=" ", dtype=float)


def origin_transform(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if origin is None:
        return transform
    transform[:3, :3] = Rotation.from_euler("xyz", _numbers(origin.get("rpy"), (0.0, 0.0, 0.0))).as_matrix()
    transform[:3, 3] = _numbers(origin.get("xyz"), (0.0, 0.0, 0.0))
    return transform


def colored(mesh: trimesh.Trimesh, rgba: np.ndarray) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1)))
    return mesh


def load_mesh_file(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type in {path}: {type(loaded).__name__}")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def urdf_geometry(geometry: ET.Element, urdf_dir: Path) -> trimesh.Trimesh:
    mesh_node = geometry.find("mesh")
    if mesh_node is not None:
        filename = mesh_node.get("filename")
        if not filename:
            raise ValueError("URDF mesh is missing filename")
        mesh = load_mesh_file((urdf_dir / filename).resolve())
        scale = _numbers(mesh_node.get("scale"), (1.0, 1.0, 1.0))
        if not np.allclose(scale, 1.0):
            mesh.apply_scale(scale)
        return mesh
    box = geometry.find("box")
    if box is not None:
        return trimesh.creation.box(extents=_numbers(box.get("size"), (1.0, 1.0, 1.0)))
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return trimesh.creation.cylinder(
            radius=float(cylinder.get("radius", "1")),
            height=float(cylinder.get("length", "1")),
            sections=32,
        )
    sphere = geometry.find("sphere")
    if sphere is not None:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(sphere.get("radius", "1")))
    raise ValueError("Unsupported URDF visual geometry")


def load_urdf(path: Path, rgba: np.ndarray) -> trimesh.Scene:
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    child_joints: dict[str, list[ET.Element]] = {name: [] for name in links}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name, child_name = parent.get("link"), child.get("link")
        if parent_name in child_joints and child_name in links:
            child_joints[parent_name].append(joint)
            child_links.add(child_name)
    roots = set(links).difference(child_links)
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root in {path}, got {sorted(roots)}")

    scene = trimesh.Scene()
    count = 0

    def visit(link_name: str, link_transform: np.ndarray) -> None:
        nonlocal count
        link = links[link_name]
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = colored(urdf_geometry(geometry, path.parent), rgba)
            scene.add_geometry(
                mesh,
                node_name=f"{link_name}_visual_{count}",
                transform=link_transform @ origin_transform(visual.find("origin")),
            )
            count += 1
        for joint in child_joints[link_name]:
            child = joint.find("child")
            assert child is not None
            visit(child.get("link"), link_transform @ origin_transform(joint.find("origin")))

    visit(next(iter(roots)), np.eye(4))
    if count == 0:
        raise ValueError(f"URDF contains no visual geometry: {path}")
    return scene


def mujoco_primitive(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        return trimesh.Trimesh(
            vertices=np.asarray(model.mesh_vert[vertex_start : vertex_start + vertex_count]),
            faces=np.asarray(model.mesh_face[face_start : face_start + face_count]),
            process=False,
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(size)
        return mesh
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]), sections=24)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]), count=[16, 16])
    return None


def load_mjcf(path: Path, rgba: np.ndarray) -> trimesh.Scene:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = trimesh.Scene()
    included = 0

    explicit_visual_groups = {int(model.geom_group[index]) for index in range(model.ngeom) if int(model.geom_group[index]) in {1, 2}}
    visual_groups = explicit_visual_groups or {0}
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) not in visual_groups:
            continue
        mesh = mujoco_primitive(model, geom_id)
        if mesh is None or mesh.is_empty:
            continue
        colored(mesh, rgba)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        transform[:3, 3] = np.asarray(data.geom_xpos[geom_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        scene.add_geometry(mesh, node_name=f"{name}_{geom_id}", transform=transform)
        included += 1
    if included == 0:
        raise ValueError(f"MJCF contains no visual geometry: {path}")
    return scene


def triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[list[int]] = []
    cursor = 0
    for count in counts.astype(int):
        polygon = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            triangles.append([int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])])
    return np.asarray(triangles, dtype=np.int64)


def load_usd(path: Path, rgba: np.ndarray) -> trimesh.Scene:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"OpenUSD could not open {path}")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    scene = trimesh.Scene()
    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() not in {UsdGeom.Tokens.default_, UsdGeom.Tokens.render}:
            continue
        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get()
        counts = mesh_prim.GetFaceVertexCountsAttr().Get()
        indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        faces = triangulate_faces(np.asarray(counts), np.asarray(indices))
        if len(faces) == 0:
            continue
        mesh = trimesh.Trimesh(vertices=np.asarray(points, dtype=float), faces=faces, process=False)
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        colored(mesh, rgba)
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=float).T
        scene.add_geometry(mesh, node_name=f"usd_mesh_{count}", transform=transform)
        count += 1
    if count == 0:
        raise ValueError(f"USD stage contains no render-purpose meshes: {path}")
    return scene


def normalize_scene(scene: trimesh.Scene, target_scale: float = 2.0) -> np.ndarray:
    """Normalize geometry and return the homogeneous normalization transform.

    The source origin is normally the wrist/root of a hand model.  Keeping it
    lets presentation code distinguish the wrist end from the fingertips
    without changing or guessing the source asset's coordinate convention.
    """
    bounds = np.asarray(scene.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Scene has invalid bounds")
    extents = bounds[1] - bounds[0]
    longest = float(extents.max())
    if longest <= 0.0:
        raise ValueError("Scene has zero extent")
    center = bounds.mean(axis=0)
    scale = target_scale / longest
    transform = np.eye(4)
    transform[:3, 3] = -center
    scene.apply_transform(transform)
    scene.apply_scale(scale)
    normalization = np.eye(4)
    normalization[:3, :3] *= scale
    normalization[:3, 3] = -center * scale
    return normalization


def source_landmark_positions(path: Path, source_format: str) -> dict[str, np.ndarray]:
    """Return neutral-pose link/body origins in the source coordinate frame."""
    if source_format == "urdf":
        root = ET.parse(path).getroot()
        links = {link.get("name") for link in root.findall("link")}
        children: dict[str, list[tuple[str, np.ndarray]]] = {}
        child_links: set[str] = set()
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_name, child_name = parent.get("link"), child.get("link")
            children.setdefault(parent_name, []).append((child_name, origin_transform(joint.find("origin"))))
            child_links.add(child_name)
        roots = links.difference(child_links)
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root in {path}, got {sorted(roots)}")
        positions: dict[str, np.ndarray] = {}

        def visit(link_name: str, transform: np.ndarray) -> None:
            positions[link_name] = transform[:3, 3].copy()
            for child_name, joint_transform in children.get(link_name, []):
                visit(child_name, transform @ joint_transform)

        visit(next(iter(roots)), np.eye(4))
        return positions

    if source_format == "mjcf":
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id): np.asarray(data.xpos[body_id]).copy()
            for body_id in range(1, model.nbody)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        }

    if source_format == "usd":
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError(f"OpenUSD could not open {path}")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        positions = {}
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Xform):
                continue
            transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
            positions[prim.GetName()] = transform[:3, 3].copy()
        return positions

    raise ValueError(f"Unsupported landmark source format: {source_format}")


def canonical_basis(
    landmarks: dict[str, np.ndarray],
    specification: tuple[str, list[str], list[str], str],
    thumb_x_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a root-part frame: X thumb-side, Y palm-normal, Z proximal-to-distal."""
    wrist_name, finger_base_names, fingertip_names, thumb_name = specification
    missing = [name for name in [wrist_name, *finger_base_names, *fingertip_names, thumb_name] if name not in landmarks]
    if missing:
        raise KeyError(f"Missing anatomical landmarks: {missing}")
    wrist = landmarks[wrist_name]
    finger_bases = np.asarray([landmarks[name] for name in finger_base_names])
    fingertips = np.asarray([landmarks[name] for name in fingertip_names])
    thumb = landmarks[thumb_name]

    # The landmarks only identify which *root coordinate axes* mean palm
    # length and width.  Snap to signed cardinal axes instead of fitting a
    # slightly tilted geometric line: q=0 finger splay and unequal lengths must
    # never rotate the root part by a few degrees.
    longitudinal = fingertips.mean(axis=0) - finger_bases.mean(axis=0)
    z_index = int(np.argmax(np.abs(longitudinal)))
    z_axis = np.zeros(3)
    z_axis[z_index] = 1.0 if longitudinal[z_index] >= 0.0 else -1.0

    spreads = np.var(finger_bases, axis=0)
    spreads[z_index] = -np.inf
    x_index = int(np.argmax(spreads))
    x_axis = np.zeros(3)
    thumb_side = thumb - wrist
    x_axis[x_index] = thumb_x_sign * (1.0 if thumb_side[x_index] >= 0.0 else -1.0)
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    return wrist, np.column_stack([x_axis, y_axis, z_axis])


def geometry_totals(scene: trimesh.Scene) -> tuple[int, int]:
    return (
        sum(len(geometry.vertices) for geometry in scene.geometry.values()),
        sum(len(geometry.faces) for geometry in scene.geometry.values()),
    )


def load_all_hands(side: str = "right") -> list[HandScene]:
    if side not in {"left", "right"}:
        raise ValueError(f"Expected left or right, got {side!r}")
    hands = json.loads(REGISTRY.read_text(encoding="utf-8"))["hands"]
    order = ["mano", *(hand_id for hand_id in hands if hand_id != "mano")]
    result: list[HandScene] = []
    for index, hand_id in enumerate(order):
        metadata = hands[hand_id]
        entry = metadata["entries"][side]
        path = (ASSETS / entry["path"]).resolve()
        rgba = PALETTE[index % len(PALETTE)]
        if entry["format"] == "urdf":
            scene = load_urdf(path, rgba)
        elif entry["format"] == "mjcf":
            scene = load_mjcf(path, rgba)
        elif entry["format"] == "usd":
            scene = load_usd(path, rgba)
        else:
            raise ValueError(f"Unsupported format for {hand_id}: {entry['format']}")
        raw_landmarks = source_landmark_positions(path, entry["format"])
        normalization = normalize_scene(scene)
        landmarks = {
            name: normalization[:3, :3] @ position + normalization[:3, 3]
            for name, position in raw_landmarks.items()
        }
        specifications = RIGHT_HAND_LANDMARKS if side == "right" else LEFT_HAND_LANDMARKS
        root_origin, basis = canonical_basis(
            landmarks,
            specifications[hand_id],
            thumb_x_sign=1.0 if side == "right" else -1.0,
        )
        vertices, faces = geometry_totals(scene)
        result.append(
            HandScene(
                hand_id=hand_id,
                side=side,
                display_name=metadata["display_name"],
                source_format=entry["format"],
                source_path=path,
                scene=scene,
                root_origin=root_origin,
                canonical_basis=basis,
                vertices=vertices,
                faces=faces,
            )
        )
        print(
            f"[{side[0].upper()}{index + 1:02d}/{len(order):02d}] {metadata['display_name']:<34} "
            f"{entry['format'].upper():<4} vertices={vertices:,} faces={faces:,}",
            flush=True,
        )
    return result
