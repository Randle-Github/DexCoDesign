"""Fixed-WUJI contact-surface observations, independent of Isaac Sim.

All quaternions use wxyz. Each of the 20 moving finger links contributes an
ordered, non-collinear triangle on its inward collision surface: a pad-center
point at half the kinematic segment length and left/right points at 80% of the
length. Reference landmarks are produced by FK of ``hand_q``. The live
landmarks use the same link-local offsets with body poses reported by Isaac.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import trimesh


MOVING_LINK_COUNT = 20
SURFACE_POINTS_PER_LINK = 3
LANDMARK_COUNT = MOVING_LINK_COUNT * SURFACE_POINTS_PER_LINK
# Current/reference surface XYZ, reference palm pose, current/reference object
# poses, palm-frame gravity, and phase. The distal-link surface triangles already
# encode fingertip position and orientation, so separate fingertip goals would be
# redundant.
OBSERVATION_DIM = 6 * LANDMARK_COUNT + 3 * 7 + 3 + 1

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    uv = torch.linalg.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(q[..., 1:], uv))


def palm_points(points: torch.Tensor, palm_pose: torch.Tensor) -> torch.Tensor:
    q = quat_inverse(palm_pose[..., 3:]).unsqueeze(-2).expand(*points.shape[:-1], 4)
    return quat_apply(q, points - palm_pose[..., None, :3])


def palm_poses(poses: torch.Tensor, palm_pose: torch.Tensor) -> torch.Tensor:
    inverse = quat_inverse(palm_pose[..., 3:])
    while inverse.ndim < poses.ndim:
        inverse = inverse.unsqueeze(-2)
        palm_pose = palm_pose.unsqueeze(-2)
    inverse = inverse.expand(*poses.shape[:-1], 4)
    rotation = quat_mul(inverse, poses[..., 3:])
    # q and -q represent the same orientation; give the networks one convention.
    rotation = torch.where(rotation[..., :1] < 0, -rotation, rotation)
    return torch.cat((quat_apply(inverse, poses[..., :3] - palm_pose[..., :3]), rotation), -1)


def build_observation(
    current_points: torch.Tensor,
    reference_points: torch.Tensor,
    palm_pose: torch.Tensor,
    reference_palm_pose: torch.Tensor,
    object_pose: torch.Tensor,
    reference_object_pose: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    gravity = torch.zeros_like(palm_pose[..., :3])
    gravity[..., 2] = -1
    return torch.cat((
        palm_points(current_points, palm_pose).flatten(1),
        palm_points(reference_points, palm_pose).flatten(1),
        palm_poses(reference_palm_pose, palm_pose),
        palm_poses(object_pose, palm_pose),
        palm_poses(reference_object_pose, palm_pose),
        quat_apply(quat_inverse(palm_pose[..., 3:]), gravity),
        phase.unsqueeze(-1),
    ), -1)


class FixedWujiGeometry:
    """Read the fixed WUJI URDF and construct collision-surface landmarks."""

    _POINT_LABELS = ("pad_center", "distal_left", "distal_right")

    def __init__(
        self,
        urdf_path: Path,
        joint_names: list[str],
        palm_name: str,
        reference_q: torch.Tensor | None = None,
        reference_object_positions: torch.Tensor | None = None,
        collision_approximation: str | Mapping[str, str] = "convexHull",
        inward_direction_mode: str = "reference_object",
        joint_origin_overrides: Mapping[str, list[float]] | None = None,
        collision_mesh_transforms: Mapping[str, list[list[float]]] | None = None,
        collision_mesh_deformations: Mapping[str, dict | None] | None = None,
    ):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.joints = list(root.findall("joint"))
        self.links = {link.get("name"): link for link in root.findall("link")}
        self.joint_names = joint_names
        self.palm_name = palm_name
        self.collision_approximation = collision_approximation
        self.inward_direction_mode = inward_direction_mode
        self.joint_origin_overrides = dict(joint_origin_overrides or {})
        self.collision_mesh_transforms = dict(collision_mesh_transforms or {})
        self.collision_mesh_deformations = dict(collision_mesh_deformations or {})
        self.finger_joints = [
            joint for joint in self.joints
            if joint.get("name", "").startswith("finger__")
            and joint.get("type") in ("revolute", "continuous")
        ]
        if len(self.finger_joints) != MOVING_LINK_COUNT:
            raise ValueError(
                f"palm_geometry requires {MOVING_LINK_COUNT} moving WUJI finger links; "
                f"found {len(self.finger_joints)}"
            )
        children = {joint.find("child").get("link") for joint in self.joints}
        roots = {link.get("name") for link in root.findall("link")} - children
        if len(roots) != 1:
            raise ValueError("reference URDF must have one root")
        self.root_name = roots.pop()
        missing = {
            joint.get("name") for joint in self.joints if joint.get("type") != "fixed"
        } - set(joint_names)
        if missing:
            raise ValueError(f"reference joint_names missing URDF joints: {sorted(missing)}")

        self.segment_body_names = [joint.find("child").get("link") for joint in self.finger_joints]
        self.body_names = [
            body_name
            for body_name in self.segment_body_names
            for _ in range(SURFACE_POINTS_PER_LINK)
        ]
        self.landmark_names = [
            f"{joint.get('name')}:{label}"
            for joint in self.finger_joints
            for label in self._POINT_LABELS
        ]
        inward_signs = self._inward_signs(reference_q, reference_object_positions)
        triangles = [
            self._surface_triangle(joint, inward_sign)
            for joint, inward_sign in zip(self.finger_joints, inward_signs, strict=True)
        ]
        self.offsets = torch.as_tensor(np.concatenate(triangles), dtype=torch.float32)
        if self.offsets.shape != (LANDMARK_COUNT, 3):
            raise RuntimeError(f"unexpected WUJI surface landmark shape: {self.offsets.shape}")

    @staticmethod
    def _mapped(mapping: Mapping, name: str):
        """Resolve raw graph names against wrapped URDF link/joint names."""
        if name in mapping:
            return mapping[name]
        matches = [
            value for key, value in mapping.items()
            if name.endswith(f"__{key}") or key.endswith(f"__{name}")
        ]
        if len(matches) > 1:
            raise ValueError(f"ambiguous morphology override for {name!r}")
        return None if not matches else matches[0]

    def _origin_xyz(self, joint: ET.Element) -> np.ndarray:
        override = self._mapped(self.joint_origin_overrides, joint.get("name", ""))
        if override is not None:
            value = np.asarray(override, dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"invalid morphology joint origin for {joint.get('name')}")
            return value
        return np.asarray(
            self._vector(joint.find("origin"), "xyz", "0 0 0"), dtype=np.float64
        )

    def _deform_body_points(self, body_name: str, points: np.ndarray) -> np.ndarray:
        """Apply the same deformation and affine used by the runtime USD overlay."""
        result = np.asarray(points, dtype=np.float64)
        deformation = self._mapped(self.collision_mesh_deformations, body_name)
        if deformation is not None:
            from dexcodesign.morphology.parametric_mesh import deform_template_points

            result = deform_template_points(result, deformation)
        transform = self._mapped(self.collision_mesh_transforms, body_name)
        if transform is not None:
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"invalid morphology mesh transform for {body_name}")
            homogeneous = np.column_stack((result, np.ones(len(result))))
            result = (homogeneous @ matrix)[:, :3]
        return result

    @staticmethod
    def _vector(element, key, default):
        return [float(x) for x in (default if element is None else element.get(key, default)).split()]

    @staticmethod
    def _rpy_matrix(rpy: list[float]) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.asarray((
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ), dtype=np.float64)

    def _successor_origin(
        self, body_name: str, mesh: trimesh.Trimesh | None = None
    ) -> np.ndarray:
        successors = [
            joint for joint in self.joints
            if joint.find("parent").get("link") == body_name
            and (
                joint.get("name", "").startswith("finger__")
                or joint.get("name", "").endswith("tip_fixed")
            )
        ]
        if len(successors) > 1:
            raise ValueError(
                f"WUJI moving link {body_name!r} needs one distal joint/tip; "
                f"found {[joint.get('name') for joint in successors]}"
            )
        if not successors:
            # Prototype-bank URDFs intentionally omit massless fingertip links.
            # Recover the distal direction from the actual overlaid collision
            # body rather than inventing a fixed tip offset.
            mesh = self._physics_collision_mesh(body_name) if mesh is None else mesh
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            centered = vertices - vertices.mean(axis=0)
            values, vectors = np.linalg.eigh(centered.T @ centered)
            longitudinal = vectors[:, int(np.argmax(values))]
            if float(vertices.mean(axis=0) @ longitudinal) < 0.0:
                longitudinal *= -1.0
            projection = vertices @ longitudinal
            length = float(np.quantile(projection, 0.98))
            if not np.isfinite(length) or length <= 1.0e-6:
                raise ValueError(
                    f"cannot infer terminal segment length for {body_name}"
                )
            return length * longitudinal
        successor = successors[0]
        explicit = self._mapped(
            self.joint_origin_overrides, successor.get("name", "")
        )
        if explicit is not None:
            return np.asarray(explicit, dtype=np.float64)
        origin = self._origin_xyz(successor)
        # The terminal fixed tip has no graph-joint override. Its position is
        # nevertheless part of the deformed distal-link geometry.
        return self._deform_body_points(body_name, origin[None])[0]

    def _collision_mesh(self, body_name: str) -> trimesh.Trimesh:
        """Load the URDF collision triangles in the corresponding link frame."""
        link = self.links[body_name]
        meshes: list[trimesh.Trimesh] = []
        for collision in link.findall("collision"):
            mesh_element = collision.find("geometry/mesh")
            if mesh_element is None:
                continue
            mesh_path = Path(mesh_element.get("filename"))
            if not mesh_path.is_absolute():
                mesh_path = self.urdf_path.parent / mesh_path
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.size == 0:
                raise ValueError(f"cannot load collision mesh for {body_name}: {mesh_path}")
            vertices = np.asarray(loaded.vertices, dtype=np.float64).copy()
            scale = np.asarray(self._vector(mesh_element, "scale", "1 1 1"), dtype=np.float64)
            vertices *= scale
            origin = collision.find("origin")
            rotation = self._rpy_matrix(self._vector(origin, "rpy", "0 0 0"))
            translation = np.asarray(self._vector(origin, "xyz", "0 0 0"), dtype=np.float64)
            vertices = vertices @ rotation.T + translation
            meshes.append(trimesh.Trimesh(vertices=vertices, faces=loaded.faces.copy(), process=False))
        if not meshes:
            raise ValueError(f"WUJI moving link {body_name!r} has no collision mesh")
        mesh = trimesh.util.concatenate(meshes)
        mesh.vertices = self._deform_body_points(body_name, mesh.vertices)
        return mesh

    def _physics_collision_mesh(self, body_name: str) -> trimesh.Trimesh:
        """Reproduce the collision approximation authored for this USD link.

        ``convexHull`` contacts the convex hull of the source triangles, while
        ``none`` uses the source triangle surface. Other PhysX approximations
        cannot be reconstructed exactly from the URDF alone and are rejected
        instead of silently producing landmarks on the wrong surface.
        """
        configured = self.collision_approximation
        approximation = (
            self._mapped(configured, body_name)
            if isinstance(configured, Mapping)
            else configured
        )
        aliases = {
            "convexHull": "convexHull",
            "convex_hull": "convexHull",
            "none": "none",
            "source_mesh": "none",
            "triangle_mesh": "none",
        }
        if approximation not in aliases:
            raise ValueError(
                f"unsupported collision approximation {approximation!r} for {body_name}; "
                "exact contact landmarks currently support only convexHull and none"
            )
        approximation = aliases[approximation]
        mesh = self._collision_mesh(body_name)
        if approximation == "convexHull":
            mesh = mesh.convex_hull
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
            raise ValueError(f"cannot construct {approximation} collision surface for {body_name}")
        return mesh

    @staticmethod
    def _normal_sign_from_reference(
        normal: np.ndarray,
        link_positions: torch.Tensor,
        link_rotations: torch.Tensor,
        object_positions: torch.Tensor,
    ) -> float:
        local_vectors = quat_apply(
            quat_inverse(link_rotations), object_positions - link_positions
        )
        distances = torch.linalg.vector_norm(local_vectors, dim=-1)
        count = max(1, min(local_vectors.shape[0], local_vectors.shape[0] // 4))
        nearest = torch.topk(distances, k=count, largest=False).indices
        score = torch.median(
            local_vectors[nearest] @ torch.as_tensor(normal, dtype=local_vectors.dtype)
        ).item()
        return 1.0 if score >= 0.0 else -1.0

    def _inward_signs(
        self,
        reference_q: torch.Tensor | None,
        reference_object_positions: torch.Tensor | None,
    ) -> list[float]:
        if self.inward_direction_mode == "kinematic_normal":
            return [1.0] * len(self.finger_joints)
        if self.inward_direction_mode == "negative_kinematic_normal":
            return [-1.0] * len(self.finger_joints)
        if self.inward_direction_mode != "reference_object":
            raise ValueError(
                f"unknown inward_direction_mode: {self.inward_direction_mode!r}; "
                "expected reference_object, kinematic_normal, or negative_kinematic_normal"
            )
        if reference_q is None or reference_object_positions is None:
            raise ValueError(
                "reference_object inward direction requires reference_q and "
                "reference_object_positions"
            )
        if reference_q.ndim != 2 or reference_object_positions.shape != (reference_q.shape[0], 3):
            raise ValueError("reference_q and reference_object_positions have incompatible shapes")
        frames = self._forward_frames(reference_q.detach().cpu())
        object_positions = reference_object_positions.detach().cpu().to(reference_q)
        signs: list[float] = []
        for joint in self.finger_joints:
            body_name = joint.find("child").get("link")
            longitudinal = self._successor_origin(body_name)
            longitudinal /= np.linalg.norm(longitudinal)
            lateral = np.asarray(self._vector(joint.find("axis"), "xyz", "1 0 0"), dtype=np.float64)
            lateral -= longitudinal * np.dot(lateral, longitudinal)
            lateral /= np.linalg.norm(lateral)
            normal = np.cross(lateral, longitudinal)
            normal /= np.linalg.norm(normal)
            position, rotation = frames[body_name]
            signs.append(self._normal_sign_from_reference(
                normal, position, rotation, object_positions
            ))
        return signs

    @staticmethod
    def _cross_section_range(
        vertices: np.ndarray,
        longitudinal: np.ndarray,
        lateral: np.ndarray,
        station: float,
        length: float,
    ) -> tuple[float, float]:
        along = vertices @ longitudinal
        tolerance = max(0.04 * length, 1.0e-4)
        selected = np.abs(along - station) <= tolerance
        if selected.sum() < 8:
            nearest = np.argsort(np.abs(along - station))[:max(8, len(vertices) // 50)]
            section = vertices[nearest]
        else:
            section = vertices[selected]
        across = section @ lateral
        return float(across.min()), float(across.max())

    @staticmethod
    def _surface_intersection(
        mesh: trimesh.Trimesh, base: np.ndarray, inward: np.ndarray
    ) -> np.ndarray:
        extent = max(float(np.linalg.norm(mesh.extents)), 0.1)
        origin = base + inward * (2.0 * extent)
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origin[None], ray_directions=(-inward)[None], multiple_hits=True
        )
        if not len(locations):
            closest, distance, _ = trimesh.proximity.closest_point(mesh, base[None])
            if not np.isfinite(distance[0]):
                raise ValueError("cannot project WUJI surface landmark onto collision mesh")
            return closest[0]
        # The inward surface is the first intersection seen from inward.
        return locations[np.argmax(locations @ inward)]

    def _surface_triangle(self, joint: ET.Element, inward_sign: float) -> np.ndarray:
        body_name = joint.find("child").get("link")
        mesh = self._physics_collision_mesh(body_name)
        distal = self._successor_origin(body_name, mesh)
        length = float(np.linalg.norm(distal))
        if not np.isfinite(length) or length <= 1.0e-6:
            raise ValueError(f"invalid WUJI segment length for {body_name}: {length}")
        longitudinal = distal / length
        lateral = np.asarray(self._vector(joint.find("axis"), "xyz", "1 0 0"), dtype=np.float64)
        lateral -= longitudinal * np.dot(lateral, longitudinal)
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm <= 1.0e-6:
            raise ValueError(f"joint axis is parallel to segment for {joint.get('name')}")
        lateral /= lateral_norm
        inward = np.cross(lateral, longitudinal)
        inward = inward_sign * inward / np.linalg.norm(inward)
        center_station = 0.50 * length
        distal_station = 0.80 * length
        center_min, center_max = self._cross_section_range(
            mesh.vertices, longitudinal, lateral, center_station, length
        )
        distal_min, distal_max = self._cross_section_range(
            mesh.vertices, longitudinal, lateral, distal_station, length
        )
        center_lateral = 0.5 * (center_min + center_max)
        distal_lateral = 0.5 * (distal_min + distal_max)
        distal_width = distal_max - distal_min
        queries = (
            center_station * longitudinal + center_lateral * lateral,
            distal_station * longitudinal + (distal_lateral - 0.4 * distal_width) * lateral,
            distal_station * longitudinal + (distal_lateral + 0.4 * distal_width) * lateral,
        )
        points = np.stack([
            self._surface_intersection(mesh, query, inward) for query in queries
        ])
        area = 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
        if not np.isfinite(area) or area <= 1.0e-9:
            raise ValueError(f"degenerate WUJI contact triangle for {body_name}: area={area}")
        return points

    def _forward_frames(self, q: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        identity = q.new_zeros((*q.shape[:-1], 4))
        identity[..., 0] = 1
        frames = {self.root_name: (q.new_zeros((*q.shape[:-1], 3)), identity)}
        pending = self.joints.copy()
        while pending:
            remaining = []
            for joint in pending:
                parent = joint.find("parent").get("link")
                if parent not in frames:
                    remaining.append(joint)
                    continue
                pos, rot = frames[parent]
                origin = joint.find("origin")
                xyz = q.new_tensor(self._origin_xyz(joint)).expand_as(pos)
                rpy = self._vector(origin, "rpy", "0 0 0")
                origin_rot = identity.clone()
                # URDF fixed-axis RPY is Rz(yaw) Ry(pitch) Rx(roll).
                for axis_index in (2, 1, 0):
                    axis_rot = q.new_zeros(identity.shape)
                    axis_rot[..., 0] = math.cos(rpy[axis_index] / 2)
                    axis_rot[..., axis_index + 1] = math.sin(rpy[axis_index] / 2)
                    origin_rot = quat_mul(origin_rot, axis_rot)
                pos = pos + quat_apply(rot, xyz)
                rot = quat_mul(rot, origin_rot)
                kind = joint.get("type")
                if kind != "fixed":
                    angle = q[..., self.joint_names.index(joint.get("name"))]
                    axis = q.new_tensor(self._vector(joint.find("axis"), "xyz", "1 0 0"))
                    axis = axis / torch.linalg.vector_norm(axis)
                    if kind == "prismatic":
                        pos = pos + quat_apply(rot, angle[..., None] * axis)
                    elif kind in ("revolute", "continuous"):
                        motion = torch.cat((torch.cos(angle / 2)[..., None],
                                            torch.sin(angle / 2)[..., None] * axis), -1)
                        rot = quat_mul(rot, motion)
                    else:
                        raise ValueError(f"unsupported URDF joint type: {kind}")
                frames[joint.find("child").get("link")] = (pos, rot)
            if len(remaining) == len(pending):
                raise ValueError("disconnected or cyclic reference URDF")
            pending = remaining
        return frames

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return surface landmarks and palm pose in the URDF world frame."""
        frames = self._forward_frames(q)
        points = torch.stack([
            frames[name][0] + quat_apply(
                frames[name][1], self.offsets[index].to(q).expand_as(frames[name][0])
            )
            for index, name in enumerate(self.body_names)
        ], -2)
        return points, torch.cat(frames[self.palm_name], -1)
