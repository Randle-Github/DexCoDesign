#!/usr/bin/env python3
"""GPU-batched warm-start retargeting for thousands of WUJI morphologies.

This is the high-throughput proposal stage of morphology search.  It reuses
the exact source-hand retargeted trajectory as the initial value, evaluates a
same-topology parametric WUJI kinematic model in Torch, and applies the same
damped-least-squares update used by Isaac Lab's Differential IK controller.
Visual/collision meshes are deliberately absent here; only exact top-k
candidates are compiled and physically rescored by MuJoCo/MJX.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_URDF = (
    REPO_ROOT
    / "assets"
    / "robot_hands"
    / "direct_motor"
    / "wuji_hand_2"
    / "left"
    / "hand.urdf"
)
DEFAULT_CAPTURE = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "subject_7"
    / "20231022_192832"
    / "isaaclab_reference.npz"
)
OBJECT_MESH = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "models"
    / "G04_1"
    / "cleaned_mesh_2000.obj"
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_LINKS = {
    "thumb": "l_thumb_tip",
    "index": "l_index_finger_tip",
    "middle": "l_middle_finger_tip",
    "ring": "l_ring_finger_tip",
    "pinky": "l_pinky_tip",
}
VECTOR_NAMES = (
    "palm_expansion",
    "palm_scale_x",
    "palm_scale_z",
    "palm_yaw",
    *(f"{finger}_length" for finger in FINGERS),
    *(f"{finger}_radius" for finger in FINGERS),
)
LOWER_BOUNDS = np.asarray(
    [0.0, 0.90, 0.90, -0.12, *([0.82] * 5), *([0.85] * 5)],
    dtype=np.float32,
)
UPPER_BOUNDS = np.asarray(
    [0.35, 1.12, 1.12, 0.12, *([1.20] * 5), *([1.15] * 5)],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    xyz: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    movable: bool


def rpy_matrix(text: str | None) -> np.ndarray:
    roll, pitch, yaw = np.fromstring(text or "0 0 0", sep=" ")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def parse_urdf(path: Path) -> tuple[dict[str, Joint], dict[str, Joint]]:
    root = ET.parse(path).getroot()
    by_child: dict[str, Joint] = {}
    movable: dict[str, Joint] = {}
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            continue
        origin = element.find("origin")
        axis_element = element.find("axis")
        limit = element.find("limit")
        kind = element.get("type", "fixed")
        name = str(element.get("name"))
        joint = Joint(
            name=name,
            parent=str(parent.get("link")),
            child=str(child.get("link")),
            xyz=np.fromstring(
                "0 0 0" if origin is None else origin.get("xyz", "0 0 0"),
                sep=" ",
                dtype=np.float64,
            ),
            rotation=rpy_matrix(None if origin is None else origin.get("rpy")),
            axis=np.fromstring(
                "1 0 0" if axis_element is None else axis_element.get("xyz", "1 0 0"),
                sep=" ",
                dtype=np.float64,
            ),
            lower=-math.pi if limit is None else float(limit.get("lower", -math.pi)),
            upper=math.pi if limit is None else float(limit.get("upper", math.pi)),
            movable=kind != "fixed",
        )
        by_child[joint.child] = joint
        if joint.movable:
            movable[name] = joint
    return by_child, movable


def chain_to_link(by_child: dict[str, Joint], link: str, root: str = "l_wrist") -> list[Joint]:
    chain = []
    current = link
    while current != root:
        if current not in by_child:
            raise ValueError(f"{link}: cannot trace kinematic chain to {root}")
        joint = by_child[current]
        chain.append(joint)
        current = joint.parent
    return list(reversed(chain))


def axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(skew)
    sine = torch.sin(angle)[..., None, None]
    cosine = torch.cos(angle)[..., None, None]
    return eye + sine * skew + (1.0 - cosine) * (skew @ skew)


def yaw_matrix(yaw: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1
    ).reshape(*yaw.shape, 3, 3)


def quat_xyzw_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    x, y, z, w = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def direction_error(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-8)
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1.0e-8)
    cross = torch.linalg.cross(current, target, dim=-1)
    norm = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
    dot = (current * target).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.atan2(norm, dot)
    return angle * cross / norm.clamp_min(1.0e-8)


class WujiBatchKinematics:
    def __init__(self, joint_names: list[str], device: torch.device):
        by_child, movable = parse_urdf(SOURCE_URDF)
        self.device = device
        self.dtype = torch.float32
        self.joint_names = joint_names
        self.name_to_q = {name: index for index, name in enumerate(joint_names)}
        self.chains = {
            finger: chain_to_link(by_child, TIP_LINKS[finger]) for finger in FINGERS
        }
        missing = sorted(set(movable) - set(joint_names))
        if missing:
            raise ValueError(f"warm-start trajectory misses joints: {missing}")
        self.lower = torch.tensor(
            [movable[name].lower for name in joint_names], device=device, dtype=self.dtype
        )
        self.upper = torch.tensor(
            [movable[name].upper for name in joint_names], device=device, dtype=self.dtype
        )

    def forward_finger(
        self,
        finger_index: int,
        vector: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tip pose, Jacobians, and distal-link start point."""
        chain = self.chains[FINGERS[finger_index]]
        batch_shape = q.shape[:-1]
        rotation = torch.eye(3, device=self.device, dtype=self.dtype).expand(
            *batch_shape, 3, 3
        ).clone()
        position = torch.zeros(*batch_shape, 3, device=self.device, dtype=self.dtype)
        joint_positions = []
        joint_axes = []
        joint_q_indices = []
        length = vector[..., 4 + finger_index]
        palm_scale = torch.stack(
            (vector[..., 1], torch.ones_like(vector[..., 1]), vector[..., 2]), dim=-1
        )
        palm_yaw = yaw_matrix(vector[..., 3])
        expansion = 1.0 + vector[..., 0]
        movable_rank = 0
        for chain_rank, joint in enumerate(chain):
            xyz = torch.tensor(joint.xyz, device=self.device, dtype=self.dtype)
            xyz = xyz.expand(*batch_shape, 3)
            origin_rotation = torch.tensor(
                joint.rotation, device=self.device, dtype=self.dtype
            ).expand(*batch_shape, 3, 3)
            if chain_rank == 0:
                xyz = xyz * palm_scale
                xyz = torch.stack(
                    (xyz[..., 0] * expansion, xyz[..., 1], xyz[..., 2] * expansion),
                    dim=-1,
                )
                xyz = (palm_yaw @ xyz.unsqueeze(-1)).squeeze(-1)
                origin_rotation = palm_yaw @ origin_rotation
            else:
                xyz = xyz * length.unsqueeze(-1)
            position = position + (rotation @ xyz.unsqueeze(-1)).squeeze(-1)
            rotation = rotation @ origin_rotation
            if joint.movable:
                q_index = self.name_to_q[joint.name]
                local_axis = torch.tensor(
                    joint.axis, device=self.device, dtype=self.dtype
                ).expand(*batch_shape, 3)
                axis_world = (rotation @ local_axis.unsqueeze(-1)).squeeze(-1)
                joint_positions.append(position.clone())
                joint_axes.append(axis_world)
                joint_q_indices.append(q_index)
                rotation = rotation @ axis_angle(local_axis, q[..., q_index])
                movable_rank += 1
        if movable_rank != 4:
            raise ValueError(f"{FINGERS[finger_index]} expected 4 movable joints")
        # The final fixed joint origin places the physical fingertip. Its local
        # +Z direction is the same semantic direction used by retargeting.
        direction = rotation[..., :, 2]
        joint_positions_tensor = torch.stack(joint_positions, dim=-2)
        joint_axes_tensor = torch.stack(joint_axes, dim=-2)
        lever = position.unsqueeze(-2) - joint_positions_tensor
        jacobian_position = torch.linalg.cross(joint_axes_tensor, lever, dim=-1).mT
        jacobian_rotation = joint_axes_tensor.mT
        distal_start = joint_positions_tensor[..., -1, :]
        return position, direction, jacobian_position, jacobian_rotation, distal_start

    def solve(
        self,
        vectors: torch.Tensor,
        seed_q: torch.Tensor,
        iterations: int,
        candidate_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        candidate_count = vectors.shape[0]
        frame_count = seed_q.shape[0]
        identity_vector = torch.tensor(
            [0.0, 1.0, 1.0, 0.0, *([1.0] * 10)],
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, 1, -1)
        source_q = seed_q.unsqueeze(0)
        target_points = []
        target_directions = []
        for finger_index in range(5):
            point, direction, _, _, _ = self.forward_finger(
                finger_index, identity_vector, source_q
            )
            target_points.append(point[0])
            target_directions.append(direction[0])
        target_points_tensor = torch.stack(target_points, dim=1)
        target_directions_tensor = torch.stack(target_directions, dim=1)

        all_q = torch.empty(
            candidate_count, frame_count, len(self.joint_names),
            device="cpu", dtype=self.dtype, pin_memory=self.device.type == "cuda",
        )
        all_wrist_delta = torch.empty(
            candidate_count, frame_count, 3,
            device="cpu", dtype=self.dtype, pin_memory=self.device.type == "cuda",
        )
        start = time.perf_counter()
        orientation_length = 0.15
        temporal = 0.015
        damping = 0.025
        variable_count = 3 + len(self.joint_names)
        for begin in range(0, candidate_count, candidate_chunk):
            end = min(begin + candidate_chunk, candidate_count)
            count = end - begin
            vector = vectors[begin:end, None, :].expand(count, frame_count, -1)
            q = seed_q[None].expand(count, -1, -1).clone()
            wrist_delta = torch.zeros(count, frame_count, 3, device=self.device)
            for _ in range(iterations):
                rows = []
                errors = []
                for finger_index in range(5):
                    point, direction, jac_p, jac_r, _ = self.forward_finger(
                        finger_index, vector, q
                    )
                    row_position = torch.zeros(
                        count, frame_count, 3, variable_count,
                        device=self.device, dtype=self.dtype,
                    )
                    row_position[..., :3] = torch.eye(
                        3, device=self.device, dtype=self.dtype
                    )
                    movable_names = [
                        joint.name
                        for joint in self.chains[FINGERS[finger_index]]
                        if joint.movable
                    ]
                    q_indices = [self.name_to_q[name] for name in movable_names]
                    row_position[..., 3 + torch.tensor(q_indices, device=self.device)] = (
                        jac_p
                    )
                    rows.append(row_position)
                    errors.append(
                        target_points_tensor[None, :, finger_index]
                        - (point + wrist_delta)
                    )

                    target_direction = target_directions_tensor[None, :, finger_index]
                    projector = torch.eye(
                        3, device=self.device, dtype=self.dtype
                    ) - target_direction.unsqueeze(-1) * target_direction.unsqueeze(-2)
                    row_direction = torch.zeros_like(row_position)
                    row_direction[..., 3 + torch.tensor(q_indices, device=self.device)] = (
                        orientation_length * (projector @ jac_r)
                    )
                    rows.append(row_direction)
                    errors.append(
                        orientation_length
                        * direction_error(target_direction, direction)
                    )
                regularization = torch.zeros(
                    count, frame_count, len(self.joint_names), variable_count,
                    device=self.device, dtype=self.dtype,
                )
                q_diag = torch.arange(len(self.joint_names), device=self.device)
                regularization[..., q_diag, 3 + q_diag] = temporal
                rows.append(regularization)
                errors.append(temporal * (seed_q[None] - q))
                jacobian = torch.cat(rows, dim=-2)
                error = torch.cat(errors, dim=-1)
                jacobian_t = jacobian.mT
                normal = jacobian_t @ jacobian
                normal.diagonal(dim1=-2, dim2=-1).add_(damping * damping)
                rhs = (jacobian_t @ error.unsqueeze(-1)).squeeze(-1)
                delta = torch.linalg.solve(normal, rhs)
                max_abs = delta.abs().amax(dim=-1, keepdim=True).clamp_min(0.12)
                delta = delta * (0.12 / max_abs)
                wrist_delta += delta[..., :3]
                q += delta[..., 3:]
                q = torch.maximum(
                    torch.minimum(q, seed_q[None] + 0.20),
                    seed_q[None] - 0.20,
                )
                q = torch.maximum(torch.minimum(q, self.upper), self.lower)
            all_q[begin:end].copy_(q.detach(), non_blocking=False)
            all_wrist_delta[begin:end].copy_(wrist_delta.detach(), non_blocking=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        return all_q, all_wrist_delta, {
            "candidate_count": candidate_count,
            "frame_count": frame_count,
            "iterations": iterations,
            "seconds": elapsed,
            "candidate_trajectories_per_second": candidate_count / elapsed,
            "frame_solves_per_second": candidate_count * frame_count / elapsed,
        }

    def score_contact_proxy(
        self,
        vectors: torch.Tensor,
        q_cpu: torch.Tensor,
        wrist_delta_cpu: torch.Tensor,
        wrist_position: torch.Tensor,
        wrist_quaternion_xyzw: torch.Tensor,
        object_pose_wxyz: torch.Tensor,
        candidate_chunk: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """GPU ranker using the reference object surface and distal radii.

        This is intentionally only a proposal score. Exact C-error + binary
        pinch reward is always recomputed for top-k in the physical backend.
        """
        vertices = []
        with OBJECT_MESH.open(encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if line.startswith("v "):
                    vertices.append([float(value) for value in line.split()[1:4]])
        mesh_vertices = np.asarray(vertices, dtype=np.float32)
        center_xz = mesh_vertices[:, (0, 2)].mean(axis=0)
        local_y = mesh_vertices[:, 1]
        y_min, y_max = float(local_y.min()), float(local_y.max())
        bin_count = 96
        bin_index = np.clip(
            ((local_y - y_min) / (y_max - y_min) * (bin_count - 1)).astype(int),
            0,
            bin_count - 1,
        )
        radial = np.linalg.norm(mesh_vertices[:, (0, 2)] - center_xz, axis=1)
        profile = np.zeros(bin_count, dtype=np.float32)
        for index in range(bin_count):
            values = radial[bin_index == index]
            profile[index] = float(values.max()) if len(values) else np.nan
        valid = np.flatnonzero(np.isfinite(profile))
        profile = np.interp(np.arange(bin_count), valid, profile[valid]).astype(np.float32)
        profile_t = torch.from_numpy(profile).to(self.device)
        center_t = torch.tensor(center_xz, device=self.device, dtype=self.dtype)
        wrist_rotation = quat_xyzw_matrix(wrist_quaternion_xyzw)
        object_quaternion_xyzw = torch.cat(
            (object_pose_wxyz[:, 4:7], object_pose_wxyz[:, 3:4]), dim=-1
        )
        object_rotation = quat_xyzw_matrix(object_quaternion_xyzw)
        object_position = object_pose_wxyz[:, :3]
        scores = torch.empty(vectors.shape[0], dtype=self.dtype, device="cpu")
        hard_pinch_frames = torch.empty_like(scores)
        minimum_distance = torch.empty(
            vectors.shape[0], 5, dtype=self.dtype, device="cpu"
        )
        start = time.perf_counter()
        for begin in range(0, vectors.shape[0], candidate_chunk):
            end = min(begin + candidate_chunk, vectors.shape[0])
            count = end - begin
            vector = vectors[begin:end, None, :].expand(count, q_cpu.shape[1], -1)
            q = q_cpu[begin:end].to(self.device, non_blocking=True)
            wrist_delta = wrist_delta_cpu[begin:end].to(self.device, non_blocking=True)
            local_segments = []
            for finger_index in range(5):
                point, _, _, _, distal_start = self.forward_finger(
                    finger_index, vector, q
                )
                alpha = torch.linspace(
                    0.0, 1.0, 7, device=self.device, dtype=self.dtype
                )
                segment = (
                    distal_start.unsqueeze(-2) * (1.0 - alpha[None, None, :, None])
                    + point.unsqueeze(-2) * alpha[None, None, :, None]
                    + wrist_delta.unsqueeze(-2)
                )
                local_segments.append(segment)
            local_segment = torch.stack(local_segments, dim=2)
            world_segment = wrist_position[None, :, None, None] + (
                wrist_rotation[None, :, None, None]
                @ local_segment.unsqueeze(-1)
            ).squeeze(-1)
            object_local = (
                object_rotation.mT[None, :, None, None]
                @ (
                    world_segment - object_position[None, :, None, None]
                ).unsqueeze(-1)
            ).squeeze(-1)
            y = object_local[..., 1]
            normalized_y = ((y - y_min) / (y_max - y_min) * (bin_count - 1)).clamp(
                0.0, bin_count - 1.0001
            )
            low = normalized_y.floor().long()
            fraction = normalized_y - low
            radius_at_y = profile_t[low] * (1.0 - fraction) + profile_t[low + 1] * fraction
            point_radius = torch.linalg.vector_norm(
                object_local[..., (0, 2)] - center_t, dim=-1
            )
            side_distance = (point_radius - radius_at_y).abs()
            cap_distance = torch.where(
                y < y_min,
                torch.sqrt(side_distance.square() + (y_min - y).square()),
                torch.where(
                    y > y_max,
                    torch.sqrt(side_distance.square() + (y - y_max).square()),
                    side_distance,
                ),
            )
            cap_distance = cap_distance.amin(dim=-1)
            tip_radius = 0.009 * vector[..., 9:14]
            # Smooth proposal score avoids the zero-gradient/zero-ranking
            # problem of exact binary contact. Exact top-k uses binary reward.
            soft_contact = torch.sigmoid((tip_radius - cap_distance) / 0.0025)
            soft_pinch = soft_contact[..., 0] * soft_contact[..., 1:].amax(dim=-1)
            hard_contact = cap_distance <= tip_radius
            hard_pinch = hard_contact[..., 0] & hard_contact[..., 1:].any(dim=-1)
            scores[begin:end] = (2.0 * soft_pinch.sum(dim=-1)).detach().cpu()
            hard_pinch_frames[begin:end] = hard_pinch.sum(dim=-1).detach().cpu()
            minimum_distance[begin:end] = cap_distance.amin(dim=1).detach().cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        return torch.cat(
            (scores[:, None], hard_pinch_frames[:, None], minimum_distance), dim=-1
        ), {
            "seconds": elapsed,
            "candidates_per_second": vectors.shape[0] / elapsed,
        }


def joint_names_from_seed(seed: np.lib.npyio.NpzFile) -> list[str]:
    if "joint_names" in seed:
        return [str(value) for value in seed["joint_names"].tolist()]
    # Existing retarget caches predate explicit names. Recover them from the
    # exact MuJoCo qpos addresses once; subsequent GPU passes store names.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import retarget_all_hands as retarget

    retarget.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    scene = retarget.make_scene("wuji_hand_2")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    by_qpos = {
        int(model.jnt_qposadr[joint_id]): str(model.joint(joint_id).name)
        for joint_id in range(model.njnt)
    }
    return [by_qpos[int(qpos_id)] for qpos_id in seed["qpos_ids"]]


def sample_vectors(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normalized = rng.uniform(0.0, 1.0, size=(count, len(VECTOR_NAMES))).astype(np.float32)
    vectors = LOWER_BOUNDS + normalized * (UPPER_BOUNDS - LOWER_BOUNDS)
    vectors[0] = np.asarray([0.0, 1.0, 1.0, 0.0, *([1.0] * 10)], dtype=np.float32)
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    if args.candidates < 1:
        parser.error("--candidates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    with np.load(args.seed_trajectory) as seed:
        joint_names = joint_names_from_seed(seed)
        seed_q_np = seed["qpos"].astype(np.float32)
        wrist_position = seed["wrist_position"].astype(np.float32)
        wrist_quaternion = seed["wrist_quaternion_xyzw"].astype(np.float32)
        frame_ids = seed["frame_ids"].astype(np.int64)
    vectors_np = sample_vectors(args.candidates, args.seed)
    vectors = torch.from_numpy(vectors_np).to(device)
    seed_q = torch.from_numpy(seed_q_np).to(device)
    kinematics = WujiBatchKinematics(joint_names, device)
    q, wrist_delta, benchmark = kinematics.solve(
        vectors, seed_q, args.iterations, args.candidate_chunk
    )
    with np.load(DEFAULT_CAPTURE) as capture:
        object_pose = capture["object_pose_wxyz"][: len(seed_q_np)].astype(np.float32)
    proxy, proxy_benchmark = kinematics.score_contact_proxy(
        vectors,
        q,
        wrist_delta,
        torch.from_numpy(wrist_position).to(device),
        torch.from_numpy(wrist_quaternion).to(device),
        torch.from_numpy(object_pose).to(device),
        args.candidate_chunk,
    )
    top_count = min(max(args.top_k, 1), args.candidates)
    top_indices = torch.topk(proxy[:, 0], k=top_count).indices
    payload = {
        "schema_version": 1,
        "backend": "torch_gpu_batched_dls",
        "isaaclab_algorithm": "damped_least_squares",
        "warm_start": str(args.seed_trajectory.resolve()),
        "vector_names": list(VECTOR_NAMES),
        "benchmark": benchmark,
        "contact_proxy_benchmark": proxy_benchmark,
        "top_k": top_count,
        "maximum_soft_pinch_return": float(proxy[:, 0].max()),
        "maximum_hard_pinch_frames": int(proxy[:, 1].max()),
        "minimum_distal_surface_distance_m": proxy[:, 2:].amin(dim=0).tolist(),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "vectors": vectors_np,
        "joint_names": np.asarray(joint_names),
        "frame_ids": frame_ids,
        "wrist_position": wrist_position,
        "wrist_quaternion_xyzw": wrist_quaternion,
        "metadata_json": np.asarray(json.dumps(payload)),
        "proxy_soft_pinch_return": proxy[:, 0].numpy(),
        "proxy_hard_pinch_frames": proxy[:, 1].numpy(),
        "proxy_minimum_surface_distance_m": proxy[:, 2:].numpy(),
        "top_indices": top_indices.numpy(),
        "top_vectors": vectors_np[top_indices.numpy()],
        "top_qpos": q[top_indices].numpy(),
        "top_wrist_delta_local": wrist_delta[top_indices].numpy(),
    }
    if args.save_trajectories:
        arrays["qpos"] = q.numpy()
        arrays["wrist_delta_local"] = wrist_delta.numpy()
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
