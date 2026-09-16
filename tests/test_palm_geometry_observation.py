"""Palm observations preserve geometry, reference errors, and action semantics."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "palm_geometry_observation",
    ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/palm_geometry_observation.py",
)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def pose(position=(0., 0., 0.), quaternion=(1., 0., 0., 0.), count=3):
    return torch.tensor([*position, *quaternion], dtype=torch.float64).repeat(count, 1)


def test_frames_match_independent_rotation_math():
    rotation = Rotation.from_euler("xyz", [.3, -.6, .8])
    palm = pose((1., 2., 3.), np.roll(rotation.as_quat(), 1))
    points = torch.randn(3, geometry.LANDMARK_COUNT, 3, dtype=torch.float64)
    actual = geometry.palm_points(points, palm)
    expected = (points.numpy() - np.array([1., 2., 3.])) @ rotation.as_matrix()
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    target_rotation = Rotation.from_euler("xyz", [-.2, .5, -.7])
    target = pose((.2, -.5, .8), np.roll(target_rotation.as_quat(), 1))
    relative = geometry.palm_poses(target, palm)
    np.testing.assert_allclose(
        Rotation.from_quat(np.roll(relative[0, 3:].numpy(), -1)).as_matrix(),
        (rotation.inv() * target_rotation).as_matrix(), atol=1e-12,
    )


def test_observation_translation_and_yaw_invariance_and_target_error():
    torch.manual_seed(42)
    current = torch.randn(3, geometry.LANDMARK_COUNT, 3, dtype=torch.float64)
    reference = current + .01
    palm, goal_palm = pose(), pose((.1, .2, .3))
    obj, goal_obj = pose((.2, .3, .4)), pose((.3, .4, .5))
    phase = torch.tensor([0., .5, 1.], dtype=torch.float64)
    obs = geometry.build_observation(current, reference, palm, goal_palm, obj, goal_obj, phase)
    assert obs.shape == (3, geometry.OBSERVATION_DIM)
    assert torch.isfinite(obs).all()
    point_values = 3 * geometry.LANDMARK_COUNT
    torch.testing.assert_close(
        obs[:, point_values:2 * point_values] - obs[:, :point_values],
        torch.full((3, point_values), .01, dtype=torch.float64),
    )
    torch.testing.assert_close(obs[:, -1], phase)
    yaw = pose(quaternion=(math.cos(.4), 0., 0., math.sin(.4)))[:, 3:]
    translation = torch.tensor([4., -5., 6.], dtype=torch.float64)

    def transform_points(x):
        return geometry.quat_apply(yaw[:, None].expand(*x.shape[:-1], 4), x) + translation

    def transform_pose(x):
        q = yaw
        if x.ndim == 3:
            q = q[:, None].expand(*x.shape[:-1], 4)
        return torch.cat((geometry.quat_apply(q, x[..., :3]) + translation,
                          geometry.quat_mul(q, x[..., 3:])), -1)

    transformed = geometry.build_observation(
        transform_points(current), transform_points(reference), transform_pose(palm),
        transform_pose(goal_palm), transform_pose(obj), transform_pose(goal_obj),
        phase,
    )
    torch.testing.assert_close(obs, transformed, atol=1e-12, rtol=1e-12)


def test_quaternion_sign_is_canonical():
    target = pose((.1, .2, .3))
    negative = target.clone()
    negative[:, 3:] *= -1
    torch.testing.assert_close(geometry.palm_poses(target, pose()), geometry.palm_poses(negative, pose()))


@pytest.fixture
def simple_hand(tmp_path):
    import xml.etree.ElementTree as ET
    import trimesh

    root = ET.Element("robot", name="test_wuji")
    ET.SubElement(root, "link", name="world")
    ET.SubElement(root, "link", name="palm")
    joint = ET.SubElement(root, "joint", name="root_x", type="prismatic")
    ET.SubElement(joint, "parent", link="world")
    ET.SubElement(joint, "child", link="palm")
    ET.SubElement(joint, "axis", xyz="1 0 0")
    names = ["root_x"]
    for finger in range(5):
        parent = "palm"
        for index in range(4):
            child = f"finger{finger}_part{index}"
            link = ET.SubElement(root, "link", name=child)
            length = .05 if index == 3 else .1
            mesh_path = tmp_path / f"{child}.obj"
            mesh = trimesh.creation.box(extents=(length, .02, .04))
            mesh.apply_translation((.5 * length, 0., 0.))
            mesh.export(mesh_path)
            collision = ET.SubElement(link, "collision")
            geometry_node = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry_node, "mesh", filename=mesh_path.name)
            name = f"finger__{finger}_{index}"
            names.append(name)
            joint = ET.SubElement(root, "joint", name=name, type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "origin", xyz="0.1 0 0")
            ET.SubElement(joint, "axis", xyz="0 0 1")
            parent = child
        child = f"tip{finger}"
        ET.SubElement(root, "link", name=child)
        joint = ET.SubElement(root, "joint", name=f"finger__{finger}_tip_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz="0.05 0 0")
    path = tmp_path / "hand.urdf"
    ET.ElementTree(root).write(path)
    return path, names


def test_surface_landmarks_use_true_mesh_and_half_length_centers(simple_hand):
    path, names = simple_hand
    reference_q = torch.zeros(4, 21, dtype=torch.float64)
    object_positions = torch.tensor([[0., 1., 0.]] * 4, dtype=torch.float64)
    model = geometry.FixedWujiGeometry(
        path, names, "palm", reference_q, object_positions
    )
    assert model.offsets.shape == (60, 3)
    assert len(model.body_names) == 60
    assert len(model.landmark_names) == 60
    first = model.offsets[:3].double()
    torch.testing.assert_close(first[:, 0], torch.tensor([.05, .08, .08], dtype=torch.float64))
    torch.testing.assert_close(first[:, 1], torch.full((3,), .01, dtype=torch.float64))
    torch.testing.assert_close(first[:, 2], torch.tensor([0., -.016, .016], dtype=torch.float64))
    area = .5 * torch.linalg.vector_norm(torch.linalg.cross(first[1] - first[0], first[2] - first[0]))
    assert area > 0
    # The fourth moving link uses the fixed tip at x=.05 as its kinematic length.
    torch.testing.assert_close(model.offsets[9, 0], torch.tensor(.025))


def test_collision_surface_respects_requested_approximation(simple_hand):
    import trimesh

    path, names = simple_hand
    reference_q = torch.zeros(1, 21, dtype=torch.float64)
    object_positions = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    model = geometry.FixedWujiGeometry(
        path, names, "palm", reference_q, object_positions,
        collision_approximation="none",
    )

    # Replace one source mesh with a disconnected, non-convex shape after the
    # constructor has generated its landmarks. This makes the distinction
    # between the raw collision surface and its convex hull unambiguous.
    lower = trimesh.creation.box(extents=(.08, .02, .02))
    upper = lower.copy()
    lower.apply_translation((.05, 0., -.02))
    upper.apply_translation((.05, 0., .02))
    source = trimesh.util.concatenate((lower, upper))
    source.export(path.parent / "finger0_part0.obj")

    raw = model._physics_collision_mesh("finger0_part0")
    model.collision_approximation = "convexHull"
    hull = model._physics_collision_mesh("finger0_part0")
    assert hull.volume > raw.volume
    assert len(hull.faces) < len(raw.faces)

    model.collision_approximation = {"finger0_part0": "source_mesh"}
    mapped_raw = model._physics_collision_mesh("finger0_part0")
    np.testing.assert_allclose(mapped_raw.volume, raw.volume)

    model.collision_approximation = "convexDecomposition"
    with pytest.raises(ValueError, match="support only convexHull and none"):
        model._physics_collision_mesh("finger0_part0")


def test_kinematic_inward_direction_does_not_depend_on_object(simple_hand):
    path, names = simple_hand
    reference_q = torch.zeros(2, 21, dtype=torch.float64)
    positive_object = torch.tensor([[0., 1., 0.]] * 2, dtype=torch.float64)
    negative_object = -positive_object
    positive = geometry.FixedWujiGeometry(
        path, names, "palm", reference_q, positive_object,
        inward_direction_mode="kinematic_normal",
    )
    negative = geometry.FixedWujiGeometry(
        path, names, "palm", reference_q, negative_object,
        inward_direction_mode="kinematic_normal",
    )
    torch.testing.assert_close(positive.offsets, negative.offsets)
    torch.testing.assert_close(
        positive.offsets[0], torch.tensor([.05, .01, 0.], dtype=torch.float32)
    )

    reversed_normal = geometry.FixedWujiGeometry(
        path, names, "palm", reference_q, positive_object,
        inward_direction_mode="negative_kinematic_normal",
    )
    torch.testing.assert_close(
        reversed_normal.offsets[0], torch.tensor([.05, -.01, 0.], dtype=torch.float32)
    )

    with pytest.raises(ValueError, match="unknown inward_direction_mode"):
        geometry.FixedWujiGeometry(
            path, names, "palm", reference_q, positive_object,
            inward_direction_mode="not_a_mode",
        )


def test_candidate_joint_and_collision_overlays_change_offsets_and_fk(simple_hand):
    path, names = simple_hand
    q = torch.zeros(2, 21, dtype=torch.float64)
    object_positions = torch.tensor([[0., 1., 0.]] * 2, dtype=torch.float64)
    transform = np.eye(4)
    transform[0, 0] = 1.2
    candidate = geometry.FixedWujiGeometry(
        path,
        names,
        "palm",
        q,
        object_positions,
        inward_direction_mode="kinematic_normal",
        joint_origin_overrides={"finger__0_1": [.12, 0., 0.]},
        collision_mesh_transforms={"finger0_part0": transform.tolist()},
    )
    # Segment stationing follows the candidate joint origin, while the ray
    # intersection uses the candidate's transformed collision surface.
    torch.testing.assert_close(
        candidate.offsets[0, 0], torch.tensor(.06, dtype=torch.float32)
    )
    points, _ = candidate.forward(q)
    torch.testing.assert_close(
        points[:, 3, 0], torch.full((2,), .27, dtype=torch.float64)
    )


def test_batched_surface_fk_uses_joint_axes_and_named_order(simple_hand):
    path, names = simple_hand
    reference_q = torch.zeros(2, 21, dtype=torch.float64)
    object_positions = torch.tensor([[0., 1., 0.]] * 2, dtype=torch.float64)
    model = geometry.FixedWujiGeometry(path, names, "palm", reference_q, object_positions)
    q = torch.zeros(2, 21, dtype=torch.float64)
    q[1, 0] = 2
    q[1, 1] = math.pi / 2
    points, palm = model.forward(q)
    assert points.shape == (2, 60, 3)
    torch.testing.assert_close(points[0, 0], torch.tensor([.15, .01, 0.], dtype=q.dtype))
    torch.testing.assert_close(points[1, 0], torch.tensor([2.09, .05, 0.], dtype=q.dtype))
    torch.testing.assert_close(palm[:, 0], q[:, 0])
    reversed_model = geometry.FixedWujiGeometry(
        path, names[::-1], "palm", reference_q.flip(-1), object_positions
    )
    reversed_points, reversed_palm = reversed_model.forward(q.flip(-1))
    torch.testing.assert_close(points, reversed_points)
    torch.testing.assert_close(palm, reversed_palm)


def test_original_reference_produces_sixty_non_degenerate_surface_landmarks():
    prepared = ROOT / "artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2"
    if not (prepared / "reference.npz").is_file():
        pytest.skip("original WUJI prepared reference is not available")
    with np.load(prepared / "reference.npz") as reference:
        hand_q = torch.from_numpy(reference["hand_q"])
        model = geometry.FixedWujiGeometry(
            prepared / "hand_rl.urdf",
            reference["joint_names"].tolist(),
            str(reference["palm_body_name"]),
            hand_q,
            torch.from_numpy(reference["object_pose_wxyz"][:, :3]),
            collision_approximation="convexHull",
        )
        points, _ = model.forward(hand_q)
        assert points.shape == (hand_q.shape[0], 60, 3)
        triangles = model.offsets.reshape(20, 3, 3)
        areas = .5 * torch.linalg.vector_norm(
            torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            dim=-1,
        )
        assert torch.isfinite(points).all()
        assert torch.all(areas > 1.0e-9)
        inward_signs = model._inward_signs(
            hand_q, torch.from_numpy(reference["object_pose_wxyz"][:, :3])
        )
        # The USD converter authors these links as convexHull colliders. Every
        # landmark must therefore lie on the boundary of that same convex hull,
        # rather than on a concavity of the source triangle mesh. All three
        # points must also remain on the object-facing half of that hull.
        for link_index, (joint, body_name, inward_sign) in enumerate(zip(
            model.finger_joints,
            model.segment_body_names,
            inward_signs,
            strict=True,
        )):
            hull = model._physics_collision_mesh(body_name)
            equations = ConvexHull(hull.vertices).equations
            triangle = triangles[link_index].numpy()
            signed_distances = equations[:, :3] @ triangle.T + equations[:, 3:4]
            np.testing.assert_allclose(
                signed_distances.max(axis=0), np.zeros(3), atol=1.0e-7
            )
            distal = model._successor_origin(body_name)
            longitudinal = distal / np.linalg.norm(distal)
            lateral = np.asarray(
                model._vector(joint.find("axis"), "xyz", "1 0 0"), dtype=np.float64
            )
            lateral -= longitudinal * np.dot(lateral, longitudinal)
            lateral /= np.linalg.norm(lateral)
            inward = inward_sign * np.cross(lateral, longitudinal)
            inward /= np.linalg.norm(inward)
            hull_projection = hull.vertices @ inward
            point_fraction = (
                triangle @ inward - hull_projection.min()
            ) / np.ptp(hull_projection)
            assert np.all(point_fraction >= 0.5)
