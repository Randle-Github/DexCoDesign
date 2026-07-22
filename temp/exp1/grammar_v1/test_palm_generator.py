#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from palm_generator import (  # noqa: E402
    AttachmentPatch,
    PalmGeometryParams,
    deform_template_palm,
    deform_template_to_house_palm,
    generate_house_hull_palm,
    generate_palm_mesh,
    infer_palm_params,
    patch_corners_2d,
    patches_from_hand_ir,
    transform_from_rotation_translation,
)
from generate_hands import apply_global_palm_layout, instantiate_protected_platform  # noqa: E402


def frame(x: float, z: float, yaw: float = 0.0, y: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return transform_from_rotation_translation(rotation, [x, y, z])


def layout(count: int) -> tuple[list[AttachmentPatch], AttachmentPatch]:
    if count == 3:
        specs = [(-0.28, 0.86, -0.24), (0.0, 1.02, 0.0), (0.28, 0.86, 0.24)]
    elif count == 4:
        specs = [(-0.32, 0.88, -0.18), (-0.11, 1.05, -0.04), (0.12, 1.0, 0.05), (0.34, 0.82, 0.28)]
    elif count == 5:
        specs = [
            (-0.39, 0.82, -0.12), (-0.14, 1.03, -0.03),
            (0.10, 1.08, 0.02), (0.33, 0.96, 0.10),
            (0.47, 0.38, 1.02),  # opposed thumb
        ]
    else:
        raise ValueError(count)
    patches = [
        AttachmentPatch(f"finger_{index}", frame(x, z, yaw), 0.18, 0.16, 0.07)
        for index, (x, z, yaw) in enumerate(specs)
    ]
    wrist = AttachmentPatch("wrist", frame(0.0, 0.0), 0.46, 0.18, 0.08)
    return patches, wrist


def params(**changes) -> PalmGeometryParams:
    values = dict(
        thickness=0.24,
        boundary_margin=0.07,
        edge_rounding_radius=0.035,
        wrist_width=0.46,
        transverse_arch=0.025,
        longitudinal_arch=0.018,
        central_cup=0.022,
        deformation_resolution=0.30,
    )
    values.update(changes)
    return PalmGeometryParams(**values)


class PalmGeneratorTests(unittest.TestCase):
    def test_source_fixed_layout_preserves_order_and_attachment_pose(self) -> None:
        anchor = np.asarray([0.21, -0.04, 0.73])
        frame_value = frame(0.0, 0.0)[:3, :3]
        slots = {"thumb": {"anchor": anchor.copy(), "frame": frame_value.copy()}}
        result = apply_global_palm_layout(
            np.random.default_rng(1), slots, {}, {}, np.eye(3), "source_fixed"
        )
        self.assertTrue(np.array_equal(slots["thumb"]["anchor"], anchor))
        self.assertTrue(np.array_equal(slots["thumb"]["frame"], frame_value))
        self.assertTrue(result["source_order_locked"])
        self.assertTrue(result["attachment_poses_locked"])
        self.assertEqual(result["edits"], [])

    def test_protected_platform_is_copied_with_identity_geometry(self) -> None:
        source = {
            "hand_id": "fake_tendon",
            "parts": [
                {"id": 0, "parent": None},
                {"id": 1, "parent": 0, "role": "base", "joint_type": "hinge",
                 "joint_axis": [1, 0, 0], "joint_range": [-1, 1], "joint_name": "base_pitch",
                 "relative_pos": [0.0, 0.0, -0.2], "mesh": {"file": "base.obj"}},
                {"id": 2, "parent": 1, "role": "other", "joint_type": "fixed",
                 "joint_axis": [0, 0, 0], "joint_range": [0, 0], "joint_name": "housing",
                 "relative_pos": [0.0, 0.0, -0.1], "mesh": {"file": "housing.obj"}},
                {"id": 3, "parent": 0, "role": "index", "joint_type": "hinge",
                 "joint_axis": [1, 0, 0], "joint_range": [0, 1], "joint_name": "index",
                 "relative_pos": [0.2, 0.0, 0.5], "mesh": {"file": "finger.obj"}},
            ],
        }
        output = [{"id": 0}]
        created = instantiate_protected_platform(
            output, source, [{"source_part_ids": [3]}]
        )
        self.assertEqual(created, [1, 2])
        for node in output[1:]:
            self.assertTrue(np.array_equal(np.asarray(node["mesh_linear"]), np.eye(3)))
            self.assertTrue(node["geometry_locked"])
        self.assertEqual(output[2]["parent"], 1)

    def test_global_symmetric_layout_preserves_thickness_and_motor_clearance(self) -> None:
        roles = ("thumb", "index", "middle", "ring", "pinky")
        slots = {}
        selected = {}
        candidates = {}
        for index, role in enumerate(roles):
            slots[role] = {
                "anchor": np.asarray([0.30 - 0.15 * index, 0.04 + 0.01 * index, 0.82]),
                "frame": np.eye(3),
            }
            source = f"fake_{role}"
            selected[role] = {"source_hand_id": source, "root_part_id": 0}
            candidates[f"{source}:part:0"] = {
                "source_mesh": {"size": [0.18, 0.16, 0.24]}
            }
        result = apply_global_palm_layout(
            np.random.default_rng(4),
            slots,
            selected,
            candidates,
            np.eye(3),
            "symmetric",
        )
        slots_out = np.sort(np.asarray(result["mount_slots"], dtype=int))
        gaps = np.diff(np.r_[slots_out, slots_out[0] + result["slot_count"]])
        self.assertLessEqual(int(gaps.max() - gaps.min()), 1)
        self.assertGreaterEqual(result["minimum_arc_clearance"], -1.0e-12)
        self.assertTrue(result["cyclic_order_preserved"])
        self.assertEqual(result["source_cyclic_roles"], result["target_cyclic_roles"])
        self.assertEqual(result["thickness_scale_locked"], 1.0)
        self.assertTrue(result["all_slots_outside_root_mount_exclusion"])
        for index, role in enumerate(roles):
            self.assertEqual(slots[role]["anchor"][1], 0.04 + 0.01 * index)
            self.assertAlmostEqual(slots[role]["frame"][1, 2], 0.0)
            self.assertIn("reference_anchor", slots[role])

    def test_house_hull_uses_graph_motor_footprints_without_template_warp(self) -> None:
        patches, wrist = layout(5)
        slots = []
        for index, patch in enumerate(patches):
            slots.append({
                "slot_id": index,
                "role": "thumb" if index == 4 else f"finger_{index}",
                "attachment_translation": patch.transform[:3, 3].tolist(),
                "attachment_rotation": patch.transform[:3, :3].tolist(),
            })
        template = trimesh.creation.box(extents=[0.82, 0.24, 1.0])
        hand = {
            "palm_layout": {"mode": "symmetric", "center_local_xz": [0.0, 0.4]},
            "finger_slots": slots,
            "parts": [{"world_pos": [0.0, 0.0, 0.0]}],
        }
        result = generate_house_hull_palm(
            hand, patches, wrist, template, np.eye(3)
        )
        self.assertEqual(result.metadata["mode"], "house_hull")
        self.assertTrue(result.metadata["all_motor_centers_covered"])
        self.assertAlmostEqual(result.metadata["thickness_error"], 0.0, places=12)
        self.assertTrue(result.visual_mesh.is_watertight)
        self.assertLess(len(result.visual_mesh.faces), 100)

    def test_joint_frame_invariance_under_all_geometry_edits(self) -> None:
        patches, wrist = layout(5)
        expected = {patch.name: patch.transform.copy() for patch in [*patches, wrist]}
        variants = [
            params(thickness=0.18),
            params(thickness=0.32, boundary_margin=0.11),
            params(transverse_arch=0.05, longitudinal_arch=0.0, central_cup=0.0),
            params(transverse_arch=0.0, longitudinal_arch=0.05, central_cup=0.0),
            params(transverse_arch=0.0, longitudinal_arch=0.0, central_cup=0.05),
        ]
        for variant in variants:
            result = generate_palm_mesh(patches, wrist, variant, mode="parametric_2_5d")
            for name, transform in expected.items():
                self.assertTrue(np.array_equal(result.attachment_frames[name], transform), name)

    def test_attachment_coverage_and_coordinate_consistency(self) -> None:
        patches, wrist = layout(5)
        configuration = params()
        result = generate_palm_mesh(patches, wrist, configuration, mode="attachment_hull")
        self.assertTrue(all(result.metadata["attachment_coverage"].values()))
        outline = Polygon(result.metadata["outline_coordinates"])
        normal_axis = result.metadata["thickness_axis"]
        normal_bounds = result.visual_mesh.bounds[:, normal_axis]
        for patch in [*patches, wrist]:
            origin = result.attachment_frames[patch.name][:3, 3]
            center_2d = origin[list(configuration.plane_axes)]
            self.assertTrue(outline.buffer(1.0e-9).covers(Point(center_2d)))
            self.assertGreaterEqual(origin[normal_axis], normal_bounds[0] - 1.0e-9)
            self.assertLessEqual(origin[normal_axis], normal_bounds[1] + 1.0e-9)
            # The graph origin is exactly the center of the projected patch.
            corners = patch_corners_2d(patch, configuration)
            self.assertTrue(np.allclose(corners.mean(axis=0), center_2d, atol=1.0e-12))

    def test_watertight_winding_and_degenerate_faces(self) -> None:
        patches, wrist = layout(4)
        for mode in ("attachment_hull", "parametric_2_5d"):
            result = generate_palm_mesh(patches, wrist, params(), mode=mode)
            for mesh in (result.visual_mesh, result.collision_mesh):
                self.assertTrue(mesh.is_watertight, mode)
                self.assertTrue(mesh.is_winding_consistent, mode)
                self.assertEqual(mesh.body_count, 1, mode)
            self.assertEqual(result.metadata["visual_quality"]["degenerate_faces"], 0)
            self.assertEqual(result.metadata["collision_quality"]["degenerate_faces"], 0)

    def test_three_four_five_finger_layouts(self) -> None:
        for count in (3, 4, 5):
            patches, wrist = layout(count)
            result = generate_palm_mesh(patches, wrist, params(), mode="parametric_2_5d")
            self.assertEqual(len(result.attachment_frames), count + 1)
            self.assertTrue(all(result.metadata["attachment_coverage"].values()))
            self.assertTrue(result.visual_mesh.is_volume)

    def test_fixed_template_is_backward_compatible(self) -> None:
        patches, wrist = layout(3)
        template = trimesh.creation.box(extents=[0.8, 0.25, 1.0])
        result = generate_palm_mesh(
            patches, wrist, params(), mode="fixed_template", fixed_template_mesh=template
        )
        self.assertTrue(np.allclose(result.visual_mesh.bounds, template.bounds))
        self.assertEqual(len(result.visual_mesh.faces), len(template.faces))
        self.assertEqual(result.metadata["mode"], "fixed_template")

    def test_template_deformation_preserves_thickness_and_surface_topology(self) -> None:
        reference_patches, wrist = layout(5)
        slots = []
        current_patches = []
        for index, patch in enumerate(reference_patches):
            reference = patch.transform.copy()
            current = reference.copy()
            # Normal fingers move along palm width; the opposed thumb moves
            # along palm length. Both remain boundary-tangent edits.
            if index < 4:
                current[0, 3] += (-0.018 + 0.012 * index)
            else:
                current[2, 3] += 0.018
            current_patches.append(AttachmentPatch(
                patch.name, current, patch.width, patch.depth, patch.thickness
            ))
            slots.append({
                "slot_id": index,
                "role": "thumb" if index == 4 else f"finger_{index}",
                "attachment_translation": current[:3, 3].tolist(),
                "attachment_rotation": current[:3, :3].tolist(),
                "reference_attachment_translation": reference[:3, 3].tolist(),
                "reference_attachment_rotation": reference[:3, :3].tolist(),
            })
        template = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        template.vertices *= np.asarray([0.58, 0.14, 0.72])
        template.vertices[:, 2] += 0.42
        hand = {"finger_slots": slots, "parts": [{"world_pos": [0.0, 0.0, 0.0]}]}
        result = deform_template_palm(
            hand, current_patches, wrist, template, np.eye(3)
        )
        self.assertEqual(len(result.visual_mesh.vertices), len(template.vertices))
        self.assertEqual(len(result.visual_mesh.faces), len(template.faces))
        self.assertEqual(result.metadata["local_thickness_coordinate_max_error"], 0.0)
        self.assertEqual(
            result.metadata["original_local_thickness"],
            result.metadata["deformed_local_thickness"],
        )
        self.assertTrue(result.metadata["all_mounting_points_on_boundary"])
        self.assertTrue(result.collision_mesh.is_watertight)
        self.assertGreater(result.metadata["root_mount_locked_vertex_count"], 0)
        self.assertEqual(result.metadata["root_mount_locked_max_error"], 0.0)
        for patch in current_patches:
            self.assertTrue(np.array_equal(
                result.attachment_frames[patch.name], patch.transform
            ))

    def test_dense_surface_controls_do_not_exhaust_sparse_convex_hull(self) -> None:
        reference_patches, wrist = layout(5)
        current_patches = []
        slots = []
        for index, patch in enumerate(reference_patches):
            reference = patch.transform.copy()
            current = reference.copy()
            current[0, 3] += 0.008 * (index - 2)
            current_patches.append(AttachmentPatch(
                patch.name,
                current,
                patch.width,
                patch.depth,
                patch.thickness,
                footprint_center_offset=np.asarray([0.012, 0.0, 0.0]),
                reference_footprint_center_offset=np.asarray([0.006, 0.0, 0.0]),
            ))
            slots.append({
                "slot_id": index,
                "role": "thumb" if index == 4 else f"finger_{index}",
                "attachment_translation": current[:3, 3].tolist(),
                "attachment_rotation": current[:3, :3].tolist(),
                "reference_attachment_translation": reference[:3, 3].tolist(),
                "reference_attachment_rotation": reference[:3, :3].tolist(),
            })
        template = trimesh.creation.box(extents=[1.25, 0.20, 1.15])
        template.apply_translation([0.0, 0.0, 0.48])
        for _ in range(3):
            vertices, faces = trimesh.remesh.subdivide(template.vertices, template.faces)
            template = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        hand = {"finger_slots": slots, "parts": [{"world_pos": [0.0, 0.0, 0.0]}]}
        result = deform_template_palm(
            hand, current_patches, wrist, template, np.eye(3)
        )
        vertices_xz = np.asarray(result.visual_mesh.vertices)[:, [0, 2]]
        for record in result.metadata["mounting_points"]:
            center_control = int(record["control_vertex_ids"][1])
            self.assertTrue(np.allclose(
                vertices_xz[center_control],
                np.asarray(record["control_target_local"]),
                atol=1.0e-10,
            ))
        for patch in current_patches:
            self.assertTrue(np.array_equal(
                result.attachment_frames[patch.name], patch.transform
            ))

    def test_source_palm_finger_interface_follows_joint_frame(self) -> None:
        template = trimesh.creation.box(extents=[1.0, 0.20, 1.0])
        template.apply_translation([0.0, 0.0, 0.45])
        for _ in range(3):
            vertices, faces = trimesh.remesh.subdivide(template.vertices, template.faces)
            template = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        root_mesh = trimesh.creation.box(extents=[0.20, 0.18, 0.24])
        reference = frame(0.18, 0.93, 0.0)
        current = frame(0.27, 0.96, 0.16)
        root_rotation = current[:3, :3] @ reference[:3, :3].T
        hand = {
            "palm_transform": np.eye(3).tolist(),
            "finger_slots": [{
                "slot_id": 0,
                "role": "index",
                "root_node_id": 1,
                "attachment_translation": current[:3, 3].tolist(),
                "attachment_rotation": current[:3, :3].tolist(),
                "reference_attachment_translation": reference[:3, 3].tolist(),
                "reference_attachment_rotation": reference[:3, :3].tolist(),
            }],
            "parts": [
                {"world_pos": [0.0, 0.0, 0.0]},
                {
                    "world_pos": current[:3, 3].tolist(),
                    "mesh_linear": root_rotation.tolist(),
                    "reference_mesh_linear": np.eye(3).tolist(),
                    "source_mesh": {
                        "file": "root.obj",
                        "bounds": np.asarray(root_mesh.bounds).tolist(),
                    },
                },
            ],
        }
        configuration = infer_palm_params(template)
        patches, wrist = patches_from_hand_ir(
            hand,
            template,
            configuration,
            root_mesh_loader=lambda _: root_mesh,
        )
        result = deform_template_palm(
            hand, patches, wrist, template, np.eye(3)
        )
        records = result.metadata["joint_interface_patches"]
        self.assertEqual(len(records), 1)
        self.assertGreater(records[0]["source_interface_vertex_count"], 0)
        self.assertEqual(records[0]["maximum_free_interface_frame_error"], 0.0)
        self.assertEqual(records[0]["maximum_locked_interface_conflict"], 0.0)
        self.assertTrue(np.array_equal(
            result.attachment_frames[patches[0].name], current
        ))

    def test_house_cage_preserves_source_topology_thickness_and_mount(self) -> None:
        patches, wrist = layout(5)
        slots = [{
            "slot_id": index,
            "role": "thumb" if index == 4 else f"finger_{index}",
            "attachment_translation": patch.transform[:3, 3].tolist(),
            "attachment_rotation": patch.transform[:3, :3].tolist(),
        } for index, patch in enumerate(patches)]
        template = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        template.vertices *= np.asarray([0.62, 0.14, 0.78])
        template.vertices[:, 2] += 0.43
        hand = {
            "palm_layout": {"mode": "symmetric", "center_local_xz": [0.0, 0.43]},
            "finger_slots": slots,
            "parts": [{"world_pos": [0.0, 0.0, 0.0]}],
        }
        result = deform_template_to_house_palm(
            hand, patches, wrist, template, np.eye(3)
        )
        self.assertEqual(result.metadata["mode"], "source_topology_house")
        self.assertEqual(len(result.visual_mesh.vertices), len(template.vertices))
        self.assertEqual(len(result.visual_mesh.faces), len(template.faces))
        self.assertEqual(result.metadata["local_thickness_coordinate_max_error"], 0.0)
        self.assertGreater(result.metadata["root_mount_locked_vertex_count"], 0)
        self.assertEqual(result.metadata["root_mount_locked_max_error"], 0.0)
        self.assertTrue(result.metadata["all_motor_centers_covered"])
        self.assertTrue(result.metadata["all_motor_interfaces_reached"])
        self.assertLessEqual(result.metadata["maximum_planar_displacement_fraction"], 0.30)
        self.assertTrue(result.collision_mesh.is_watertight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
