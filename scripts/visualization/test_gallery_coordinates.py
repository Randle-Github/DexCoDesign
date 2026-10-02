"""Regression checks for source-frame articulation and floor annotations."""

import unittest

import numpy as np
import trimesh

from render_geometry_gallery import ground_and_offset, posed_vertices


class GalleryCoordinateTests(unittest.TestCase):
    def test_arctic_positive_q_rotates_top_about_local_negative_z_once(self):
        mesh = trimesh.Trimesh(vertices=[[1.0, 0.0, 0.0]], faces=[], process=False)
        data = {
            "object_root_pose_wxyz": np.array([[[0, 0, 0, 1, 0, 0, 0]]], dtype=float),
            "object_joint_positions_rad": np.array([[[np.pi / 2]]], dtype=float),
        }
        # An old manifest contains -1. The official ARCTIC axis is already
        # negative Z, so that stale field must never negate q a second time.
        spec = {"object_index": 0, "part": "top", "articulation_sign": -1.0}
        np.testing.assert_allclose(
            posed_vertices(mesh, spec, data, 0, "arctic", 0.0),
            [[0.0, -1.0, 0.0]], atol=1e-7,
        )
        spec["part"] = "bottom"
        np.testing.assert_allclose(
            posed_vertices(mesh, spec, data, 0, "arctic", 0.0),
            [[1.0, 0.0, 0.0]], atol=1e-7,
        )

    def test_gigahands_does_not_claim_a_table(self):
        roots = np.array([[[0, 0, 0, 1, 0, 0, 0]]], dtype=float)
        ground, label, offset = ground_and_offset(
            {"dataset": "gigahands", "clip_start": 0},
            {"object_root_pose_wxyz": roots}, [],
        )
        self.assertIsNone(ground)
        self.assertIn("not calibrated", label)
        self.assertEqual(offset, 0.0)


if __name__ == "__main__":
    unittest.main()
