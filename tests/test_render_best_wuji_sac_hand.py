"""Geometry-only WUJI comparison must preserve source origins and metric scale."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_best_wuji_sac_hand", ROOT / "temp/hocap_mano_replay/scripts/render_best_wuji_sac_hand.py"
)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_zero_pose_accumulates_parent_rotation_and_geometry_origin():
    # Deliberately list joints out of topological order. A parent's yaw rotates
    # the next joint's local +X offset onto world +Y.
    root = ET.fromstring(
        """
        <robot name="test">
          <link name="base"/><link name="middle"/><link name="tip"/>
          <joint name="second" type="revolute">
            <parent link="middle"/><child link="tip"/><origin xyz="2 0 0"/>
          </joint>
          <joint name="first" type="fixed">
            <parent link="base"/><child link="middle"/>
            <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
          </joint>
        </robot>
    """
    )
    transforms = renderer.zero_pose_link_transforms(root)
    np.testing.assert_allclose(transforms["tip"][:3, 3], [1, 2, 0], atol=1e-14)
    geometry = ET.fromstring('<origin xyz="0 1 0" rpy="0 0 0"/>')
    point = transforms["tip"] @ renderer.origin_transform(geometry) @ [0, 0, 0, 1]
    np.testing.assert_allclose(point[:3], [0, 2, 0], atol=1e-14)


def test_original_mesh_preserves_scale_mesh_origin_and_source_files(tmp_path):
    source = tmp_path / "source.obj"
    trimesh.creation.box(extents=[2, 2, 2]).export(source)
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test">
          <link name="base"/>
          <link name="tip"><collision>
            <origin xyz="1 0 0" rpy="0 0 0"/>
            <geometry><mesh filename="source.obj" scale="2 1 3"/></geometry>
          </collision></link>
          <joint name="finger" type="revolute">
            <parent link="base"/><child link="tip"/>
            <origin xyz="0 5 0" rpy="0 0 1.5707963267948966"/>
          </joint>
        </robot>
    """
    )
    original_bytes = (source.read_bytes(), urdf.read_bytes())
    records, bounds = renderer.original_meshes(urdf, tmp_path / "render")
    np.testing.assert_allclose(bounds, [[-1, 4, -3], [1, 8, 3]], atol=1e-14)
    assert len(records) == 1
    np.testing.assert_allclose(trimesh.load(records[0][1], force="mesh").bounds, bounds, atol=1e-6)
    assert (source.read_bytes(), urdf.read_bytes()) == original_bytes


def test_template_urdf_prefers_bank_over_materialized_candidate(tmp_path):
    bank = tmp_path / "bank/prepared/candidates/source"
    urdf = bank / "runtime/source/left/hand.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text('<robot name="source"/>')
    manifest = {
        "hand_usd_paths": [str(tmp_path / "new_candidate/asset/hand.usd")],
        "parametric_template_usd_paths": [str(bank / "asset/hand.usd")],
    }
    assert renderer.template_urdf(manifest, 0) == urdf


def test_best_proposal_uses_proposal_mean_not_one_lucky_rollout(tmp_path):
    (tmp_path / "training_history.json").write_text(
        json.dumps([{"generation": 0, "best_reward": 100}, {"generation": 1, "best_reward": 200}])
    )
    generation = tmp_path / "generation_001"
    generation.mkdir()
    (generation / "physx_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {"candidate_index": 0, "total_reward": 150},
                    {"candidate_index": 1, "total_reward": 200},
                ],
                "rollout_results": [
                    {"candidate_index": 0, "total_reward": 500},
                    {"candidate_index": 1, "total_reward": 190},
                    {"candidate_index": 1, "total_reward": 210},
                ],
            }
        )
    )
    number, _, best, rollouts = renderer.best_proposal(tmp_path)
    assert number == 1 and best["candidate_index"] == 1
    assert len(rollouts) == 2 and all(r["candidate_index"] == 1 for r in rollouts)


def test_original_source_tree_has_one_base_and_all_links():
    root = ET.parse(renderer.ORIGINAL_URDF).getroot()
    transforms = renderer.zero_pose_link_transforms(root)
    assert set(transforms) == {link.attrib["name"] for link in root.findall("link")}
    np.testing.assert_array_equal(transforms["l_base_link"], np.eye(4))
    assert not np.allclose(transforms["l_wrist"], np.eye(4))


def test_invalid_joint_tree_is_rejected():
    root = ET.fromstring('<robot name="bad"><link name="a"/><link name="b"/></robot>')
    with pytest.raises(ValueError, match="one original hand base"):
        renderer.zero_pose_link_transforms(root)
