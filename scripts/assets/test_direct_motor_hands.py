#!/usr/bin/env python3
"""Regression checks for the generated direct-motor hand URDF library."""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "robot_hands"
REGISTRY = ASSETS / "direct_motor" / "registry.json"
AUDIT = ASSETS / "direct_motor" / "conversion_audit.json"


class DirectMotorHandsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    def test_all_registered_bilateral_assets_exist(self) -> None:
        self.assertEqual(len(self.registry["hands"]), 14)
        for hand in self.registry["hands"].values():
            self.assertEqual(set(hand["entries"]), {"left", "right"})
            for entry in hand["entries"].values():
                self.assertTrue((ASSETS / entry["path"]).exists())

    def test_every_movable_joint_is_active_or_mimic(self) -> None:
        for hand_id, hand in self.registry["hands"].items():
            for side, entry in hand["entries"].items():
                path = ASSETS / entry["path"]
                robot = ET.parse(path).getroot()
                active = {
                    joint.get("name")
                    for transmission in robot.findall("transmission")
                    if (joint := transmission.find("joint")) is not None
                }
                passive = set()
                for joint in robot.findall("joint"):
                    if joint.get("type") not in {
                        "revolute",
                        "continuous",
                        "prismatic",
                    }:
                        continue
                    name = joint.get("name")
                    mimic = joint.find("mimic")
                    if mimic is None:
                        self.assertIn(name, active, f"{hand_id}/{side}/{name}")
                    else:
                        passive.add(name)
                        self.assertNotIn(name, active, f"{hand_id}/{side}/{name}")
                        self.assertIn(
                            mimic.get("joint"),
                            active,
                            f"{hand_id}/{side}/{name}",
                        )
                self.assertEqual(len(active), entry["active_dofs"])
                self.assertEqual(len(passive), entry["passive_mimic_dofs"])
                self.assertEqual(len(active) + len(passive), entry["scalar_dofs"])

    def test_no_tendon_or_nonlinear_constraint_survives(self) -> None:
        for hand in self.registry["hands"].values():
            for entry in hand["entries"].values():
                text = (ASSETS / entry["path"]).read_text(encoding="utf-8")
                self.assertNotIn("<tendon", text)
                self.assertNotIn("<equality", text)
                self.assertNotIn("coupled_joint", text)

    def test_ranges_meshes_and_face_counts_are_preserved(self) -> None:
        for entry in self.audit["entries"]:
            self.assertFalse(entry["output_audit"]["missing_meshes"])
            self.assertFalse(entry["output_audit"]["loose_or_invalid_dofs"])
            self.assertEqual(
                entry["scalar_dofs"],
                entry["active_dofs"] + entry["passive_mimic_dofs"],
            )
            if "mesh_faces_preserved" in entry:
                self.assertTrue(entry["mesh_faces_preserved"], entry["output_path"])
            for joint in entry["joint_records"]:
                lower, upper = joint["range"]
                self.assertLess(lower, upper, joint["joint"] if "joint" in joint else joint["name"])

    def test_mano_is_fully_active(self) -> None:
        for side in ("left", "right"):
            entry = self.registry["hands"]["mano"]["entries"][side]
            self.assertEqual(entry["scalar_dofs"], 28)
            self.assertEqual(entry["active_dofs"], 28)
            self.assertEqual(entry["passive_mimic_dofs"], 0)

    def test_all_generated_urdfs_compile_in_mujoco(self) -> None:
        for hand_id, hand in self.registry["hands"].items():
            for side, entry in hand["entries"].items():
                model = mujoco.MjModel.from_xml_path(str(ASSETS / entry["path"]))
                self.assertEqual(model.nv, entry["scalar_dofs"], f"{hand_id}/{side}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
