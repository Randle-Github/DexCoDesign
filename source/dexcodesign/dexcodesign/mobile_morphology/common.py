"""Shared paths, source specifications, and rigid-transform helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ASSETS = ROOT / "assets" / "mobile_manipulators"
ARTIFACT_ROOT = ROOT / "artifacts" / "mobile_manipulator_morphology"
GENERATED_COUNT = int(os.environ.get("MOBILE_MORPHOLOGY_COUNT", "32"))
GENERATED_ROOT = ARTIFACT_ROOT / f"generated_{GENERATED_COUNT}"
REFERENCE_GRAPHS = ARTIFACT_ROOT / "reference_graphs.json"
GRAMMAR = ARTIFACT_ROOT / "grammar.json"
PRECOMPUTED_ROOT = ARTIFACT_ROOT / "precomputed_link_variants"
PRECOMPUTED_VARIANTS = ARTIFACT_ROOT / "precomputed_link_variants.json"
ROBOT_IR = GENERATED_ROOT / "robot_ir.json"
COMPILED_ROBOTS = GENERATED_ROOT / "compiled_robots.json"

@dataclass(frozen=True)
class RobotSpec:
    source_id: str
    urdf: Path
    arm_link_pairs: tuple[tuple[str, str], ...]
    arm_joint_pairs: tuple[tuple[str, str], ...]
    torso_links: tuple[str, ...]
    display_joint_positions: tuple[tuple[str, float], ...] = ()
    display_yaw_degrees: float = -90.0


SPECS = (
    RobotSpec(
        "dexmate_vega",
        ASSETS / "dexmate_vega" / "eef_free" / "robot.urdf",
        tuple((f"L_arm_l{i}", f"R_arm_l{i}") for i in range(1, 8)),
        tuple((f"L_arm_j{i}", f"R_arm_j{i}") for i in range(1, 8)),
        ("torso_l1", "torso_l2", "torso_l3"),
        (
            ("torso_j1", 1.3666),
            ("torso_j2", 2.7021),
            ("torso_j3", 1.2036),
            ("L_arm_j1", math.pi / 2),
            ("R_arm_j1", -math.pi / 2),
        ),
        display_yaw_degrees=90.0,
    ),
    RobotSpec(
        "rby1",
        ASSETS / "rby1" / "eef_free" / "robot.urdf",
        tuple((f"link_left_arm_{i}", f"link_right_arm_{i}") for i in range(7)),
        tuple((f"left_arm_{i}", f"right_arm_{i}") for i in range(7)),
        tuple(f"link_torso_{i}" for i in range(6)),
    ),
    RobotSpec(
        "galaxea_r1",
        ASSETS / "galaxea_r1" / "eef_free" / "robot.urdf",
        tuple((f"left_arm_link{i}", f"right_arm_link{i}") for i in range(1, 7)),
        tuple((f"left_arm_joint{i}", f"right_arm_joint{i}") for i in range(1, 7)),
        tuple(f"torso_link{i}" for i in range(1, 5)),
        (
            ("left_arm_joint1", -math.pi / 2),
            ("left_arm_joint3", -math.pi),
            ("right_arm_joint1", math.pi / 2),
            ("right_arm_joint3", -math.pi),
        ),
    ),
)
SPEC_BY_ID = {spec.source_id: spec for spec in SPECS}


def numbers(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(value) for value in text.replace(",", " ").split()])


def fmt(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def rpy_matrix(rpy) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def longitudinal_linear(connector_vector, factor: float) -> np.ndarray:
    """Hand-pipeline affine map shared by one rigid mesh node and its graph edges."""
    vector = np.asarray(connector_vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("zero connector vector")
    axis = vector / norm
    return np.eye(3) + (float(factor) - 1.0) * np.outer(axis, axis)


def transform(xyz=None, rpy=None) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rpy_matrix((0, 0, 0) if rpy is None else rpy)
    result[:3, 3] = (0, 0, 0) if xyz is None else xyz
    return result
