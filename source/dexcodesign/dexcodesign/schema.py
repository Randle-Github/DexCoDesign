"""Versioned HandIR and source-bound mechanism-module schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LinkCandidateSpec:
    candidate_id: str
    bundle_id: str
    source_hand_id: str
    source_part_id: int
    semantic_role: str
    visual_path: str | None
    canonical_dimensions: tuple[float, float, float]
    proximal_connector: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    distal_connectors: tuple[tuple[float, ...], ...] = ()
    deformation_bounds: tuple[float, float] = (0.78, 1.22)


@dataclass(frozen=True)
class MechanismBundleSpec:
    bundle_id: str
    source_hand_id: str
    source_joint_name: str
    actuator_id: str | None
    binding_status: str
    transmission_type: str
    driven_joint_names: tuple[str, ...]
    link_candidate_ids: tuple[str, ...]
    searchable_parameters: tuple[str, ...] = ("length_scale", "radial_scale")


@dataclass
class LinkNodeIR:
    node_id: int
    source_node_id: int
    semantic_role: str
    finger_slot: int | None
    bundle_id: str
    candidate_id: str | None
    length_scale: float = 1.0
    radial_scale: float = 1.0
    palm_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class JointEdgeIR:
    joint_id: int
    parent_node: int
    child_node: int
    source_joint_name: str
    joint_type: str
    origin_translation: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    zero_position: float
    active: bool
    motor_link_binding_id: str


@dataclass
class FingerSlotIR:
    slot_id: int
    role: str
    active: bool
    palm_node_id: int
    attachment_translation: tuple[float, float, float]
    root_bundle_id: str


@dataclass
class HandIR:
    hand_id: str
    source_hand_id: str
    handedness: str
    schema_version: str = "0.1"
    compiler_version: str = "0.1"
    nodes: list[LinkNodeIR] = field(default_factory=list)
    joints: list[JointEdgeIR] = field(default_factory=list)
    finger_slots: list[FingerSlotIR] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleDatabase:
    schema_version: str = "0.1"
    bundles: dict[str, MechanismBundleSpec] = field(default_factory=dict)
    candidates: dict[str, LinkCandidateSpec] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundles": {key: asdict(value) for key, value in self.bundles.items()},
            "candidates": {key: asdict(value) for key, value in self.candidates.items()},
        }
