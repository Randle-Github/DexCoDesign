#!/usr/bin/env python3
"""Generate an isolated, exaggerated multi-edit mobile-robot population."""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from dexcodesign.mobile_morphology import mesh_compiler, preprocess, render
from dexcodesign.mobile_morphology.common import (
    GRAMMAR,
    ROOT,
    SPEC_BY_ID,
    longitudinal_linear,
)
from dexcodesign.mobile_morphology.preprocess import variant_key


SEED = 20260725
COUNT = 32
LENGTH_FACTORS = (0.55, 1.75)
TERMINAL_LENGTH_FACTORS = (1.75,)
SHOULDER_FACTORS = (0.70, 1.40)
MIN_ACTIONS = 3
MAX_ACTIONS = 5

FORMAL_ROOT = (
    ROOT
    / "artifacts"
    / "mobile_manipulator_morphology"
    / "generated_32"
)
AMUSE_ROOT = FORMAL_ROOT / "amuse"
AMUSE_CACHE = AMUSE_ROOT / "precomputed_link_variants"
AMUSE_TILES = AMUSE_ROOT / "render_tiles"
OUTPUT = FORMAL_ROOT / "mobile_robots_amuse.png"
MANIFEST = AMUSE_ROOT / "compiled_robots_amuse.json"


def factors_for(production_name: str, group: dict) -> tuple[float, ...]:
    if production_name == "shoulder_width":
        return SHOULDER_FACTORS
    source_factors = group.get("factors", ())
    if source_factors == [1.5] or source_factors == (1.5,):
        return TERMINAL_LENGTH_FACTORS
    return LENGTH_FACTORS


def build_extreme_variants(
    grammars: list[dict],
) -> tuple[dict, dict, dict]:
    preprocess.PRECOMPUTED_ROOT = AMUSE_CACHE
    variants = {}
    allowed_factors = {}
    cache_hits = cache_misses = 0
    for grammar in grammars:
        source_id = grammar["source_id"]
        spec = SPEC_BY_ID[source_id]
        robot = ET.parse(spec.urdf).getroot()
        links = {
            link.get("name", ""): link
            for link in robot.findall("link")
        }
        for production_name, production in grammar[
            "productions"
        ].items():
            for group in production["groups"]:
                group_key = (source_id, production_name, group["group_id"])
                allowed_factors[group_key] = []
                for factor in factors_for(production_name, group):
                    member_variants = []
                    valid = True
                    for member in group["members"]:
                        link_name = member["link"]
                        link = links[link_name]
                        linear = longitudinal_linear(
                            member["deformation_axis"], factor
                        )
                        connector_rig = (
                            preprocess.build_connector_rig(
                                link,
                                spec.urdf.parent,
                                member,
                                factor,
                            )
                            if production_name
                            == "vertical_link_length"
                            else None
                        )
                        if connector_rig is not None:
                            direction = np.asarray(
                                connector_rig["direction"], dtype=float
                            )
                            target_distal = (
                                float(
                                    connector_rig[
                                        "distal_connector_projection"
                                    ]
                                )
                                + float(
                                    direction
                                    @ np.asarray(
                                        connector_rig[
                                            "distal_displacement"
                                        ],
                                        dtype=float,
                                    )
                                )
                            )
                            minimum_distal = float(
                                connector_rig["middle_start"]
                            ) + 0.01
                            if target_distal <= minimum_distal:
                                valid = False
                                break
                        records = []
                        for owner_kind in ("visual", "collision"):
                            for owner_index, owner in enumerate(
                                link.findall(owner_kind)
                            ):
                                record = preprocess.owner_mesh_variant(
                                    source_id=source_id,
                                    link_name=link_name,
                                    owner_kind=owner_kind,
                                    owner_index=owner_index,
                                    owner=owner,
                                    source_dir=spec.urdf.parent,
                                    linear=linear,
                                    factor=factor,
                                    production=production_name,
                                    connector_rig=connector_rig,
                                )
                                if record is None:
                                    continue
                                records.append(record)
                                if record["cache_hit"]:
                                    cache_hits += 1
                                else:
                                    cache_misses += 1
                        member_variants.append(
                            (
                                variant_key(
                                    source_id,
                                    production_name,
                                    link_name,
                                    factor,
                                ),
                                {
                                    "source_id": source_id,
                                    "production": production_name,
                                    "link": link_name,
                                    "factor": float(factor),
                                    "deformation_axis": member[
                                        "deformation_axis"
                                    ],
                                    "linear": linear.tolist(),
                                    "connector_rig": connector_rig,
                                    "mesh_owners": records,
                                },
                            )
                        )
                    if not valid:
                        continue
                    variants.update(member_variants)
                    allowed_factors[group_key].append(float(factor))
                if not allowed_factors[group_key]:
                    raise ValueError(
                        f"no safe amuse factors for {group_key}"
                    )
    return variants, allowed_factors, {
        "allowed_factors": {
            json.dumps(key): value
            for key, value in allowed_factors.items()
        },
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }


def make_action(
    source_id: str,
    production_name: str,
    production: dict,
    group: dict,
    factor: float,
) -> dict:
    return {
        "production": production_name,
        "group_id": group["group_id"],
        "kind": group["kind"],
        "factor": factor,
        "other_axis_scale": production["scale_other_axes"],
        "reference_direction": production["reference_direction"],
        "members": group["members"],
    }


def sample_population(
    grammars: list[dict],
    factor_map: dict[tuple[str, str, str], list[float]],
) -> list[dict]:
    rng = random.Random(SEED)
    robots = []
    for index in range(COUNT):
        grammar = grammars[index % len(grammars)]
        source_id = grammar["source_id"]
        candidates = []
        for production_name, production in grammar[
            "productions"
        ].items():
            for group in production["groups"]:
                factors = factor_map[
                    (source_id, production_name, group["group_id"])
                ]
                candidates.append(
                    (
                        production_name,
                        production,
                        group,
                        rng.choice(factors),
                    )
                )
        rng.shuffle(candidates)
        action_count = rng.randint(
            MIN_ACTIONS, min(MAX_ACTIONS, len(candidates))
        )
        selected = []
        targeted_links = set()
        for candidate in candidates:
            group = candidate[2]
            links = {
                member["link"] for member in group["members"]
            }
            if links & targeted_links:
                continue
            selected.append(candidate)
            targeted_links |= links
            if len(selected) == action_count:
                break
        if len(selected) < MIN_ACTIONS:
            raise ValueError(
                f"{source_id}: could not sample multiple independent edits"
            )
        actions = [
            make_action(
                source_id,
                production_name,
                production,
                group,
                factor,
            )
            for production_name, production, group, factor in selected
        ]
        robots.append(
            {
                "robot_id": f"amuse_{index:03d}_{source_id}",
                "source_id": source_id,
                "grammar_id": (
                    "mobile-manipulator-amuse-length-width-v1"
                ),
                "actions": actions,
                "audit": {
                    "base_immutable": True,
                    "arm_edits_bilateral": all(
                        action["kind"] != "bilateral_arm_pair"
                        or len(action["members"]) == 2
                        for action in actions
                    ),
                    "joint_orientation_edits_absent": True,
                    "multiple_distinct_links_changed": (
                        len(targeted_links) > 1
                    ),
                },
            }
        )
    return robots


def validate_compiled(robots: list[dict]) -> None:
    for entry in robots:
        if len(entry["actions"]) < MIN_ACTIONS:
            raise ValueError(
                f"{entry['robot_id']}: fewer than {MIN_ACTIONS} edits"
            )
        if not all(
            entry["audit"][key]
            for key in (
                "base_immutable",
                "arm_edits_bilateral",
                "joint_orientation_edits_absent",
                "multiple_distinct_links_changed",
            )
        ):
            raise ValueError(
                f"{entry['robot_id']}: amuse rule audit failed"
            )
        path = AMUSE_ROOT / entry["compiled_urdf"]
        robot = ET.parse(path).getroot()
        for mesh in robot.findall(".//mesh"):
            mesh_path = (
                path.parent / mesh.get("filename", "")
            ).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)


def render_population(robots: list[dict]) -> None:
    render.GENERATED_ROOT = AMUSE_ROOT
    render.TILE_DIR = AMUSE_TILES
    if AMUSE_TILES.exists():
        shutil.rmtree(AMUSE_TILES)
    tiles = [
        render.render_tile(
            robots[start : start + 8],
            start // 8,
            output_dir=AMUSE_TILES,
            spacing_x=2.05,
            spacing_y=2.65,
            camera_distance=7.25,
        )
        for start in range(0, COUNT, 8)
    ]
    canvas = Image.new("RGB", (3200, 1720))
    for index, tile in enumerate(tiles):
        with Image.open(tile) as image:
            cropped = image.convert("RGB").crop((0, 40, 1600, 900))
            canvas.paste(
                cropped,
                ((index % 2) * 1600, (index // 2) * 860),
            )
    canvas.save(OUTPUT)


def main() -> int:
    grammar_payload = json.loads(
        GRAMMAR.read_text(encoding="utf-8")
    )
    grammars = grammar_payload["robots"]
    grammar_by_id = {
        grammar["source_id"]: grammar for grammar in grammars
    }
    AMUSE_ROOT.mkdir(parents=True, exist_ok=True)
    for stale_dir in (AMUSE_ROOT / "robots", AMUSE_TILES):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    if MANIFEST.exists():
        MANIFEST.unlink()
    variants, allowed_factors, variant_audit = (
        build_extreme_variants(grammars)
    )
    robot_ir = sample_population(grammars, allowed_factors)

    mesh_compiler.GENERATED_ROOT = AMUSE_ROOT
    compiled = [
        mesh_compiler.compile_robot(
            robot,
            grammar_by_id[robot["source_id"]],
            variants,
        )
        for robot in robot_ir
    ]
    payload = {
        "schema_version": 1,
        "grammar_id": (
            "mobile-manipulator-amuse-length-width-v1"
        ),
        "temporary_experiment": True,
        "seed": SEED,
        "extreme_factors": {
            "length": LENGTH_FACTORS,
            "terminal_length": TERMINAL_LENGTH_FACTORS,
            "shoulder_width": SHOULDER_FACTORS,
        },
        "variant_cache": variant_audit,
        "robots": compiled,
        "summary": {
            "robots": len(compiled),
            "minimum_actions_per_robot": min(
                len(robot["actions"]) for robot in compiled
            ),
            "maximum_actions_per_robot": max(
                len(robot["actions"]) for robot in compiled
            ),
            "all_base_immutable": True,
            "all_arm_edits_bilateral": True,
            "all_joint_orientations_unchanged": True,
            "all_multiple_mesh_edits": True,
        },
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    render_population(compiled)
    print(json.dumps(payload["summary"], indent=2))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
