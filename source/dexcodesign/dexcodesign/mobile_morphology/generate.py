"""Sample length-only rewrites from the mobile-manipulator grammar."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from . import SCHEMA_VERSION
from .common import GENERATED_ROOT, GRAMMAR, ROBOT_IR


MODES = ("vertical_link_length", "shoulder_width")


def make_action(grammar: dict, mode: str, rng: random.Random) -> dict:
    production = grammar["productions"][mode]
    group = rng.choice(production["groups"])
    return {
        "production": mode,
        "group_id": group["group_id"],
        "kind": group["kind"],
        "factor": rng.choice(
            group.get("factors", production["factors"])
        ),
        "other_axis_scale": production["scale_other_axes"],
        "reference_direction": production["reference_direction"],
        "members": group["members"],
    }


def validate_action(action: dict, grammar: dict) -> None:
    if action["production"] not in MODES:
        raise ValueError(
            f"orientation/rotation production is forbidden: "
            f"{action['production']}"
        )
    production = grammar["productions"][action["production"]]
    groups = {
        group["group_id"]: group for group in production["groups"]
    }
    if action["group_id"] not in groups:
        raise ValueError(
            f"unknown production group: {action['group_id']}"
        )
    reference = groups[action["group_id"]]
    if action["kind"] != reference["kind"]:
        raise ValueError("production kind changed")
    if action["factor"] not in reference.get(
        "factors", production["factors"]
    ):
        raise ValueError("invalid directional deformation factor")
    if action["other_axis_scale"] != 1.0:
        raise ValueError("length production changes thickness")
    if len(action["members"]) != len(reference["members"]):
        raise ValueError("production member count changed")
    for member, expected in zip(
        action["members"], reference["members"]
    ):
        for key in (
            "link",
            "side",
            "deformation_axis",
            "reference_world_direction",
            "reference_pose",
        ):
            if member.get(key) != expected.get(key):
                raise ValueError(
                    f"grammar-selected length contract changed: {key}"
                )
        if (
            action["production"] == "shoulder_width"
            and member.get("attachment_edges")
            != expected.get("attachment_edges")
        ):
            raise ValueError(
                "grammar-selected shoulder attachment path changed"
            )
    if action["kind"] == "bilateral_arm_pair":
        if [member["side"] for member in action["members"]] != [
            "left",
            "right",
        ]:
            raise ValueError(
                "arm production is not an atomic left/right pair"
            )
    immutable_links = set(
        grammar["hard_constraints"]["immutable_links"]
    )
    if any(
        member["link"] in immutable_links
        for member in action["members"]
    ):
        raise ValueError("immutable/base link targeted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(args.seed)
    payload = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    grammars = payload["robots"]
    robots = []
    source_occurrences = Counter()
    for index in range(args.count):
        grammar = grammars[index % len(grammars)]
        available_modes = [
            mode
            for mode in MODES
            if grammar["productions"][mode]["groups"]
        ]
        if not available_modes:
            raise ValueError(
                f"{grammar['source_id']} has no constructible productions"
            )
        occurrence = source_occurrences[grammar["source_id"]]
        mode = available_modes[occurrence % len(available_modes)]
        source_occurrences[grammar["source_id"]] += 1
        actions = [make_action(grammar, mode, rng)]
        for action in actions:
            validate_action(action, grammar)
        robots.append(
            {
                "robot_id": (
                    f"mobile_{index:03d}_{grammar['source_id']}"
                ),
                "source_id": grammar["source_id"],
                "grammar_id": payload["grammar_id"],
                "actions": actions,
                "audit": {
                    "base_immutable": True,
                    "arm_edits_bilateral": all(
                        action["kind"] != "bilateral_arm_pair"
                        or len(action["members"]) == 2
                        for action in actions
                    ),
                    "directions_grammar_locked": True,
                    "joint_orientation_edits_absent": True,
                },
            }
        )
    counts = Counter(robot["source_id"] for robot in robots)
    mode_counts = Counter(
        action["production"]
        for robot in robots
        for action in robot["actions"]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "grammar_id": payload["grammar_id"],
        "seed": args.seed,
        "robots": robots,
        "summary": {
            "robots": len(robots),
            "by_source": dict(counts),
            "production_applications": dict(mode_counts),
            "all_base_immutable": all(
                robot["audit"]["base_immutable"]
                for robot in robots
            ),
            "all_arm_edits_bilateral": all(
                robot["audit"]["arm_edits_bilateral"]
                for robot in robots
            ),
            "all_directions_grammar_locked": all(
                robot["audit"]["directions_grammar_locked"]
                for robot in robots
            ),
            "all_joint_orientation_edits_absent": all(
                robot["audit"]["joint_orientation_edits_absent"]
                for robot in robots
            ),
        },
    }
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    ROBOT_IR.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
