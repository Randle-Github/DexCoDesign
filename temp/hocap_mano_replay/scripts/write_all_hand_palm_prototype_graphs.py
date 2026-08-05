#!/usr/bin/env python3
"""Create the 32-level real-palm mesh bank graph for every robot hand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROBOT_HANDS = (
    "ability_hand",
    "schunk_svh",
    "wuji_hand_2",
    "sharpa_wave_01",
    "tesollo_dg5f",
    "unitree_dex5_1",
    "robotera_xhand1",
    "orca_hand_v2",
    "shadow_hand_e",
    "allegro_hand_v5",
    "midas_hand",
    "ruka_v2",
    "inspire_rh56dfx",
)
LEVELS = 32
MAX_EXPANSION = 0.35


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-hand", action="append", default=[])
    args = parser.parse_args()
    selected = tuple(args.source_hand) if args.source_hand else ROBOT_HANDS
    unknown = set(selected) - set(ROBOT_HANDS)
    if unknown:
        raise ValueError(f"unknown/non-robot hand sources: {sorted(unknown)}")
    hands = []
    for source in selected:
        for index in range(LEVELS):
            expansion = MAX_EXPANSION * index / (LEVELS - 1)
            hands.append(
                {
                    "hand_id": f"{source}__palm_{index:02d}",
                    "source_hand": source,
                    "palm": {
                        "layout_mode": "anthropomorphic",
                        "expansion_index": index,
                        "expansion_levels": LEVELS,
                        "expansion": expansion,
                        "scale_x": 1.0,
                        "scale_z": 1.0,
                        "yaw": 0.0,
                    },
                    "fingers": {
                        "default_length_scale": 1.0,
                        "default_radius_scale": 1.0,
                    },
                }
            )
    payload = {
        "schema_version": 1,
        "palm_prototype_bank": {
            "levels": LEVELS,
            "minimum": 0.0,
            "maximum": MAX_EXPANSION,
            "sources": list(selected),
            "mano_excluded": True,
        },
        "hands": hands,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"ALL_HAND_PALM_BANK_GRAPHS sources={len(selected)} "
        f"prototypes={len(hands)} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

