#!/usr/bin/env python3
"""Build the first model-free HandIR/grammar/compiler reference experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source/dexcodesign"))

from dexcodesign import build_wuji_demo_variants, compile_hand, import_reference_library, validate_hand
from dexcodesign.io import write_json
from dexcodesign.visualization import draw_structure_graph

OUTPUTS = REPO_ROOT / "temp/design_grammar/outputs"


def main() -> int:
    hands, database = import_reference_library(REPO_ROOT)
    reference_dir = OUTPUTS / "reference_handir"
    graph_dir = OUTPUTS / "reference_graphs"
    audits = {}
    for hand in hands:
        audit = validate_hand(hand, database, REPO_ROOT)
        audits[hand.hand_id] = audit
        write_json(reference_dir / f"{hand.hand_id}.json", hand.to_dict())
        draw_structure_graph(hand, graph_dir / f"{hand.hand_id}.png")
    write_json(OUTPUTS / "module_database.json", database.to_dict())
    write_json(
        OUTPUTS / "reference_index.json",
        {
            "right_hand_references": len(hands),
            "source_hand_ids": [hand.hand_id for hand in hands],
            "total_rigid_functional_parts": sum(value["nodes"] for value in audits.values()),
            "total_joints": sum(value["joints"] for value in audits.values()),
            "total_movable_dofs": sum(value["dof"] for value in audits.values()),
            "audits": audits,
            "importer_boundary": (
                "The first refactor imports the previously audited canonical graph/rigid-part library. "
                "Native URDF/MJCF/USD fixed-cluster parsing is the next importer replacement."
            ),
        },
    )

    wuji = next(hand for hand in hands if hand.hand_id == "wuji_hand_2")
    source_compiled = compile_hand(wuji, database, REPO_ROOT, OUTPUTS / "compiled_source_wuji/meshes")
    write_json(OUTPUTS / "compiled_source_wuji/compiled.json", source_compiled)

    variants = build_wuji_demo_variants(wuji)
    variant_index = []
    for index, hand in enumerate(variants, start=1):
        output = OUTPUTS / "variants" / f"{index:02d}_{hand.metadata['variant_spec']['name']}"
        compiled = compile_hand(hand, database, REPO_ROOT, output / "meshes")
        write_json(output / "handir.json", hand.to_dict())
        write_json(output / "compiled.json", compiled)
        draw_structure_graph(hand, output / "structure_graph.png")
        variant_index.append(
            {
                "index": index,
                "hand_id": hand.hand_id,
                "name": hand.metadata["variant_spec"]["name"],
                "spec": hand.metadata["variant_spec"],
                "audit": compiled["audit"],
                "directory": str(output.relative_to(REPO_ROOT)),
            }
        )
    write_json(
        OUTPUTS / "variant_index.json",
        {
            "source_hand": "wuji_hand_2",
            "variant_count": len(variant_index),
            "cross_source_meshes": 0,
            "variants": variant_index,
        },
    )
    print(
        f"references={len(hands)} parts={sum(v['nodes'] for v in audits.values())} "
        f"joints={sum(v['joints'] for v in audits.values())} variants={len(variants)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
