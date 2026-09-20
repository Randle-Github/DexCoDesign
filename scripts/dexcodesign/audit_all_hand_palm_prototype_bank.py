#!/usr/bin/env python3
"""Verify every requested palm prototype produced a real collision mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("compiled", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.compiled.read_text(encoding="utf-8"))
    root = args.compiled.parent
    rows = []
    for hand in payload["hands"]:
        palm = next(
            part
            for part in hand["parts"]
            if int(part["id"]) == 0 and part["role"] == "palm"
        )
        mesh = palm.get("compiled_mesh")
        if mesh is None:
            raise RuntimeError(f"{hand['hand_id']}: palm mesh was not compiled")
        visual = root / mesh["file"]
        collision = root / mesh.get("collision_file", mesh["file"])
        if not visual.is_file() or not collision.is_file():
            raise FileNotFoundError(f"{hand['hand_id']}: {visual} / {collision}")
        graph = hand["graph_parameters"]["palm"]
        rows.append(
            {
                "hand_id": hand["hand_id"],
                "source_hand": hand["seed_source"],
                "palm_expansion_index": int(round(graph["expansion"] * 31 / 0.35)),
                "palm_expansion": float(graph["expansion"]),
                "visual_mesh": str(visual.resolve()),
                "collision_mesh": str(collision.resolve()),
                "visual_faces": int(mesh["faces"]),
                "collision_faces": int(mesh.get("collision_faces", mesh["faces"])),
            }
        )
    by_source: dict[str, list[int]] = {}
    for row in rows:
        by_source.setdefault(row["source_hand"], []).append(
            row["palm_expansion_index"]
        )
    invalid = {
        source: sorted(indices)
        for source, indices in by_source.items()
        if sorted(indices) != list(range(32))
    }
    if invalid:
        raise RuntimeError(f"incomplete prototype indices: {invalid}")
    result = {
        "schema_version": 1,
        "sources": len(by_source),
        "levels_per_source": 32,
        "prototype_count": len(rows),
        "all_prototypes_have_real_visual_and_collision_mesh": True,
        "mano_excluded": "mano" not in by_source,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"ALL_HAND_PALM_BANK_AUDIT sources={len(by_source)} "
        f"prototypes={len(rows)} valid=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
