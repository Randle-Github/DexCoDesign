#!/usr/bin/env python3
"""Write the complete discrete WUJI palm prototype design matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wuji_general_space import (
    PALM_EXPANSION_LEVELS,
    PALM_EXPANSION_MAX,
    SCHEMA,
    SOURCE_VECTOR,
    VECTOR_NAMES,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    vectors = np.repeat(SOURCE_VECTOR[None, :], PALM_EXPANSION_LEVELS, axis=0)
    vectors[:, 0] = np.arange(PALM_EXPANSION_LEVELS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors.astype(np.float32))
    signature = {
        "schema_version": 1,
        "grammar_id": SCHEMA["grammar_id"],
        "source_hand": SCHEMA["source_hand"],
        "vector_dimension": len(VECTOR_NAMES),
        "vector_names": list(VECTOR_NAMES),
        "palm_layout_mode": "source_star_fusion",
        "palm_prototype_count": PALM_EXPANSION_LEVELS,
        "palm_expansion_range": [0.0, PALM_EXPANSION_MAX],
        "zero_prototype_is_exact_source": True,
    }
    args.output.with_suffix(".schema.json").write_text(
        json.dumps(signature, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"WUJI_PALM_PROTOTYPE_VECTORS levels={PALM_EXPANSION_LEVELS} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
