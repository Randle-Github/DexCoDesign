#!/usr/bin/env python3
"""Write the complete discrete WUJI palm prototype design matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wuji_morphology_space import PALM_EXPANSION_LEVELS, SOURCE_VECTOR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    vectors = np.repeat(SOURCE_VECTOR[None, :], PALM_EXPANSION_LEVELS, axis=0)
    vectors[:, 0] = np.arange(PALM_EXPANSION_LEVELS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors.astype(np.float32))
    print(
        f"WUJI_PALM_PROTOTYPE_VECTORS levels={PALM_EXPANSION_LEVELS} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

