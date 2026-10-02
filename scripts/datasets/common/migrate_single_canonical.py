#!/usr/bin/env python3
"""One-time SkyNet migration to five public dataset roots plus Supp.

Nothing is deleted: superseded material is moved to a separate recovery root.
Run without --apply to preflight the exact moves first.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dataset_catalog import datasets_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    data = datasets_root()
    archive = args.archive_root.expanduser().resolve()
    if archive == data or data in archive.parents:
        raise ValueError("Recovery archive must be outside the active datasets tree")
    repo = data.parent
    moves: list[tuple[Path, Path]] = [
        (repo / "artifacts/datasets/hocap_v1/canonical", data / "hocap_v1/canonical"),
        (repo / "temp/hocap_mano_replay/data/tasks/models", data / "hocap_v1/assets/objects"),
        (repo / "temp/hocap_mano_replay/data/raw", data / "hocap_v1/raw"),
        (repo / "temp/hocap_mano_replay/data/tasks", data / "hocap_v1/source_tasks"),
        (data / "arctic_v1/canonical_100", data / "arctic_v1/raw/canonical_source"),
        (data / "gigahands_v1/canonical_candidates_v2", data / "gigahands_v1/canonical"),
        (data / "taco_v1/canonical_100", data / "taco_v1/canonical"),
    ]
    corrected = sorted((data / "arctic_v1/canonical_100_tabletop_semantic_v3").glob("*/trajectory.npz"))
    if len(corrected) != 10:
        raise ValueError(f"Expected 10 corrected ARCTIC records, found {len(corrected)}")
    for path in corrected:
        sample = path.parent.name
        base = data / "arctic_v1/canonical_100_tabletop" / sample
        if not (base / "trajectory.npz").is_file():
            raise FileNotFoundError(base)
        moves.append((base, archive / "arctic/tabletop_before_semantic_correction" / sample))
    # Remove the old versions of the corrected samples first, then rename the
    # remaining 90-sample parent, then insert the corrected samples. Every
    # source therefore still exists when its move is executed.
    moves.append((data / "arctic_v1/canonical_100_tabletop", data / "arctic_v1/canonical"))
    moves.extend((path.parent, data / "arctic_v1/canonical" / path.parent.name) for path in corrected)
    archived = [
        "arctic_v1/canonical_100_table_fixed",
        "arctic_v1/canonical_100_tabletop_semantic_v2",
        "arctic_v1/canonical_100_tabletop_semantic_v3",
        "arctic_v1/benchmark_100_table_fixed",
        "arctic_v1/benchmark_100_tabletop",
        "arctic_v1/benchmark_100_tabletop_semantic_v2",
        "arctic_v1/benchmark_100_tabletop_semantic_v3",
        "gigahands_v1/canonical_candidates",
        "taco_v1_candidates500",
        "penspin_v1",
        "bidexhands_v1",
        "_staging",
    ]
    moves.extend((data / name, archive / name) for name in archived if (data / name).exists())
    if os.stat(data).st_dev != os.stat(archive.parent).st_dev:
        raise ValueError("Recovery archive must be on the same filesystem for atomic moves")
    for source, destination in moves:
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        print(f"MOVE {source} -> {destination}")
    if not args.apply:
        print("COPY Bi-DexHands source cube mesh into Supp assets for reproducible re-export")
        print("DRY RUN: pass --apply on SkyNet after reviewing these moves")
        return
    cube_source = data / "bidexhands_v1/assets/objects/source/cube_multicolor.obj"
    cube_target = data / "supp_v1/assets/source/cube_multicolor.obj"
    if not cube_source.is_file() or cube_target.exists():
        raise FileNotFoundError(f"Cannot preserve source cube mesh: {cube_source} -> {cube_target}")
    cube_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cube_source, cube_target)
    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            completed.append((source, destination))
    except Exception:
        for source, destination in reversed(completed):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
        raise
    print("MIGRATION_COMPLETE: active data has five public roots plus Supp")


if __name__ == "__main__":
    main()
