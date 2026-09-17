"""Load reference arrays without retaining archive file descriptors."""

from pathlib import Path
from collections.abc import Sequence

import numpy as np


def reference_layout(manifest: dict | None, count: int) -> tuple[list[int], list[int]]:
    """Return representative environment indices and an environment-to-bank map.

    Only explicitly replicated morphologies share banks. Reject inconsistent
    replica metadata rather than silently using another hand's reference/FK.
    """
    if manifest is None or not manifest.get("grouped_physics_replication", False):
        return list(range(count)), list(range(count))
    labels = manifest["morphology_indices"]
    if len(labels) != count:
        raise ValueError("morphology mapping length differs from reference count")
    fields = [key for key, value in manifest.items()
              if key in ("reference_paths", "hand_usd_paths", "hand_urdf_paths", "vectors")
              or key.startswith("parametric_") and isinstance(value, list)]
    for key in fields:
        if len(manifest[key]) != count:
            raise ValueError(f"replicated morphology metadata length mismatch: {key}")
    representatives, mapping, bank = [], [], {}
    for index, label in enumerate(labels):
        if label not in bank:
            bank[label] = len(representatives)
            representatives.append(index)
        slot = bank[label]
        first = representatives[slot]
        for key in fields:
            if manifest[key][index] != manifest[key][first]:
                raise ValueError(f"replicas of morphology {label} disagree on {key}")
        mapping.append(slot)
    return representatives, mapping


def load_references(paths: Sequence[str | Path]) -> list[dict[str, np.ndarray]]:
    """Preserve replica order while loading each unique NPZ only once.

    Replicas share the materialized arrays, which callers must treat as read-only.
    The cache is local to this call so later environments reload updated files.
    """
    cache: dict[Path, dict[str, np.ndarray]] = {}
    references = []
    for path in paths:
        key = Path(path).expanduser().resolve()
        if key not in cache:
            with np.load(key, allow_pickle=False) as archive:
                cache[key] = {name: archive[name] for name in archive.files}
        references.append(cache[key])
    return references
