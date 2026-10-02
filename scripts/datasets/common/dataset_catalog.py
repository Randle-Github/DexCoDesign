"""One read interface for the active DexCoDesign pose datasets.

Raw downloads and simulator-specific retargeted references are not trajectories
in this catalog. Each public dataset has one active ``canonical`` directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from trajectory_schema import audit_trajectory


DATASET_NAMES = ("hocap", "arctic", "taco", "gigahands", "dexterhand", "supp")
REPO_ROOT = Path(__file__).resolve().parents[3]


def datasets_root() -> Path:
    """Allow SkyNet data to live outside a checkout without changing callers."""
    return Path(os.environ.get("DEXCODESIGN_DATASETS_ROOT", REPO_ROOT / "datasets")).expanduser().resolve()


def dataset_root(dataset: str) -> Path:
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_NAMES}")
    return datasets_root() / f"{dataset}_v1"


def load_index(dataset: str) -> dict:
    path = dataset_root(dataset) / "index.json"
    index = json.loads(path.read_text())
    if index.get("schema") != "dexcodesign.dataset_index.v1" or index.get("dataset") != dataset:
        raise ValueError(f"Invalid dataset index: {path}")
    return index


def sample_record(dataset: str, sample_id: str) -> dict:
    for record in load_index(dataset)["samples"]:
        if record["sample_id"] == sample_id:
            return record
    raise KeyError(f"{dataset}/{sample_id} is not in the active index")


def load_sample(dataset: str, sample_id: str) -> tuple[dict, dict]:
    """Return arrays and record; units/frame conventions are shared by all datasets."""
    record = sample_record(dataset, sample_id)
    path = dataset_root(dataset) / record["trajectory"]
    audit = audit_trajectory(path)
    if not audit["ok"]:
        raise ValueError(f"{path}: {audit['error']}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
        arrays["metadata"] = json.loads(str(data["metadata_json"]))
    return arrays, record


def iter_samples(dataset: str):
    """Yield the same record structure regardless of the source dataset."""
    yield from load_index(dataset)["samples"]
