"""Package fresh Torch IK with the original WUJI USD and reference topology."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from prepare_wuji_parametric_training_assets import save_reference


def prepare_original_source_hand(retarget_path, task_targets_path, output, repo_root):
    start = time.perf_counter()
    output, repo_root = Path(output), Path(repo_root)
    asset_root = repo_root / "artifacts/isaaclab_all_hands_residual"
    usd = asset_root / "assets/wuji_hand_2/hand.usd"
    urdf = asset_root / "prepared/wuji_hand_2/hand_rl.urdf"
    template = asset_root / "prepared/wuji_hand_2/reference.npz"
    for path in (usd, urdf, template):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(retarget_path, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        if vectors.shape != (1, 23) or np.any(vectors != 0):
            raise ValueError("Original WUJI assets require exactly one zero source vector")
        with np.load(template, allow_pickle=False) as source:
            expected = [str(name).removeprefix("finger__") for name in source["joint_names"][6:]]
        actual = data["joint_names"].tolist()
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ValueError(f"Torch/source joint names disagree: {actual} vs {expected}")
        # Match by name, since references store the original articulation order.
        order = [actual.index(name) for name in expected]
        reference = output / "reference.npz"
        save_reference(
            template, reference, "wuji_hand_2", data["qpos"][0][:, order],
            data["wrist_position_all"][0], data["wrist_quaternion_xyzw_all"][0],
            task_targets=Path(task_targets_path),
        )
    manifest = {
        "schema_version": 3,
        "original_source_hand": True,
        "candidate_ids": ["wuji_physx_000000"],
        "vectors": vectors.tolist(),
        "hand_usd_paths": [str(usd.resolve())],
        "hand_urdf_paths": [str(urdf.resolve())],
        "reference_paths": [str(reference.resolve())],
        "runtime_parametric_overlays": False,
    }
    (output / "physx_batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ORIGINAL_SOURCE_HAND] usd={usd} reference={reference} asset_generation=skipped", flush=True)
    return manifest, {"asset_generation_skipped": True, "reference_prepare_seconds": time.perf_counter() - start}
