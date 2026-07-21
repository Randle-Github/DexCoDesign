#!/usr/bin/env python3
"""Sample 100 conservative right-hand designs from the trained latent manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import HandGraphAutoencoder


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset = json.loads((OUTPUTS / "dataset.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(OUTPUTS / "graph_autoencoder.pt", map_location="cpu", weights_only=False)
    model = HandGraphAutoencoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    embeddings = checkpoint["embeddings"].detach().cpu().numpy()
    latent_scale = np.std(embeddings, axis=0)
    raw_targets = np.asarray([record["target"] for record in dataset["records"]], dtype=np.float32)
    target_mean = checkpoint["target_mean"].detach().cpu().numpy()
    target_std = checkpoint["target_std"].detach().cpu().numpy()
    target_min = checkpoint["target_min"].detach().cpu().numpy()
    target_max = checkpoint["target_max"].detach().cpu().numpy()
    target_names = list(checkpoint["target_names"])

    samples = []
    for sample_id in range(args.count):
        anchor = sample_id % len(embeddings)
        partner = int(rng.integers(0, len(embeddings) - 1))
        if partner >= anchor:
            partner += 1
        alpha = float(rng.beta(2.5, 2.5))
        latent = alpha * embeddings[anchor] + (1.0 - alpha) * embeddings[partner]
        latent += rng.normal(0.0, 0.035, size=latent.shape) * np.maximum(latent_scale, 0.05)
        with torch.no_grad():
            standardized = model.decode(torch.tensor(latent[None], dtype=torch.float32))["target"][0].numpy()
        parameters = standardized * target_std + target_mean
        # Never extrapolate far outside the tiny 14-hand training support.
        margin = np.maximum(0.04 * (target_max - target_min), 0.005)
        parameters = np.clip(parameters, target_min - margin, target_max + margin)

        # Balance all source mesh/graph families. The legal graph executor
        # below mutates this decoded template instead of silently collapsing
        # every sample back to the nearest fixed topology.
        template = dataset["records"][anchor]
        param_dict = {name: float(value) for name, value in zip(target_names, parameters)}
        desired_digits = 3 + (sample_id % 4)  # exactly balanced 3/4/5/6-finger designs
        segment_delta = (-1, 0, 1)[sample_id % 3]
        param_dict["digit_count"] = desired_digits
        param_dict["segment_count"] = int(np.clip(template["segment_count"] + segment_delta, 2, 5))
        samples.append(
            {
                "sample_id": sample_id,
                "side": "right",
                "anchor": dataset["records"][anchor]["hand_id"],
                "partner": dataset["records"][partner]["hand_id"],
                "alpha": alpha,
                "topology_template": template["hand_id"],
                "graph_mutation": {
                    "desired_digit_count": desired_digits,
                    "segment_delta": segment_delta,
                    "fixed_to_hinge": sample_id % 3,
                    "hinge_to_fixed": (sample_id // 3) % 3,
                    "base_joint_count": (sample_id // 5) % 3,
                },
                "parameters": param_dict,
                "latent": latent.tolist(),
            }
        )

    output = {
        "schema_version": 1,
        "method": "GCN latent interpolation + neural morphology decoder + balanced source mesh + legal graph grammar mutations",
        "training_samples": len(dataset["records"]),
        "count": len(samples),
        "samples": samples,
    }
    path = OUTPUTS / "generated_hands.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(samples)} right-hand designs: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
