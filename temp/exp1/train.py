#!/usr/bin/env python3
"""Train the right-hand graph autoencoder/regressor on CPU."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import HandGraphAutoencoder


HERE = Path(__file__).resolve().parent
DATASET = HERE / "outputs" / "dataset.json"
CHECKPOINT = HERE / "outputs" / "graph_autoencoder.pt"
METRICS = HERE / "outputs" / "training_metrics.json"


def tensors(payload: dict[str, object]) -> tuple[torch.Tensor, ...]:
    records = payload["records"]
    x = torch.tensor([record["x"] for record in records], dtype=torch.float32)
    adjacency = torch.tensor([record["adjacency"] for record in records], dtype=torch.float32)
    mask = torch.tensor([record["mask"] for record in records], dtype=torch.float32)
    target = torch.tensor([record["target"] for record in records], dtype=torch.float32)
    return x, adjacency, mask, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    records = payload["records"]
    if len(records) != 14 or any(record["side"] != "right" for record in records):
        raise ValueError("Experiment 1 intentionally expects exactly the 14 registered right hands")
    x, adjacency, mask, raw_target = tensors(payload)
    target_mean = raw_target.mean(dim=0)
    target_std = raw_target.std(dim=0).clamp_min(0.025)
    target = (raw_target - target_mean) / target_std

    config = {
        "node_dim": x.shape[-1],
        "max_nodes": x.shape[1],
        "target_dim": target.shape[-1],
        "hidden_dim": 64,
        "latent_dim": 16,
    }
    model = HandGraphAutoencoder(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-4)

    pair_mask = mask[:, :, None] * mask[:, None, :]
    eye = torch.eye(adjacency.shape[-1]).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - eye)
    positive = (adjacency * pair_mask).sum()
    negative = pair_mask.sum() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)

    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        result = model(x, adjacency, mask)
        target_loss = F.mse_loss(result["target"], target)
        node_error = (result["nodes"] - x).square() * mask[:, :, None]
        node_loss = node_error.sum() / (mask.sum() * x.shape[-1])
        edge_raw = F.binary_cross_entropy_with_logits(
            result["adjacency_logits"], adjacency, reduction="none", pos_weight=positive_weight
        )
        edge_loss = (edge_raw * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
        latent_loss = result["latent"].square().mean()
        loss = 6.0 * target_loss + 0.7 * node_loss + 0.12 * edge_loss + 2.0e-4 * latent_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            row = {
                "epoch": epoch,
                "loss": value,
                "target_mse_z": float(target_loss.detach()),
                "node_mse": float(node_loss.detach()),
                "edge_bce": float(edge_loss.detach()),
            }
            history.append(row)
            print(
                f"epoch={epoch:4d} loss={value:.6f} target_z={row['target_mse_z']:.6f} "
                f"node={row['node_mse']:.6f} edge={row['edge_bce']:.6f}",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        result = model(x, adjacency, mask)
        prediction = result["target"] * target_std + target_mean
        absolute_error = (prediction - raw_target).abs()
        rmse = torch.sqrt((prediction - raw_target).square().mean(dim=0))
        edge_prediction = (torch.sigmoid(result["adjacency_logits"]) >= 0.5).float()
        true_positive = (edge_prediction * adjacency * pair_mask).sum()
        predicted_positive = (edge_prediction * pair_mask).sum()
        actual_positive = (adjacency * pair_mask).sum()
        precision = true_positive / predicted_positive.clamp_min(1.0)
        recall = true_positive / actual_positive.clamp_min(1.0)
        edge_f1 = 2 * precision * recall / (precision + recall).clamp_min(1.0e-8)

    checkpoint = {
        "schema_version": 1,
        "model_config": config,
        "model_state": model.state_dict(),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_min": raw_target.min(dim=0).values,
        "target_max": raw_target.max(dim=0).values,
        "embeddings": result["latent"],
        "record_keys": [f"{record['hand_id']}:{record['side']}" for record in records],
        "feature_names": payload["feature_names"],
        "target_names": payload["target_names"],
        "seed": args.seed,
    }
    torch.save(checkpoint, CHECKPOINT)
    metrics = {
        "samples": len(records),
        "sides": ["right"],
        "epochs": args.epochs,
        "best_loss": best_loss,
        "mean_absolute_error": float(absolute_error.mean()),
        "target_rmse": dict(zip(payload["target_names"], (float(x) for x in rmse))),
        "edge_f1": float(edge_f1),
        "history": history,
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {CHECKPOINT}")
    print(f"MAE={metrics['mean_absolute_error']:.6f}, edge_f1={metrics['edge_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

