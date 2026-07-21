#!/usr/bin/env python3
"""Small dense-GCN autoencoder used by the CPU-only morphology experiment."""

from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    """GCN layer for a padded batch of small hand graphs."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0)
        active_pair = mask[:, :, None] * mask[:, None, :]
        a = (adjacency + identity) * active_pair
        degree = a.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        a_hat = inv_sqrt[:, :, None] * a * inv_sqrt[:, None, :]
        result = torch.bmm(a_hat, self.linear(x))
        return torch.nn.functional.gelu(self.norm(result)) * mask[:, :, None]


class HandGraphAutoencoder(nn.Module):
    """Encode a variable-size graph and regress/reconstruct its morphology.

    Canonical BFS node ordering makes the padded reconstruction head meaningful.
    The encoder itself is permutation-equivariant until masked graph pooling.
    """

    def __init__(
        self,
        node_dim: int,
        max_nodes: int,
        target_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.target_dim = target_dim
        self.latent_dim = latent_dim

        self.gcn1 = GraphConv(node_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)
        self.gcn3 = GraphConv(hidden_dim, hidden_dim)
        self.to_latent = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.target_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.node_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_nodes * node_dim),
        )
        self.adjacency_seed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_nodes * 16),
        )

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, adjacency, mask)
        h = self.gcn2(h, adjacency, mask)
        h = self.gcn3(h, adjacency, mask)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = h.sum(dim=1) / count
        masked_h = h.masked_fill(mask[:, :, None] == 0, -1.0e9)
        max_pool = masked_h.max(dim=1).values
        return self.to_latent(torch.cat([mean_pool, max_pool], dim=-1))

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        nodes = self.node_head(latent).view(-1, self.max_nodes, self.node_dim)
        edge_embedding = self.adjacency_seed(latent).view(-1, self.max_nodes, 16)
        adjacency_logits = torch.bmm(edge_embedding, edge_embedding.transpose(1, 2))
        return {
            "target": self.target_head(latent),
            "nodes": nodes,
            "adjacency_logits": adjacency_logits,
        }

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(x, adjacency, mask)
        result = self.decode(latent)
        result["latent"] = latent
        return result

