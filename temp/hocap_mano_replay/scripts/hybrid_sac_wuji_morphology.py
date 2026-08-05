#!/usr/bin/env python3
"""Episode-level hybrid SAC for WUJI morphology selection.

The task is a one-step contextual bandit: select one morphology, execute the
complete physical reference rollout, and observe its exact terminal return.
Palm expansion is categorical; the remaining dimensions are continuous.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from wuji_morphology_space import (
    CONTINUOUS_LOWER_BOUNDS,
    CONTINUOUS_UPPER_BOUNDS,
    PALM_EXPANSION_LEVELS,
    SOURCE_VECTOR,
    VECTOR_NAMES,
    validate_design_vectors,
)


CONTINUOUS_DIM = len(CONTINUOUS_LOWER_BOUNDS)
CRITIC_INPUT_DIM = PALM_EXPANSION_LEVELS + CONTINUOUS_DIM


def mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, output_dim),
    )


class HybridActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.palm_logits = nn.Parameter(torch.zeros(PALM_EXPANSION_LEVELS))
        self.continuous = mlp(
            PALM_EXPANSION_LEVELS, 2 * CONTINUOUS_DIM
        )

    def distribution(
        self, palm_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = torch.nn.functional.one_hot(
            palm_index, PALM_EXPANSION_LEVELS
        ).float()
        mean, log_std = self.continuous(one_hot).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 1.0)

    def sample(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        palm = Categorical(logits=self.palm_logits).sample((count,))
        mean, log_std = self.distribution(palm)
        normal = Normal(mean, log_std.exp())
        latent = normal.rsample()
        continuous = torch.tanh(latent)
        log_prob = normal.log_prob(latent) - torch.log(
            1.0 - continuous.square() + 1.0e-6
        )
        log_prob = log_prob.sum(-1) + torch.log_softmax(
            self.palm_logits, dim=0
        )[palm]
        return palm, continuous, log_prob


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = mlp(CRITIC_INPUT_DIM, 1)

    def forward(
        self, palm: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(
            palm, PALM_EXPANSION_LEVELS
        ).float()
        return self.network(torch.cat((one_hot, continuous), dim=-1)).squeeze(-1)


def physical_to_normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = validate_design_vectors(vectors)
    span = CONTINUOUS_UPPER_BOUNDS - CONTINUOUS_LOWER_BOUNDS
    return np.clip(
        2.0 * (vectors[:, 1:] - CONTINUOUS_LOWER_BOUNDS) / span - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)


def normalized_to_physical(
    palm: np.ndarray, continuous: np.ndarray
) -> np.ndarray:
    span = CONTINUOUS_UPPER_BOUNDS - CONTINUOUS_LOWER_BOUNDS
    values = CONTINUOUS_LOWER_BOUNDS + 0.5 * (continuous + 1.0) * span
    return np.concatenate((palm[:, None], values), axis=1).astype(np.float32)


def load_replay(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, CONTINUOUS_DIM), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    with np.load(path) as data:
        return (
            data["palm_index"].astype(np.int64),
            data["continuous_normalized"].astype(np.float32),
            data["reward"].astype(np.float32),
        )


def append_exact_results(
    replay: tuple[np.ndarray, np.ndarray, np.ndarray], summary_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("all_candidates_physically_evaluated", False):
        raise ValueError("hybrid SAC accepts only complete physical evaluations")
    if summary.get("proxy_used", True) or summary.get("top_k_prefilter_used", True):
        raise ValueError("proxy/prefiltered rewards cannot enter hybrid SAC replay")
    rows = list(summary["results"])
    expected = int(summary["candidate_count"])
    if len(rows) != expected or int(summary["completed"]) != expected:
        raise ValueError(f"incomplete physical batch: {len(rows)}/{expected}")
    vectors = validate_design_vectors(
        np.asarray([row["vector"] for row in rows], dtype=np.float64)
    )
    rewards = np.asarray([row["total_reward"] for row in rows], dtype=np.float32)
    old_palm, old_continuous, old_reward = replay
    return (
        np.concatenate((old_palm, vectors[:, 0].astype(np.int64))),
        np.concatenate((old_continuous, physical_to_normalized(vectors)), axis=0),
        np.concatenate((old_reward, rewards)),
    )


def save_replay(
    path: Path, replay: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        palm_index=replay[0],
        continuous_normalized=replay[1],
        reward=replay[2],
    )


def actor_loss_exact_categories(
    actor: HybridActor,
    critic1: Critic,
    critic2: Critic,
    alpha_discrete: torch.Tensor,
    alpha_continuous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    palm = torch.arange(PALM_EXPANSION_LEVELS, device=actor.palm_logits.device)
    probability = torch.softmax(actor.palm_logits, dim=0)
    log_probability = torch.log_softmax(actor.palm_logits, dim=0)
    mean, log_std = actor.distribution(palm)
    normal = Normal(mean, log_std.exp())
    latent = normal.rsample()
    continuous = torch.tanh(latent)
    continuous_log_probability = (
        normal.log_prob(latent)
        - torch.log(1.0 - continuous.square() + 1.0e-6)
    ).sum(-1)
    q = torch.minimum(
        critic1(palm, continuous), critic2(palm, continuous)
    )
    objective = (
        alpha_discrete * log_probability
        + alpha_continuous * continuous_log_probability
        - q
    )
    loss = (probability * objective).sum()
    expected_continuous_log_probability = (
        probability.detach() * continuous_log_probability
    ).sum()
    expected_discrete_log_probability = (
        probability.detach() * log_probability
    ).sum()
    return loss, expected_discrete_log_probability, expected_continuous_log_probability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-vectors", type=Path, required=True)
    parser.add_argument("--exact-summary", type=Path)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--uniform-fraction", type=float, default=0.15)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.population < 2:
        parser.error("--population must be at least 2")
    if not 0.0 <= args.uniform_fraction <= 1.0:
        parser.error("--uniform-fraction must lie in [0, 1]")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    actor, critic1, critic2 = HybridActor().to(device), Critic().to(device), Critic().to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3.0e-4)
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=3.0e-4
    )
    log_alpha_discrete = torch.tensor(
        math.log(0.2), device=device, requires_grad=True
    )
    log_alpha_continuous = torch.tensor(
        math.log(0.2), device=device, requires_grad=True
    )
    alpha_optimizer = torch.optim.Adam(
        [log_alpha_discrete, log_alpha_continuous], lr=1.0e-4
    )
    generation = 0
    if args.state.is_file():
        state = torch.load(args.state, map_location=device, weights_only=False)
        actor.load_state_dict(state["actor"])
        critic1.load_state_dict(state["critic1"])
        critic2.load_state_dict(state["critic2"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        log_alpha_discrete.data.copy_(state["log_alpha_discrete"].to(device))
        log_alpha_continuous.data.copy_(state["log_alpha_continuous"].to(device))
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        generation = int(state["generation"])

    replay = load_replay(args.replay)
    if args.exact_summary is not None:
        replay = append_exact_results(replay, args.exact_summary)
        generation += 1
        save_replay(args.replay, replay)

    update_start = time.perf_counter()
    losses: list[tuple[float, float]] = []
    if len(replay[2]) >= max(64, min(args.batch_size, 256)):
        replay_palm = torch.from_numpy(replay[0]).to(device)
        replay_continuous = torch.from_numpy(replay[1]).to(device)
        replay_reward = torch.from_numpy(replay[2]).to(device) * args.reward_scale
        batch_size = min(args.batch_size, len(replay_reward))
        for _ in range(args.updates):
            indices = torch.randint(len(replay_reward), (batch_size,), device=device)
            palm = replay_palm[indices]
            continuous = replay_continuous[indices]
            target = replay_reward[indices]
            q1, q2 = critic1(palm, continuous), critic2(palm, continuous)
            critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss, logp_discrete, logp_continuous = actor_loss_exact_categories(
                actor,
                critic1,
                critic2,
                log_alpha_discrete.exp().detach(),
                log_alpha_continuous.exp().detach(),
            )
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()

            target_discrete_entropy = 0.80 * math.log(PALM_EXPANSION_LEVELS)
            target_continuous_entropy = float(CONTINUOUS_DIM)
            alpha_loss = (
                log_alpha_discrete
                * (-logp_discrete - target_discrete_entropy).detach()
                + log_alpha_continuous
                * (-logp_continuous - target_continuous_entropy).detach()
            )
            alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_optimizer.step()
            losses.append((float(critic_loss.item()), float(actor_loss.item())))
    update_seconds = time.perf_counter() - update_start

    sample_start = time.perf_counter()
    actor_count = int(round(args.population * (1.0 - args.uniform_fraction)))
    random_count = args.population - actor_count
    with torch.inference_mode():
        palm, continuous, _ = actor.sample(actor_count)
    palm_np = palm.cpu().numpy().astype(np.int64)
    continuous_np = continuous.cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(args.seed + generation)
    if random_count:
        palm_np = np.concatenate(
            (palm_np, rng.integers(0, PALM_EXPANSION_LEVELS, random_count))
        )
        continuous_np = np.concatenate(
            (
                continuous_np,
                rng.uniform(-1.0, 1.0, (random_count, CONTINUOUS_DIM)).astype(
                    np.float32
                ),
            ),
            axis=0,
        )
    order = rng.permutation(args.population)
    vectors = normalized_to_physical(palm_np[order], continuous_np[order])
    vectors[0] = SOURCE_VECTOR.astype(np.float32)
    args.output_vectors.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_vectors, vectors)
    sample_seconds = time.perf_counter() - sample_start

    args.state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "generation": generation,
            "vector_names": VECTOR_NAMES,
            "actor": actor.state_dict(),
            "critic1": critic1.state_dict(),
            "critic2": critic2.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "log_alpha_discrete": log_alpha_discrete.detach().cpu(),
            "log_alpha_continuous": log_alpha_continuous.detach().cpu(),
            "alpha_optimizer": alpha_optimizer.state_dict(),
        },
        args.state,
    )
    status = {
        "schema_version": 1,
        "generation": generation,
        "replay_size": len(replay[2]),
        "population": args.population,
        "uniform_fraction": args.uniform_fraction,
        "updates": len(losses),
        "update_seconds": update_seconds,
        "sample_seconds": sample_seconds,
        "alpha_discrete": float(log_alpha_discrete.exp().item()),
        "alpha_continuous": float(log_alpha_continuous.exp().item()),
        "last_critic_loss": None if not losses else losses[-1][0],
        "last_actor_loss": None if not losses else losses[-1][1],
        "reward_best": None if not len(replay[2]) else float(replay[2].max()),
        "reward_mean": None if not len(replay[2]) else float(replay[2].mean()),
    }
    args.output_vectors.with_suffix(".json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print("WUJI_HYBRID_SAC " + json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
