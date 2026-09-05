#!/usr/bin/env python3
"""SKRL SAC adapter for one-step continuous WUJI morphology optimization.

The semantic design action is fully continuous, including palm expansion.
Only the PhysX execution boundary maps expansion to the nearest precompiled
palm collision prototype. This preserves ordering and critic gradients while
keeping exact, precompiled collision geometry during high-throughput search.
Every rollout is a terminal one-step transition, so the SAC Bellman target is
the measured physical return.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

from wuji_general_space import (
    CONTINUOUS_LOWER_BOUNDS,
    CONTINUOUS_SOURCE_VECTOR,
    CONTINUOUS_UPPER_BOUNDS,
    PALM_EXPANSION_LEVELS,
    PALM_EXPANSION_MAX,
    PALM_EXPANSION_MIN,
    SOURCE_VECTOR,
)


ACTION_LOWER_BOUNDS = np.concatenate(
    ([PALM_EXPANSION_MIN], CONTINUOUS_LOWER_BOUNDS)
)
ACTION_UPPER_BOUNDS = np.concatenate(
    ([PALM_EXPANSION_MAX], CONTINUOUS_UPPER_BOUNDS)
)
SOURCE_SEMANTIC_VECTOR = np.concatenate(
    ([PALM_EXPANSION_MIN], CONTINUOUS_SOURCE_VECTOR)
)
ACTION_DIM = len(ACTION_LOWER_BOUNDS)


def normalized_to_semantic(actions: np.ndarray) -> np.ndarray:
    """Map normalized SAC actions to continuous physical design values."""
    span = ACTION_UPPER_BOUNDS - ACTION_LOWER_BOUNDS
    return (
        ACTION_LOWER_BOUNDS + 0.5 * (np.asarray(actions) + 1.0) * span
    ).astype(np.float32)


def semantic_to_prototype_vectors(semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize only palm collision execution; preserve all other values."""
    semantic = np.asarray(semantic, dtype=np.float32)
    normalized = (semantic[:, 0] - PALM_EXPANSION_MIN) / (
        PALM_EXPANSION_MAX - PALM_EXPANSION_MIN
    )
    indices = np.rint(normalized * (PALM_EXPANSION_LEVELS - 1)).astype(np.int64)
    indices = np.clip(indices, 0, PALM_EXPANSION_LEVELS - 1)
    vectors = semantic.copy()
    vectors[:, 0] = indices
    return vectors, indices


@dataclass(frozen=True)
class ProposalBatch:
    """A morphology population and the exact actions that generated it."""

    vectors: np.ndarray
    semantic_vectors: np.ndarray
    observations: torch.Tensor
    actions: torch.Tensor
    palm_indices: np.ndarray
    sample_sources: tuple[str, ...]


def _net(input_name: str) -> list[dict[str, object]]:
    return [
        {
            "name": "net",
            "input": input_name,
            "layers": [256, 256],
            "activations": ["silu", "silu"],
        }
    ]


class SkrlConditionalMorphologySAC:
    """Thin orchestration layer around the unmodified SKRL 2.1 SAC agent."""

    def __init__(
        self,
        *,
        population: int,
        generations: int,
        gradient_steps: int,
        batch_size: int,
        uniform_fraction: float,
        elite_mutation_fraction: float,
        elite_replay_fraction: float,
        elite_mutation_sigma: float,
        reward_scale: float,
        seed: int,
        output_root: Path,
        device: str | torch.device = "cuda",
        wandb: bool = False,
        wandb_project: str = "DexCoDesign",
        wandb_group: str = "wuji-hybrid-sac",
        wandb_run_name: str = "conditional_morphology_sac",
    ) -> None:
        if not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in [0, 1]")
        if not 0.0 <= elite_mutation_fraction <= 1.0:
            raise ValueError("elite_mutation_fraction must be in [0, 1]")
        if not 0.0 <= elite_replay_fraction <= 1.0:
            raise ValueError("elite_replay_fraction must be in [0, 1]")
        if uniform_fraction + elite_mutation_fraction + elite_replay_fraction >= 1.0:
            raise ValueError("proposal mixture fractions must sum to less than 1")
        self.population = int(population)
        self.generations = int(generations)
        self.uniform_fraction = float(uniform_fraction)
        self.elite_mutation_fraction = float(elite_mutation_fraction)
        self.elite_replay_fraction = float(elite_replay_fraction)
        self.elite_mutation_sigma = float(elite_mutation_sigma)
        self.reward_scale = float(reward_scale)
        self.seed = int(seed)
        self.wandb_enabled = bool(wandb)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.archive_actions = np.empty((0, ACTION_DIM), dtype=np.float32)
        self.archive_rewards = np.empty(0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        policy = gaussian_model(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            clip_actions=True,
            clip_mean_actions=True,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=1.0,
            network=_net("OBSERVATIONS"),
            output="ACTIONS",
        )
        critic_1 = deterministic_model(
            observation_space=self.observation_space,
            state_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            network=_net("concatenate([STATES, ACTIONS])"),
            output="ONE",
        )
        critic_2 = deterministic_model(
            observation_space=self.observation_space,
            state_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            network=_net("concatenate([STATES, ACTIONS])"),
            output="ONE",
        )
        target_critic_1 = deterministic_model(
            observation_space=self.observation_space,
            state_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            network=_net("concatenate([STATES, ACTIONS])"),
            output="ONE",
        )
        target_critic_2 = deterministic_model(
            observation_space=self.observation_space,
            state_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            network=_net("concatenate([STATES, ACTIONS])"),
            output="ONE",
        )
        memory = RandomMemory(
            memory_size=self.generations,
            num_envs=self.population,
            device=self.device,
            replacement=False,
        )
        cfg = SAC_CFG()
        cfg.gradient_steps = int(gradient_steps)
        cfg.batch_size = int(batch_size)
        cfg.discount_factor = 0.99  # terminal transitions make this immaterial
        cfg.polyak = 0.005
        cfg.learning_rate = (1.0e-4, 1.0e-4, 1.0e-4)
        cfg.random_timesteps = 0
        cfg.learning_starts = 0
        cfg.learn_entropy = True
        cfg.initial_entropy_value = 0.05
        cfg.target_entropy = -0.5 * float(ACTION_DIM)
        cfg.mixed_precision = False
        cfg.rewards_shaper = (
            lambda rewards, _timestep, _timesteps: rewards * self.reward_scale
        )
        cfg.experiment.directory = str(self.output_root / "skrl_logs")
        cfg.experiment.experiment_name = wandb_run_name
        cfg.experiment.write_interval = 1 if wandb else 0
        cfg.experiment.checkpoint_interval = 0
        cfg.experiment.wandb = bool(wandb)
        cfg.experiment.wandb_kwargs = {
            "project": wandb_project,
            "group": wandb_group,
            "tags": ["WUJI", "morphology", "SAC"],
            # SKRL binds its custom EventFileWriter before wandb.init can patch
            # TensorBoard. Log generation metrics explicitly instead.
            "sync_tensorboard": False,
        }
        self.memory = memory
        self.agent = SAC(
            models={
                "policy": policy,
                "critic_1": critic_1,
                "critic_2": critic_2,
                "target_critic_1": target_critic_1,
                "target_critic_2": target_critic_2,
            },
            memory=memory,
            observation_space=self.observation_space,
            state_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            cfg=cfg,
        )
        # No trainer is needed: the persistent Isaac loop supplies one exact
        # terminal transition batch per generation through the public Agent API.
        self.agent.init()
        self.agent.enable_training_mode(True)

    def propose(self, generation: int) -> ProposalBatch:
        # This is a reward-only, one-step bandit. A constant observation keeps
        # morphology entirely in the ordered continuous action coordinates.
        observations = torch.ones(
            (self.population, 1), dtype=torch.float32, device=self.device
        )
        # Use no_grad rather than inference_mode: sampled actions are inserted
        # into replay and later consumed by critic autograd.
        with torch.no_grad():
            actions, _ = self.agent.act(
                observations,
                observations,
                timestep=generation,
                timesteps=self.generations,
            )
        actions = actions.clamp(-1.0, 1.0)
        sources = np.full(self.population, "skrl_policy", dtype=object)
        random_count = int(round(self.population * self.uniform_fraction))
        rng = np.random.default_rng(self.seed + 100_000 + generation)
        available = np.arange(1, self.population)
        rng.shuffle(available)
        cursor = 0
        if random_count:
            selected = available[cursor : cursor + random_count]
            cursor += random_count
            actions[selected] = torch.from_numpy(
                rng.uniform(-1.0, 1.0, (random_count, ACTION_DIM)).astype(
                    np.float32
                )
            ).to(self.device)
            sources[selected] = "uniform"

        if len(self.archive_actions):
            mutation_count = int(
                round(self.population * self.elite_mutation_fraction)
            )
            replay_count = int(round(self.population * self.elite_replay_fraction))
            mutation_indices = available[cursor : cursor + mutation_count]
            cursor += mutation_count
            replay_indices = available[cursor : cursor + replay_count]
            elite_count = min(16, len(self.archive_actions))
            parent_indices = rng.integers(0, elite_count, size=len(mutation_indices))
            mutated = self.archive_actions[parent_indices] + rng.normal(
                0.0,
                self.elite_mutation_sigma,
                size=(len(mutation_indices), ACTION_DIM),
            ).astype(np.float32)
            actions[mutation_indices] = torch.from_numpy(
                np.clip(mutated, -1.0, 1.0)
            ).to(self.device)
            sources[mutation_indices] = "elite_mutation"
            if len(replay_indices):
                replay_choices = np.arange(len(replay_indices)) % elite_count
                actions[replay_indices] = torch.from_numpy(
                    self.archive_actions[replay_choices]
                ).to(self.device)
                sources[replay_indices] = "elite_replay"

        # Preserve an exact source-hand baseline in every generation.
        observations[0].fill_(1.0)
        source_action = 2.0 * (
            (SOURCE_SEMANTIC_VECTOR - ACTION_LOWER_BOUNDS)
            / (ACTION_UPPER_BOUNDS - ACTION_LOWER_BOUNDS)
        ) - 1.0
        actions[0] = torch.from_numpy(source_action.astype(np.float32)).to(
            self.device
        )
        sources[0] = "source_baseline"
        semantic_vectors = normalized_to_semantic(
            actions.detach().cpu().numpy().astype(np.float32)
        )
        vectors, palms = semantic_to_prototype_vectors(semantic_vectors)
        vectors[0] = SOURCE_VECTOR.astype(np.float32)
        semantic_vectors[0] = SOURCE_SEMANTIC_VECTOR.astype(np.float32)
        return ProposalBatch(
            vectors=vectors,
            semantic_vectors=semantic_vectors,
            observations=observations,
            actions=actions.detach(),
            palm_indices=palms,
            sample_sources=tuple(str(value) for value in sources),
        )

    def observe(
        self, generation: int, proposal: ProposalBatch, rewards: np.ndarray
    ) -> dict[str, object]:
        reward_tensor = torch.from_numpy(
            np.asarray(rewards, dtype=np.float32)[:, None]
        ).to(self.device)
        terminal = torch.ones(
            (self.population, 1), dtype=torch.bool, device=self.device
        )
        truncated = torch.zeros_like(terminal)
        observations = proposal.observations
        self.agent.record_transition(
            observations=observations,
            states=observations,
            actions=proposal.actions,
            rewards=reward_tensor,
            next_observations=observations,
            next_states=observations,
            terminated=terminal,
            truncated=truncated,
            infos={},
            timestep=generation,
            timesteps=self.generations,
        )
        self.agent.post_interaction(
            timestep=generation, timesteps=self.generations
        )
        candidate_actions = proposal.actions.detach().cpu().numpy().astype(np.float32)
        candidate_rewards = np.asarray(rewards, dtype=np.float32)
        all_actions = np.concatenate((self.archive_actions, candidate_actions), axis=0)
        all_rewards = np.concatenate((self.archive_rewards, candidate_rewards), axis=0)
        order = np.argsort(all_rewards)[::-1]
        # Keep a small, deduplicated high-value archive. Exact replay estimates
        # robustness; local mutation supplies dense critic data near good modes.
        kept: list[int] = []
        seen: set[bytes] = set()
        for index in order:
            key = np.round(all_actions[index], 5).tobytes()
            if key in seen:
                continue
            seen.add(key)
            kept.append(int(index))
            if len(kept) == 64:
                break
        self.archive_actions = all_actions[kept]
        self.archive_rewards = all_rewards[kept]
        if self.wandb_enabled:
            try:
                import wandb

                wandb.log(
                    {
                        "Reward / Instantaneous reward (max)": float(
                            candidate_rewards.max()
                        ),
                        "Reward / Instantaneous reward (mean)": float(
                            candidate_rewards.mean()
                        ),
                        "Reward / Instantaneous reward (min)": float(
                            candidate_rewards.min()
                        ),
                        "Morphology / Elite archive best reward": float(
                            self.archive_rewards.max()
                        ),
                        "Morphology / Replay size": int(len(self.memory)),
                    },
                    step=generation,
                )
            except Exception as exc:
                print(
                    f"WARNING: failed to log generation {generation} to W&B: {exc}",
                    flush=True,
                )
        np.savez_compressed(
            self.output_root / "elite_archive.npz",
            actions=self.archive_actions,
            rewards=self.archive_rewards,
        )
        checkpoint = self.output_root / "skrl_sac_agent.pt"
        self.agent.save(str(checkpoint))
        status = {
            "backend": "skrl-2.1-sac",
            "generation": generation,
            "replay_size": int(len(self.memory)),
            "population": self.population,
            "gradient_steps": int(self.agent.cfg.gradient_steps),
            "batch_size": int(self.agent.cfg.batch_size),
            "uniform_fraction": self.uniform_fraction,
            "elite_mutation_fraction": self.elite_mutation_fraction,
            "elite_replay_fraction": self.elite_replay_fraction,
            "elite_mutation_sigma": self.elite_mutation_sigma,
            "elite_archive_size": len(self.archive_actions),
            "reward_scale": self.reward_scale,
            "palm_representation": "continuous_action_quantized_at_physx_boundary",
            "terminal_one_step_transitions": True,
            "checkpoint": str(checkpoint),
        }
        (self.output_root / "skrl_sac_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        return status
