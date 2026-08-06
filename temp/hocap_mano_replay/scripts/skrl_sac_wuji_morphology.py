#!/usr/bin/env python3
"""SKRL SAC adapter for one-step WUJI morphology optimization.

SKRL's maintained SAC implementation supports continuous ``Box`` actions. The
discrete palm prototype is therefore treated as an observed context selected by
an auditable scheduler. SAC learns only the conditional continuous morphology.
Every exact PhysX rollout is recorded as a terminal one-step transition, so the
standard SAC Bellman target reduces to the measured rollout return.
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

from wuji_morphology_space import (
    CONTINUOUS_LOWER_BOUNDS,
    CONTINUOUS_SOURCE_VECTOR,
    CONTINUOUS_UPPER_BOUNDS,
    PALM_EXPANSION_LEVELS,
    SOURCE_VECTOR,
)


CONTINUOUS_DIM = len(CONTINUOUS_LOWER_BOUNDS)


def normalized_to_physical(
    palm: np.ndarray, continuous: np.ndarray
) -> np.ndarray:
    span = CONTINUOUS_UPPER_BOUNDS - CONTINUOUS_LOWER_BOUNDS
    values = CONTINUOUS_LOWER_BOUNDS + 0.5 * (continuous + 1.0) * span
    return np.concatenate((palm[:, None], values), axis=1).astype(np.float32)


@dataclass(frozen=True)
class ProposalBatch:
    """A morphology population and the exact actions that generated it."""

    vectors: np.ndarray
    observations: torch.Tensor
    actions: torch.Tensor
    palm_indices: np.ndarray
    sample_sources: tuple[str, ...]


class BalancedPalmScheduler:
    """Cover all palm prototypes evenly, with deterministic shuffled ordering."""

    def __init__(self, levels: int, seed: int) -> None:
        self.levels = int(levels)
        self.seed = int(seed)

    def sample(self, count: int, generation: int) -> np.ndarray:
        if count < self.levels:
            raise ValueError(
                f"population {count} cannot cover all {self.levels} palm prototypes"
            )
        base = np.arange(count, dtype=np.int64) % self.levels
        rng = np.random.default_rng(self.seed + generation)
        rng.shuffle(base)
        # Candidate zero is reserved for the exact source baseline. Swap a
        # prototype-zero entry into that slot so coverage remains balanced.
        zero_index = int(np.flatnonzero(base == 0)[0])
        base[0], base[zero_index] = base[zero_index], base[0]
        return base


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
        reward_scale: float,
        seed: int,
        output_root: Path,
        device: str | torch.device = "cuda",
    ) -> None:
        if not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in [0, 1]")
        self.population = int(population)
        self.generations = int(generations)
        self.uniform_fraction = float(uniform_fraction)
        self.reward_scale = float(reward_scale)
        self.seed = int(seed)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.scheduler = BalancedPalmScheduler(PALM_EXPANSION_LEVELS, seed)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(PALM_EXPANSION_LEVELS,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(CONTINUOUS_DIM,),
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
        cfg.learning_rate = (3.0e-4, 3.0e-4, 1.0e-4)
        cfg.random_timesteps = 0
        cfg.learning_starts = 0
        cfg.learn_entropy = True
        cfg.initial_entropy_value = 0.2
        cfg.target_entropy = -float(CONTINUOUS_DIM)
        cfg.mixed_precision = False
        cfg.rewards_shaper = (
            lambda rewards, _timestep, _timesteps: rewards * self.reward_scale
        )
        cfg.experiment.directory = str(self.output_root / "skrl_logs")
        cfg.experiment.experiment_name = "conditional_morphology_sac"
        cfg.experiment.write_interval = 0
        cfg.experiment.checkpoint_interval = 0
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
        palms = self.scheduler.sample(self.population, generation)
        palm_tensor = torch.from_numpy(palms).to(self.device)
        observations = torch.nn.functional.one_hot(
            palm_tensor, PALM_EXPANSION_LEVELS
        ).float()
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
        if random_count:
            selected = rng.choice(self.population, size=random_count, replace=False)
            actions[selected] = torch.from_numpy(
                rng.uniform(-1.0, 1.0, (random_count, CONTINUOUS_DIM)).astype(
                    np.float32
                )
            ).to(self.device)
            sources[selected] = "uniform"

        # Preserve an exact source-hand baseline in every generation.
        observations[0].zero_()
        observations[0, palms[0]] = 1.0
        source_action = 2.0 * (
            (CONTINUOUS_SOURCE_VECTOR - CONTINUOUS_LOWER_BOUNDS)
            / (CONTINUOUS_UPPER_BOUNDS - CONTINUOUS_LOWER_BOUNDS)
        ) - 1.0
        actions[0] = torch.from_numpy(source_action.astype(np.float32)).to(
            self.device
        )
        sources[0] = "source_baseline"
        vectors = normalized_to_physical(
            palms, actions.detach().cpu().numpy().astype(np.float32)
        )
        vectors[0] = SOURCE_VECTOR.astype(np.float32)
        return ProposalBatch(
            vectors=vectors,
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
            "reward_scale": self.reward_scale,
            "palm_scheduler": "balanced",
            "terminal_one_step_transitions": True,
            "checkpoint": str(checkpoint),
        }
        (self.output_root / "skrl_sac_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        return status
