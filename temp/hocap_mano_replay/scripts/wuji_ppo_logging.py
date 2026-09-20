"""Mirror SKRL PPO aggregates to the co-training W&B run."""

import warnings

import numpy as np


def enable_ppo_logging(agent, run, *, generation, steps_per_generation, global_envs):
    """Use SKRL's one-based writer step and preserve its TensorBoard output."""
    if run is None:
        raise RuntimeError("PPO logging requires the outer co-training W&B run")
    run.define_metric("PPO/train_steps")
    run.define_metric("PPO/*", step_metric="PPO/train_steps")
    original_write = agent.write_tracking_data
    warned = False

    def write_tracking_data(*, timestep, timesteps):
        nonlocal warned
        metrics = {}
        for tag, values in agent.tracking_data.items():
            if not len(values):
                continue
            reducer = np.min if tag.endswith("(min)") else np.max if tag.endswith("(max)") else np.mean
            name = tag.replace("Reward / Total reward", "Reward / Episode return")
            metrics[f"PPO/{name}"] = float(reducer(values))
        # The original writer clears tracking_data; aggregate before calling it.
        original_write(timestep=timestep, timesteps=timesteps)
        step = generation * steps_per_generation + timestep
        metrics.update({"PPO/train_steps": step,
                        "PPO/train_transitions": step * global_envs,
                        "PPO/generation": generation})
        try:
            # Do not supply W&B's global step: morphology logs interleave here.
            run.log(metrics)
        except Exception as exc:
            if not warned:
                warnings.warn(f"PPO W&B logging failed: {exc}")
                warned = True

    agent.write_tracking_data = write_tracking_data
