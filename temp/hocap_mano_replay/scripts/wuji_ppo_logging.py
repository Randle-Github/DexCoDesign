"""Mirror SKRL PPO aggregates to the co-training W&B run."""

import warnings

import numpy as np
import torch


class CompletedEpisodeWindow:
    """Episode-weighted training statistics over persistent, nonoverlapping windows.

    Only completed episodes enter the means. Window totals survive generation
    boundaries; per-environment unfinished episodes do not. All ranks must call
    observe/finish_generation/flush together when distributed=True.
    """

    def __init__(self, *, window_steps=1600, global_envs, log=None, distributed=False):
        if window_steps < 1:
            raise ValueError("window_steps must be positive")
        self.window_steps = window_steps
        self.global_envs = global_envs
        self.log = log
        self.distributed = distributed
        self.total_steps = 0
        self.pending_steps = 0
        self.returns = None
        self.lengths = None
        self.totals = None  # return sum, length sum, episode count, interrupted count
        self.warned = False

    @torch.no_grad()
    def observe(self, rewards, terminated, truncated):
        rewards = rewards.detach().reshape(-1)
        if self.returns is None:
            self.returns = torch.zeros_like(rewards, dtype=torch.float64)
            self.lengths = torch.zeros_like(rewards, dtype=torch.long)
        if self.totals is None:
            self.totals = torch.zeros(4, device=rewards.device, dtype=torch.float64)
        self.returns.add_(rewards)
        self.lengths.add_(1)
        done = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
        self.totals[0] += self.returns[done].sum()
        self.totals[1] += self.lengths[done].sum()
        self.totals[2] += done.sum()
        self.returns[done] = 0
        self.lengths[done] = 0
        self.total_steps += 1
        self.pending_steps += 1
        if self.pending_steps == self.window_steps:
            self.flush()

    @torch.no_grad()
    def finish_generation(self):
        if self.lengths is not None:
            self.totals[3] += (self.lengths > 0).sum()
        # Evaluation resets the simulation. Never join two episodes across it.
        self.returns = self.lengths = None

    @torch.no_grad()
    def flush(self):
        if self.totals is None:
            return
        totals = self.totals.clone()
        if self.distributed:
            torch.distributed.all_reduce(totals)
        return_sum, length_sum, count, interrupted = totals.cpu().tolist()
        if self.log is not None and (self.pending_steps or interrupted):
            prefix = "PPO/Completed episodes / "
            metrics = {
                "PPO/train_steps": self.total_steps,
                "PPO/train_transitions": self.total_steps * self.global_envs,
                prefix + "Interrupted episodes (total)": int(interrupted),
            }
            if self.pending_steps:
                metrics[prefix + "Count (window)"] = int(count)
                metrics[prefix + "Window steps"] = self.pending_steps
            # No completed episodes means no estimate, not a zero return.
            if count:
                metrics[prefix + "Return mean (window)"] = return_sum / count
                metrics[prefix + "Length mean (window)"] = length_sum / count
            try:
                self.log(metrics)
            except Exception as exc:
                if not self.warned:
                    warnings.warn(f"PPO completed-episode W&B logging failed: {exc}")
                    self.warned = True
        self.totals[:3].zero_()
        self.pending_steps = 0


def track_completed_episodes(agent, window):
    """Observe raw training rewards before reward shaping or PPO bootstrapping."""
    original_record = agent.record_transition

    def record_transition(**kwargs):
        window.observe(kwargs["rewards"], kwargs["terminated"], kwargs["truncated"])
        return original_record(**kwargs)

    agent.record_transition = record_transition


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
