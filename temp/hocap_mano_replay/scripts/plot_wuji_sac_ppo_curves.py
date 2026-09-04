#!/usr/bin/env python3
"""Plot directly comparable WUJI morphology-SAC and fixed-hand PPO rewards."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    return (
        np.asarray([event.step / 16.0 for event in events], dtype=np.float64),
        np.asarray([event.value for event in events], dtype=np.float64),
    )


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        result[index] = values[max(0, index - window + 1) : index + 1].mean()
    return result


def exact_best_curve(
    log_path: Path, baseline: float, final_iteration: int
) -> tuple[np.ndarray, np.ndarray]:
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    current_step = 0
    points: list[tuple[float, float]] = [(0.0, baseline)]
    best = baseline
    for line in text.splitlines():
        progress = re.search(r"(\d+)/32000", line)
        if progress:
            current_step = int(progress.group(1))
        marker = re.search(
            r"HAND_BEST_ROLLOUT_CAPTURED.*?total_reward=([-+0-9.eE]+)", line
        )
        if marker:
            best = max(best, float(marker.group(1)))
            points.append((current_step / 16.0, best))
    points.append((float(final_iteration), best))
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points])


def sac_statistics(root: Path) -> list[dict[str, float]]:
    rows = []
    for summary_path in sorted(root.glob("generation_*/physx_results.json")):
        summary = json.loads(summary_path.read_text())
        results = summary["results"]
        rewards = np.asarray([row["total_reward"] for row in results])
        rows.append(
            {
                "generation": int(summary_path.parent.name.split("_")[-1]),
                "mean": float(rewards.mean()),
                "median": float(np.median(rewards)),
                "p90": float(np.quantile(rewards, 0.90)),
                "p99": float(np.quantile(rewards, 0.99)),
                "best": float(rewards.max()),
                "success_count": int(sum(row["success"] for row in results)),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-run-dir", type=Path, required=True)
    parser.add_argument("--ppo-log", type=Path, required=True)
    parser.add_argument("--sac-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-reward", type=float, required=True)
    parser.add_argument("--ppo-iterations", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    iteration, reward_mean = scalar(args.ppo_run_dir, "Reward / Total reward (mean)")
    _, reward_max = scalar(args.ppo_run_dir, "Reward / Total reward (max)")
    iteration = np.concatenate(([0.0], iteration))
    reward_mean = np.concatenate(([args.baseline_reward], reward_mean))
    reward_max = np.concatenate(([args.baseline_reward], reward_max))
    best_x, best_y = exact_best_curve(
        args.ppo_log, args.baseline_reward, args.ppo_iterations
    )

    ppo_rows = [
        {
            "iteration": float(x),
            "completed_episode_reward_mean": float(mean),
            "completed_episode_reward_mean_smooth5": float(smooth),
            "completed_episode_reward_max": float(maximum),
        }
        for x, mean, smooth, maximum in zip(
            iteration, reward_mean, rolling_mean(reward_mean), reward_max
        )
    ]
    write_csv(
        args.output_dir / "ppo_reward_curve.csv", list(ppo_rows[0]), ppo_rows
    )
    write_csv(
        args.output_dir / "ppo_exact_best_curve.csv",
        ["iteration", "exact_best_total_reward"],
        [
            {"iteration": float(x), "exact_best_total_reward": float(y)}
            for x, y in zip(best_x, best_y)
        ],
    )

    fig, axis = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    axis.plot(iteration, reward_mean, color="#78a6d8", alpha=0.35, linewidth=1.0)
    axis.plot(
        iteration,
        rolling_mean(reward_mean),
        color="#2676bd",
        linewidth=2.2,
        label="Completed-episode mean (5-point smooth)",
    )
    axis.plot(
        iteration,
        reward_max,
        color="#ef9b31",
        linewidth=1.5,
        alpha=0.8,
        label="Completed-episode max",
    )
    axis.step(
        best_x,
        best_y,
        where="post",
        color="#16865c",
        linewidth=2.4,
        label="Exact captured rollout best-so-far",
    )
    axis.axhline(
        args.baseline_reward,
        color="#7c8188",
        linestyle="--",
        linewidth=1.2,
        label=f"Pure replay baseline = {args.baseline_reward:.2f}",
    )
    axis.scatter([0], [args.baseline_reward], s=55, color="#b32828", zorder=5)
    axis.annotate(
        "pure replay",
        (0, args.baseline_reward),
        xytext=(35, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#7c8188"},
    )
    axis.set(xlabel="PPO iteration", ylabel="Accumulated total reward")
    axis.set_xlim(0, args.ppo_iterations)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", frameon=False)
    fig.savefig(args.output_dir / "ppo_reward_curve.png", dpi=180)
    plt.close(fig)

    sac_rows = sac_statistics(args.sac_root)
    write_csv(
        args.output_dir / "sac_reward_curve.csv", list(sac_rows[0]), sac_rows
    )
    generation = np.asarray([row["generation"] for row in sac_rows])
    fig, (reward_axis, success_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    for key, label, color, width in (
        ("mean", "Population mean", "#2676bd", 2.4),
        ("median", "Median", "#7957b8", 1.6),
        ("p90", "P90", "#ef9b31", 1.8),
        ("p99", "P99", "#d45252", 2.0),
        ("best", "Generation best", "#16865c", 2.0),
    ):
        reward_axis.plot(
            generation,
            [row[key] for row in sac_rows],
            marker="o",
            linewidth=width,
            label=label,
            color=color,
        )
    reward_axis.axhline(
        args.baseline_reward,
        color="#7c8188",
        linestyle="--",
        linewidth=1.2,
        label=f"Pure replay baseline = {args.baseline_reward:.2f}",
    )
    reward_axis.set_ylabel("Total rollout reward")
    reward_axis.grid(alpha=0.2)
    reward_axis.legend(ncol=3, frameon=False)
    success_axis.bar(
        generation,
        [row["success_count"] for row in sac_rows],
        color="#16865c",
        width=0.62,
    )
    success_axis.set(xlabel="Morphology generation", ylabel="Successes")
    success_axis.set_xticks(generation)
    success_axis.grid(axis="y", alpha=0.2)
    fig.savefig(args.output_dir / "sac_reward_curve.png", dpi=180)
    plt.close(fig)

    # Coarse sample-efficiency comparison. One PPO iteration contains 16
    # control steps in every one of 4096 environments. Each morphology
    # generation executes one 446-frame reference rollout in 4096 environments.
    parallel_envs = 4096
    ppo_steps_per_iteration = 16 * parallel_envs
    sac_steps_per_generation = 446 * parallel_envs
    sac_sim_steps = np.concatenate(
        ([0.0], (generation + 1).astype(np.float64) * sac_steps_per_generation)
    )
    sac_best_so_far = np.concatenate(
        (
            [args.baseline_reward],
            np.maximum.accumulate([row["best"] for row in sac_rows]),
        )
    )
    ppo_sim_steps = best_x * ppo_steps_per_iteration
    sac_limit = float(sac_sim_steps[-1])
    plot_limit = 4.0 * sac_limit
    visible = ppo_sim_steps <= plot_limit
    visible_ppo_steps = ppo_sim_steps[visible]
    visible_ppo_reward = best_y[visible]
    if visible_ppo_steps[-1] < plot_limit:
        visible_ppo_steps = np.append(visible_ppo_steps, plot_limit)
        visible_ppo_reward = np.append(visible_ppo_reward, visible_ppo_reward[-1])
    combined_rows = [
        {"algorithm": "SAC morphology", "sim_steps": int(x), "best_reward": float(y)}
        for x, y in zip(sac_sim_steps, sac_best_so_far)
    ] + [
        {"algorithm": "PPO control", "sim_steps": int(x), "best_reward": float(y)}
        for x, y in zip(visible_ppo_steps, visible_ppo_reward)
    ]
    write_csv(
        args.output_dir / "sac_ppo_sim_steps.csv",
        ["algorithm", "sim_steps", "best_reward"],
        combined_rows,
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    axis.step(
        visible_ppo_steps,
        visible_ppo_reward,
        where="post",
        linewidth=2.5,
        color="#2676bd",
        label="PPO control · best-so-far",
    )
    axis.step(
        sac_sim_steps,
        sac_best_so_far,
        where="post",
        linewidth=2.5,
        marker="o",
        color="#16865c",
        label="SAC morphology · best-so-far",
    )
    axis.axhline(
        args.baseline_reward,
        color="#7c8188",
        linestyle="--",
        linewidth=1.2,
        label=f"Pure replay = {args.baseline_reward:.2f}",
    )
    axis.set(
        xlabel="Cumulative simulator transitions",
        ylabel="Best accumulated total reward",
        xlim=(0, plot_limit),
    )
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(args.output_dir / "sac_ppo_sim_steps.png", dpi=180)
    plt.close(fig)

    summary = {
        "pure_replay_baseline": args.baseline_reward,
        "ppo_final_training_mean": float(reward_mean[-1]),
        "ppo_max_training_mean": float(reward_mean.max()),
        "ppo_exact_best": float(best_y.max()),
        "sac_completed_generations": len(sac_rows),
        "sac_latest": sac_rows[-1],
        "sac_best": max(sac_rows, key=lambda row: row["best"]),
        "sac_sim_step_limit": int(sac_limit),
        "combined_plot_sim_step_limit": int(plot_limit),
        "ppo_best_at_plot_limit": float(visible_ppo_reward[-1]),
        "sac_best_at_sac_limit": float(sac_best_so_far[-1]),
    }
    (args.output_dir / "curve_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
