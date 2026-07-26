#!/usr/bin/env python3
"""Export a compact reward curve and copy the final Isaac Lab rollout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def choose_reward_tag(tags: list[str]) -> str:
    preferred = (
        "Reward / Total reward (mean)",
        "Reward/Total",
        "Environment/Return",
        "Episode / Total reward (mean)",
    )
    for tag in preferred:
        if tag in tags:
            return tag
    candidates = [
        tag
        for tag in tags
        if "reward" in tag.lower() or "return" in tag.lower()
    ]
    if not candidates:
        raise RuntimeError(f"No reward/return scalar found. Available scalar tags: {tags}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_files = sorted(args.run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file below {args.run_dir}")
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    tag = choose_reward_tag(accumulator.Tags()["scalars"])
    samples = accumulator.Scalars(tag)
    steps = [sample.step for sample in samples]
    values = [sample.value for sample in samples]

    csv_path = args.output_dir / "reward.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "reward"))
        writer.writerows(zip(steps, values))

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, color="#2563eb", linewidth=2)
    plt.xlabel("training step")
    plt.ylabel(tag)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_dir / "reward.png", dpi=180)
    plt.close()

    videos = sorted((args.run_dir / "videos" / "play").glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No play video below {args.run_dir}")
    shutil.copy2(videos[-1], args.output_dir / "final_rollout.mp4")

    summary = {
        "run_dir": str(args.run_dir),
        "reward_tag": tag,
        "samples": len(samples),
        "first_reward": values[0],
        "last_reward": values[-1],
        "best_reward": max(values),
        "source_video": str(videos[-1]),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
