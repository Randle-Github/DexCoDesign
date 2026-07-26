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
        "Info / pose_tracking_reward",
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


def scalar_summary(accumulator: EventAccumulator, tag: str) -> dict[str, float] | None:
    if tag not in accumulator.Tags()["scalars"]:
        return None
    values = [sample.value for sample in accumulator.Scalars(tag)]
    return {
        "first": values[0],
        "last": values[-1],
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-weight", type=float)
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
    auxiliary_tags = {
        "pinch_contact_reward": "Info / pinch_contact_reward",
        "thumb_contact_fraction": "Info / thumb_object_contact",
        "other_finger_contact_fraction": "Info / other_finger_object_contact",
        "thumb_contact_force_n": "Info / thumb_object_contact_force_n",
        "other_finger_contact_force_n": "Info / other_finger_object_contact_force_n",
        "finger_residual_abs_mean_rad": "Info / finger_residual_abs_mean_rad",
        "finger_residual_abs_max_rad": "Info / finger_residual_abs_max_rad",
        "object_position_error_m": "Info / object_position_error_m",
        "object_rotation_error_rad": "Info / object_rotation_error_rad",
    }
    auxiliary = {
        name: value
        for name, tag_name in auxiliary_tags.items()
        if (value := scalar_summary(accumulator, tag_name)) is not None
    }
    if args.contact_weight is not None:
        summary["contact_weight"] = args.contact_weight
        pinch_summary = auxiliary.get("pinch_contact_reward")
        if pinch_summary is not None:
            auxiliary["pinch_contact_fraction"] = {
                key: value / args.contact_weight
                for key, value in pinch_summary.items()
            }
    summary["training_auxiliary"] = auxiliary
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
