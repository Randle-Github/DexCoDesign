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


def scalar_by_step(
    accumulator: EventAccumulator, tag: str
) -> dict[int, float]:
    if tag not in accumulator.Tags()["scalars"]:
        return {}
    return {sample.step: sample.value for sample in accumulator.Scalars(tag)}


def accumulated_pose_return(
    accumulator: EventAccumulator,
) -> tuple[list[int], list[float], str]:
    exact_tag = "Info / pose_tracking_return"
    exact = scalar_by_step(accumulator, exact_tag)
    if exact:
        return (
            list(exact),
            list(exact.values()),
            "exact completed-episode pose return (contact excluded)",
        )

    # Legacy runs only recorded the mean per-step pose reward and the mean
    # completed episode length. Their product is the closest recoverable
    # contact-free accumulated return; label it explicitly as an estimate.
    pose = scalar_by_step(accumulator, "Info / pose_tracking_reward")
    length = scalar_by_step(accumulator, "Episode / Total timesteps (mean)")
    common_steps = sorted(pose.keys() & length.keys())
    if common_steps:
        return (
            common_steps,
            [pose[step] * length[step] for step in common_steps],
            "estimated pose return = mean pose reward × mean episode length (contact excluded)",
        )

    raise RuntimeError(
        "No pose-only accumulated return can be recovered. Available scalar tags: "
        f"{accumulator.Tags()['scalars']}"
    )


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
    steps, values, reward_semantics = accumulated_pose_return(accumulator)

    csv_path = args.output_dir / "reward.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "accumulated_pose_reward"))
        writer.writerows(zip(steps, values))

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, color="#93c5fd", linewidth=1, alpha=0.75, label="raw")
    smoothing_window = min(7, len(values))
    if smoothing_window > 1:
        smoothed = [
            sum(values[max(0, index - smoothing_window + 1) : index + 1])
            / min(index + 1, smoothing_window)
            for index in range(len(values))
        ]
        plt.plot(steps, smoothed, color="#2563eb", linewidth=2.4, label="moving average")
    plt.xlabel("training step")
    plt.ylabel("accumulated pose reward")
    plt.title("Pose-only episode return (contact reward excluded)")
    plt.legend(frameon=False)
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
        "reward_semantics": reward_semantics,
        "samples": len(values),
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
