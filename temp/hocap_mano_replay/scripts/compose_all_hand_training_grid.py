#!/usr/bin/env python3
"""Compose all captured Isaac hand rollouts into the reviewed 4x4 layout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


HAND_ORDER = (
    "mano",
    "ability_hand",
    "schunk_svh",
    "wuji_hand_2",
    "sharpa_wave_01",
    "tesollo_dg5f",
    "unitree_dex5_1",
    "robotera_xhand1",
    "orca_hand_v2",
    "shadow_hand_e",
    "allegro_hand_v5",
    "midas_hand",
    "ruka_v2",
    "inspire_rh56dfx",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    )
    args = parser.parse_args()

    inputs: list[str] = []
    filters: list[str] = []
    summary: list[dict[str, object]] = []
    for index, hand_id in enumerate(HAND_ORDER):
        hand_root = args.result_root / hand_id
        status = json.loads(
            (hand_root / "training_status.json").read_text(encoding="utf-8")
        )
        metadata = status["selected_rollout_metadata"]
        video = Path(status["selected_rollout"]).with_suffix(".mp4").name
        video_path = hand_root / video
        if not video_path.is_file():
            video_path = hand_root / "farthest_rollout.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        success = bool(status["success"])
        phase = int(metadata["final_phase"])
        last_phase = int(metadata["reference_last_phase"])
        state = "SUCCESS" if success else "FAIL"
        text_color = "0x70ff70" if success else "0xffb050"
        label = f"{hand_id} | {state} | {phase}/{last_phase}"
        inputs.extend(("-i", str(video_path)))
        chain = (
            f"[{index}:v]fps=30,"
            "scale=480:360:force_original_aspect_ratio=decrease,"
            "pad=480:360:(ow-iw)/2:(oh-ih)/2:black,"
            "tpad=stop_mode=clone:stop_duration=15,"
            "trim=end_frame=446,setpts=PTS-STARTPTS,"
            "drawbox=x=0:y=0:w=iw:h=44:color=black@0.72:t=fill,"
            f"drawtext=fontfile='{args.font}':text='{label}':"
            f"fontcolor={text_color}:fontsize=20:x=10:y=11"
        )
        if hand_id == "mano":
            chain += ",drawbox=x=2:y=2:w=iw-4:h=ih-4:color=red:t=8"
        filters.append(chain + f"[v{index}]")
        summary.append(
            {
                "hand_id": hand_id,
                "success": success,
                "final_phase": phase,
                "reference_last_phase": last_phase,
                "video": str(video_path.resolve()),
            }
        )

    filters.extend(
        (
            "color=c=black:s=480x360:r=30:d=14.8666667[blank14]",
            "color=c=black:s=480x360:r=30:d=14.8666667[blank15]",
        )
    )
    layout = "|".join(f"{(i % 4) * 480}_{(i // 4) * 360}" for i in range(16))
    stack_inputs = "".join(f"[v{i}]" for i in range(14)) + "[blank14][blank15]"
    filters.append(
        f"{stack_inputs}xstack=inputs=16:layout={layout}:fill=black[outv]"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outv]",
        "-frames:v",
        "446",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
