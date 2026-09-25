#!/usr/bin/env python3
"""Blend a validated grasp residual into an ARCTIC hand-control reference."""

import argparse
import json
from pathlib import Path

import numpy as np


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("residual", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fade-in-start", type=int, default=30)
    parser.add_argument("--hold-start", type=int, default=52)
    parser.add_argument("--hold-end", type=int, default=162)
    parser.add_argument("--fade-out-end", type=int, default=210)
    parser.add_argument(
        "--blend-mode", choices=("smoothstep", "linear"), default="smoothstep"
    )
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    residual = json.loads(args.residual.read_text(encoding="utf-8"))
    joint_names = payload["joint_names"].tolist()
    control = payload["hand_ctrl"].copy()
    frames = np.arange(len(control), dtype=np.float32)
    blend_curve = smoothstep if args.blend_mode == "smoothstep" else lambda x: np.clip(x, 0.0, 1.0)
    fade_in = blend_curve(
        (frames - args.fade_in_start) / max(args.hold_start - args.fade_in_start, 1)
    )
    fade_out = 1.0 - blend_curve(
        (frames - args.hold_end) / max(args.fade_out_end - args.hold_end, 1)
    )
    blend = np.minimum(fade_in, fade_out).astype(np.float32)
    scale = float(residual["residual_finger_scale"])
    for name, action in zip(
        residual["finger_action_names"], residual["finger_actions"], strict=True
    ):
        control[:, joint_names.index(name)] += blend * scale * float(action)
    payload["hand_ctrl"] = control.astype(np.float32)
    payload["grasp_residual_blend"] = blend
    payload["grasp_residual_metadata_json"] = np.asarray(
        json.dumps(
            {
                "residual": str(args.residual),
                "fade_in_start": args.fade_in_start,
                "hold_start": args.hold_start,
                "hold_end": args.hold_end,
                "fade_out_end": args.fade_out_end,
                "blend_mode": args.blend_mode,
            }
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"ARCTIC_GRASP_REFERENCE_WRITTEN path={args.output} frames={len(control)}")


if __name__ == "__main__":
    main()
