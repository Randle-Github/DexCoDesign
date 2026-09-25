#!/usr/bin/env python3
"""Apply one smoothed iterative-learning update to an ARCTIC hand controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    with np.load(args.trace, allow_pickle=False) as trace:
        phases = trace["phase"].astype(np.int64)
        actual = trace["hand_q_actual"].astype(np.float32)
        target = trace["hand_q_reference"].astype(np.float32)
        trace_names = trace["joint_names"].astype(str).tolist()

    reference_names = payload["joint_names"].astype(str).tolist()
    if set(trace_names) != set(reference_names):
        raise ValueError("Trace and reference joint sets differ")
    trace_order = [trace_names.index(name) for name in reference_names]
    actual = actual[:, trace_order]
    target = target[:, trace_order]
    hand_ctrl = payload["hand_ctrl"].astype(np.float32).copy()
    if actual.shape != target.shape or actual.shape[1] != hand_ctrl.shape[1]:
        raise ValueError(
            f"Incompatible trace shapes: actual={actual.shape}, target={target.shape}, "
            f"control={hand_ctrl.shape}"
        )

    error = np.zeros_like(hand_ctrl)
    counts = np.zeros((len(hand_ctrl), 1), dtype=np.float32)
    np.add.at(error, phases, target - actual)
    np.add.at(counts, phases, 1.0)
    observed = counts[:, 0] > 0
    error[observed] /= counts[observed]
    for column in range(error.shape[1]):
        error[:, column] = np.interp(
            np.arange(len(error)),
            np.flatnonzero(observed),
            error[observed, column],
        )
    smoothed = gaussian_filter1d(error, sigma=args.smooth_sigma, axis=0, mode="nearest")
    correction = args.alpha * smoothed
    correction[:, :3] = np.clip(correction[:, :3], -0.02, 0.02)
    correction[:, 3:6] = np.clip(correction[:, 3:6], -0.10, 0.10)
    correction[:, 6:] = np.clip(correction[:, 6:], -0.20, 0.20)
    payload["hand_ctrl"] = hand_ctrl + correction.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "schema": "dexcodesign.arctic_control_ilc.v1",
        "reference": str(args.reference.resolve()),
        "trace": str(args.trace.resolve()),
        "alpha": args.alpha,
        "smooth_sigma": args.smooth_sigma,
        "mean_abs_tracking_error": float(np.abs(error).mean()),
        "mean_abs_correction": float(np.abs(correction).mean()),
        "max_abs_correction": float(np.abs(correction).max()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "ARCTIC_CONTROL_REFERENCE_REFINED "
        f"mean_error={report['mean_abs_tracking_error']:.9g} "
        f"mean_correction={report['mean_abs_correction']:.9g} "
        f"output={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
