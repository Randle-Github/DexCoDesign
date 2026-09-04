#!/usr/bin/env python3
"""Benchmark MJX GPU stepping with a batch dimension on model parameters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cpu_model = mujoco.MjModel.from_xml_path(str(args.model))
    model = mjx.put_model(cpu_model)
    data = mjx.make_data(model)
    scales = jp.linspace(0.90, 1.10, args.candidates, dtype=jp.float32)
    data_batch = jax.vmap(lambda _: data)(scales)

    # A batch dimension on model parameters is the key capability needed for
    # morphology search. This benchmark varies all non-root body offsets while
    # preserving identical topology and array shapes.
    def one_step(scale, state):
        body_scale = jp.ones((model.body_pos.shape[0], 1), dtype=model.body_pos.dtype)
        body_scale = body_scale.at[1:].set(scale)
        candidate_model = model.replace(body_pos=model.body_pos * body_scale)
        return mjx.step(candidate_model, state)

    batched_step = jax.jit(jax.vmap(one_step, in_axes=(0, 0)))
    state = batched_step(scales, data_batch)
    jax.block_until_ready(state.qpos)
    start = time.perf_counter()
    for _ in range(args.steps):
        state = batched_step(scales, state)
    jax.block_until_ready(state.qpos)
    elapsed = time.perf_counter() - start
    payload = {
        "backend": "mjx",
        "device": str(jax.devices()[0]),
        "candidate_models": args.candidates,
        "steps": args.steps,
        "seconds": elapsed,
        "environment_steps_per_second": args.candidates * args.steps / elapsed,
        "different_model_parameters": True,
        "topology_shared": True,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
