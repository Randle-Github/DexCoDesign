#!/usr/bin/env python3
"""Simulate and visualize the closed-loop MIDAS whole hand MJCF."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict

import mujoco
import mujoco.viewer
import numpy as np


FINGERS = ("index", "middle", "ring")
FINGER_PHASE = {
    "index": 0.0,
    "middle": 0.7,
    "ring": 1.4,
}
FINGER_ABAD = {
    "index": -1.0,
    "middle": 0.0,
    "ring": 1.0,
}
THUMB_OPEN_TARGETS = {
    "thumb_cmc_roll_joint": 0.25,
    "thumb_cmc_side_joint": -0.30,
    "thumb_mcp_joint": -0.10,
    "thumb_dip_joint": 0.02,
}
THUMB_OPPOSE_TARGETS = {
    "thumb_cmc_roll_joint": 1.30,
    "thumb_cmc_side_joint": 0.42,
    "thumb_mcp_joint": -0.72,
    "thumb_dip_joint": -0.46,
}

FINGER_ACTIVE_JOINTS = [
    f"{finger}_{joint}"
    for finger in FINGERS
    for joint in ("mcp_abad_joint", "mcp_pitch_joint", "pip_joint")
]
THUMB_ACTIVE_JOINTS = [
    "thumb_cmc_roll_joint",
    "thumb_cmc_side_joint",
    "thumb_mcp_joint",
    "thumb_dip_joint",
]
ACTIVE_JOINTS = FINGER_ACTIVE_JOINTS + THUMB_ACTIVE_JOINTS


def name2id(model: mujoco.MjModel, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"Could not find {name!r} in the MuJoCo model.")
    return obj_id


def find_actuator_for_joint(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id:
            return actuator_id
    raise RuntimeError(f"No actuator found for joint {joint_name!r}.")


def joint_range(model: mujoco.MjModel, joint_name: str) -> tuple[float, float]:
    joint_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    lower, upper = model.jnt_range[joint_id]
    return float(lower), float(upper)


def clip_joint(model: mujoco.MjModel, joint_name: str, value: float) -> float:
    lower, upper = joint_range(model, joint_name)
    return float(np.clip(value, lower, upper))


def smoothstep(amount: float) -> float:
    amount = float(np.clip(amount, 0.0, 1.0))
    return amount * amount * (3.0 - 2.0 * amount)


def blend(a: float, b: float, amount: float) -> float:
    amount = float(np.clip(amount, 0.0, 1.0))
    return a + amount * (b - a)


def limit_blend(model: mujoco.MjModel, joint_name: str, amount: float) -> float:
    """Blend from the upper/open limit to the lower/closed limit."""

    lower, upper = joint_range(model, joint_name)
    amount = float(np.clip(amount, 0.0, 1.0))
    return upper + amount * (lower - upper)


def signed_limit_blend(model: mujoco.MjModel, joint_name: str, amount: float, sign: float) -> float:
    lower, upper = joint_range(model, joint_name)
    amount = float(np.clip(amount, 0.0, 1.0))
    if sign < 0.0:
        return amount * lower
    if sign > 0.0:
        return amount * upper
    return 0.0


def thumb_opposition_targets(
    model: mujoco.MjModel,
    base_amount: float,
) -> Dict[str, float]:
    """Human-like thumb opposition with CMC leading MCP/DIP curl."""

    cmc_amount = smoothstep(base_amount)
    curl_amount = smoothstep((base_amount - 0.18) / 0.82)
    targets = {
        "thumb_cmc_roll_joint": blend(
            THUMB_OPEN_TARGETS["thumb_cmc_roll_joint"],
            THUMB_OPPOSE_TARGETS["thumb_cmc_roll_joint"],
            cmc_amount,
        ),
        "thumb_cmc_side_joint": blend(
            THUMB_OPEN_TARGETS["thumb_cmc_side_joint"],
            THUMB_OPPOSE_TARGETS["thumb_cmc_side_joint"],
            cmc_amount,
        ),
        "thumb_mcp_joint": blend(
            THUMB_OPEN_TARGETS["thumb_mcp_joint"],
            THUMB_OPPOSE_TARGETS["thumb_mcp_joint"],
            curl_amount,
        ),
        "thumb_dip_joint": blend(
            THUMB_OPEN_TARGETS["thumb_dip_joint"],
            THUMB_OPPOSE_TARGETS["thumb_dip_joint"],
            curl_amount,
        ),
    }
    return {joint: clip_joint(model, joint, target) for joint, target in targets.items()}


def build_actuator_map(model: mujoco.MjModel) -> Dict[str, int]:
    return {joint_name: find_actuator_for_joint(model, joint_name) for joint_name in ACTIVE_JOINTS}


def apply_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: Dict[str, int],
    targets: Dict[str, float],
) -> None:
    for joint_name, target in targets.items():
        data.ctrl[actuator_id[joint_name]] = clip_joint(model, joint_name, target)


def motion_targets(
    model: mujoco.MjModel,
    sim_time: float,
    *,
    sine: bool,
    manual_flexion: float,
    amp_rad: float,
    offset_rad: float,
    freq_hz: float,
    thumb_enabled: bool,
) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    _, pip_open = joint_range(model, "index_pip_joint")
    pip_closed, _ = joint_range(model, "index_pip_joint")
    pip_span = max(abs(pip_open - pip_closed), 1e-9)

    for finger in FINGERS:
        if sine:
            phase = FINGER_PHASE[finger]
            flexion = offset_rad + amp_rad * math.sin(2.0 * math.pi * freq_hz * sim_time + phase)
        else:
            flexion = manual_flexion

        flexion_amount = float(np.clip(flexion / pip_span, 0.0, 1.0))
        targets[f"{finger}_pip_joint"] = limit_blend(model, f"{finger}_pip_joint", flexion_amount)
        targets[f"{finger}_mcp_pitch_joint"] = limit_blend(
            model,
            f"{finger}_mcp_pitch_joint",
            flexion_amount,
        )
        targets[f"{finger}_mcp_abad_joint"] = signed_limit_blend(
            model,
            f"{finger}_mcp_abad_joint",
            flexion_amount,
            FINGER_ABAD[finger],
        )

    if thumb_enabled:
        if sine:
            thumb_amount = 0.5 + 0.5 * math.sin(2.0 * math.pi * freq_hz * sim_time - 0.45)
        else:
            thumb_amount = float(np.clip(manual_flexion / pip_span, 0.0, 1.0))

        targets.update(thumb_opposition_targets(model, thumb_amount))
    else:
        targets.update(
            {
                "thumb_cmc_roll_joint": 0.0,
                "thumb_cmc_side_joint": 0.0,
                "thumb_mcp_joint": 0.0,
                "thumb_dip_joint": 0.0,
            }
        )

    return {joint: clip_joint(model, joint, target) for joint, target in targets.items()}


def step_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: Dict[str, int],
    state: Dict[str, float | bool],
    args: argparse.Namespace,
    sim_time: float,
) -> Dict[str, float]:
    targets = motion_targets(
        model,
        sim_time,
        sine=bool(state["sine"]),
        manual_flexion=float(state["manual_flexion"]),
        amp_rad=np.deg2rad(args.amp_deg),
        offset_rad=np.deg2rad(args.offset_deg),
        freq_hz=args.freq_hz,
        thumb_enabled=bool(state["thumb"]),
    )
    apply_targets(model, data, actuator_id, targets)
    mujoco.mj_step(model, data)
    return targets


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: Dict[str, int],
    state: Dict[str, float | bool],
    args: argparse.Namespace,
) -> None:
    for step in range(args.steps):
        step_model(model, data, actuator_id, state, args, step * model.opt.timestep)

    residual = float(np.max(np.abs(data.efc_pos))) if data.efc_pos.size else 0.0
    print(
        f"Headless simulation complete: steps={args.steps}, "
        f"time={data.time:.3f}s, max_constraint_residual={residual:.3e}"
    )


def main() -> None:
    default_xml = Path(__file__).resolve().parent / "midas_description" / "midas_whole_hand.xml"
    parser = argparse.ArgumentParser(description="Run the closed-loop MIDAS whole hand in MuJoCo.")
    parser.add_argument("--xml", type=Path, default=default_xml, help="MJCF XML file.")
    parser.add_argument("--manual", action="store_true", help="Start in keyboard/manual flexion mode.")
    parser.add_argument("--amp-deg", type=float, default=41.5, help="Automatic flexion sine amplitude.")
    parser.add_argument("--offset-deg", type=float, default=41.5, help="Automatic flexion sine offset.")
    parser.add_argument("--freq-hz", type=float, default=0.18, help="Automatic flexion frequency.")
    parser.add_argument("--step-deg", type=float, default=3.0, help="Keyboard flexion increment.")
    parser.add_argument("--no-thumb", action="store_true", help="Hold thumb joints at zero.")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless instead of launching the viewer.")
    parser.add_argument("--steps", type=int, default=2000, help="Headless simulation steps.")
    parser.add_argument("--print-rate", type=float, default=5.0, help="Status print rate in Hz.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    actuator_id = build_actuator_map(model)

    state: Dict[str, float | bool] = {
        "sine": not args.manual,
        "manual_flexion": np.deg2rad(args.offset_deg),
        "thumb": not args.no_thumb,
        "close": False,
    }

    def key_callback(keycode: int) -> None:
        if keycode in (ord("="), ord("]")):
            state["manual_flexion"] = float(state["manual_flexion"]) + np.deg2rad(args.step_deg)
            state["sine"] = False
        elif keycode in (ord("-"), ord("[")):
            state["manual_flexion"] = max(0.0, float(state["manual_flexion"]) - np.deg2rad(args.step_deg))
            state["sine"] = False
        elif keycode == ord("0"):
            state["manual_flexion"] = 0.0
            state["sine"] = False
        elif keycode in (ord("S"), ord("s")):
            state["sine"] = not bool(state["sine"])
            print(f"\nSine mode: {state['sine']}")
        elif keycode in (ord("T"), ord("t")):
            state["thumb"] = not bool(state["thumb"])
            print(f"\nThumb motion: {state['thumb']}")
        elif keycode == 256:  # Esc
            state["close"] = True

    if args.no_viewer:
        run_headless(model, data, actuator_id, state, args)
        return

    print("\nLaunching MuJoCo viewer for MIDAS whole hand.")
    print("Controls: = or ] flex, - or [ extend, 0 open, S sine/manual, T thumb motion, Esc close.")
    print("Finger DIP and linkage joints are passive closed loops constrained in MJCF.\n")

    start_wall = time.time()
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not bool(state["close"]):
            now = time.time()
            sim_time = now - start_wall

            with viewer.lock():
                targets = step_model(model, data, actuator_id, state, args, sim_time)

            viewer.sync()

            if now - last_print > 1.0 / max(args.print_rate, 1e-6):
                last_print = now
                residual = float(np.max(np.abs(data.efc_pos))) if data.efc_pos.size else 0.0
                print(
                    "\r"
                    f"index_pip={np.rad2deg(targets['index_pip_joint']):7.2f} deg | "
                    f"middle_pip={np.rad2deg(targets['middle_pip_joint']):7.2f} deg | "
                    f"ring_pip={np.rad2deg(targets['ring_pip_joint']):7.2f} deg | "
                    f"residual={residual:.2e} | "
                    f"{'sine' if state['sine'] else 'manual'}"
                    "     ",
                    end="",
                    flush=True,
                )

            time.sleep(0.001)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
