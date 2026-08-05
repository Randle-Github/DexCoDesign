#!/usr/bin/env python3
"""Run every retargeted robot hand against one freely simulated object.

The retargeted joint and wrist trajectories are control targets, not states:

* the wrist is a kinematic mocap base following the retargeted wrist target;
* finger joints are driven by MuJoCo position actuators;
* the object is initialized once as a free body and is never reset;
* hand-object and object-table contacts are enabled;
* hand-table and hand self-collision are disabled.

Every numerically complete rollout is saved, regardless of whether the hand
grasps, drops, or pushes the object. A model or numerical failure is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
REGISTRY = DIRECT_ROOT / "registry.json"
RETARGET_ROOT = REPO_ROOT / "artifacts" / "all_hands_success_action_retarget"
DEFAULT_CAPTURE = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "subject_7"
    / "20231022_192832"
    / "isaaclab_reference.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "all_hands_physical_rollouts"
PHYSICS_CACHE = EXPERIMENT_ROOT / "artifacts" / "all_hands_physics_cache"

sys.path.insert(0, str(SCRIPT_ROOT))
from retarget_all_hands import (  # noqa: E402
    TIP_LINKS,
    lowest_common_ancestor,
    make_scene,
    rotation_matrix,
)


def _set(element: ET.Element, **attributes: object) -> None:
    for key, value in attributes.items():
        element.set(key, str(value))


def build_physics_scene(hand_id: str) -> Path:
    """Convert the retarget visualization scene to a physical control scene."""
    source = make_scene(hand_id)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", f"{root.get('model', hand_id)}_physical_rollout")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    _set(
        option,
        timestep=1.0 / 240.0,
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="elliptic",
        iterations=80,
        ls_iterations=20,
    )

    world = root.find("worldbody")
    assert world is not None
    wrapper = world.find("./body[@name='retarget_wrapper']")
    object_body = world.find("./body[@name='hocap_object_body']")
    table = world.find("./geom[@name='table']")
    assert wrapper is not None and object_body is not None and table is not None

    # URDF import creates a visual geom (group=1, collision disabled) and a
    # collision geom for each link. Keep visuals visible and collision proxies
    # physically active but transparent.
    for geom in wrapper.iter("geom"):
        is_visual = (
            geom.get("group") == "1"
            or (
                geom.get("contype") == "0"
                and geom.get("conaffinity") == "0"
                and geom.get("density") == "0"
            )
        )
        if is_visual:
            _set(geom, contype=0, conaffinity=0, density=0)
        else:
            _set(
                geom,
                contype=1,
                conaffinity=2,
                rgba="0 0 0 0",
                friction="1.2 0.02 0.002",
                condim=4,
                solref="0.006 1",
                solimp="0.95 0.99 0.001",
            )

    object_body.attrib.pop("mocap", None)
    object_body.insert(0, ET.Element("freejoint", {"name": "hocap_object_free"}))
    object_geom = object_body.find("./geom[@name='hocap_object_geom']")
    assert object_geom is not None
    _set(
        object_geom,
        contype=2,
        conaffinity=5,
        density=350,
        friction="1.1 0.02 0.002",
        condim=4,
        solref="0.006 1",
        solimp="0.95 0.99 0.001",
    )
    _set(
        table,
        contype=4,
        conaffinity=2,
        friction="1.0 0.02 0.002",
        condim=4,
        solref="0.006 1",
        solimp="0.95 0.99 0.001",
    )

    existing_actuator = root.find("actuator")
    if existing_actuator is not None:
        root.remove(existing_actuator)
    actuator = ET.SubElement(root, "actuator")
    for joint in wrapper.iter("joint"):
        name = joint.get("name")
        if not name:
            continue
        joint.set("frictionloss", "0")
        joint.set("damping", "0.25")
        low, high = -3.14159, 3.14159
        if joint.get("range"):
            low, high = (float(value) for value in joint.get("range").split())
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"track_{name}",
                "joint": name,
                "kp": "35",
                "ctrllimited": "true",
                "ctrlrange": f"{low} {high}",
                "forcelimited": "true",
                "forcerange": "-15 15",
            },
        )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    _set(global_visual, offwidth=960, offheight=720)

    PHYSICS_CACHE.mkdir(parents=True, exist_ok=True)
    output = PHYSICS_CACHE / f"{hand_id}_physical.xml"
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def quaternion_angle_wxyz(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(abs(np.dot(first, second)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def set_physics_wrist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wrist_id: int,
    wrapper_mocap_id: int,
    wrist_position: np.ndarray,
    wrist_rotation: Rotation,
) -> None:
    data.mocap_pos[wrapper_mocap_id] = 0.0
    data.mocap_quat[wrapper_mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    local_position = data.xpos[wrist_id].copy()
    local_rotation = Rotation.from_matrix(rotation_matrix(data, wrist_id))
    wrapper_rotation = wrist_rotation * local_rotation.inv()
    wrapper_position = wrist_position - wrapper_rotation.apply(local_position)
    data.mocap_pos[wrapper_mocap_id] = wrapper_position
    data.mocap_quat[wrapper_mocap_id] = np.roll(
        wrapper_rotation.as_quat(), 1
    )


def hand_object_contact_count(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> int:
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        first_type = int(model.geom_contype[first])
        second_type = int(model.geom_contype[second])
        if {first_type, second_type} == {1, 2}:
            count += 1
    return count


def simulate_hand(
    hand_id: str,
    display_name: str,
    trajectory_path: Path,
    object_reference: np.ndarray,
    output_dir: Path,
    fps: int,
    width: int,
    height: int,
    *,
    render_video: bool = True,
) -> dict[str, object]:
    targets = np.load(trajectory_path)
    q_targets = targets["qpos"].astype(np.float64)
    wrist_positions = targets["wrist_position"].astype(np.float64)
    wrist_quaternions = targets["wrist_quaternion_xyzw"].astype(np.float64)
    frame_count = min(
        len(q_targets),
        len(wrist_positions),
        len(object_reference),
    )
    q_targets = q_targets[:frame_count]
    wrist_positions = wrist_positions[:frame_count]
    wrist_quaternions = wrist_quaternions[:frame_count]
    object_reference = object_reference[:frame_count]

    # Recover the exact body and qpos-name semantics used when the target
    # cache was written. The stored qpos ids are stable for this source scene.
    reference_scene = make_scene(hand_id)
    reference_model = mujoco.MjModel.from_xml_path(str(reference_scene))
    reference_qpos_ids = targets["qpos_ids"].astype(int)
    qpos_to_joint_name = {
        int(reference_model.jnt_qposadr[joint_id]): reference_model.joint(
            joint_id
        ).name
        for joint_id in range(reference_model.njnt)
    }
    target_joint_names = [
        qpos_to_joint_name[int(qpos_id)] for qpos_id in reference_qpos_ids
    ]
    reference_tip_ids = [
        reference_model.body(link).id
        for link in TIP_LINKS[hand_id].values()
    ]
    reference_wrist_id = lowest_common_ancestor(
        reference_model, reference_tip_ids
    )
    reference_wrist_name = reference_model.body(reference_wrist_id).name

    scene = build_physics_scene(hand_id)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] == 1:
            model.geom_rgba[geom_id] = (0.16, 0.58, 0.82, 1.0)
    object_geom_id = model.geom("hocap_object_geom").id
    model.geom_rgba[object_geom_id] = (0.96, 0.48, 0.14, 1.0)
    model.geom_rgba[model.geom("table").id] = (0.72, 0.74, 0.78, 1.0)

    wrist_id = model.body(reference_wrist_name).id
    wrapper_mocap_id = int(model.body_mocapid[model.body("retarget_wrapper").id])
    object_body_id = model.body("hocap_object_body").id
    object_joint_id = model.joint("hocap_object_free").id
    object_qpos_id = int(model.jnt_qposadr[object_joint_id])
    target_qpos_ids = np.asarray(
        [model.joint(name).qposadr[0] for name in target_joint_names],
        dtype=int,
    )
    actuator_ids = np.asarray(
        [model.actuator(f"track_{name}").id for name in target_joint_names],
        dtype=int,
    )

    mujoco.mj_resetData(model, data)
    data.qpos[target_qpos_ids] = q_targets[0]
    data.ctrl[actuator_ids] = q_targets[0]
    data.qpos[object_qpos_id : object_qpos_id + 7] = object_reference[0]
    set_physics_wrist(
        model,
        data,
        wrist_id,
        wrapper_mocap_id,
        wrist_positions[0],
        Rotation.from_quat(wrist_quaternions[0]),
    )
    mujoco.mj_forward(model, data)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hand_id}_physical_rollout.mp4"
    temporary_path = output_dir / f".{hand_id}_physical_rollout.tmp.mp4"
    preview_path = output_dir / f"{hand_id}_physical_preview.png"
    diagnostics_path = output_dir / f"{hand_id}_physical_diagnostics.npz"
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-0.09, 0.01, 0.14)
    camera.distance = 0.72
    camera.azimuth = 132
    camera.elevation = -25
    renderer = (
        mujoco.Renderer(model, height=height, width=width)
        if render_video
        else None
    )
    ffmpeg = None
    if render_video:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                str(temporary_path),
            ],
            stdin=subprocess.PIPE,
        )

    position_errors = []
    orientation_errors = []
    contact_counts = []
    object_poses = []
    q_tracking_errors = []
    substeps = max(1, round((1.0 / fps) / model.opt.timestep))
    try:
        for frame_index in range(frame_count):
            data.ctrl[actuator_ids] = q_targets[frame_index]
            set_physics_wrist(
                model,
                data,
                wrist_id,
                wrapper_mocap_id,
                wrist_positions[frame_index],
                Rotation.from_quat(wrist_quaternions[frame_index]),
            )
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(
                    f"non-finite physics state at frame {frame_index}"
                )

            object_pose = np.concatenate(
                (data.xpos[object_body_id], data.xquat[object_body_id])
            )
            object_poses.append(object_pose.copy())
            position_errors.append(
                float(np.linalg.norm(object_pose[:3] - object_reference[frame_index, :3]))
            )
            orientation_errors.append(
                quaternion_angle_wxyz(
                    object_pose[3:7], object_reference[frame_index, 3:7]
                )
            )
            contact_counts.append(hand_object_contact_count(model, data))
            q_tracking_errors.append(
                float(
                    np.mean(
                        np.abs(
                            data.qpos[target_qpos_ids] - q_targets[frame_index]
                        )
                    )
                )
            )

            if render_video:
                assert renderer is not None and ffmpeg is not None
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render()
                assert ffmpeg.stdin is not None
                ffmpeg.stdin.write(pixels.tobytes())
                if frame_index == frame_count // 2:
                    Image.fromarray(pixels).save(preview_path)
    except Exception:
        if ffmpeg is not None:
            if ffmpeg.stdin is not None:
                ffmpeg.stdin.close()
            ffmpeg.wait()
        if renderer is not None:
            renderer.close()
        raise
    else:
        if render_video:
            assert ffmpeg is not None and ffmpeg.stdin is not None
            ffmpeg.stdin.close()
            return_code = ffmpeg.wait()
            assert renderer is not None
            renderer.close()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
            os.replace(temporary_path, output_path)

    np.savez_compressed(
        diagnostics_path,
        object_pose_wxyz=np.asarray(object_poses),
        reference_object_pose_wxyz=object_reference,
        object_position_error_m=np.asarray(position_errors),
        object_orientation_error_rad=np.asarray(orientation_errors),
        hand_object_contact_count=np.asarray(contact_counts),
        mean_abs_joint_tracking_error_rad=np.asarray(q_tracking_errors),
    )
    position_errors_array = np.asarray(position_errors)
    orientation_errors_array = np.asarray(orientation_errors)
    pose_errors = np.sqrt(
        position_errors_array**2 + (0.3 * orientation_errors_array) ** 2
    )
    reward_offset_c = float(np.sqrt(0.05**2 + (0.3 * 1.5) ** 2))
    pose_rewards = reward_offset_c - pose_errors
    failures = np.flatnonzero(
        (position_errors_array > 0.05)
        | (orientation_errors_array > 1.5)
    )
    final_phase = int(failures[0]) if len(failures) else frame_count - 1
    evaluated_steps = final_phase + 1
    return {
        "status": "completed",
        "frames": frame_count,
        "video": str(output_path) if render_video else None,
        "preview": str(preview_path) if render_video else None,
        "diagnostics": str(diagnostics_path),
        "final_object_position_error_m": position_errors[-1],
        "max_object_position_error_m": float(np.max(position_errors)),
        "final_object_orientation_error_rad": orientation_errors[-1],
        "contact_frame_fraction": float(np.mean(np.asarray(contact_counts) > 0)),
        "mean_abs_joint_tracking_error_rad": float(np.mean(q_tracking_errors)),
        "strict_final_phase": final_phase,
        "strict_success": bool(final_phase == frame_count - 1),
        "pose_tracking_return": float(np.sum(pose_rewards[:evaluated_steps])),
        "reward_offset_c": reward_offset_c,
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--retarget-root", type=Path, default=RETARGET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hands", nargs="*", default=None)
    args = parser.parse_args()

    captured = np.load(args.capture)
    object_reference = captured["object_pose_wxyz"].astype(np.float64)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hand_ids = [
        hand_id for hand_id in registry["hands"] if hand_id != "mano"
    ]
    if args.hands:
        requested = set(args.hands)
        hand_ids = [hand_id for hand_id in hand_ids if hand_id in requested]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "source_capture": str(args.capture),
        "object_semantics": (
            "initialized at frame 0, then free rigid-body dynamics; never reset"
        ),
        "control_semantics": (
            "retargeted wrist as kinematic mocap base; retargeted finger joints "
            "as position-actuator targets"
        ),
        "collision_semantics": (
            "hand-object and object-table enabled; hand-table and hand self-collision disabled"
        ),
        "hands": {},
    }
    for index, hand_id in enumerate(hand_ids, 1):
        trajectory = (
            args.retarget_root / hand_id / "retargeted_trajectory.npz"
        )
        try:
            if not trajectory.exists():
                raise FileNotFoundError(trajectory)
            result = simulate_hand(
                hand_id,
                registry["hands"][hand_id]["display_name"],
                trajectory,
                object_reference,
                args.output_dir / hand_id,
                args.fps,
                args.width,
                args.height,
            )
        except Exception as error:
            result = {
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        report["hands"][hand_id] = result
        print(
            f"[{index:02d}/{len(hand_ids)}] {hand_id}: "
            f"{result['status']}"
            + (
                f", {result['frames']}/{len(object_reference)} frames"
                if result["status"] == "completed"
                else f" ({result['failure_reason']})"
            ),
            flush=True,
        )

    completed = [
        hand_id
        for hand_id, result in report["hands"].items()
        if result["status"] == "completed"
    ]
    failed = [
        hand_id
        for hand_id, result in report["hands"].items()
        if result["status"] == "failed"
    ]
    report["summary"] = {
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed_hands": completed,
        "failed_hands": failed,
    }
    report_path = args.output_dir / "physical_rollout_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
