from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path


# Edit these angles directly. Positive angles curl toward the sole.
LEFT_TOE_TARGET_DEG = {
    "left_big_toe": 32.0,
    "left_toe_2": 38.0,
    "left_toe_3": 40.0,
    "left_toe_4": 38.0,
    "left_toe_5": 34.0,
}

RIGHT_TOE_TARGET_DEG = {
    "right_big_toe": 32.0,
    "right_toe_2": 38.0,
    "right_toe_3": 40.0,
    "right_toe_4": 38.0,
    "right_toe_5": 34.0,
}

CYCLE_SECONDS = 5.0
POSITION_KP = 2.5
POSITION_KV = 0.16
MAX_TOE_TORQUE = 1.2


HERE = Path(__file__).resolve().parent


def _import_mujoco():
    try:
        import mujoco

        return mujoco
    except ModuleNotFoundError:
        # The containing DexCoDesign workspace already ships this environment.
        # Re-exec through mjpython so `python view_feet.py` also opens the native
        # macOS viewer correctly.
        bin_dir = HERE.parent / ".venv-morphology" / "bin"
        runner = bin_dir / ("python" if "--headless-check" in sys.argv else "mjpython")
        if runner.is_file() and os.environ.get("_FEMALE_FEET_REEXEC") != "1":
            env = os.environ.copy()
            env["_FEMALE_FEET_REEXEC"] = "1"
            os.execve(str(runner), [str(runner), str(Path(__file__).resolve()), *sys.argv[1:]], env)
        raise RuntimeError(
            "MuJoCo is not installed for this Python. Install `mujoco`, or run with "
            f"{runner}."
        )


mujoco = _import_mujoco()


def _attach_urdf(scene, urdf_path: Path, prefix: str, mount_name: str, mount_pos):
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    foot = mujoco.MjSpec.from_file(str(urdf_path))
    mount = scene.worldbody.add_frame(name=mount_name, pos=mount_pos)
    scene.attach(foot, prefix=prefix, frame=mount)


def build_model():
    scene = mujoco.MjSpec.from_string(
        """
<mujoco model="female_supr_feet">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75" specular="0.15 0.15 0.15"/>
    <rgba haze="0.92 0.94 0.97 1"/>
  </visual>
  <worldbody>
    <light name="key" pos="-0.4 -0.5 1.0" dir="0.4 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" group="2" size="1 1 0.05" pos="0 0 0"
          rgba="0.72 0.75 0.78 1" contype="2" conaffinity="1"
          friction="0.9 0.03 0.003" solref="0.035 1" solimp="0.85 0.95 0.002 0.5 2"/>
  </worldbody>
</mujoco>
"""
    )
    _attach_urdf(
        scene,
        HERE / "left_foot.urdf",
        prefix="L_",
        mount_name="left_mount",
        mount_pos=(-0.12, 0.075, 0.14),
    )
    _attach_urdf(
        scene,
        HERE / "right_foot.urdf",
        prefix="R_",
        mount_name="right_mount",
        mount_pos=(-0.12, -0.075, 0.14),
    )

    # URDF has no native soft-contact parameters. Configure only collision
    # geoms here; visual geoms remain contact-free. Bit masks keep both feet
    # and sibling toes from colliding with each other while retaining ground
    # contact if the mount height is lowered.
    for geom in scene.geoms:
        if geom.group == 0 and geom.name != "ground":
            geom.contype = 1
            geom.conaffinity = 2
            geom.friction = [0.85, 0.025, 0.002]
            geom.solref = [0.035, 1.0]
            geom.solimp = [0.85, 0.95, 0.002, 0.5, 2.0]
            geom.margin = 0.0008
            geom.gap = 0.0002

    target_items = [
        *(("L_" + name, angle) for name, angle in LEFT_TOE_TARGET_DEG.items()),
        *(("R_" + name, angle) for name, angle in RIGHT_TOE_TARGET_DEG.items()),
    ]
    for prefixed_joint, _ in target_items:
        actuator = scene.add_actuator(name=prefixed_joint + "_position", target=prefixed_joint)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_position(kp=POSITION_KP, kv=POSITION_KV)
        actuator.ctrllimited = True
        actuator.ctrlrange = [-0.5, 0.8]
        actuator.forcelimited = True
        actuator.forcerange = [-MAX_TOE_TORQUE, MAX_TOE_TORQUE]

    model = scene.compile()
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    targets = [math.radians(angle) for _, angle in target_items]
    actuator_names = [name + "_position" for name, _ in target_items]
    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names
    ]
    if any(actuator_id < 0 for actuator_id in actuator_ids):
        raise RuntimeError("Not all ten toe position actuators were created.")
    return model, actuator_ids, targets


def set_periodic_targets(model, data, actuator_ids, targets):
    blend = 0.5 - 0.5 * math.cos(2.0 * math.pi * data.time / CYCLE_SECONDS)
    for actuator_id, target in zip(actuator_ids, targets):
        data.ctrl[actuator_id] = blend * target


def headless_check(seconds: float = 8.0):
    model, actuator_ids, targets = build_model()
    data = mujoco.MjData(model)
    max_abs_qpos = 0.0
    steps = int(seconds / model.opt.timestep)
    for _ in range(steps):
        set_periodic_targets(model, data, actuator_ids, targets)
        mujoco.mj_step(model, data)
        if not (all(map(math.isfinite, data.qpos)) and all(map(math.isfinite, data.qvel))):
            raise RuntimeError("Simulation became non-finite.")
        max_abs_qpos = max(max_abs_qpos, float(max(abs(data.qpos))))
    if model.nu != 10 or model.njnt != 10 or max_abs_qpos < 0.15:
        raise RuntimeError(
            f"Unexpected model state: nu={model.nu}, njnt={model.njnt}, "
            f"max_abs_qpos={max_abs_qpos:.4f}"
        )
    print(
        f"OK: {model.nbody} bodies, {model.njnt} active toe joints, "
        f"{model.nu} soft position actuators; max |q|={max_abs_qpos:.3f} rad."
    )


def main():
    if "--headless-check" in sys.argv:
        headless_check()
        return

    import mujoco.viewer

    model, actuator_ids, targets = build_model()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0  # hide collision hulls; keep them active
        viewer.opt.geomgroup[1] = 1  # SUPR visual surface
        viewer.opt.geomgroup[2] = 1  # ground
        viewer.cam.lookat[:] = [0.0, 0.0, 0.17]
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -28
        while viewer.is_running():
            frame_start = time.perf_counter()
            set_periodic_targets(model, data, actuator_ids, targets)
            mujoco.mj_step(model, data)
            viewer.sync()
            delay = model.opt.timestep - (time.perf_counter() - frame_start)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
