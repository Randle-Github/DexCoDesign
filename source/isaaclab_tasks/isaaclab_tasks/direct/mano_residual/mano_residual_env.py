# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL over a reviewed HO-Cap hand reference trajectory.

The command is exactly ``q_target = q_reference + scale * residual``. The
training reward combines object-pose tracking with EgoEngine-MPC's binary
pinch-contact term.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import get_all_matching_child_prims
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
ASSET_ROOT = REPO_ROOT / "artifacts" / "isaaclab_mano_residual" / "assets"
MANO_REFERENCE_PATH = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "subject_7"
    / "20231022_192832"
    / "isaaclab_reference.npz"
)
REFERENCE_PATH_OVERRIDE = os.environ.get("DEXCODESIGN_REFERENCE_PATH")
BIMANUAL_REFERENCE_PATH_OVERRIDE = os.environ.get(
    "DEXCODESIGN_BIMANUAL_REFERENCE_PATH"
)
OBJECT_USD_PATH = Path(
    os.environ.get(
        "DEXCODESIGN_OBJECT_USD_PATH",
        str(ASSET_ROOT / "g04_1.usd"),
    )
).expanduser().resolve()
ALL_HAND_ROOT = REPO_ROOT / "artifacts" / "isaaclab_all_hands_residual"
MORPHOLOGY_BATCH_MANIFEST_PATH = os.environ.get(
    "DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST"
)
MORPHOLOGY_BATCH_MANIFEST = None
if MORPHOLOGY_BATCH_MANIFEST_PATH:
    MORPHOLOGY_BATCH_MANIFEST = json.loads(
        Path(MORPHOLOGY_BATCH_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    _batch_usd_paths = [
        Path(value).expanduser().resolve()
        for value in MORPHOLOGY_BATCH_MANIFEST["hand_usd_paths"]
    ]
    _batch_reference_paths = [
        Path(value).expanduser().resolve()
        for value in MORPHOLOGY_BATCH_MANIFEST["reference_paths"]
    ]
    if not _batch_usd_paths or len(_batch_usd_paths) != len(_batch_reference_paths):
        raise ValueError(
            "Morphology batch manifest needs equally-sized, non-empty "
            "hand_usd_paths and reference_paths"
        )
    HAND_ID = "wuji_morphology_batch"
else:
    _batch_usd_paths = []
    _batch_reference_paths = []
    HAND_ID = os.environ.get("DEXCODESIGN_HAND_ID", "mano")
if HAND_ID == "mano":
    MANO_SIDE = os.environ.get("DEXCODESIGN_MANO_SIDE", "left").strip().lower()
    if MANO_SIDE not in {"left", "right"}:
        raise ValueError(f"DEXCODESIGN_MANO_SIDE must be left or right, got {MANO_SIDE!r}")
    _mano = MANO_SIDE
    REFERENCE_PATH = (
        Path(REFERENCE_PATH_OVERRIDE).expanduser().resolve()
        if REFERENCE_PATH_OVERRIDE
        else MANO_REFERENCE_PATH
    )
    HAND_USD_PATH = ASSET_ROOT / f"mano_{_mano}.usd"
    ROOT_POSITION_JOINT_NAMES = tuple(f"{_mano}_pos_{axis}" for axis in "xyz")
    ROOT_ROTATION_JOINT_NAMES = tuple(f"{_mano}_rot_{axis}" for axis in "xyz")
    # Scissor handles and other tools are commonly supported by middle and
    # proximal phalanges, not only fingertips. Reward all collision-bearing
    # finger bodies while keeping palm and virtual serial-joint bodies out of
    # the opposing-finger pinch definition.
    THUMB_CONTACT_LINK_NAMES = tuple(
        f"{_mano}_{name}" for name in ("thumb1z", "thumb2z", "thumb3")
    )
    OTHER_FINGER_CONTACT_LINK_NAMES = tuple(
        f"{_mano}_{finger}{segment}"
        for finger in ("index", "middle", "ring", "pinky")
        for segment in ("1z", "2", "3")
    )
    ALL_HAND_CONTACT_LINK_NAMES = (
        f"{_mano}_palm",
        f"{_mano}_index1z",
        f"{_mano}_index2",
        f"{_mano}_index3",
        f"{_mano}_middle1z",
        f"{_mano}_middle2",
        f"{_mano}_middle3",
        f"{_mano}_ring1z",
        f"{_mano}_ring2",
        f"{_mano}_ring3",
        f"{_mano}_pinky1z",
        f"{_mano}_pinky2",
        f"{_mano}_pinky3",
        f"{_mano}_thumb1z",
        f"{_mano}_thumb2z",
        f"{_mano}_thumb3",
    )
    PALM_BODY_NAME = f"{_mano}_palm"
    MIDDLE_TIP_BODY_NAME = f"{_mano}_middle3"
else:
    if MORPHOLOGY_BATCH_MANIFEST is not None:
        REFERENCE_PATH = _batch_reference_paths[0]
        HAND_USD_PATH = _batch_usd_paths[0]
    else:
        REFERENCE_PATH = (
            Path(REFERENCE_PATH_OVERRIDE).expanduser().resolve()
            if REFERENCE_PATH_OVERRIDE
            else ALL_HAND_ROOT / "prepared" / HAND_ID / "reference.npz"
        )
        HAND_USD_PATH = ALL_HAND_ROOT / "assets" / HAND_ID / "hand.usd"
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing prepared reference for {HAND_ID}: {REFERENCE_PATH}"
        )
    with np.load(REFERENCE_PATH) as _schema:
        THUMB_CONTACT_LINK_NAMES = tuple(
            _schema["thumb_contact_link_names"].tolist()
        )
        OTHER_FINGER_CONTACT_LINK_NAMES = tuple(
            _schema["other_finger_contact_link_names"].tolist()
        )
        if "contact_link_names" not in _schema:
            raise RuntimeError(
                f"{REFERENCE_PATH} predates full collision coverage; "
                "rebuild all-hand assets"
            )
        ALL_HAND_CONTACT_LINK_NAMES = tuple(
            _schema["contact_link_names"].tolist()
        )
        PALM_BODY_NAME = str(_schema["palm_body_name"])
        MIDDLE_TIP_BODY_NAME = str(_schema["middle_tip_body_name"])
    ROOT_POSITION_JOINT_NAMES = ("root_pos_x", "root_pos_y", "root_pos_z")
    ROOT_ROTATION_JOINT_NAMES = ("root_rot_x", "root_rot_y", "root_rot_z")

with np.load(REFERENCE_PATH) as _reference_schema:
    CONTROL_DIM = int(_reference_schema["hand_q"].shape[1])
    ACTION_DIM = (
        len(_reference_schema["action_joint_names"])
        if "action_joint_names" in _reference_schema
        else CONTROL_DIM
    )
    OBJECT_JOINT_DIM = (
        int(np.asarray(_reference_schema["object_joint_position_rad"]).reshape(
            len(_reference_schema["hand_q"]), -1
        ).shape[1])
        if "object_joint_position_rad" in _reference_schema
        else 0
    )
OBSERVATION_DIM = 2 * CONTROL_DIM + 34
ROOT_POSITION_EXPR = f"{MANO_SIDE}_pos_.*" if HAND_ID == "mano" else "root_pos_.*"
ROOT_ROTATION_EXPR = f"{MANO_SIDE}_rot_.*" if HAND_ID == "mano" else "root_rot_.*"
FINGER_JOINT_EXPR = f"{MANO_SIDE}_j_.*" if HAND_ID == "mano" else "finger__.*"

if HAND_ID == "mano":
    SECOND_MANO_SIDE = "left" if MANO_SIDE == "right" else "right"
    SECOND_HAND_USD_PATH = ASSET_ROOT / f"mano_{SECOND_MANO_SIDE}.usd"
    SECOND_CONTACT_LINK_NAMES = (
        f"{SECOND_MANO_SIDE}_palm",
        *(
            f"{SECOND_MANO_SIDE}_{finger}{segment}"
            for finger in ("index", "middle", "ring", "pinky")
            for segment in ("1z", "2", "3")
        ),
        *(
            f"{SECOND_MANO_SIDE}_{name}"
            for name in ("thumb1z", "thumb2z", "thumb3")
        ),
    )
else:
    SECOND_MANO_SIDE = "left"
    SECOND_HAND_USD_PATH = ASSET_ROOT / "mano_left.usd"
    SECOND_CONTACT_LINK_NAMES = ()


def _points_in_elliptical_prism(
    points: torch.Tensor,
    center: torch.Tensor,
    radii_xy: torch.Tensor,
    half_height: float,
) -> torch.Tensor:
    """Return whether local-frame points lie inside an elliptical hole volume."""

    centered = points - center
    radial = (centered[..., :2] / radii_xy).square().sum(dim=-1)
    return (radial <= 1.0) & (centered[..., 2].abs() <= half_height)


@configclass
class ManoResidualEnvCfg(DirectRLEnvCfg):
    decimation = 4
    # Every training episode follows the reference from phase zero. PPO rollout
    # boundaries must not reset or randomly seek within the reference.
    episode_length_s = 15.0
    # The PPO action is the normalized residual itself. Keep its declared
    # bounds identical to the residual executed by _pre_physics_step so the
    # policy likelihood is evaluated on the action that reaches the robot.
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
    )
    # EgoEngine-style goal conditioning:
    # current q (N), current thumb/index tip positions (6), current object
    # pose (7), goal thumb/index tip poses (14), goal q (N), goal object
    # pose (7): 2N + 34 values.
    observation_space = OBSERVATION_DIM
    # Opt-in morphology conditioning. The morphology search driver sets this
    # to the design-vector width and appends the matching normalized context.
    morphology_context_dim = 0
    # Keep the legacy MANO/object observation layout unless a WUJI run
    # explicitly opts into the palm-geometry representation.
    observation_mode = "legacy"
    state_space = 0

    # Rigid mode keeps the original 7D object pose. Articulate mode changes
    # only the object-pose representation to [root xyz, root quat wxyz,
    # internal joint positions]. The policy sees this representation for both
    # current and goal object pose.
    articulate_mode = False
    articulated_object_fix_root_link = False
    # Formal manipulation runs must use a free object and a reference that
    # retains the demonstrated world-space root motion.  These guards prevent
    # fixed-base collision diagnostics from being mistaken for grasp training.
    require_free_object_root = False
    require_dynamic_object_root = False

    # Bimanual mode jointly controls both MANO hands. The policy action is the
    # concatenation [primary residual, opposite-hand residual]. It remains
    # opt-in so existing single-hand checkpoints and action dimensions do not
    # change.
    bimanual_mode = False

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        # 4096 synchronized WUJI evaluations exceeded the default 163840
        # contact patches (observed >175000). Overflow drops contacts and
        # invalidates returns; reserve headroom for the hand/object contacts.
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=2**19),
        log_dir=str(
            REPO_ROOT
            / "artifacts"
            / "isaaclab_all_hands_residual"
            / "until_success"
            / "isaaclab_logs"
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # Use Isaac Lab's official dexterous-hand baseline.  The previous
            # value (2.0) was a project-side grasp aid, not an asset/vendor or
            # EgoEngine parameter, and can amplify tangential contact chatter.
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.45, 0.42, 0.38),
        lookat=(-0.10, 0.0, 0.17),
        origin_type="env",
        resolution=(960, 720),
    )

    hand_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=(
            sim_utils.MultiUsdFileCfg(
                usd_path=[str(path) for path in _batch_usd_paths],
                random_choice=False,
                reuse_duplicate_assets=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    # Every canonical hand has a virtual ``world`` link followed
                    # by six actuated wrist joints. Keep that virtual base fixed;
                    # otherwise contact moves the entire articulation while the
                    # six reported wrist coordinates remain deceptively correct.
                    fix_root_link=True,
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            )
            if MORPHOLOGY_BATCH_MANIFEST is not None
            else sim_utils.UsdFileCfg(
                usd_path=str(HAND_USD_PATH),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    # Adjacent MANO collision shells intentionally overlap at
                    # the knuckles. Enabling articulation-wide self collision
                    # makes PhysX fight the joint constraints and can pull the
                    # palm/fingers metres apart after the first object contact.
                    # Hand-object and object-support collisions remain active.
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            # EgoEngine's MANO model uses kp=1000 for the six virtual wrist
            # joints and kp=300 for all articulated finger joints.  Its
            # joints have unit armature and position actuators use a critical
            # damping ratio.
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[ROOT_POSITION_EXPR, ROOT_ROTATION_EXPR],
                stiffness=1000.0,
                damping=63.2455532,
                effort_limit_sim=1000.0,
                velocity_limit_sim=20.0,
                armature=1.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[FINGER_JOINT_EXPR],
                stiffness=300.0,
                damping=34.6410162,
                effort_limit_sim=1000.0,
                velocity_limit_sim=20.0,
                armature=1.0,
            ),
        },
    )

    second_hand_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SecondHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SECOND_HAND_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[
                    f"{SECOND_MANO_SIDE}_pos_.*",
                    f"{SECOND_MANO_SIDE}_rot_.*",
                ],
                stiffness=1000.0,
                damping=63.2455532,
                effort_limit_sim=1000.0,
                velocity_limit_sim=20.0,
                armature=1.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[f"{SECOND_MANO_SIDE}_j_.*"],
                stiffness=300.0,
                damping=34.6410162,
                effort_limit_sim=1000.0,
                velocity_limit_sim=20.0,
                armature=1.0,
            ),
        },
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBJECT_USD_PATH),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.38, 0.08),
                roughness=0.55,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            # Match EgoEngine's object_mass_scale=0.1 against this asset's
            # original 0.15 kg nominal mass.
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    articulated_object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBJECT_USD_PATH),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.38, 0.08),
                roughness=0.55,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            # Preserve the source URDF's per-link inertial properties. ARCTIC's
            # scissors links are 22.15 g and 26.96 g; replacing both with a
            # generic rigid-object mass makes the tool unrealistically easy to
            # launch and also destroys its center-of-mass distribution.
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={},
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=(
            len(_batch_usd_paths)
            if MORPHOLOGY_BATCH_MANIFEST is not None
            else 1024
        ),
        env_spacing=0.65,
        # Heterogeneous morphology assets must keep independent joint frames
        # and collision geometry. MultiUsdFileCfg spawns them deterministically
        # in manifest order when physics replication is disabled.
        replicate_physics=MORPHOLOGY_BATCH_MANIFEST is None,
        # ContactSensor discovers one USD reporting prim per environment.
        # Fabric-only clones are absent from that discovery stage.
        clone_in_fabric=False,
    )

    # Residual bounds are applied directly to the reference controller target.
    # The reference is already close, so root translation and finger residuals
    # retain the validated conservative ranges. Wrist rotation alone receives
    # the wider range needed to clear the object during release and retreat.
    residual_root_position_scale = 0.10
    residual_root_rotation_scale = 0.30
    residual_finger_scale = 0.30
    object_position_sigma = 0.04
    object_rotation_sigma = 0.50
    # Match EgoEngine-MPC's Aria residual-RL reward geometry:
    # reward = C - ||[w_pos * position_error, w_rot * rotation_error]||_2
    #          + contact_scale * pinch_contact.
    object_position_reward_weight = 1.0
    # Extra emphasis on vertical tracking while preserving the established
    # isotropic position reward when left at one.
    object_z_reward_multiplier = 1.0
    object_rotation_reward_weight = 0.3
    object_articulation_reward_weight = 0.3
    # Optional bounded reward for contact-rich free-object tasks. Unlike the
    # legacy C-error form, its baseline cannot grow when failure thresholds are
    # relaxed, so a dropped object does not keep earning a large positive value.
    exponential_pose_reward = False
    pose_tracking_reward_scale = 1.0
    contact_reward_weight = 2.0
    # Optional support-clearance bonus. The initial reference height is the
    # settled tabletop height, so exceeding it by this margin is a conservative
    # proxy for the object no longer contacting the support plane.
    object_airborne_reward_weight = 0.0
    object_airborne_clearance = 0.01
    # For tools such as scissors, a useful pinch must oppose two different
    # articulated links rather than press both finger groups onto one link.
    # Keep this optional because single-link objects and some articulated
    # tasks legitimately use same-link grasps.
    require_cross_link_pinch = False
    # Opt-in gate for the ARCTIC scissors demonstration. It is disabled by
    # default so every existing dataset retains the original contact reward.
    # Coordinates are expressed in the corresponding scissors-link frame.
    scissors_handle_contact_gate = False
    scissors_top_hole_center = (-0.0654, 0.0292, 0.0035)
    scissors_top_hole_radii_xy = (0.0140, 0.0140)
    scissors_bottom_hole_center = (-0.0680, -0.0145, 0.0015)
    scissors_bottom_hole_radii_xy = (0.0260, 0.0120)
    scissors_hole_half_height = 0.015
    # Positive values gate contact reward by object-position tracking quality:
    # exp(-position_error / sigma). This prevents stationary table contact from
    # outscoring a grasp that actually follows a lifting reference.
    contact_tracking_sigma = 0.0
    # Optional penalty on excessive hand-object force. It is disabled by
    # default to preserve established rigid-object experiments; articulated
    # tasks can enable it to prevent a binary contact reward from being
    # satisfied by physically implausible crushing forces.
    contact_force_safe_threshold = 50.0
    contact_force_penalty_weight = 0.0
    # Keeps an otherwise object-only task from learning to move the hand away
    # while the already stable object earns pose reward without interaction.
    # Zero preserves the established rigid-object benchmark behavior; ARCTIC
    # enables this explicitly from its training launcher.
    residual_action_penalty_weight = 0.0
    # EgoEngine treats any penetrating thumb/object and other-finger/object
    # contact pair as active.  A positive Isaac contact force is its analogue.
    contact_force_threshold = 0.0
    object_failure_distance = 0.05
    object_failure_orientation = 1.50
    object_failure_articulation = 1.50
    disable_object_failure_termination = False
    randomize_start_phase = False
    randomize_start_phase_fraction = 1.0
    # Inclusive phase bounds for contact-rich curriculum resets. ``-1`` keeps
    # the upper bound at the penultimate reference frame.
    random_start_phase_min = 0
    random_start_phase_max = -1
    # Optional local curriculum horizon in control steps. Zero keeps the full
    # remaining reference. A finite horizon prevents easy post-manipulation
    # table frames from dominating contact/lift learning.
    random_start_episode_length = 0
    log_rollout_diagnostics = False


@configclass
class ManoResidualPlayEnvCfg(ManoResidualEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.65,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    episode_length_s = 15.0
    # Evaluation must play the complete reference once. Training intentionally
    # terminates failed rollouts early, but inheriting that threshold here can
    # reset the one-environment video to frame zero on every failed step.
    # Keep the finite training reward normalization constants while allowing a
    # complete diagnostic/video rollout even after a tracking failure.
    disable_object_failure_termination = True
    randomize_start_phase = False
    log_rollout_diagnostics = True


@configclass
class ManoResidualEvalEnvCfg(ManoResidualEnvCfg):
    """Full-reference evaluation with training termination thresholds enabled."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.65,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    episode_length_s = 15.0
    randomize_start_phase = False
    log_rollout_diagnostics = False


class ManoResidualEnv(DirectRLEnv):
    cfg: ManoResidualEnvCfg

    @property
    def num_envs(self) -> int:
        if getattr(self, "_grouped_physics_replicas", False):
            return len(_batch_reference_paths)
        return self.scene.num_envs

    def __init__(self, cfg: ManoResidualEnvCfg, render_mode: str | None = None, **kwargs):
        self._bimanual_mode = bool(getattr(cfg, "bimanual_mode", False))
        if self._bimanual_mode:
            if HAND_ID != "mano" or MORPHOLOGY_BATCH_MANIFEST is not None:
                raise ValueError(
                    "bimanual_mode currently supports one canonical MANO pair only"
                )
            if not BIMANUAL_REFERENCE_PATH_OVERRIDE:
                raise ValueError(
                    "bimanual_mode requires DEXCODESIGN_BIMANUAL_REFERENCE_PATH"
                )
            second_reference_path = Path(
                BIMANUAL_REFERENCE_PATH_OVERRIDE
            ).expanduser().resolve()
            if not second_reference_path.is_file():
                raise FileNotFoundError(
                    f"Missing second MANO reference: {second_reference_path}"
                )
            with np.load(second_reference_path) as second_reference_file:
                self._second_reference = {
                    key: second_reference_file[key].copy()
                    for key in second_reference_file.files
                }
            cfg.second_hand_cfg.spawn.activate_contact_sensors = bool(
                cfg.articulate_mode
            )
        if (
            MORPHOLOGY_BATCH_MANIFEST is not None
            and not MORPHOLOGY_BATCH_MANIFEST.get("grouped_physics_replication", False)
            and cfg.scene.num_envs != len(_batch_reference_paths)
        ):
            raise ValueError(
                f"Morphology manifest contains {len(_batch_reference_paths)} environment rows, "
                f"but scene.num_envs={cfg.scene.num_envs}. --num_envs does not resize "
                "the hand assets, references, or geometry metadata in a manifest. "
                "Use a manifest with one row per requested environment, or set "
                f"--num_envs {len(_batch_reference_paths)}. "
                f"Manifest: {MORPHOLOGY_BATCH_MANIFEST_PATH}"
            )
        if cfg.observation_mode not in ("legacy", "palm_geometry"):
            raise ValueError(f"Unknown observation_mode: {cfg.observation_mode}")
        if cfg.observation_mode == "palm_geometry":
            if HAND_ID not in ("wuji_hand_2", "wuji_morphology_batch"):
                raise ValueError("palm_geometry currently supports only WUJI topology")
            if cfg.observation_space != PALM_GEOMETRY_OBSERVATION_DIM or cfg.morphology_context_dim:
                raise ValueError(
                    f"palm_geometry requires {PALM_GEOMETRY_OBSERVATION_DIM} "
                    "observations and no morphology context"
                )
        reference_paths = (
            _batch_reference_paths
            if MORPHOLOGY_BATCH_MANIFEST is not None
            else [REFERENCE_PATH]
        )
        references = [np.load(path) for path in reference_paths]
        reference = references[0]
        if self._bimanual_mode:
            required_second_keys = (
                "joint_names",
                "hand_q",
                "hand_ctrl",
                "object_pose_wxyz",
                "fingertip_pose_wxyz",
                "fingertip_link_names",
                "fingertip_offsets",
            )
            missing = [
                key
                for key in required_second_keys
                if key not in self._second_reference
            ]
            if missing:
                raise RuntimeError(
                    f"Second MANO reference is missing required fields: {missing}"
                )
            if len(self._second_reference["hand_q"]) != len(reference["hand_q"]):
                raise RuntimeError(
                    "Primary and second MANO references have different lengths"
                )
            if not np.allclose(
                self._second_reference["object_pose_wxyz"],
                reference["object_pose_wxyz"],
                atol=1.0e-5,
            ):
                raise RuntimeError(
                    "Primary and second MANO references disagree on object pose"
                )
        self._reference_fps = float(np.asarray(reference.get("fps", 0.0)))
        self._control_dt = float(cfg.sim.dt * cfg.decimation)
        if self._reference_fps <= 0.0:
            self._reference_fps = 1.0 / self._control_dt
        reference_duration = (len(reference["hand_q"]) - 1) / self._reference_fps
        # A reference frame is not necessarily one RL control step. ARCTIC is
        # 10 Hz while this environment controls at 30 Hz, so each source frame
        # must remain active for three control steps.
        cfg.episode_length_s = max(
            float(cfg.episode_length_s),
            reference_duration + 2.0 * self._control_dt,
        )
        if cfg.articulate_mode and OBJECT_JOINT_DIM == 0:
            raise ValueError(
                "articulate_mode requires object_joint_position_rad in the reference"
            )
        if cfg.articulate_mode:
            cfg.articulated_object_cfg.spawn.articulation_props.fix_root_link = (
                cfg.articulated_object_fix_root_link
            )
        cfg.observation_space = OBSERVATION_DIM + (
            2 * OBJECT_JOINT_DIM if cfg.articulate_mode else 0
        )
        if self._bimanual_mode:
            second_control_dim = int(
                self._second_reference["hand_q"].shape[1]
            )
            second_tip_count = int(
                self._second_reference["fingertip_pose_wxyz"].shape[1]
            )
            # Second current/goal q plus current xyz and goal xyz+quat for
            # each tracked fingertip. Object state remains represented once.
            cfg.observation_space += (
                2 * second_control_dim + 10 * second_tip_count
            )
        if cfg.articulate_mode and cfg.hand_cfg.spawn is not None:
            # Articulated-object contact filtering reports from one hand link
            # and filters against all object links, so the hand bodies need
            # the contact reporter API in this mode only.
            cfg.hand_cfg.spawn.activate_contact_sensors = True
        self._morphology_batch = MORPHOLOGY_BATCH_MANIFEST is not None
        self._grouped_physics_replicas = bool(
            self._morphology_batch
            and MORPHOLOGY_BATCH_MANIFEST is not None
            and MORPHOLOGY_BATCH_MANIFEST.get("grouped_physics_replication", False)
        )
        context = (
            MORPHOLOGY_BATCH_MANIFEST.get("policy_morphology_context")
            if MORPHOLOGY_BATCH_MANIFEST is not None
            else None
        )
        self._morphology_context_cpu = (
            None
            if context is None
            else torch.as_tensor(context, dtype=torch.float32)
        )
        self._reference_joint_names = reference["joint_names"].tolist()
        self._action_joint_names = (
            reference["action_joint_names"].tolist()
            if "action_joint_names" in reference
            else list(self._reference_joint_names)
        )
        self._action_to_control_cpu = torch.from_numpy(
            reference["action_to_control_matrix"]
            if "action_to_control_matrix" in reference
            else np.eye(len(self._reference_joint_names), dtype=np.float32)
        )
        self._primary_action_dim = len(self._action_joint_names)
        if self._bimanual_mode:
            self._second_reference_joint_names = self._second_reference[
                "joint_names"
            ].tolist()
            self._second_action_joint_names = (
                self._second_reference["action_joint_names"].tolist()
                if "action_joint_names" in self._second_reference
                else list(self._second_reference_joint_names)
            )
            self._second_action_to_control_cpu = torch.from_numpy(
                self._second_reference["action_to_control_matrix"]
                if "action_to_control_matrix" in self._second_reference
                else np.eye(
                    len(self._second_reference_joint_names), dtype=np.float32
                )
            )
        else:
            self._second_reference_joint_names = []
            self._second_action_joint_names = []
            self._second_action_to_control_cpu = None
        self._second_action_dim = len(self._second_action_joint_names)
        self._all_action_joint_names = [
            *self._action_joint_names,
            *self._second_action_joint_names,
        ]
        cfg.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self._all_action_joint_names),),
            dtype=np.float32,
        )
        def stacked(key: str) -> np.ndarray:
            values = [entry[key] for entry in references]
            if self._morphology_batch:
                return np.stack(values, axis=0)
            return values[0]

        for candidate in references[1:]:
            for key in (
                "joint_names",
                "action_joint_names",
                "fingertip_link_names",
                "thumb_contact_link_names",
                "other_finger_contact_link_names",
                "contact_link_names",
            ):
                if key in reference and not np.array_equal(reference[key], candidate[key]):
                    raise RuntimeError(
                        f"Morphology batch references disagree on {key}; all "
                        "candidates must preserve WUJI topology"
                    )
        self._reference_hand_q_cpu = torch.from_numpy(stacked("hand_q"))
        self._initialization_hand_q_cpu = (
            torch.from_numpy(stacked("initialization_hand_q"))
            if "initialization_hand_q" in reference
            else None
        )
        if "hand_ctrl" not in reference:
            raise RuntimeError(
                f"{REFERENCE_PATH} has no hand_ctrl; regenerate the "
                "EgoEngine-style reference before training"
            )
        self._reference_hand_ctrl_cpu = torch.from_numpy(stacked("hand_ctrl"))
        self._reference_object_pose_cpu = torch.from_numpy(
            stacked("object_pose_wxyz")
        )
        if cfg.require_free_object_root and cfg.articulated_object_fix_root_link:
            raise ValueError(
                "Formal manipulation training requires a free articulated-object "
                "root; fixed-root mode is diagnostic-only"
            )
        if cfg.require_dynamic_object_root:
            root_positions = self._reference_object_pose_cpu[..., :3]
            time_axis = 1 if self._morphology_batch else 0
            root_motion = root_positions.amax(dim=time_axis) - root_positions.amin(
                dim=time_axis
            )
            if float(root_motion.amax()) < 1.0e-4:
                raise ValueError(
                    "Formal manipulation training requires a dynamic object-root "
                    "trajectory; the selected reference is table-fixed"
                )
        if cfg.articulate_mode:
            object_joint_values = stacked("object_joint_position_rad")
            object_joint_values = object_joint_values.reshape(
                object_joint_values.shape[0], object_joint_values.shape[1], -1
            ) if self._morphology_batch else object_joint_values.reshape(
                object_joint_values.shape[0], -1
            )
            self._reference_object_joint_cpu = torch.from_numpy(object_joint_values)
            if "initialization_object_joint_position_rad" in reference:
                initialization_object_joint_values = stacked(
                    "initialization_object_joint_position_rad"
                )
                initialization_object_joint_values = (
                    initialization_object_joint_values.reshape(
                        initialization_object_joint_values.shape[0],
                        initialization_object_joint_values.shape[1],
                        -1,
                    )
                    if self._morphology_batch
                    else initialization_object_joint_values.reshape(
                        initialization_object_joint_values.shape[0], -1
                    )
                )
                self._initialization_object_joint_cpu = torch.from_numpy(
                    initialization_object_joint_values
                )
            else:
                self._initialization_object_joint_cpu = None
            if "object_joint_names" not in reference:
                raise RuntimeError(
                    "articulate_mode requires object_joint_names in the reference"
                )
            self._reference_object_joint_names = reference[
                "object_joint_names"
            ].tolist()
        else:
            self._reference_object_joint_cpu = None
            self._initialization_object_joint_cpu = None
            self._reference_object_joint_names = []
        required_fingertip_keys = (
            "fingertip_pose_wxyz",
            "fingertip_link_names",
            "fingertip_offsets",
        )
        missing_fingertip_keys = [
            key for key in required_fingertip_keys if key not in reference
        ]
        if missing_fingertip_keys:
            raise RuntimeError(
                f"{REFERENCE_PATH} is missing {missing_fingertip_keys}; regenerate "
                "the EgoEngine-style reference before training"
            )
        self._reference_fingertip_pose_cpu = torch.from_numpy(
            stacked("fingertip_pose_wxyz")
        )
        self._reference_fingertip_link_names = reference[
            "fingertip_link_names"
        ].tolist()
        self._fingertip_offsets_cpu = torch.from_numpy(stacked("fingertip_offsets"))
        self._reference_length = int(reference["hand_q"].shape[0])

        super().__init__(cfg, render_mode, **kwargs)
        if cfg.articulate_mode:
            object_masses = self.object.data.default_mass[0].detach().cpu().tolist()
            print(
                f"[HAND_OBJECT_MASS:{HAND_ID}] body_names={self.object.body_names} "
                f"mass_kg={object_masses} total_kg={sum(object_masses):.9f}",
                flush=True,
            )

        self.action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.action_dim != len(self._all_action_joint_names):
            raise RuntimeError(
                f"Action space has {self.action_dim} dimensions but {HAND_ID} "
                f"defines {len(self._all_action_joint_names)} active joints"
            )
        if self.hand.num_joints != len(self._reference_joint_names):
            raise RuntimeError(
                f"Expected {len(self._reference_joint_names)} controlled {HAND_ID} "
                f"joints, found {self.hand.num_joints}: {self.hand.joint_names}"
            )
        missing = sorted(set(self._reference_joint_names) - set(self.hand.joint_names))
        if missing:
            raise RuntimeError(
                f"Reference joints missing from imported {HAND_ID} articulation: {missing}"
            )

        reference_order = [self._reference_joint_names.index(name) for name in self.hand.joint_names]
        joint_axis = 2 if self._morphology_batch else 1
        self.reference_hand_q = self._reference_hand_q_cpu.index_select(
            joint_axis, torch.tensor(reference_order)
        ).to(self.device)
        self.initialization_hand_q = (
            None
            if self._initialization_hand_q_cpu is None
            else self._initialization_hand_q_cpu.index_select(
                joint_axis, torch.tensor(reference_order)
            ).to(self.device)
        )
        self.reference_hand_ctrl = self._reference_hand_ctrl_cpu.index_select(
            joint_axis, torch.tensor(reference_order)
        ).to(self.device)
        self.action_to_control = self._action_to_control_cpu[reference_order].to(
            self.device
        )
        if self.action_to_control.shape != (
            self.hand.num_joints,
            self._primary_action_dim,
        ):
            raise RuntimeError(
                f"{HAND_ID} action-to-control map has shape "
                f"{tuple(self.action_to_control.shape)}, expected "
                f"({self.hand.num_joints}, {self._primary_action_dim})"
            )
        self.reference_object_pose = self._reference_object_pose_cpu.to(self.device)
        if self.cfg.articulate_mode:
            if self.object.num_joints != len(self._reference_object_joint_names):
                raise RuntimeError(
                    "Articulated object joint count does not match reference: "
                    f"asset={self.object.joint_names}, "
                    f"reference={self._reference_object_joint_names}"
                )
            missing_object_joints = sorted(
                set(self._reference_object_joint_names) - set(self.object.joint_names)
            )
            if missing_object_joints:
                raise RuntimeError(
                    f"Reference object joints missing from asset: {missing_object_joints}"
                )
            object_order = [
                self._reference_object_joint_names.index(name)
                for name in self.object.joint_names
            ]
            object_joint_axis = 2 if self._morphology_batch else 1
            self.reference_object_joint = self._reference_object_joint_cpu.index_select(
                object_joint_axis, torch.tensor(object_order)
            ).to(self.device)
            self.initialization_object_joint = (
                None
                if self._initialization_object_joint_cpu is None
                else self._initialization_object_joint_cpu.index_select(
                    object_joint_axis, torch.tensor(object_order)
                ).to(self.device)
            )
        else:
            self.reference_object_joint = None
            self.initialization_object_joint = None
        self.reference_fingertip_pose = self._reference_fingertip_pose_cpu.to(
            self.device
        )
        self.fingertip_offsets = self._fingertip_offsets_cpu.to(self.device)
        if self._bimanual_mode:
            second_joint_names = self._second_reference_joint_names
            if self.second_hand.num_joints != len(second_joint_names):
                raise RuntimeError(
                    "Second MANO joint count does not match its reference: "
                    f"asset={self.second_hand.joint_names}, "
                    f"reference={second_joint_names}"
                )
            missing_second_joints = sorted(
                set(second_joint_names) - set(self.second_hand.joint_names)
            )
            if missing_second_joints:
                raise RuntimeError(
                    "Second reference joints missing from imported MANO hand: "
                    f"{missing_second_joints}"
                )
            second_order = [
                second_joint_names.index(name)
                for name in self.second_hand.joint_names
            ]
            second_order_tensor = torch.tensor(second_order)
            self.second_reference_hand_q = torch.from_numpy(
                self._second_reference["hand_q"]
            ).index_select(1, second_order_tensor).to(self.device)
            self.second_reference_hand_ctrl = torch.from_numpy(
                self._second_reference["hand_ctrl"]
            ).index_select(1, second_order_tensor).to(self.device)
            self.second_action_to_control = (
                self._second_action_to_control_cpu[second_order]
                .to(self.device)
            )
            if self.second_action_to_control.shape != (
                self.second_hand.num_joints,
                self._second_action_dim,
            ):
                raise RuntimeError(
                    "Second MANO action-to-control map has shape "
                    f"{tuple(self.second_action_to_control.shape)}, expected "
                    f"({self.second_hand.num_joints}, "
                    f"{self._second_action_dim})"
                )
            self.second_reference_fingertip_pose = torch.from_numpy(
                self._second_reference["fingertip_pose_wxyz"]
            ).to(self.device)
            self.second_fingertip_offsets = torch.from_numpy(
                self._second_reference["fingertip_offsets"]
            ).to(self.device)
            second_tip_names = self._second_reference[
                "fingertip_link_names"
            ].tolist()
            missing_second_tips = sorted(
                set(second_tip_names) - set(self.second_hand.body_names)
            )
            if missing_second_tips:
                raise RuntimeError(
                    "Second fingertip links missing from imported MANO hand: "
                    f"{missing_second_tips}"
                )
            self._second_fingertip_body_indices = [
                self.second_hand.body_names.index(name)
                for name in second_tip_names
            ]
            second_limits = (
                self.second_hand.root_physx_view.get_dof_limits().to(self.device)
            )
            self.second_joint_lower_limits = second_limits[..., 0]
            self.second_joint_upper_limits = second_limits[..., 1]
        self.morphology_context = (
            None
            if self._morphology_context_cpu is None
            else self._morphology_context_cpu.to(self.device)
        )
        if self.morphology_context is not None:
            expected = (self.num_envs, self.cfg.morphology_context_dim)
            if tuple(self.morphology_context.shape) != expected:
                raise RuntimeError(
                    f"morphology context has shape {tuple(self.morphology_context.shape)}, "
                    f"expected {expected}"
                )

        limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.joint_lower_limits = limits[..., 0]
        self.joint_upper_limits = limits[..., 1]

        self.residual_scale = torch.full(
            (self.action_dim,), self.cfg.residual_finger_scale, device=self.device
        )
        root_position_action_names = [
            *ROOT_POSITION_JOINT_NAMES,
            *(
                tuple(f"{SECOND_MANO_SIDE}_pos_{axis}" for axis in "xyz")
                if self._bimanual_mode
                else ()
            ),
        ]
        root_rotation_action_names = [
            *ROOT_ROTATION_JOINT_NAMES,
            *(
                tuple(f"{SECOND_MANO_SIDE}_rot_{axis}" for axis in "xyz")
                if self._bimanual_mode
                else ()
            ),
        ]
        self._root_position_action_indices = torch.tensor(
            [
                self._all_action_joint_names.index(name)
                for name in root_position_action_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._root_rotation_action_indices = torch.tensor(
            [
                self._all_action_joint_names.index(name)
                for name in root_rotation_action_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._finger_action_indices = torch.tensor(
            [
                index
                for index, name in enumerate(self._all_action_joint_names)
                if name
                not in set(root_position_action_names + root_rotation_action_names)
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.residual_scale[self._root_position_action_indices] = (
            self.cfg.residual_root_position_scale
        )
        self.residual_scale[self._root_rotation_action_indices] = (
            self.cfg.residual_root_rotation_scale
        )

        self.phase_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reference_time_buf = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float, device=self.device
        )
        self.joint_targets = torch.zeros(
            (self.num_envs, self.hand.num_joints),
            dtype=torch.float,
            device=self.device,
        )
        if self._bimanual_mode:
            self.second_joint_targets = torch.zeros(
                (self.num_envs, self.second_hand.num_joints),
                dtype=torch.float,
                device=self.device,
            )
        self._object_position_error = torch.zeros(self.num_envs, device=self.device)
        self._object_rotation_error = torch.zeros(self.num_envs, device=self.device)
        # Evaluation curves report the accumulated C-error pose reward only.
        # Contact remains part of the optimization reward but is intentionally
        # excluded from this episode-return diagnostic.
        self._pose_episode_return = torch.zeros(self.num_envs, device=self.device)
        self._last_evaluated_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        success_capture_path = os.environ.get(
            "HAND_SUCCESS_TRAJECTORY_PATH",
            os.environ.get("MANO_SUCCESS_TRAJECTORY_PATH"),
        )
        best_rollout_path = os.environ.get("HAND_BEST_ROLLOUT_PATH")
        self._success_capture_path = (
            Path(success_capture_path).expanduser().resolve()
            if success_capture_path
            else None
        )
        self._best_rollout_path = (
            Path(best_rollout_path).expanduser().resolve()
            if best_rollout_path
            else None
        )
        self._best_rollout_phase = -1
        self._best_rollout_return = float("-inf")
        if self._best_rollout_path is not None and self._best_rollout_path.is_file():
            try:
                with np.load(self._best_rollout_path) as previous_rollout:
                    previous_metadata = json.loads(
                        str(previous_rollout["metadata_json"])
                    )
                self._best_rollout_phase = int(
                    previous_metadata.get("final_phase", -1)
                )
                self._best_rollout_return = float(
                    previous_metadata.get("pose_tracking_return", 0.0)
                    + previous_metadata.get("contact_return", 0.0)
                )
                print(
                    "HAND_BEST_ROLLOUT_RESUMED "
                    f"hand_id={HAND_ID} path={self._best_rollout_path} "
                    f"phase={self._best_rollout_phase}",
                    flush=True,
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                # A malformed prior capture must not prevent training. It will
                # be replaced as soon as the resumed run records a valid phase.
                self._best_rollout_phase = -1
        self._capture_enabled = (
            self._success_capture_path is not None
            or self._best_rollout_path is not None
        )
        if self._capture_enabled:
            capture_shape = (self.num_envs, self._reference_length)
            self._capture_hand_q = torch.zeros(
                (*capture_shape, self.hand.num_joints),
                dtype=torch.float32,
                device=self.device,
            )
            if self._bimanual_mode:
                self._capture_second_hand_q = torch.zeros(
                    (*capture_shape, self.second_hand.num_joints),
                    dtype=torch.float32,
                    device=self.device,
                )
                self._capture_second_joint_targets = torch.zeros(
                    (*capture_shape, self.second_hand.num_joints),
                    dtype=torch.float32,
                    device=self.device,
                )
            self._capture_object_pose = torch.zeros(
                (*capture_shape, 7),
                dtype=torch.float32,
                device=self.device,
            )
            self._capture_object_joint = (
                torch.zeros(
                    (*capture_shape, self.object.num_joints),
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.cfg.articulate_mode
                else None
            )
            self._capture_actions = torch.zeros(
                (*capture_shape, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self._capture_joint_targets = torch.zeros(
                (*capture_shape, self.hand.num_joints),
                dtype=torch.float32,
                device=self.device,
            )
            self._capture_pose_reward = torch.zeros(
                capture_shape, dtype=torch.float32, device=self.device
            )
            self._capture_contact_reward = torch.zeros_like(
                self._capture_pose_reward
            )
            self._capture_position_error = torch.zeros_like(
                self._capture_pose_reward
            )
            self._capture_rotation_error = torch.zeros_like(
                self._capture_pose_reward
            )
            self._capture_articulation_error = torch.zeros_like(
                self._capture_pose_reward
            )
        self._last_diagnostic_phase = -1
        self._palm_body_index = self.hand.body_names.index(PALM_BODY_NAME)
        self._middle_tip_body_index = self.hand.body_names.index(
            MIDDLE_TIP_BODY_NAME
        )
        missing_fingertip_links = sorted(
            set(self._reference_fingertip_link_names) - set(self.hand.body_names)
        )
        if missing_fingertip_links:
            raise RuntimeError(
                f"Reference fingertip links missing from imported {HAND_ID} articulation: "
                f"{missing_fingertip_links}"
            )
        self._fingertip_body_indices = [
            self.hand.body_names.index(name)
            for name in self._reference_fingertip_link_names
        ]
        self._setup_scissors_handle_contact_gate()

    def _setup_scissors_handle_contact_gate(self) -> None:
        """Resolve the two handle frames without changing the default task."""

        self._scissors_handle_gate_enabled = bool(
            self.cfg.scissors_handle_contact_gate
        )
        if not self._scissors_handle_gate_enabled:
            return
        if not self.cfg.articulate_mode:
            raise RuntimeError("scissors_handle_contact_gate requires articulate_mode")

        required_object_bodies = {"bottom", "top"}
        missing_object_bodies = required_object_bodies - set(self.object.body_names)
        if missing_object_bodies:
            raise RuntimeError(
                "scissors_handle_contact_gate needs object bodies bottom/top; "
                f"missing {sorted(missing_object_bodies)} from {self.object.body_names}"
            )
        self._scissors_bottom_body_index = self.object.body_names.index("bottom")
        self._scissors_top_body_index = self.object.body_names.index("top")

        filter_names = list(self._articulated_contact_filter_body_names)
        missing_filter_bodies = required_object_bodies - set(filter_names)
        if missing_filter_bodies:
            raise RuntimeError(
                "Scissors contact filters do not expose bottom/top; "
                f"missing {sorted(missing_filter_bodies)} from {filter_names}"
            )
        self._scissors_bottom_filter_index = filter_names.index("bottom")
        self._scissors_top_filter_index = filter_names.index("top")

        thumb_names = list(THUMB_CONTACT_LINK_NAMES)
        other_names = list(OTHER_FINGER_CONTACT_LINK_NAMES)
        if not thumb_names or not other_names:
            raise RuntimeError("Scissors gate requires thumb and opposing-finger links")
        self._scissors_thumb_body_indices = [
            self.hand.body_names.index(name) for name in thumb_names
        ]
        self._scissors_other_body_indices = [
            self.hand.body_names.index(name) for name in other_names
        ]
        self._scissors_thumb_contact_sensors = tuple(
            self._articulated_contact_sensors[name] for name in thumb_names
        )
        self._scissors_other_contact_sensors = tuple(
            self._articulated_contact_sensors[name] for name in other_names
        )

        def vector(values: Sequence[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._scissors_top_hole_center = vector(
            self.cfg.scissors_top_hole_center
        )
        self._scissors_top_hole_radii_xy = vector(
            self.cfg.scissors_top_hole_radii_xy
        )
        self._scissors_bottom_hole_center = vector(
            self.cfg.scissors_bottom_hole_center
        )
        self._scissors_bottom_hole_radii_xy = vector(
            self.cfg.scissors_bottom_hole_radii_xy
        )
        if (
            (self._scissors_top_hole_radii_xy <= 0).any()
            or (self._scissors_bottom_hole_radii_xy <= 0).any()
            or self.cfg.scissors_hole_half_height <= 0
        ):
            raise ValueError("Scissors hole radii and half-height must be positive")

    def _current_fingertip_positions_w(self) -> torch.Tensor:
        fingertip_body_quat = self.hand.data.body_quat_w[
            :, self._fingertip_body_indices
        ]
        fingertip_body_pos = self.hand.data.body_pos_w[
            :, self._fingertip_body_indices
        ]
        fingertip_offsets = (
            self.fingertip_offsets
            if self._morphology_batch
            else self.fingertip_offsets[None, :, :].expand(self.num_envs, -1, -1)
        )
        fingertip_offset_w = quat_apply(
            fingertip_body_quat.reshape(-1, 4),
            fingertip_offsets.reshape(-1, 3),
        ).reshape(self.num_envs, len(self._fingertip_body_indices), 3)
        return fingertip_body_pos + fingertip_offset_w

    def _current_second_fingertip_positions_w(self) -> torch.Tensor:
        fingertip_body_quat = self.second_hand.data.body_quat_w[
            :, self._second_fingertip_body_indices
        ]
        fingertip_body_pos = self.second_hand.data.body_pos_w[
            :, self._second_fingertip_body_indices
        ]
        fingertip_offsets = self.second_fingertip_offsets[
            None, :, :
        ].expand(self.num_envs, -1, -1)
        fingertip_offset_w = quat_apply(
            fingertip_body_quat.reshape(-1, 4),
            fingertip_offsets.reshape(-1, 3),
        ).reshape(
            self.num_envs,
            len(self._second_fingertip_body_indices),
            3,
        )
        return fingertip_body_pos + fingertip_offset_w

    def _reference_at(
        self,
        tensor: torch.Tensor,
        phases: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Gather phase-aligned references for homogeneous or batched hands."""

        if not self._morphology_batch:
            return tensor[phases]
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return tensor[ids, phases]

    def _reference_velocity_at(
        self,
        tensor: torch.Tensor,
        phases: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Finite-difference velocity for a phase-aligned reference state.

        Random-phase resets must not place a moving hand or articulated object
        at an intermediate pose with zero velocity.  That creates a nonphysical
        impact transient exactly at contact-rich phases and makes the reset
        distribution differ from the demonstrated trajectory.
        """

        previous = torch.clamp(phases - 1, min=0)
        following = torch.clamp(phases + 1, max=self._reference_length - 1)
        duration = (following - previous).clamp(min=1).to(torch.float32)
        before = self._reference_at(tensor, previous, env_ids)
        after = self._reference_at(tensor, following, env_ids)
        while duration.ndim < before.ndim:
            duration = duration.unsqueeze(-1)
        return (after - before) * (self._reference_fps / duration)

    def _reference_root_velocity_at(
        self,
        poses: torch.Tensor,
        phases: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Finite-difference world linear/angular velocity for xyz+wxyz poses."""

        previous = torch.clamp(phases - 1, min=0)
        following = torch.clamp(phases + 1, max=self._reference_length - 1)
        duration_frames = (following - previous).clamp(min=1).to(torch.float32)
        before = self._reference_at(poses, previous, env_ids)
        after = self._reference_at(poses, following, env_ids)
        inverse_dt = self._reference_fps / duration_frames
        linear_velocity = (after[:, :3] - before[:, :3]) * inverse_dt[:, None]
        quaternion_delta = quat_mul(after[:, 3:7], quat_conjugate(before[:, 3:7]))
        # q and -q encode the same rotation.  Select the short arc before the
        # logarithm so interpolation across a sign flip cannot inject a huge
        # reset angular velocity.
        quaternion_delta = torch.where(
            quaternion_delta[:, :1] < 0.0,
            -quaternion_delta,
            quaternion_delta,
        )
        angular_velocity = axis_angle_from_quat(quaternion_delta) * inverse_dt[:, None]
        return torch.cat((linear_velocity, angular_velocity), dim=-1)

    def _environment_root(self, index: int) -> str:
        if not self._grouped_physics_replicas:
            return f"/World/envs/env_{index}"
        assert MORPHOLOGY_BATCH_MANIFEST is not None
        unique = int(MORPHOLOGY_BATCH_MANIFEST["unique_morphology_count"])
        replica_index, morphology_index = divmod(index, unique)
        return (
            f"/World/envs/env_{replica_index}/SuperEnvironment"
            f"/morph_{morphology_index:06d}"
        )

    def _environment_regex(self) -> str:
        if self._grouped_physics_replicas:
            return "/World/envs/env_.*/SuperEnvironment/morph_.*"
        return "/World/envs/env_.*"

    def _setup_scene(self) -> None:
        grouped_replicas = self._grouped_physics_replicas
        if grouped_replicas:
            self._spawn_grouped_morphology_sources()
            self._apply_morphology_batch_overlays(
                count=int(MORPHOLOGY_BATCH_MANIFEST["unique_morphology_count"])
            )
            for morphology_index in range(
                int(MORPHOLOGY_BATCH_MANIFEST["unique_morphology_count"])
            ):
                source_root = self._environment_root(morphology_index)
                self._validate_collision_coverage(
                    hand_root_path=f"{source_root}/Hand",
                    object_root_path=f"{source_root}/Object",
                )
            self._spawn_support_ground()
            self._spawn_dome_light()
            for index in range(
                int(MORPHOLOGY_BATCH_MANIFEST["unique_morphology_count"])
            ):
                self._filter_hand_support_collisions(
                    hand_root_path=f"{self._environment_root(index)}/Hand",
                    support_root_path="/World/ground",
                )
        else:
            self.hand = Articulation(self.cfg.hand_cfg)
            if self._bimanual_mode:
                self.second_hand = Articulation(self.cfg.second_hand_cfg)
            self._apply_morphology_batch_overlays()
        visual_manifest_path = ASSET_ROOT / "mano_visuals.json"
        if HAND_ID == "mano" and MANO_SIDE == "left" and visual_manifest_path.is_file():
            visual_manifest = json.loads(visual_manifest_path.read_text(encoding="utf-8"))
            for link_name, relative_usd_path in visual_manifest.items():
                visual_cfg = sim_utils.UsdFileCfg(
                    usd_path=str(ASSET_ROOT / relative_usd_path),
                )
                visual_cfg.func(
                    f"/World/envs/env_0/Hand/{link_name}/visual_overlay",
                    visual_cfg,
                )
        if (
            self._bimanual_mode
            and SECOND_MANO_SIDE == "left"
            and visual_manifest_path.is_file()
        ):
            visual_manifest = json.loads(
                visual_manifest_path.read_text(encoding="utf-8")
            )
            for link_name, relative_usd_path in visual_manifest.items():
                visual_cfg = sim_utils.UsdFileCfg(
                    usd_path=str(ASSET_ROOT / relative_usd_path),
                )
                visual_cfg.func(
                    f"/World/envs/env_0/SecondHand/{link_name}/visual_overlay",
                    visual_cfg,
                )
        if grouped_replicas:
            self._clone_grouped_morphology_environments()
            print("[MORPHOLOGY_GROUPED_STAGE] clone_complete", flush=True)
            hand_cfg = copy.deepcopy(self.cfg.hand_cfg)
            hand_cfg.spawn = None
            object_cfg = copy.deepcopy(
                self.cfg.articulated_object_cfg
                if self.cfg.articulate_mode
                else self.cfg.object_cfg
            )
            object_cfg.spawn = None
            print("[MORPHOLOGY_GROUPED_STAGE] constructing_articulation", flush=True)
            self.hand = Articulation(hand_cfg)
            print("[MORPHOLOGY_GROUPED_STAGE] articulation_constructed", flush=True)
            self.object = (
                Articulation(object_cfg)
                if self.cfg.articulate_mode
                else RigidObject(object_cfg)
            )
            print("[MORPHOLOGY_GROUPED_STAGE] object_constructed", flush=True)
        else:
            self.object = (
                Articulation(self.cfg.articulated_object_cfg)
                if self.cfg.articulate_mode
                else RigidObject(self.cfg.object_cfg)
            )
        env_regex = self._environment_regex()
        if self.cfg.articulate_mode:
            # ContactSensor filtering is one-to-many. Report one hand body per
            # sensor and filter it against every rigid link of the object.
            # Discover exact rigid-body suffixes instead of using a wildcard:
            # the imported USD also contains joints and grouping Xforms.
            first_object_root = f"{self._environment_root(0)}/Object"
            object_body_prims = get_all_matching_child_prims(
                first_object_root,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
                stage=self.scene.stage,
                traverse_instance_prims=True,
            )
            object_filter_paths = [
                f"{env_regex}/Object{str(prim.GetPath())[len(first_object_root):]}"
                for prim in object_body_prims
            ]
            self._articulated_contact_filter_body_names = [
                prim.GetName() for prim in object_body_prims
            ]
            if not object_filter_paths:
                raise RuntimeError(
                    f"Articulated object has no rigid links below {first_object_root}"
                )
            scissors_region_links = set(
                (*THUMB_CONTACT_LINK_NAMES, *OTHER_FINGER_CONTACT_LINK_NAMES)
            )
            self._articulated_contact_sensors = {
                link_name: ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{env_regex}/Hand/{link_name}",
                        update_period=0.0,
                        history_length=0,
                        filter_prim_paths_expr=object_filter_paths,
                        track_contact_points=bool(
                            self.cfg.scissors_handle_contact_gate
                            and link_name in scissors_region_links
                        ),
                        max_contact_data_count_per_prim=(
                            32
                            if self.cfg.scissors_handle_contact_gate
                            and link_name in scissors_region_links
                            else 4
                        ),
                    )
                )
                for link_name in ALL_HAND_CONTACT_LINK_NAMES
            }
            self._thumb_contact_sensors = tuple(
                self._articulated_contact_sensors[name]
                for name in THUMB_CONTACT_LINK_NAMES
            )
            self._other_finger_contact_sensors = tuple(
                self._articulated_contact_sensors[name]
                for name in OTHER_FINGER_CONTACT_LINK_NAMES
            )
            self._all_hand_contact_sensors = tuple(
                self._articulated_contact_sensors.values()
            )
            if self._bimanual_mode:
                self._second_articulated_contact_sensors = {
                    link_name: ContactSensor(
                        ContactSensorCfg(
                            prim_path=f"{env_regex}/SecondHand/{link_name}",
                            update_period=0.0,
                            history_length=0,
                            filter_prim_paths_expr=object_filter_paths,
                            max_contact_data_count_per_prim=4,
                        )
                    )
                    for link_name in SECOND_CONTACT_LINK_NAMES
                }
                self._all_hand_contact_sensors += tuple(
                    self._second_articulated_contact_sensors.values()
                )
        else:
            # A rigid object has one reporting body, filtered against groups
            # of hand links exactly as in the original environment.
            def object_contact_sensor(
                link_names: Sequence[str], *, include_second: bool = False
            ) -> ContactSensor:
                filter_paths = [
                    f"{env_regex}/Hand/{name}" for name in link_names
                ]
                if include_second and self._bimanual_mode:
                    filter_paths.extend(
                        f"{env_regex}/SecondHand/{name}"
                        for name in SECOND_CONTACT_LINK_NAMES
                    )
                return ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{env_regex}/Object",
                        update_period=0.0,
                        history_length=0,
                        filter_prim_paths_expr=filter_paths,
                    )
                )

            self._thumb_contact_sensor = object_contact_sensor(
                THUMB_CONTACT_LINK_NAMES
            )
            self._other_finger_contact_sensor = object_contact_sensor(
                OTHER_FINGER_CONTACT_LINK_NAMES
            )
            self._all_hand_contact_sensor = object_contact_sensor(
                ALL_HAND_CONTACT_LINK_NAMES,
                include_second=True,
            )
        if not grouped_replicas:
            self._validate_collision_coverage(
                hand_root_path=f"{self._environment_root(0)}/Hand",
                object_root_path=f"{self._environment_root(0)}/Object",
            )
            if self._bimanual_mode:
                self._validate_collision_coverage(
                    hand_root_path=f"{self._environment_root(0)}/SecondHand",
                    object_root_path=f"{self._environment_root(0)}/Object",
                    contact_link_names=SECOND_CONTACT_LINK_NAMES,
                    hand_label=f"mano_{SECOND_MANO_SIDE}",
                )
        if not grouped_replicas:
            self._spawn_support_ground()
            hand_roots = (
                [f"/World/envs/env_{index}/Hand" for index in range(self.num_envs)]
                if self._morphology_batch
                else ["/World/envs/env_0/Hand"]
            )
            for hand_root in hand_roots:
                self._filter_hand_support_collisions(
                    hand_root_path=hand_root,
                    support_root_path="/World/ground",
                )
            if self._bimanual_mode:
                self._filter_hand_support_collisions(
                    hand_root_path="/World/envs/env_0/SecondHand",
                    support_root_path="/World/ground",
                )
        # InteractiveScene already creates all independent environment Xforms
        # before _setup_scene when replicate_physics=False. Re-cloning here
        # would overwrite the deterministic MultiUsdFileCfg assignments.
        if self.cfg.scene.replicate_physics and not grouped_replicas:
            self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        if self._bimanual_mode:
            self.scene.articulations["second_hand"] = self.second_hand
        if self.cfg.articulate_mode:
            self.scene.articulations["object"] = self.object
        else:
            self.scene.rigid_objects["object"] = self.object
        if self.cfg.articulate_mode:
            for link_name, sensor in self._articulated_contact_sensors.items():
                self.scene.sensors[f"object_contact_{link_name}"] = sensor
            if self._bimanual_mode:
                for link_name, sensor in (
                    self._second_articulated_contact_sensors.items()
                ):
                    self.scene.sensors[
                        f"object_contact_second_{link_name}"
                    ] = sensor
        else:
            self.scene.sensors["object_thumb_contact"] = self._thumb_contact_sensor
            self.scene.sensors["object_other_finger_contact"] = self._other_finger_contact_sensor
            self.scene.sensors["object_all_hand_contact"] = self._all_hand_contact_sensor
        if grouped_replicas:
            print("[MORPHOLOGY_GROUPED_STAGE] setup_scene_complete", flush=True)
        if not grouped_replicas:
            self._spawn_dome_light()

    def _spawn_dome_light(self) -> None:
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1800.0, color=(0.85, 0.85, 0.85)
        )
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_support_ground(self) -> None:
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                color=(0.25, 0.27, 0.30),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

    def _spawn_grouped_morphology_sources(self) -> None:
        """Spawn only one physical source environment per unique morphology."""

        manifest = MORPHOLOGY_BATCH_MANIFEST
        assert manifest is not None
        unique = int(manifest["unique_morphology_count"])
        super_usd = manifest.get("hand_super_environment_usd")
        if not super_usd:
            raise ValueError("grouped replication requires a static hand super-environment USD")
        local_origins = np.asarray(
            manifest["hand_super_environment_origins"][:unique], dtype=np.float32
        )
        source_replica = "/World/envs/env_0/SuperEnvironment"
        # The hands are already referenced inside this USD. Articulation below
        # uses spawn=None, so its spawn overrides would otherwise never run.
        # Apply the same physics configuration as ordinary hand spawning before
        # PhysX parses/clones the source (the objects are spawned separately).
        hand_spawn = self.cfg.hand_cfg.spawn
        super_cfg = sim_utils.UsdFileCfg(
            usd_path=str(Path(super_usd).resolve()),
            rigid_props=hand_spawn.rigid_props,
            collision_props=hand_spawn.collision_props,
            mass_props=hand_spawn.mass_props,
            articulation_props=hand_spawn.articulation_props,
            fixed_tendons_props=hand_spawn.fixed_tendons_props,
            spatial_tendons_props=hand_spawn.spatial_tendons_props,
            joint_drive_props=hand_spawn.joint_drive_props,
            activate_contact_sensors=hand_spawn.activate_contact_sensors,
        )
        super_cfg.func(source_replica, super_cfg)
        source_parent_expression = (
            "/World/envs/env_0/SuperEnvironment/morph_.*"
        )
        object_spawn = copy.deepcopy(
            self.cfg.articulated_object_cfg.spawn
            if getattr(self.cfg, "articulate_mode", False)
            else self.cfg.object_cfg.spawn
        )
        object_spawn.func(
            f"{source_parent_expression}/Object",
            object_spawn,
            replicate_physics=False,
            clone_in_fabric=False,
        )
        print(
            "[MORPHOLOGY_GROUPED_SOURCES] "
            f"unique={unique} total_envs={self.num_envs}",
            flush=True,
        )
        self._morphology_local_origins = local_origins

    def _clone_grouped_morphology_environments(self) -> None:
        """Clone one heterogeneous super-environment into PPO replicas.

        The source contains every unique morphology. Isaac Sim therefore sees
        one homogeneous replication operation even though each replica block
        contains heterogeneous hands.
        """

        manifest = MORPHOLOGY_BATCH_MANIFEST
        assert manifest is not None
        unique = int(manifest["unique_morphology_count"])
        replicas = int(manifest["morphology_replicas"])
        if unique * replicas != self.num_envs:
            raise ValueError(
                f"grouped morphology shape {unique}x{replicas} != {self.num_envs}"
            )
        local_origins = self._morphology_local_origins
        self.scene.clone_environments(copy_from_source=False)
        block_origins = self.scene.env_origins.detach().cpu().numpy()
        if len(block_origins) != replicas:
            raise RuntimeError(
                f"official scene clone returned {len(block_origins)} blocks, "
                f"expected {replicas}"
            )
        flattened_origins = np.concatenate(
            [local_origins + block_origin for block_origin in block_origins], axis=0
        )
        self.scene._default_env_origins = torch.as_tensor(
            flattened_origins, device=self.device, dtype=torch.float32
        )
        print(
            "[MORPHOLOGY_GROUPED_PHYSX_CLONES] "
            f"sources={unique} replicas={replicas} total={self.num_envs}",
            flush=True,
        )

    def _apply_morphology_batch_overlays(self, count: int | None = None) -> None:
        """Author per-env continuous morphology before PhysX parses the stage.

        The discrete palm mesh comes from one of the canonical prototype USDs.
        Only the affine link geometry and joint-frame opinions vary per env.
        Keeping those opinions in the scene root lets the multi-asset spawner
        reuse 32 prototypes for thousands of candidates.
        """

        manifest = MORPHOLOGY_BATCH_MANIFEST
        if not (
            self._morphology_batch
            and manifest is not None
            and manifest.get("runtime_parametric_overlays", False)
        ):
            return
        count = len(manifest["hand_usd_paths"]) if count is None else count
        required = (
            "parametric_link_names",
            "parametric_relative_transforms",
            "parametric_link_translations",
            "parametric_joint_names",
            "parametric_joint_local_positions",
        )
        for key in required:
            if len(manifest[key]) < count:
                raise ValueError(f"runtime morphology overlay count mismatch for {key}")
        stage = self.scene.stage
        resolved_links: list[list[str]] = []
        resolved_joints: list[list[str]] = []
        source_indices = (
            list(range(count))
            if self._grouped_physics_replicas
            else list(range(count))
        )
        for logical_index, manifest_index in enumerate(source_indices):
            hand_root = f"{self._environment_root(manifest_index)}/Hand"
            hand_prim = stage.GetPrimAtPath(hand_root)
            if not hand_prim.IsValid():
                raise ValueError(f"missing spawned morphology hand at {hand_root}")
            # The importer can nest instance roots at the hand, link, and
            # geometry levels. Recursively deinstance the candidate subtree
            # so no collision descendant remains a read-only instance proxy.
            pending = [hand_prim]
            while pending:
                prim = pending.pop()
                if prim.IsInstance() or prim.IsInstanceable():
                    prim.SetInstanceable(False)
                pending.extend(prim.GetChildren())
            link_children = {child.GetName() for child in hand_prim.GetChildren()}
            joint_scope = stage.GetPrimAtPath(f"{hand_root}/joints")
            joint_children = (
                {child.GetName() for child in joint_scope.GetChildren()}
                if joint_scope.IsValid()
                else set()
            )

            def resolve(raw_name: str, available: set[str], kind: str) -> str:
                if raw_name in available:
                    return raw_name
                matches = [
                    value
                    for value in available
                    if value.endswith(f"__{raw_name}")
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"cannot uniquely resolve morphology {kind} {raw_name} "
                        f"at env {manifest_index}: {matches}"
                    )
                return matches[0]

            resolved_links.append(
                [
                    resolve(name, link_children, "link")
                    for name in manifest["parametric_link_names"][manifest_index]
                ]
            )
            resolved_joints.append(
                [
                    resolve(name, joint_children, "joint")
                    for name in manifest["parametric_joint_names"][manifest_index]
                ]
            )
        with Sdf.ChangeBlock():
            for logical_index, manifest_index in enumerate(source_indices):
                hand_root = f"{self._environment_root(manifest_index)}/Hand"
                links = resolved_links[logical_index]
                transforms = manifest["parametric_relative_transforms"][manifest_index]
                translations = manifest["parametric_link_translations"][manifest_index]
                if not (len(links) == len(transforms) == len(translations)):
                    raise ValueError(
                        "runtime morphology link overlay mismatch at env "
                        f"{manifest_index}"
                    )
                for link_name, transform, translation in zip(
                    links, transforms, translations, strict=True
                ):
                    link_path = f"{hand_root}/{link_name}"
                    link = stage.GetPrimAtPath(link_path)
                    link.GetAttribute("xformOp:translate").Set(
                        Gf.Vec3d(*translation)
                    )
                    matrix = np.asarray(transform, dtype=np.float64)
                    if not np.allclose(matrix, np.eye(4), atol=1.0e-12):
                        collision = stage.GetPrimAtPath(
                            f"{link_path}/collisions"
                        )
                        matrix_value = Gf.Matrix4d(
                            *matrix.reshape(-1).tolist()
                        )
                        # Search assets contain only ``collisions`` while
                        # inspection assets also contain ``visuals``. Apply
                        # the identical morphology transform to both scopes so
                        # adding render geometry cannot change the collision
                        # geometry that was evaluated during optimization.
                        for geometry_scope in ("collisions", "visuals"):
                            geometry = stage.GetPrimAtPath(
                                f"{link_path}/{geometry_scope}"
                            )
                            if not geometry.IsValid():
                                continue
                            xform = UsdGeom.Xformable(geometry)
                            xform.ClearXformOpOrder()
                            xform.AddTransformOp(opSuffix="morphology").Set(
                                matrix_value
                            )
                            from dexcodesign.morphology.wuji_palm_collision import preserve_fixed_base
                            preserve_fixed_base(stage, geometry.GetPath(), matrix)
                joint_names = resolved_joints[logical_index]
                positions = manifest["parametric_joint_local_positions"][manifest_index]
                if len(joint_names) != len(positions):
                    raise ValueError(
                        f"runtime morphology joint overlay mismatch at env {index}"
                    )
                for joint_name, position in zip(
                    joint_names, positions, strict=True
                ):
                    joint = stage.GetPrimAtPath(
                        f"{hand_root}/joints/{joint_name}"
                    )
                    joint.GetAttribute("physics:localPos0").Set(
                        Gf.Vec3f(*position)
                    )
        print(
            "[MORPHOLOGY_RUNTIME_OVERLAYS] "
            f"authored={count} prototypes={len(set(manifest['hand_usd_paths']))}",
            flush=True,
        )

    def _validate_collision_coverage(
        self,
        hand_root_path: str,
        object_root_path: str,
        contact_link_names: Sequence[str] = ALL_HAND_CONTACT_LINK_NAMES,
        hand_label: str = HAND_ID,
    ) -> None:
        """Fail before simulation if a physical hand part or object lacks collision."""

        stage = self.scene.stage
        missing_links: list[str] = []
        collider_count = 0
        for link_name in contact_link_names:
            link_path = f"{hand_root_path}/{link_name}"
            link = stage.GetPrimAtPath(link_path)
            colliders = []
            if link.IsValid():
                colliders = get_all_matching_child_prims(
                    link_path,
                    predicate=lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI),
                    stage=stage,
                    traverse_instance_prims=True,
                )
            if not colliders:
                missing_links.append(link_name)
            collider_count += len(colliders)
        if missing_links:
            raise RuntimeError(
                f"{hand_label} has physical hand parts without collision USD prims: "
                f"{missing_links}"
            )

        object_root = stage.GetPrimAtPath(object_root_path)
        object_colliders = []
        if object_root.IsValid():
            object_colliders = get_all_matching_child_prims(
                object_root_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI),
                stage=stage,
                traverse_instance_prims=True,
            )
        if not object_colliders:
            raise RuntimeError(f"Object has no collision USD prim below {object_root_path}")
        object_approximations = sorted(
            {
                str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                for prim in object_colliders
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            }
        )
        allowed_object_approximations = (
            {"convexHull", "convexDecomposition"}
            if self.cfg.articulate_mode
            else {"convexDecomposition"}
        )
        if (
            not object_approximations
            or not set(object_approximations).issubset(
                allowed_object_approximations
            )
        ):
            raise RuntimeError(
                "Object collision approximation is incompatible with "
                f"articulate_mode={self.cfg.articulate_mode}: "
                f"{object_approximations or ['missing MeshCollisionAPI']}"
            )
        self._object_collision_approximations = object_approximations
        print(
            f"[HAND_COLLISION_COVERAGE:{hand_label}] "
            f"physical_links={len(contact_link_names)} "
            f"hand_colliders={collider_count} "
            f"object_colliders={len(object_colliders)} "
            f"object_approximation={object_approximations[0]}"
        )

    def _filter_hand_support_collisions(
        self,
        hand_root_path: str,
        support_root_path: str,
    ) -> None:
        """Disable only hand-support contacts while preserving object contacts.

        EgoEngine lets the hand interact with the manipulated object but not
        with the static table/support. The object is intentionally untouched,
        so object-hand and object-support collision pairs remain active.
        """
        stage = self.scene.stage
        hand_root = stage.GetPrimAtPath(hand_root_path)
        support_root = stage.GetPrimAtPath(support_root_path)
        if not hand_root.IsValid() or not support_root.IsValid():
            raise RuntimeError(
                "Cannot configure hand-support collision filtering: "
                f"hand={hand_root_path}, support={support_root_path}"
            )

        support_colliders = [
            prim.GetPath()
            for prim in Usd.PrimRange(support_root)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not support_colliders:
            raise RuntimeError(f"No collision shapes found below {support_root_path}")

        filtered_prims = 0
        for prim in Usd.PrimRange(hand_root):
            if not (
                prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                continue
            relationship = (
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            )
            for support_collider in support_colliders:
                relationship.AddTarget(support_collider)
            filtered_prims += 1

        if filtered_prims == 0:
            raise RuntimeError(f"No hand collision bodies found below {hand_root_path}")
        if not self._morphology_batch or hand_root_path.endswith("/env_0/Hand"):
            suffix = (
                f" across {self.num_envs} heterogeneous environments"
                if self._morphology_batch
                else ""
            )
            print(
                f"[HAND_COLLISION_FILTER:{HAND_ID}] "
                f"disabled hand-support pairs for {filtered_prims} hand prims"
                f"{suffix}; object-hand and object-support pairs remain enabled"
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        base_targets = self._reference_at(self.reference_hand_ctrl, self.phase_buf)
        primary_actions = self.actions[:, : self._primary_action_dim]
        active_residual = (
            self.residual_scale[: self._primary_action_dim] * primary_actions
        )
        control_residual = active_residual @ self.action_to_control.T
        targets = base_targets + control_residual
        self.joint_targets = torch.clamp(
            targets,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )
        if self._bimanual_mode:
            second_base_targets = self._reference_at(
                self.second_reference_hand_ctrl,
                self.phase_buf,
            )
            second_actions = self.actions[:, self._primary_action_dim :]
            second_residual = (
                self.residual_scale[self._primary_action_dim :]
                * second_actions
            )
            second_control_residual = (
                second_residual @ self.second_action_to_control.T
            )
            second_targets = (
                second_base_targets + second_control_residual
            )
            self.second_joint_targets = torch.clamp(
                second_targets,
                self.second_joint_lower_limits,
                self.second_joint_upper_limits,
            )
        if self._capture_enabled:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._capture_actions[env_ids, self.phase_buf] = self.actions
            self._capture_joint_targets[env_ids, self.phase_buf] = self.joint_targets
            if self._bimanual_mode:
                self._capture_second_joint_targets[
                    env_ids, self.phase_buf
                ] = self.second_joint_targets

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self.joint_targets)
        if self._bimanual_mode:
            self.second_hand.set_joint_position_target(
                self.second_joint_targets
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        object_pose = torch.cat((object_pos, self.object.data.root_quat_w), dim=-1)
        fingertip_pos = (
            self._current_fingertip_positions_w()
            - self.scene.env_origins[:, None, :]
        )
        goal_fingertip_pose = self._reference_at(
            self.reference_fingertip_pose, self.phase_buf
        )
        goal_object_pose = self._reference_at(
            self.reference_object_pose, self.phase_buf
        )
        if self.cfg.articulate_mode:
            object_pose = torch.cat((object_pose, self.object.data.joint_pos), dim=-1)
            goal_object_pose = torch.cat(
                (
                    goal_object_pose,
                    self._reference_at(self.reference_object_joint, self.phase_buf),
                ),
                dim=-1,
            )
        if self.cfg.log_rollout_diagnostics and self.num_envs == 1:
            phase_index = int(self.phase_buf[0].item())
            if phase_index != self._last_diagnostic_phase and (
                phase_index < 3 or phase_index % 110 == 0
            ):
                actual_q = self.hand.data.joint_pos[0]
                target_q = self.joint_targets[0]
                if self._morphology_batch:
                    reference_q = self.reference_hand_q[0, phase_index]
                    reference_object_pos = self.reference_object_pose[0, phase_index, :3]
                else:
                    reference_q = self.reference_hand_q[phase_index]
                    reference_object_pos = self.reference_object_pose[phase_index, :3]
                palm_pos = self.hand.data.body_pos_w[0, self._palm_body_index]
                middle_tip_pos = self.hand.data.body_pos_w[0, self._middle_tip_body_index]
                print(
                    f"[HAND_ROLLOUT:{HAND_ID}] "
                    f"phase={phase_index} "
                    f"hand_actual_root={actual_q[:6].detach().cpu().tolist()} "
                    f"hand_target_root={target_q[:6].detach().cpu().tolist()} "
                    f"hand_reference_root={reference_q[:6].detach().cpu().tolist()} "
                    f"palm_pos={palm_pos.detach().cpu().tolist()} "
                    f"middle_tip_pos={middle_tip_pos.detach().cpu().tolist()} "
                    f"object_actual_pos={object_pos[0].detach().cpu().tolist()} "
                    f"object_reference_pos={reference_object_pos.detach().cpu().tolist()}"
                )
                self._last_diagnostic_phase = phase_index
        observation = torch.cat(
            (
                self.hand.data.joint_pos,
                fingertip_pos.flatten(start_dim=1),
                object_pose,
                goal_fingertip_pose.flatten(start_dim=1),
                self._reference_at(self.reference_hand_q, self.phase_buf),
                goal_object_pose,
            ),
            dim=-1,
        )
        if self._bimanual_mode:
            second_fingertip_pos = (
                self._current_second_fingertip_positions_w()
                - self.scene.env_origins[:, None, :]
            )
            second_goal_fingertip_pose = self._reference_at(
                self.second_reference_fingertip_pose,
                self.phase_buf,
            )
            second_observation = torch.cat(
                (
                    self.second_hand.data.joint_pos,
                    second_fingertip_pos.flatten(start_dim=1),
                    second_goal_fingertip_pose.flatten(start_dim=1),
                    self._reference_at(
                        self.second_reference_hand_q,
                        self.phase_buf,
                    ),
                ),
                dim=-1,
            )
            observation = torch.cat(
                (observation, second_observation), dim=-1
            )
        if self.morphology_context is not None:
            observation = torch.cat((observation, self.morphology_context), dim=-1)
        return {"policy": observation}

    def _compute_object_errors(self) -> None:
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        reference_pose = self._reference_at(
            self.reference_object_pose, self.phase_buf
        )
        self._object_position_error = torch.linalg.vector_norm(
            object_pos - reference_pose[:, :3], dim=-1
        )
        self._object_rotation_error = quat_error_magnitude(
            self.object.data.root_quat_w,
            reference_pose[:, 3:7],
        )
        if self.cfg.articulate_mode:
            reference_joint = self._reference_at(
                self.reference_object_joint, self.phase_buf
            )
            joint_delta = self.object.data.joint_pos - reference_joint
            joint_delta = torch.atan2(torch.sin(joint_delta), torch.cos(joint_delta))
            self._object_articulation_error = torch.linalg.vector_norm(
                joint_delta, dim=-1
            )
        else:
            self._object_articulation_error = torch.zeros_like(
                self._object_position_error
            )

    def _contact_sensor_force(
        self, sensors: ContactSensor | Sequence[ContactSensor]
    ) -> torch.Tensor:
        """Return the maximum filtered force over bodies and object links."""

        if isinstance(sensors, ContactSensor):
            sensors = (sensors,)
        forces = []
        for sensor in sensors:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is not None:
                forces.append(
                    torch.linalg.vector_norm(
                        force_matrix.reshape(self.num_envs, -1, 3), dim=-1
                    ).amax(dim=-1)
                )
        if not forces:
            return torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        return torch.stack(forces, dim=-1).amax(dim=-1)

    def _contact_sensor_forces_by_filter(
        self, sensors: ContactSensor | Sequence[ContactSensor]
    ) -> torch.Tensor:
        """Maximum force per filtered object link over reporting hand bodies."""

        if isinstance(sensors, ContactSensor):
            sensors = (sensors,)
        forces = []
        for sensor in sensors:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is not None:
                # (environment, reporting body, filtered object body, xyz)
                forces.append(
                    torch.linalg.vector_norm(force_matrix, dim=-1).amax(dim=1)
                )
        if not forces:
            return torch.zeros(
                (self.num_envs, 0), dtype=torch.float32, device=self.device
            )
        return torch.stack(forces, dim=-1).amax(dim=-1)

    def _compute_scissors_handle_contact(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the union of the two hole ROIs to the existing contact reward."""

        def group_contact(
            sensors: Sequence[ContactSensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            in_either_hole = []
            valid_contact = []
            all_forces = []
            for sensor in sensors:
                if sensor.data.contact_pos_w is None:
                    raise RuntimeError(
                        "Scissors handle reward requires track_contact_points=True"
                    )
                force_by_filter = self._contact_sensor_forces_by_filter(sensor)
                for body_index, filter_index, center, radii_xy in (
                    (
                        self._scissors_top_body_index,
                        self._scissors_top_filter_index,
                        self._scissors_top_hole_center,
                        self._scissors_top_hole_radii_xy,
                    ),
                    (
                        self._scissors_bottom_body_index,
                        self._scissors_bottom_filter_index,
                        self._scissors_bottom_hole_center,
                        self._scissors_bottom_hole_radii_xy,
                    ),
                ):
                    point_w = sensor.data.contact_pos_w[:, 0, filter_index]
                    finite = torch.isfinite(point_w).all(dim=-1)
                    point_local = quat_apply(
                        quat_conjugate(self.object.data.body_quat_w[:, body_index]),
                        torch.nan_to_num(point_w)
                        - self.object.data.body_pos_w[:, body_index],
                    )
                    inside = finite & _points_in_elliptical_prism(
                        point_local,
                        center,
                        radii_xy,
                        self.cfg.scissors_hole_half_height,
                    )
                    force = force_by_filter[:, filter_index]
                    in_either_hole.append(inside)
                    valid_contact.append(
                        inside & (force > self.cfg.contact_force_threshold)
                    )
                    all_forces.append(force)
            return (
                torch.stack(in_either_hole, dim=1).any(dim=1),
                torch.stack(valid_contact, dim=1).any(dim=1),
                torch.stack(all_forces, dim=1).amax(dim=1),
            )

        thumb_in_hole, thumb_contact, thumb_force = group_contact(
            self._scissors_thumb_contact_sensors
        )
        other_in_hole, other_contact, other_force = group_contact(
            self._scissors_other_contact_sensors
        )
        self._last_thumb_in_scissors_hole = thumb_in_hole
        self._last_other_finger_in_scissors_hole = other_in_hole
        return (
            thumb_contact,
            other_contact,
            thumb_contact & other_contact,
            thumb_force,
            other_force,
        )

    def _compute_pinch_contact(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._scissors_handle_gate_enabled:
            return self._compute_scissors_handle_contact()
        if self.cfg.articulate_mode and self.cfg.require_cross_link_pinch:
            thumb_link_force = self._contact_sensor_forces_by_filter(
                self._thumb_contact_sensors
            )
            other_link_force = self._contact_sensor_forces_by_filter(
                self._other_finger_contact_sensors
            )
            if thumb_link_force.shape[1] < 2:
                raise RuntimeError(
                    "require_cross_link_pinch needs at least two articulated object links"
                )
            thumb_by_link = thumb_link_force > self.cfg.contact_force_threshold
            other_by_link = other_link_force > self.cfg.contact_force_threshold
            different_link = ~torch.eye(
                thumb_link_force.shape[1], dtype=torch.bool, device=self.device
            )
            cross_link_pairs = (
                thumb_by_link[:, :, None]
                & other_by_link[:, None, :]
                & different_link[None, :, :]
            )
            thumb_force = thumb_link_force.amax(dim=1)
            other_finger_force = other_link_force.amax(dim=1)
            thumb_contact = thumb_by_link.any(dim=1)
            other_finger_contact = other_by_link.any(dim=1)
            return (
                thumb_contact,
                other_finger_contact,
                cross_link_pairs.any(dim=(1, 2)),
                thumb_force,
                other_finger_force,
            )
        thumb_force = self._contact_sensor_force(
            self._thumb_contact_sensors
            if self.cfg.articulate_mode
            else self._thumb_contact_sensor
        )
        other_finger_force = self._contact_sensor_force(
            self._other_finger_contact_sensors
            if self.cfg.articulate_mode
            else self._other_finger_contact_sensor
        )
        thumb_contact = thumb_force > self.cfg.contact_force_threshold
        other_finger_contact = other_finger_force > self.cfg.contact_force_threshold
        return (
            thumb_contact,
            other_finger_contact,
            thumb_contact & other_finger_contact,
            thumb_force,
            other_finger_force,
        )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_object_errors()
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        reference_pos = self._reference_at(
            self.reference_object_pose, self.phase_buf
        )[:, :3]
        position_delta = object_pos - reference_pos
        anisotropic_position_error = torch.sqrt(
            position_delta[:, :2].square().sum(dim=-1)
            + (
                self.cfg.object_z_reward_multiplier * position_delta[:, 2]
            ).square()
        )
        weighted_position_error = (
            self.cfg.object_position_reward_weight
            * anisotropic_position_error
        )
        weighted_rotation_error = (
            self.cfg.object_rotation_reward_weight
            * self._object_rotation_error
        )
        weighted_articulation_error = (
            self.cfg.object_articulation_reward_weight
            * self._object_articulation_error
        )
        pose_tracking_error = torch.sqrt(
            weighted_position_error.square()
            + weighted_rotation_error.square()
            + weighted_articulation_error.square()
        )
        reward_offset_c = math.sqrt(
            (
                self.cfg.object_position_reward_weight
                * self.cfg.object_failure_distance
            )
            ** 2
            + (
                self.cfg.object_rotation_reward_weight
                * self.cfg.object_failure_orientation
            )
            ** 2
            + (
                self.cfg.object_articulation_reward_weight
                * self.cfg.object_failure_articulation
            )
            ** 2
            * float(self.cfg.articulate_mode)
        )
        pose_tracking_reward = (
            self.cfg.pose_tracking_reward_scale * torch.exp(-pose_tracking_error)
            if self.cfg.exponential_pose_reward
            else reward_offset_c - pose_tracking_error
        )
        (
            thumb_contact,
            other_finger_contact,
            pinch_contact,
            thumb_force,
            other_finger_force,
        ) = self._compute_pinch_contact()
        all_hand_force = self._contact_sensor_force(
            self._all_hand_contact_sensors
            if self.cfg.articulate_mode
            else self._all_hand_contact_sensor
        )
        all_hand_contact = all_hand_force > self.cfg.contact_force_threshold
        if self.cfg.contact_tracking_sigma > 0.0:
            contact_tracking_quality = torch.exp(
                -self._object_position_error / self.cfg.contact_tracking_sigma
            )
        else:
            contact_tracking_quality = torch.ones_like(self._object_position_error)
        contact_reward = (
            pinch_contact.to(torch.float32)
            * contact_tracking_quality
            * self.cfg.contact_reward_weight
        )
        settled_object_height = self._reference_at(
            self.reference_object_pose,
            torch.zeros_like(self.phase_buf),
        )[:, 2]
        object_airborne = (
            object_pos[:, 2]
            > settled_object_height + self.cfg.object_airborne_clearance
        )
        object_airborne_reward = (
            object_airborne.to(torch.float32)
            * self.cfg.object_airborne_reward_weight
        )
        excess_contact_force = torch.relu(
            all_hand_force - self.cfg.contact_force_safe_threshold
        ) / 100.0
        contact_force_penalty = (
            excess_contact_force.square() * self.cfg.contact_force_penalty_weight
        )
        residual_action_penalty = (
            self.actions.square().mean(dim=-1)
            * self.cfg.residual_action_penalty_weight
        )
        total_reward = (
            pose_tracking_reward
            + contact_reward
            + object_airborne_reward
            - contact_force_penalty
            - residual_action_penalty
        )
        # Keep per-environment components available after DirectRLEnv performs
        # its automatic reset. Morphology evaluation consumes the exact same
        # reward tensors returned to PPO, without reconstructing a proxy.
        self._last_pose_tracking_reward = pose_tracking_reward
        self._last_contact_reward = contact_reward
        self._last_pinch_contact = pinch_contact
        self._last_thumb_contact_force = thumb_force
        self._last_other_finger_contact_force = other_finger_force
        self._pose_episode_return += pose_tracking_reward
        if self._capture_enabled:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._capture_pose_reward[env_ids, self.phase_buf] = pose_tracking_reward
            self._capture_contact_reward[env_ids, self.phase_buf] = contact_reward
        finger_residual = (
            self.actions[:, self._finger_action_indices]
            * self.residual_scale[self._finger_action_indices]
        ).abs()
        log = {
            "object_position_error_m": self._object_position_error.mean(),
            "object_rotation_error_rad": self._object_rotation_error.mean(),
            "object_articulation_error_rad": self._object_articulation_error.mean(),
            "weighted_object_position_error": weighted_position_error.mean(),
            "weighted_object_rotation_error": weighted_rotation_error.mean(),
            "weighted_object_articulation_error": weighted_articulation_error.mean(),
            "pose_tracking_error": pose_tracking_error.mean(),
            "reward_offset_c": reward_offset_c,
            "pose_tracking_reward": pose_tracking_reward.mean(),
            "thumb_object_contact": thumb_contact.to(torch.float32).mean(),
            "other_finger_object_contact": other_finger_contact.to(torch.float32).mean(),
            "pinch_contact_reward": contact_reward.mean(),
            "object_airborne": object_airborne.to(torch.float32).mean(),
            "object_airborne_reward": object_airborne_reward.mean(),
            "contact_tracking_quality": contact_tracking_quality.mean(),
            "residual_action_penalty": residual_action_penalty.mean(),
            "thumb_object_contact_force_n": thumb_force.mean(),
            "other_finger_object_contact_force_n": other_finger_force.mean(),
            "all_hand_object_contact": all_hand_contact.to(torch.float32).mean(),
            "all_hand_object_contact_force_n": all_hand_force.mean(),
            "excess_contact_force_penalty": contact_force_penalty.mean(),
            "finger_residual_abs_mean_rad": finger_residual.mean(),
            "finger_residual_abs_max_rad": finger_residual.amax(dim=-1).mean(),
            "reference_phase_fraction": (
                self.phase_buf.to(torch.float32).mean()
                / float(self._reference_length - 1)
            ),
        }
        if self._scissors_handle_gate_enabled:
            log["thumb_inside_scissors_top_hole"] = (
                self._last_thumb_in_scissors_hole.to(torch.float32).mean()
            )
            log["other_finger_inside_scissors_bottom_hole"] = (
                self._last_other_finger_in_scissors_hole.to(torch.float32).mean()
            )
        completed = self.reset_buf
        if completed.any():
            log["pose_tracking_return"] = self._pose_episode_return[completed].mean()
            log["completed_episode_steps"] = (
                self.episode_length_buf[completed].to(torch.float32).mean()
            )
        self.extras["log"] = log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # EgoEngine applies ctrl_ref[t] and evaluates against reference t+1.
        self.reference_time_buf += self._control_dt
        self.phase_buf = torch.clamp(
            torch.floor(self.reference_time_buf * self._reference_fps).to(torch.long),
            max=self._reference_length - 1,
        )
        self._compute_object_errors()
        # DirectRLEnv resets finished environments before returning from step().
        # Preserve the phase used for termination so external evaluation can
        # distinguish a true last-reference timeout from an early reset.
        self._last_evaluated_phase.copy_(self.phase_buf)
        invalid = (
            ~torch.isfinite(self._object_position_error)
            | ~torch.isfinite(self._object_rotation_error)
            | ~torch.isfinite(self._object_articulation_error)
        )
        articulation_lost = (
            self._object_articulation_error
            > self.cfg.object_failure_articulation
            if self.cfg.articulate_mode
            else torch.zeros_like(self._object_position_error, dtype=torch.bool)
        )
        tracking_failed = (
            (self._object_position_error > self.cfg.object_failure_distance)
            | (self._object_rotation_error > self.cfg.object_failure_orientation)
            | articulation_lost
        )
        # Playback may deliberately continue after a tracking failure so that
        # diagnostics and videos cover the complete reference.  That must not
        # turn a failed rollout into a successful one: termination policy and
        # task-success classification are separate decisions.
        object_lost = tracking_failed
        if self.cfg.disable_object_failure_termination:
            object_lost = torch.zeros_like(object_lost)
        end_of_reference = self.phase_buf >= self._reference_length - 1
        if self._capture_enabled:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._capture_hand_q[env_ids, self.phase_buf] = self.hand.data.joint_pos
            if self._bimanual_mode:
                self._capture_second_hand_q[env_ids, self.phase_buf] = (
                    self.second_hand.data.joint_pos
                )
            object_position = (
                self.object.data.root_pos_w - self.scene.env_origins
            )
            object_pose = torch.cat(
                (object_position, self.object.data.root_quat_w), dim=-1
            )
            self._capture_object_pose[env_ids, self.phase_buf] = object_pose
            if self.cfg.articulate_mode:
                self._capture_object_joint[env_ids, self.phase_buf] = (
                    self.object.data.joint_pos
                )
            self._capture_position_error[env_ids, self.phase_buf] = (
                self._object_position_error
            )
            self._capture_rotation_error[env_ids, self.phase_buf] = (
                self._object_rotation_error
            )
            self._capture_articulation_error[env_ids, self.phase_buf] = (
                self._object_articulation_error
            )
            trajectory_tracking_failed = (
                self._capture_position_error.amax(dim=1)
                > self.cfg.object_failure_distance
            ) | (
                self._capture_rotation_error.amax(dim=1)
                > self.cfg.object_failure_orientation
            )
            if self.cfg.articulate_mode:
                trajectory_tracking_failed |= (
                    self._capture_articulation_error.amax(dim=1)
                    > self.cfg.object_failure_articulation
                )
            successful = end_of_reference & ~invalid & ~trajectory_tracking_failed
            if successful.any():
                successful_ids = successful.nonzero(as_tuple=False).flatten()
                if os.environ.get("HAND_CAPTURE_BEST_SUCCESS", "0") == "1":
                    success_returns = (
                        self._capture_pose_reward[successful_ids].sum(dim=1)
                        + self._capture_contact_reward[successful_ids].sum(dim=1)
                    )
                    success_env_id = int(
                        successful_ids[torch.argmax(success_returns)].item()
                    )
                else:
                    success_env_id = int(successful_ids[0].item())
                self._save_success_trajectory(success_env_id)
            finished = invalid | object_lost | end_of_reference
            if finished.any():
                finished_ids = finished.nonzero(as_tuple=False).flatten()
                finished_phases = self.phase_buf[finished_ids]
                max_phase = finished_phases.max()
                phase_mask = finished_phases == max_phase
                phase_ids = finished_ids[phase_mask]
                phase = int(max_phase.item())
                frame_mask = (
                    torch.arange(self._reference_length, device=self.device)[None]
                    <= phase
                )
                candidate_returns = (
                    (self._capture_pose_reward[phase_ids] * frame_mask).sum(dim=1)
                    + (self._capture_contact_reward[phase_ids] * frame_mask).sum(dim=1)
                )
                candidate_offset = int(torch.argmax(candidate_returns).item())
                candidate_env_id = int(phase_ids[candidate_offset].item())
                # Save before DirectRLEnv resets the completed environment;
                # otherwise phase_buf and the captured episode are overwritten.
                self._save_best_trajectory(candidate_env_id)
        episode_limit = self.max_episode_length
        if (
            self.cfg.randomize_start_phase
            and self.cfg.random_start_episode_length > 0
        ):
            episode_limit = min(
                episode_limit, int(self.cfg.random_start_episode_length)
            )
        time_out = (self.episode_length_buf >= episode_limit) | end_of_reference
        return invalid | object_lost, time_out

    def _save_rollout_trajectory(
        self,
        output_path: Path,
        env_id: int,
        last_phase: int,
        *,
        success: bool,
    ) -> dict[str, object]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "hand_id": HAND_ID,
            "bimanual_mode": self._bimanual_mode,
            "second_hand_side": (
                SECOND_MANO_SIDE if self._bimanual_mode else None
            ),
            "status": "success" if success else "farthest",
            "success": success,
            "env_id": env_id,
            "final_phase": last_phase,
            "reference_last_phase": self._reference_length - 1,
            "position_error_m": float(self._object_position_error[env_id].item()),
            "rotation_error_rad": float(self._object_rotation_error[env_id].item()),
            "articulation_error_rad": float(
                self._object_articulation_error[env_id].item()
            ),
            "pose_tracking_return": float(
                self._capture_pose_reward[env_id, : last_phase + 1].sum().item()
            ),
            "contact_return": float(
                self._capture_contact_reward[env_id, : last_phase + 1].sum().item()
            ),
            "mean_position_error_m": float(
                self._capture_position_error[env_id, : last_phase + 1].mean().item()
            ),
            "max_position_error_m": float(
                self._capture_position_error[env_id, : last_phase + 1].amax().item()
            ),
            "mean_rotation_error_rad": float(
                self._capture_rotation_error[env_id, : last_phase + 1].mean().item()
            ),
            "max_rotation_error_rad": float(
                self._capture_rotation_error[env_id, : last_phase + 1].amax().item()
            ),
            "mean_articulation_error_rad": float(
                self._capture_articulation_error[
                    env_id, : last_phase + 1
                ].mean().item()
            ),
            "max_articulation_error_rad": float(
                self._capture_articulation_error[
                    env_id, : last_phase + 1
                ].amax().item()
            ),
            "joint_names": list(self.hand.joint_names),
            "action_joint_names": list(self._all_action_joint_names),
            "primary_action_joint_names": list(self._action_joint_names),
            "second_action_joint_names": list(
                self._second_action_joint_names
            ),
            "residual_root_position_scale": self.cfg.residual_root_position_scale,
            "residual_root_rotation_scale": self.cfg.residual_root_rotation_scale,
            "residual_finger_scale": self.cfg.residual_finger_scale,
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        payload = {
            "hand_q": self._capture_hand_q[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "object_pose_wxyz": self._capture_object_pose[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "actions": self._capture_actions[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "joint_targets": self._capture_joint_targets[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "pose_reward": self._capture_pose_reward[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "contact_reward": self._capture_contact_reward[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "object_position_error_m": self._capture_position_error[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "object_rotation_error_rad": self._capture_rotation_error[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "object_articulation_error_rad": self._capture_articulation_error[
                env_id, : last_phase + 1
            ].detach().cpu().numpy(),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        if self._bimanual_mode:
            payload["second_hand_q"] = self._capture_second_hand_q[
                env_id, : last_phase + 1
            ].detach().cpu().numpy()
            payload["second_joint_targets"] = (
                self._capture_second_joint_targets[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy()
            )
            payload["second_joint_names"] = np.asarray(
                self.second_hand.joint_names
            )
        if self.cfg.articulate_mode:
            payload["object_joint_position_rad"] = self._capture_object_joint[
                env_id, : last_phase + 1
            ].detach().cpu().numpy()
            payload["object_joint_names"] = np.asarray(self.object.joint_names)
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary_path.replace(output_path)
        return metadata

    def _save_best_trajectory(self, env_id: int) -> None:
        if self._best_rollout_path is None:
            return
        last_phase = int(self.phase_buf[env_id].item())
        candidate_return = float(
            self._capture_pose_reward[env_id, : last_phase + 1].sum().item()
            + self._capture_contact_reward[env_id, : last_phase + 1].sum().item()
        )
        if last_phase < self._best_rollout_phase or (
            last_phase == self._best_rollout_phase
            and candidate_return <= self._best_rollout_return
        ):
            return
        trajectory_tracking_failed = (
            self._capture_position_error[env_id, : last_phase + 1].amax()
            > self.cfg.object_failure_distance
        ) | (
            self._capture_rotation_error[env_id, : last_phase + 1].amax()
            > self.cfg.object_failure_orientation
        )
        if self.cfg.articulate_mode:
            trajectory_tracking_failed |= (
                self._capture_articulation_error[
                    env_id, : last_phase + 1
                ].amax()
                > self.cfg.object_failure_articulation
            )
        success = bool(
            last_phase >= self._reference_length - 1
            and torch.isfinite(self._object_position_error[env_id]).item()
            and torch.isfinite(self._object_rotation_error[env_id]).item()
            and torch.isfinite(self._object_articulation_error[env_id]).item()
            and not trajectory_tracking_failed.item()
        )
        metadata = self._save_rollout_trajectory(
            self._best_rollout_path,
            env_id,
            last_phase,
            success=success,
        )
        self._best_rollout_phase = last_phase
        self._best_rollout_return = candidate_return
        print(
            "HAND_BEST_ROLLOUT_CAPTURED "
            f"hand_id={HAND_ID} path={self._best_rollout_path} "
            f"env_id={env_id} phase={last_phase} "
            f"total_reward={candidate_return:.9f} "
            f"position_error_m={metadata['position_error_m']:.9f} "
            f"rotation_error_rad={metadata['rotation_error_rad']:.9f}",
            flush=True,
        )

    def _save_success_trajectory(self, env_id: int) -> None:
        if self._success_capture_path is None:
            return
        last_phase = self._reference_length - 1
        metadata = self._save_rollout_trajectory(
            self._success_capture_path,
            env_id,
            last_phase,
            success=True,
        )
        print(
            "HAND_SUCCESS_TRAJECTORY_CAPTURED "
            f"hand_id={HAND_ID} path={self._success_capture_path} "
            f"env_id={env_id} phase={last_phase} "
            f"position_error_m={metadata['position_error_m']:.9f} "
            f"rotation_error_rad={metadata['rotation_error_rad']:.9f}",
            flush=True,
        )
        raise SystemExit(0)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_tensor = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )

        if self.cfg.randomize_start_phase:
            # Tracking references can be longer than a single useful learning
            # segment. Sampling across the full reference gives PPO local
            # supervision at every manipulation phase; the end-of-reference
            # timeout naturally shortens late-starting episodes.
            min_start = int(self.cfg.random_start_phase_min)
            max_start = int(self.cfg.random_start_phase_max)
            if max_start < 0:
                max_start = self._reference_length - 2
            max_start = min(max_start, self._reference_length - 2)
            if min_start < 0 or min_start > max_start:
                raise ValueError(
                    "random start phase bounds must satisfy "
                    f"0 <= min <= max <= {self._reference_length - 2}, got "
                    f"min={min_start}, max={max_start}"
                )
            self.phase_buf[env_ids] = torch.randint(
                low=min_start,
                high=max_start + 1,
                size=(len(env_ids),),
                device=self.device,
            )
            random_fraction = float(self.cfg.randomize_start_phase_fraction)
            if not 0.0 <= random_fraction <= 1.0:
                raise ValueError("randomize_start_phase_fraction must be in [0, 1]")
            if random_fraction < 1.0:
                keep_random = torch.rand(len(env_ids), device=self.device) < random_fraction
                self.phase_buf[env_ids_tensor[~keep_random]] = 0
        else:
            self.phase_buf[env_ids] = 0
        self.reference_time_buf[env_ids] = (
            self.phase_buf[env_ids].to(torch.float32) / self._reference_fps
        )

        hand_root_state = self.hand.data.default_root_state[env_ids].clone()
        hand_root_state[:, :3] += self.scene.env_origins[env_ids]
        hand_root_state[:, 7:] = 0.0
        hand_initialization = (
            self.initialization_hand_q
            if self.cfg.randomize_start_phase
            and self.initialization_hand_q is not None
            else self.reference_hand_q
        )
        joint_pos = self._reference_at(
            hand_initialization,
            self.phase_buf[env_ids_tensor],
            env_ids_tensor,
        )
        joint_vel = (
            self._reference_velocity_at(
                hand_initialization,
                self.phase_buf[env_ids_tensor],
                env_ids_tensor,
            )
            if self.cfg.randomize_start_phase
            else torch.zeros_like(joint_pos)
        )
        self.hand.write_root_pose_to_sim(hand_root_state[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(hand_root_state[:, 7:], env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        ctrl = self._reference_at(
            self.reference_hand_ctrl,
            self.phase_buf[env_ids_tensor],
            env_ids_tensor,
        )
        self.hand.set_joint_position_target(ctrl, env_ids=env_ids)
        self.joint_targets[env_ids] = ctrl

        if self._bimanual_mode:
            second_root_state = self.second_hand.data.default_root_state[
                env_ids
            ].clone()
            second_root_state[:, :3] += self.scene.env_origins[env_ids]
            second_root_state[:, 7:] = 0.0
            second_joint_pos = self._reference_at(
                self.second_reference_hand_q,
                self.phase_buf[env_ids_tensor],
                env_ids_tensor,
            )
            second_joint_vel = (
                self._reference_velocity_at(
                    self.second_reference_hand_q,
                    self.phase_buf[env_ids_tensor],
                    env_ids_tensor,
                )
                if self.cfg.randomize_start_phase
                else torch.zeros_like(second_joint_pos)
            )
            self.second_hand.write_root_pose_to_sim(
                second_root_state[:, :7], env_ids
            )
            self.second_hand.write_root_velocity_to_sim(
                second_root_state[:, 7:], env_ids
            )
            self.second_hand.write_joint_state_to_sim(
                second_joint_pos,
                second_joint_vel,
                env_ids=env_ids,
            )
            second_ctrl = self._reference_at(
                self.second_reference_hand_ctrl,
                self.phase_buf[env_ids_tensor],
                env_ids_tensor,
            )
            self.second_hand.set_joint_position_target(
                second_ctrl, env_ids=env_ids
            )
            self.second_joint_targets[env_ids] = second_ctrl

        object_pose = self._reference_at(
            self.reference_object_pose,
            self.phase_buf[env_ids_tensor],
            env_ids_tensor,
        ).clone()
        object_pose[:, :3] += self.scene.env_origins[env_ids]
        object_velocity = (
            self._reference_root_velocity_at(
                self.reference_object_pose,
                self.phase_buf[env_ids_tensor],
                env_ids_tensor,
            )
            if self.cfg.randomize_start_phase
            else torch.zeros((len(env_ids), 6), device=self.device)
        )
        self.object.write_root_pose_to_sim(object_pose, env_ids)
        self.object.write_root_velocity_to_sim(object_velocity, env_ids)
        if self.cfg.articulate_mode:
            object_joint_initialization = (
                self.initialization_object_joint
                if self.cfg.randomize_start_phase
                and self.initialization_object_joint is not None
                else self.reference_object_joint
            )
            object_joint_pos = self._reference_at(
                object_joint_initialization,
                self.phase_buf[env_ids_tensor],
                env_ids_tensor,
            )
            object_joint_vel = (
                self._reference_velocity_at(
                    object_joint_initialization,
                    self.phase_buf[env_ids_tensor],
                    env_ids_tensor,
                )
                if self.cfg.randomize_start_phase
                else torch.zeros_like(object_joint_pos)
            )
            self.object.write_joint_state_to_sim(
                object_joint_pos,
                object_joint_vel,
                env_ids=env_ids,
            )
        self.actions[env_ids] = 0.0
        self._pose_episode_return[env_ids] = 0.0
        if self._capture_enabled:
            self._capture_hand_q[env_ids] = 0.0
            if self._bimanual_mode:
                self._capture_second_hand_q[env_ids] = 0.0
                self._capture_second_joint_targets[env_ids] = 0.0
            self._capture_object_pose[env_ids] = 0.0
            if self.cfg.articulate_mode:
                self._capture_object_joint[env_ids] = 0.0
            self._capture_actions[env_ids] = 0.0
            self._capture_joint_targets[env_ids] = 0.0
            self._capture_pose_reward[env_ids] = 0.0
            self._capture_contact_reward[env_ids] = 0.0
            self._capture_position_error[env_ids] = 0.0
            self._capture_rotation_error[env_ids] = 0.0
            self._capture_articulation_error[env_ids] = 0.0
            self._capture_hand_q[env_ids, 0] = joint_pos
            initial_object_pose = self._reference_at(
                self.reference_object_pose,
                torch.zeros(len(env_ids_tensor), device=self.device, dtype=torch.long),
                env_ids_tensor,
            )
            self._capture_object_pose[env_ids, 0] = initial_object_pose
            if self.cfg.articulate_mode:
                self._capture_object_joint[env_ids, 0] = self._reference_at(
                    self.reference_object_joint,
                    torch.zeros(
                        len(env_ids_tensor), device=self.device, dtype=torch.long
                    ),
                    env_ids_tensor,
                )
            self._capture_actions[env_ids, 0] = 0.0
            self._capture_joint_targets[env_ids, 0] = ctrl
            if self._bimanual_mode:
                self._capture_second_hand_q[env_ids, 0] = second_joint_pos
                self._capture_second_joint_targets[
                    env_ids, 0
                ] = second_ctrl
            self._capture_pose_reward[env_ids, 0] = 0.0
            self._capture_contact_reward[env_ids, 0] = 0.0
