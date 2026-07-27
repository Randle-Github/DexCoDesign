# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL over a reviewed HO-Cap hand reference trajectory.

The command is exactly ``q_target = q_reference + scale * residual``. The
training reward combines object-pose tracking with EgoEngine-MPC's binary
pinch-contact term.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_error_magnitude


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
ALL_HAND_ROOT = REPO_ROOT / "artifacts" / "isaaclab_all_hands_residual"
HAND_ID = os.environ.get("DEXCODESIGN_HAND_ID", "mano")
if HAND_ID == "mano":
    REFERENCE_PATH = MANO_REFERENCE_PATH
    HAND_USD_PATH = ASSET_ROOT / "mano_left.usd"
    ROOT_POSITION_JOINT_NAMES = ("left_pos_x", "left_pos_y", "left_pos_z")
    ROOT_ROTATION_JOINT_NAMES = ("left_rot_x", "left_rot_y", "left_rot_z")
    THUMB_CONTACT_LINK_NAMES = ("left_thumb3",)
    OTHER_FINGER_CONTACT_LINK_NAMES = (
        "left_index3",
        "left_middle3",
        "left_ring3",
        "left_pinky3",
    )
    PALM_BODY_NAME = "left_palm"
    MIDDLE_TIP_BODY_NAME = "left_middle3"
else:
    REFERENCE_PATH = ALL_HAND_ROOT / "prepared" / HAND_ID / "reference.npz"
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
OBSERVATION_DIM = 2 * CONTROL_DIM + 34
ROOT_POSITION_EXPR = "left_pos_.*" if HAND_ID == "mano" else "root_pos_.*"
ROOT_ROTATION_EXPR = "left_rot_.*" if HAND_ID == "mano" else "root_rot_.*"
FINGER_JOINT_EXPR = "left_j_.*" if HAND_ID == "mano" else "finger__.*"


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
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        log_dir=str(
            REPO_ROOT
            / "artifacts"
            / "isaaclab_all_hands_residual"
            / "until_success"
            / "isaaclab_logs"
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
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
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(HAND_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=HAND_ID == "mano",
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
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

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_ROOT / "g04_1.usd"),
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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=0.65,
        replicate_physics=True,
        # ContactSensor discovers one USD reporting prim per environment.
        # Fabric-only clones are absent from that discovery stage.
        clone_in_fabric=False,
    )

    # Residual bounds are applied directly to the reference controller target.
    # The reference is already close, so root translation and finger residuals
    # retain the validated conservative ranges. Wrist rotation alone receives
    # the wider range needed to clear the object during release and retreat.
    residual_root_position_scale = 0.04
    residual_root_rotation_scale = 0.30
    residual_finger_scale = 0.15
    object_position_sigma = 0.04
    object_rotation_sigma = 0.50
    # Match EgoEngine-MPC's Aria residual-RL reward geometry:
    # reward = C - ||[w_pos * position_error, w_rot * rotation_error]||_2
    #          + contact_scale * pinch_contact.
    object_position_reward_weight = 1.0
    object_rotation_reward_weight = 0.3
    contact_reward_weight = 2.0
    # EgoEngine treats any penetrating thumb/object and other-finger/object
    # contact pair as active.  A positive Isaac contact force is its analogue.
    contact_force_threshold = 0.0
    object_failure_distance = 0.05
    object_failure_orientation = 1.50
    randomize_start_phase = False
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
    object_failure_distance = float("inf")
    object_failure_orientation = float("inf")
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

    def __init__(self, cfg: ManoResidualEnvCfg, render_mode: str | None = None, **kwargs):
        reference = np.load(REFERENCE_PATH)
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
        self._reference_hand_q_cpu = torch.from_numpy(reference["hand_q"])
        if "hand_ctrl" not in reference:
            raise RuntimeError(
                f"{REFERENCE_PATH} has no hand_ctrl; regenerate the "
                "EgoEngine-style reference before training"
            )
        self._reference_hand_ctrl_cpu = torch.from_numpy(reference["hand_ctrl"])
        self._reference_object_pose_cpu = torch.from_numpy(reference["object_pose_wxyz"])
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
            reference["fingertip_pose_wxyz"]
        )
        self._reference_fingertip_link_names = reference[
            "fingertip_link_names"
        ].tolist()
        self._fingertip_offsets_cpu = torch.from_numpy(reference["fingertip_offsets"])
        self._reference_length = len(self._reference_hand_q_cpu)

        super().__init__(cfg, render_mode, **kwargs)

        self.action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.action_dim != len(self._action_joint_names):
            raise RuntimeError(
                f"Action space has {self.action_dim} dimensions but {HAND_ID} "
                f"defines {len(self._action_joint_names)} active joints"
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
        self.reference_hand_q = self._reference_hand_q_cpu[:, reference_order].to(self.device)
        self.reference_hand_ctrl = self._reference_hand_ctrl_cpu[:, reference_order].to(
            self.device
        )
        self.action_to_control = self._action_to_control_cpu[reference_order].to(
            self.device
        )
        if self.action_to_control.shape != (
            self.hand.num_joints,
            self.action_dim,
        ):
            raise RuntimeError(
                f"{HAND_ID} action-to-control map has shape "
                f"{tuple(self.action_to_control.shape)}, expected "
                f"({self.hand.num_joints}, {self.action_dim})"
            )
        self.reference_object_pose = self._reference_object_pose_cpu.to(self.device)
        self.reference_fingertip_pose = self._reference_fingertip_pose_cpu.to(
            self.device
        )
        self.fingertip_offsets = self._fingertip_offsets_cpu.to(self.device)

        limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.joint_lower_limits = limits[..., 0]
        self.joint_upper_limits = limits[..., 1]

        self.residual_scale = torch.full(
            (self.action_dim,), self.cfg.residual_finger_scale, device=self.device
        )
        self._root_position_action_indices = torch.tensor(
            [
                self._action_joint_names.index(name)
                for name in ROOT_POSITION_JOINT_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._root_rotation_action_indices = torch.tensor(
            [
                self._action_joint_names.index(name)
                for name in ROOT_ROTATION_JOINT_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._finger_action_indices = torch.tensor(
            [
                index
                for index, name in enumerate(self._action_joint_names)
                if name
                not in set(ROOT_POSITION_JOINT_NAMES + ROOT_ROTATION_JOINT_NAMES)
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
        self.actions = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float, device=self.device
        )
        self.joint_targets = torch.zeros(
            (self.num_envs, self.hand.num_joints),
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
        if self._best_rollout_path is not None and self._best_rollout_path.is_file():
            try:
                with np.load(self._best_rollout_path) as previous_rollout:
                    previous_metadata = json.loads(
                        str(previous_rollout["metadata_json"])
                    )
                self._best_rollout_phase = int(
                    previous_metadata.get("final_phase", -1)
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
            self._capture_object_pose = torch.zeros(
                (*capture_shape, 7),
                dtype=torch.float32,
                device=self.device,
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

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.hand_cfg)
        visual_manifest_path = ASSET_ROOT / "mano_visuals.json"
        if HAND_ID == "mano" and visual_manifest_path.is_file():
            visual_manifest = json.loads(visual_manifest_path.read_text(encoding="utf-8"))
            for link_name, relative_usd_path in visual_manifest.items():
                visual_cfg = sim_utils.UsdFileCfg(
                    usd_path=str(ASSET_ROOT / relative_usd_path),
                )
                visual_cfg.func(
                    f"/World/envs/env_0/Hand/{link_name}/visual_overlay",
                    visual_cfg,
                )
        self.object = RigidObject(self.cfg.object_cfg)
        # ContactSensor filtering is one-to-many: its prim_path must resolve to
        # one reporting rigid body per environment.  Use the object as that
        # body and filter its contacts against the two hand-link groups.
        self._thumb_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Object",
                update_period=0.0,
                history_length=0,
                filter_prim_paths_expr=[
                    f"/World/envs/env_.*/Hand/{link_name}"
                    for link_name in THUMB_CONTACT_LINK_NAMES
                ],
            )
        )
        self._other_finger_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Object",
                update_period=0.0,
                history_length=0,
                filter_prim_paths_expr=[
                    f"/World/envs/env_.*/Hand/{link_name}"
                    for link_name in OTHER_FINGER_CONTACT_LINK_NAMES
                ],
            )
        )
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
        self._filter_hand_support_collisions(
            hand_root_path="/World/envs/env_0/Hand",
            support_root_path="/World/ground",
        )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["object_thumb_contact"] = self._thumb_contact_sensor
        self.scene.sensors["object_other_finger_contact"] = self._other_finger_contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

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
        print(
            f"[HAND_COLLISION_FILTER:{HAND_ID}] "
            f"disabled hand-support pairs for {filtered_prims} hand prims; "
            "object-hand and object-support pairs remain enabled"
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        base_targets = self.reference_hand_ctrl[self.phase_buf]
        active_residual = self.residual_scale * self.actions
        control_residual = active_residual @ self.action_to_control.T
        targets = base_targets + control_residual
        self.joint_targets = torch.clamp(
            targets,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )
        if self._capture_enabled:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._capture_actions[env_ids, self.phase_buf] = self.actions
            self._capture_joint_targets[env_ids, self.phase_buf] = self.joint_targets

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self.joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        object_pose = torch.cat((object_pos, self.object.data.root_quat_w), dim=-1)
        fingertip_body_quat = self.hand.data.body_quat_w[
            :, self._fingertip_body_indices
        ]
        fingertip_body_pos = (
            self.hand.data.body_pos_w[:, self._fingertip_body_indices]
            - self.scene.env_origins[:, None, :]
        )
        fingertip_offset_w = quat_apply(
            fingertip_body_quat.reshape(-1, 4),
            self.fingertip_offsets[None, :, :]
            .expand(self.num_envs, -1, -1)
            .reshape(-1, 3),
        ).reshape(self.num_envs, len(self._fingertip_body_indices), 3)
        fingertip_pos = fingertip_body_pos + fingertip_offset_w
        goal_fingertip_pose = self.reference_fingertip_pose[self.phase_buf]
        goal_object_pose = self.reference_object_pose[self.phase_buf]
        if self.cfg.log_rollout_diagnostics and self.num_envs == 1:
            phase_index = int(self.phase_buf[0].item())
            if phase_index != self._last_diagnostic_phase and (
                phase_index < 3 or phase_index % 110 == 0
            ):
                actual_q = self.hand.data.joint_pos[0]
                target_q = self.joint_targets[0]
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
                self.reference_hand_q[self.phase_buf],
                goal_object_pose,
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _compute_object_errors(self) -> None:
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        reference_pose = self.reference_object_pose[self.phase_buf]
        self._object_position_error = torch.linalg.vector_norm(
            object_pos - reference_pose[:, :3], dim=-1
        )
        self._object_rotation_error = quat_error_magnitude(
            self.object.data.root_quat_w,
            reference_pose[:, 3:7],
        )

    def _contact_sensor_force(self, sensor: ContactSensor) -> torch.Tensor:
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        return torch.linalg.vector_norm(
            force_matrix.reshape(self.num_envs, -1, 3),
            dim=-1,
        ).amax(dim=-1)

    def _compute_pinch_contact(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        thumb_force = self._contact_sensor_force(self._thumb_contact_sensor)
        other_finger_force = self._contact_sensor_force(self._other_finger_contact_sensor)
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
        weighted_position_error = (
            self.cfg.object_position_reward_weight
            * self._object_position_error
        )
        weighted_rotation_error = (
            self.cfg.object_rotation_reward_weight
            * self._object_rotation_error
        )
        pose_tracking_error = torch.sqrt(
            weighted_position_error.square()
            + weighted_rotation_error.square()
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
        )
        pose_tracking_reward = reward_offset_c - pose_tracking_error
        (
            thumb_contact,
            other_finger_contact,
            pinch_contact,
            thumb_force,
            other_finger_force,
        ) = self._compute_pinch_contact()
        contact_reward = pinch_contact.to(torch.float32) * self.cfg.contact_reward_weight
        total_reward = pose_tracking_reward + contact_reward
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
            "weighted_object_position_error": weighted_position_error.mean(),
            "weighted_object_rotation_error": weighted_rotation_error.mean(),
            "pose_tracking_error": pose_tracking_error.mean(),
            "reward_offset_c": reward_offset_c,
            "pose_tracking_reward": pose_tracking_reward.mean(),
            "thumb_object_contact": thumb_contact.to(torch.float32).mean(),
            "other_finger_object_contact": other_finger_contact.to(torch.float32).mean(),
            "pinch_contact_reward": contact_reward.mean(),
            "thumb_object_contact_force_n": thumb_force.mean(),
            "other_finger_object_contact_force_n": other_finger_force.mean(),
            "finger_residual_abs_mean_rad": finger_residual.mean(),
            "finger_residual_abs_max_rad": finger_residual.amax(dim=-1).mean(),
            "reference_phase_fraction": (
                self.phase_buf.to(torch.float32).mean()
                / float(self._reference_length - 1)
            ),
        }
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
        self.phase_buf = torch.clamp(
            self.phase_buf + 1,
            max=self._reference_length - 1,
        )
        self._compute_object_errors()
        # DirectRLEnv resets finished environments before returning from step().
        # Preserve the phase used for termination so external evaluation can
        # distinguish a true last-reference timeout from an early reset.
        self._last_evaluated_phase.copy_(self.phase_buf)
        invalid = ~torch.isfinite(self._object_position_error) | ~torch.isfinite(
            self._object_rotation_error
        )
        object_lost = (
            (self._object_position_error > self.cfg.object_failure_distance)
            | (self._object_rotation_error > self.cfg.object_failure_orientation)
        )
        end_of_reference = self.phase_buf >= self._reference_length - 1
        if self._capture_enabled:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._capture_hand_q[env_ids, self.phase_buf] = self.hand.data.joint_pos
            object_position = (
                self.object.data.root_pos_w - self.scene.env_origins
            )
            object_pose = torch.cat(
                (object_position, self.object.data.root_quat_w), dim=-1
            )
            self._capture_object_pose[env_ids, self.phase_buf] = object_pose
            successful = end_of_reference & ~invalid & ~object_lost
            if successful.any():
                success_env_id = int(successful.nonzero(as_tuple=False)[0, 0].item())
                self._save_success_trajectory(success_env_id)
            finished = invalid | object_lost | end_of_reference
            if finished.any():
                finished_ids = finished.nonzero(as_tuple=False).flatten()
                finished_phases = self.phase_buf[finished_ids]
                candidate_offset = int(torch.argmax(finished_phases).item())
                candidate_env_id = int(finished_ids[candidate_offset].item())
                candidate_phase = int(self.phase_buf[candidate_env_id].item())
                if candidate_phase > self._best_rollout_phase:
                    # Save before DirectRLEnv resets the completed environment;
                    # otherwise phase_buf and the captured episode are overwritten.
                    self._save_best_trajectory(candidate_env_id)
        time_out = (self.episode_length_buf >= self.max_episode_length) | end_of_reference
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
            "status": "success" if success else "farthest",
            "success": success,
            "env_id": env_id,
            "final_phase": last_phase,
            "reference_last_phase": self._reference_length - 1,
            "position_error_m": float(self._object_position_error[env_id].item()),
            "rotation_error_rad": float(self._object_rotation_error[env_id].item()),
            "pose_tracking_return": float(
                self._capture_pose_reward[env_id, : last_phase + 1].sum().item()
            ),
            "contact_return": float(
                self._capture_contact_reward[env_id, : last_phase + 1].sum().item()
            ),
            "joint_names": list(self.hand.joint_names),
            "action_joint_names": list(self._action_joint_names),
            "residual_root_position_scale": self.cfg.residual_root_position_scale,
            "residual_root_rotation_scale": self.cfg.residual_root_rotation_scale,
            "residual_finger_scale": self.cfg.residual_finger_scale,
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                hand_q=self._capture_hand_q[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy(),
                object_pose_wxyz=self._capture_object_pose[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy(),
                actions=self._capture_actions[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy(),
                joint_targets=self._capture_joint_targets[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy(),
                pose_reward=self._capture_pose_reward[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy(),
                contact_reward=self._capture_contact_reward[
                    env_id, : last_phase + 1
                ].detach().cpu().numpy(),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
        temporary_path.replace(output_path)
        return metadata

    def _save_best_trajectory(self, env_id: int) -> None:
        if self._best_rollout_path is None:
            return
        last_phase = int(self.phase_buf[env_id].item())
        if last_phase <= self._best_rollout_phase:
            return
        metadata = self._save_rollout_trajectory(
            self._best_rollout_path,
            env_id,
            last_phase,
            success=last_phase >= self._reference_length - 1,
        )
        self._best_rollout_phase = last_phase
        print(
            "HAND_BEST_ROLLOUT_CAPTURED "
            f"hand_id={HAND_ID} path={self._best_rollout_path} "
            f"env_id={env_id} phase={last_phase} "
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

        if self.cfg.randomize_start_phase:
            max_start = self._reference_length - self.max_episode_length - 2
            self.phase_buf[env_ids] = torch.randint(
                low=0,
                high=max(max_start + 1, 1),
                size=(len(env_ids),),
                device=self.device,
            )
        else:
            self.phase_buf[env_ids] = 0

        hand_root_state = self.hand.data.default_root_state[env_ids].clone()
        hand_root_state[:, :3] += self.scene.env_origins[env_ids]
        hand_root_state[:, 7:] = 0.0
        joint_pos = self.reference_hand_q[self.phase_buf[env_ids]]
        joint_vel = torch.zeros_like(joint_pos)
        self.hand.write_root_pose_to_sim(hand_root_state[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(hand_root_state[:, 7:], env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        ctrl = self.reference_hand_ctrl[self.phase_buf[env_ids]]
        self.hand.set_joint_position_target(ctrl, env_ids=env_ids)
        self.joint_targets[env_ids] = ctrl

        object_pose = self.reference_object_pose[self.phase_buf[env_ids]].clone()
        object_pose[:, :3] += self.scene.env_origins[env_ids]
        object_velocity = torch.zeros((len(env_ids), 6), device=self.device)
        self.object.write_root_pose_to_sim(object_pose, env_ids)
        self.object.write_root_velocity_to_sim(object_velocity, env_ids)
        self.actions[env_ids] = 0.0
        self._pose_episode_return[env_ids] = 0.0
        if self._capture_enabled:
            self._capture_hand_q[env_ids, 0] = joint_pos
            self._capture_object_pose[env_ids, 0] = self.reference_object_pose[0]
            self._capture_actions[env_ids, 0] = 0.0
            self._capture_joint_targets[env_ids, 0] = ctrl
            self._capture_pose_reward[env_ids, 0] = 0.0
            self._capture_contact_reward[env_ids, 0] = 0.0
