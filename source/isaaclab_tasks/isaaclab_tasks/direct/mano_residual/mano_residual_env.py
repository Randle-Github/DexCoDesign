# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL over the reviewed HO-Cap/MANO reference trajectory.

The command is exactly ``q_target = q_reference + scale * residual``. The
training reward combines object-pose tracking with EgoEngine-MPC's binary
pinch-contact term.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude


REPO_ROOT = Path(__file__).resolve().parents[5]
ASSET_ROOT = REPO_ROOT / "artifacts" / "isaaclab_mano_residual" / "assets"
REFERENCE_PATH = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "subject_7"
    / "20231022_192832"
    / "isaaclab_reference.npz"
)

THUMB_CONTACT_LINK_NAMES = ("left_thumb3",)
OTHER_FINGER_CONTACT_LINK_NAMES = (
    "left_index3",
    "left_middle3",
    "left_ring3",
    "left_pinky3",
)


@configclass
class ManoResidualEnvCfg(DirectRLEnvCfg):
    decimation = 4
    # EgoEngine trains on randomly sampled 64-control-step windows and only
    # uses a full sequence for evaluation.
    episode_length_s = 64.0 / 30.0
    # The PPO action is the normalized residual itself. Keep its declared
    # bounds identical to the residual executed by _pre_physics_step so the
    # policy likelihood is evaluated on the action that reaches the robot.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)
    observation_space = 63
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        log_dir=str(
            REPO_ROOT
            / "artifacts"
            / "isaaclab_mano_residual"
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
            usd_path=str(ASSET_ROOT / "mano_left.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
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
                joint_names_expr=["left_pos_.*", "left_rot_.*"],
                stiffness=1000.0,
                damping=63.2455532,
                effort_limit_sim=1000.0,
                velocity_limit_sim=20.0,
                armature=1.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["left_j_.*"],
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

    # EgoEngine action scale = component_noise_scale * first_ctrl_noise_scale.
    residual_root_position_scale = 0.08 * 0.5
    residual_root_rotation_scale = 0.05 * 0.5
    residual_finger_scale = 0.30 * 0.5
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
    randomize_start_phase = True
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
        self._reference_hand_q_cpu = torch.from_numpy(reference["hand_q"])
        if "hand_ctrl" not in reference:
            raise RuntimeError(
                f"{REFERENCE_PATH} has no hand_ctrl; regenerate the "
                "EgoEngine-style reference before training"
            )
        self._reference_hand_ctrl_cpu = torch.from_numpy(reference["hand_ctrl"])
        self._reference_object_pose_cpu = torch.from_numpy(reference["object_pose_wxyz"])
        self._reference_length = len(self._reference_hand_q_cpu)

        super().__init__(cfg, render_mode, **kwargs)

        if self.hand.num_joints != self.cfg.action_space:
            raise RuntimeError(
                f"Expected {self.cfg.action_space} MANO joints, found {self.hand.num_joints}: "
                f"{self.hand.joint_names}"
            )
        missing = sorted(set(self._reference_joint_names) - set(self.hand.joint_names))
        if missing:
            raise RuntimeError(f"Reference joints missing from imported MANO articulation: {missing}")

        reference_order = [self._reference_joint_names.index(name) for name in self.hand.joint_names]
        self.reference_hand_q = self._reference_hand_q_cpu[:, reference_order].to(self.device)
        self.reference_hand_ctrl = self._reference_hand_ctrl_cpu[:, reference_order].to(
            self.device
        )
        self.reference_object_pose = self._reference_object_pose_cpu.to(self.device)

        limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.joint_lower_limits = limits[..., 0]
        self.joint_upper_limits = limits[..., 1]

        self.residual_scale = torch.full(
            (self.hand.num_joints,), self.cfg.residual_finger_scale, device=self.device
        )
        self.residual_scale[:3] = self.cfg.residual_root_position_scale
        self.residual_scale[3:6] = self.cfg.residual_root_rotation_scale

        self.phase_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device
        )
        self.joint_targets = torch.zeros_like(self.actions)
        self._object_position_error = torch.zeros(self.num_envs, device=self.device)
        self._object_rotation_error = torch.zeros(self.num_envs, device=self.device)
        self._last_evaluated_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._last_diagnostic_phase = -1
        self._palm_body_index = self.hand.body_names.index("left_palm")
        self._middle_tip_body_index = self.hand.body_names.index("left_middle3")

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.hand_cfg)
        visual_manifest_path = ASSET_ROOT / "mano_visuals.json"
        if visual_manifest_path.is_file():
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
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["object_thumb_contact"] = self._thumb_contact_sensor
        self.scene.sensors["object_other_finger_contact"] = self._other_finger_contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        base_targets = self.reference_hand_ctrl[self.phase_buf]
        targets = base_targets + self.residual_scale * self.actions
        self.joint_targets = torch.clamp(
            targets,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self.joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        object_pose = torch.cat((object_pos, self.object.data.root_quat_w), dim=-1)
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
                    "[MANO_ROLLOUT] "
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
                object_pose,
                self.reference_hand_ctrl[self.phase_buf],
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
        finger_residual = (
            self.actions[:, 6:] * self.residual_scale[6:]
        ).abs()
        self.extras["log"] = {
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
        time_out = (self.episode_length_buf >= self.max_episode_length) | end_of_reference
        return invalid | object_lost, time_out

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
