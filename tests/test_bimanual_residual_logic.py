"""CPU checks for the rigid two-hand observation and reward wiring."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/mano_residual_env.py"
)


def _method(name: str):
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    env_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManoResidualEnv"
    )
    method = next(
        node for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENV_PATH), "exec"), scope)
    return scope[name]


class BimanualResidualLogicTest(unittest.TestCase):
    def test_second_pinch_uses_only_second_hand_contacts(self):
        env = SimpleNamespace(
            cfg=SimpleNamespace(
                contact_force_threshold=0., articulate_mode=False,
                require_cross_link_pinch=False,
            ),
            _second_thumb_contact_sensor=torch.tensor([1., 0.]),
            _second_other_finger_contact_sensor=torch.tensor([1., 1.]),
            _contact_sensor_force=lambda sensor: sensor,
        )
        thumb, other, pinch, _, _ = _method("_compute_second_pinch_contact")(env)
        self.assertEqual(thumb.tolist(), [True, False])
        self.assertEqual(other.tolist(), [True, True])
        self.assertEqual(pinch.tolist(), [True, False])

    def test_second_articulated_pinch_uses_its_own_object_link_contacts(self):
        thumb = torch.tensor([[1., 0.], [1., 0.]])
        other = torch.tensor([[0., 1.], [1., 0.]])
        env = SimpleNamespace(
            cfg=SimpleNamespace(
                contact_force_threshold=0., articulate_mode=True,
                require_cross_link_pinch=True,
            ),
            device="cpu",
            _second_thumb_contact_sensors=thumb,
            _second_other_finger_contact_sensors=other,
            _contact_sensor_forces_by_filter=lambda sensors: sensors,
        )
        _, _, pinch, _, _ = _method("_compute_second_pinch_contact")(env)
        self.assertEqual(pinch.tolist(), [True, False])

    def test_observation_contains_two_hands_and_one_shared_object(self):
        count, joints, frames = 2, 28, 3
        joint_pos = torch.ones(count, joints)
        second_joint_pos = torch.full((count, joints), 2.)
        tip_pos = torch.ones(count, 2, 3)
        second_tip_pos = torch.full((count, 2, 3), 3.)
        tip_ref = torch.full((frames, 2, 7), 4.)
        second_tip_ref = torch.full((frames, 2, 7), 5.)
        object_pose = torch.tensor([[.2, .3, .4, 1., 0., 0., 0.]]).repeat(count, 1)
        object_ref = torch.tensor([[1., 2., 3., 1., 0., 0., 0.]]).repeat(frames, 1)
        env = SimpleNamespace(
            object=SimpleNamespace(data=SimpleNamespace(
                root_pos_w=object_pose[:, :3], root_quat_w=object_pose[:, 3:]
            )),
            scene=SimpleNamespace(env_origins=torch.zeros(count, 3)),
            hand=SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos)),
            second_hand=SimpleNamespace(data=SimpleNamespace(joint_pos=second_joint_pos)),
            cfg=SimpleNamespace(
                observation_mode="legacy", articulate_mode=False,
                log_rollout_diagnostics=False, observation_space=166,
            ),
            _bimanual_mode=True,
            num_envs=count,
            phase_buf=torch.zeros(count, dtype=torch.long),
            _current_fingertip_positions_w=lambda: tip_pos,
            _current_second_fingertip_positions_w=lambda: second_tip_pos,
            _reference_at=lambda values, phases: values[phases],
            reference_fingertip_pose=tip_ref,
            second_reference_fingertip_pose=second_tip_ref,
            reference_object_pose=object_ref,
            reference_hand_q=torch.full((frames, joints), 6.),
            second_reference_hand_q=torch.full((frames, joints), 7.),
            morphology_context=None,
        )
        observation = _method("_get_observations")(env)["policy"]
        self.assertEqual(tuple(observation.shape), (count, 166))
        torch.testing.assert_close(observation[:, :joints], joint_pos)
        torch.testing.assert_close(observation[:, 28:34], tip_pos.flatten(start_dim=1))
        torch.testing.assert_close(observation[:, 34:41], object_pose)
        torch.testing.assert_close(observation[:, 41:55], tip_ref[0].flatten().repeat(count, 1))
        torch.testing.assert_close(observation[:, 55:83], env.reference_hand_q[0].repeat(count, 1))
        torch.testing.assert_close(observation[:, 83:90], object_ref[0].repeat(count, 1))
        torch.testing.assert_close(observation[:, 90:118], second_joint_pos)
        torch.testing.assert_close(observation[:, 118:124], second_tip_pos.flatten(start_dim=1))
        torch.testing.assert_close(observation[:, 124:138], second_tip_ref[0].flatten().repeat(count, 1))
        torch.testing.assert_close(observation[:, 138:166], env.second_reference_hand_q[0].repeat(count, 1))

        env.cfg.articulate_mode = True
        env.cfg.observation_space = 168
        env.object.data.joint_pos = torch.full((count, 1), 8.)
        env.reference_object_joint = torch.full((frames, 1), 9.)
        articulated_observation = _method("_get_observations")(env)["policy"]
        self.assertEqual(tuple(articulated_observation.shape), (count, 168))
        torch.testing.assert_close(articulated_observation[:, 34:42],
                                   torch.cat((object_pose, env.object.data.joint_pos), dim=-1))
        torch.testing.assert_close(articulated_observation[:, 84:92],
                                   torch.cat((object_ref[0].repeat(count, 1),
                                              env.reference_object_joint[0].repeat(count, 1)), dim=-1))
        torch.testing.assert_close(articulated_observation[:, 92:120], second_joint_pos)
        torch.testing.assert_close(articulated_observation[:, 140:168],
                                   env.second_reference_hand_q[0].repeat(count, 1))

    def test_each_hand_has_its_own_pinch_reward(self):
        count = 2
        primary = torch.tensor([True, True])
        second = torch.tensor([True, False])
        contact = lambda mask: (mask, mask, mask, mask.float(), mask.float())
        cfg = SimpleNamespace(
            articulate_mode=False,
            object_z_reward_multiplier=1.,
            object_position_reward_weight=1.,
            object_rotation_reward_weight=.3,
            object_articulation_reward_weight=.3,
            object_failure_distance=.05,
            object_failure_orientation=1.5,
            object_failure_articulation=1.5,
            pose_tracking_reward_scale=1.,
            exponential_pose_reward=False,
            contact_tracking_sigma=0.,
            contact_reward_weight=2.,
            contact_force_threshold=0.,
            object_airborne_clearance=.01,
            object_airborne_reward_weight=0.,
            contact_force_safe_threshold=50.,
            contact_force_penalty_weight=0.,
            residual_action_penalty_weight=0.,
        )
        env = SimpleNamespace(
            cfg=cfg,
            _bimanual_mode=True,
            num_envs=count,
            device="cpu",
            object=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(count, 3))),
            scene=SimpleNamespace(env_origins=torch.zeros(count, 3)),
            phase_buf=torch.zeros(count, dtype=torch.long),
            _reference_length=3,
            reference_object_pose=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).repeat(3, 1),
            _reference_at=lambda values, phases: values[phases],
            _compute_object_errors=lambda: None,
            _object_position_error=torch.zeros(count),
            _object_rotation_error=torch.zeros(count),
            _object_articulation_error=torch.zeros(count),
            _compute_pinch_contact=lambda: contact(primary),
            _compute_second_pinch_contact=lambda: contact(second),
            _all_hand_contact_sensor=None,
            _primary_all_hand_contact_sensor=None,
            _second_all_hand_contact_sensor=None,
            _contact_sensor_force=lambda _: torch.zeros(count),
            actions=torch.zeros(count, 2),
            _finger_action_indices=torch.tensor([1]),
            residual_scale=torch.ones(2),
            _pose_episode_return=torch.zeros(count),
            _capture_enabled=False,
            reset_buf=torch.zeros(count, dtype=torch.bool),
            extras={},
        )
        reward = _method("_get_rewards")(env)
        shared_pose = math.sqrt(.05**2 + (.3 * 1.5)**2)
        torch.testing.assert_close(reward, torch.tensor([shared_pose + 4, shared_pose + 2]))
        torch.testing.assert_close(env._last_contact_reward, torch.tensor([4., 2.]))
        self.assertEqual(env._last_primary_pinch_contact.tolist(), [True, True])
        self.assertEqual(env._last_second_pinch_contact.tolist(), [True, False])
        self.assertEqual(env._last_pinch_contact.tolist(), [True, True])

        env.cfg.articulate_mode = True
        env._object_articulation_error = torch.full((count,), .2)
        env._all_hand_contact_sensors = ()
        env._primary_all_hand_contact_sensors = ()
        env._second_all_hand_contact_sensors = ()
        articulated_reward = _method("_get_rewards")(env)
        shared_articulated_pose = math.sqrt(.05**2 + (.3 * 1.5)**2 + (.3 * 1.5)**2) - .3 * .2
        torch.testing.assert_close(articulated_reward,
                                   torch.tensor([shared_articulated_pose + 4,
                                                 shared_articulated_pose + 2]))


if __name__ == "__main__":
    unittest.main()
