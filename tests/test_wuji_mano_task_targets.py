"""Direct target frame conversion and reference provenance contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'temp/hocap_mano_replay/scripts'))
from wuji_mano_task_targets import targets_in_wrist_frame
from prepare_wuji_parametric_training_assets import save_reference


def test_world_targets_round_trip_with_rotated_translated_wrist():
    r = Rotation.from_euler('xyz', [[.2,.3,.4],[-.4,.5,.2]])
    wrist = np.array([[1.,2.,3.],[3.,2.,1.]])
    points = np.arange(30).reshape(2,5,3) / 100
    directions = np.tile([0.,0.,1.], (2,5,1))
    p, d = targets_in_wrist_frame(points, directions, wrist, r.as_quat())
    np.testing.assert_allclose(np.einsum('tij,tfj->tfi',r.as_matrix(),p)+wrist[:,None],points,atol=1e-12)
    np.testing.assert_allclose(np.einsum('tij,tfj->tfi',r.as_matrix(),d),directions,atol=1e-12)


def test_reference_uses_explicit_task_targets_not_template(tmp_path):
    template=tmp_path/'template.npz';targets=tmp_path/'targets.npz';output=tmp_path/'out.npz'
    np.savez(template,hand_q=np.zeros((2,26)),object_pose_wxyz=np.zeros((2,7)),
             fingertip_pose_wxyz=np.zeros((2,2,7)),joint_names=np.array(['candidate_joint']))
    obj=np.arange(14,dtype=np.float32).reshape(2,7)
    tips=np.arange(28,dtype=np.float32).reshape(2,2,7)
    np.savez(targets,object_pose_wxyz=obj,fingertip_pose_wxyz=tips,frame_ids=[0,1],
             source_mano_rollout='capture.npz',source_mano_rollout_sha256='abc')
    save_reference(template,output,'candidate',np.ones((2,20)),np.zeros((2,3)),
                   np.tile([0.,0.,0.,1.],(2,1)),task_targets=targets)
    with np.load(output) as f:
        np.testing.assert_array_equal(f['object_pose_wxyz'],obj)
        np.testing.assert_array_equal(f['fingertip_pose_wxyz'],tips)
        np.testing.assert_array_equal(f['hand_ctrl'][:,6:],np.ones((2,20)))
        assert f['joint_names'].tolist()==['candidate_joint']
        assert str(f['source_mano_rollout'])=='capture.npz'


def test_reference_rejects_mismatched_capture_length(tmp_path):
    np.savez(tmp_path/'template.npz',hand_q=np.zeros((2,26)))
    np.savez(tmp_path/'targets.npz',object_pose_wxyz=np.zeros((3,7)))
    with pytest.raises(ValueError,match='object_pose_wxyz'):
        save_reference(tmp_path/'template.npz',tmp_path/'out.npz','candidate',np.ones((2,20)),
                       np.zeros((2,3)),np.tile([0.,0.,0.,1.],(2,1)),task_targets=tmp_path/'targets.npz')


def test_original_source_assets_keep_topology_and_reorder_torch_joints(tmp_path):
    from wuji_original_source_assets import prepare_original_source_hand

    root = tmp_path / 'artifacts/isaaclab_all_hands_residual'
    usd = root / 'assets/wuji_hand_2/hand.usd'
    urdf = root / 'prepared/wuji_hand_2/hand_rl.urdf'
    usd.parent.mkdir(parents=True)
    urdf.parent.mkdir(parents=True)
    usd.write_bytes(b'original-usd')
    urdf.write_bytes(b'original-urdf')
    names = [f'root_{i}' for i in range(6)] + ['finger__joint_a', 'finger__joint_b']
    np.savez(urdf.parent / 'reference.npz', hand_q=np.zeros((2, 8)),
             joint_names=names, palm_body_name='original_palm', contact_link_names=['original_tip'])
    targets = tmp_path / 'targets.npz'
    np.savez(targets, object_pose_wxyz=np.ones((2, 7)), fingertip_pose_wxyz=np.ones((2, 2, 7)),
             frame_ids=[0, 1], source_mano_rollout='capture.npz', source_mano_rollout_sha256='abc')
    retarget = tmp_path / 'retarget.npz'
    values = dict(vectors=np.zeros((1, 23)), joint_names=['joint_b', 'joint_a'],
                  qpos=np.array([[[2., 1.], [4., 3.]]]), wrist_position_all=np.zeros((1, 2, 3)),
                  wrist_quaternion_xyzw_all=np.tile([0., 0., 0., 1.], (1, 2, 1)))
    np.savez(retarget, **values)
    manifest, _ = prepare_original_source_hand(retarget, targets, tmp_path / 'prepared', tmp_path)
    assert manifest['hand_usd_paths'] == [str(usd)]
    assert usd.read_bytes() == b'original-usd'
    assert urdf.read_bytes() == b'original-urdf'
    with np.load(manifest['reference_paths'][0]) as ref:
        np.testing.assert_array_equal(ref['hand_q'][:, 6:], [[1., 2.], [3., 4.]])
        assert ref['joint_names'].tolist() == names
        assert ref['palm_body_name'] == 'original_palm'
        assert ref['contact_link_names'].tolist() == ['original_tip']
        np.testing.assert_array_equal(ref['object_pose_wxyz'], np.ones((2, 7)))
    values['vectors'][0, 1] = .1
    np.savez(retarget, **values)
    with pytest.raises(ValueError, match='zero source vector'):
        prepare_original_source_hand(retarget, targets, tmp_path / 'invalid', tmp_path)
