"""Sequential candidate IK starts neutral and regularizes to its own past."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'temp/hocap_mano_replay/scripts'))
from gpu_wuji_retarget import WujiBatchKinematics, SOURCE_URDF, parse_urdf, resolve_design_vectors
from wuji_sequential_retarget import solve_mano_sequence


def fixture():
    torch.set_num_threads(1)
    k = WujiBatchKinematics(list(parse_urdf(SOURCE_URDF)[1]), torch.device('cpu'))
    v = resolve_design_vectors(np.zeros((1,23))).astype(np.float32)
    vt = torch.from_numpy(v)
    neutral = torch.zeros(1,20)
    points0 = torch.stack([k.forward_finger(i,vt,neutral)[0] for i in range(5)],dim=1).numpy()
    ps, ds = [], []
    for angle in [.65,.7,.75]:
        q = torch.full((1,20),angle)
        ps.append(torch.stack([k.forward_finger(i,vt,q)[0] for i in range(5)],dim=1)[0].numpy())
        ds.append(torch.stack([k.forward_finger(i,vt,q)[1] for i in range(5)],dim=1)[0].numpy())
    targets=dict(frame_ids=np.arange(3),target_points_world=np.array(ps),target_directions_world=np.array(ds),
                 mano_tip_positions_local=np.repeat(points0,3,axis=0),mano_wrist_position=np.zeros((3,3)),
                 mano_wrist_quaternion_xyzw=np.tile([0.,0.,0.,1.],(3,1)))
    return k,v,targets


def test_neutral_first_frame_is_not_capped_and_later_frames_are_temporal():
    k,v,t=fixture()
    out,stats=solve_mano_sequence(k,v,t,14)
    assert np.max(np.abs(out['qpos'][0,0])) > .20
    assert np.max(np.abs(np.diff(out['qpos'],axis=1))) <= .200001
    assert np.isfinite(out['qpos']).all()
    assert stats['initialization']=='neutral'
    assert stats['temporal_anchor']=='previous_candidate_frame'


def test_candidate_batch_does_not_share_trajectory_state():
    k,v,t=fixture()
    single,_=solve_mano_sequence(k,v,t,14)
    batch,_=solve_mano_sequence(k,np.repeat(v,2,axis=0),t,14)
    for key in ['qpos','wrist_position_all','wrist_quaternion_xyzw_all']:
        np.testing.assert_allclose(batch[key][0],single[key][0],atol=2e-5)
        np.testing.assert_allclose(batch[key][1],single[key][0],atol=2e-5)
