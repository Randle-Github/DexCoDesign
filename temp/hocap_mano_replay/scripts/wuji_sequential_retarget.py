"""Candidate-batched MANO IK with the original sequential retargeting rules.

No saved WUJI trajectory is used. Neutral geometry determines wrist alignment;
each candidate carries only its own previous solution between frames.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gpu_wuji_retarget import FINGERS, SOURCE_URDF, direction_error, parse_urdf


def solve_mano_sequence(kinematics, vectors, targets, iterations=14):
    if iterations < 1:
        raise ValueError("Retarget iterations must be positive")
    start = time.perf_counter()
    k = kinematics
    device, dtype = k.device, k.dtype
    vectors = torch.as_tensor(vectors, device=device, dtype=dtype)
    count, joints = len(vectors), len(k.joint_names)
    frames = len(targets['frame_ids'])
    points = torch.as_tensor(targets['target_points_world'], device=device, dtype=dtype)
    directions = torch.as_tensor(targets['target_directions_world'], device=device, dtype=dtype)
    if points.shape != (frames, 5, 3) or directions.shape != points.shape:
        raise ValueError("Expected world-space targets (frames, 5, 3)")
    if not torch.isfinite(points).all() or not torch.isfinite(directions).all():
        raise ValueError("Non-finite MANO targets")
    q = torch.zeros((count, joints), device=device, dtype=dtype)
    neutral = torch.stack([k.forward_finger(i, vectors, q)[0] for i in range(5)], dim=1).cpu().numpy()
    reference = np.asarray(targets['mano_tip_positions_local'][0], dtype=np.float64)
    if np.any(np.linalg.norm(reference,axis=-1) < 1e-8) or np.any(np.linalg.norm(neutral,axis=-1) < 1e-8):
        raise ValueError("Cannot align degenerate neutral fingertip vectors")
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    mappings = [Rotation.align_vectors(n / np.linalg.norm(n,axis=-1,keepdims=True), reference)[0]
                for n in neutral]
    mano_rotation = Rotation.from_quat(targets['mano_wrist_quaternion_xyzw'])
    wrist_rotations = np.stack([(mano_rotation * m.inv()).as_matrix() for m in mappings])
    rotations = torch.as_tensor(wrist_rotations, device=device, dtype=dtype)
    wrist = torch.as_tensor(targets['mano_wrist_position'][0], device=device, dtype=dtype).repeat(count,1)
    q_history = torch.empty((count,frames,joints),device=device,dtype=dtype)
    wrist_history = torch.empty((count,frames,3),device=device,dtype=dtype)
    tip_errors = torch.empty((count,frames),device=device,dtype=dtype)
    direction_errors = torch.empty_like(tip_errors)
    eye3 = torch.eye(3,device=device,dtype=dtype).expand(count,3,3)
    eyeq = torch.eye(joints,device=device,dtype=dtype).expand(count,joints,joints)
    q_indices = [[k.name_to_q[j.name] for j in k.chains[f] if j.movable] for f in FINGERS]
    for t in range(frames):
        frame_start = q.clone()
        rotation = rotations[:,t]
        active = torch.ones(count,device=device,dtype=torch.bool)
        for _ in range(iterations):
            rows, errors = [], []
            for i in range(5):
                p, d, jp, jr, _ = k.forward_finger(i, vectors, q)
                p_world = wrist + (rotation @ p.unsqueeze(-1)).squeeze(-1)
                d_world = (rotation @ d.unsqueeze(-1)).squeeze(-1)
                pos_row = torch.zeros((count,3,3+joints),device=device,dtype=dtype)
                pos_row[:,:,:3] = eye3
                pos_row[:,:,3+torch.tensor(q_indices[i],device=device)] = rotation @ jp
                rows.append(pos_row)
                errors.append(points[t,i]-p_world)
                projector = eye3-directions[t,i,:,None]*directions[t,i,None,:]
                dir_row = torch.zeros_like(pos_row)
                dir_row[:,:,3+torch.tensor(q_indices[i],device=device)] = .15 * (projector @ rotation @ jr)
                rows.append(dir_row)
                errors.append(.15 * direction_error(directions[t,i].expand_as(d_world),d_world))
            if t:
                reg = torch.zeros((count,joints,3+joints),device=device,dtype=dtype)
                reg[:,:,3:] = .015 * eyeq
                rows.append(reg)
                errors.append(.015*(frame_start-q))
            jac = torch.cat(rows,dim=1)
            error = torch.cat(errors,dim=1)
            # Primal DLS is algebraically equivalent to the original solver's
            # J.T @ solve(J @ J.T + damping**2 I, error), but smaller.
            normal = jac.mT @ jac
            normal.diagonal(dim1=-2,dim2=-1).add_(.025**2)
            step = torch.linalg.solve(normal,(jac.mT @ error.unsqueeze(-1))).squeeze(-1)
            step *= .12 / step.abs().amax(dim=-1,keepdim=True).clamp_min(.12)
            step *= active[:,None]
            wrist += step[:,:3]
            q += step[:,3:]
            if t:
                q = torch.maximum(torch.minimum(q,frame_start+.20),frame_start-.20)
            q = torch.maximum(torch.minimum(q,k.upper),k.lower)
            active &= error.norm(dim=-1) >= 5e-4
            if not active.any():
                break
        q_history[:,t] = q
        wrist_history[:,t] = wrist
        distances,angles = [],[]
        for i in range(5):
            p,d,*_ = k.forward_finger(i,vectors,q)
            distances.append((wrist+(rotation@p.unsqueeze(-1)).squeeze(-1)-points[t,i]).norm(dim=-1))
            angles.append(direction_error(directions[t,i].expand_as(d),(rotation@d.unsqueeze(-1)).squeeze(-1)).norm(dim=-1))
        tip_errors[:,t] = torch.stack(distances,dim=1).mean(dim=1)
        direction_errors[:,t] = torch.stack(angles,dim=1).mean(dim=1)
    # Convert optimized l_wrist poses to the virtual-root/base pose expected
    # by save_reference, accounting for the fixed source URDF transform.
    by_child,_ = parse_urdf(SOURCE_URDF)
    fixed = by_child['l_wrist']
    if fixed.parent != 'l_base_link' or fixed.movable:
        raise ValueError('Expected fixed base-to-wrist transform')
    base_rotations = wrist_rotations @ fixed.rotation.T
    base_positions = wrist_history.cpu().numpy() - np.einsum('ktij,j->kti',base_rotations,fixed.xyz)
    base_quaternions = Rotation.from_matrix(base_rotations.reshape(-1,3,3)).as_quat().reshape(count,frames,4)
    if not torch.isfinite(q_history).all() or not np.isfinite(base_positions).all():
        raise ValueError('Non-finite retarget solution')
    return dict(qpos=q_history.cpu().numpy(),wrist_position_all=base_positions.astype(np.float32),
                wrist_quaternion_xyzw_all=base_quaternions.astype(np.float32),
                tip_position_error_m=tip_errors.cpu().numpy(),tip_direction_error_rad=direction_errors.cpu().numpy()), dict(
                    seconds=time.perf_counter()-start,candidate_count=count,frame_count=frames,
                    iterations=iterations,initialization='neutral',temporal_anchor='previous_candidate_frame',
                    wrist_orientation_source='mano_with_candidate_neutral_alignment')
