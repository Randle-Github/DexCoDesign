"""Reconstruct Bi-DexHands left/right fingertip poses from recorded joint states.

Some public 398-D trajectory mirrors duplicate the first hand's fingertip
observation into the second hand's slice.  This script uses the official Shadow
MJCF, rather than trusting that corrupted slice.  It first checks FK against
the recorded first-hand fingertips; no corrected data are emitted when the FK
agreement is poor.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


FINGERS = ("ff", "mf", "rf", "lf", "th")


def load_legacy_shadow(source: Path, side_index: int, temporary: Path):
    hand_dir = source / "mjcf/open_ai_assets/hand"
    stls_dir = source / "mjcf/open_ai_assets/stls/hand"
    textures_dir = source / "mjcf/open_ai_assets/textures"
    copied_hand = temporary / "hand"
    shutil.copytree(hand_dir, copied_hand)
    shutil.copytree(stls_dir, temporary / "stls/hand")
    if textures_dir.is_dir():
        shutil.copytree(textures_dir, temporary / "textures")
    for xml in copied_hand.glob("*.xml"):
        text = xml.read_text()
        text = text.replace(' coordinate="local"', "").replace(' apirate="200"', "")
        xml.write_text(text)
    path = copied_hand / ("shadow_hand.xml" if side_index == 0 else "shadow_hand1.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [f"robot{side_index}:{finger}distal" for finger in FINGERS]
    body_ids = [model.body(name).id for name in names]
    return model, body_ids


def local_fk(model, body_ids, scaled_qpos: np.ndarray) -> np.ndarray:
    if scaled_qpos.shape[1] != model.nq:
        raise ValueError(f"Recorded qpos {scaled_qpos.shape[1]} != official MJCF nq {model.nq}")
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    qpos = lower + 0.5 * (scaled_qpos + 1.0) * (upper - lower)
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), len(body_ids), 3))
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[body_ids]
    return positions


def align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    a, b = source.reshape(-1, 3), target.reshape(-1, 3)
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    translation = bc - ac @ rotation
    error = float(np.linalg.norm(a @ rotation + translation - b, axis=-1).mean())
    return rotation, translation, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample = np.load(args.sample)
    with tempfile.TemporaryDirectory(prefix="bidex-shadow-fk-") as temp:
        # Separate temp trees because both source files share includes and names.
        right_model, right_ids = load_legacy_shadow(args.asset_root, 0, Path(temp) / "right")
        left_model, left_ids = load_legacy_shadow(args.asset_root, 1, Path(temp) / "left")
        right_local = local_fk(right_model, right_ids, sample["right_qpos_scaled"])
        left_local = local_fk(left_model, left_ids, sample["left_qpos_scaled"])

    right_recorded = sample["right_fingertip_state_world"][:, :, :3]
    rotation, translation, right_error = align(right_local, right_recorded)
    print(f"right_fk_alignment_mean_error_m={right_error:.6f}")
    print(f"right_global_alignment_rotation={rotation.tolist()}")
    print(f"right_global_alignment_translation={translation.tolist()}")
    print(f"right_local_first_tip={right_local[0, 0].tolist()}")
    print(f"left_local_first_tip={left_local[0, 0].tolist()}")
    if right_error > 0.02:
        raise RuntimeError("Official Shadow FK does not agree with recorded right fingertips")
    right_world = right_local @ rotation + translation

    # The legacy MJCF puts both hand roots at an offset within the XML.  Isaac
    # Gym re-centers that root at each actor start pose.  Subtract the XML root
    # before applying the official second-hand start transform.
    left_root = np.array([0.0, -1.0, 0.5])
    xml_root = left_model.body("robot1:hand mount").pos.copy()
    print(f"left_xml_root={xml_root.tolist()}")
    left_local = Rotation.from_euler("x", np.pi).apply((left_local - xml_root).reshape(-1, 3)).reshape(left_local.shape)
    left_world = left_local + left_root
    # Verify the left fingertips remain within a physically plausible reach of
    # the object.  This is a proxy only; no Isaac Gym replay is implied.
    object_pos = sample["object_pose_xyzw"][:, None, :3]
    left_object_gap = np.linalg.norm(left_world - object_pos, axis=-1).min(axis=1)
    print(f"left_object_gap_median_m={np.median(left_object_gap):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        right_fingertips_world=right_world,
        left_fingertips_world=left_world,
        object_pose_xyzw=sample["object_pose_xyzw"],
        right_fk_alignment_mean_error_m=right_error,
        left_object_gap_median_m=np.median(left_object_gap),
    )


if __name__ == "__main__":
    main()
