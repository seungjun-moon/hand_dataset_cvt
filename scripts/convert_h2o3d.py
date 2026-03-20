#!/usr/bin/env python3
"""
Convert H2O-3D dataset into egodex format.

H2O-3D structure:
    DATASET/h2o3d/
        train/
            {sequence_name}/
                rgb/{frame_id}.jpg
                depth/{frame_id}.png
                segr/{frame_id}.png
                meta/{frame_id}.pkl
        evaluation/
            {sequence_name}/
                rgb/{frame_id}.jpg
                depth/{frame_id}.png
                segr/{frame_id}.png
                meta/{frame_id}.pkl
        train.txt          # <seq_name>/<frame_id> per line
        evaluation.txt

Annotation keys (training, per-frame .pkl):
    leftHandPose:       (48,)  axis-angle (global_orient[0:3] + 15 joints[3:48])
    rightHandPose:      (48,)  axis-angle
    leftHandTrans:      (3,)   translation
    rightHandTrans:     (3,)   translation
    handBeta:           (10,)  shared MANO shape parameters
    leftHandJoints3D:   (21,3) 3D joint positions (camera space, OpenGL)
    rightHandJoints3D:  (21,3)
    jointValidLeft:     (21,)  per-joint validity
    jointValidRight:    (21,)
    poseValidLeft:      (48,)  per-MANO-param validity
    poseValidRight:     (48,)
    camMat:             (3,3)  camera intrinsics
    objName:            str    YCB object name

All annotations are in OpenGL camera space (origin at camera).

Joint ordering:
    H2O-3D stores joints in MANO FK output order:
        0: Wrist
        1-3: Index (MCP, PIP, DIP)
        4-6: Middle (MCP, PIP, DIP)
        7-9: Pinky (MCP, PIP, DIP)
        10-12: Ring (MCP, PIP, DIP)
        13-15: Thumb (CMC, MCP, IP)
        16: Thumb Tip, 17: Index Tip, 18: Middle Tip, 19: Ring Tip, 20: Pinky Tip

    Egodex expects "simple" order:
        0: Wrist
        1-4: Thumb (CMC, MCP, IP, Tip)
        5-8: Index (MCP, PIP, DIP, Tip)
        9-12: Middle (MCP, PIP, DIP, Tip)
        13-16: Ring (MCP, PIP, DIP, Tip)
        17-20: Little (MCP, PIP, DIP, Tip)

    MANO pose parameters (48 = 16×3) also need reordering from FK order
    to simple order for the 15 hand joints (excluding global orient).

Output structure:
    CONVERTED/h2o3d/
        {object_name}/
            {seq_idx:06d}_label_00.hdf5
            {seq_idx:06d}_video_00.mp4

Sequences are clustered by grasped YCB object name.
Only training split is converted (evaluation lacks hand pose annotations).

Usage:
    python scripts/convert_h2o3d.py --src ../ho3d/data/h2o3d --dst CONVERTED/h2o3d
    python scripts/convert_h2o3d.py --max-samples 5
"""

import argparse
import os
import pickle
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import (
    images_to_mp4,
    write_egodex_hdf5,
)
from utils.joint_mapping import (
    BODY_JOINTS,
    MANO_TO_EGODEX_SUFFIX,
    METACARPAL_INTERPOLATION,
)
from utils.transforms import (
    interpolate_joint,
    joints_to_transforms,
    make_transform,
)

# H2O-3D stores 21 joints in MANO FK output order. Egodex expects "simple"
# order (Wrist, Thumb×4, Index×4, Middle×4, Ring×4, Little×4).
# This array maps: joints_simple = joints_h2o[JOINT_REORDER]
# From vis_H2O3D.py: jointsMapManoToSimple
JOINT_REORDER = [
    0,                  # Wrist
    13, 14, 15, 16,     # Thumb (CMC, MCP, IP, Tip)
    1, 2, 3, 17,        # Index (MCP, PIP, DIP, Tip)
    4, 5, 6, 18,        # Middle (MCP, PIP, DIP, Tip)
    10, 11, 12, 19,     # Ring (MCP, PIP, DIP, Tip)
    7, 8, 9, 20,        # Little/Pinky (MCP, PIP, DIP, Tip)
]



def load_h2o3d_pkl(pkl_path: str) -> dict:
    """Load an H2O-3D annotation pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def convert_mano_axisangle_to_rotmat(hand_pose_batch: np.ndarray,
                                      hand_trans_batch: np.ndarray):
    """Convert H2O-3D axis-angle MANO params to rotation matrices.

    MANO hand_pose is stored in native MANO FK rotation order (matching
    DexYCB and HO-Cap conventions): Index, Middle, Pinky, Ring, Thumb.

    Args:
        hand_pose_batch: (M, 48) axis-angle pose params in MANO FK order.
            [0:3] global_orient, [3:48] hand_pose (15 joints × 3).
        hand_trans_batch: (M, 3) translation (camera/world space).

    Returns:
        global_orient: (M, 3, 3) global orientation as rotation matrices.
        hand_pose: (M, 15, 3, 3) per-joint rotation matrices in MANO FK order.
        transl: (M, 3) translation.
    """
    M = hand_pose_batch.shape[0]
    global_orient = np.zeros((M, 3, 3), dtype=np.float32)
    hand_pose = np.zeros((M, 15, 3, 3), dtype=np.float32)
    transl = hand_trans_batch.astype(np.float32)

    for i in range(M):
        R, _ = cv2.Rodrigues(hand_pose_batch[i, :3].astype(np.float64))
        global_orient[i] = R.astype(np.float32)

        for j in range(15):
            aa = hand_pose_batch[i, 3 + j * 3:3 + (j + 1) * 3].astype(np.float64)
            hand_pose[i, j], _ = cv2.Rodrigues(aa)

    return global_orient, hand_pose, transl


def _is_hand_degenerate(joints: np.ndarray) -> bool:
    """Check if 21×3 joint positions are degenerate (all zeros or identical)."""
    if np.all(np.abs(joints) < 1e-6):
        return True
    if np.std(joints) < 1e-6:
        return True
    return False


def build_egodex_data_for_sequence(
    seq_dir: str,
    frame_ids: list,
):
    """Build world-space transforms and confidences for one H2O-3D sequence.

    H2O-3D annotations are in OpenGL camera space (objects along -Z, Y up).
    cam_pose flips Y and Z so the egodex pipeline (OpenCV convention) works.

    Returns:
        intrinsic: (3, 3) array
        transforms_dict: {joint_name: (M, 4, 4)} world-space
        confidences_dict: {joint_name: (M,)}
        valid_indices: (M,) indices into frame_ids that are valid
        mano_dicts: list of mano_dict per active hand side
        obj_name: str, YCB object name for this sequence
    """
    # H2O-3D annotations are in OpenGL camera space (objects along -Z, Y up).
    # The egodex pipeline expects OpenCV convention (objects along +Z, Y down).
    # cam_pose encodes the transform from OpenCV camera frame to world (=OpenGL).
    # Flipping Y and Z converts OpenGL world coords to OpenCV camera coords.
    cam_pose = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    identity = np.eye(4, dtype=np.float32)

    num_frames = len(frame_ids)

    # Pre-load all annotations
    all_left_joints = np.zeros((num_frames, 21, 3), dtype=np.float32)
    all_right_joints = np.zeros((num_frames, 21, 3), dtype=np.float32)
    all_left_pose = np.zeros((num_frames, 48), dtype=np.float32)
    all_right_pose = np.zeros((num_frames, 48), dtype=np.float32)
    all_left_trans = np.zeros((num_frames, 3), dtype=np.float32)
    all_right_trans = np.zeros((num_frames, 3), dtype=np.float32)
    all_joint_valid_left = np.zeros((num_frames, 21), dtype=np.float32)
    all_joint_valid_right = np.zeros((num_frames, 21), dtype=np.float32)
    frame_valid = np.zeros(num_frames, dtype=bool)
    intrinsic = None
    obj_name = None
    hand_beta = None

    for i, fid in enumerate(frame_ids):
        pkl_path = os.path.join(seq_dir, "meta", f"{fid}.pkl")
        rgb_path = os.path.join(seq_dir, "rgb", f"{fid}.jpg")
        if not os.path.exists(pkl_path) or not os.path.exists(rgb_path):
            continue

        ann = load_h2o3d_pkl(pkl_path)

        # Extract camera intrinsics (same across frames)
        if intrinsic is None:
            intrinsic = np.array(ann["camMat"], dtype=np.float32).reshape(3, 3)
        if obj_name is None:
            obj_name = ann.get("objName", "unknown")
        if hand_beta is None:
            hand_beta = np.array(ann["handBeta"], dtype=np.float32).flatten()

        # Joint positions and validity
        left_j3d_raw = np.array(ann["leftHandJoints3D"], dtype=np.float32)
        right_j3d_raw = np.array(ann["rightHandJoints3D"], dtype=np.float32)
        jv_left_raw = np.array(ann["jointValidLeft"], dtype=np.float32).flatten()
        jv_right_raw = np.array(ann["jointValidRight"], dtype=np.float32).flatten()

        # Reorder joints from MANO FK order to simple/egodex order
        left_j3d = left_j3d_raw[JOINT_REORDER]
        right_j3d = right_j3d_raw[JOINT_REORDER]
        jv_left = jv_left_raw[JOINT_REORDER]
        jv_right = jv_right_raw[JOINT_REORDER]

        # Check validity: joint flags AND non-degenerate positions
        left_valid = np.all(jv_left > 0) and not _is_hand_degenerate(left_j3d)
        right_valid = np.all(jv_right > 0) and not _is_hand_degenerate(right_j3d)

        # Require at least one hand to be valid
        if not (left_valid or right_valid):
            continue

        all_left_joints[i] = left_j3d
        all_right_joints[i] = right_j3d
        # Store per-joint validity incorporating degeneracy check
        all_joint_valid_left[i] = jv_left if left_valid else 0.0
        all_joint_valid_right[i] = jv_right if right_valid else 0.0

        all_left_pose[i] = np.array(ann["leftHandPose"], dtype=np.float32).flatten()
        all_right_pose[i] = np.array(ann["rightHandPose"], dtype=np.float32).flatten()
        all_left_trans[i] = np.array(ann["leftHandTrans"], dtype=np.float32).flatten()
        all_right_trans[i] = np.array(ann["rightHandTrans"], dtype=np.float32).flatten()

        frame_valid[i] = True

    if intrinsic is None:
        return None, None, None, np.array([], dtype=int), [], obj_name

    valid_indices = np.where(frame_valid)[0]
    M = len(valid_indices)

    if M == 0:
        return intrinsic, None, None, valid_indices, [], obj_name

    transforms_dict = {}
    confidences_dict = {}

    # Camera transform (repeated for all valid frames)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Per-hand data
    mano_dicts = []
    for side in ["left", "right"]:
        if side == "left":
            joints_all = all_left_joints
            jv_all = all_joint_valid_left
            pose_all = all_left_pose
            trans_all = all_left_trans
        else:
            joints_all = all_right_joints
            jv_all = all_joint_valid_right
            pose_all = all_right_pose
            trans_all = all_right_trans

        joints_valid = joints_all[valid_indices]    # (M, 21, 3)
        jv_valid = jv_all[valid_indices]            # (M, 21)
        pose_valid = pose_all[valid_indices]        # (M, 48)
        trans_valid = trans_all[valid_indices]      # (M, 3)

        # Per-frame confidence: 1.0 only if all 21 joints valid AND not degenerate
        per_frame_conf = np.all(jv_valid > 0, axis=1).astype(np.float32)  # (M,)

        # Double-check: set confidence to 0 for degenerate frames
        for i in range(M):
            if per_frame_conf[i] > 0 and _is_hand_degenerate(joints_valid[i]):
                per_frame_conf[i] = 0.0

        # Compute transforms for this side (joints already in simple order)
        all_transforms = np.zeros((M, 21, 4, 4), dtype=np.float32)
        for i in range(M):
            if per_frame_conf[i] > 0:
                all_transforms[i] = joints_to_transforms(joints_valid[i])
            else:
                all_transforms[i] = np.tile(identity, (21, 1, 1))

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            transforms_dict[name] = all_transforms[:, mano_idx]
            confidences_dict[name] = per_frame_conf.copy()

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            mc = np.zeros((M, 4, 4), dtype=np.float32)
            for i in range(M):
                if per_frame_conf[i] > 0:
                    pos = interpolate_joint(joints_valid[i], idx_a, idx_b, alpha=0.3)
                    direction = joints_valid[i, idx_b] - joints_valid[i, idx_a]
                    mc[i] = make_transform(pos, direction)
                else:
                    mc[i] = identity.copy()
            transforms_dict[name] = mc
            confidences_dict[name] = per_frame_conf.copy()

        # Build MANO dict if this side has any valid frames
        if np.any(per_frame_conf > 0):
            go, hp, tr = convert_mano_axisangle_to_rotmat(pose_valid, trans_valid)

            # World-space 3D keypoints from transforms
            kpt3d = np.zeros((M, 21, 3), dtype=np.float32)
            for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
                name = f"{side}{suffix}"
                kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

            mano_dicts.append({
                "betas": hand_beta,
                "global_orient_worldspace": go,
                "hand_pose": hp,
                "transl_worldspace": tr,
                "kpt3d": kpt3d,
                "side": side,
            })

    return intrinsic, transforms_dict, confidences_dict, valid_indices, mano_dicts, obj_name


def _verify_output(out_dir: str, seq_idx: int, expected_frames: int):
    """Verify the converted output."""
    prefix = f"{seq_idx:06d}"
    hdf5_path = os.path.join(out_dir, f"{prefix}_label_00.hdf5")
    with h5py.File(hdf5_path, "r") as f:
        sample_key = list(f["transforms"].keys())[0]
        if sample_key == "gravity":
            sample_key = list(f["transforms"].keys())[1]
        hdf5_frames = f[f"transforms/{sample_key}"].shape[0]
        if hdf5_frames != expected_frames:
            print(f"  WARNING: HDF5 has {hdf5_frames} frames, expected {expected_frames}")

    video_path = os.path.join(out_dir, f"{prefix}_video_00.mp4")
    if not os.path.exists(video_path):
        print(f"  WARNING: {os.path.basename(video_path)} not created")
        return
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_frames != expected_frames:
        print(f"  WARNING: {os.path.basename(video_path)} has {video_frames} frames, expected {expected_frames}")
    else:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  OK: {os.path.basename(video_path)} ({video_frames} frames, {size_mb:.1f} MB)")


def convert_h2o3d(src_dir: str, dst_dir: str, fps: float = 30.0,
                  max_samples: int = 0):
    """Convert H2O-3D training sequences to egodex format, clustered by object."""
    train_dir = os.path.join(src_dir, "train")
    os.makedirs(dst_dir, exist_ok=True)

    # Discover all training sequences
    seq_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    # First pass: discover sequences and their objects
    # {obj_name: [(seq_name, [frame_ids]), ...]}
    clusters = {}
    for seq_name in seq_names:
        seq_dir = os.path.join(train_dir, seq_name)
        meta_dir = os.path.join(seq_dir, "meta")
        if not os.path.isdir(meta_dir):
            continue

        # Get sorted frame IDs from meta dir
        frame_ids = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(meta_dir) if f.endswith(".pkl")
        ])
        if not frame_ids:
            continue

        # Read first pkl to get object name
        first_pkl = os.path.join(meta_dir, f"{frame_ids[0]}.pkl")
        ann = load_h2o3d_pkl(first_pkl)
        obj_name = ann.get("objName", "unknown")

        clusters.setdefault(obj_name, []).append((seq_name, frame_ids))

    # Second pass: convert
    global_count = 0
    for obj_name in sorted(clusters.keys()):
        sequences = clusters[obj_name]
        obj_dir = os.path.join(dst_dir, obj_name)
        os.makedirs(obj_dir, exist_ok=True)

        for seq_idx, (seq_name, frame_ids) in enumerate(sequences):
            if max_samples > 0 and global_count >= max_samples:
                break

            seq_dir = os.path.join(train_dir, seq_name)

            result = build_egodex_data_for_sequence(seq_dir, frame_ids)
            intrinsic, transforms_dict, confidences_dict, valid_indices, mano_dicts, _ = result

            n_valid = len(valid_indices)
            active_sides = [d["side"] for d in mano_dicts]
            print(f"[{obj_name}/{seq_idx:06d}] {seq_name} "
                  f"(frames={len(frame_ids)}, valid={n_valid}, hands={active_sides})")

            if n_valid == 0:
                print(f"  Skipping: no valid frames")
                continue

            prefix = f"{seq_idx:06d}"

            # HDF5 label
            mano_dict = mano_dicts[0] if mano_dicts else None
            hdf5_path = os.path.join(obj_dir, f"{prefix}_label_00.hdf5")
            write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                              confidences_dict, mano_dict=mano_dict)

            # Write additional MANO groups for second hand
            if len(mano_dicts) > 1:
                with h5py.File(hdf5_path, "a") as f:
                    for extra_mano in mano_dicts[1:]:
                        side = extra_mano["side"]
                        grp = f.create_group(f"mano_{side}")
                        grp.create_dataset("betas",
                                           data=extra_mano["betas"].astype(np.float32))
                        grp.create_dataset("global_orient_worldspace",
                                           data=extra_mano["global_orient_worldspace"].astype(np.float32))
                        grp.create_dataset("hand_pose",
                                           data=extra_mano["hand_pose"].astype(np.float32))
                        grp.create_dataset("transl_worldspace",
                                           data=extra_mano["transl_worldspace"].astype(np.float32))
                        grp.create_dataset("kpt3d",
                                           data=extra_mano["kpt3d"].astype(np.float32))

            # RGB video (valid frames only)
            all_rgb_paths = sorted([
                os.path.join(seq_dir, "rgb", f"{fid}.jpg")
                for fid in frame_ids
            ])
            valid_rgb_paths = [all_rgb_paths[i] for i in valid_indices
                               if i < len(all_rgb_paths)]
            rgb_path = os.path.join(obj_dir, f"{prefix}_video_00.mp4")
            images_to_mp4(valid_rgb_paths, rgb_path, fps=fps)

            # Verify
            _verify_output(obj_dir, seq_idx, n_valid)
            global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    print(f"\nDone. Converted {global_count} sequences to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert H2O-3D to egodex format")
    parser.add_argument("--src", default="../ho3d/data/h2o3d",
                        help="H2O-3D source directory (containing train/ and evaluation/)")
    parser.add_argument("--dst", default="CONVERTED/h2o3d",
                        help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences to convert (0=all)")
    args = parser.parse_args()

    convert_h2o3d(args.src, args.dst, fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
