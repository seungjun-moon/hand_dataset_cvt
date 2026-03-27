#!/usr/bin/env python3
"""
Convert HIC (Hand-in-Contact) dataset into egodex format.

HIC structure:
    ROOT/
        {seq_id}/
            1/
                rgb/
                    {frame_idx:03d}.png
        IJCV16___Results_MANO___parms_for___joints21/
            IJCV16___fakeGT___IJCV___{seq_id}___Model_Hand_{L,R}___ncomps45/
                {frame_idx:03d}.pkl   (pose, trans, betas, J_transformed___j21, v)
                {frame_idx:03d}.ply
        data/
            HIC.json   (COCO-style annotations with image paths and mano ply paths)

MANO pkl format:
    pose:                  (48,) axis-angle [0:3] global_orient + [3:48] hand_pose
    trans:                 (3,)  translation (camera space)
    betas:                 (10,) shape parameters
    J_transformed___j21:   (21, 3) 3D joint positions (camera space)
    v:                     (778, 3) vertices (camera space)

Camera:
    Fixed intrinsics: fx=fy=525.0, cx=319.5, cy=239.5
    No extrinsics (single camera, identity world-to-camera)

Output structure:
    CONVERTED/hic/
        {seq_name}/
            000000_label_00.hdf5
            000000_video_00.mp4

Usage:
    python scripts/convert_hic.py --src ../InterWild/data/HIC --dst CONVERTED/hic
    python scripts/convert_hic.py --max-samples 3
"""

import argparse
import json
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

# Fixed HIC camera intrinsics
HIC_INTRINSIC = np.array([
    [525.0, 0, 319.5],
    [0, 525.0, 239.5],
    [0, 0, 1],
], dtype=np.float32)


def load_mano_pkl(pkl_path: str) -> dict:
    """Load a HIC MANO pkl file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def convert_mano_axisangle_to_rotmat(pose: np.ndarray):
    """Convert (48,) axis-angle MANO pose to rotation matrices.

    Args:
        pose: (48,) axis-angle. [0:3] global_orient, [3:48] hand_pose (15×3).

    Returns:
        global_orient: (3, 3) rotation matrix.
        hand_pose: (15, 3, 3) per-joint rotation matrices.
    """
    global_orient, _ = cv2.Rodrigues(pose[:3].astype(np.float64))
    global_orient = global_orient.astype(np.float32)

    hand_pose = np.zeros((15, 3, 3), dtype=np.float32)
    for j in range(15):
        aa = pose[3 + j * 3:3 + (j + 1) * 3].astype(np.float64)
        hand_pose[j], _ = cv2.Rodrigues(aa)

    return global_orient, hand_pose


def build_egodex_data_for_sequence(
    src_dir: str,
    seq_name: str,
    annotations: list,
    images: dict,
):
    """Build world-space transforms for one HIC sequence.

    HIC uses a single camera with identity extrinsics, so camera space == world space.

    Returns:
        intrinsic: (3, 3)
        transforms_dict: {joint_name: (M, 4, 4)}
        confidences_dict: {joint_name: (M,)}
        valid_image_paths: list of str (absolute paths to valid RGB frames)
        mano_dicts: list of mano_dict per active hand side
    """
    data_dir = os.path.join(src_dir, "data")
    mano_fits_dir = os.path.join(
        src_dir, "IJCV16___Results_MANO___parms_for___joints21")

    # Determine which hands are active across the sequence
    has_right = any(a["right_mano_path"] is not None for a in annotations)
    has_left = any(a["left_mano_path"] is not None for a in annotations)

    # Sort annotations by frame index for consistent ordering
    def frame_idx_from_ann(ann):
        img = images[ann["image_id"]]
        fname = os.path.basename(img["file_name"])
        return int(os.path.splitext(fname)[0])

    annotations = sorted(annotations, key=frame_idx_from_ann)

    M = len(annotations)
    identity = np.eye(4, dtype=np.float32)
    # Camera is identity (single camera, camera space == world space)
    cam_pose = np.eye(4, dtype=np.float32)

    transforms_dict = {}
    confidences_dict = {}
    conf_ones = np.ones(M, dtype=np.float32)
    conf_zeros = np.zeros(M, dtype=np.float32)

    # Camera transform (static identity)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = conf_zeros.copy()

    # Collect per-frame joint data
    active_sides = {}
    if has_right:
        active_sides["right"] = "right_mano_path"
    if has_left:
        active_sides["left"] = "left_mano_path"

    # Per-side arrays
    all_joint_3d = {side: np.zeros((M, 21, 3), dtype=np.float32)
                    for side in active_sides}
    all_poses = {side: np.zeros((M, 48), dtype=np.float64)
                 for side in active_sides}
    all_trans = {side: np.zeros((M, 3), dtype=np.float64)
                 for side in active_sides}
    all_betas = {side: None for side in active_sides}

    valid_image_paths = []

    for i, ann in enumerate(annotations):
        img = images[ann["image_id"]]
        img_path = os.path.join(src_dir, img["file_name"])
        valid_image_paths.append(img_path)

        for side, mano_key in active_sides.items():
            mano_path = ann[mano_key]
            if mano_path is None:
                continue

            # pkl path: replace .ply with .pkl
            pkl_rel = mano_path.replace(".ply", ".pkl")
            pkl_path = os.path.join(data_dir, pkl_rel)
            if not os.path.exists(pkl_path):
                # Try from src_dir directly
                pkl_path = os.path.join(src_dir, pkl_rel)

            if not os.path.exists(pkl_path):
                print(f"  WARNING: pkl not found: {pkl_path}")
                continue

            mano_data = load_mano_pkl(pkl_path)
            all_joint_3d[side][i] = mano_data["J_transformed___j21"].astype(np.float32)
            all_poses[side][i] = mano_data["pose"]
            all_trans[side][i] = mano_data["trans"]
            if all_betas[side] is None:
                all_betas[side] = mano_data["betas"].astype(np.float32)

    # Build transforms for each hand side
    for side in ["left", "right"]:
        is_active = side in active_sides

        if is_active:
            joint_3d = all_joint_3d[side]  # (M, 21, 3) already in camera/world space
            all_transforms = np.zeros((M, 21, 4, 4), dtype=np.float32)
            for i in range(M):
                all_transforms[i] = joints_to_transforms(joint_3d[i])

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms[:, mano_idx]
                confidences_dict[name] = conf_ones.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = conf_zeros.copy()

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc = np.zeros((M, 4, 4), dtype=np.float32)
                for i in range(M):
                    pos = interpolate_joint(joint_3d[i], idx_a, idx_b, alpha=0.3)
                    direction = joint_3d[i, idx_b] - joint_3d[i, idx_a]
                    mc[i] = make_transform(pos, direction)
                transforms_dict[name] = mc
                confidences_dict[name] = conf_ones.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = conf_zeros.copy()

    # Build MANO dicts
    mano_dicts = []
    for side in active_sides:
        global_orients = np.zeros((M, 3, 3), dtype=np.float32)
        hand_poses = np.zeros((M, 15, 3, 3), dtype=np.float32)

        for i in range(M):
            go, hp = convert_mano_axisangle_to_rotmat(all_poses[side][i])
            global_orients[i] = go
            hand_poses[i] = hp

        # World-space 3D keypoints from transforms
        kpt3d = np.zeros((M, 21, 3), dtype=np.float32)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

        mano_dicts.append({
            "betas": all_betas[side],
            "global_orient_worldspace": global_orients,
            "hand_pose": hand_poses,
            "transl_worldspace": all_trans[side].astype(np.float32),
            "kpt3d": kpt3d,
            "side": side,
        })

    return HIC_INTRINSIC, transforms_dict, confidences_dict, valid_image_paths, mano_dicts


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


def convert_hic(src_dir: str, dst_dir: str, fps: float = 30.0,
                max_samples: int = 0):
    """Convert HIC sequences to egodex format."""
    os.makedirs(dst_dir, exist_ok=True)

    # Load HIC.json
    json_path = os.path.join(src_dir, "data", "HIC.json")
    with open(json_path) as f:
        hic_data = json.load(f)

    # Index images by id
    images = {img["id"]: img for img in hic_data["images"]}

    # Group annotations by sequence
    seq_annotations = {}
    for ann in hic_data["annotations"]:
        img = images[ann["image_id"]]
        seq_name = img["seq_name"]
        seq_annotations.setdefault(seq_name, []).append(ann)

    seq_names = sorted(seq_annotations.keys())
    count = 0

    for seq_idx, seq_name in enumerate(seq_names):
        if max_samples > 0 and count >= max_samples:
            break

        annotations = seq_annotations[seq_name]
        n_frames = len(annotations)
        active_types = set(a["hand_type"] for a in annotations)

        intrinsic, transforms_dict, confidences_dict, valid_paths, mano_dicts = \
            build_egodex_data_for_sequence(src_dir, seq_name, annotations, images)

        active_sides = [d["side"] for d in mano_dicts]
        print(f"[{seq_idx:06d}] seq={seq_name} "
              f"(frames={n_frames}, type={active_types}, hands={active_sides})")

        if not valid_paths:
            print(f"  Skipping: no valid frames")
            continue

        # Create a subdirectory per sequence
        seq_out_dir = os.path.join(dst_dir, seq_name)
        os.makedirs(seq_out_dir, exist_ok=True)
        prefix = "000000"

        # Write HDF5
        mano_dict = mano_dicts[0] if mano_dicts else None
        hdf5_path = os.path.join(seq_out_dir, f"{prefix}_label_00.hdf5")
        write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                          confidences_dict, mano_dict=mano_dict)

        # Write additional MANO groups for multi-hand sequences
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

        # RGB video
        rgb_path = os.path.join(seq_out_dir, f"{prefix}_video_00.mp4")
        images_to_mp4(valid_paths, rgb_path, fps=fps)

        # Verify
        _verify_output(seq_out_dir, 0, len(valid_paths))
        count += 1

    print(f"\nDone. Converted {count} sequences to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert HIC to egodex format")
    parser.add_argument("--src", default="../InterWild/data/HIC",
                        help="HIC dataset directory")
    parser.add_argument("--dst", default="CONVERTED/hic",
                        help="Output directory (sequences stored as subdirs)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences to convert (0=all)")
    args = parser.parse_args()

    convert_hic(args.src, args.dst, fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
