#!/usr/bin/env python3
"""Convert DexYCB dataset into egodex format.

DexYCB structure:
    datasets/dex_ycb/
        {date}-{subject}/
            {timestamp}/
                meta.yml
                pose.npz
                {camera_serial}/
                    color_XXXXXX.jpg
                    aligned_depth_to_color_XXXXXX.png
                    labels_XXXXXX.npz  (seg, pose_y, pose_m, joint_3d, joint_2d)
        calibration/
            intrinsics/{serial}_640x480.yml
            extrinsics_{name}/extrinsics.yml
            mano_{name}/

Egodex structure:
    datasets/dex_ycb_cvt/
        {idx}_{subject}_{seq}/
            0.hdf5  (camera/intrinsic, transforms/*, confidences/*)
            0.mp4

Usage:
    python scripts/convert_dex_ycb.py [--src datasets/dex_ycb] [--dst datasets/dex_ycb_cvt]
                                      [--camera-idx 0] [--fps 30]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import (
    collect_color_paths,
    images_to_mp4,
    load_extrinsics,
    load_frame_labels,
    load_intrinsics,
    load_yaml,
    write_egodex_hdf5,
)
from utils.joint_mapping import (
    BODY_JOINTS,
    MANO_TO_EGODEX_SUFFIX,
    METACARPAL_INTERPOLATION,
)
from utils.transforms import interpolate_joint, joints_to_transforms, make_transform


def build_egodex_data_for_sequence(
    camera_dir: str,
    meta: dict,
    calibration_dir: str,
    serial: str,
    num_frames: int,
):
    """Build transforms and confidences dicts for one sequence+camera.

    Returns:
        intrinsic: (3, 3) array
        transforms_dict: {joint_name: (N, 4, 4)}
        confidences_dict: {joint_name: (N,)}
    """
    intrinsic = load_intrinsics(calibration_dir, serial)
    extrinsic = load_extrinsics(calibration_dir, meta["extrinsics"], serial)

    # DexYCB has exactly one hand side per sequence
    mano_side = meta["mano_sides"][0].lower()  # e.g. "right"

    transforms_dict = {}
    confidences_dict = {}

    # Camera transform (static, repeated per frame)
    cam_tf = np.tile(extrinsic, (num_frames, 1, 1))
    transforms_dict["camera"] = cam_tf

    # Identity transform for missing joints
    identity = np.eye(4, dtype=np.float32)

    # Body joints: not available in DexYCB
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (num_frames, 1, 1))
        confidences_dict[name] = np.zeros(num_frames, dtype=np.float32)

    # Collect per-frame joint_3d (in camera coordinates, always index 0)
    all_joint_3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
    frame_valid = np.zeros(num_frames, dtype=bool)

    for frame_i in range(num_frames):
        label_path = os.path.join(camera_dir, f"labels_{frame_i:06d}.npz")
        if not os.path.exists(label_path):
            continue
        labels = load_frame_labels(label_path)
        if "joint_3d" not in labels:
            continue
        # Always index [0]: one hand per label file (matching HaWoR loading)
        j3d = labels["joint_3d"][0].astype(np.float32)  # (21, 3)
        # DexYCB uses -1 for missing/invalid joint annotations
        if np.any(j3d == -1):
            continue
        all_joint_3d[frame_i] = j3d
        frame_valid[frame_i] = True

    # Convert joint positions to 4x4 transforms
    all_transforms = np.zeros((num_frames, 21, 4, 4), dtype=np.float32)
    for i in range(num_frames):
        if frame_valid[i]:
            all_transforms[i] = joints_to_transforms(all_joint_3d[i])
        else:
            all_transforms[i] = np.tile(identity, (21, 1, 1))

    conf = frame_valid.astype(np.float32)

    # Assign to each hand side
    for side in ["left", "right"]:
        is_active = (side == mano_side)

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms[:, mano_idx]
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (num_frames, 1, 1))
                confidences_dict[name] = np.zeros(num_frames, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc_transforms = np.zeros((num_frames, 4, 4), dtype=np.float32)
                for i in range(num_frames):
                    if frame_valid[i]:
                        pos = interpolate_joint(all_joint_3d[i], idx_a, idx_b, alpha=0.3)
                        direction = all_joint_3d[i, idx_b] - all_joint_3d[i, idx_a]
                        mc_transforms[i] = make_transform(pos, direction)
                    else:
                        mc_transforms[i] = identity
                transforms_dict[name] = mc_transforms
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (num_frames, 1, 1))
                confidences_dict[name] = np.zeros(num_frames, dtype=np.float32)

    return intrinsic, transforms_dict, confidences_dict


def convert_dex_ycb(src_dir: str, dst_dir: str, camera_idx: int = 0, fps: float = 30.0):
    """Convert all DexYCB sequences to egodex format."""
    calibration_dir = os.path.join(src_dir, "calibration")
    os.makedirs(dst_dir, exist_ok=True)

    # Discover all subject dirs
    subject_dirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and d.startswith("20")
    ])

    global_idx = 0
    for subject_dir_name in subject_dirs:
        subject_path = os.path.join(src_dir, subject_dir_name)
        subject_label = subject_dir_name.split("-", 1)[1] if "-" in subject_dir_name else subject_dir_name

        seq_dirs = sorted([
            d for d in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, d))
        ])

        for seq_name in seq_dirs:
            seq_path = os.path.join(subject_path, seq_name)
            meta_path = os.path.join(seq_path, "meta.yml")
            if not os.path.exists(meta_path):
                continue

            meta = load_yaml(meta_path)  # plain YAML, FullLoader is fine
            serials = meta["serials"]
            num_frames = meta["num_frames"]

            if camera_idx >= len(serials):
                print(f"  Skipping {subject_dir_name}/{seq_name}: camera_idx {camera_idx} out of range")
                continue

            serial = serials[camera_idx]
            camera_dir = os.path.join(seq_path, serial)
            if not os.path.isdir(camera_dir):
                print(f"  Skipping {subject_dir_name}/{seq_name}: camera dir not found")
                continue

            out_name = f"{global_idx:06d}_{subject_label}_{seq_name}"
            out_dir = os.path.join(dst_dir, out_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"[{global_idx:06d}] {subject_dir_name}/{seq_name} (camera={serial}, frames={num_frames})")

            # Build egodex data
            intrinsic, transforms_dict, confidences_dict = build_egodex_data_for_sequence(
                camera_dir, meta, calibration_dir, serial, num_frames,
            )

            # Write HDF5
            hdf5_path = os.path.join(out_dir, "0.hdf5")
            write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict, confidences_dict)

            # Write MP4
            color_paths = collect_color_paths(camera_dir)
            mp4_path = os.path.join(out_dir, "0.mp4")
            images_to_mp4(color_paths, mp4_path, fps=fps)

            global_idx += 1

    print(f"\nDone. Converted {global_idx} sequences to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert DexYCB to egodex format")
    parser.add_argument("--src", default="datasets/dex_ycb", help="DexYCB source directory")
    parser.add_argument("--dst", default="datasets/dex_ycb_cvt", help="Output directory")
    parser.add_argument("--camera-idx", type=int, default=0, help="Which camera index to use (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    args = parser.parse_args()

    convert_dex_ycb(args.src, args.dst, camera_idx=args.camera_idx, fps=args.fps)


if __name__ == "__main__":
    main()
