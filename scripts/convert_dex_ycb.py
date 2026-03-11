#!/usr/bin/env python3
"""Convert DexYCB dataset into egodex format.

DexYCB structure:
    DATASET/dex_ycb/
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
    CONVERTED/dex_ycb/
        {idx}_{subject}_{seq}/
            0.hdf5  (camera/intrinsic, transforms/*, transforms_cam/*, confidences/*)
            0.mp4   (color video, valid frames only)
            0_depth.mp4  (colorized depth video, valid frames only)

Usage:
    python scripts/convert_dex_ycb.py --src DATASET/dex_ycb --dst CONVERTED/dex_ycb
    --camera-idx 0 --fps 30 --max-samples 5
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import (
    collect_color_paths,
    collect_depth_paths,
    depth_images_to_mp4,
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
from utils.transforms import (
    interpolate_joint,
    invert_rigid,
    joints_to_transforms,
    make_transform,
)


def build_egodex_data_for_sequence(
    camera_dir: str,
    meta: dict,
    calibration_dir: str,
    serial: str,
    num_frames: int,
):
    """Build transforms, transforms_cam, and confidences dicts for one sequence+camera.

    DexYCB joint_3d is in camera coordinates. The extrinsic maps world->camera,
    so inv(extrinsic) = camera pose in world.

    Invalid frames (missing labels or joint_3d == -1) are filtered out entirely.

    Returns:
        intrinsic: (3, 3) array
        transforms_dict: {joint_name: (M, 4, 4)} world-space (M = valid frames)
        transforms_cam_dict: {joint_name: (M, 4, 4)} camera-space
        confidences_dict: {joint_name: (M,)}
        gravity: (3, 3) gravity alignment rotation
        valid_indices: (M,) original frame indices that are valid
    """
    intrinsic = load_intrinsics(calibration_dir, serial)
    extrinsic = load_extrinsics(calibration_dir, meta["extrinsics"], serial)

    # Camera pose in world = inv(extrinsic)
    cam_pose = invert_rigid(extrinsic)

    # DexYCB has exactly one hand side per sequence
    mano_side = meta["mano_sides"][0].lower()

    transforms_dict = {}
    transforms_cam_dict = {}
    confidences_dict = {}

    identity = np.eye(4, dtype=np.float32)

    # Collect per-frame joint_3d (in camera coordinates)
    all_joint_3d_cam = np.zeros((num_frames, 21, 3), dtype=np.float32)
    frame_valid = np.zeros(num_frames, dtype=bool)

    for frame_i in range(num_frames):
        label_path = os.path.join(camera_dir, f"labels_{frame_i:06d}.npz")
        if not os.path.exists(label_path):
            continue
        labels = load_frame_labels(label_path)
        if "joint_3d" not in labels:
            continue
        j3d = labels["joint_3d"][0].astype(np.float32)  # (21, 3)
        if np.any(j3d == -1):
            continue
        all_joint_3d_cam[frame_i] = j3d
        frame_valid[frame_i] = True

    # Filter to valid frames only
    valid_indices = np.where(frame_valid)[0]
    M = len(valid_indices)
    joint_3d_valid = all_joint_3d_cam[valid_indices]  # (M, 21, 3)

    # Convert camera-space joint positions to 4x4 transforms
    all_transforms_cam = np.zeros((M, 21, 4, 4), dtype=np.float32)
    for i in range(M):
        all_transforms_cam[i] = joints_to_transforms(joint_3d_valid[i])

    # Convert to world-space transforms: T_world = cam_pose @ T_cam
    all_transforms_world = cam_pose @ all_transforms_cam  # broadcast (4,4) @ (M,21,4,4)

    # Camera transform in world space (static, repeated for valid frames)
    cam_tf = np.tile(cam_pose, (M, 1, 1))
    transforms_dict["camera"] = cam_tf

    # Gravity: use the camera pose rotation as gravity alignment
    gravity = cam_pose[:3, :3].copy()

    # Body joints: not available in DexYCB
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        transforms_cam_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # All valid frames have confidence 1.0
    conf = np.ones(M, dtype=np.float32)

    # Assign to each hand side
    for side in ["left", "right"]:
        is_active = (side == mano_side)

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_cam_dict[name] = all_transforms_cam[:, mano_idx]
                transforms_dict[name] = all_transforms_world[:, mano_idx]
                confidences_dict[name] = conf.copy()
            else:
                transforms_cam_dict[name] = np.tile(identity, (M, 1, 1))
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc_cam = np.zeros((M, 4, 4), dtype=np.float32)
                mc_world = np.zeros((M, 4, 4), dtype=np.float32)
                for i in range(M):
                    pos = interpolate_joint(joint_3d_valid[i], idx_a, idx_b, alpha=0.3)
                    direction = joint_3d_valid[i, idx_b] - joint_3d_valid[i, idx_a]
                    mc_cam[i] = make_transform(pos, direction)
                    mc_world[i] = cam_pose @ mc_cam[i]
                transforms_cam_dict[name] = mc_cam
                transforms_dict[name] = mc_world
                confidences_dict[name] = conf.copy()
            else:
                transforms_cam_dict[name] = np.tile(identity, (M, 1, 1))
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

    return intrinsic, transforms_dict, transforms_cam_dict, confidences_dict, gravity, valid_indices


def _verify_output(out_dir: str, expected_frames: int):
    """Verify the converted output: check HDF5 frame counts and video frame counts."""
    hdf5_path = os.path.join(out_dir, "0.hdf5")
    with h5py.File(hdf5_path, "r") as f:
        # Check a sample transform has expected frame count
        sample_key = list(f["transforms"].keys())[0]
        if sample_key == "gravity":
            sample_key = list(f["transforms"].keys())[1]
        hdf5_frames = f[f"transforms/{sample_key}"].shape[0]
        if hdf5_frames != expected_frames:
            print(f"  WARNING: HDF5 has {hdf5_frames} frames, expected {expected_frames}")

    for video_name in ["0.mp4", "0_depth.mp4"]:
        video_path = os.path.join(out_dir, video_name)
        if not os.path.exists(video_path):
            print(f"  WARNING: {video_name} not created")
            continue
        cap = cv2.VideoCapture(video_path)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if video_frames != expected_frames:
            print(f"  WARNING: {video_name} has {video_frames} frames, expected {expected_frames}")
        else:
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  OK: {video_name} ({video_frames} frames, {size_mb:.1f} MB)")


def convert_dex_ycb(src_dir: str, dst_dir: str, camera_idx: int = 0,
                    fps: float = 30.0, max_samples: int = 0):
    """Convert DexYCB sequences to egodex format."""
    calibration_dir = os.path.join(src_dir, "calibration")
    os.makedirs(dst_dir, exist_ok=True)

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
            if max_samples > 0 and global_idx >= max_samples:
                break

            seq_path = os.path.join(subject_path, seq_name)
            meta_path = os.path.join(seq_path, "meta.yml")
            if not os.path.exists(meta_path):
                continue

            meta = load_yaml(meta_path)
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

            intrinsic, transforms_dict, transforms_cam_dict, confidences_dict, gravity, valid_indices = \
                build_egodex_data_for_sequence(
                    camera_dir, meta, calibration_dir, serial, num_frames,
                )

            n_valid = len(valid_indices)
            print(f"[{global_idx:06d}] {subject_dir_name}/{seq_name} "
                  f"(camera={serial}, frames={num_frames}, valid={n_valid})")

            if n_valid == 0:
                print(f"  Skipping: no valid frames")
                continue

            hdf5_path = os.path.join(out_dir, "0.hdf5")
            write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                              transforms_cam_dict, confidences_dict, gravity)

            # Color video (valid frames only)
            all_color_paths = collect_color_paths(camera_dir)
            valid_color_paths = [all_color_paths[i] for i in valid_indices
                                 if i < len(all_color_paths)]
            mp4_path = os.path.join(out_dir, "0.mp4")
            images_to_mp4(valid_color_paths, mp4_path, fps=fps)

            # Depth video (valid frames only)
            all_depth_paths = collect_depth_paths(camera_dir)
            valid_depth_paths = [all_depth_paths[i] for i in valid_indices
                                 if i < len(all_depth_paths)]
            depth_mp4_path = os.path.join(out_dir, "0_depth.mp4")
            depth_images_to_mp4(valid_depth_paths, depth_mp4_path, fps=fps)

            # Verify: check frame counts match
            _verify_output(out_dir, n_valid)

            global_idx += 1

        if max_samples > 0 and global_idx >= max_samples:
            break

    print(f"\nDone. Converted {global_idx} sequences to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert DexYCB to egodex format")
    parser.add_argument("--src", default="DATASET/dex_ycb", help="DexYCB source directory")
    parser.add_argument("--dst", default="CONVERTED/dex_ycb", help="Output directory")
    parser.add_argument("--camera-idx", type=int, default=0, help="Which camera index to use (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max sequences to convert (0=all)")
    args = parser.parse_args()

    convert_dex_ycb(args.src, args.dst, camera_idx=args.camera_idx,
                    fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
