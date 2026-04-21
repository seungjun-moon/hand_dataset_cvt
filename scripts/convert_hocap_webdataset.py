#!/usr/bin/env python3
"""Convert HOCap dataset to WebDataset format (tar files).

Reads the HOCap multi-view hand-object dataset and writes tar files.
Each sample consists of:
    {prefix}/{sample_id}.jpg        — RGB image (640x480)
    {prefix}/{sample_id}.data.pyd   — pickled annotation list

Each RealSense camera view of each active hand is one sample.

Usage:
    python scripts/convert_hocap_webdataset.py \
        --src /rlwrld3/home/seungjun/HO-Cap/datasets \
        --dst CONVERTED/hocap-webdataset \
        --mano-pkl /rlwrld3/home/seungjun/hand_tracking_ablation/_DATA/data/mano/MANO_RIGHT.pkl \
        --max-subjects 1 --max-seqs 2 --cameras 105322251564
"""

import argparse
import io
import os
import pickle
import sys
import tarfile

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLES_PER_SHARD = 1000
DATASET_PREFIX = "hocap"
RS_WIDTH = 640
RS_HEIGHT = 480

# RealSense camera serials (excludes HoloLens)
ALL_RS_SERIALS = [
    "105322251564", "043422252387", "037522251142", "105322251225",
    "108222250342", "117222250549", "046122250168", "115422250549",
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_mano_pca(mano_pkl_path):
    """Load MANO PCA components and mean from the .pkl file."""
    with open(mano_pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    hands_components = np.array(data["hands_components"], dtype=np.float64)  # (45, 45)
    hands_mean = np.array(data["hands_mean"], dtype=np.float64)  # (45,)
    return hands_components, hands_mean


def pca_to_axis_angle(pca_coeffs, hands_components, hands_mean):
    """Convert PCA hand pose coefficients to axis-angle.

    manopth with use_pca=True, ncomps=45, flat_hand_mean=False computes:
        pose = coeffs @ components + mean
    This gives absolute axis-angle rotations (flat_hand_mean=True equivalent).
    """
    return (pca_coeffs @ hands_components + hands_mean).astype(np.float32)


def load_extrinsics(calib_dir):
    """Load camera extrinsics as world-to-camera transforms per serial."""
    ext_path = os.path.join(calib_dir, "extrinsics", "extrinsics_20231014.yaml")
    extrinsics = load_yaml(ext_path)["extrinsics"]

    def make_mat(values):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :] = np.array(values, dtype=np.float64).reshape(3, 4)
        return mat

    tag_1 = make_mat(extrinsics["tag_1"])
    tag_1_inv = np.linalg.inv(tag_1)

    cam_RTs = {}
    for serial in ALL_RS_SERIALS:
        if serial in extrinsics:
            rt_master = make_mat(extrinsics[serial])
            rt_world = tag_1_inv @ rt_master
            cam_RTs[serial] = rt_world
    return cam_RTs


def world_mano_to_camera(mano_pose_51, cam_RT_inv):
    """Transform MANO global orient + translation from world to camera space.

    Args:
        mano_pose_51: (51,) MANO params [global_orient(3), hand_pose(45), translation(3)]
        cam_RT_inv: (4,4) inverse of camera RT (world-to-camera transform)

    Returns:
        (51,) MANO params with global orient and translation in camera space.
    """
    pose = mano_pose_51.copy()
    global_orient = pose[:3]
    translation = pose[48:51]

    # Build 4x4 SE(3) matrix from axis-angle + translation
    R_w, _ = cv2.Rodrigues(global_orient.astype(np.float64))
    mat_w = np.eye(4, dtype=np.float64)
    mat_w[:3, :3] = R_w
    mat_w[:3, 3] = translation

    # Transform to camera space
    mat_c = cam_RT_inv @ mat_w

    # Extract back
    rvec_c, _ = cv2.Rodrigues(mat_c[:3, :3])
    pose[:3] = rvec_c.flatten().astype(np.float32)
    pose[48:51] = mat_c[:3, 3].astype(np.float32)
    return pose


def shift_kp3d_for_centered_pp(kp3d, cam_K, img_w, img_h):
    """Shift 3D keypoints so centered-PP projection matches actual K projection.

    The WebDataset visualizer projects with: u = f*X/Z + W/2
    But actual projection is: u = fx*X/Z + cx
    So shift X' = X + (cx - W/2) * Z / fx
    """
    shifted = kp3d.copy()
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    Z = kp3d[:, 2]
    shifted[:, 0] += (cx - img_w / 2.0) * Z / fx
    shifted[:, 1] += (cy - img_h / 2.0) * Z / fy
    return shifted


def compute_center_scale(kpts_2d, img_w, img_h, scale_factor=1.5):
    """Compute bounding box center and scale from 2D keypoints.

    Args:
        kpts_2d: (21, 2) or (21, 3) pixel keypoints.
        img_w, img_h: image dimensions.
        scale_factor: enlargement factor for the bounding box.

    Returns:
        center: (2,) float64
        scale: (2,) float64 (box_size / 200)
    """
    pts = kpts_2d[:, :2].astype(np.float64)
    valid = np.all(pts >= 0, axis=1)
    if valid.sum() < 2:
        return np.array([img_w / 2.0, img_h / 2.0]), np.array([img_w / 200.0, img_h / 200.0])

    pts = pts[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
    bbox_size = max(x_max - x_min, y_max - y_min) * scale_factor
    bbox_size = max(bbox_size, 50.0)  # minimum size
    scale = np.array([bbox_size / 200.0, bbox_size / 200.0])
    return center, scale


def build_annotation(kpts_2d, kpts_3d, cam_K, mano_pose_cam, betas, is_right,
                     img_w, img_h):
    """Build a single annotation dict matching HaMER WebDataset format."""
    # keypoints_2d: (21, 3) with confidence
    kp2d = np.zeros((21, 3), dtype=np.float32)
    kp2d[:, :2] = kpts_2d.astype(np.float32)
    # Mark valid keypoints (not -1)
    valid = np.all(kpts_2d >= 0, axis=1)
    kp2d[:, 2] = valid.astype(np.float32)

    # keypoints_3d: (21, 4) shifted for centered PP projection
    kp3d_shifted = shift_kp3d_for_centered_pp(kpts_3d, cam_K, img_w, img_h)
    kp3d = np.zeros((21, 4), dtype=np.float32)
    kp3d[:, :3] = kp3d_shifted.astype(np.float32)
    # Mark valid (joints with -1 values are invalid)
    valid_3d = ~np.all(kpts_3d == -1, axis=1)
    kp3d[:, 3] = valid_3d.astype(np.float32)

    # hand_pose: (48,) [global_orient(3) + hand_pose_aa(45)]
    hand_pose = np.zeros(48, dtype=np.float32)
    hand_pose[:3] = mano_pose_cam[:3]   # global orient in camera space
    hand_pose[3:48] = mano_pose_cam[3:48]  # axis-angle hand pose

    center, scale = compute_center_scale(kp2d, img_w, img_h)

    ann = {
        "keypoints_2d": kp2d,
        "keypoints_3d": kp3d,
        "center": center,
        "scale": scale,
        "right": np.float32(1.0 if is_right else 0.0),
        "hand_pose": hand_pose,
        "betas": betas.astype(np.float32),
        "has_hand_pose": np.float32(1.0),
        "has_betas": np.float32(1.0),
        "personid": 0,
        "extra_info": {},
    }
    return [ann]


def add_to_tar(tw, sample_id, img_bytes, annotation):
    """Add a jpg + pyd pair to a tar writer."""
    jpg_name = f"{DATASET_PREFIX}/{sample_id}.jpg"
    jpg_info = tarfile.TarInfo(name=jpg_name)
    jpg_info.size = len(img_bytes)
    tw.addfile(jpg_info, io.BytesIO(img_bytes))

    pyd_name = f"{DATASET_PREFIX}/{sample_id}.data.pyd"
    pyd_bytes = pickle.dumps(annotation)
    pyd_info = tarfile.TarInfo(name=pyd_name)
    pyd_info.size = len(pyd_bytes)
    tw.addfile(pyd_info, io.BytesIO(pyd_bytes))


def main():
    parser = argparse.ArgumentParser(
        description="Convert HOCap dataset to WebDataset tar files")
    parser.add_argument("--src", required=True,
                        help="Path to HOCap datasets root (containing subject_* dirs)")
    parser.add_argument("--dst", required=True,
                        help="Output directory for tar files")
    parser.add_argument("--mano-pkl",
                        default="/rlwrld3/home/seungjun/hand_tracking_ablation/_DATA/data/mano/MANO_RIGHT.pkl",
                        help="Path to MANO_RIGHT.pkl (for PCA components)")
    parser.add_argument("--samples-per-shard", type=int, default=SAMPLES_PER_SHARD)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="RealSense camera serials to use (default: all 8)")
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Max subjects to process (0=all)")
    parser.add_argument("--max-seqs", type=int, default=0,
                        help="Max sequences per subject (0=all)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    args = parser.parse_args()

    # Load MANO PCA components
    print("Loading MANO PCA components...")
    hands_components, hands_mean = load_mano_pca(args.mano_pkl)
    # Also load left hand PCA (typically same as right, but use the correct file)
    left_pkl = args.mano_pkl.replace("RIGHT", "LEFT")
    if os.path.exists(left_pkl):
        hands_components_left, hands_mean_left = load_mano_pca(left_pkl)
    else:
        print(f"  WARNING: {left_pkl} not found, using RIGHT for both hands")
        hands_components_left, hands_mean_left = hands_components, hands_mean

    # Load camera extrinsics
    calib_dir = os.path.join(args.src, "calibration")
    print("Loading camera extrinsics...")
    cam_RTs = load_extrinsics(calib_dir)

    # Determine which cameras to use
    cameras = args.cameras if args.cameras else ALL_RS_SERIALS
    cameras = [c for c in cameras if c in cam_RTs]
    print(f"Using {len(cameras)} cameras: {cameras}")

    # Pre-compute world-to-camera transforms
    cam_RT_invs = {serial: np.linalg.inv(cam_RTs[serial]) for serial in cameras}

    # Discover subjects
    subject_dirs = sorted([
        d for d in os.listdir(args.src)
        if d.startswith("subject_") and os.path.isdir(os.path.join(args.src, d))
    ])
    if args.max_subjects > 0:
        subject_dirs = subject_dirs[:args.max_subjects]
    print(f"Processing {len(subject_dirs)} subjects")

    os.makedirs(args.dst, exist_ok=True)

    # Collect all samples first, then write shards
    all_samples = []  # list of (img_path, annotation)
    total_skipped = 0

    for sub_id in subject_dirs:
        sub_dir = os.path.join(args.src, sub_id)
        # Load betas for this subject
        betas_path = os.path.join(calib_dir, "mano", f"{sub_id}.yaml")
        betas_data = load_yaml(betas_path)
        betas = np.array(betas_data["betas"], dtype=np.float32)  # (10,)

        # Get sequences
        seq_dirs = sorted([
            d for d in os.listdir(sub_dir)
            if os.path.isdir(os.path.join(sub_dir, d)) and d[0].isdigit()
        ])
        if args.max_seqs > 0:
            seq_dirs = seq_dirs[:args.max_seqs]

        for seq_id in seq_dirs:
            seq_dir = os.path.join(sub_dir, seq_id)
            meta = load_yaml(os.path.join(seq_dir, "meta.yaml"))
            mano_sides = meta["mano_sides"]
            num_frames = meta["num_frames"]

            # Load world-space MANO poses
            poses_m = np.load(os.path.join(seq_dir, "poses_m.npy"))  # (2, T, 51)

            print(f"  {sub_id}/{seq_id}: {num_frames} frames, sides={mano_sides}")

            for cam_serial in cameras:
                cam_dir = os.path.join(seq_dir, cam_serial)
                if not os.path.isdir(cam_dir):
                    continue

                cam_RT_inv = cam_RT_invs[cam_serial]

                for frame_idx in range(0, num_frames, args.frame_step):
                    img_path = os.path.join(cam_dir, f"color_{frame_idx:06d}.jpg")
                    label_path = os.path.join(cam_dir, f"label_{frame_idx:06d}.npz")

                    if not os.path.exists(img_path) or not os.path.exists(label_path):
                        total_skipped += 1
                        continue

                    label = np.load(label_path, allow_pickle=True)
                    cam_K = label["cam_K"]
                    hand_joints_2d = label["hand_joints_2d"]  # (2, 21, 2)
                    hand_joints_3d = label["hand_joints_3d"]  # (2, 21, 3)

                    for side in mano_sides:
                        hand_idx = 0 if side == "right" else 1
                        is_right = (side == "right")

                        # Check if hand data is valid
                        pose_world = poses_m[hand_idx, frame_idx]
                        if np.all(pose_world == -1):
                            total_skipped += 1
                            continue

                        j2d = hand_joints_2d[hand_idx]  # (21, 2)
                        j3d = hand_joints_3d[hand_idx]  # (21, 3)

                        # Skip if 2D keypoints are all invalid
                        if np.all(j2d == -1):
                            total_skipped += 1
                            continue

                        # Skip if most 2D keypoints are outside image
                        valid_2d = np.all(j2d >= 0, axis=1)
                        if valid_2d.sum() < 5:
                            total_skipped += 1
                            continue

                        # Transform MANO pose to camera space
                        pose_cam = world_mano_to_camera(pose_world, cam_RT_inv)

                        # Convert hand pose PCA coefficients to axis-angle
                        if is_right:
                            hand_pose_aa = pca_to_axis_angle(
                                pose_cam[3:48], hands_components, hands_mean)
                        else:
                            hand_pose_aa = pca_to_axis_angle(
                                pose_cam[3:48], hands_components_left, hands_mean_left)
                        pose_cam[3:48] = hand_pose_aa

                        ann = build_annotation(
                            j2d, j3d, cam_K, pose_cam, betas, is_right,
                            RS_WIDTH, RS_HEIGHT)

                        all_samples.append((img_path, ann))

            if len(all_samples) % 5000 < 100:
                print(f"    Collected {len(all_samples)} samples so far...")

    print(f"\nTotal samples: {len(all_samples)}, skipped: {total_skipped}")

    if not all_samples:
        print("No samples to write!")
        return

    # Write tar shards
    sps = args.samples_per_shard
    num_shards = (len(all_samples) + sps - 1) // sps
    print(f"Writing {num_shards} shards ({sps} samples/shard)...")

    total_written = 0
    for shard_idx in range(num_shards):
        start = shard_idx * sps
        end = min(start + sps, len(all_samples))

        tar_name = f"{shard_idx:06d}.tar"
        tar_path = os.path.join(args.dst, tar_name)

        with tarfile.open(tar_path, "w") as tw:
            for i in range(start, end):
                img_path, ann = all_samples[i]
                sample_id = f"{i:08d}"

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                add_to_tar(tw, sample_id, img_bytes, ann)
                total_written += 1

        print(f"  shard {shard_idx}: {tar_name} ({end - start} samples)")

    print(f"\nDone. Wrote {total_written} samples across {num_shards} shards to {args.dst}/")


if __name__ == "__main__":
    main()
