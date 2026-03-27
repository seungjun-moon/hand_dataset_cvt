#!/usr/bin/env python3
"""
Convert FreiHAND dataset into image-wise NPZ format.

FreiHAND consists of non-continuous images (no temporal coherence), so storing
as HDF5+MP4 sequences has no advantage. Instead, each image is saved as a
separate pair of files:
    {idx:06d}_{cam_idx:02d}.npz   (labels)
    {idx:06d}_{cam_idx:02d}.jpg   (image copy)

FreiHAND structure:
    {base_path}/
        training/
            rgb/00000000.jpg ... 00130239.jpg
        training_K.json       (32560 entries, 3x3 intrinsics)
        training_mano.json    (32560 entries, 61-dim: 48 pose + 10 shape + 2 uv_root + 1 scale)
        training_xyz.json     (32560 entries, 21x3 keypoints in camera space)
        evaluation/
            rgb/00000000.jpg ... 00003959.jpg
        evaluation_K.json     (3960 entries)
        evaluation_mano.json  (3960 entries)
        evaluation_xyz.json   (3960 entries)

    Training images have 4 versions (x32560 each):
        gs (0..32559), hom (32560..65119), sample (65120..97679), auto (97680..130239)

    All samples are right hand. Camera pose = identity (world = camera space).

    MANO params (61-dim): [axis-angle pose (48), shape betas (10), uv_root (2), scale (1)]
        pose[:3]   = global orientation (axis-angle)
        pose[3:48] = 15 joint rotations (axis-angle, NOT PCA)

Output structure:
    CONVERTED/freihand_train/
        cluster_00000/                          (images 0-999)
            {idx:06d}_{cam_idx:02d}.npz
            {idx:06d}_{cam_idx:02d}.jpg
        cluster_00001/                          (images 1000-1999)
            ...
    CONVERTED/freihand_eval/
        cluster_00000/
            {idx:06d}_00.npz
            {idx:06d}_00.jpg
        ...

NPZ contents per image:
    intrinsic:          (3, 3) float32 - camera intrinsic matrix
    cam_ext:            (4, 4) float32 - camera extrinsic (identity for FreiHAND)
    kpt3d_world:        (21, 3) float32 - 3D keypoints in world space
    side:               str - 'right' (always for FreiHAND)
    mano_betas:         (10,) float32
    mano_global_orient: (3, 3) float32 - rotation matrix
    mano_hand_pose:     (15, 3, 3) float32 - per-joint rotation matrices
    mano_transl:        (3,) float32 - wrist translation
    mano_kpt3d:         (21, 3) float32 - MANO keypoints

Usage:
    python scripts/convert_freihand.py --src RAW/freihand
    python scripts/convert_freihand.py --src RAW/freihand --max-images 100
    python scripts/convert_freihand.py --src RAW/freihand --skip-train
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.joint_mapping import (
    MANO_TO_EGODEX_SUFFIX,
    METACARPAL_INTERPOLATION,
)
from utils.transforms import (
    interpolate_joint,
    joints_to_transforms,
    make_transform,
)

VERSION_NAMES = ["gs", "hom", "sample", "auto"]
VERSION_OFFSETS = {name: i for i, name in enumerate(VERSION_NAMES)}
TRAIN_UNIQUE = 32560
EVAL_SIZE = 3960
IMAGES_PER_CLUSTER = 1000


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def convert_mano_to_rotmat(mano_params):
    """Convert FreiHAND 61-dim MANO params to rotation matrices."""
    mano_params = np.array(mano_params, dtype=np.float64).flatten()
    poses = mano_params[:48]
    betas = mano_params[48:58].astype(np.float32)

    global_orient, _ = cv2.Rodrigues(poses[:3].astype(np.float64))
    global_orient = global_orient.astype(np.float32)

    hand_pose = np.zeros((15, 3, 3), dtype=np.float32)
    for j in range(15):
        aa = poses[3 + j * 3: 3 + (j + 1) * 3].astype(np.float64)
        hand_pose[j], _ = cv2.Rodrigues(aa)

    return global_orient, hand_pose, betas


def build_single_image_data(K, mano_params, xyz):
    """Build label data for a single FreiHAND image.

    Args:
        K: (3, 3) camera intrinsics
        mano_params: (61,) MANO parameters
        xyz: (21, 3) 3D keypoints in camera space

    Returns:
        dict of label arrays ready for np.savez
    """
    intrinsic = np.array(K, dtype=np.float32)
    cam_ext = np.eye(4, dtype=np.float32)  # identity — world = camera space
    xyz = np.array(xyz, dtype=np.float32)

    # Joint transforms for egodex compatibility
    joint_transforms = joints_to_transforms(xyz)  # (21, 4, 4)

    # Extract kpt3d from joint transforms (world-space positions)
    kpt3d_world = np.zeros((21, 3), dtype=np.float32)
    for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
        kpt3d_world[mano_idx] = joint_transforms[mano_idx, :3, 3]

    # MANO rotation matrices
    global_orient, hand_pose, betas = convert_mano_to_rotmat(mano_params)

    side = "right"

    return {
        "intrinsic": intrinsic,                     # (3, 3)
        "cam_ext": cam_ext,                         # (4, 4)
        "kpt3d_world": kpt3d_world,                 # (21, 3)
        "side": side,
        "mano_betas": betas,                        # (10,)
        "mano_global_orient": global_orient,        # (3, 3)
        "mano_hand_pose": hand_pose,                # (15, 3, 3)
        "mano_transl": xyz[0].copy(),               # (3,) wrist position
        "mano_kpt3d": kpt3d_world.copy(),           # (21, 3)
    }


def convert_training(base_path, dst_dir, max_images=0,
                     images_per_cluster=IMAGES_PER_CLUSTER):
    """Convert FreiHAND training split to image-wise NPZ format."""
    k_path = os.path.join(base_path, "training_K.json")
    mano_path = os.path.join(base_path, "training_mano.json")
    xyz_path = os.path.join(base_path, "training_xyz.json")

    for p in [k_path, mano_path, xyz_path]:
        if not os.path.exists(p):
            print(f"Annotation file not found: {p} — skipping training split")
            return

    K_list = load_json(k_path)
    mano_list = load_json(mano_path)
    xyz_list = load_json(xyz_path)

    n_unique = len(K_list)
    assert len(mano_list) == n_unique and len(xyz_list) == n_unique
    print(f"Loaded {n_unique} training annotations")

    n_images = n_unique
    if max_images > 0:
        n_images = min(n_images, max_images)

    n_clusters = (n_images + images_per_cluster - 1) // images_per_cluster
    print(f"{n_images} unique images x {len(VERSION_NAMES)} views "
          f"-> {n_clusters} clusters ({images_per_cluster} images/cluster)")

    os.makedirs(dst_dir, exist_ok=True)

    total_count = 0
    for img_idx in range(n_images):
        cluster_idx = img_idx // images_per_cluster
        cluster_dir = os.path.join(dst_dir, f"cluster_{cluster_idx:05d}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Build label data (shared across all 4 views — same annotations)
        label_data = build_single_image_data(
            K_list[img_idx], mano_list[img_idx], xyz_list[img_idx])

        # Write 4 views (one per augmentation version)
        for cam_idx, version in enumerate(VERSION_NAMES):
            src_img_idx = img_idx + n_unique * VERSION_OFFSETS[version]
            src_path = os.path.join(
                base_path, "training", "rgb", f"{src_img_idx:08d}.jpg")

            if not os.path.exists(src_path):
                print(f"  WARNING: image not found: {src_path}")
                continue

            prefix = f"{img_idx:06d}_{cam_idx:02d}"

            # Save NPZ label
            npz_path = os.path.join(cluster_dir, f"{prefix}.npz")
            np.savez(npz_path, **label_data)

            # Copy image
            jpg_path = os.path.join(cluster_dir, f"{prefix}.jpg")
            shutil.copy2(src_path, jpg_path)

            total_count += 1

        if (img_idx + 1) % 1000 == 0 or img_idx == n_images - 1:
            print(f"  {img_idx + 1}/{n_images} images converted "
                  f"({total_count} files, cluster_{cluster_idx:05d})")

    print(f"\nDone. Converted {total_count} image+label pairs to {dst_dir}")


def convert_evaluation(base_path, dst_dir, max_images=0,
                       images_per_cluster=IMAGES_PER_CLUSTER):
    """Convert FreiHAND evaluation split to image-wise NPZ format."""
    k_path = os.path.join(base_path, "evaluation_K.json")
    mano_path = os.path.join(base_path, "evaluation_mano.json")
    xyz_path = os.path.join(base_path, "evaluation_xyz.json")

    for p in [k_path, mano_path, xyz_path]:
        if not os.path.exists(p):
            print(f"Annotation file not found: {p} — skipping evaluation split")
            return

    K_list = load_json(k_path)
    mano_list = load_json(mano_path)
    xyz_list = load_json(xyz_path)

    n_unique = len(K_list)
    assert len(mano_list) == n_unique and len(xyz_list) == n_unique
    print(f"Loaded {n_unique} evaluation annotations")

    n_images = n_unique
    if max_images > 0:
        n_images = min(n_images, max_images)

    n_clusters = (n_images + images_per_cluster - 1) // images_per_cluster
    print(f"{n_images} images -> {n_clusters} clusters")

    os.makedirs(dst_dir, exist_ok=True)

    total_count = 0
    for img_idx in range(n_images):
        cluster_idx = img_idx // images_per_cluster
        cluster_dir = os.path.join(dst_dir, f"cluster_{cluster_idx:05d}")
        os.makedirs(cluster_dir, exist_ok=True)

        label_data = build_single_image_data(
            K_list[img_idx], mano_list[img_idx], xyz_list[img_idx])

        src_path = os.path.join(
            base_path, "evaluation", "rgb", f"{img_idx:08d}.jpg")

        if not os.path.exists(src_path):
            print(f"  WARNING: image not found: {src_path}")
            continue

        prefix = f"{img_idx:06d}_00"

        npz_path = os.path.join(cluster_dir, f"{prefix}.npz")
        np.savez(npz_path, **label_data)

        jpg_path = os.path.join(cluster_dir, f"{prefix}.jpg")
        shutil.copy2(src_path, jpg_path)

        total_count += 1

        if (img_idx + 1) % 500 == 0 or img_idx == n_images - 1:
            print(f"  {img_idx + 1}/{n_images} images converted "
                  f"(cluster_{cluster_idx:05d})")

    print(f"\nDone. Converted {total_count} image+label pairs to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FreiHAND to image-wise NPZ format")
    parser.add_argument("--src", required=True,
                        help="Path to unpacked FreiHAND dataset root")
    parser.add_argument("--dst-train", default="CONVERTED/freihand_train",
                        help="Output directory for training split")
    parser.add_argument("--dst-eval", default="CONVERTED/freihand_eval",
                        help="Output directory for evaluation split")
    parser.add_argument("--images-per-cluster", type=int, default=IMAGES_PER_CLUSTER,
                        help=f"Images per cluster directory (default: {IMAGES_PER_CLUSTER})")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max unique images to convert (0=all)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training split")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation split")
    args = parser.parse_args()

    if not args.skip_train:
        print("=== Converting training split (image-wise NPZ) ===")
        convert_training(args.src, args.dst_train,
                         max_images=args.max_images,
                         images_per_cluster=args.images_per_cluster)

    if not args.skip_eval:
        print("\n=== Converting evaluation split (image-wise NPZ) ===")
        convert_evaluation(args.src, args.dst_eval,
                           max_images=args.max_images,
                           images_per_cluster=args.images_per_cluster)


if __name__ == "__main__":
    main()
