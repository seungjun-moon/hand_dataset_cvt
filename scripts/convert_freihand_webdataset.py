#!/usr/bin/env python3
"""
Convert FreiHAND evaluation dataset to WebDataset format (tar files).

Reads the FreiHAND_eval directory and writes tar files with 1000 samples each.
Each sample consists of:
    {prefix}/{sample_id}.jpg        — RGB image (224x224)
    {prefix}/{sample_id}.data.pyd   — pickled annotation list

Annotation format matches existing HaMER WebDataset tars:
    [{
        "keypoints_2d": (21, 3) float32,
        "keypoints_3d": (21, 4) float32,
        "center": (2,) float64,
        "scale": (2,) float64,
        "right": float32 scalar (1.0),
        "hand_pose": (48,) float32,
        "betas": (10,) float32,
        "has_hand_pose": float32 scalar (1.0),
        "has_betas": float32 scalar (1.0),
        "personid": int (0),
        "extra_info": dict {},
    }]

Usage:
    python scripts/convert_freihand_webdataset.py \
        --src ../hand_tracking_ablation/_DATA/datasets/FreiHAND_eval \
        --dst ../hand_tracking_ablation/hamer_evaluation_data/dataset_tars/freihand-eval/
"""

import argparse
import io
import json
import os
import pickle
import sys
import tarfile

import cv2
import numpy as np
import smplx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLES_PER_SHARD = 1000
DATASET_PREFIX = "freihand-eval"
DEFAULT_MANO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "_DATA", "data", "mano"
)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def project_keypoints_2d(xyz, K):
    """Project 3D keypoints to 2D using camera intrinsics."""
    xyz = np.array(xyz, dtype=np.float32)
    K = np.array(K, dtype=np.float32)
    pts_2d = (K @ xyz.T).T  # (21, 3)
    pts_2d[:, :2] /= pts_2d[:, 2:3]
    result = np.zeros((21, 3), dtype=np.float32)
    result[:, :2] = pts_2d[:, :2]
    result[:, 2] = 1.0
    return result


def build_annotation(K, mano_params, xyz, img_h, img_w):
    """Build a single annotation dict matching HaMER WebDataset format."""
    mano_params = np.array(mano_params, dtype=np.float32).flatten()
    xyz = np.array(xyz, dtype=np.float32)
    K_arr = np.array(K, dtype=np.float32)

    hand_pose = mano_params[:48].copy()
    betas = mano_params[48:58]

    # FreiHAND uses flat_hand_mean=False (offsets from natural grasping mean).
    # HaMER uses flat_hand_mean=True (absolute rotations from flat hand).
    # Add the mean pose to convert FreiHAND convention → HaMER convention.
    hand_pose[3:48] += HAND_MEAN_POSE

    kpts_2d = project_keypoints_2d(xyz, K_arr)

    kpts_3d = np.zeros((21, 4), dtype=np.float32)
    kpts_3d[:, :3] = xyz
    kpts_3d[:, 3] = 1.0

    # FreiHAND images are already 224x224 crops centered on the hand.
    # Use fixed center/scale matching HaMER training convention.
    center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float64)
    scale = np.array([img_w / 200.0, img_h / 200.0], dtype=np.float64)

    ann = {
        "keypoints_2d": kpts_2d,
        "keypoints_3d": kpts_3d,
        "center": center,
        "scale": scale,
        "right": np.float32(1.0),
        "hand_pose": hand_pose,
        "betas": betas,
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
        description="Convert FreiHAND eval to WebDataset tar files")
    parser.add_argument("--src",
                        default="../hand_tracking_ablation/_DATA/datasets/FreiHAND_eval",
                        help="Path to FreiHAND_eval dataset root")
    parser.add_argument("--dst",
                        default="../hand_tracking_ablation/hamer_evaluation_data/dataset_tars/freihand-eval/",
                        help="Output directory for tar files")
    parser.add_argument("--samples-per-shard", type=int,
                        default=SAMPLES_PER_SHARD,
                        help=f"Samples per tar shard (default: {SAMPLES_PER_SHARD})")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to convert (0=all)")
    args = parser.parse_args()

    # Load annotations
    k_path = os.path.join(args.src, "evaluation_K.json")
    mano_path = os.path.join(args.src, "evaluation_mano.json")
    xyz_path = os.path.join(args.src, "evaluation_xyz.json")

    for p in [k_path, mano_path, xyz_path]:
        if not os.path.exists(p):
            print(f"ERROR: annotation file not found: {p}")
            sys.exit(1)

    # Load MANO mean hand pose for convention conversion
    global HAND_MEAN_POSE
    mano_model = smplx.MANOLayer(
        model_path=os.path.normpath(DEFAULT_MANO_DIR),
        is_rhand=True, flat_hand_mean=False)
    HAND_MEAN_POSE = mano_model.hand_mean.numpy()  # (45,)

    print("Loading annotations...")
    K_list = load_json(k_path)
    mano_list = load_json(mano_path)
    xyz_list = load_json(xyz_path)

    n_total = len(K_list)
    assert len(mano_list) == n_total and len(xyz_list) == n_total
    print(f"  {n_total} evaluation samples")

    if args.max_images > 0:
        n_total = min(n_total, args.max_images)
        print(f"  Limiting to {n_total} images")

    sps = args.samples_per_shard
    num_shards = (n_total + sps - 1) // sps
    print(f"  {num_shards} shards ({sps} samples/shard)")

    os.makedirs(args.dst, exist_ok=True)

    total_written = 0

    for shard_idx in range(num_shards):
        start = shard_idx * sps
        end = min(start + sps, n_total)
        this_shard_size = end - start

        tar_name = f"{shard_idx:06d}.tar"
        tar_path = os.path.join(args.dst, tar_name)

        with tarfile.open(tar_path, "w") as tw:
            for img_idx in range(start, end):
                sample_id = f"{img_idx:08d}"

                img_path = os.path.join(
                    args.src, "evaluation", "rgb", f"{img_idx:08d}.jpg")
                if not os.path.exists(img_path):
                    print(f"  WARNING: image not found: {img_path}")
                    continue

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                img = cv2.imdecode(
                    np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                ann = build_annotation(
                    K_list[img_idx], mano_list[img_idx],
                    xyz_list[img_idx], img.shape[0], img.shape[1])

                add_to_tar(tw, sample_id, img_bytes, ann)
                total_written += 1

        print(f"  shard {shard_idx}: {tar_name} ({this_shard_size} samples)")

    print(f"\nDone. Wrote {total_written} samples across {num_shards} shards "
          f"to {args.dst}/")


if __name__ == "__main__":
    main()
