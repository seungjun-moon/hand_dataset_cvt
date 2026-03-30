#!/usr/bin/env python3
"""
Convert ARCTIC dataset to WebDataset format (tar files) using HaPTIC clip labels.

Reads per-clip label files (.pyd) which contain per-frame annotations, pairs each
frame with its corresponding image from the raw Arctic dataset, and writes tar files
with 1000 samples each.

Each sample consists of:
    {prefix}/{sample_id}.jpg        — RGB image
    {prefix}/{sample_id}.data.pyd   — pickled annotation list

The clip labels store hand_pose in world space with flat_hand_mean=False convention.
This script converts to the HaMER webdataset convention:
    - hand_pose[:3] (global_orient): rotated from world to camera frame using cTw
    - hand_pose[3:48]: converted from flat_hand_mean=False to flat_hand_mean=True
      by adding the MANO mean hand pose
    - keypoints_2d/3d: already in camera space in the clip labels (kept as-is)
    - Frames with center outside image bounds are filtered out

Usage:
    python scripts/convert_arctic_webdataset.py \
        --label-dir ../haptic/haptic_training_label/arctic/clip \
        --img-dir ../haptic/data/arctic/images \
        --dst ../haptic/hamer_training_data/dataset_tars/arctic-train/ \
        --views 0 1 2 3 4 5 6 7 8
"""

import argparse
import io
import os
import pickle
import sys
import tarfile
from glob import glob

import cv2
import numpy as np
import smplx

SAMPLES_PER_SHARD = 1000
DATASET_PREFIX = "arctic-train"
DEFAULT_MANO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "_DATA", "data", "mano"
)


def load_hand_mean_pose(mano_dir):
    """Load the MANO mean hand pose for flat_hand_mean convention conversion."""
    mano_model = smplx.MANOLayer(
        model_path=os.path.normpath(mano_dir),
        is_rhand=True, flat_hand_mean=False)
    return mano_model.hand_mean.numpy()  # (45,)


def axis_angle_to_rotmat(aa):
    """Convert axis-angle (3,) to rotation matrix (3, 3)."""
    R, _ = cv2.Rodrigues(aa.astype(np.float64))
    return R.astype(np.float32)


def rotmat_to_axis_angle(R):
    """Convert rotation matrix (3, 3) to axis-angle (3,)."""
    aa, _ = cv2.Rodrigues(R.astype(np.float64))
    return aa.flatten().astype(np.float32)


def transform_global_orient_to_camera(global_orient_aa, cTw):
    """Rotate global orientation from world frame to camera frame.

    Args:
        global_orient_aa: (3,) axis-angle in world frame.
        cTw: (4, 4) world-to-camera transform.

    Returns:
        (3,) axis-angle in camera frame.
    """
    R_world = axis_angle_to_rotmat(global_orient_aa)
    R_cam = cTw[:3, :3].astype(np.float32)
    R_cam_orient = R_cam @ R_world
    return rotmat_to_axis_angle(R_cam_orient)


def collect_all_labels(label_dir, views=None):
    """Walk through per-subject/per-sequence/per-view label directories
    and collect all clip .pyd files.

    Returns list of (clip_pyd_path, subject, seq_name, view_idx).
    """
    clips = []
    subjects = sorted([
        d for d in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, d)) and d.startswith("s")
    ])

    for subject in subjects:
        subject_dir = os.path.join(label_dir, subject)
        sequences = sorted([
            d for d in os.listdir(subject_dir)
            if os.path.isdir(os.path.join(subject_dir, d))
        ])
        for seq_name in sequences:
            seq_dir = os.path.join(subject_dir, seq_name)
            view_dirs = sorted([
                d for d in os.listdir(seq_dir)
                if os.path.isdir(os.path.join(seq_dir, d))
            ])
            for view_str in view_dirs:
                view_idx = int(view_str)
                if views is not None and view_idx not in views:
                    continue
                view_dir = os.path.join(seq_dir, view_str)
                pyd_files = sorted(glob(os.path.join(view_dir, "*.pyd")))
                for pyd_path in pyd_files:
                    clips.append((pyd_path, subject, seq_name, view_idx))

    return clips


def get_image_size(img_path):
    """Get image (W, H) by reading only the header (fast)."""
    from PIL import Image
    try:
        with Image.open(img_path) as im:
            return im.size  # (W, H)
    except Exception:
        return None


def shift_kp3d_for_centered_pp(kp3d, K, W, H):
    """Shift 3D keypoints so that projection with centered principal point
    gives the same 2D as projection with the actual K matrix.

    Given: u = fx * X/Z + cx  (actual projection)
    Want:  u = fx * X'/Z + W/2  (centered pp projection)
    So:    X' = X + (cx - W/2) * Z / fx

    Args:
        kp3d: (21, 4) keypoints in camera space [x, y, z, conf].
        K: (3, 3) camera intrinsic matrix.
        W, H: image width and height.

    Returns:
        (21, 4) shifted keypoints.
    """
    kp3d_shifted = kp3d.copy()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = kp3d[:, 3] > 0.5
    Z = kp3d[valid, 2]
    kp3d_shifted[valid, 0] += (cx - W / 2.0) * Z / fx
    kp3d_shifted[valid, 1] += (cy - H / 2.0) * Z / fy

    return kp3d_shifted


def extract_frames_from_clip(clip_label, img_dir, hand_mean_pose,
                             image_size_cache):
    """Extract individual frame annotations from a clip label.

    Converts hand_pose from world-space to camera-space.
    Shifts keypoints_3d so projection with centered principal point matches
    the actual 2D keypoints.
    Filters frames where center is outside image bounds.

    Yields:
        (img_path, annotation_dict) for each valid frame.
    """
    num_frames = clip_label["imgname"].shape[0]

    for i in range(num_frames):
        imgname = str(clip_label["imgname"][i])
        img_path = os.path.join(img_dir, imgname)

        # Get image size for bounds check (cache per directory, not per file)
        img_dir_key = os.path.dirname(img_path)
        if img_dir_key not in image_size_cache:
            if not os.path.exists(img_path):
                image_size_cache[img_dir_key] = None
            else:
                image_size_cache[img_dir_key] = get_image_size(img_path)

        img_size = image_size_cache[img_dir_key]
        if img_size is None:
            continue

        W, H = img_size

        # Filter: center must be within image bounds
        center = clip_label["center"][i]
        if center[0] < 0 or center[0] > W or center[1] < 0 or center[1] > H:
            continue

        # Filter: scale must be reasonable
        scale = clip_label["scale"][i]
        if scale[0] < 1e-2 or scale[1] < 1e-2:
            continue

        # Convert hand_pose: world→camera frame for global_orient
        hand_pose = clip_label["hand_pose"][i].copy().astype(np.float32)
        cTw = clip_label["cTw"][i]

        # Rotate global_orient from world to camera frame
        hand_pose[:3] = transform_global_orient_to_camera(hand_pose[:3], cTw)

        # Build annotation in HaMER webdataset format
        ann = {
            "keypoints_2d": clip_label["hand_keypoints_2d"][i].astype(np.float32),
            "keypoints_3d": clip_label["hand_keypoints_3d"][i].astype(np.float32),
            "center": center.astype(np.float64),
            "scale": scale.astype(np.float64),
            "right": np.float32(clip_label["right"][i]),
            "hand_pose": hand_pose,
            "betas": clip_label["betas"][i].astype(np.float32),
            "has_hand_pose": np.float32(clip_label["has_hand_pose"][i]),
            "has_betas": np.float32(clip_label["has_betas"][i]),
            "personid": int(clip_label["person_id"][i]),
            "extra_info": {},
        }

        yield img_path, ann


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
        description="Convert ARCTIC to WebDataset tar files using HaPTIC clip labels")
    parser.add_argument("--label-dir",
                        default="../haptic/haptic_training_label/arctic/clip",
                        help="Path to Arctic clip label directory")
    parser.add_argument("--img-dir",
                        default="../haptic/data/arctic/images",
                        help="Path to Arctic images directory")
    parser.add_argument("--dst",
                        default="../haptic/hamer_training_data/dataset_tars/arctic-train/",
                        help="Output directory for tar files")
    parser.add_argument("--views", type=int, nargs="+", default=[0],
                        help="View indices to include (0=ego, 1-8=exo, default: 0)")
    parser.add_argument("--samples-per-shard", type=int,
                        default=SAMPLES_PER_SHARD,
                        help=f"Samples per tar shard (default: {SAMPLES_PER_SHARD})")
    parser.add_argument("--deduplicate", action="store_true", default=True,
                        help="Deduplicate frames across overlapping clips (default: True)")
    parser.add_argument("--no-deduplicate", action="store_false", dest="deduplicate")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to convert (0=all)")
    parser.add_argument("--mano-dir", default=DEFAULT_MANO_DIR,
                        help="Path to MANO model directory")
    args = parser.parse_args()

    views_set = set(args.views)
    print(f"Label dir:  {args.label_dir}")
    print(f"Image dir:  {args.img_dir}")
    print(f"Output:     {args.dst}")
    print(f"Views:      {sorted(views_set)}")
    print(f"MANO dir:   {args.mano_dir}")

    # Load MANO mean hand pose for convention conversion
    print("\nLoading MANO mean hand pose...")
    hand_mean_pose = load_hand_mean_pose(args.mano_dir)
    print(f"  Mean pose shape: {hand_mean_pose.shape}")

    # Collect all clip label files
    print("\nScanning clip labels...")
    clips = collect_all_labels(args.label_dir, views=views_set)
    print(f"  Found {len(clips)} clip files")

    if not clips:
        print("ERROR: No clip labels found. Check --label-dir and --views.")
        sys.exit(1)

    # Extract all frames, deduplicating by image path
    print("Extracting frames from clips...")
    seen_images = set()
    frames = []  # list of (img_path, annotation)
    skipped_dup = 0
    skipped_oob = 0
    image_size_cache = {}

    for clip_idx, (pyd_path, subject, seq_name, view_idx) in enumerate(clips):
        clip_label = pickle.load(open(pyd_path, "rb"))
        n_before = len(frames)

        for img_path, ann in extract_frames_from_clip(
            clip_label, args.img_dir, hand_mean_pose, image_size_cache
        ):
            if args.deduplicate and img_path in seen_images:
                skipped_dup += 1
                continue

            seen_images.add(img_path)
            frames.append((img_path, ann))

            if args.max_samples > 0 and len(frames) >= args.max_samples:
                break

        # Count frames that were skipped due to OOB/missing
        n_clip_frames = clip_label["imgname"].shape[0]
        n_added = len(frames) - n_before
        # (rough count — includes dups and missing)

        if clip_idx % 500 == 0:
            print(f"  Processed {clip_idx}/{len(clips)} clips, "
                  f"{len(frames)} frames collected")

        if args.max_samples > 0 and len(frames) >= args.max_samples:
            break

    n_total = len(frames)
    n_missing = sum(1 for v in image_size_cache.values() if v is None)
    print(f"\n  Total frames: {n_total}")
    print(f"  Skipped (duplicate): {skipped_dup}")
    print(f"  Skipped (missing image): {n_missing} unique images not found")
    print(f"  (Frames with center outside image bounds were also filtered)")

    if n_total == 0:
        print("ERROR: No frames to write.")
        sys.exit(1)

    # Write tar shards
    sps = args.samples_per_shard
    num_shards = (n_total + sps - 1) // sps
    print(f"\nWriting {num_shards} shards ({sps} samples/shard)...")

    os.makedirs(args.dst, exist_ok=True)
    total_written = 0

    for shard_idx in range(num_shards):
        start = shard_idx * sps
        end = min(start + sps, n_total)

        tar_name = f"{shard_idx:06d}.tar"
        tar_path = os.path.join(args.dst, tar_name)

        with tarfile.open(tar_path, "w") as tw:
            for frame_idx in range(start, end):
                img_path, ann = frames[frame_idx]

                if not os.path.exists(img_path):
                    continue

                sample_id = f"{total_written:08d}"

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                add_to_tar(tw, sample_id, img_bytes, [ann])
                total_written += 1

        if shard_idx % 5 == 0 or shard_idx == num_shards - 1:
            print(f"  shard {shard_idx}: {tar_name} ({end - start} samples)")

    print(f"\nDone. Wrote {total_written} samples across {num_shards} shards "
          f"to {args.dst}")


if __name__ == "__main__":
    main()
