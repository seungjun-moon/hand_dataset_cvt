#!/usr/bin/env python3
"""Visualize randomly sampled frames with projected 3D keypoints.

Reads CONVERTED HDF5 files and source videos, projects world-space 3D
keypoints to 2D, and draws skeleton overlays on images. No MANO required.

Usage:
    python scripts/visualize.py --src CONVERTED/dex_ycb --n 100 --out outputs
    python scripts/visualize.py --src CONVERTED/ho_cap --n 10 --out outputs --seed 0
"""

import argparse
import os
import random
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.camera_utils import HAND_JOINT_SUFFIXES, get_active_sides
from utils.image_utils import project_3d_to_2d

# Skeleton: MANO parent indices for 21 joints
PARENTS = [
    -1,  # 0: wrist
    0, 1, 2, 3,      # thumb
    0, 5, 6, 7,      # index
    0, 9, 10, 11,    # middle
    0, 13, 14, 15,   # ring
    0, 17, 18, 19,   # little
]

# Per-finger colors (BGR)
FINGER_COLORS = [
    (0, 255, 255),   # thumb - yellow
    (0, 0, 255),     # index - red
    (255, 0, 0),     # middle - blue
    (255, 0, 255),   # ring - magenta
    (0, 165, 255),   # little - orange
]

WRIST_COLOR = (0, 255, 0)  # green

# Side indicator colors
SIDE_COLORS = {"right": (0, 200, 0), "left": (200, 200, 0)}


def collect_samples(src_dir: str):
    """Collect all (hdf5_path, video_path, frame_count, active_sides) tuples."""
    samples = []
    for cluster in sorted(os.listdir(src_dir)):
        cluster_dir = os.path.join(src_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        for fname in sorted(os.listdir(cluster_dir)):
            if not fname.endswith(".hdf5"):
                continue
            hdf5_path = os.path.join(cluster_dir, fname)
            video_name = fname.replace("_label_", "_video_").replace(".hdf5", ".mp4")
            video_path = os.path.join(cluster_dir, video_name)
            if not os.path.exists(video_path):
                continue
            sides = get_active_sides(hdf5_path)
            if not sides:
                continue
            with h5py.File(hdf5_path, "r") as f:
                sample_key = f"{sides[0]}Hand"
                n_frames = f[f"transforms/{sample_key}"].shape[0]
            samples.append((hdf5_path, video_path, n_frames, sides))
    return samples


def read_video_frame(video_path: str, frame_idx: int):
    """Read a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _joint_color(joint_idx: int):
    """Return BGR color for a joint index."""
    if joint_idx == 0:
        return WRIST_COLOR
    return FINGER_COLORS[(joint_idx - 1) // 4]


def render_frame(hdf5_path: str, video_path: str, frame_idx: int, sides: list):
    """Render a single frame with projected 3D keypoint skeleton.

    Returns BGR image or None on failure.
    """
    img = read_video_frame(video_path, frame_idx)
    if img is None:
        return None

    with h5py.File(hdf5_path, "r") as f:
        intrinsic = f["camera/intrinsic"][:]
        cam_pose = f["transforms/camera"][frame_idx]

        for side in sides:
            # Load 21 world-space joint positions from transforms
            kp_world = np.zeros((21, 3), dtype=np.float32)
            for j, suffix in enumerate(HAND_JOINT_SUFFIXES):
                name = f"{side}{suffix}"
                kp_world[j] = f[f"transforms/{name}"][frame_idx, :3, 3]

            # Project to 2D
            kp_2d = project_3d_to_2d(kp_world, cam_pose, intrinsic)

            # Check if keypoints are within image bounds (at least partially)
            h, w = img.shape[:2]
            in_bounds = np.any(
                (kp_2d[:, 0] >= -w) & (kp_2d[:, 0] < 2 * w) &
                (kp_2d[:, 1] >= -h) & (kp_2d[:, 1] < 2 * h)
            )
            if not in_bounds:
                continue

            # Draw skeleton connections
            for i in range(1, 21):
                p = PARENTS[i]
                pt1 = (int(kp_2d[i, 0]), int(kp_2d[i, 1]))
                pt2 = (int(kp_2d[p, 0]), int(kp_2d[p, 1]))
                color = _joint_color(i)
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

            # Draw keypoints
            for i, (x, y) in enumerate(kp_2d):
                color = _joint_color(i)
                cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)

            # Label the hand side
            wrist_2d = kp_2d[0]
            label_pos = (int(wrist_2d[0]) - 10, int(wrist_2d[1]) - 15)
            cv2.putText(img, side[0].upper(), label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        SIDE_COLORS[side], 2, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize frames with projected 3D keypoints")
    parser.add_argument("--src", default="CONVERTED/dex_ycb",
                        help="Converted dataset directory")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of frames to sample")
    parser.add_argument("--out", default="outputs",
                        help="Output directory for saved images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Collecting samples...")
    samples = collect_samples(args.src)
    print(f"Found {len(samples)} sequences with active hands")

    if not samples:
        print("No samples found!")
        return

    # Sample random (sequence, frame) pairs
    picks = []
    for _ in range(args.n):
        seq = random.choice(samples)
        frame_idx = random.randint(0, seq[2] - 1)
        picks.append((seq[0], seq[1], frame_idx, seq[3]))

    os.makedirs(args.out, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.src))

    # Render and save individual images
    print(f"Rendering {args.n} frames...")
    saved = 0
    for i, (hdf5_path, video_path, frame_idx, sides) in enumerate(picks):
        img = render_frame(hdf5_path, video_path, frame_idx, sides)
        if img is not None:
            # Build descriptive filename
            cluster = os.path.basename(os.path.dirname(hdf5_path))
            seq_name = os.path.basename(hdf5_path).replace(".hdf5", "")
            out_path = os.path.join(args.out, f"{dataset_name}_{cluster}_{seq_name}_f{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, img)
            saved += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.n}")

    print(f"Saved {saved} images to {args.out}/")


if __name__ == "__main__":
    main()
