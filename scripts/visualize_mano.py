#!/usr/bin/env python3
"""Visualize randomly sampled frames with 2D keypoints and MANO mesh overlay.

Reads CONVERTED HDF5 files (with MANO params) and source videos.
Projects 3D keypoints to 2D and renders MANO mesh wireframe on images.

Usage:
    python scripts/visualize_mano.py --src CONVERTED/dex_ycb --n 100
    python scripts/visualize_mano.py --src CONVERTED/dex_ycb --n 100 --out vis_mano.png
"""

import argparse
import math
import os
import random
import sys

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manopth.manolayer import ManoLayer
from utils.camera_utils import HAND_JOINT_SUFFIXES
from utils.image_utils import project_3d_to_2d

MANO_ROOT = "/rlwrld3/home/seungjun/HO-Cap-Annotation/config/mano_models"


def get_mano_layer(side: str):
    """Create ManoLayer for given side (axis-angle, no PCA)."""
    return ManoLayer(
        mano_root=MANO_ROOT, side=side,
        use_pca=False, flat_hand_mean=False,
    )


def collect_samples(src_dir: str):
    """Collect all (hdf5_path, video_path, frame_count) tuples from CONVERTED."""
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
            with h5py.File(hdf5_path, "r") as f:
                has_mano = "mano_right" in f or "mano_left" in f
                if not has_mano:
                    continue
                mano_key = "mano_right" if "mano_right" in f else "mano_left"
                # Require latest format with all keys
                if "transl_camspace" not in f[mano_key]:
                    continue
                n_frames = f[f"{mano_key}/hand_pose"].shape[0]
            samples.append((hdf5_path, video_path, n_frames))
    return samples


def read_video_frame(video_path: str, frame_idx: int):
    """Read a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def render_frame(hdf5_path: str, video_path: str, frame_idx: int,
                 get_mano_fn):
    """Render a single frame with keypoints and MANO mesh overlay.

    Returns BGR image or None on failure.
    """
    img = read_video_frame(video_path, frame_idx)
    if img is None:
        return None

    with h5py.File(hdf5_path, "r") as f:
        intrinsic = f["camera/intrinsic"][:]
        cam_pose = f["transforms/camera"][frame_idx]
        # Find which mano side exists
        if "mano_right" in f:
            mano_side = "right"
        elif "mano_left" in f:
            mano_side = "left"
        else:
            return None
        mano_key = f"mano_{mano_side}"
        betas = f[f"{mano_key}/betas"][:]
        global_orient_w = f[f"{mano_key}/global_orient_worldspace"][frame_idx]
        hand_pose_rotmats = f[f"{mano_key}/hand_pose"][frame_idx]  # (15,3,3)
        transl = f[f"{mano_key}/transl_camspace"][frame_idx]  # (3,)

        # Load keypoints from stored transforms (not MANO)
        kp_world = np.zeros((21, 3), dtype=np.float32)
        for j, suffix in enumerate(HAND_JOINT_SUFFIXES):
            name = f"{mano_side}{suffix}"
            kp_world[j] = f[f"transforms/{name}"][frame_idx, :3, 3]

    # Project stored keypoints to 2D
    kp_2d = project_3d_to_2d(kp_world, cam_pose, intrinsic)

    # Convert stored rotmats back to axis-angle for MANO forward pass
    R_cam = cam_pose[:3, :3]
    global_orient_cam = R_cam.T @ global_orient_w  # world -> camera
    global_aa, _ = cv2.Rodrigues(global_orient_cam.astype(np.float64))
    global_aa = global_aa.flatten()

    # Hand pose rotmats -> axis-angle, subtract hands_mean
    mano_layer = get_mano_fn(mano_side)
    hands_mean = mano_layer.th_hands_mean.numpy().squeeze()  # (45,)
    hand_aa = np.zeros(45, dtype=np.float64)
    for j in range(15):
        aa, _ = cv2.Rodrigues(hand_pose_rotmats[j].astype(np.float64))
        hand_aa[j * 3:(j + 1) * 3] = aa.flatten()
    # use_pca=False adds hands_mean internally, so subtract it
    hand_aa_input = (hand_aa - hands_mean).astype(np.float32)

    pose_48 = np.concatenate([global_aa.astype(np.float32), hand_aa_input])
    verts, _ = mano_layer(
        torch.tensor(pose_48).unsqueeze(0).float(),
        torch.tensor(betas).unsqueeze(0).float(),
        torch.tensor(transl).unsqueeze(0).float(),
    )
    verts_cam = verts[0].detach().numpy() / 1000.0  # (778, 3)

    # Project vertices to 2D
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    verts_world = (verts_cam @ R.T) + t
    verts_2d = project_3d_to_2d(verts_world, cam_pose, intrinsic)

    # Draw mesh wireframe
    faces = mano_layer.th_faces.numpy()
    overlay = img.copy()
    for face in faces:
        pts = verts_2d[face].astype(np.int32)
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], True,
                      (200, 200, 200), 1, cv2.LINE_AA)

    # Blend mesh overlay
    img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)

    # Draw keypoints
    colors = [(0, 255, 0)] * 21  # green for all joints
    # Thumb=1-4, Index=5-8, Middle=9-12, Ring=13-16, Little=17-20
    finger_colors = [
        (0, 255, 255),   # thumb - yellow
        (0, 0, 255),     # index - red
        (255, 0, 0),     # middle - blue
        (255, 0, 255),   # ring - magenta
        (0, 165, 255),   # little - orange
    ]
    for i, (x, y) in enumerate(kp_2d):
        if i == 0:
            color = (0, 255, 0)  # wrist green
        else:
            finger_idx = (i - 1) // 4
            color = finger_colors[finger_idx]
        cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

    # Draw skeleton connections
    parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    for i in range(1, 21):
        p = parents[i]
        pt1 = (int(kp_2d[i, 0]), int(kp_2d[i, 1]))
        pt2 = (int(kp_2d[p, 0]), int(kp_2d[p, 1]))
        finger_idx = (i - 1) // 4
        color = finger_colors[finger_idx]
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize frames with keypoints and MANO mesh overlay")
    parser.add_argument("--src", default="CONVERTED/dex_ycb",
                        help="Converted dataset directory")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of frames to sample")
    parser.add_argument("--out", default="vis_mano.png",
                        help="Output image path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Collecting samples...")
    samples = collect_samples(args.src)
    print(f"Found {len(samples)} sequences with MANO data")

    if not samples:
        print("No samples found!")
        return

    # Sample random (sequence, frame) pairs
    picks = []
    for _ in range(args.n):
        seq = random.choice(samples)
        frame_idx = random.randint(0, seq[2] - 1)
        picks.append((seq[0], seq[1], frame_idx))

    # Load MANO layers
    print("Loading MANO models...")
    mano_layers = {}

    def get_or_load_mano(side):
        if side not in mano_layers:
            mano_layers[side] = get_mano_layer(side)
        return mano_layers[side]

    # Render frames
    print(f"Rendering {args.n} frames...")
    rendered = []
    for i, (hdf5_path, video_path, frame_idx) in enumerate(picks):
        img = render_frame(hdf5_path, video_path, frame_idx, get_or_load_mano)
        if img is not None:
            # Resize to uniform size for grid
            img = cv2.resize(img, (480, 360))
            rendered.append(img)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.n}")

    print(f"Rendered {len(rendered)} frames")

    if not rendered:
        print("No frames rendered!")
        return

    # Create grid
    n = len(rendered)
    cols = min(10, n)
    rows = math.ceil(n / cols)

    h, w = rendered[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(rendered):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    cv2.imwrite(args.out, grid)
    print(f"Saved visualization to {args.out} ({cols}x{rows} grid)")


if __name__ == "__main__":
    main()
