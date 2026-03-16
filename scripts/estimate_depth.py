#!/usr/bin/env python3
"""Estimate depth maps for CONVERTED dataset videos using MoGe.

Reads CONVERTED/ layout, estimates metric depth for each video frame using
MoGe-2, and saves as a lossless depth video alongside the RGB video.

Source/output structure:
    CONVERTED/{dataset}/{cluster}/
        {seq_idx:06d}_video_{cam_idx:02d}.mp4      (input RGB)
        {seq_idx:06d}_label_{cam_idx:02d}.hdf5      (input labels, for FOV)
        {seq_idx:06d}_depth_{cam_idx:02d}.mp4       (output depth)

Depth encoding: uint16 millimeters packed into BGR24 (B=high byte, G=low byte,
R=0), encoded with libx264rgb CRF 0 (lossless).

To decode:
    depth_mm = B.astype(uint16) * 256 + G.astype(uint16)
    depth_m  = depth_mm / 1000.0

Usage:
    python scripts/estimate_depth.py --src CONVERTED/dex_ycb
    python scripts/estimate_depth.py --src CONVERTED --datasets dex_ycb egodex
    python scripts/estimate_depth.py --src CONVERTED/dex_ycb --gpu 0
    python scripts/estimate_depth.py --src CONVERTED/dex_ycb --resolution-level 7
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.depth_utils import (
    estimate_depth_for_video,
    load_moge_model,
    save_depth_video,
)
from utils.file_utils import discover_files, get_video_dimensions


def estimate_depth_for_cluster(cluster_dir: str, depth_model,
                               resolution_level: int = 9,
                               fps: float = 30.0):
    """Estimate depth for all videos in a cluster directory.

    Skips videos that already have a corresponding depth file.

    Returns number of depth videos created.
    """
    files = discover_files(cluster_dir)
    created = 0

    for seq_idx, cam_idx, hdf5_path, video_path in files:
        depth_path = os.path.join(
            cluster_dir, f"{seq_idx}_depth_{cam_idx}.mp4")

        if os.path.exists(depth_path):
            continue

        # Compute FOV from intrinsics
        with h5py.File(hdf5_path, "r") as f:
            intrinsic = f["camera/intrinsic"][:]
        img_w, _ = get_video_dimensions(video_path)
        fov_x = float(2 * np.arctan(img_w / (2 * intrinsic[0, 0]))
                       * 180 / np.pi)

        print(f"  Estimating depth: {seq_idx}_video_{cam_idx}.mp4"
              f" (fov={fov_x:.1f})")
        depths = estimate_depth_for_video(
            depth_model, video_path, fov_x=fov_x,
            resolution_level=resolution_level)

        save_depth_video(depths, depth_path, fps=fps)
        size_mb = os.path.getsize(depth_path) / (1024 * 1024)
        print(f"  Saved: {os.path.basename(depth_path)}"
              f" ({len(depths)} frames, {size_mb:.1f} MB)")
        created += 1

    return created


def estimate_depth_for_dir(src_dir: str, depth_model,
                           resolution_level: int = 9,
                           fps: float = 30.0):
    """Recursively find cluster directories and estimate depth.

    A cluster directory is one that contains {seq_idx}_label_{cam_idx}.hdf5
    files. Walks the directory tree to find all such directories.

    Returns total number of depth videos created.
    """
    total = 0
    pattern = re.compile(r"^\d{6}_label_\d{2}\.hdf5$")

    for root, dirs, filenames in os.walk(src_dir):
        has_hdf5 = any(pattern.match(f) for f in filenames)
        if has_hdf5:
            rel = os.path.relpath(root, src_dir)
            print(f"Processing: {rel}")
            n = estimate_depth_for_cluster(
                root, depth_model,
                resolution_level=resolution_level, fps=fps)
            if n > 0:
                print(f"  Created {n} depth videos")
            total += n

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Estimate depth maps for CONVERTED dataset videos"
    )
    parser.add_argument("--src", default="CONVERTED",
                        help="Source directory (default: CONVERTED)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset subdirs to process (default: all under src)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output depth video FPS (default: 30)")
    parser.add_argument("--resolution-level", type=int, default=9,
                        help="MoGe resolution level 0-9 (default: 9)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Loading MoGe depth model...")
    depth_model = load_moge_model()
    print("MoGe model loaded.")

    if args.datasets is not None:
        dirs = [os.path.join(args.src, d) for d in args.datasets]
    else:
        dirs = [args.src]

    grand_total = 0
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Skipping {d}: not found")
            continue
        grand_total += estimate_depth_for_dir(
            d, depth_model,
            resolution_level=args.resolution_level, fps=args.fps)

    print(f"\nDone. Created {grand_total} depth videos.")


if __name__ == "__main__":
    main()
