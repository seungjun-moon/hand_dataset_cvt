#!/usr/bin/env python3
"""Convert egodex datasets from ARKit coordinates (+Y up) to +Z up coordinates.

ARKit coordinate system: +X right, +Y up, -Z forward
Target coordinate system: +X right, +Y forward, +Z up

Source structure (RAW):
    RAW/egodex/
        {part}/
            {task_name}/
                {N}.hdf5
                {N}.mp4

Output structure (CONVERTED):
    CONVERTED/egodex/
        {task_name}/
            {seq_idx:06d}_label_00.hdf5
            {seq_idx:06d}_video_00.mp4

Sequences are clustered by task name, matching the DexYCB output convention.

Usage:
    python scripts/convert_egodex.py --src RAW/egodex --dst CONVERTED/egodex
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import write_egodex_hdf5

# +90 deg rotation around X-axis: ARKit (Y-up) -> Z-up
T_WORLD = np.array([
    [0,  0,  -1,  0],
    [-1,  0, 0,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  1],
], dtype=np.float32)


def convert_hdf5(src_path: str, dst_path: str):
    """Convert a single egodex HDF5 from ARKit coords to +Z up."""
    with h5py.File(src_path, "r") as f_in:
        intrinsic = f_in["camera/intrinsic"][:]

        # Read all world transforms (ARKit coords), skip gravity
        transform_names = [name for name in f_in["transforms"] if name != "gravity"]
        transforms_arkit = {}
        for name in transform_names:
            transforms_arkit[name] = f_in[f"transforms/{name}"][:]

        # Read confidences
        confidences = {}
        for name in f_in["confidences"]:
            confidences[name] = f_in[f"confidences/{name}"][:]

    # Convert world transforms to Z-up: T_world @ T_arkit
    transforms_zup = {}
    for name, tf in transforms_arkit.items():
        transforms_zup[name] = T_WORLD @ tf  # (N,4,4): broadcast left-multiply

    write_egodex_hdf5(dst_path, intrinsic, transforms_zup, confidences)


def _numeric_sort_key(path: str):
    """Sort key that handles non-zero-padded numeric filenames (0, 1, 2, ..., 10, ...)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def collect_sequences(src_dir: str):
    """Collect all (hdf5_path, mp4_path, task_name) tuples across all parts.

    Groups by task_name. Within each task, sequences are sorted numerically
    to handle non-zero-padded filenames (0, 1, 2, ..., 10, ...).

    Returns dict {task_name: [(hdf5_path, mp4_path_or_None), ...]}.
    """
    clusters = {}  # task_name -> [(hdf5_path, mp4_path_or_None)]

    subdirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ])

    for subdir in subdirs:
        subdir_path = os.path.join(src_dir, subdir)
        # Check if this is a part directory (contains task subdirs)
        # or a task directory (contains .hdf5 files directly)
        hdf5_in_subdir = glob.glob(os.path.join(subdir_path, "*.hdf5"))
        if hdf5_in_subdir:
            # Single-level: src_dir/{task}/{N}.hdf5
            _collect_task(subdir_path, subdir, clusters)
        else:
            # Two-level: src_dir/{part}/{task}/{N}.hdf5
            task_dirs = sorted([
                d for d in os.listdir(subdir_path)
                if os.path.isdir(os.path.join(subdir_path, d))
            ])
            for task_name in task_dirs:
                task_path = os.path.join(subdir_path, task_name)
                _collect_task(task_path, task_name, clusters)

    return clusters


def _collect_task(task_path: str, task_name: str, clusters: dict):
    """Collect hdf5/mp4 pairs from a single task directory, sorted numerically."""
    hdf5_files = sorted(
        glob.glob(os.path.join(task_path, "*.hdf5")),
        key=_numeric_sort_key,
    )
    for hdf5_path in hdf5_files:
        stem = os.path.splitext(os.path.basename(hdf5_path))[0]
        mp4_path = os.path.join(task_path, f"{stem}.mp4")
        if not os.path.exists(mp4_path):
            mp4_path = None
        clusters.setdefault(task_name, []).append((hdf5_path, mp4_path))


def convert_egodex(src_dir: str, dst_dir: str, max_samples: int = 0):
    """Convert all egodex sequences from ARKit coords to +Z up."""
    clusters = collect_sequences(src_dir)

    if not clusters:
        print(f"No HDF5 files found in {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    global_count = 0
    for task_name in sorted(clusters.keys()):
        sequences = clusters[task_name]
        task_dir = os.path.join(dst_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        for seq_idx, (hdf5_src, mp4_src) in enumerate(sequences):
            if max_samples > 0 and global_count >= max_samples:
                break

            prefix = f"{seq_idx:06d}"

            # HDF5 label
            hdf5_dst = os.path.join(task_dir, f"{prefix}_label_00.hdf5")
            convert_hdf5(hdf5_src, hdf5_dst)

            # RGB video (re-encode to h264)
            if mp4_src is not None:
                mp4_dst = os.path.join(task_dir, f"{prefix}_video_00.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp4_src,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                     mp4_dst],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=True,
                )

            print(f"[{task_name}/{prefix}] <- {os.path.basename(hdf5_src)}")
            global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    print(f"\nDone. Converted {global_count} sequences -> {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert egodex datasets from ARKit coords to +Z up"
    )
    parser.add_argument("--src", default="RAW/egodex",
                        help="Source egodex directory (default: RAW/egodex)")
    parser.add_argument("--dst", default="CONVERTED/egodex",
                        help="Output directory (default: CONVERTED/egodex)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences to convert (0=all)")
    args = parser.parse_args()

    convert_egodex(args.src, args.dst, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
