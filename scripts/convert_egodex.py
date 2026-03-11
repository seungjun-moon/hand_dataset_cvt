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
        {idx:06d}_{task_name}/
            0.hdf5  (transforms/ in Z-up, transforms_cam/, confidences/, camera/)
            0.mp4

Usage:
    python scripts/convert_egodex.py --src RAW/egodex --dst CONVERTED/egodex
"""

import argparse
import glob
import os
import re
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.transforms import invert_rigid

# +90 deg rotation around X-axis: ARKit (Y-up) -> Z-up
T_WORLD = np.array([
    [0,  0,  -1,  0],
    [-1,  0, 0,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  1],
], dtype=np.float32)


def convert_hdf5(src_path: str, dst_path: str):
    """Convert a single egodex HDF5 from ARKit coords to +Z up and add transforms_cam."""
    with h5py.File(src_path, "r") as f_in:
        intrinsic = f_in["camera/intrinsic"][:]

        # Read all world transforms (ARKit coords)
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

    # Compute camera-space transforms: inv(cam_c2w) @ joint_world
    # Note: T_world cancels out, so this is just inv(cam_arkit) @ joint_arkit
    cam_arkit = transforms_arkit["camera"]
    transforms_cam = {}
    N = cam_arkit.shape[0]
    for name, tf in transforms_arkit.items():
        if name == "camera":
            continue
        tc = np.zeros((N, 4, 4), dtype=np.float32)
        for i in range(N):
            tc[i] = invert_rigid(cam_arkit[i]) @ tf[i]
        transforms_cam[name] = tc

    # Write output
    with h5py.File(dst_path, "w") as f_out:
        cam_grp = f_out.create_group("camera")
        cam_grp.create_dataset("intrinsic", data=intrinsic)

        tf_grp = f_out.create_group("transforms")
        for name, data in transforms_zup.items():
            tf_grp.create_dataset(name, data=data.astype(np.float32))

        tf_cam_grp = f_out.create_group("transforms_cam")
        for name, data in transforms_cam.items():
            tf_cam_grp.create_dataset(name, data=data.astype(np.float32))

        conf_grp = f_out.create_group("confidences")
        for name, data in confidences.items():
            conf_grp.create_dataset(name, data=data.astype(np.float32))


def collect_hdf5_pairs(src_dir: str):
    """Collect all (hdf5_path, mp4_path, task_name) tuples across all parts.

    Scans src_dir for part directories containing task subdirectories,
    each with numbered .hdf5/.mp4 pairs.

    Returns sorted list of (hdf5_path, mp4_path_or_None, task_name).
    """
    pairs = []

    # Detect structure: either src_dir/{part}/{task}/{N}.hdf5
    # or src_dir/{task}/{N}.hdf5 (single-level)
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
            _collect_task_pairs(subdir_path, subdir, pairs)
        else:
            # Two-level: src_dir/{part}/{task}/{N}.hdf5
            task_dirs = sorted([
                d for d in os.listdir(subdir_path)
                if os.path.isdir(os.path.join(subdir_path, d))
            ])
            for task_name in task_dirs:
                task_path = os.path.join(subdir_path, task_name)
                _collect_task_pairs(task_path, task_name, pairs)

    return pairs


def _collect_task_pairs(task_path: str, task_name: str, pairs: list):
    """Collect hdf5/mp4 pairs from a single task directory."""
    hdf5_files = sorted(glob.glob(os.path.join(task_path, "*.hdf5")))
    for hdf5_path in hdf5_files:
        stem = os.path.splitext(os.path.basename(hdf5_path))[0]
        mp4_path = os.path.join(task_path, f"{stem}.mp4")
        if not os.path.exists(mp4_path):
            mp4_path = None
        pairs.append((hdf5_path, mp4_path, task_name))


def convert_egodex(src_dir: str, dst_dir: str, max_samples: int = 0):
    """Convert all egodex sequences from ARKit coords to +Z up."""
    pairs = collect_hdf5_pairs(src_dir)

    if not pairs:
        print(f"No HDF5 files found in {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    for idx, (hdf5_src, mp4_src, task_name) in enumerate(pairs):
        if max_samples > 0 and idx >= max_samples:
            break

        out_name = f"{idx:06d}_{task_name}"
        out_dir = os.path.join(dst_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        hdf5_dst = os.path.join(out_dir, "0.hdf5")
        convert_hdf5(hdf5_src, hdf5_dst)

        if mp4_src is not None:
            mp4_dst = os.path.join(out_dir, "0.mp4")
            shutil.copy2(mp4_src, mp4_dst)

        print(f"[{idx:06d}] {task_name} <- {os.path.basename(hdf5_src)}")

    total = min(len(pairs), max_samples) if max_samples > 0 else len(pairs)
    print(f"\nDone. Converted {total} sequences -> {dst_dir}")


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
