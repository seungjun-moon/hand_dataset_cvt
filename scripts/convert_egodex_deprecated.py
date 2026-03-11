#!/usr/bin/env python3
"""Convert egodex datasets from ARKit coordinates (+Y up) to +Z up coordinates.

ARKit coordinate system: +X right, +Y up, -Z forward
Target coordinate system: +X right, +Y forward, +Z up

Conversion (90-degree rotation around X-axis):
    Camera c2w:  T_world @ arkit_c2w
    3D points:   R_world @ points
    transforms_cam: inv(camera_c2w) @ joint_world  (per frame)

Usage:
    python scripts/convert_egodex.py --src DATASET/egodex --dst CONVERT/egodex
"""

import argparse
import glob
import os
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

R_WORLD = T_WORLD[:3, :3]


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


def convert_egodex(src_dir: str, dst_dir: str, inplace: bool = False):
    """Convert all egodex sequences from ARKit coords to +Z up."""
    if inplace:
        dst_dir = src_dir

    seq_dirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ])

    if not seq_dirs:
        print(f"No sequences found in {src_dir}")
        return

    if not inplace:
        os.makedirs(dst_dir, exist_ok=True)

    for i, seq_name in enumerate(seq_dirs):
        src_seq = os.path.join(src_dir, seq_name)
        dst_seq = os.path.join(dst_dir, seq_name)

        hdf5_files = sorted(glob.glob(os.path.join(src_seq, "*.hdf5")))
        if not hdf5_files:
            print(f"  Skipping {seq_name}: no .hdf5 files")
            continue

        if not inplace:
            os.makedirs(dst_seq, exist_ok=True)

        for hdf5_src in hdf5_files:
            basename = os.path.basename(hdf5_src)
            stem = os.path.splitext(basename)[0]

            if inplace:
                hdf5_tmp = hdf5_src + ".tmp"
                convert_hdf5(hdf5_src, hdf5_tmp)
                os.replace(hdf5_tmp, hdf5_src)
            else:
                hdf5_dst = os.path.join(dst_seq, basename)
                convert_hdf5(hdf5_src, hdf5_dst)

                # Copy matching mp4 if it exists
                mp4_src = os.path.join(src_seq, f"{stem}.mp4")
                mp4_dst = os.path.join(dst_seq, f"{stem}.mp4")
                if os.path.exists(mp4_src) and not os.path.exists(mp4_dst):
                    shutil.copy2(mp4_src, mp4_dst)

        print(f"[{i:04d}] {seq_name}")

    print(f"\nDone. Converted {len(seq_dirs)} sequences -> {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert egodex datasets from ARKit coords to +Z up"
    )
    parser.add_argument("--src", default="DATASET/egodex",
                        help="Source egodex directory (default: DATASET/egodex)")
    parser.add_argument("--dst", default="CONVERT/egodex",
                        help="Output directory (default: CONVERT/egodex)")
    parser.add_argument("--inplace", action="store_true",
                        help="Convert in-place (overwrite source files)")
    args = parser.parse_args()

    convert_egodex(args.src, args.dst, inplace=args.inplace)


if __name__ == "__main__":
    main()
