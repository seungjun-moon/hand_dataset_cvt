#!/usr/bin/env python3
"""Evaluate finger joint planarity on converted egodex-format datasets.

Human finger DIP and PIP joints are 1-DOF hinges, so the 4 keypoints
of each finger (Knuckle, IntermediateBase, IntermediateTip, Tip) should
be coplanar. Lower planarity error = more anatomically plausible.

Usage:
    python scripts/eval_planarity.py --src CONVERTED/dex_ycb_cam_000
    python scripts/eval_planarity.py --src CONVERTED/egodex --normalize
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Finger joint suffixes in egodex convention (Knuckle=MCP, IntBase=PIP, IntTip=DIP, Tip)
FINGER_SUFFIXES = {
    "Index":  ["IndexFingerKnuckle", "IndexFingerIntermediateBase",
               "IndexFingerIntermediateTip", "IndexFingerTip"],
    "Middle": ["MiddleFingerKnuckle", "MiddleFingerIntermediateBase",
               "MiddleFingerIntermediateTip", "MiddleFingerTip"],
    "Ring":   ["RingFingerKnuckle", "RingFingerIntermediateBase",
               "RingFingerIntermediateTip", "RingFingerTip"],
    "Little": ["LittleFingerKnuckle", "LittleFingerIntermediateBase",
               "LittleFingerIntermediateTip", "LittleFingerTip"],
}

def planarity_error(points: np.ndarray, normalize: bool = False) -> float:
    """
    Compute planarity error for (4, 3) points.
    Fits a plane via SVD of the centered points.
    Returns the smallest singular value, which is the RMS out-of-plane distance * sqrt(N).
    If normalize=True, divides by finger length to get a scale-invariant ratio.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    err = float(s[-1])
    if normalize:
        length = finger_length(points)
        if length < 1e-8:
            return 0.0
        return err / length
    return err

def finger_length(points):
    """Sum of bone segment lengths."""
    return sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1))


def extract_positions(transforms):
    """Extract translation column from (N, 4, 4) transforms -> (N, 3)."""
    return transforms[:, :3, 3]


def eval_sequence(hdf5_path, normalize=False, hop=1):
    """Evaluate planarity for one HDF5 file.

    Args:
        hop: Evaluate every N-th frame (1=all frames, 10=every 10th).

    Returns:
        dict {side_finger: (M,) array of errors} for fingers with confidence > 0
    """
    results = {}

    with h5py.File(hdf5_path, "r") as f:
        tf_group = f["transforms"]
        conf_group = f["confidences"]

        for side in ["left", "right"]:
            for finger_name, suffixes in FINGER_SUFFIXES.items():
                joint_names = [f"{side}{s}" for s in suffixes]

                # Skip if any joint is missing
                if not all(name in tf_group for name in joint_names):
                    continue

                # Skip if confidence is zero (inactive hand)
                conf_key = joint_names[0]
                if conf_key in conf_group:
                    conf = conf_group[conf_key][:]
                    if np.all(conf == 0):
                        continue

                # Load positions: (N, 3) for each of 4 joints
                positions = [extract_positions(tf_group[name][::hop]) for name in joint_names]

                N = positions[0].shape[0]

                errors = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    pts = np.array([p[i] for p in positions])  # (4, 3)
                    err = planarity_error(pts)
                    if normalize:
                        flen = finger_length(pts)
                        err = err / flen if flen > 1e-10 else 0.0
                    errors[i] = err

                key = f"{side}_{finger_name}"
                results[key] = errors

    return results


def eval_dataset(src_dir, normalize=False, max_samples=0, hop=1):
    """Evaluate planarity across all sequences in a converted dataset."""
    seq_dirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ])

    all_errors = {}  # finger -> list of arrays
    n_seqs = 0

    if max_samples > 0:
        seq_dirs = seq_dirs[:max_samples]

    for seq_name in tqdm(seq_dirs, desc="Evaluating"):
        hdf5_files = sorted(glob.glob(os.path.join(src_dir, seq_name, "*_00.hdf5"))) # global keypoints are same anywhere.
        if not hdf5_files:
            continue

        for hdf5_path in tqdm(hdf5_files, desc=f"  {seq_name}", leave=False):
            results = eval_sequence(hdf5_path, normalize=normalize, hop=hop)
            for key, errors in results.items():
                all_errors.setdefault(key, []).append(errors)

        n_seqs += 1

    if not all_errors:
        print(f"No valid sequences found in {src_dir}")
        return

    # Print report
    unit = "% of finger length" if normalize else "mm"
    scale = 100.0 if normalize else 1000.0  # m -> mm, or ratio -> %

    print(f"\nPlanarity Error ({unit})")
    print(f"Dataset: {src_dir} ({n_seqs} sequences)")
    print(f"{'Finger':<20s} {'Mean':>10s} {'Median':>10s} {'Std':>10s} {'Max':>10s} {'Frames':>8s}")
    print("-" * 68)

    grand_all = []

    # Group by finger (across sides)
    finger_errors = {}
    for key, arrays in all_errors.items():
        _, finger = key.split("_", 1)
        combined = np.concatenate(arrays) * scale
        finger_errors.setdefault(finger, []).append(combined)

    for finger in ["Index", "Middle", "Ring", "Little"]:
        if finger not in finger_errors:
            continue
        combined = np.concatenate(finger_errors[finger])
        valid = combined[~np.isnan(combined)]
        grand_all.append(valid)
        print(f"{finger:<20s} {np.mean(valid):>10.3f} {np.median(valid):>10.3f} "
              f"{np.std(valid):>10.3f} {np.max(valid):>10.3f} {len(valid):>8d}")

    if grand_all:
        total = np.concatenate(grand_all)
        print("-" * 68)
        print(f"{'ALL':<20s} {np.mean(total):>10.3f} {np.median(total):>10.3f} "
              f"{np.std(total):>10.3f} {np.max(total):>10.3f} {len(total):>8d}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate finger joint planarity on converted datasets"
    )
    parser.add_argument("--src", default="CONVERTED/dex_ycb",
                        help="Converted dataset directory (default: CONVERTED/dex_ycb)")
    parser.add_argument("--normalize", action="store_true",
                        help="Report error as %% of finger length")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences to evaluate (0=all)")
    parser.add_argument("--hop", type=int, default=1,
                        help="Evaluate every N-th frame (default: 1=all)")
    args = parser.parse_args()

    eval_dataset(args.src, normalize=args.normalize, max_samples=args.max_samples,
                 hop=args.hop)


if __name__ == "__main__":
    main()
