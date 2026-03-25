#!/usr/bin/env python3
"""Evaluate finger joint planarity on converted egodex-format datasets.

Human finger DIP and PIP joints are 1-DOF hinges, so the 4 keypoints
of each finger (Knuckle, IntermediateBase, IntermediateTip, Tip) should
be coplanar. Lower planarity error = more anatomically plausible.

Automatically detects the dataset format:
  - HDF5: *_label_*.hdf5 (egodex-format, video sequences)
  - WebDataset: *.tar containing {id}.data.pyd (HaMER format)

Joint source (--joints):
  - kpt3d: Use annotated 3D keypoints (default)
  - mano:  Use joints from MANO mesh (HDF5: reads mano_{side}/kpt3d,
           WebDataset: runs MANO forward pass, requires --mano-dir)

Usage:
    python scripts/eval_planarity.py --src CONVERTED/dex_ycb
    python scripts/eval_planarity.py --src CONVERTED/egodex --normalize
    python scripts/eval_planarity.py --src CONVERTED/dex_ycb --joints mano
    python scripts/eval_planarity.py --src ../hamer/hamer_training_data/dataset_tars/freihand-train
    python scripts/eval_planarity.py --src ../hamer/hamer_training_data/dataset_tars/freihand-train --joints mano --mano-dir /path/to/mano
"""

import argparse
import glob
import os
import pickle
import sys
import tarfile

import h5py
import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MANO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "hamer", "_DATA", "data", "mano"
)

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


# MANO joint indices per finger (Knuckle/MCP, PIP, DIP, Tip)
MANO_FINGER_INDICES = {
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Little": [17, 18, 19, 20],
}


def detect_format(src_dir: str) -> str:
    """Detect dataset format by checking file extensions."""
    for fname in os.listdir(src_dir):
        if fname.endswith(".tar"):
            return "webdataset"
    return "hdf5"


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


def eval_sequence_mano(hdf5_path, normalize=False, hop=1):
    """Evaluate planarity using MANO-derived joints from mano_{side}/kpt3d.

    Returns:
        dict {side_finger: (M,) array of errors} for sides with MANO data.
    """
    results = {}

    with h5py.File(hdf5_path, "r") as f:
        for side in ["left", "right"]:
            grp_name = f"mano_{side}"
            if grp_name not in f:
                continue

            kpt3d = f[grp_name]["kpt3d"][::hop]  # (M, 21, 3)
            M = kpt3d.shape[0]

            for finger_name, indices in MANO_FINGER_INDICES.items():
                errors = np.zeros(M, dtype=np.float32)
                for i in range(M):
                    pts = kpt3d[i, indices, :]  # (4, 3)
                    err = planarity_error(pts)
                    if normalize:
                        flen = finger_length(pts)
                        err = err / flen if flen > 1e-10 else 0.0
                    errors[i] = err

                key = f"{side}_{finger_name}"
                results[key] = errors

    return results


def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert (3,) axis-angle to (3, 3) rotation matrix."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


# MANO mesh vertex indices for fingertips (thumb, index, middle, ring, little)
MANO_FINGERTIP_VERTEX_IDS = [744, 320, 443, 554, 671]


def _mano_forward_batch(mano_model, hand_pose_aa_batch, betas_batch, is_right,
                        device="cpu"):
    """Batched MANO forward pass, return (B, 21, 3) joint positions.

    Args:
        mano_model: dict with 'right' and 'left' MANOLayer.
        hand_pose_aa_batch: (B, 48) axis-angle params.
        betas_batch: (B, 10) shape params.
        is_right: bool, same side for entire batch.
        device: 'cpu' or 'cuda'.

    MANO outputs 16 joints (wrist + 3 per finger, no tips).
    Fingertips are extracted from mesh vertices to form the full 21 joints:
        [wrist, thumb×4, index×4, middle×4, ring×4, little×4]
    """
    import torch

    B = hand_pose_aa_batch.shape[0]
    model = mano_model["right"] if is_right else mano_model["left"]

    # Convert axis-angle to rotation matrices in batch
    # hand_pose_aa_batch: (B, 48) -> global_orient (B, 1, 3, 3), hand_pose (B, 15, 3, 3)
    aa_all = torch.from_numpy(hand_pose_aa_batch).float().to(device)  # (B, 48)
    global_aa = aa_all[:, :3]          # (B, 3)
    hand_aa = aa_all[:, 3:48].reshape(B, 15, 3)  # (B, 15, 3)

    # Batch axis-angle -> rotation matrix using Rodrigues
    def _batch_rodrigues(aa):
        """(N, 3) axis-angle -> (N, 3, 3) rotation matrices."""
        angle = torch.norm(aa, dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        axis = aa / angle  # (N, 3)
        cos_a = torch.cos(angle).unsqueeze(-1)  # (N, 1, 1)
        sin_a = torch.sin(angle).unsqueeze(-1)  # (N, 1, 1)
        # Skew-symmetric matrix
        K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        eye = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
        return eye + sin_a * K + (1 - cos_a) * (K @ K)

    global_orient = _batch_rodrigues(global_aa).unsqueeze(1)  # (B, 1, 3, 3)
    hand_pose = _batch_rodrigues(hand_aa.reshape(-1, 3)).reshape(B, 15, 3, 3)

    betas_t = torch.from_numpy(betas_batch).float().to(device)  # (B, 10)

    with torch.no_grad():
        out = model(global_orient=global_orient, hand_pose=hand_pose,
                    betas=betas_t, pose2rot=False)

    joints_16 = out.joints.cpu().numpy()    # (B, 16, 3)
    vertices = out.vertices.cpu().numpy()   # (B, 778, 3)
    tips = vertices[:, MANO_FINGERTIP_VERTEX_IDS, :]  # (B, 5, 3)

    # Build 21-joint array: interleave 3 joints + 1 tip per finger
    # MANO 16 layout: [0:wrist, 1-3:thumb, 4-6:index, 7-9:middle, 10-12:ring, 13-15:little]
    # Target 21 layout: [0:wrist, 1-4:thumb(+tip), 5-8:index(+tip), ...]
    joints_21 = np.zeros((B, 21, 3), dtype=np.float32)
    joints_21[:, 0] = joints_16[:, 0]
    for fi in range(5):
        src_start = 1 + fi * 3
        dst_start = 1 + fi * 4
        joints_21[:, dst_start:dst_start + 3] = joints_16[:, src_start:src_start + 3]
        joints_21[:, dst_start + 3] = tips[:, fi]

    return joints_21


def eval_sample_webdataset(kpts_3d, normalize=False):
    """Evaluate planarity for a single webdataset sample.

    Args:
        kpts_3d: (21, 4) array with [x, y, z, conf] in camera space.

    Returns:
        dict {finger_name: scalar error} for fingers with valid joints.
    """
    results = {}
    for finger_name, indices in MANO_FINGER_INDICES.items():
        # Check confidence for all 4 joints
        confs = kpts_3d[indices, 3]
        if np.any(confs < 0.5):
            continue

        pts = kpts_3d[indices, :3]  # (4, 3)
        err = planarity_error(pts)
        if normalize:
            flen = finger_length(pts)
            err = err / flen if flen > 1e-10 else 0.0
        results[finger_name] = err

    return results


def _eval_planarity_from_joints(kpt3d_batch, normalize=False):
    """Compute per-finger planarity errors from (B, 21, 3) joint positions.

    Returns:
        dict {finger_name: (B,) array of errors}.
    """
    B = kpt3d_batch.shape[0]
    results = {}
    for finger_name, indices in MANO_FINGER_INDICES.items():
        errors = np.zeros(B, dtype=np.float32)
        for i in range(B):
            pts = kpt3d_batch[i, indices, :]
            err = planarity_error(pts)
            if normalize:
                flen = finger_length(pts)
                err = err / flen if flen > 1e-10 else 0.0
            errors[i] = err
        results[finger_name] = errors
    return results


def eval_dataset_webdataset(src_dir, normalize=False, max_samples=0, max_tars=0,
                            joints="kpt3d", mano_model=None, device="cpu",
                            batch_size=256):
    """Evaluate planarity across all samples in a webdataset directory."""
    tar_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".tar")])
    if max_tars > 0:
        tar_files = tar_files[:max_tars]

    all_errors = {}  # finger -> list of arrays
    n_samples = 0

    if joints == "mano":
        # Collect annotations per side, then batch MANO forward pass
        # {side: {"hand_pose": [], "betas": []}}
        side_anns = {"right": {"hand_pose": [], "betas": []},
                     "left":  {"hand_pose": [], "betas": []}}

        for tar_name in tqdm(tar_files, desc="Loading annotations"):
            tar_path = os.path.join(src_dir, tar_name)
            with tarfile.open(tar_path) as tf:
                names = set(tf.getnames())
                pyd_names = sorted([n for n in names if n.endswith(".data.pyd")])

                for pyd_name in pyd_names:
                    if max_samples > 0 and n_samples >= max_samples:
                        break
                    try:
                        pyd_bytes = tf.extractfile(pyd_name).read()
                        ann = pickle.loads(pyd_bytes)[0]
                    except Exception:
                        continue
                    if not bool(ann.get("has_hand_pose", 0) > 0.5):
                        continue
                    side = "right" if bool(ann["right"] > 0.5) else "left"
                    side_anns[side]["hand_pose"].append(ann["hand_pose"])
                    side_anns[side]["betas"].append(ann["betas"])
                    n_samples += 1

            if max_samples > 0 and n_samples >= max_samples:
                break

        # Batched MANO forward + planarity eval per side
        for side in ["right", "left"]:
            poses = side_anns[side]["hand_pose"]
            if not poses:
                continue
            is_right = side == "right"
            poses_np = np.array(poses, dtype=np.float32)
            betas_np = np.array(side_anns[side]["betas"], dtype=np.float32)
            N = len(poses)

            print(f"Running batched MANO ({side}, {N} samples, device={device})...")
            # Process in batches
            all_joints = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                j21 = _mano_forward_batch(
                    mano_model, poses_np[start:end], betas_np[start:end],
                    is_right, device=device)
                all_joints.append(j21)
            all_joints = np.concatenate(all_joints, axis=0)  # (N, 21, 3)

            results = _eval_planarity_from_joints(all_joints, normalize=normalize)
            for finger, errors in results.items():
                all_errors.setdefault(finger, []).append(errors)

    else:  # kpt3d
        for tar_name in tqdm(tar_files, desc="Scanning tars"):
            tar_path = os.path.join(src_dir, tar_name)
            with tarfile.open(tar_path) as tf:
                names = set(tf.getnames())
                pyd_names = sorted([n for n in names if n.endswith(".data.pyd")])

                for pyd_name in pyd_names:
                    if max_samples > 0 and n_samples >= max_samples:
                        break
                    try:
                        pyd_bytes = tf.extractfile(pyd_name).read()
                        ann = pickle.loads(pyd_bytes)[0]
                    except Exception:
                        continue

                    kpts_3d = ann.get("keypoints_3d")
                    if kpts_3d is None:
                        continue
                    results = eval_sample_webdataset(kpts_3d, normalize=normalize)
                    for finger, err in results.items():
                        all_errors.setdefault(finger, []).append(err)
                    n_samples += 1

            if max_samples > 0 and n_samples >= max_samples:
                break

    if not all_errors:
        print(f"No valid samples found in {src_dir}")
        return

    print_report(src_dir, all_errors, normalize, n_label=f"{n_samples} samples")


def eval_dataset_hdf5(src_dir, normalize=False, max_samples=0, hop=1, joints="kpt3d"):
    """Evaluate planarity across all sequences in a converted HDF5 dataset."""
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

        eval_fn = eval_sequence_mano if joints == "mano" else eval_sequence
        for hdf5_path in tqdm(hdf5_files, desc=f"  {seq_name}", leave=False):
            results = eval_fn(hdf5_path, normalize=normalize, hop=hop)
            for key, errors in results.items():
                all_errors.setdefault(key, []).append(errors)

        n_seqs += 1

    if not all_errors:
        print(f"No valid sequences found in {src_dir}")
        return

    # Group by finger (across sides)
    finger_errors = {}
    for key, arrays in all_errors.items():
        _, finger = key.split("_", 1)
        combined = np.concatenate(arrays)
        finger_errors.setdefault(finger, []).append(combined)

    # Merge arrays per finger
    merged = {}
    for finger, arrays in finger_errors.items():
        merged[finger] = np.concatenate(arrays)

    print_report(src_dir, merged, normalize, n_label=f"{n_seqs} sequences")


def print_report(src_dir, all_errors, normalize, n_label=""):
    """Print planarity error report.

    Args:
        all_errors: dict {finger_name: array or list of errors} (in raw units).
        normalize: whether errors are normalized by finger length.
        n_label: description string for count (e.g. "42 sequences").
    """
    unit = "% of finger length" if normalize else "mm"
    scale = 100.0 if normalize else 1000.0  # m -> mm, or ratio -> %

    print(f"\nPlanarity Error ({unit})")
    print(f"Dataset: {src_dir} ({n_label})")
    print(f"{'Finger':<20s} {'Mean':>10s} {'Median':>10s} {'Std':>10s} {'Max':>10s} {'Frames':>8s}")
    print("-" * 68)

    grand_all = []

    for finger in ["Index", "Middle", "Ring", "Little"]:
        if finger not in all_errors:
            continue
        raw = all_errors[finger]
        scaled = np.asarray(raw) * scale
        valid = scaled[~np.isnan(scaled)]
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
                        help="Converted dataset directory (HDF5) or WebDataset tar directory")
    parser.add_argument("--normalize", action="store_true",
                        help="Report error as %% of finger length")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences/samples to evaluate (0=all)")
    parser.add_argument("--hop", type=int, default=1,
                        help="Evaluate every N-th frame (default: 1=all, HDF5 only)")
    parser.add_argument("--max-tars", type=int, default=0,
                        help="Max tar files to scan (0=all, WebDataset only)")
    parser.add_argument("--joints", choices=["kpt3d", "mano"], default="kpt3d",
                        help="Joint source: kpt3d (annotated 3D keypoints) or mano (MANO mesh joints)")
    parser.add_argument("--mano-dir", default=DEFAULT_MANO_DIR,
                        help="MANO model directory (WebDataset + --joints mano only)")
    parser.add_argument("--device", default="cuda",
                        help="Device for MANO forward pass (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for MANO forward pass (default: 256)")
    args = parser.parse_args()

    fmt = detect_format(args.src)
    print(f"Detected format: {fmt}, joints: {args.joints}")

    if fmt == "webdataset":
        mano_model = None
        device = args.device
        if args.joints == "mano":
            import torch
            import smplx
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
            mano_dir = os.path.normpath(args.mano_dir)
            print(f"Loading MANO from {mano_dir} (device={device})")
            mano_model = {
                "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True, flat_hand_mean=False).to(device),
                "left": smplx.MANOLayer(model_path=mano_dir, is_rhand=False, flat_hand_mean=False).to(device),
            }
        eval_dataset_webdataset(args.src, normalize=args.normalize,
                                max_samples=args.max_samples,
                                max_tars=args.max_tars,
                                joints=args.joints, mano_model=mano_model,
                                device=device, batch_size=args.batch_size)
    else:
        eval_dataset_hdf5(args.src, normalize=args.normalize,
                          max_samples=args.max_samples, hop=args.hop,
                          joints=args.joints)


if __name__ == "__main__":
    main()
