#!/usr/bin/env python3
"""Generate random hand keypoints using smplx MANO and manotorch, save as egodex HDF5.

Produces datasets compatible with eval_alignment.py for comparing planarity
of joints from different MANO implementations.

Output:
    datasets/mano/000000_synthetic/0.hdf5          (smplx MANO, unconstrained)
    datasets/manotorch/000000_synthetic/0.hdf5      (manotorch, unconstrained)
    datasets/manotorch_constr/000000_synthetic/0.hdf5  (manotorch, anatomy-constrained)

Usage:
    python scripts/generate_random_keypoints.py
    python scripts/generate_random_keypoints.py --n-samples 5000 --pose-scale 1.0
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HaWoR"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HaWoR", "thirdparty", "manotorch"))

from utils.io import write_egodex_hdf5
from utils.joint_mapping import BODY_JOINTS, MANO_TO_EGODEX_SUFFIX, METACARPAL_INTERPOLATION
from utils.transforms import joints_to_transforms, interpolate_joint, make_transform

# smplx MANO joint order -> OpenPose 21-joint order
# smplx.MANOLayer outputs 16 joints; adding 5 fingertip vertices gives 21 in
# MANO-native order. This reindex maps to OpenPose/DexYCB convention:
#   0:wrist, 1-4:thumb, 5-8:index, 9-12:middle, 13-16:ring, 17-20:little
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# smplx vertex indices for fingertips (thumb, index, middle, ring, little)
FINGERTIP_VERTEX_IDS = [744, 320, 443, 554, 671]


def load_smplx_mano(mano_path: str):
    """Load smplx.MANOLayer."""
    import smplx
    model = smplx.MANOLayer(mano_path, is_rhand=True, use_pca=False, flat_hand_mean=True)
    return model


def load_manotorch(assets_path: str, constrain: bool = False):
    """Load MANOTorch wrapper."""
    from lib.models.mano_torch_wrapper import MANOTorch
    model = MANOTorch(side="right", mano_assets_root=assets_path, constrain=constrain)
    return model


def run_smplx_mano(model, pose_aa: torch.Tensor, betas: torch.Tensor) -> np.ndarray:
    """Run smplx MANO and return (B, 21, 3) joints in OpenPose order."""
    from hawor.utils.geometry import aa_to_rotmat
    B = pose_aa.shape[0]
    go_aa = pose_aa[:, :3]
    hp_aa = pose_aa[:, 3:].reshape(-1, 15, 3)
    go_rot = aa_to_rotmat(go_aa.reshape(-1, 3)).reshape(B, 1, 3, 3)
    hp_rot = aa_to_rotmat(hp_aa.reshape(-1, 3)).reshape(B, 15, 3, 3)
    with torch.no_grad():
        out = model(global_orient=go_rot, hand_pose=hp_rot, betas=betas, pose2rot=False)
    # 16 base joints + 5 fingertip vertices -> reorder to OpenPose
    verts = out.vertices  # (B, 778, 3)
    joints_base = out.joints  # (B, 16, 3)
    fingertips = verts[:, FINGERTIP_VERTEX_IDS, :]  # (B, 5, 3)
    joints_21 = torch.cat([joints_base, fingertips], dim=1)  # (B, 21, 3)
    joints_openpose = joints_21[:, MANO_TO_OPENPOSE, :]  # (B, 21, 3)
    return joints_openpose.numpy()


def run_manotorch(model, pose_aa: torch.Tensor, betas: torch.Tensor) -> np.ndarray:
    """Run MANOTorch and return (B, 21, 3) joints."""
    with torch.no_grad():
        out = model(pose_aa, betas)
    return out.joints.numpy()  # (B, 21, 3)


def joints_to_egodex_dicts(all_joints: np.ndarray, side: str = "right"):
    """Convert (N, 21, 3) joint positions to egodex-format dicts.

    Args:
        all_joints: (N, 21, 3) in OpenPose/DexYCB order.
        side: which hand side is active.

    Returns:
        transforms_dict, transforms_cam_dict, confidences_dict
    """
    N = all_joints.shape[0]
    identity = np.eye(4, dtype=np.float32)

    transforms_dict = {}
    transforms_cam_dict = {}
    confidences_dict = {}

    # Camera: identity (synthetic, no camera)
    transforms_dict["camera"] = np.tile(identity, (N, 1, 1))

    # Body joints: unavailable
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (N, 1, 1))
        transforms_cam_dict[name] = np.tile(identity, (N, 1, 1))
        confidences_dict[name] = np.zeros(N, dtype=np.float32)

    # Compute per-frame transforms from joint positions
    all_transforms = np.zeros((N, 21, 4, 4), dtype=np.float32)
    for i in range(N):
        all_transforms[i] = joints_to_transforms(all_joints[i])

    conf = np.ones(N, dtype=np.float32)
    other_side = "left" if side == "right" else "right"

    for s in [side, other_side]:
        is_active = (s == side)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{s}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms[:, mano_idx]
                transforms_cam_dict[name] = all_transforms[:, mano_idx]
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (N, 1, 1))
                transforms_cam_dict[name] = np.tile(identity, (N, 1, 1))
                confidences_dict[name] = np.zeros(N, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{s}{suffix}"
            if is_active:
                mc = np.zeros((N, 4, 4), dtype=np.float32)
                for i in range(N):
                    pos = interpolate_joint(all_joints[i], idx_a, idx_b, alpha=0.3)
                    direction = all_joints[i, idx_b] - all_joints[i, idx_a]
                    mc[i] = make_transform(pos, direction)
                transforms_dict[name] = mc
                transforms_cam_dict[name] = mc.copy()
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (N, 1, 1))
                transforms_cam_dict[name] = np.tile(identity, (N, 1, 1))
                confidences_dict[name] = np.zeros(N, dtype=np.float32)

    return transforms_dict, transforms_cam_dict, confidences_dict


def save_as_egodex(joints: np.ndarray, output_dir: str, side: str = "right"):
    """Save (N, 21, 3) joints as an egodex HDF5."""
    seq_dir = os.path.join(output_dir, "000000_synthetic")
    os.makedirs(seq_dir, exist_ok=True)

    transforms_dict, transforms_cam_dict, confidences_dict = \
        joints_to_egodex_dicts(joints, side=side)

    intrinsic = np.eye(3, dtype=np.float32)
    gravity = np.eye(3, dtype=np.float32)

    hdf5_path = os.path.join(seq_dir, "0.hdf5")
    write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                      transforms_cam_dict, confidences_dict, gravity)
    return hdf5_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate random MANO keypoints and save as egodex HDF5"
    )
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of random poses to generate (default: 1000)")
    parser.add_argument("--pose-scale", type=float, default=1.5,
                        help="Scale of random pose axis-angle noise (default: 1.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dst", default="datasets",
                        help="Base output directory (default: datasets)")
    parser.add_argument("--mano-path",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "HaWoR", "_DATA", "data", "mano"),
                        help="Path to smplx MANO model directory")
    parser.add_argument("--manotorch-assets",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "HaWoR", "thirdparty",
                                             "manotorch", "assets", "mano"),
                        help="Path to manotorch assets directory")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N = args.n_samples
    print(f"Generating {N} random poses (scale={args.pose_scale}, seed={args.seed})")

    # Generate shared random poses and betas
    pose_aa = torch.randn(N, 48) * args.pose_scale
    betas = torch.zeros(N, 10)

    # --- smplx MANO ---
    print("\nLoading smplx MANO...")
    mano_model = load_smplx_mano(os.path.abspath(args.mano_path))
    print("Running smplx MANO forward pass...")
    # Process in batches to avoid OOM
    batch_size = 128
    joints_mano = []
    for i in range(0, N, batch_size):
        j = run_smplx_mano(mano_model, pose_aa[i:i+batch_size], betas[i:i+batch_size])
        joints_mano.append(j)
    joints_mano = np.concatenate(joints_mano, axis=0)
    print(f"  joints shape: {joints_mano.shape}")

    path = save_as_egodex(joints_mano, os.path.join(args.dst, "mano"))
    print(f"  Saved: {path}")

    # --- manotorch (unconstrained) ---
    print("\nLoading manotorch (unconstrained)...")
    mt_raw = load_manotorch(os.path.abspath(args.manotorch_assets), constrain=False)
    print("Running manotorch forward pass...")
    joints_mt_raw = []
    for i in range(0, N, batch_size):
        j = run_manotorch(mt_raw, pose_aa[i:i+batch_size], betas[i:i+batch_size])
        joints_mt_raw.append(j)
    joints_mt_raw = np.concatenate(joints_mt_raw, axis=0)
    print(f"  joints shape: {joints_mt_raw.shape}")

    path = save_as_egodex(joints_mt_raw, os.path.join(args.dst, "manotorch"))
    print(f"  Saved: {path}")

    # --- manotorch (anatomy-constrained) ---
    print("\nLoading manotorch (anatomy-constrained)...")
    mt_constr = load_manotorch(os.path.abspath(args.manotorch_assets), constrain=True)
    print("Running manotorch constrained forward pass...")
    joints_mt_constr = []
    for i in range(0, N, batch_size):
        j = run_manotorch(mt_constr, pose_aa[i:i+batch_size], betas[i:i+batch_size])
        joints_mt_constr.append(j)
    joints_mt_constr = np.concatenate(joints_mt_constr, axis=0)
    print(f"  joints shape: {joints_mt_constr.shape}")

    path = save_as_egodex(joints_mt_constr, os.path.join(args.dst, "manotorch_constr"))
    print(f"  Saved: {path}")

    # Quick summary
    print(f"\nDone. Generated {N} poses for 3 variants:")
    print(f"  datasets/mano/             - smplx MANO (unconstrained)")
    print(f"  datasets/manotorch/        - manotorch (unconstrained)")
    print(f"  datasets/manotorch_constr/ - manotorch (anatomy-constrained)")
    print(f"\nRun eval_alignment.py to compare:")
    print(f"  python scripts/eval_alignment.py --dataset_dir datasets/mano")
    print(f"  python scripts/eval_alignment.py --dataset_dir datasets/manotorch")
    print(f"  python scripts/eval_alignment.py --dataset_dir datasets/manotorch_constr")


if __name__ == "__main__":
    main()
