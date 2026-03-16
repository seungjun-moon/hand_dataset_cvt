#!/usr/bin/env python3
"""
Convert DexYCB dataset into egodex format.

DexYCB structure:
    DATASET/dex_ycb/
        {date}-{subject}/
            {timestamp}/
                meta.yml
                pose.npz
                {camera_serial}/
                    color_XXXXXX.jpg
                    aligned_depth_to_color_XXXXXX.png
                    labels_XXXXXX.npz  (seg, pose_y, pose_m, joint_3d, joint_2d) # joint_3d is in cam_space
        calibration/
            intrinsics/{serial}_640x480.yml
            extrinsics_{name}/extrinsics.yml
            mano_{name}/

Output structure:
    CONVERTED/dex_ycb/
        {object_name}/
            {seq_idx:06d}_label_{cam_idx:02d}.hdf5
            {seq_idx:06d}_video_{cam_idx:02d}.mp4

Sequences are clustered by grasped object. Each sequence produces files
for all requested cameras.

Usage:
    python scripts/convert_dex_ycb.py --src DATASET/dex_ycb --dst CONVERTED/dex_ycb
    python scripts/convert_dex_ycb.py --cameras 0 1 2 --max-samples 5
"""

import argparse
import os
import sys

import cv2
import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manopth.manolayer import ManoLayer
from utils.io import (
    collect_color_paths,
    images_to_mp4,
    load_extrinsics,
    load_frame_labels,
    load_intrinsics,
    load_yaml,
    write_egodex_hdf5,
)

MANO_ROOT = "/rlwrld3/home/seungjun/HO-Cap-Annotation/config/mano_models"
from utils.joint_mapping import (
    BODY_JOINTS,
    MANO_TO_EGODEX_SUFFIX,
    METACARPAL_INTERPOLATION,
)
from utils.transforms import (
    interpolate_joint,
    joints_to_transforms,
    make_transform,
)

# YCB object id -> name
YCB_OBJECTS = {
    1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box",
    4: "005_tomato_soup_can", 5: "006_mustard_bottle", 6: "007_tuna_fish_can",
    7: "008_pudding_box", 8: "009_gelatin_box", 9: "010_potted_meat_can",
    10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
    13: "024_bowl", 14: "025_mug", 15: "035_power_drill",
    16: "036_wood_block", 17: "037_scissors", 18: "040_large_marker",
    19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick",
}


def load_mano_betas(calibration_dir: str, mano_calib: str) -> np.ndarray:
    """Load MANO beta parameters from calibration mano.yml.

    Returns:
        (10,) array of shape parameters.
    """
    mano_yml = os.path.join(calibration_dir, f"mano_{mano_calib}", "mano.yml")
    data = load_yaml(mano_yml)
    return np.array(data["betas"], dtype=np.float32)


# Cache PCA components per side
_mano_pca_cache = {}


def _get_mano_pca(side: str):
    """Get PCA components and hand mean for a given side."""
    if side not in _mano_pca_cache:
        mano = ManoLayer(
            mano_root=MANO_ROOT, side=side,
            use_pca=True, ncomps=45, flat_hand_mean=False,
        )
        _mano_pca_cache[side] = (
            mano.th_selected_comps.numpy(),   # (45, 45)
            mano.th_hands_mean.numpy().squeeze(),  # (45,)
        )
    return _mano_pca_cache[side]


def convert_mano_pca_to_rotmat(pose_m_batch: np.ndarray, side: str,
                                cam_R: np.ndarray, cam_t: np.ndarray):
    """Convert PCA MANO params to rotation matrices.

    Args:
        pose_m_batch: (M, 51) PCA pose params (3 global + 45 PCA + 3 transl).
        side: 'left' or 'right'.
        cam_R: (3, 3) camera-to-world rotation matrix.
        cam_t: (3,) camera-to-world translation.

    Returns:
        global_orient_worldspace: (M, 3, 3) global orientation in world space.
        hand_pose: (M, 15, 3, 3) per-joint rotation matrices.
        transl_worldspace: (M, 3) translation in world space.
    """
    pca_comps, hands_mean = _get_mano_pca(side)
    M = pose_m_batch.shape[0]

    global_orient_world = np.zeros((M, 3, 3), dtype=np.float32)
    hand_pose = np.zeros((M, 15, 3, 3), dtype=np.float32)
    transl_cam = pose_m_batch[:, 48:].astype(np.float32)
    transl_world = (transl_cam @ cam_R.T + cam_t).astype(np.float32)

    for i in range(M):
        # Global orientation: axis-angle -> rotmat
        R_cam, _ = cv2.Rodrigues(pose_m_batch[i, :3].astype(np.float64))
        global_orient_world[i] = (cam_R @ R_cam).astype(np.float32)

        # Hand pose: PCA -> full axis-angle -> per-joint rotmats
        pca_coeffs = pose_m_batch[i, 3:48]
        raw_hand_aa = pca_coeffs @ pca_comps + hands_mean  # (45,)
        for j in range(15):
            aa = raw_hand_aa[j * 3:(j + 1) * 3].astype(np.float64)
            hand_pose[i, j], _ = cv2.Rodrigues(aa)

    return global_orient_world, hand_pose, transl_world


def build_egodex_data_for_sequence(
    camera_dir: str,
    meta: dict,
    calibration_dir: str,
    serial: str,
    num_frames: int,
):
    """Build world-space transforms and confidences for one sequence+camera.

    DexYCB joint_3d is in camera coordinates. Extrinsics map cam->world.
    Camera-space transforms can be recovered as inv(camera) @ transforms.

    Invalid frames (missing labels or joint_3d == -1) are filtered out entirely.

    Returns:
        intrinsic: (3, 3) array
        transforms_dict: {joint_name: (M, 4, 4)} world-space (M = valid frames)
        confidences_dict: {joint_name: (M,)}
        valid_indices: (M,) original frame indices that are valid
        mano_dict: dict with 'betas' (10,), 'pose' (M, 51), 'side' str
    """
    intrinsic = load_intrinsics(calibration_dir, serial)
    # DexYCB extrinsics map cam->world (cam_pose directly)
    cam_pose = load_extrinsics(calibration_dir, meta["extrinsics"], serial)

    # DexYCB has exactly one hand side per sequence
    mano_side = meta["mano_sides"][0].lower()

    transforms_dict = {}
    confidences_dict = {}

    identity = np.eye(4, dtype=np.float32)

    # Load MANO betas (same for all frames in a sequence)
    mano_calib = meta["mano_calib"][0]
    mano_betas = load_mano_betas(calibration_dir, mano_calib)

    # Collect per-frame joint_3d and pose_m (in camera coordinates)
    all_joint_3d_cam = np.zeros((num_frames, 21, 3), dtype=np.float32)
    all_pose_m = np.zeros((num_frames, 51), dtype=np.float32)
    frame_valid = np.zeros(num_frames, dtype=bool)

    for frame_i in range(num_frames):
        label_path = os.path.join(camera_dir, f"labels_{frame_i:06d}.npz")
        if not os.path.exists(label_path):
            continue
        labels = load_frame_labels(label_path)
        if "joint_3d" not in labels:
            continue
        j3d = labels["joint_3d"][0].astype(np.float32)  # (21, 3)
        if np.any(j3d == -1):
            continue
        all_joint_3d_cam[frame_i] = j3d
        if "pose_m" in labels:
            all_pose_m[frame_i] = labels["pose_m"][0].astype(np.float32)
        frame_valid[frame_i] = True

    # Filter to valid frames only
    valid_indices = np.where(frame_valid)[0]
    M = len(valid_indices)
    joint_3d_valid = all_joint_3d_cam[valid_indices]  # (M, 21, 3)
    pose_m_valid = all_pose_m[valid_indices]  # (M, 51)

    # Convert camera-space joint positions to 4x4 transforms
    all_transforms_cam = np.zeros((M, 21, 4, 4), dtype=np.float32)
    for i in range(M):
        all_transforms_cam[i] = joints_to_transforms(joint_3d_valid[i])

    # Convert to world-space transforms: T_world = cam_pose @ T_cam
    all_transforms_world = cam_pose @ all_transforms_cam  # broadcast (4,4) @ (M,21,4,4)

    # Camera transform in world space (static, repeated for valid frames)
    cam_tf = np.tile(cam_pose, (M, 1, 1))
    transforms_dict["camera"] = cam_tf

    # Body joints: not available in DexYCB
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # All valid frames have confidence 1.0
    conf = np.ones(M, dtype=np.float32)

    # Assign to each hand side
    for side in ["left", "right"]:
        is_active = (side == mano_side)

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms_world[:, mano_idx]
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc_world = np.zeros((M, 4, 4), dtype=np.float32)
                for i in range(M):
                    pos = interpolate_joint(joint_3d_valid[i], idx_a, idx_b, alpha=0.3)
                    direction = joint_3d_valid[i, idx_b] - joint_3d_valid[i, idx_a]
                    mc_cam = make_transform(pos, direction)
                    mc_world[i] = cam_pose @ mc_cam
                transforms_dict[name] = mc_world
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

    go_world, hand_pose, transl_world = \
        convert_mano_pca_to_rotmat(
            pose_m_valid, mano_side, cam_pose[:3, :3], cam_pose[:3, 3])

    # World-space 3D keypoints: extract translation from world-space transforms
    # for the active side's joints (21 MANO joints)
    kpt3d = np.zeros((M, 21, 3), dtype=np.float32)
    for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
        name = f"{mano_side}{suffix}"
        kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

    mano_dict = {
        "betas": mano_betas,
        "global_orient_worldspace": go_world,
        "hand_pose": hand_pose,
        "transl_worldspace": transl_world,
        "kpt3d": kpt3d,
        "side": mano_side,
    }

    return intrinsic, transforms_dict, confidences_dict, valid_indices, mano_dict


def _verify_output(out_dir: str, seq_idx: int, cam_idx: int, expected_frames: int):
    """Verify the converted output: check HDF5 frame counts and video frame counts."""
    prefix = f"{seq_idx:06d}"
    hdf5_path = os.path.join(out_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
    with h5py.File(hdf5_path, "r") as f:
        sample_key = list(f["transforms"].keys())[0]
        if sample_key == "gravity":
            sample_key = list(f["transforms"].keys())[1]
        hdf5_frames = f[f"transforms/{sample_key}"].shape[0]
        if hdf5_frames != expected_frames:
            print(f"  WARNING: HDF5 has {hdf5_frames} frames, expected {expected_frames}")

    for suffix in ["video"]:
        video_path = os.path.join(out_dir, f"{prefix}_{suffix}_{cam_idx:02d}.mp4")
        if not os.path.exists(video_path):
            print(f"  WARNING: {os.path.basename(video_path)} not created")
            continue
        cap = cv2.VideoCapture(video_path)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if video_frames != expected_frames:
            print(f"  WARNING: {os.path.basename(video_path)} has {video_frames} frames, expected {expected_frames}")
        else:
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  OK: {os.path.basename(video_path)} ({video_frames} frames, {size_mb:.1f} MB)")


def convert_dex_ycb(src_dir: str, dst_dir: str, cameras: list = None,
                    fps: float = 30.0, max_samples: int = 0):
    """Convert DexYCB sequences to egodex format, clustered by grasped object."""
    calibration_dir = os.path.join(src_dir, "calibration")
    os.makedirs(dst_dir, exist_ok=True)

    # First pass: collect all sequences grouped by grasped object
    subject_dirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and d.startswith("20")
    ])

    # {object_name: [(subject_dir_name, seq_name, meta), ...]}
    clusters = {}
    for subject_dir_name in subject_dirs:
        subject_path = os.path.join(src_dir, subject_dir_name)
        seq_dirs = sorted([
            d for d in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, d))
        ])
        for seq_name in seq_dirs:
            meta_path = os.path.join(subject_path, seq_name, "meta.yml")
            if not os.path.exists(meta_path):
                continue
            meta = load_yaml(meta_path)
            grasp_idx = meta["ycb_grasp_ind"]
            grasped_obj_id = meta["ycb_ids"][grasp_idx]
            obj_name = YCB_OBJECTS.get(grasped_obj_id, f"ycb_{grasped_obj_id:03d}")
            clusters.setdefault(obj_name, []).append(
                (subject_dir_name, seq_name, meta)
            )

    # Second pass: convert, respecting max_samples globally
    global_count = 0
    for obj_name in sorted(clusters.keys()):
        sequences = clusters[obj_name]
        obj_dir = os.path.join(dst_dir, obj_name)
        os.makedirs(obj_dir, exist_ok=True)

        for seq_idx, (subject_dir_name, seq_name, meta) in enumerate(sequences):
            if max_samples > 0 and global_count >= max_samples:
                break

            serials = meta["serials"]
            num_frames = meta["num_frames"]
            seq_path = os.path.join(src_dir, subject_dir_name, seq_name)

            # Use all cameras if not specified
            cam_list = cameras if cameras is not None else list(range(len(serials)))

            any_camera_ok = False
            for cam_idx in cam_list:
                if cam_idx >= len(serials):
                    print(f"  Skipping {subject_dir_name}/{seq_name} cam {cam_idx}: out of range")
                    continue

                serial = serials[cam_idx]
                camera_dir = os.path.join(seq_path, serial)
                if not os.path.isdir(camera_dir):
                    print(f"  Skipping {subject_dir_name}/{seq_name} cam {cam_idx}: dir not found")
                    continue

                intrinsic, transforms_dict, confidences_dict, valid_indices, mano_dict = \
                    build_egodex_data_for_sequence(
                        camera_dir, meta, calibration_dir, serial, num_frames,
                    )

                n_valid = len(valid_indices)
                print(f"[{obj_name}/{seq_idx:06d}] {subject_dir_name}/{seq_name} "
                      f"(cam={cam_idx:02d}, serial={serial}, frames={num_frames}, valid={n_valid})")

                if n_valid == 0:
                    print(f"  Skipping: no valid frames")
                    continue

                prefix = f"{seq_idx:06d}"

                # HDF5 label
                hdf5_path = os.path.join(obj_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
                write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                                  confidences_dict, mano_dict=mano_dict)

                # RGB video (valid frames only)
                all_color_paths = collect_color_paths(camera_dir)
                valid_color_paths = [all_color_paths[i] for i in valid_indices
                                     if i < len(all_color_paths)]
                rgb_path = os.path.join(obj_dir, f"{prefix}_video_{cam_idx:02d}.mp4")
                images_to_mp4(valid_color_paths, rgb_path, fps=fps)

                # Verify
                _verify_output(obj_dir, seq_idx, cam_idx, n_valid)
                any_camera_ok = True

            if any_camera_ok:
                global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    cam_str = "all" if cameras is None else str(len(cameras))
    print(f"\nDone. Converted {global_count} sequences ({cam_str} cam(s) each) to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert DexYCB to egodex format")
    parser.add_argument("--src", default="DATASET/dex_ycb", help="DexYCB source directory")
    parser.add_argument("--dst", default="CONVERTED/dex_ycb", help="Output directory")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Camera indices to extract (default: all)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max sequences to convert (0=all)")
    args = parser.parse_args()

    convert_dex_ycb(args.src, args.dst, cameras=args.cameras,
                    fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
