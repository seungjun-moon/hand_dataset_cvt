#!/usr/bin/env python3
"""
Convert HO-Cap dataset into egodex format.

HO-Cap structure:
    DATASET/
        subject_N/
            {timestamp}/
                meta.yaml
                poses_m.npy           (2, num_frames, 51) MANO params in world space
                {camera_serial}/
                    color_XXXXXX.jpg
                    depth_XXXXXX.png
                    label_XXXXXX.npz  (cam_K, hand_joints_3d, hand_joints_2d, ...)
        calibration/
            intrinsics/{serial}.yaml
            extrinsics/extrinsics_{name}.yaml
            mano/{subject_id}.yaml

poses_m format (axis-angle, NOT PCA):
    [0:3]   global_orient (axis-angle, world space)
    [3:48]  hand_pose (15 joints × 3 axis-angle)
    [48:51] translation (world space)
    Index 0 = right hand, Index 1 = left hand. -1 indicates inactive.

Output structure:
    CONVERTED/ho_cap/
        {subject_id}/
            {seq_idx:06d}_label_{cam_idx:02d}.hdf5
            {seq_idx:06d}_video_{cam_idx:02d}.mp4

Usage:
    python scripts/convert_ho_cap.py --src ../HO-Cap/datasets --dst CONVERTED/ho_cap
    python scripts/convert_ho_cap.py --cameras 0 1 2 --max-samples 5
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import (
    images_to_mp4,
    load_yaml,
    write_egodex_hdf5,
)
from utils.joint_mapping import (
    BODY_JOINTS,
    MANO_TO_EGODEX_SUFFIX,
    METACARPAL_INTERPOLATION,
)
from utils.transforms import (
    extrinsics_tuple_to_4x4,
    interpolate_joint,
    joints_to_transforms,
    make_transform,
)

# Hand index convention: 0 = right, 1 = left
HAND_INDEX_TO_SIDE = {0: "right", 1: "left"}


def load_ho_cap_intrinsics(calibration_dir: str, serial: str) -> np.ndarray:
    """Load HO-Cap camera intrinsics as a (3, 3) matrix."""
    path = os.path.join(calibration_dir, "intrinsics", f"{serial}.yaml")
    data = load_yaml(path)
    intr = data["color"]
    K = np.array([
        [intr["fx"], 0, intr["ppx"]],
        [0, intr["fy"], intr["ppy"]],
        [0, 0, 1],
    ], dtype=np.float32)
    return K


def load_ho_cap_extrinsics(calibration_dir: str, extrinsics_file: str,
                            serial: str) -> np.ndarray:
    """Load HO-Cap camera extrinsics as a (4, 4) camera-to-world matrix."""
    path = os.path.join(calibration_dir, "extrinsics", extrinsics_file)
    data = load_yaml(path)
    vals = data["extrinsics"][serial]
    return extrinsics_tuple_to_4x4(vals)


def load_ho_cap_mano_betas(calibration_dir: str, subject_id: str) -> np.ndarray:
    """Load per-subject MANO betas from calibration/mano/{subject_id}.yaml."""
    path = os.path.join(calibration_dir, "mano", f"{subject_id}.yaml")
    data = load_yaml(path)
    return np.array(data["betas"], dtype=np.float32)


def convert_mano_axisangle_to_rotmat(poses_m_batch: np.ndarray):
    """Convert axis-angle MANO params to rotation matrices.

    Args:
        poses_m_batch: (M, 51) axis-angle pose params, already in world space.
            [0:3] global_orient, [3:48] hand_pose (15×3), [48:51] translation.

    Returns:
        global_orient: (M, 3, 3) global orientation as rotation matrices.
        hand_pose: (M, 15, 3, 3) per-joint rotation matrices.
        transl: (M, 3) world-space translation.
    """
    M = poses_m_batch.shape[0]
    global_orient = np.zeros((M, 3, 3), dtype=np.float32)
    hand_pose = np.zeros((M, 15, 3, 3), dtype=np.float32)
    transl = poses_m_batch[:, 48:51].astype(np.float32)

    for i in range(M):
        R, _ = cv2.Rodrigues(poses_m_batch[i, :3].astype(np.float64))
        global_orient[i] = R.astype(np.float32)

        for j in range(15):
            aa = poses_m_batch[i, 3 + j * 3:3 + (j + 1) * 3].astype(np.float64)
            hand_pose[i, j], _ = cv2.Rodrigues(aa)

    return global_orient, hand_pose, transl


def build_egodex_data_for_sequence(
    seq_path: str,
    meta: dict,
    calibration_dir: str,
    serial: str,
    poses_m: np.ndarray,
    num_frames: int,
    subject_id: str,
):
    """Build world-space transforms and confidences for one sequence+camera.

    hand_joints_3d from labels is in camera space. Extrinsics map cam→world.
    poses_m is already in world space.

    Returns:
        intrinsic: (3, 3) array
        transforms_dict: {joint_name: (M, 4, 4)} world-space
        confidences_dict: {joint_name: (M,)}
        valid_indices: (M,) original frame indices that are valid
        mano_dicts: list of mano_dict per active hand side
    """
    intrinsic = load_ho_cap_intrinsics(calibration_dir, serial)
    cam_pose = load_ho_cap_extrinsics(
        calibration_dir, meta["extrinsics"], serial)

    camera_dir = os.path.join(seq_path, serial)

    # Determine active hand sides from poses_m data
    active_hands = {}  # {side: hand_index}
    for hand_idx in range(2):
        side = HAND_INDEX_TO_SIDE[hand_idx]
        if not np.all(poses_m[hand_idx] == -1):
            active_hands[side] = hand_idx

    # Collect per-frame joint data
    all_joint_3d_cam = {side: np.full((num_frames, 21, 3), -1.0, dtype=np.float32)
                        for side in active_hands}
    frame_valid = np.ones(num_frames, dtype=bool)

    for frame_i in range(num_frames):
        label_path = os.path.join(camera_dir, f"label_{frame_i:06d}.npz")
        if not os.path.exists(label_path):
            frame_valid[frame_i] = False
            continue

        labels = dict(np.load(label_path, allow_pickle=True))
        if "hand_joints_3d" not in labels:
            frame_valid[frame_i] = False
            continue

        j3d = labels["hand_joints_3d"]  # (2, 21, 3)
        for side, hand_idx in active_hands.items():
            joints = j3d[hand_idx].astype(np.float32)
            if np.any(joints == -1):
                frame_valid[frame_i] = False
                break
            all_joint_3d_cam[side][frame_i] = joints

    # Filter to valid frames
    valid_indices = np.where(frame_valid)[0]
    M = len(valid_indices)

    transforms_dict = {}
    confidences_dict = {}
    identity = np.eye(4, dtype=np.float32)
    conf_ones = np.ones(M, dtype=np.float32)
    conf_zeros = np.zeros(M, dtype=np.float32)

    # Camera transform (static, repeated for valid frames)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = conf_zeros.copy()

    # Hand joints
    for side in ["left", "right"]:
        is_active = side in active_hands

        if is_active:
            hand_idx = active_hands[side]
            joint_3d_valid = all_joint_3d_cam[side][valid_indices]  # (M, 21, 3)

            # Camera-space joints → 4x4 transforms → world space
            all_transforms_cam = np.zeros((M, 21, 4, 4), dtype=np.float32)
            for i in range(M):
                all_transforms_cam[i] = joints_to_transforms(joint_3d_valid[i])
            all_transforms_world = cam_pose @ all_transforms_cam

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms_world[:, mano_idx]
                confidences_dict[name] = conf_ones.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = conf_zeros.copy()

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
                confidences_dict[name] = conf_ones.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = conf_zeros.copy()

    # Build MANO dicts for each active hand
    mano_dicts = []
    mano_betas = load_ho_cap_mano_betas(calibration_dir, subject_id)

    for side, hand_idx in active_hands.items():
        poses_valid = poses_m[hand_idx][valid_indices]  # (M, 51)
        go, hp, tr = convert_mano_axisangle_to_rotmat(poses_valid)

        # World-space 3D keypoints from transforms
        kpt3d = np.zeros((M, 21, 3), dtype=np.float32)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

        mano_dicts.append({
            "betas": mano_betas,
            "global_orient_worldspace": go,
            "hand_pose": hp,
            "transl_worldspace": tr,
            "kpt3d": kpt3d,
            "side": side,
        })

    return intrinsic, transforms_dict, confidences_dict, valid_indices, mano_dicts


def _verify_output(out_dir: str, seq_idx: int, cam_idx: int, expected_frames: int):
    """Verify the converted output."""
    prefix = f"{seq_idx:06d}"
    hdf5_path = os.path.join(out_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
    with h5py.File(hdf5_path, "r") as f:
        sample_key = list(f["transforms"].keys())[0]
        if sample_key == "gravity":
            sample_key = list(f["transforms"].keys())[1]
        hdf5_frames = f[f"transforms/{sample_key}"].shape[0]
        if hdf5_frames != expected_frames:
            print(f"  WARNING: HDF5 has {hdf5_frames} frames, expected {expected_frames}")

    video_path = os.path.join(out_dir, f"{prefix}_video_{cam_idx:02d}.mp4")
    if not os.path.exists(video_path):
        print(f"  WARNING: {os.path.basename(video_path)} not created")
        return
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_frames != expected_frames:
        print(f"  WARNING: {os.path.basename(video_path)} has {video_frames} frames, expected {expected_frames}")
    else:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  OK: {os.path.basename(video_path)} ({video_frames} frames, {size_mb:.1f} MB)")


def convert_ho_cap(src_dir: str, dst_dir: str, cameras: list = None,
                   fps: float = 30.0, max_samples: int = 0):
    """Convert HO-Cap sequences to egodex format, grouped by subject."""
    calibration_dir = os.path.join(src_dir, "calibration")
    os.makedirs(dst_dir, exist_ok=True)

    # Collect all subjects and their sequences
    subject_dirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and d.startswith("subject_")
    ])

    global_count = 0
    for subject_id in subject_dirs:
        subject_path = os.path.join(src_dir, subject_id)
        seq_dirs = sorted([
            d for d in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, d))
        ])

        out_dir = os.path.join(dst_dir, subject_id)
        os.makedirs(out_dir, exist_ok=True)

        for seq_idx, seq_name in enumerate(seq_dirs):
            if max_samples > 0 and global_count >= max_samples:
                break

            seq_path = os.path.join(subject_path, seq_name)
            meta_path = os.path.join(seq_path, "meta.yaml")
            if not os.path.exists(meta_path):
                continue
            meta = load_yaml(meta_path)

            poses_m_path = os.path.join(seq_path, "poses_m.npy")
            if not os.path.exists(poses_m_path):
                print(f"  Skipping {subject_id}/{seq_name}: poses_m.npy not found")
                continue
            poses_m = np.load(poses_m_path)  # (2, num_frames, 51)

            serials = meta["realsense"]["serials"]
            num_frames = meta["num_frames"]

            cam_list = cameras if cameras is not None else list(range(len(serials)))

            any_camera_ok = False
            for cam_idx in cam_list:
                if cam_idx >= len(serials):
                    print(f"  Skipping {subject_id}/{seq_name} cam {cam_idx}: out of range")
                    continue

                serial = serials[cam_idx]
                camera_dir = os.path.join(seq_path, serial)
                if not os.path.isdir(camera_dir):
                    print(f"  Skipping {subject_id}/{seq_name} cam {cam_idx}: dir not found")
                    continue

                intrinsic, transforms_dict, confidences_dict, valid_indices, mano_dicts = \
                    build_egodex_data_for_sequence(
                        seq_path, meta, calibration_dir, serial,
                        poses_m, num_frames, subject_id,
                    )

                n_valid = len(valid_indices)
                active_sides = [d["side"] for d in mano_dicts]
                print(f"[{subject_id}/{seq_idx:06d}] {seq_name} "
                      f"(cam={cam_idx:02d}, serial={serial}, frames={num_frames}, "
                      f"valid={n_valid}, hands={active_sides})")

                if n_valid == 0:
                    print(f"  Skipping: no valid frames")
                    continue

                prefix = f"{seq_idx:06d}"

                # Write one HDF5 per active hand (first hand as primary mano_dict)
                mano_dict = mano_dicts[0] if mano_dicts else None
                hdf5_path = os.path.join(out_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
                write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                                  confidences_dict, mano_dict=mano_dict)

                # Write additional MANO groups for multi-hand sequences
                if len(mano_dicts) > 1:
                    with h5py.File(hdf5_path, "a") as f:
                        for extra_mano in mano_dicts[1:]:
                            side = extra_mano["side"]
                            grp = f.create_group(f"mano_{side}")
                            grp.create_dataset("betas",
                                               data=extra_mano["betas"].astype(np.float32))
                            grp.create_dataset("global_orient_worldspace",
                                               data=extra_mano["global_orient_worldspace"].astype(np.float32))
                            grp.create_dataset("hand_pose",
                                               data=extra_mano["hand_pose"].astype(np.float32))
                            grp.create_dataset("transl_worldspace",
                                               data=extra_mano["transl_worldspace"].astype(np.float32))
                            grp.create_dataset("kpt3d",
                                               data=extra_mano["kpt3d"].astype(np.float32))

                # RGB video (valid frames only)
                all_color_paths = sorted([
                    os.path.join(camera_dir, f)
                    for f in os.listdir(camera_dir) if f.startswith("color_") and f.endswith(".jpg")
                ])
                valid_color_paths = [all_color_paths[i] for i in valid_indices
                                     if i < len(all_color_paths)]
                rgb_path = os.path.join(out_dir, f"{prefix}_video_{cam_idx:02d}.mp4")
                images_to_mp4(valid_color_paths, rgb_path, fps=fps)

                # Verify
                _verify_output(out_dir, seq_idx, cam_idx, n_valid)
                any_camera_ok = True

            if any_camera_ok:
                global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    cam_str = "all" if cameras is None else str(len(cameras))
    print(f"\nDone. Converted {global_count} sequences ({cam_str} cam(s) each) to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert HO-Cap to egodex format")
    parser.add_argument("--src", default="../HO-Cap/datasets",
                        help="HO-Cap dataset directory")
    parser.add_argument("--dst", default="CONVERTED/ho_cap",
                        help="Output directory")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Camera indices to extract (default: all 8 RealSense)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences to convert (0=all)")
    args = parser.parse_args()

    convert_ho_cap(args.src, args.dst, cameras=args.cameras,
                   fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
