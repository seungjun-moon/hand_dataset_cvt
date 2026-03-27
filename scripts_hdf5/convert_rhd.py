#!/usr/bin/env python3
"""
Convert RHD (Rendered Hand Dataset) synthetic data into egodex format.

RHD structure:
    rhd/data/synthetic_train_val/
        images/
            l{XX}/          (lighting conditions: l01..l30, skipping l19)
                cam{YY}/    (25 cameras per lighting)
                    handV2_..._{lXX}_{camYY}_.{ZZZZ}.png  (500 poses)
        3D_labels/
            camPosition.txt     (500 poses x 25 cameras x 7 params)
            handGestures.txt    (500 poses x 21 joints x 3 coords, in cm)
            val-camera.txt      (validation camera names)
        hand_3D_mesh/           (OBJ files, 500 poses)

Each lighting condition (lXX) is treated as a separate sequence.
Within a sequence, each camera (camYY) provides 500 consecutive frames.

Output structure:
    CONVERTED/rhd/
        l{XX}/
            {seq_idx:06d}_label_{cam_idx:02d}.hdf5
            {seq_idx:06d}_video_{cam_idx:02d}.mp4

Usage:
    python scripts/convert_rhd.py --src rhd/data --dst CONVERTED/rhd
    python scripts/convert_rhd.py --cameras 0 1 2 --lightings l01 l02
    python scripts/convert_rhd.py --max-samples 3
"""

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import images_to_mp4, write_egodex_hdf5
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

# RHD joint order (same as MANO 21-joint ordering):
#  0: palm (wrist)
#  1-4: thumb (CMC, MCP, IP, tip)
#  5-8: index (MCP, PIP, DIP, tip)
#  9-12: middle (MCP, PIP, DIP, tip)
#  13-16: ring (MCP, PIP, DIP, tip)
#  17-20: pinky (MCP, PIP, DIP, tip)

NUM_POSES = 500
NUM_CAMERAS = 25


def load_camera_params(cam_param_path):
    """Load camera parameters from camPosition.txt.

    Returns:
        (N_pose, N_cam, 7) array.
        Each entry: [focal_length, tx, ty, tz, euler_x, euler_y, euler_z]
    """
    all_camera_names = np.loadtxt(cam_param_path, usecols=(0,), dtype=str)
    num_cameras = len(np.unique(all_camera_names))
    all_camera_params = np.loadtxt(cam_param_path, usecols=(1, 2, 3, 4, 5, 6, 7))
    all_camera_params = all_camera_params.reshape((-1, num_cameras, 7))
    return all_camera_params


def load_global_pose3d(pose3d_gt_path):
    """Load global 3D hand pose ground truth from handGestures.txt.

    Returns:
        (N_pose, 21, 3) array in centimeters.
    """
    all_joint_names = np.loadtxt(pose3d_gt_path, usecols=(0,), dtype=str)
    num_joints = len(np.unique(all_joint_names))
    all_global_pose3d = np.loadtxt(pose3d_gt_path, usecols=(1, 2, 3))
    all_global_pose3d = all_global_pose3d.reshape((-1, num_joints, 3))
    return all_global_pose3d


def euler_xyz_to_rot_mx(euler_angle):
    """Convert xyz euler angles (degrees) to 3x3 rotation matrix."""
    rad = euler_angle * math.pi / 180.0
    sins = np.sin(rad)
    coss = np.cos(rad)
    rot_x = np.array([[1, 0, 0],
                       [0, coss[0], -sins[0]],
                       [0, sins[0], coss[0]]])
    rot_y = np.array([[coss[1], 0, sins[1]],
                       [0, 1, 0],
                       [-sins[1], 0, coss[1]]])
    rot_z = np.array([[coss[2], -sins[2], 0],
                       [sins[2], coss[2], 0],
                       [0, 0, 1]])
    return rot_z.dot(rot_y).dot(rot_x)


def build_cam_extrinsic(cam_param):
    """Build a 4x4 world-to-camera extrinsic matrix from RHD camera params.

    Args:
        cam_param: (7,) [focal_length, tx, ty, tz, euler_x, euler_y, euler_z]

    Returns:
        cam_extrinsic: (4, 4) world-to-camera transform
        cam_pose: (4, 4) camera-to-world transform (camera pose in world)
    """
    # RHD stores translation in cm; convert to meters.
    translation = cam_param[1:4] / 100.0
    theta = cam_param[4:]
    rot_mx = euler_xyz_to_rot_mx(theta)
    # RHD applies a sign flip on Y and Z axes
    aux_mx = np.eye(3, dtype=np.float64)
    aux_mx[1, 1] = -1.0
    aux_mx[2, 2] = -1.0
    rot_mx = rot_mx.dot(aux_mx)

    # RHD row-vector convention: cam_point = (world_point - translation) @ rot_mx
    # In column convention: cam_point = rot_mx^T @ (world_point - translation)
    # So world-to-camera: R_w2c = rot_mx^T, t_w2c = -rot_mx^T @ translation
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rot_mx.T.astype(np.float32)
    extrinsic[:3, 3] = (-rot_mx.T @ translation).astype(np.float32)

    # Camera pose (camera-to-world): R_c2w = rot_mx, t_c2w = translation
    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, :3] = rot_mx.astype(np.float32)
    cam_pose[:3, 3] = translation.astype(np.float32)

    return extrinsic, cam_pose


def build_intrinsic(cam_param, im_width, im_height):
    """Build a 3x3 intrinsic matrix from RHD camera params.

    RHD uses a single focal length with principal point at image center.
    """
    fl = cam_param[0]
    K = np.array([
        [fl, 0.0, im_width / 2.0],
        [0.0, fl, im_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K


def transform_global_to_cam(global_3d, cam_param):
    """Transform 3D points from global to camera coordinate system.

    Args:
        global_3d: (N, 3) points in world space (cm)
        cam_param: (7,) camera parameters

    Returns:
        (N, 3) points in camera space (cm)
    """
    translation = cam_param[1:4]
    theta = cam_param[4:]
    rot_mx = euler_xyz_to_rot_mx(theta)
    aux_mx = np.eye(3, dtype=np.float64)
    aux_mx[1, 1] = -1.0
    aux_mx[2, 2] = -1.0
    rot_mx = rot_mx.dot(aux_mx)

    pose3d = global_3d - translation
    camera_3d = pose3d.dot(rot_mx)
    return camera_3d


def get_image_paths(images_dir, lighting, cam_name):
    """Get sorted list of image paths for a given lighting and camera."""
    cam_dir = os.path.join(images_dir, lighting, cam_name)
    paths = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    return paths


def build_egodex_data_for_sequence(
    all_camera_params,
    all_global_pose3d,
    cam_idx,
    im_width,
    im_height,
):
    """Build world-space transforms and confidences for one camera across all poses.

    In RHD, each camera has fixed parameters per pose (camera params vary per pose).
    We treat the 500 poses as a temporal sequence.

    Args:
        all_camera_params: (500, 25, 7) camera parameters
        all_global_pose3d: (500, 21, 3) global joint positions (cm)
        cam_idx: camera index (0-24)
        im_width: image width
        im_height: image height

    Returns:
        intrinsics: (500, 3, 3) per-frame intrinsic matrices
        transforms_dict: {joint_name: (500, 4, 4)} world-space transforms
        confidences_dict: {joint_name: (500,)}
    """
    N = all_global_pose3d.shape[0]  # 500
    identity = np.eye(4, dtype=np.float32)

    transforms_dict = {}
    confidences_dict = {}

    # Per-frame intrinsics (focal length varies per pose for some cameras)
    intrinsics = np.zeros((N, 3, 3), dtype=np.float32)

    # Compute world-space transforms for all frames
    all_transforms_world = np.zeros((N, 21, 4, 4), dtype=np.float32)
    all_cam_poses = np.zeros((N, 4, 4), dtype=np.float32)
    # RHD stores joint positions in cm; convert to meters.
    all_joint_3d_world = all_global_pose3d.astype(np.float32) / 100.0  # (500, 21, 3) cm -> m

    for i in range(N):
        cam_param = all_camera_params[i, cam_idx]
        intrinsics[i] = build_intrinsic(cam_param, im_width, im_height)
        _, cam_pose = build_cam_extrinsic(cam_param)
        all_cam_poses[i] = cam_pose

        # Compute transforms from joint positions in world space (already in meters)
        joint_3d = all_joint_3d_world[i]  # (21, 3) in meters
        transforms_local = joints_to_transforms(joint_3d)
        all_transforms_world[i] = transforms_local  # already in world space

    # Camera transform
    transforms_dict["camera"] = all_cam_poses

    # Body joints: not available
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (N, 1, 1))
        confidences_dict[name] = np.zeros(N, dtype=np.float32)

    conf = np.ones(N, dtype=np.float32)

    # RHD is a right-hand dataset (despite "_L" suffix in joint names,
    # the rendered hand is right hand based on the model name "rgt01")
    active_side = "right"

    for side in ["left", "right"]:
        is_active = (side == active_side)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms_world[:, mano_idx]
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (N, 1, 1))
                confidences_dict[name] = np.zeros(N, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc_world = np.zeros((N, 4, 4), dtype=np.float32)
                for i in range(N):
                    pos = interpolate_joint(
                        all_joint_3d_world[i], idx_a, idx_b, alpha=0.3
                    )
                    direction = all_joint_3d_world[i, idx_b] - all_joint_3d_world[i, idx_a]
                    mc_world[i] = make_transform(pos, direction)
                transforms_dict[name] = mc_world
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(identity, (N, 1, 1))
                confidences_dict[name] = np.zeros(N, dtype=np.float32)

    # World-space 3D keypoints
    kpt3d = all_joint_3d_world.copy()  # (N, 21, 3)

    mano_dict = {
        "betas": np.zeros(10, dtype=np.float32),  # unknown, set to zero
        "global_orient_worldspace": np.zeros((N, 3, 3), dtype=np.float32),  # not available
        "hand_pose": np.zeros((N, 15, 3, 3), dtype=np.float32),  # not available
        "transl_worldspace": all_joint_3d_world[:, 0],  # wrist position
        "kpt3d": kpt3d,
        "side": active_side,
    }

    return intrinsics, transforms_dict, confidences_dict, mano_dict


def convert_rhd(src_dir, dst_dir, lightings=None, cameras=None,
                fps=30.0, max_samples=0):
    """Convert RHD synthetic sequences to egodex format.

    Each lighting condition is a separate cluster. Within each lighting,
    each camera produces a 500-frame sequence.
    """
    synthetic_dir = os.path.join(src_dir, "synthetic_train_val")
    images_dir = os.path.join(synthetic_dir, "images")
    labels_dir = os.path.join(synthetic_dir, "3D_labels")

    cam_param_path = os.path.join(labels_dir, "camPosition.txt")
    pose3d_path = os.path.join(labels_dir, "handGestures.txt")

    print("Loading camera parameters...")
    all_camera_params = load_camera_params(cam_param_path)
    print(f"  Camera params shape: {all_camera_params.shape}")

    print("Loading 3D pose ground truth...")
    all_global_pose3d = load_global_pose3d(pose3d_path)
    print(f"  Pose3d shape: {all_global_pose3d.shape}")

    # Discover available lightings
    available_lightings = sorted([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d)) and d.startswith("l")
    ])
    if lightings is not None:
        available_lightings = [l for l in available_lightings if l in lightings]

    print(f"  Available lightings: {len(available_lightings)}")

    # Get image dimensions from first image
    sample_img_path = get_image_paths(
        images_dir, available_lightings[0], "cam01"
    )[0]
    sample_img = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
    im_height, im_width = sample_img.shape[:2]
    print(f"  Image size: {im_width}x{im_height}")

    os.makedirs(dst_dir, exist_ok=True)

    cam_indices = cameras if cameras is not None else list(range(NUM_CAMERAS))
    global_count = 0

    for lighting in available_lightings:
        if max_samples > 0 and global_count >= max_samples:
            break

        lighting_dir = os.path.join(dst_dir, lighting)
        os.makedirs(lighting_dir, exist_ok=True)

        seq_idx = 0  # one sequence per lighting

        for cam_idx in cam_indices:
            cam_name = f"cam{cam_idx + 1:02d}"
            cam_dir = os.path.join(images_dir, lighting, cam_name)
            if not os.path.isdir(cam_dir):
                print(f"  Skipping {lighting}/{cam_name}: directory not found")
                continue

            image_paths = get_image_paths(images_dir, lighting, cam_name)
            if len(image_paths) != NUM_POSES:
                print(f"  WARNING: {lighting}/{cam_name} has {len(image_paths)} images, expected {NUM_POSES}")
                if len(image_paths) == 0:
                    continue

            intrinsics, transforms_dict, confidences_dict, mano_dict = \
                build_egodex_data_for_sequence(
                    all_camera_params, all_global_pose3d,
                    cam_idx, im_width, im_height,
                )

            n_frames = len(image_paths)
            print(f"[{lighting}/{seq_idx:06d}] {cam_name} "
                  f"(cam={cam_idx:02d}, frames={n_frames})")

            prefix = f"{seq_idx:06d}"

            # HDF5 label
            hdf5_path = os.path.join(lighting_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
            write_egodex_hdf5(
                hdf5_path, intrinsics, transforms_dict,
                confidences_dict, mano_dict=mano_dict,
            )

            # RGB video
            rgb_path = os.path.join(lighting_dir, f"{prefix}_video_{cam_idx:02d}.mp4")
            images_to_mp4(image_paths, rgb_path, fps=fps)

            # Verify
            size_mb = os.path.getsize(rgb_path) / (1024 * 1024)
            print(f"  OK: {os.path.basename(rgb_path)} ({n_frames} frames, {size_mb:.1f} MB)")

        global_count += 1

    cam_str = "all" if cameras is None else str(len(cameras))
    print(f"\nDone. Converted {global_count} lighting(s) ({cam_str} cam(s) each) to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert RHD to egodex format")
    parser.add_argument("--src", default="/rlwrld3/home/seungjun/rhd/data",
                        help="RHD data source directory")
    parser.add_argument("--dst", default="/rlwrld3/home/seungjun/hand_dataset_cvt/CONVERTED/rhd",
                        help="Output directory")
    parser.add_argument("--lightings", nargs="+", default=None,
                        help="Specific lightings to convert (e.g. l01 l02). Default: all")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Camera indices (0-24) to extract. Default: all")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS (default: 30)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max lighting conditions to convert (0=all)")
    args = parser.parse_args()

    convert_rhd(
        args.src, args.dst,
        lightings=args.lightings,
        cameras=args.cameras,
        fps=args.fps,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
