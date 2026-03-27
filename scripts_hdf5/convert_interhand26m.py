#!/usr/bin/env python3
"""
Convert InterHand2.6M dataset (5fps) into egodex format.

InterHand2.6M structure:
    ROOT/
        images/train/CaptureXX/seq_name/camXXXXXX/imageXXXX.jpg
        annotations/train/
            InterHand2.6M_train_data.json       (COCO-format: images + annotations)
            InterHand2.6M_train_camera.json      (per-capture, per-camera intrinsics/extrinsics)
            InterHand2.6M_train_joint_3d.json    (per-capture, per-frame world-space 3D joints)
            InterHand2.6M_train_MANO_NeuralAnnot.json  (per-capture, per-frame MANO params)

Annotation keys:
    cameras[capture_id]['campos'/'camrot'/'focal'/'princpt'][cam_id]
    joints[capture_id][frame_idx]['world_coord' (42,3), 'joint_valid' (42,), 'seq']
    mano_params[capture_id][frame_idx]['right'/'left'] -> {'pose' (48,), 'shape' (10,), 'trans' (3,)}

InterHand2.6M has 42 joints: 21 right (indices 0-20) + 21 left (indices 21-41).
Joint ordering per hand follows MANO convention:
    0: Wrist, 1-4: Thumb (CMC/MCP/IP/Tip), 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Little

Conversion strategy:
    - Per capture, sort sequences alphabetically and split into chunks of ~28.
    - Per (capture, chunk), create one output sequence folder.
    - Within each folder, each camera gets an indexed pair of (label HDF5, video MP4).
    - Frames from all sequences in the chunk are concatenated in sequence order,
      then frame order within each sequence.
    - Camera extrinsics convert world-space joints to camera-space for transforms.

Output structure:
    CONVERTED/interhand26m/
        capture_XXX_chunk_XXX/
            XXXXXX_label_00.hdf5   # per camera
            XXXXXX_video_00.mp4

Usage:
    python scripts/convert_interhand26m.py --src ../InterWild/data/InterHand26M --dst CONVERTED/interhand26m
    python scripts/convert_interhand26m.py --split train --chunk-size 28 --max-samples 3
"""

import argparse
import json
import math
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import (
    images_to_mp4,
    write_egodex_hdf5,
)
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

# InterHand2.6M joint indices per hand (21 joints each)
# Right hand: 0-20, Left hand: 21-41
# Per-hand ordering: 0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Little
# This matches MANO joint convention.
IH26M_RIGHT_INDICES = list(range(0, 21))
IH26M_LEFT_INDICES = list(range(21, 42))
IH26M_WRIST_RIGHT = 20  # R_Wrist in InterHand26M ordering
IH26M_WRIST_LEFT = 41   # L_Wrist in InterHand26M ordering

# InterHand2.6M joint names (from InterWild code)
IH26M_JOINTS = (
    'R_Thumb_4', 'R_Thumb_3', 'R_Thumb_2', 'R_Thumb_1',
    'R_Index_4', 'R_Index_3', 'R_Index_2', 'R_Index_1',
    'R_Middle_4', 'R_Middle_3', 'R_Middle_2', 'R_Middle_1',
    'R_Ring_4', 'R_Ring_3', 'R_Ring_2', 'R_Ring_1',
    'R_Pinky_4', 'R_Pinky_3', 'R_Pinky_2', 'R_Pinky_1',
    'R_Wrist',
    'L_Thumb_4', 'L_Thumb_3', 'L_Thumb_2', 'L_Thumb_1',
    'L_Index_4', 'L_Index_3', 'L_Index_2', 'L_Index_1',
    'L_Middle_4', 'L_Middle_3', 'L_Middle_2', 'L_Middle_1',
    'L_Ring_4', 'L_Ring_3', 'L_Ring_2', 'L_Ring_1',
    'L_Pinky_4', 'L_Pinky_3', 'L_Pinky_2', 'L_Pinky_1',
    'L_Wrist',
)

# Mapping from InterHand2.6M per-hand index to MANO joint index.
# IH26M per-hand: 0-3=Thumb(tip->CMC), 4-7=Index(tip->MCP), ..., 20=Wrist
# MANO: 0=Wrist, 1-4=Thumb(CMC->tip), 5-8=Index(MCP->tip), ...
# So IH26M needs reordering: wrist=20->0, thumb_4=0->4, thumb_3=1->3, thumb_2=2->2, thumb_1=3->1
IH26M_TO_MANO = [None] * 21
IH26M_TO_MANO[20] = 0   # Wrist
# Thumb: IH26M 0(tip)->MANO 4, 1->3, 2->2, 3->1(CMC)
IH26M_TO_MANO[0] = 4; IH26M_TO_MANO[1] = 3; IH26M_TO_MANO[2] = 2; IH26M_TO_MANO[3] = 1
# Index: IH26M 4(tip)->MANO 8, 5->7, 6->6, 7->5(MCP)
IH26M_TO_MANO[4] = 8; IH26M_TO_MANO[5] = 7; IH26M_TO_MANO[6] = 6; IH26M_TO_MANO[7] = 5
# Middle: IH26M 8(tip)->MANO 12, 9->11, 10->10, 11->9(MCP)
IH26M_TO_MANO[8] = 12; IH26M_TO_MANO[9] = 11; IH26M_TO_MANO[10] = 10; IH26M_TO_MANO[11] = 9
# Ring: IH26M 12(tip)->MANO 16, 13->15, 14->14, 15->13(MCP)
IH26M_TO_MANO[12] = 16; IH26M_TO_MANO[13] = 15; IH26M_TO_MANO[14] = 14; IH26M_TO_MANO[15] = 13
# Pinky: IH26M 16(tip)->MANO 20, 17->19, 18->18, 19->17(MCP)
IH26M_TO_MANO[16] = 20; IH26M_TO_MANO[17] = 19; IH26M_TO_MANO[18] = 18; IH26M_TO_MANO[19] = 17


def reorder_ih26m_to_mano(joints_ih26m_21: np.ndarray) -> np.ndarray:
    """Reorder (N, 21, 3) or (21, 3) IH26M per-hand joints to MANO ordering."""
    out = np.zeros_like(joints_ih26m_21)
    for ih_idx, mano_idx in enumerate(IH26M_TO_MANO):
        if joints_ih26m_21.ndim == 3:
            out[:, mano_idx] = joints_ih26m_21[:, ih_idx]
        else:
            out[mano_idx] = joints_ih26m_21[ih_idx]
    return out


def get_cam_extrinsic(cameras: dict, capture_id: str, cam_id: str) -> np.ndarray:
    """Build a 4x4 world-to-camera extrinsic matrix.

    InterHand2.6M stores campos (camera position in world) and camrot (world-to-camera rotation).
    Convention: t = -R @ campos, so the extrinsic [R|t] maps world->camera.

    Returns cam_pose (4x4) that maps camera->world (inverse of extrinsic).
    """
    # InterHand2.6M stores campos/camrot in mm; convert to meters.
    campos = np.array(cameras[capture_id]['campos'][cam_id], dtype=np.float32).reshape(3) / 1000.0
    camrot = np.array(cameras[capture_id]['camrot'][cam_id], dtype=np.float32).reshape(3, 3)

    # camrot is world-to-camera rotation, campos is camera position in world
    # world-to-camera: p_cam = R @ p_world + t, where t = -R @ campos
    R = camrot
    t = -R @ campos

    # cam_pose: camera-to-world (inverse extrinsic)
    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = campos  # R.T @ (-t) = R.T @ R @ campos = campos

    return cam_pose


def get_cam_intrinsic(cameras: dict, capture_id: str, cam_id: str) -> np.ndarray:
    """Build (3, 3) intrinsic matrix from focal length and principal point."""
    focal = np.array(cameras[capture_id]['focal'][cam_id], dtype=np.float32)
    princpt = np.array(cameras[capture_id]['princpt'][cam_id], dtype=np.float32)
    K = np.array([
        [focal[0], 0, princpt[0]],
        [0, focal[1], princpt[1]],
        [0, 0, 1],
    ], dtype=np.float32)
    return K


def convert_mano_axisangle_to_rotmat(pose: np.ndarray):
    """Convert (48,) axis-angle MANO pose to rotation matrices.

    Returns:
        global_orient: (3, 3) rotation matrix.
        hand_pose: (15, 3, 3) per-joint rotation matrices.
    """
    global_orient, _ = cv2.Rodrigues(pose[:3].astype(np.float64))
    global_orient = global_orient.astype(np.float32)
    hand_pose = np.zeros((15, 3, 3), dtype=np.float32)
    for j in range(15):
        aa = pose[3 + j * 3:3 + (j + 1) * 3].astype(np.float64)
        hand_pose[j], _ = cv2.Rodrigues(aa)
    return global_orient, hand_pose


def build_ordered_frames(
    joints: dict,
    mano_params: dict,
    capture_id: str,
    seq_names: list,
    img_dir: str,
    cam_ids: list,
):
    """Collect ordered frames for a chunk of sequences.

    For each sequence in order, collect frame_idx values sorted numerically.
    Returns:
        frame_list: list of (frame_idx_str, seq_name) in concatenated order
        cam_frame_images: {cam_id: [image_path, ...]} for each camera
    """
    frame_list = []
    for seq_name in seq_names:
        # Find frames belonging to this sequence from joints data
        seq_frames = []
        for frame_idx_str, frame_data in joints[capture_id].items():
            if frame_data.get('seq') == seq_name:
                seq_frames.append(frame_idx_str)
        seq_frames.sort(key=lambda x: int(x))
        for fid in seq_frames:
            frame_list.append((fid, seq_name))

    # Build image paths per camera
    cam_frame_images = {}
    for cam_id in cam_ids:
        paths = []
        for frame_idx_str, seq_name in frame_list:
            img_path = os.path.join(
                img_dir, seq_name, f"cam{cam_id}", f"image{frame_idx_str}.jpg"
            )
            paths.append(img_path)
        cam_frame_images[cam_id] = paths

    return frame_list, cam_frame_images


def build_egodex_data_for_camera(
    cameras: dict,
    joints: dict,
    mano_params: dict,
    capture_id: str,
    cam_id: str,
    frame_list: list,
):
    """Build world-space transforms and MANO data for one camera in a chunk.

    Args:
        frame_list: list of (frame_idx_str, seq_name) in order.

    Returns:
        intrinsic: (3, 3)
        transforms_dict: {joint_name: (M, 4, 4)}
        confidences_dict: {joint_name: (M,)}
        mano_dicts: list of mano_dict (one per active side)
    """
    M = len(frame_list)
    identity = np.eye(4, dtype=np.float32)
    intrinsic = get_cam_intrinsic(cameras, capture_id, cam_id)
    cam_pose = get_cam_extrinsic(cameras, capture_id, cam_id)

    transforms_dict = {}
    confidences_dict = {}

    # Camera transform (static for this camera)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Collect per-frame data
    # World-space joints: (M, 21, 3) per side, in MANO ordering
    right_joints_world = np.zeros((M, 21, 3), dtype=np.float32)
    left_joints_world = np.zeros((M, 21, 3), dtype=np.float32)
    right_valid = np.zeros(M, dtype=np.float32)
    left_valid = np.zeros(M, dtype=np.float32)
    # Per-joint validity: (M, 21) per side, in MANO ordering
    right_joint_valid = np.zeros((M, 21), dtype=np.float32)
    left_joint_valid = np.zeros((M, 21), dtype=np.float32)

    # MANO params per side
    right_poses = np.zeros((M, 48), dtype=np.float64)
    right_trans = np.zeros((M, 3), dtype=np.float64)
    right_betas = None
    left_poses = np.zeros((M, 48), dtype=np.float64)
    left_trans = np.zeros((M, 3), dtype=np.float64)
    left_betas = None

    for i, (frame_idx_str, seq_name) in enumerate(frame_list):
        frame_joints = joints[capture_id][frame_idx_str]
        # InterHand2.6M stores world_coord in mm; convert to meters.
        world_coord = np.array(frame_joints['world_coord'], dtype=np.float32).reshape(42, 3) / 1000.0
        joint_valid = np.array(frame_joints['joint_valid'], dtype=np.float32).flatten()

        # Right hand (IH26M indices 0-20)
        right_world_ih = world_coord[:21]   # (21, 3) in IH26M order
        right_jv = joint_valid[:21]
        if np.any(right_jv > 0):
            right_joints_world[i] = reorder_ih26m_to_mano(right_world_ih)
            right_valid[i] = 1.0
            # Reorder per-joint validity to MANO ordering
            for ih_idx, mano_idx in enumerate(IH26M_TO_MANO):
                right_joint_valid[i, mano_idx] = right_jv[ih_idx]

        # Left hand (IH26M indices 21-41)
        left_world_ih = world_coord[21:]    # (21, 3) in IH26M order
        left_jv = joint_valid[21:]
        if np.any(left_jv > 0):
            left_joints_world[i] = reorder_ih26m_to_mano(left_world_ih)
            left_valid[i] = 1.0
            for ih_idx, mano_idx in enumerate(IH26M_TO_MANO):
                left_joint_valid[i, mano_idx] = left_jv[ih_idx]

        # MANO params
        try:
            mano_frame = mano_params[capture_id][frame_idx_str]
        except KeyError:
            mano_frame = {'right': None, 'left': None}

        if mano_frame.get('right') is not None:
            right_poses[i] = np.array(mano_frame['right']['pose'], dtype=np.float64)
            right_trans[i] = np.array(mano_frame['right']['trans'], dtype=np.float64) / 1000.0
            if right_betas is None:
                right_betas = np.array(mano_frame['right']['shape'], dtype=np.float32)

        if mano_frame.get('left') is not None:
            left_poses[i] = np.array(mano_frame['left']['pose'], dtype=np.float64)
            left_trans[i] = np.array(mano_frame['left']['trans'], dtype=np.float64) / 1000.0
            if left_betas is None:
                left_betas = np.array(mano_frame['left']['shape'], dtype=np.float32)

    # Interpolate single-frame joint gaps; mask entire hand for 2+ consecutive gaps
    for joints_world, jv, hand_valid in [
        (right_joints_world, right_joint_valid, right_valid),
        (left_joints_world, left_joint_valid, left_valid),
    ]:
        for j in range(21):
            col = jv[:, j]  # (M,) validity for this joint
            i = 0
            while i < M:
                if col[i] > 0:
                    i += 1
                    continue
                # Find the run of invalid frames for this joint
                run_start = i
                while i < M and col[i] == 0:
                    i += 1
                run_end = i  # exclusive
                run_len = run_end - run_start

                if run_len == 1 and run_start > 0 and run_end < M \
                        and col[run_start - 1] > 0 and col[run_end] > 0 \
                        and hand_valid[run_start] > 0:
                    # Single-frame gap with valid neighbors: interpolate
                    joints_world[run_start, j] = 0.5 * (
                        joints_world[run_start - 1, j] +
                        joints_world[run_end, j]
                    )
                    jv[run_start, j] = 1.0
                else:
                    # 2+ consecutive gap: mask out the entire hand for these frames
                    for k in range(run_start, run_end):
                        hand_valid[k] = 0.0

    # Build transforms for each hand
    has_right = np.any(right_valid > 0)
    has_left = np.any(left_valid > 0)

    for side, joints_world, valid_mask in [
        ("right", right_joints_world, right_valid),
        ("left", left_joints_world, left_valid),
    ]:
        is_active = (side == "right" and has_right) or (side == "left" and has_left)

        if is_active:
            # Compute world-space transforms from joint positions
            all_transforms_world = np.zeros((M, 21, 4, 4), dtype=np.float32)
            for i in range(M):
                if valid_mask[i] > 0:
                    all_transforms_world[i] = joints_to_transforms(joints_world[i])
                else:
                    all_transforms_world[i] = np.tile(identity, (21, 1, 1))

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms_world[:, mano_idx]
                confidences_dict[name] = valid_mask.copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc = np.zeros((M, 4, 4), dtype=np.float32)
                mc_conf = np.zeros(M, dtype=np.float32)
                for i in range(M):
                    if valid_mask[i] > 0:
                        pos = interpolate_joint(joints_world[i], idx_a, idx_b, alpha=0.3)
                        direction = joints_world[i, idx_b] - joints_world[i, idx_a]
                        mc[i] = make_transform(pos, direction)
                        mc_conf[i] = 1.0
                    else:
                        mc[i] = identity
                transforms_dict[name] = mc
                confidences_dict[name] = mc_conf
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Build MANO dicts
    cam_R = cam_pose[:3, :3]  # camera-to-world rotation
    cam_t = cam_pose[:3, 3]   # camera-to-world translation

    mano_dicts = []
    for side, poses, trans, betas, joints_world, valid_mask in [
        ("right", right_poses, right_trans, right_betas, right_joints_world, right_valid),
        ("left", left_poses, left_trans, left_betas, left_joints_world, left_valid),
    ]:
        if not np.any(valid_mask > 0):
            continue
        if betas is None:
            betas = np.zeros(10, dtype=np.float32)

        global_orients = np.zeros((M, 3, 3), dtype=np.float32)
        hand_poses_rot = np.zeros((M, 15, 3, 3), dtype=np.float32)

        for i in range(M):
            if valid_mask[i] > 0 and np.any(poses[i] != 0):
                go, hp = convert_mano_axisangle_to_rotmat(poses[i])
                global_orients[i] = go
                hand_poses_rot[i] = hp
            else:
                global_orients[i] = np.eye(3, dtype=np.float32)
                for j in range(15):
                    hand_poses_rot[i, j] = np.eye(3, dtype=np.float32)

        # World-space 3D keypoints from transforms
        kpt3d = np.zeros((M, 21, 3), dtype=np.float32)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

        mano_dicts.append({
            "betas": betas,
            "global_orient_worldspace": global_orients,
            "hand_pose": hand_poses_rot,
            "transl_worldspace": trans.astype(np.float32),
            "kpt3d": kpt3d,
            "side": side,
        })

    return intrinsic, transforms_dict, confidences_dict, mano_dicts


def _verify_output(out_dir: str, cam_idx: int, expected_frames: int):
    """Verify the converted output."""
    prefix = f"{cam_idx:06d}"
    hdf5_path = os.path.join(out_dir, f"{prefix}_label_00.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  WARNING: {os.path.basename(hdf5_path)} not created")
        return
    with h5py.File(hdf5_path, "r") as f:
        sample_key = list(f["transforms"].keys())[0]
        if sample_key == "gravity":
            sample_key = list(f["transforms"].keys())[1]
        hdf5_frames = f[f"transforms/{sample_key}"].shape[0]
        if hdf5_frames != expected_frames:
            print(f"  WARNING: HDF5 has {hdf5_frames} frames, expected {expected_frames}")

    video_path = os.path.join(out_dir, f"{prefix}_video_00.mp4")
    if not os.path.exists(video_path):
        print(f"  WARNING: {os.path.basename(video_path)} not created")
        return
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_frames != expected_frames:
        print(f"  WARNING: video has {video_frames} frames, expected {expected_frames}")
    else:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  OK: cam {cam_idx} ({video_frames} frames, {size_mb:.1f} MB)")


def convert_interhand26m(
    src_dir: str,
    dst_dir: str,
    split: str = "train",
    chunk_size: int = 28,
    fps: float = 5.0,
    max_samples: int = 0,
):
    """Convert InterHand2.6M sequences to egodex format."""
    os.makedirs(dst_dir, exist_ok=True)

    annot_dir = os.path.join(src_dir, "annotations", split)
    img_base = os.path.join(src_dir, "images", split)

    # For the images path, also check the alternative structure
    if not os.path.isdir(img_base):
        alt_img_base = os.path.join(
            src_dir, "InterHand2.6M_5fps_batch1", "images", split)
        if os.path.isdir(alt_img_base):
            img_base = alt_img_base
        else:
            raise FileNotFoundError(
                f"Image directory not found at {img_base} or {alt_img_base}")

    print(f"Loading annotations from {annot_dir} ...")
    prefix = f"InterHand2.6M_{split}"

    print("  Loading camera.json ...")
    with open(os.path.join(annot_dir, f"{prefix}_camera.json")) as f:
        cameras = json.load(f)

    print("  Loading joint_3d.json ...")
    with open(os.path.join(annot_dir, f"{prefix}_joint_3d.json")) as f:
        joints = json.load(f)

    print("  Loading MANO_NeuralAnnot.json ...")
    with open(os.path.join(annot_dir, f"{prefix}_MANO_NeuralAnnot.json")) as f:
        mano_params = json.load(f)

    print("Annotations loaded.")

    # Discover captures from image directory
    capture_dirs = sorted([
        d for d in os.listdir(img_base)
        if os.path.isdir(os.path.join(img_base, d)) and d.startswith("Capture")
    ])

    global_count = 0

    for capture_dir_name in capture_dirs:
        capture_id = capture_dir_name.replace("Capture", "")
        capture_path = os.path.join(img_base, capture_dir_name)

        if capture_id not in cameras:
            print(f"Skipping {capture_dir_name}: no camera data")
            continue

        # Get sorted list of sequences
        seq_names = sorted([
            d for d in os.listdir(capture_path)
            if os.path.isdir(os.path.join(capture_path, d))
        ])

        if not seq_names:
            continue

        # Get camera IDs for this capture
        cam_ids = sorted(cameras[capture_id]['campos'].keys())

        # Split sequences into chunks
        n_chunks = max(1, math.ceil(len(seq_names) / chunk_size))
        for chunk_idx in range(n_chunks):
            if max_samples > 0 and global_count >= max_samples:
                break

            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(seq_names))
            chunk_seqs = seq_names[chunk_start:chunk_end]

            chunk_name = f"capture_{int(capture_id):03d}_chunk_{chunk_idx:03d}"
            chunk_out_dir = os.path.join(dst_dir, chunk_name)

            print(f"\n[{chunk_name}] {capture_dir_name} seqs {chunk_start}-{chunk_end-1} "
                  f"({len(chunk_seqs)} seqs, {len(cam_ids)} cameras)")

            # Build ordered frame list for this chunk
            frame_list, cam_frame_images = build_ordered_frames(
                joints, mano_params, capture_id, chunk_seqs,
                capture_path, cam_ids,
            )

            if not frame_list:
                print("  Skipping: no frames")
                continue

            n_frames = len(frame_list)
            print(f"  Total frames: {n_frames}")

            os.makedirs(chunk_out_dir, exist_ok=True)

            for cam_idx, cam_id in enumerate(cam_ids):
                # Check that image files exist for this camera
                image_paths = cam_frame_images[cam_id]
                existing_paths = [p for p in image_paths if os.path.exists(p)]
                if not existing_paths:
                    continue

                # Filter frame_list to only frames with existing images
                valid_frame_indices = [
                    i for i, p in enumerate(image_paths) if os.path.exists(p)
                ]
                valid_frame_list = [frame_list[i] for i in valid_frame_indices]
                valid_image_paths = [image_paths[i] for i in valid_frame_indices]

                if not valid_frame_list:
                    continue

                # Build egodex data for this camera
                intrinsic, transforms_dict, confidences_dict, mano_dicts = \
                    build_egodex_data_for_camera(
                        cameras, joints, mano_params,
                        capture_id, cam_id, valid_frame_list,
                    )

                prefix_str = f"{cam_idx:06d}"

                # Write HDF5
                mano_dict = mano_dicts[0] if mano_dicts else None
                hdf5_path = os.path.join(chunk_out_dir, f"{prefix_str}_label_00.hdf5")
                write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                                  confidences_dict, mano_dict=mano_dict)

                # Write additional MANO groups
                if len(mano_dicts) > 1:
                    with h5py.File(hdf5_path, "a") as f:
                        for extra_mano in mano_dicts[1:]:
                            side = extra_mano["side"]
                            grp = f.create_group(f"mano_{side}")
                            grp.create_dataset(
                                "betas", data=extra_mano["betas"].astype(np.float32))
                            grp.create_dataset(
                                "global_orient_worldspace",
                                data=extra_mano["global_orient_worldspace"].astype(np.float32))
                            grp.create_dataset(
                                "hand_pose",
                                data=extra_mano["hand_pose"].astype(np.float32))
                            grp.create_dataset(
                                "transl_worldspace",
                                data=extra_mano["transl_worldspace"].astype(np.float32))
                            grp.create_dataset(
                                "kpt3d",
                                data=extra_mano["kpt3d"].astype(np.float32))

                # RGB video
                video_path = os.path.join(chunk_out_dir, f"{prefix_str}_video_00.mp4")
                images_to_mp4(valid_image_paths, video_path, fps=fps)

                # Verify
                _verify_output(chunk_out_dir, cam_idx, len(valid_image_paths))

            global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    print(f"\nDone. Converted {global_count} chunks to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert InterHand2.6M to egodex format")
    parser.add_argument(
        "--src", default="../InterWild/data/InterHand26M",
        help="InterHand2.6M dataset directory")
    parser.add_argument(
        "--dst", default="CONVERTED/interhand26m",
        help="Output directory")
    parser.add_argument(
        "--split", default="train",
        help="Data split to convert (train/val/test)")
    parser.add_argument(
        "--chunk-size", type=int, default=28,
        help="Number of sequences per chunk (default: 28)")
    parser.add_argument(
        "--fps", type=float, default=5.0,
        help="Video FPS (default: 5.0, matching 5fps subset)")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max chunks to convert (0=all)")
    args = parser.parse_args()

    convert_interhand26m(
        args.src, args.dst,
        split=args.split,
        chunk_size=args.chunk_size,
        fps=args.fps,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
