#!/usr/bin/env python3
"""
Convert ReInterHand dataset into egodex format.

ReInterHand structure (per capture directory):
    m--YYYYMMDD--HHMM--XXXXXX--pilot--ProjectName--Hands--two-hands/
        frame_list.txt              (seq_name frame_id per line, filtered subset)
        frame_list_orig.txt         (frame_id per line, all original frames)
        Mugsy_cameras/
            cam_params.json         (per-camera: focal, princpt, R, t)
            envmap_per_frame/
                images/
                    {cam_id}/
                        {frame_id}.png
        keypoints_orig/
            keypoints_orig/
                {frame_id}.json     (42 keypoints, 3D world coords, [x,y,z])
        mano_fits/
            params/
                {frame_id}_left.json   (pose(48), shape(10), trans(3))
                {frame_id}_right.json

Keypoint ordering (same as InterHand2.6M):
    42 joints total: 21 right (0-20) + 21 left (21-41).
    Per-hand: 0-3=Thumb(tip->CMC), 4-7=Index(tip->MCP), ..., 20=Wrist
    (tip-to-root, reversed from MANO convention)

Camera parameters:
    R: (3,3) world-to-camera rotation
    t: (3,)  world-to-camera translation
    p_cam = R @ p_world + t

Conversion strategy:
    - Per capture, parse frame_list.txt to get (seq_name, frame_id) pairs.
    - Sort sequences alphabetically and split into chunks.
    - Per (capture, chunk), create one output sequence folder.
    - Within each folder, each camera gets an indexed pair of (label HDF5, video MP4).
    - Only frames with existing images are included.

Output structure:
    CONVERTED/reinterhand/
        {capture_name}_chunk_XXX/
            XXXXXX_label_00.hdf5
            XXXXXX_video_00.mp4

Usage:
    python scripts/convert_reinterhand.py --src ../InterWild/tool/ReInterHand/download --dst CONVERTED/reinterhand
    python scripts/convert_reinterhand.py --chunk-size 28 --max-samples 3
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


def images_to_mp4_padded(image_paths: list, output_path: str, fps: float = 30.0):
    """Encode images to mp4, padding to even dimensions if needed.

    ffmpeg's yuv420p requires even width and height. ReInterHand images
    can have odd dimensions (e.g. 667x1024), so we pad with black.
    """
    import subprocess
    if not image_paths:
        return
    first = cv2.imread(image_paths[0])
    h, w = first.shape[:2]
    # Pad to even dimensions
    pad_w = w % 2
    pad_h = h % 2
    out_w = w + pad_w
    out_h = h + pad_h

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for p in image_paths:
        frame = cv2.imread(p)
        if frame is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        if pad_w or pad_h:
            frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed for {output_path}: {err}")

# ReInterHand uses the same 42-joint convention as InterHand2.6M:
# Right hand (0-20), Left hand (21-41), per-hand: tip-to-root ordering.
# Mapping from per-hand index to MANO joint index (root-to-tip).
IH_TO_MANO = [None] * 21
IH_TO_MANO[20] = 0   # Wrist
IH_TO_MANO[0] = 4; IH_TO_MANO[1] = 3; IH_TO_MANO[2] = 2; IH_TO_MANO[3] = 1      # Thumb
IH_TO_MANO[4] = 8; IH_TO_MANO[5] = 7; IH_TO_MANO[6] = 6; IH_TO_MANO[7] = 5      # Index
IH_TO_MANO[8] = 12; IH_TO_MANO[9] = 11; IH_TO_MANO[10] = 10; IH_TO_MANO[11] = 9  # Middle
IH_TO_MANO[12] = 16; IH_TO_MANO[13] = 15; IH_TO_MANO[14] = 14; IH_TO_MANO[15] = 13  # Ring
IH_TO_MANO[16] = 20; IH_TO_MANO[17] = 19; IH_TO_MANO[18] = 18; IH_TO_MANO[19] = 17  # Pinky


def reorder_ih_to_mano(joints_ih_21: np.ndarray) -> np.ndarray:
    """Reorder (21, 3) or (N, 21, 3) from IH per-hand order to MANO order."""
    out = np.zeros_like(joints_ih_21)
    for ih_idx, mano_idx in enumerate(IH_TO_MANO):
        if joints_ih_21.ndim == 3:
            out[:, mano_idx] = joints_ih_21[:, ih_idx]
        else:
            out[mano_idx] = joints_ih_21[ih_idx]
    return out


def get_cam_pose(cam_params: dict) -> np.ndarray:
    """Build camera-to-world (4x4) pose from ReInterHand camera params.

    ReInterHand stores R (world-to-camera rotation) and t (world-to-camera translation).
    p_cam = R @ p_world + t
    cam_pose (cam-to-world): R_cw = R^T, t_cw = -R^T @ t
    """
    R = np.array(cam_params['R'], dtype=np.float32).reshape(3, 3)
    # ReInterHand stores t in mm; convert to meters.
    t = np.array(cam_params['t'], dtype=np.float32).reshape(3) / 1000.0

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = -R.T @ t
    return cam_pose


def get_cam_intrinsic(cam_params: dict) -> np.ndarray:
    """Build (3, 3) intrinsic matrix."""
    focal = cam_params['focal']
    princpt = cam_params['princpt']
    K = np.array([
        [focal[0], 0, princpt[0]],
        [0, focal[1], princpt[1]],
        [0, 0, 1],
    ], dtype=np.float32)
    return K


def convert_mano_axisangle_to_rotmat(pose: np.ndarray):
    """Convert (48,) axis-angle MANO pose to rotation matrices.

    Returns:
        global_orient: (3, 3)
        hand_pose: (15, 3, 3)
    """
    global_orient, _ = cv2.Rodrigues(pose[:3].astype(np.float64))
    global_orient = global_orient.astype(np.float32)
    hand_pose = np.zeros((15, 3, 3), dtype=np.float32)
    for j in range(15):
        aa = pose[3 + j * 3:3 + (j + 1) * 3].astype(np.float64)
        hand_pose[j], _ = cv2.Rodrigues(aa)
    return global_orient, hand_pose


def parse_frame_list(frame_list_path: str):
    """Parse frame_list.txt -> list of (seq_name, frame_id_str).

    Each line: "seq_name frame_id"
    """
    entries = []
    with open(frame_list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
    return entries


def load_keypoints(kp_dir: str, frame_id: str) -> np.ndarray:
    """Load (42, 3) keypoints for a frame. Returns None if not found."""
    path = os.path.join(kp_dir, f"{frame_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # ReInterHand stores keypoints in mm; convert to meters.
    return np.array(data, dtype=np.float32).reshape(42, 3) / 1000.0


def load_mano_params(mano_dir: str, frame_id: str, side: str):
    """Load MANO params for a frame/side. Returns None if not found."""
    path = os.path.join(mano_dir, f"{frame_id}_{side}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def preload_chunk_annotations(frame_ids: list, kp_dir: str, mano_dir: str):
    """Preload all keypoints and MANO params for a chunk of frames.

    Loads all per-frame JSON files once so they can be reused across cameras.

    Returns:
        kp_cache: {frame_id: (42, 3) ndarray or None}
        mano_cache: {(frame_id, side): dict or None}
    """
    kp_cache = {}
    mano_cache = {}
    for fid in frame_ids:
        kp_cache[fid] = load_keypoints(kp_dir, fid)
        for side in ("right", "left"):
            mano_cache[(fid, side)] = load_mano_params(mano_dir, fid, side)
    return kp_cache, mano_cache


def build_egodex_data_for_camera(
    cam_params: dict,
    cam_id: str,
    frame_ids: list,
    kp_cache: dict,
    mano_cache: dict,
):
    """Build world-space transforms and MANO data for one camera in a chunk.

    Args:
        cam_params: dict of per-camera params from cam_params.json
        cam_id: camera ID string
        frame_ids: list of frame_id strings in order
        kp_cache: preloaded keypoints {fid: (42,3) or None}
        mano_cache: preloaded MANO params {(fid, side): dict or None}

    Returns:
        intrinsic: (3, 3)
        transforms_dict: {joint_name: (M, 4, 4)}
        confidences_dict: {joint_name: (M,)}
        mano_dicts: list of mano_dict (one per active side)
    """
    M = len(frame_ids)
    identity = np.eye(4, dtype=np.float32)
    intrinsic = get_cam_intrinsic(cam_params[cam_id])
    cam_pose = get_cam_pose(cam_params[cam_id])

    transforms_dict = {}
    confidences_dict = {}

    # Camera transform (static)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Per-frame data arrays
    right_joints_world = np.zeros((M, 21, 3), dtype=np.float32)
    left_joints_world = np.zeros((M, 21, 3), dtype=np.float32)
    right_valid = np.zeros(M, dtype=np.float32)
    left_valid = np.zeros(M, dtype=np.float32)
    right_joint_valid = np.zeros((M, 21), dtype=np.float32)
    left_joint_valid = np.zeros((M, 21), dtype=np.float32)

    right_poses = np.zeros((M, 48), dtype=np.float64)
    right_trans = np.zeros((M, 3), dtype=np.float64)
    right_betas = None
    left_poses = np.zeros((M, 48), dtype=np.float64)
    left_trans = np.zeros((M, 3), dtype=np.float64)
    left_betas = None

    for i, fid in enumerate(frame_ids):
        kp = kp_cache.get(fid)
        if kp is None:
            continue

        # Right hand (indices 0-20)
        right_ih = kp[:21]
        if np.any(np.abs(right_ih) > 1e-6):
            right_joints_world[i] = reorder_ih_to_mano(right_ih)
            right_valid[i] = 1.0
            for ih_idx, mano_idx in enumerate(IH_TO_MANO):
                right_joint_valid[i, mano_idx] = 1.0

        # Left hand (indices 21-41)
        left_ih = kp[21:]
        if np.any(np.abs(left_ih) > 1e-6):
            left_joints_world[i] = reorder_ih_to_mano(left_ih)
            left_valid[i] = 1.0
            for ih_idx, mano_idx in enumerate(IH_TO_MANO):
                left_joint_valid[i, mano_idx] = 1.0

        # MANO params (from cache)
        for side, poses_arr, trans_arr, betas_ref in [
            ("right", right_poses, right_trans, "right_betas"),
            ("left", left_poses, left_trans, "left_betas"),
        ]:
            mano = mano_cache.get((fid, side))
            if mano is not None:
                poses_arr[i] = np.array(mano['pose'], dtype=np.float64)
                trans_arr[i] = np.array(mano['trans'], dtype=np.float64) / 1000.0
                if betas_ref == "right_betas" and right_betas is None:
                    right_betas = np.array(mano['shape'], dtype=np.float32)
                elif betas_ref == "left_betas" and left_betas is None:
                    left_betas = np.array(mano['shape'], dtype=np.float32)

    # Interpolate single-frame gaps; mask entire hand for 2+ consecutive gaps
    for joints_world, jv, hand_valid in [
        (right_joints_world, right_joint_valid, right_valid),
        (left_joints_world, left_joint_valid, left_valid),
    ]:
        for j in range(21):
            col = jv[:, j]
            i = 0
            while i < M:
                if col[i] > 0:
                    i += 1
                    continue
                run_start = i
                while i < M and col[i] == 0:
                    i += 1
                run_end = i
                run_len = run_end - run_start

                if run_len == 1 and run_start > 0 and run_end < M \
                        and col[run_start - 1] > 0 and col[run_end] > 0 \
                        and hand_valid[run_start] > 0:
                    joints_world[run_start, j] = 0.5 * (
                        joints_world[run_start - 1, j] +
                        joints_world[run_end, j]
                    )
                    jv[run_start, j] = 1.0
                else:
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


def convert_reinterhand(
    src_dir: str,
    dst_dir: str,
    chunk_size: int = 28,
    fps: float = 30.0,
    max_samples: int = 0,
):
    """Convert ReInterHand captures to egodex format."""
    os.makedirs(dst_dir, exist_ok=True)

    # Discover capture directories
    capture_dirs = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and d.startswith("m--")
    ])

    if not capture_dirs:
        raise FileNotFoundError(f"No capture directories (m--*) found in {src_dir}")

    print(f"Found {len(capture_dirs)} captures in {src_dir}")
    global_count = 0

    for capture_name in capture_dirs:
        capture_path = os.path.join(src_dir, capture_name)

        # Check required files
        frame_list_path = os.path.join(capture_path, "frame_list.txt")
        cam_params_path = os.path.join(capture_path, "Mugsy_cameras", "cam_params.json")
        kp_dir = os.path.join(capture_path, "keypoints_orig", "keypoints_orig")
        mano_dir = os.path.join(capture_path, "mano_fits", "params")
        img_base = os.path.join(capture_path, "Mugsy_cameras", "envmap_per_frame", "images")

        for required in [frame_list_path, cam_params_path]:
            if not os.path.exists(required):
                print(f"Skipping {capture_name}: missing {os.path.basename(required)}")
                continue

        if not os.path.isdir(img_base):
            print(f"Skipping {capture_name}: no Mugsy camera images")
            continue

        # Load camera params
        with open(cam_params_path) as f:
            cam_params = json.load(f)
        cam_ids = sorted(cam_params.keys())

        # Parse frame list
        frame_entries = parse_frame_list(frame_list_path)
        if not frame_entries:
            print(f"Skipping {capture_name}: empty frame_list.txt")
            continue

        # Group frames by sequence
        from collections import OrderedDict
        seq_frames = OrderedDict()
        for seq_name, fid in frame_entries:
            if seq_name not in seq_frames:
                seq_frames[seq_name] = []
            seq_frames[seq_name].append(fid)

        seq_names = list(seq_frames.keys())
        print(f"\n{'='*60}")
        print(f"Capture: {capture_name}")
        print(f"  Sequences: {len(seq_names)}, Cameras: {len(cam_ids)}, "
              f"Total frames: {len(frame_entries)}")

        # Split sequences into chunks
        n_chunks = max(1, math.ceil(len(seq_names) / chunk_size))
        for chunk_idx in range(n_chunks):
            if max_samples > 0 and global_count >= max_samples:
                break

            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(seq_names))
            chunk_seq_names = seq_names[chunk_start:chunk_end]

            # Collect frame IDs for this chunk in order
            chunk_frame_ids = []
            for sn in chunk_seq_names:
                chunk_frame_ids.extend(seq_frames[sn])

            # Short capture name for output dir
            # Extract date and subject ID: m--YYYYMMDD--HHMM--XXXXXX--...
            parts = capture_name.split("--")
            short_name = f"{parts[1]}_{parts[3]}" if len(parts) >= 4 else capture_name[:30]
            chunk_name = f"{short_name}_chunk_{chunk_idx:03d}"
            chunk_out_dir = os.path.join(dst_dir, chunk_name)

            print(f"\n  [{chunk_name}] seqs {chunk_start}-{chunk_end-1} "
                  f"({len(chunk_seq_names)} seqs, {len(chunk_frame_ids)} frames)")

            os.makedirs(chunk_out_dir, exist_ok=True)

            # Preload all keypoints and MANO params for this chunk once
            print(f"    Preloading annotations for {len(chunk_frame_ids)} frames ...")
            kp_cache, mano_cache = preload_chunk_annotations(
                chunk_frame_ids, kp_dir, mano_dir)
            print(f"    Preloaded {sum(1 for v in kp_cache.values() if v is not None)} "
                  f"keypoints, "
                  f"{sum(1 for v in mano_cache.values() if v is not None)} MANO params")

            for cam_idx, cam_id in enumerate(cam_ids):
                cam_img_dir = os.path.join(img_base, cam_id)
                if not os.path.isdir(cam_img_dir):
                    continue

                # Filter to frames with existing images
                valid_frame_ids = []
                valid_image_paths = []
                for fid in chunk_frame_ids:
                    img_path = os.path.join(cam_img_dir, f"{fid}.png")
                    if os.path.exists(img_path):
                        valid_frame_ids.append(fid)
                        valid_image_paths.append(img_path)

                if not valid_frame_ids:
                    continue

                # Build egodex data (uses preloaded caches)
                intrinsic, transforms_dict, confidences_dict, mano_dicts = \
                    build_egodex_data_for_camera(
                        cam_params, cam_id,
                        valid_frame_ids, kp_cache, mano_cache,
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
                            for key in ["betas", "global_orient_worldspace",
                                        "hand_pose", "transl_worldspace", "kpt3d"]:
                                grp.create_dataset(
                                    key, data=extra_mano[key].astype(np.float32))

                # RGB video
                video_path = os.path.join(chunk_out_dir, f"{prefix_str}_video_00.mp4")
                images_to_mp4_padded(valid_image_paths, video_path, fps=fps)

                # Verify
                _verify_output(chunk_out_dir, cam_idx, len(valid_image_paths))

            global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    print(f"\nDone. Converted {global_count} chunks to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ReInterHand to egodex format")
    parser.add_argument(
        "--src", default="../InterWild/tool/ReInterHand/download",
        help="ReInterHand download directory containing m--* captures")
    parser.add_argument(
        "--dst", default="CONVERTED/reinterhand",
        help="Output directory")
    parser.add_argument(
        "--chunk-size", type=int, default=28,
        help="Number of sequences per chunk (default: 28)")
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video FPS (default: 30.0)")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max chunks to convert (0=all)")
    args = parser.parse_args()

    convert_reinterhand(
        args.src, args.dst,
        chunk_size=args.chunk_size,
        fps=args.fps,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
