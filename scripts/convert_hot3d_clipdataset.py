#!/usr/bin/env python3
"""Convert HOT3D (Aria) sequences to ClipDataset format.

HOT3D stores:
  - Images in VRS files (extracted via projectaria_tools)
  - MANO hand poses as 15 PCA coefficients + wrist SE3 (quaternion + translation)
  - Camera intrinsics as FISHEYE624 model parameters
  - 2D hand bounding boxes in box2d_hands.csv
  - MANO wrist_xform is in WORLD frame (SLAM world), not device frame

ClipDataset expects per-clip .pyd files with:
  imgname (T,), center (T,2), scale (T,2), right (T,),
  hand_pose (T,48), hand_tsl (T,3), betas (T,10),
  has_hand_pose (T,), has_betas (T,),
  hand_keypoints_2d (T,21,3), hand_keypoints_3d (T,21,4),  [camera frame]
  cTw (T,4,4), focal (T,3,3), person_id (T,), extra_info (T,)

Projection chain:
  world --[inv(T_world_device)]--> device --[inv(T_device_camera)]--> camera --[K]--> pixel

Usage:
  python scripts/convert_hot3d_clipdataset.py \
    --src /path/to/hot3d/dataset/P0001_10a27bf7 \
    --dst /path/to/output/hot3d \
    --mano_dir path/to/mano \
    --clip_len 90 --stream_id 214-1 --max_clips 2
"""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.calibration import (
    distort_by_calibration, get_linear_camera_calibration,
)


def quat_wxyz_to_rotmat(q_wxyz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotmat_to_axis_angle(R):
    """Convert 3x3 rotation matrix to axis-angle (3,)."""
    aa, _ = cv2.Rodrigues(R.astype(np.float64))
    return aa.flatten().astype(np.float32)


def load_mano_pca_basis(mano_dir, is_right=True):
    """Load the MANO PCA basis to convert PCA coefficients -> axis-angle."""
    fname = "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl"
    with open(os.path.join(mano_dir, fname), "rb") as f:
        mano_data = pickle.load(f, encoding="latin1")
    return mano_data["hands_components"], mano_data["hands_mean"]


def pca_to_axis_angle(pca_coeffs, components, mean):
    """Convert PCA coefficients to axis-angle: mean + pca_coeffs @ components[:n_pca]."""
    n_pca = len(pca_coeffs)
    return (mean + pca_coeffs @ components[:n_pca]).astype(np.float32)


def load_hand_poses(jsonl_path):
    """Load MANO hand poses from HOT3D jsonl. Returns dict: ts -> hand_poses_dict."""
    poses = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            poses[entry["timestamp_ns"]] = entry["hand_poses"]
    return poses


def load_mask_csv(csv_path, stream_id_str):
    """Load a HOT3D mask CSV. Returns set of timestamp_ns where mask is True
    for the given stream_id. Masks are per (ts, stream_id), not per hand.
    """
    ok = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stream_id"] != stream_id_str:
                continue
            if row["mask"].strip().lower() == "true":
                ok.add(int(row["timestamp[ns]"]))
    return ok


def load_box2d_hands(csv_path):
    """Load 2D hand bounding boxes. Returns dict: (stream_id, ts) -> {hand_idx: bbox}."""
    boxes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["x_min[pixel]"] == "":
                continue
            key = (row["stream_id"], int(row["timestamp[ns]"]))
            hand_idx = int(row["hand_index"])
            if key not in boxes:
                boxes[key] = {}
            boxes[key][hand_idx] = {
                "x_min": float(row["x_min[pixel]"]),
                "x_max": float(row["x_max[pixel]"]),
                "y_min": float(row["y_min[pixel]"]),
                "y_max": float(row["y_max[pixel]"]),
                "visibility": float(row["visibility_ratio[%]"]),
            }
    return boxes


def load_camera_model(json_path, stream_id_str):
    """Load camera intrinsics from camera_models.json."""
    with open(json_path, "r") as f:
        models = json.load(f)
    for m in models:
        if m["stream_id"] == stream_id_str:
            return m
    raise ValueError(f"Stream {stream_id_str} not found")


def get_pinhole_intrinsic(pinhole_cal):
    """Build 3x3 intrinsic matrix from a projectaria pinhole calibration."""
    fl = pinhole_cal.get_focal_lengths()
    pp = pinhole_cal.get_principal_point()
    return np.array([[fl[0], 0, pp[0]], [0, fl[1], pp[1]], [0, 0, 1]], dtype=np.float32)


def build_T_device_camera(cam_model):
    """Build 4x4 T_device_camera from camera model."""
    T_dc = cam_model["T_Device_Camera"]
    R = quat_wxyz_to_rotmat(T_dc["quaternion_wxyz"])
    t = np.array(T_dc["translation_xyz"])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_slam_trajectory(slam_csv_path, timecode_mapping_path):
    """Load closed-loop SLAM trajectory and build timestamp mapping.

    Returns dict: timecode_ns -> T_world_device (4x4 float64)
    """
    tc_to_dt = {}
    with open(timecode_mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tc_to_dt[int(row["timecode_ns"])] = int(row["devicetime_ns"])

    slam_by_dt_us = {}
    with open(slam_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_us = int(row["tracking_timestamp_us"])
            q_wxyz = [float(row[f"q{c}_world_device"]) for c in ["w", "x", "y", "z"]]
            t_xyz = [float(row[f"t{c}_world_device"]) for c in ["x", "y", "z"]]
            R = quat_wxyz_to_rotmat(q_wxyz)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t_xyz
            slam_by_dt_us[dt_us] = T

    slam_dt_us_list = sorted(slam_by_dt_us.keys())
    slam_dt_us_arr = np.array(slam_dt_us_list)
    slam_min_us = slam_dt_us_arr[0]
    slam_max_us = slam_dt_us_arr[-1]

    result = {}
    for tc_ns, dt_ns in tc_to_dt.items():
        dt_us = dt_ns // 1000
        # Skip frames outside SLAM coverage
        if dt_us < slam_min_us or dt_us > slam_max_us:
            continue
        idx = np.searchsorted(slam_dt_us_arr, dt_us)
        if idx == 0:
            closest = slam_dt_us_list[0]
        elif idx >= len(slam_dt_us_list):
            closest = slam_dt_us_list[-1]
        else:
            before = slam_dt_us_list[idx - 1]
            after = slam_dt_us_list[idx]
            closest = before if abs(dt_us - before) < abs(dt_us - after) else after
        result[tc_ns] = slam_by_dt_us[closest]

    return result


def batch_mano_fk(mano_dir, all_betas, all_pca, all_wrist_R, all_wrist_t, is_right):
    """Batch MANO forward kinematics. Returns (N, 21, 3) joints in world frame."""
    import smplx
    N = len(all_pca)
    if N == 0:
        return np.zeros((0, 21, 3), dtype=np.float32)

    model_path = os.path.join(mano_dir, "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl")
    mano_layer = smplx.create(
        model_path, "mano",
        use_pca=True, is_rhand=is_right,
        num_pca_comps=all_pca.shape[1],
        flat_hand_mean=False,
    )
    # Apply the standard smplx MANO_LEFT.pkl shapedirs fix (smplx#48) for the
    # left hand. Matches ho2o's label convention and the official HOT3D loader
    # (hot3d/data_loaders/mano_layer.py). Without this, every left-hand label
    # is shifted by ~1 cm per unit of betas[0] (up to 24 mm on extreme subjects).
    if not is_right:
        mano_layer.shapedirs.data[:, 0, :] *= -1

    global_orient = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        global_orient[i] = rotmat_to_axis_angle(all_wrist_R[i])

    BATCH = 256
    all_joints = []
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        output = mano_layer(
            betas=torch.tensor(all_betas[start:end], dtype=torch.float32),
            global_orient=torch.tensor(global_orient[start:end], dtype=torch.float32),
            hand_pose=torch.tensor(all_pca[start:end], dtype=torch.float32),
            transl=torch.tensor(all_wrist_t[start:end], dtype=torch.float32),
            return_verts=True,
        )
        joints = output.joints.detach().numpy()  # (B, 16, 3)
        if joints.shape[1] < 21:
            FINGERTIP_VERT_IDS = [744, 320, 443, 554, 671]
            verts = output.vertices.detach().numpy()
            tips = verts[:, FINGERTIP_VERT_IDS]
            joints = np.concatenate([joints, tips], axis=1)

        # Reorder MANO FK -> egodex standard 21-joint order
        MANO_FK_TO_EGODEX = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        joints = joints[:, MANO_FK_TO_EGODEX]
        all_joints.append(joints)

    return np.concatenate(all_joints, axis=0).astype(np.float32)


def project_3d_to_2d(joints_3d_cam, K):
    """Project (21,3) 3D joints in camera frame to (21,3) 2D [x,y,conf] using pinhole K."""
    proj = (K @ joints_3d_cam.T).T  # (21, 3)
    z = proj[:, 2:3]
    valid = (z > 0.01).flatten()
    xy = np.zeros((21, 2), dtype=np.float32)
    xy[valid] = (proj[valid, :2] / z[valid])
    conf = valid.astype(np.float32)
    return np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)


def save_pyd(path, **data):
    """Save data as .pyd (numpy pickled dict), matching existing ClipDataset format."""
    np.savez(path, **data)
    npz_path = path if path.endswith(".npz") else path + ".npz"
    if os.path.exists(npz_path) and not path.endswith(".npz"):
        os.rename(npz_path, path)


def convert_sequence(src_dir, dst_dir, img_dir, mano_dir, clip_len, stream_id_str,
                     max_clips=None, min_visibility=0.5, img_size=1408):
    """Convert one HOT3D sequence to ClipDataset format."""
    recording_name = os.path.basename(src_dir)
    person_id = int(recording_name.split("_")[0].replace("P", ""))
    print(f"Converting {recording_name} (person={person_id})")

    # Load metadata
    hand_poses = load_hand_poses(os.path.join(src_dir, "mano_hand_pose_trajectory.jsonl"))
    box2d = load_box2d_hands(os.path.join(src_dir, "box2d_hands.csv"))
    cam_model = load_camera_model(os.path.join(src_dir, "camera_models.json"), stream_id_str)

    # HOT3D per-frame quality masks (per stream, not per hand).
    # qa_pass is the strict gate the dataset authors use; pose_available/visible
    # are cheap extras that mostly agree with it.
    masks_dir = os.path.join(src_dir, "masks")
    qa_ok = load_mask_csv(os.path.join(masks_dir, "mask_qa_pass.csv"), stream_id_str)
    pose_ok = load_mask_csv(os.path.join(masks_dir, "mask_hand_pose_available.csv"), stream_id_str)
    vis_ok = load_mask_csv(os.path.join(masks_dir, "mask_hand_visible.csv"), stream_id_str)
    print(f"  Masks: qa_pass={len(qa_ok)} pose_avail={len(pose_ok)} hand_visible={len(vis_ok)}")

    # Build static transforms
    T_dc = build_T_device_camera(cam_model)     # device <- camera
    T_cd = np.linalg.inv(T_dc)                  # camera <- device

    # Load SLAM trajectory (wrist_xform is in world frame, we need world->camera)
    slam_csv = os.path.join(src_dir, "mps", "slam", "closed_loop_trajectory.csv")
    tc_mapping = os.path.join(src_dir, "timecode_devicetime_mapping.csv")
    slam_traj = load_slam_trajectory(slam_csv, tc_mapping)
    print(f"  Loaded SLAM trajectory: {len(slam_traj)} timestamps with coverage")

    # Load MANO PCA bases
    pca_right, mean_right = load_mano_pca_basis(mano_dir, is_right=True)
    pca_left, mean_left = load_mano_pca_basis(mano_dir, is_right=False)

    # Open VRS and build undistortion calibration
    vrs_path = os.path.join(src_dir, "recording.vrs")
    provider = data_provider.create_vrs_data_provider(vrs_path)
    stream_id = StreamId(stream_id_str)
    timestamps = sorted(provider.get_timestamps_ns(stream_id, TimeDomain.TIME_CODE))

    # Build pinhole calibration for undistortion
    device_cal = provider.get_device_calibration()
    cam_label = provider.get_label_from_stream_id(stream_id)
    fisheye_cal = device_cal.get_camera_calib(cam_label)
    image_size = fisheye_cal.get_image_size()
    focal = fisheye_cal.get_focal_lengths()
    pinhole_cal = get_linear_camera_calibration(
        int(image_size[0]), int(image_size[1]), focal[0])
    K = get_pinhole_intrinsic(pinhole_cal)
    print(f"  {len(timestamps)} RGB frames, pinhole K: f={K[0,0]:.1f} pp=({K[0,2]:.1f},{K[1,2]:.1f})")

    os.makedirs(os.path.join(img_dir, recording_name), exist_ok=True)

    for hand_idx, is_right_hand in [(1, True), (0, False)]:
        hand_label = "right" if is_right_hand else "left"
        components = pca_right if is_right_hand else pca_left
        mean_pose = mean_right if is_right_hand else mean_left

        # Step 1: Collect valid frame indices
        # Require: hand pose + bbox + SLAM coverage + bbox within image + sufficient
        # visibility + HOT3D quality masks (qa_pass, pose_available, hand_visible).
        valid_frames = []
        drop = defaultdict(int)
        for frame_i, ts in enumerate(timestamps):
            if ts not in hand_poses:
                drop["no_pose_ts"] += 1; continue
            h_key = str(hand_idx)
            if h_key not in hand_poses[ts]:
                drop["no_pose_hand"] += 1; continue
            box_key = (stream_id_str, ts)
            if box_key not in box2d or hand_idx not in box2d[box_key]:
                drop["no_bbox"] += 1; continue
            if ts not in slam_traj:
                drop["no_slam"] += 1; continue
            bbox = box2d[box_key][hand_idx]
            if bbox["visibility"] < min_visibility:
                drop["low_vis"] += 1; continue
            cx = (bbox["x_min"] + bbox["x_max"]) / 2
            cy = (bbox["y_min"] + bbox["y_max"]) / 2
            if cx < 0 or cx > img_size or cy < 0 or cy > img_size:
                drop["bbox_out"] += 1; continue
            if ts not in qa_ok:
                drop["qa_fail"] += 1; continue
            if ts not in pose_ok:
                drop["pose_unavail"] += 1; continue
            if ts not in vis_ok:
                drop["not_visible"] += 1; continue
            valid_frames.append((frame_i, ts))
        if drop:
            print(f"    drop counts: {dict(drop)}")

        n_valid = len(valid_frames)
        n_clips = n_valid // clip_len
        if max_clips is not None:
            n_clips = min(n_clips, max_clips)
        n_needed = n_clips * clip_len

        if n_clips == 0:
            print(f"  {hand_label}: {n_valid} valid frames, not enough for {clip_len}-frame clip")
            continue

        valid_frames = valid_frames[:n_needed]
        print(f"  {hand_label}: {n_valid} valid -> {n_clips} clips × {clip_len} = {n_needed} frames")

        # Step 2: Batch collect annotations
        all_pca = np.zeros((n_needed, 15), dtype=np.float32)
        all_wrist_R = np.zeros((n_needed, 3, 3), dtype=np.float64)
        all_wrist_t = np.zeros((n_needed, 3), dtype=np.float32)
        all_betas = np.zeros((n_needed, 10), dtype=np.float32)
        all_wrist_aa = np.zeros((n_needed, 3), dtype=np.float32)
        all_hand_pose_aa = np.zeros((n_needed, 45), dtype=np.float32)
        all_center = np.zeros((n_needed, 2), dtype=np.float32)
        all_scale = np.zeros((n_needed, 2), dtype=np.float64)
        all_T_cw = np.zeros((n_needed, 4, 4), dtype=np.float64)
        all_imgname = []

        for i, (frame_i, ts) in enumerate(valid_frames):
            hd = hand_poses[ts][str(hand_idx)]
            pca = np.array(hd["pose"], dtype=np.float32)
            q = np.array(hd["wrist_xform"]["q_wxyz"])
            t = np.array(hd["wrist_xform"]["t_xyz"], dtype=np.float32)
            betas = np.array(hd["betas"], dtype=np.float32)

            R = quat_wxyz_to_rotmat(q)
            all_pca[i] = pca
            all_wrist_R[i] = R
            all_wrist_t[i] = t
            all_betas[i] = betas
            all_wrist_aa[i] = rotmat_to_axis_angle(R)
            all_hand_pose_aa[i] = pca_to_axis_angle(pca, components, mean_pose)

            # Per-frame T_camera_world = T_camera_device @ inv(T_world_device)
            T_wd = slam_traj[ts]
            T_dw = np.linalg.inv(T_wd)
            all_T_cw[i] = T_cd @ T_dw

            all_imgname.append(os.path.join(recording_name, f"{frame_i:06d}.jpg"))

        # Step 3: Batch MANO FK for 3D joints (in world frame)
        print(f"    Running batch MANO FK ({n_needed} frames)...")
        joints_3d_world = batch_mano_fk(
            mano_dir, all_betas, all_pca, all_wrist_R, all_wrist_t, is_right_hand)

        # Transform world -> camera per frame, project to 2D, compute center/scale
        all_kp3d_cam = np.zeros((n_needed, 21, 3), dtype=np.float32)
        all_kp2d = np.zeros((n_needed, 21, 3), dtype=np.float32)
        for i in range(n_needed):
            R_cw = all_T_cw[i, :3, :3]
            t_cw = all_T_cw[i, :3, 3:4]
            joints_cam = (R_cw @ joints_3d_world[i].T + t_cw).T.astype(np.float32)
            all_kp3d_cam[i] = joints_cam
            kp2d = project_3d_to_2d(joints_cam, K)
            all_kp2d[i] = kp2d

            # Compute center/scale from projected keypoints in undistorted image.
            # Matches arctic/dexycb/ho3d/ho2o haptic convention: scale*200 =
            # 3x kp span, so the dataloader's default rescale_factor=2 yields a
            # final crop of 6x kp_span.
            valid = kp2d[:, 2] > 0.5
            if valid.any():
                pts = kp2d[valid, :2]
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                all_center[i] = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                s = 3.0 * max(x_max - x_min, y_max - y_min) / 200.0
                all_scale[i] = [s, s]

        # 3D keypoints in camera frame with confidence
        all_kp3d = np.concatenate(
            [all_kp3d_cam, np.ones((n_needed, 21, 1), dtype=np.float32)], axis=2)

        # Full 48-dim pose: 3 global_orient + 45 hand_pose (axis-angle)
        all_hand_pose_48 = np.concatenate(
            [all_wrist_aa, all_hand_pose_aa], axis=1).astype(np.float32)

        # Step 4: Extract only needed images
        needed_frames = set()
        for frame_i, ts in valid_frames:
            needed_frames.add((frame_i, ts))

        print(f"    Extracting & undistorting {len(needed_frames)} images...")
        for frame_i, ts in tqdm(needed_frames, desc="    Images"):
            img_path = os.path.join(img_dir, recording_name, f"{frame_i:06d}.jpg")
            if os.path.exists(img_path):
                continue
            image_data = provider.get_image_data_by_time_ns(
                stream_id, ts, TimeDomain.TIME_CODE, TimeQueryOptions.CLOSEST)
            img = image_data[0].to_numpy_array()
            img = distort_by_calibration(img, pinhole_cal, fisheye_cal)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Step 5: Save clips
        clip_dir = os.path.join(dst_dir, "clip", recording_name, str(hand_idx))
        os.makedirs(clip_dir, exist_ok=True)

        clip_names = []
        for ci in range(n_clips):
            s, e = ci * clip_len, (ci + 1) * clip_len
            clip_fname = f"{s:06d}_{e:06d}.pyd"
            clip_path = os.path.join(clip_dir, clip_fname)

            clip_data = {
                "imgname": np.array(all_imgname[s:e]),
                "center": all_center[s:e],
                "scale": all_scale[s:e],
                "right": np.full(clip_len, 1 if is_right_hand else 0, dtype=np.int64),
                "hand_pose": all_hand_pose_48[s:e],
                "hand_tsl": all_wrist_t[s:e],
                "betas": all_betas[s:e],
                "has_hand_pose": np.ones(clip_len, dtype=np.float64),
                "has_betas": np.ones(clip_len, dtype=np.float64),
                "hand_keypoints_2d": all_kp2d[s:e],
                "hand_keypoints_3d": all_kp3d[s:e],
                "cTw": all_T_cw[s:e].astype(np.float32),
                "focal": np.tile(K[None], (clip_len, 1, 1)),
                "person_id": np.full(clip_len, person_id, dtype=np.int64),
                "extra_info": np.array(
                    [defaultdict(list) for _ in range(clip_len)], dtype=object),
            }
            save_pyd(clip_path, **clip_data)
            clip_names.append(os.path.join(recording_name, str(hand_idx), clip_fname))

        index_name = f"hot3d_{recording_name}_{hand_label}_clip"
        index_path = os.path.join(dst_dir, "clip", f"{index_name}.data.pyd")
        save_pyd(index_path,
                 label_dir=os.path.join(dst_dir, "clip"),
                 labelname=np.array(clip_names))
        print(f"    Saved {n_clips} clips -> {index_path}")

    # Create combined index
    all_clip_names = []
    for hand_idx in [0, 1]:
        hand_label = "right" if hand_idx == 1 else "left"
        index_path = os.path.join(
            dst_dir, "clip", f"hot3d_{recording_name}_{hand_label}_clip.data.pyd")
        if os.path.exists(index_path):
            d = np.load(index_path, allow_pickle=True)
            all_clip_names.extend(d["labelname"].tolist())
    if all_clip_names:
        combined_path = os.path.join(
            dst_dir, "clip", f"hot3d_{recording_name}_clip.data.pyd")
        save_pyd(combined_path,
                 label_dir=os.path.join(dst_dir, "clip"),
                 labelname=np.array(all_clip_names))
        print(f"  Combined index: {combined_path} ({len(all_clip_names)} clips)")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert HOT3D to ClipDataset format")
    parser.add_argument("--src", required=True,
                        help="Path to HOT3D sequence dir (e.g., P0001_10a27bf7)")
    parser.add_argument("--dst", required=True, help="Output label directory")
    parser.add_argument("--img_dir", default=None,
                        help="Output image directory (default: dst/../images)")
    parser.add_argument("--mano_dir", default="_DATA/data/mano",
                        help="Path to MANO model files")
    parser.add_argument("--clip_len", type=int, default=90, help="Frames per clip")
    parser.add_argument("--stream_id", default="214-1",
                        help="VRS stream ID for RGB camera")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Max clips per hand (for testing)")
    parser.add_argument("--min_visibility", type=float, default=0.5,
                        help="Minimum bbox visibility ratio to include frame")
    args = parser.parse_args()

    if args.img_dir is None:
        args.img_dir = os.path.join(os.path.dirname(args.dst.rstrip("/")), "images")

    convert_sequence(args.src, args.dst, args.img_dir, args.mano_dir,
                     args.clip_len, args.stream_id, args.max_clips,
                     args.min_visibility)


if __name__ == "__main__":
    main()
