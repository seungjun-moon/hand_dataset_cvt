#!/usr/bin/env python3
"""
Convert MTC (Panoptic Studio) video dataset into egodex format.

Input:  mtc_video_dataset/
            ├── annotation.pkl
            ├── camera_data.pkl
            ├── training/{seqName}_id{id}/cam_{cam:02d}.mp4
            └── testing/{seqName}_id{id}/cam_{cam:02d}.mp4

Hand annotation:
    landmarks: 63 values (21 joints × 3) in world space
    Sparse: only a subset of frames have hand annotations
    Joint order: MANO-compatible (wrist, thumb×4, index×4, middle×4, ring×4, pinky×4)

Camera data (per sequence, per camera):
    K: (3, 3) intrinsic
    R: (3, 3) world-to-camera rotation
    t: (3, 1) world-to-camera translation
    distCoef: (5,) lens distortion coefficients

Only frames with at least one hand annotation are included in the output.
Video frames are undistorted to match the pinhole camera model.

Output:
    CONVERTED/mtc_train/
        {seqName}_id{id}/
            000000_label_{cam_idx:02d}.hdf5
            000000_video_{cam_idx:02d}.mp4
    CONVERTED/mtc_eval/
        ...

Usage:
    python scripts/convert_mtc.py --dst CONVERTED/mtc_train --split training
    python scripts/convert_mtc.py --dst CONVERTED/mtc_eval --split testing
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import write_egodex_hdf5, _pipe_frames_to_ffmpeg
from utils.joint_mapping import (
    BODY_JOINTS,
    MANO_TO_EGODEX_SUFFIX,
    METACARPAL_INTERPOLATION,
)
from utils.transforms import (
    interpolate_joint,
    joints_to_transforms,
    make_transform,
    invert_rigid,
)


def build_camera_pose(R, t):
    """Build camera-to-world 4×4 from world-to-camera R, t."""
    T_w2c = np.eye(4, dtype=np.float32)
    T_w2c[:3, :3] = R.astype(np.float32)
    T_w2c[:3, 3] = t.flatten().astype(np.float32)
    return invert_rigid(T_w2c)


def get_annotated_frames(clip_ann):
    """Return sorted list of frame indices that have at least one hand annotation."""
    annotated = set()
    for hand_key in ["left_hand", "right_hand"]:
        if hand_key in clip_ann and clip_ann[hand_key]["landmarks"]:
            annotated.update(clip_ann[hand_key]["landmarks"].keys())
    return sorted(annotated)


def build_egodex_data(clip_ann, cam_K, cam_pose, frame_indices):
    """Build egodex transforms for one clip + one camera.

    Only includes the specified frame_indices (annotated frames).

    Returns:
        intrinsic: (3, 3)
        transforms_dict: {joint_name: (M, 4, 4)}
        confidences_dict: {joint_name: (M,)}
    """
    M = len(frame_indices)
    identity = np.eye(4, dtype=np.float32)
    # Inactive joints use a transform with position far off-screen
    # so the visualizer's in_bounds check skips them.
    inactive = np.eye(4, dtype=np.float32)
    inactive[:3, 3] = 1e8

    transforms_dict = {}
    confidences_dict = {}

    # Camera transform (static per clip)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(inactive, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Process each hand side
    for side in ["left", "right"]:
        hand_key = f"{side}_hand"
        has_hand = hand_key in clip_ann and clip_ann[hand_key]["landmarks"]

        if has_hand:
            landmarks = clip_ann[hand_key]["landmarks"]

            joint_3d = np.zeros((M, 21, 3), dtype=np.float32)
            conf = np.zeros(M, dtype=np.float32)

            for out_idx, frame_idx in enumerate(frame_indices):
                if frame_idx not in landmarks:
                    continue
                lm_arr = np.array(landmarks[frame_idx], dtype=np.float32).reshape(21, 3)
                if np.linalg.norm(lm_arr) < 1e-6:
                    continue
                joint_3d[out_idx] = lm_arr
                conf[out_idx] = 1.0

            all_transforms = np.tile(inactive, (M, 21, 1, 1))
            for i in range(M):
                if conf[i] > 0:
                    all_transforms[i] = joints_to_transforms(joint_3d[i])

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if has_hand:
                transforms_dict[name] = all_transforms[:, mano_idx]
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(inactive, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if has_hand:
                mc = np.zeros((M, 4, 4), dtype=np.float32)
                for i in range(M):
                    if conf[i] > 0:
                        pos = interpolate_joint(
                            joint_3d[i], idx_a, idx_b, alpha=0.3
                        )
                        direction = joint_3d[i, idx_b] - joint_3d[i, idx_a]
                        mc[i] = make_transform(pos, direction)
                    else:
                        mc[i] = inactive
                transforms_dict[name] = mc
                confidences_dict[name] = conf.copy()
            else:
                transforms_dict[name] = np.tile(inactive, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

    return cam_K.astype(np.float32), transforms_dict, confidences_dict


def extract_undistorted_frames(src_video, dst_video, frame_indices, K, dist_coeffs, fps):
    """Extract specific frames from video, undistort, and encode to mp4."""
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        print(f"    WARNING: cannot open {src_video}")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Precompute undistort maps
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (w, h), cv2.CV_16SC2)

    target_set = set(frame_indices)
    max_target = max(frame_indices)

    def frames():
        idx = 0
        while idx <= max_target:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in target_set:
                yield cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            idx += 1

    _pipe_frames_to_ffmpeg(frames(), dst_video, fps, w, h)
    cap.release()


def is_clip_done(out_dir, cam_indices):
    """Check if all HDF5 + video files already exist for this clip."""
    for cam_idx in cam_indices:
        hdf5 = os.path.join(out_dir, f"000000_label_{cam_idx:02d}.hdf5")
        video = os.path.join(out_dir, f"000000_video_{cam_idx:02d}.mp4")
        if not os.path.exists(hdf5) or not os.path.exists(video):
            return False
    return True


def convert_mtc(src_dir, dst_dir, split="training", fps=30.0, max_samples=0,
                seq_filter=None):
    """Convert one split of MTC video dataset to egodex format."""
    print("Loading annotations...")
    with open(os.path.join(src_dir, "annotation.pkl"), "rb") as f:
        annotation = pickle.load(f)
    with open(os.path.join(src_dir, "camera_data.pkl"), "rb") as f:
        camera_data = pickle.load(f)

    split_key = f"{split}_data"
    clips = annotation[split_key]
    print(f"\n=== {split} ({len(clips)} clips) -> {dst_dir} ===")

    os.makedirs(dst_dir, exist_ok=True)

    count = 0
    skipped = 0
    for clip in clips:
        if max_samples > 0 and count >= max_samples:
            break

        seq_name = clip["seqName"]
        pid = clip["id"]
        clip_name = f"{seq_name}_id{pid}"

        if seq_filter and seq_filter not in clip_name:
            continue

        # Get annotated frame indices (union of left/right hand)
        frame_indices = get_annotated_frames(clip)
        if not frame_indices:
            print(f"  WARNING: no hand annotations for {clip_name}, skipping")
            continue

        # Get camera data for this sequence
        if seq_name not in camera_data:
            print(f"  WARNING: no camera data for {seq_name}, skipping")
            continue

        seq_cams = camera_data[seq_name]
        cam_ids = sorted(seq_cams.keys())

        # Check source videos exist
        video_dir = os.path.join(src_dir, split, clip_name)
        if not os.path.isdir(video_dir):
            print(f"  WARNING: no video dir for {clip_name}, skipping")
            continue

        # Output directory per clip
        clip_out_dir = os.path.join(dst_dir, clip_name)

        # Find which cam_ids have actual video files
        valid_cams = []
        for cam_id in cam_ids:
            src_video = os.path.join(video_dir, f"cam_{cam_id:02d}.mp4")
            if os.path.exists(src_video):
                valid_cams.append(cam_id)

        if not valid_cams:
            print(f"  WARNING: no video files for {clip_name}, skipping")
            continue

        # Skip if fully converted
        cam_indices = list(range(len(valid_cams)))
        if os.path.isdir(clip_out_dir) and is_clip_done(
            clip_out_dir, cam_indices
        ):
            skipped += 1
            count += 1
            continue

        # Count per-side annotations
        n_left = (
            len(clip["left_hand"]["landmarks"])
            if "left_hand" in clip
            else 0
        )
        n_right = (
            len(clip["right_hand"]["landmarks"])
            if "right_hand" in clip
            else 0
        )

        print(
            f"[{clip_name}] annotated={len(frame_indices)}/{clip['num_frames']}, "
            f"L:{n_left} R:{n_right}, cams={len(valid_cams)}"
        )

        os.makedirs(clip_out_dir, exist_ok=True)

        for cam_idx, cam_id in enumerate(valid_cams):
            hdf5_path = os.path.join(
                clip_out_dir, f"000000_label_{cam_idx:02d}.hdf5"
            )
            out_video = os.path.join(
                clip_out_dir, f"000000_video_{cam_idx:02d}.mp4"
            )

            # Skip if this camera already done
            if os.path.exists(hdf5_path) and os.path.exists(out_video):
                continue

            cam_info = seq_cams[cam_id]
            cam_K = np.array(cam_info["K"], dtype=np.float32)
            cam_R = np.array(cam_info["R"], dtype=np.float32)
            cam_t = np.array(cam_info["t"], dtype=np.float32)
            cam_pose = build_camera_pose(cam_R, cam_t)

            intrinsic, transforms_dict, confidences_dict = build_egodex_data(
                clip, cam_K, cam_pose, frame_indices
            )

            # Write HDF5
            write_egodex_hdf5(
                hdf5_path,
                intrinsic,
                transforms_dict,
                confidences_dict,
                mano_dict=None,
            )

            # Extract annotated frames, undistort, and encode
            src_video = os.path.join(video_dir, f"cam_{cam_id:02d}.mp4")
            dist_coeffs = np.array(cam_info["distCoef"], dtype=np.float64)
            K64 = cam_K.astype(np.float64)
            extract_undistorted_frames(
                src_video, out_video, frame_indices, K64, dist_coeffs, fps
            )

        count += 1

    print(f"{split}: {count} clips ({skipped} skipped as pre-computed)")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MTC (Panoptic Studio) to egodex format"
    )
    parser.add_argument(
        "--src",
        default="../mtc_dataset/mtc_video_dataset",
        help="MTC video dataset directory",
    )
    parser.add_argument(
        "--dst",
        default="CONVERTED/mtc_train",
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        choices=["training", "testing"],
        default="training",
        help="Which split to convert",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max clips to convert (0=all)",
    )
    parser.add_argument(
        "--seq-filter",
        type=str,
        default=None,
        help="Only process clips containing this substring",
    )
    args = parser.parse_args()

    convert_mtc(args.src, args.dst, split=args.split,
                fps=args.fps, max_samples=args.max_samples,
                seq_filter=args.seq_filter)


if __name__ == "__main__":
    main()
