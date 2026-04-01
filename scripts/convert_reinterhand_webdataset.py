#!/usr/bin/env python3
"""
Convert ReInterHand CONVERTED (HDF5+MP4) data to WebDataset format (tar files).

Reads the egodex-format HDF5 annotation files and MP4 videos from
CONVERTED/reinterhand/ and writes tar files with 1000 samples each.

Each sample consists of:
    {prefix}/{sample_id}.jpg        -- RGB image (full frame, not cropped)
    {prefix}/{sample_id}.data.pyd   -- pickled annotation list

Annotation format matches existing HaMER WebDataset tars:
    [{
        "keypoints_2d": (21, 3) float32,
        "keypoints_3d": (21, 4) float32,
        "center": (2,) float64,
        "scale": (2,) float64,
        "right": float32 scalar (1.0 or 0.0),
        "hand_pose": (48,) float32,
        "betas": (10,) float32,
        "has_hand_pose": float32 scalar (1.0),
        "has_betas": float32 scalar (1.0),
        "personid": int (0),
        "extra_info": dict {},
    }]

Both right and left hands are emitted as separate samples per frame (when valid).
Global orient is rotated from world space to camera space.

Usage:
    python scripts/convert_reinterhand_webdataset.py \
        --src CONVERTED/reinterhand \
        --dst CONVERTED/reinterhand_webdataset/
"""

import argparse
import io
import os
import pickle
import sys
import tarfile

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import smplx

from utils.image_utils import bbox_from_keypoints, expand_to_square, project_3d_to_2d

SAMPLES_PER_SHARD = 1000
DATASET_PREFIX = "reinterhand"
DEFAULT_MANO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "hand_tracking_ablation", "_DATA", "data", "mano"
)


def load_hand_mean_poses(mano_dir):
    """Load MANO mean hand poses (45,) for both hands.

    ReInterHand MANO fits store hand_pose as offsets from the natural grasping
    mean (flat_hand_mean=False). HaMER expects absolute rotations from flat hand
    (flat_hand_mean=True). We add the per-hand mean to convert.

    Returns:
        dict with "right" and "left" keys, each (45,) ndarray.
    """
    mano_dir = os.path.normpath(mano_dir)
    return {
        "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True,
                                 flat_hand_mean=False).hand_mean.numpy(),
        "left": smplx.MANOLayer(model_path=mano_dir, is_rhand=False,
                                flat_hand_mean=False).hand_mean.numpy(),
    }


def rotmat_to_axis_angle(R):
    """Convert (3,3) rotation matrix to (3,) axis-angle."""
    aa, _ = cv2.Rodrigues(R.astype(np.float64))
    return aa.flatten().astype(np.float32)


def shift_kp3d_for_centered_pp(kp3d, intrinsic, img_w, img_h):
    """Shift camera-space 3D keypoints so projection with centered principal
    point gives the same 2D as projection with the actual intrinsic matrix.

    Given: u = fx * X/Z + cx  (actual projection)
    Want:  u = fx * X'/Z + W/2  (centered pp projection)
    So:    X' = X + (cx - W/2) * Z / fx

    Args:
        kp3d: (N, 3) keypoints in camera space.
        intrinsic: (3, 3) camera intrinsic matrix.
        img_w: image width.
        img_h: image height.

    Returns:
        (N, 3) shifted keypoints.
    """
    kp3d_shifted = kp3d.copy()
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    Z = kp3d[:, 2]
    kp3d_shifted[:, 0] += (cx - img_w / 2.0) * Z / fx
    kp3d_shifted[:, 1] += (cy - img_h / 2.0) * Z / fy
    return kp3d_shifted


def extract_hand_samples(hdf5_path, video_path, hand_mean_poses):
    """Extract per-frame hand samples from one HDF5+MP4 pair.

    Args:
        hdf5_path: Path to HDF5 annotation file.
        video_path: Path to corresponding MP4 video.
        hand_mean_poses: dict with "right"/"left" keys, each (45,) mean pose.

    Yields (frame_idx, annotation_dict) for each valid hand in each frame.
    """
    # Get image dimensions from video
    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with h5py.File(hdf5_path, "r") as f:
        intrinsic = f["camera/intrinsic"][:]  # (3,3)
        cam_poses = f["transforms/camera"][:]  # (M, 4, 4)
        M = cam_poses.shape[0]

        hands = []
        for side in ("right", "left"):
            mano_key = f"mano_{side}"
            if mano_key not in f:
                continue

            grp = f[mano_key]
            betas = grp["betas"][:]  # (10,)
            global_orient = grp["global_orient_worldspace"][:]  # (M, 3, 3)
            hand_pose_rot = grp["hand_pose"][:]  # (M, 15, 3, 3)
            transl = grp["transl_worldspace"][:]  # (M, 3)
            kpt3d = grp["kpt3d"][:]  # (M, 21, 3)

            # Determine valid frames from confidences
            # Use wrist confidence (rightHand or leftHand)
            conf_key = f"confidences/{side}Hand"
            if conf_key in f:
                valid = f[conf_key][:] > 0.5
            else:
                # Fallback: check if kpt3d has non-zero values
                valid = np.any(np.abs(kpt3d) > 1e-6, axis=(1, 2))

            hands.append({
                "side": side,
                "betas": betas,
                "global_orient": global_orient,
                "hand_pose_rot": hand_pose_rot,
                "transl": transl,
                "kpt3d": kpt3d,
                "valid": valid,
            })

        if not hands:
            return

    # Now process frame by frame
    for frame_idx in range(M):
        cam_pose = cam_poses[frame_idx]  # (4, 4) camera-to-world
        # World-to-camera rotation
        R_w2c = cam_pose[:3, :3].T  # R^T

        for hand in hands:
            if not hand["valid"][frame_idx]:
                continue

            side = hand["side"]
            kpt3d_world = hand["kpt3d"][frame_idx]  # (21, 3)

            # Project to 2D
            kpt2d = project_3d_to_2d(kpt3d_world, cam_pose, intrinsic)  # (21, 2)

            # Build keypoints_2d (21, 3) with confidence
            keypoints_2d = np.zeros((21, 3), dtype=np.float32)
            keypoints_2d[:, :2] = kpt2d
            keypoints_2d[:, 2] = 1.0

            # Build keypoints_3d in camera space (21, 4) with confidence
            t_cam = cam_pose[:3, 3]
            kpt3d_cam = (kpt3d_world - t_cam) @ cam_pose[:3, :3]  # world-to-cam

            # Shift for centered principal point: HaMER/visualizer project
            # with u = f*X/Z + W/2, so we shift X so this matches the actual
            # projection u = fx*X/Z + cx.
            kpt3d_cam = shift_kp3d_for_centered_pp(
                kpt3d_cam, intrinsic, img_w, img_h)

            keypoints_3d = np.zeros((21, 4), dtype=np.float32)
            keypoints_3d[:, :3] = kpt3d_cam
            keypoints_3d[:, 3] = 1.0

            # Compute center/scale from 2D keypoints
            bbox = bbox_from_keypoints(kpt2d)
            sq_bbox = expand_to_square(bbox)
            center = np.array([
                (sq_bbox[0] + sq_bbox[2]) / 2.0,
                (sq_bbox[1] + sq_bbox[3]) / 2.0,
            ], dtype=np.float64)
            bbox_size = float(sq_bbox[2] - sq_bbox[0])
            scale = np.array([bbox_size / 200.0, bbox_size / 200.0], dtype=np.float64)

            # Convert MANO params to axis-angle
            # Global orient: world -> camera
            go_world = hand["global_orient"][frame_idx]  # (3, 3)
            go_cam = R_w2c @ go_world  # rotate to camera frame
            go_aa = rotmat_to_axis_angle(go_cam)  # (3,)

            # Hand pose: 15 joints, each (3,3) -> (3,) axis-angle
            # The HDF5 stores rotation matrices of OFFSETS from grasping mean
            # (flat_hand_mean=False convention). Convert back to offset axis-angle
            # and add the mean to get absolute axis-angle (flat_hand_mean=True).
            hp_rot = hand["hand_pose_rot"][frame_idx]  # (15, 3, 3)
            hp_aa = np.zeros(45, dtype=np.float32)
            for j in range(15):
                hp_aa[j * 3:(j + 1) * 3] = rotmat_to_axis_angle(hp_rot[j])
            hp_aa = hp_aa + hand_mean_poses[side]  # offset -> absolute

            hand_pose = np.concatenate([go_aa, hp_aa])  # (48,)

            ann = {
                "keypoints_2d": keypoints_2d,
                "keypoints_3d": keypoints_3d,
                "center": center,
                "scale": scale,
                "right": np.float32(1.0 if side == "right" else 0.0),
                "hand_pose": hand_pose,
                "betas": hand["betas"].astype(np.float32),
                "has_hand_pose": np.float32(1.0),
                "has_betas": np.float32(1.0),
                "personid": 0,
                "extra_info": {},
            }

            yield frame_idx, ann


def add_to_tar(tw, sample_id, img_bytes, annotation):
    """Add a jpg + pyd pair to a tar writer."""
    jpg_name = f"{DATASET_PREFIX}/{sample_id}.jpg"
    jpg_info = tarfile.TarInfo(name=jpg_name)
    jpg_info.size = len(img_bytes)
    tw.addfile(jpg_info, io.BytesIO(img_bytes))

    pyd_name = f"{DATASET_PREFIX}/{sample_id}.data.pyd"
    pyd_bytes = pickle.dumps(annotation)
    pyd_info = tarfile.TarInfo(name=pyd_name)
    pyd_info.size = len(pyd_bytes)
    tw.addfile(pyd_info, io.BytesIO(pyd_bytes))


def main():
    parser = argparse.ArgumentParser(
        description="Convert ReInterHand CONVERTED data to WebDataset tar files")
    parser.add_argument("--src",
                        default="CONVERTED/reinterhand",
                        help="Path to CONVERTED/reinterhand directory")
    parser.add_argument("--dst",
                        default="CONVERTED/reinterhand_webdataset/",
                        help="Output directory for tar files")
    parser.add_argument("--samples-per-shard", type=int,
                        default=SAMPLES_PER_SHARD,
                        help=f"Samples per tar shard (default: {SAMPLES_PER_SHARD})")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to convert (0=all)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Max chunk directories to process (0=all)")
    parser.add_argument("--mano-dir", default=DEFAULT_MANO_DIR,
                        help="Path to MANO model directory")
    args = parser.parse_args()

    print(f"Source: {args.src}")
    print(f"Output: {args.dst}")

    # Load MANO mean hand poses for convention conversion
    print("Loading MANO mean hand poses...")
    hand_mean_poses = load_hand_mean_poses(args.mano_dir)

    # Discover chunk directories
    chunk_dirs = sorted([
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    ])
    print(f"Found {len(chunk_dirs)} chunk directories")

    if args.max_chunks > 0:
        chunk_dirs = chunk_dirs[:args.max_chunks]
        print(f"  Limiting to {len(chunk_dirs)} chunks")

    os.makedirs(args.dst, exist_ok=True)

    # Collect all samples: (chunk_dir, hdf5_file, frame_idx, annotation)
    # We process lazily to avoid memory issues — buffer samples and write shards
    total_written = 0
    shard_idx = 0
    sps = args.samples_per_shard
    current_tar = None
    tar_path = None
    samples_in_shard = 0

    def open_new_shard():
        nonlocal current_tar, tar_path, shard_idx, samples_in_shard
        if current_tar is not None:
            current_tar.close()
            n = samples_in_shard
            print(f"  shard {shard_idx - 1}: {os.path.basename(tar_path)} ({n} samples)")
        tar_name = f"{shard_idx:06d}.tar"
        tar_path = os.path.join(args.dst, tar_name)
        current_tar = tarfile.open(tar_path, "w")
        samples_in_shard = 0
        shard_idx += 1

    open_new_shard()

    for chunk_name in chunk_dirs:
        chunk_path = os.path.join(args.src, chunk_name)

        # Find all HDF5+MP4 pairs
        hdf5_files = sorted([
            f for f in os.listdir(chunk_path) if f.endswith("_label_00.hdf5")
        ])

        for hdf5_file in hdf5_files:
            prefix = hdf5_file.replace("_label_00.hdf5", "")
            video_file = f"{prefix}_video_00.mp4"
            hdf5_path = os.path.join(chunk_path, hdf5_file)
            video_path = os.path.join(chunk_path, video_file)

            if not os.path.exists(video_path):
                continue

            # Collect annotations first (frame_idx -> list of annotations)
            frame_anns = {}
            for frame_idx, ann in extract_hand_samples(hdf5_path, video_path, hand_mean_poses):
                if frame_idx not in frame_anns:
                    frame_anns[frame_idx] = []
                frame_anns[frame_idx].append(ann)

            if not frame_anns:
                continue

            # Read video frames on demand
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  WARNING: cannot open {video_path}")
                continue

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx in frame_anns:
                    # Encode frame as JPEG
                    _, img_bytes = cv2.imencode(".jpg", frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    img_bytes = img_bytes.tobytes()

                    for ann in frame_anns[frame_idx]:
                        sample_id = f"{total_written:08d}"
                        add_to_tar(current_tar, sample_id, img_bytes, [ann])
                        total_written += 1
                        samples_in_shard += 1

                        if samples_in_shard >= sps:
                            open_new_shard()

                        if args.max_samples > 0 and total_written >= args.max_samples:
                            break

                frame_idx += 1

                if args.max_samples > 0 and total_written >= args.max_samples:
                    break

            cap.release()

            if args.max_samples > 0 and total_written >= args.max_samples:
                break

        if args.max_samples > 0 and total_written >= args.max_samples:
            break

        print(f"  Processed chunk {chunk_name} (total samples so far: {total_written})")

    # Close final shard
    if current_tar is not None:
        current_tar.close()
        print(f"  shard {shard_idx - 1}: {os.path.basename(tar_path)} "
              f"({samples_in_shard} samples)")

    print(f"\nDone. Wrote {total_written} samples across {shard_idx} shards "
          f"to {args.dst}")


if __name__ == "__main__":
    main()
