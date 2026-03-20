#!/usr/bin/env python3
"""
Convert ARCTIC dataset into egodex format.

ARCTIC structure:
    ROOT/
        raw_seqs/{subject}/{seq_name}.mano.npy
        raw_seqs/{subject}/{seq_name}.egocam.dist.npy
        meta/misc.json
        cropped_images/{subject}/{seq_name}/{view_idx}/{frame_id:05d}.jpg

Output structure:
    CONVERTED/arctic/{object_name}/
        {seq_idx:06d}_label_{cam_idx:02d}.hdf5
        {seq_idx:06d}_video_{cam_idx:02d}.mp4

Usage:
    python scripts/convert_arctic.py --src ../arctic/downloads/data --dst CONVERTED/arctic
    python scripts/convert_arctic.py --cameras 1 2 3 --max-samples 5
"""

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch

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
    invert_rigid,
    interpolate_joint,
    joints_to_transforms,
    make_transform,
)

# MANO 16-joint to standard 21-joint reorder (adding 5 fingertips from verts)
# MANO internal order: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
# Target 21-joint order: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
MANO_16_TO_21_ORDER = [
    0,           # wrist
    13, 14, 15,  # thumb CMC, MCP, IP
    None,        # thumb tip (vertex)
    1, 2, 3,     # index MCP, PIP, DIP
    None,        # index tip
    4, 5, 6,     # middle MCP, PIP, DIP
    None,        # middle tip
    10, 11, 12,  # ring MCP, PIP, DIP
    None,        # ring tip
    7, 8, 9,     # pinky MCP, PIP, DIP
    None,        # pinky tip
]
TIP_VERT_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky (from smplx)
TIP_DST_INDICES = [4, 8, 12, 16, 20]


class MANOLayer:
    """MANO forward pass using the smplx library for correct LBS."""

    def __init__(self, model_dir, side="right"):
        import smplx
        self.side = side
        is_rhand = (side == "right")
        self.model = smplx.create(
            model_dir, model_type="mano",
            is_rhand=is_rhand, use_pca=False, flat_hand_mean=False,
        )

    @torch.no_grad()
    def __call__(self, rot_np, pose_np, trans_np, betas_np):
        """Forward pass: compute 21 joints in world space.

        Args:
            rot_np: (N, 3) global orientation axis-angle (numpy).
            pose_np: (N, 45) hand pose axis-angle (numpy).
            trans_np: (N, 3) translation (numpy).
            betas_np: (10,) shape parameters (numpy).

        Returns:
            joints21: (N, 21, 3) numpy float32.
        """
        N = rot_np.shape[0]
        out = self.model(
            global_orient=torch.FloatTensor(rot_np),
            hand_pose=torch.FloatTensor(pose_np),
            transl=torch.FloatTensor(trans_np),
            betas=torch.FloatTensor(betas_np).unsqueeze(0).expand(N, -1),
        )

        joints16 = out.joints.numpy()  # (N, 16, 3)
        verts = out.vertices.numpy()   # (N, 778, 3)

        # Tips from vertices
        tips = verts[:, TIP_VERT_IDS]  # (N, 5, 3)

        # Assemble 21 joints
        joints21 = np.zeros((N, 21, 3), dtype=np.float32)
        for k, src_idx in enumerate(MANO_16_TO_21_ORDER):
            if src_idx is not None:
                joints21[:, k] = joints16[:, src_idx]
        for t_idx, dst_idx in enumerate(TIP_DST_INDICES):
            joints21[:, dst_idx] = tips[:, t_idx]

        return joints21


def convert_mano_axisangle_to_rotmat(rot, pose):
    """Convert axis-angle MANO params to rotation matrices."""
    N = rot.shape[0]
    global_orient = np.zeros((N, 3, 3), dtype=np.float32)
    hand_pose = np.zeros((N, 15, 3, 3), dtype=np.float32)

    for i in range(N):
        R, _ = cv2.Rodrigues(rot[i].astype(np.float64))
        global_orient[i] = R.astype(np.float32)
        for j in range(15):
            aa = pose[i, j * 3:(j + 1) * 3].astype(np.float64)
            hand_pose[i, j], _ = cv2.Rodrigues(aa)

    return global_orient, hand_pose


def get_intrinsic_for_view(misc_subject, egocam, view_idx):
    """Get camera intrinsic (3, 3) for a given view.

    For ego (view 0), returns the undistorted intrinsic matrix so that
    it matches the undistorted video frames.
    """
    if view_idx == 0:
        K = np.array(egocam["intrinsics"], dtype=np.float64)
        dist8 = np.array(egocam["dist8"], dtype=np.float64)
        # Image size: ego is 2800x2000 (from misc image_size[0])
        w = int(misc_subject["image_size"][0][0])
        h = int(misc_subject["image_size"][0][1])
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, dist8, (w, h), alpha=0, newImgSize=(w, h))
        return new_K.astype(np.float32)
    else:
        return np.array(misc_subject["intris_mat"][view_idx - 1],
                        dtype=np.float32)


def build_egodex_data_for_sequence(
    mano_data, egocam, misc_subject, view_idx, mano_layers,
):
    """Build world-space transforms for one ARCTIC sequence+camera."""
    N = mano_data["right"]["rot"].shape[0]
    identity = np.eye(4, dtype=np.float32)
    intrinsic = get_intrinsic_for_view(misc_subject, egocam, view_idx)

    transforms_dict = {}
    confidences_dict = {}
    conf_ones = np.ones(N, dtype=np.float32)
    conf_zeros = np.zeros(N, dtype=np.float32)

    # Camera transform: cam2world
    if view_idx == 0:
        cam_poses = np.zeros((N, 4, 4), dtype=np.float32)
        for i in range(N):
            R = egocam["R_k_cam_np"][i]
            T = egocam["T_k_cam_np"][i]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R.astype(np.float32)
            w2c[:3, 3] = T.squeeze().astype(np.float32)
            cam_poses[i] = invert_rigid(w2c)
    else:
        w2c = np.array(misc_subject["world2cam"][view_idx - 1],
                        dtype=np.float32)
        cam2world = invert_rigid(w2c)
        cam_poses = np.tile(cam2world, (N, 1, 1))

    transforms_dict["camera"] = cam_poses

    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (N, 1, 1))
        confidences_dict[name] = conf_zeros.copy()

    mano_dicts = []
    for side in ["right", "left"]:
        side_data = mano_data[side]
        rot = side_data["rot"]
        pose = side_data["pose"]
        trans = side_data["trans"]
        shape = side_data["shape"]

        joints_world = mano_layers[side](rot, pose, trans, shape)

        all_transforms = np.zeros((N, 21, 4, 4), dtype=np.float32)
        for i in range(N):
            all_transforms[i] = joints_to_transforms(joints_world[i])

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            transforms_dict[name] = all_transforms[:, mano_idx]
            confidences_dict[name] = conf_ones.copy()

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            mc = np.zeros((N, 4, 4), dtype=np.float32)
            for i in range(N):
                pos = interpolate_joint(joints_world[i], idx_a, idx_b,
                                        alpha=0.3)
                direction = joints_world[i, idx_b] - joints_world[i, idx_a]
                mc[i] = make_transform(pos, direction)
            transforms_dict[name] = mc
            confidences_dict[name] = conf_ones.copy()

        global_orient, hand_pose = convert_mano_axisangle_to_rotmat(rot, pose)

        kpt3d = np.zeros((N, 21, 3), dtype=np.float32)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

        mano_dicts.append({
            "betas": shape.astype(np.float32),
            "global_orient_worldspace": global_orient,
            "hand_pose": hand_pose,
            "transl_worldspace": trans.astype(np.float32),
            "kpt3d": kpt3d,
            "side": side,
        })

    return intrinsic, transforms_dict, confidences_dict, mano_dicts


def collect_image_paths(images_dir, seq_name, view_idx, ioi_offset,
                        num_frames):
    """Collect sorted image paths for a sequence/view."""
    view_dir = os.path.join(images_dir, seq_name, str(view_idx))
    paths = []
    for frame_idx in range(num_frames):
        img_id = ioi_offset + frame_idx
        img_path = os.path.join(view_dir, f"{img_id:05d}.jpg")
        paths.append(img_path)
    return paths


def undistort_images_to_mp4(image_paths, output_path, K, dist8, fps=30.0):
    """Undistort ego camera images and encode to mp4.

    Args:
        image_paths: list of image file paths.
        K: (3, 3) camera intrinsic matrix (for distorted images).
        dist8: (8,) distortion coefficients.
        fps: video frame rate.
    """
    if not image_paths:
        return

    first = cv2.imread(image_paths[0])
    h, w = first.shape[:2]

    # Compute undistortion maps once (new camera matrix = same as original)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist8, (w, h), alpha=0,
                                                newImgSize=(w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist8, None, new_K,
                                              (w, h), cv2.CV_32FC1)

    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for p in image_paths:
        frame = cv2.imread(p)
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        proc.stdin.write(undistorted.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed for {output_path}: {err}")


def _verify_output(out_dir, seq_idx, cam_idx, expected_frames):
    """Verify the converted output."""
    prefix = f"{seq_idx:06d}"
    hdf5_path = os.path.join(out_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
    with h5py.File(hdf5_path, "r") as f:
        sample_key = list(f["transforms"].keys())[0]
        if sample_key == "gravity":
            sample_key = list(f["transforms"].keys())[1]
        hdf5_frames = f[f"transforms/{sample_key}"].shape[0]
        if hdf5_frames != expected_frames:
            print(f"  WARNING: HDF5 has {hdf5_frames} frames, "
                  f"expected {expected_frames}")

    video_path = os.path.join(out_dir, f"{prefix}_video_{cam_idx:02d}.mp4")
    if not os.path.exists(video_path):
        print(f"  WARNING: {os.path.basename(video_path)} not created")
        return
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_frames != expected_frames:
        print(f"  WARNING: {os.path.basename(video_path)} has "
              f"{video_frames} frames, expected {expected_frames}")
    else:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  OK: {os.path.basename(video_path)} "
              f"({video_frames} frames, {size_mb:.1f} MB)")


def convert_arctic(src_dir, dst_dir, cameras=None, fps=30.0,
                   max_samples=0, mano_model_dir=None,
                   subject_filter=None):
    """Convert ARCTIC sequences to egodex format, clustered by object."""
    os.makedirs(dst_dir, exist_ok=True)

    meta_path = os.path.join(src_dir, "meta", "misc.json")
    with open(meta_path) as f:
        misc = json.load(f)

    if mano_model_dir is None:
        candidates = [
            os.path.join(src_dir, "..", "unpack", "body_models"),
            "/rlwrld3/home/seungjun/arctic/unpack/body_models",
        ]
        for c in candidates:
            if os.path.isdir(c):
                mano_model_dir = c
                break
        if mano_model_dir is None:
            raise FileNotFoundError(
                "MANO model directory not found. Use --mano-model-dir")

    mano_layers = {
        "right": MANOLayer(mano_model_dir, "right"),
        "left": MANOLayer(mano_model_dir, "left"),
    }
    print(f"Loaded MANO models from {mano_model_dir}")

    raw_seqs_dir = os.path.join(src_dir, "raw_seqs")
    images_dir = os.path.join(src_dir, "images")

    clusters = {}
    all_subjects = sorted([d for d in os.listdir(raw_seqs_dir)
                           if os.path.isdir(os.path.join(raw_seqs_dir, d))])
    if subject_filter:
        all_subjects = [s for s in all_subjects if s in subject_filter]
    for subject in all_subjects:
        subject_dir = os.path.join(raw_seqs_dir, subject)
        mano_files = sorted([f for f in os.listdir(subject_dir)
                             if f.endswith(".mano.npy")])
        for mano_file in mano_files:
            seq_name = mano_file.replace(".mano.npy", "")
            obj_name = seq_name.split("_")[0]
            clusters.setdefault(obj_name, []).append((subject, seq_name))

    if cameras is None:
        cameras = list(range(9))

    global_count = 0
    for obj_name in sorted(clusters.keys()):
        sequences = clusters[obj_name]
        obj_dir = os.path.join(dst_dir, obj_name)
        os.makedirs(obj_dir, exist_ok=True)

        for seq_idx, (subject, seq_name) in enumerate(sequences):
            if max_samples > 0 and global_count >= max_samples:
                break

            mano_path = os.path.join(raw_seqs_dir, subject,
                                     f"{seq_name}.mano.npy")
            ego_path = os.path.join(raw_seqs_dir, subject,
                                    f"{seq_name}.egocam.dist.npy")

            mano_data = np.load(mano_path, allow_pickle=True).item()
            egocam = np.load(ego_path, allow_pickle=True).item()
            misc_subject = misc[subject]
            ioi_offset = misc_subject["ioi_offset"]
            num_frames = mano_data["right"]["rot"].shape[0]

            any_camera_ok = False
            for cam_idx in cameras:
                if cam_idx > 8:
                    continue

                img_dir = os.path.join(images_dir, subject, seq_name,
                                       str(cam_idx))
                if not os.path.isdir(img_dir):
                    print(f"  Skipping {subject}/{seq_name} cam {cam_idx}: "
                          f"images not found")
                    continue

                intrinsic, transforms_dict, confidences_dict, mano_dicts = \
                    build_egodex_data_for_sequence(
                        mano_data, egocam, misc_subject, cam_idx,
                        mano_layers,
                    )

                print(f"[{obj_name}/{seq_idx:06d}] {subject}/{seq_name} "
                      f"(cam={cam_idx:02d}, frames={num_frames})")

                prefix = f"{seq_idx:06d}"

                hdf5_path = os.path.join(
                    obj_dir, f"{prefix}_label_{cam_idx:02d}.hdf5")
                write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                                  confidences_dict,
                                  mano_dict=mano_dicts[0])

                with h5py.File(hdf5_path, "a") as f:
                    extra = mano_dicts[1]
                    grp = f.create_group(f"mano_{extra['side']}")
                    grp.create_dataset(
                        "betas",
                        data=extra["betas"].astype(np.float32))
                    grp.create_dataset(
                        "global_orient_worldspace",
                        data=extra["global_orient_worldspace"].astype(
                            np.float32))
                    grp.create_dataset(
                        "hand_pose",
                        data=extra["hand_pose"].astype(np.float32))
                    grp.create_dataset(
                        "transl_worldspace",
                        data=extra["transl_worldspace"].astype(np.float32))
                    grp.create_dataset(
                        "kpt3d",
                        data=extra["kpt3d"].astype(np.float32))

                image_paths = collect_image_paths(
                    os.path.join(images_dir, subject),
                    seq_name, cam_idx, ioi_offset, num_frames)

                valid_paths = [p for p in image_paths if os.path.exists(p)]
                if len(valid_paths) != num_frames:
                    print(f"  WARNING: found {len(valid_paths)}/{num_frames} "
                          f"images for cam {cam_idx}")

                if valid_paths:
                    rgb_path = os.path.join(
                        obj_dir, f"{prefix}_video_{cam_idx:02d}.mp4")
                    if cam_idx == 0:
                        # Ego camera: undistort frames
                        ego_K = np.array(egocam["intrinsics"],
                                         dtype=np.float64)
                        ego_dist = np.array(egocam["dist8"],
                                            dtype=np.float64)
                        undistort_images_to_mp4(
                            valid_paths, rgb_path, ego_K, ego_dist,
                            fps=fps)
                    else:
                        images_to_mp4(valid_paths, rgb_path, fps=fps)

                _verify_output(obj_dir, seq_idx, cam_idx, num_frames)
                any_camera_ok = True

            if any_camera_ok:
                global_count += 1

        if max_samples > 0 and global_count >= max_samples:
            break

    cam_str = "all 9" if len(cameras) == 9 else str(len(cameras))
    print(f"\nDone. Converted {global_count} sequences "
          f"({cam_str} cam(s) each) to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ARCTIC to egodex format")
    parser.add_argument("--src", default="../arctic/downloads/data",
                        help="ARCTIC data directory")
    parser.add_argument("--dst", default="CONVERTED/arctic",
                        help="Output directory")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Camera indices (0=ego, 1-8=allocentric, "
                             "default: all 9)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS (default: 30)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max sequences to convert (0=all)")
    parser.add_argument("--mano-model-dir", type=str, default=None,
                        help="Path to directory containing mano/ subdir")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Filter to specific subjects (e.g. s01 s02)")
    args = parser.parse_args()

    convert_arctic(args.src, args.dst, cameras=args.cameras,
                   fps=args.fps, max_samples=args.max_samples,
                   mano_model_dir=args.mano_model_dir,
                   subject_filter=args.subjects)


if __name__ == "__main__":
    main()
