#!/usr/bin/env python3
"""
Convert WHIM dataset into egodex format.

Downloads YouTube videos one by one, extracts annotated frames,
and converts hand annotations to egodex HDF5 format.

WHIM annotation format (per-frame .npy):
    Array of N dicts, one per detected hand:
        bbox: (4,) [x1, y1, x2, y2]
        joints_3d: (21, 3) MANO joints in local space
        side: scalar (0.0=right, 1.0=left)
        trans: (3,) translation to camera space
        K: (3, 3) camera intrinsics
        mano: dict with global_orient (1,3,3), hand_pose (15,3,3), betas (10,)

    Camera-space joints: joints_3d + trans
    cam_pose = identity (no extrinsics, world = camera space)

Multi-hand policy:
    Keep at most one hand per side (left/right) per frame.
    When multiple same-side hands exist, keep the largest bbox.

Output structure:
    CONVERTED/whim_train/
        {mode}/
            {seq_idx:06d}_label_00.hdf5
            {seq_idx:06d}_video_00.mp4
    Tracking files in dst:
        completed_train.json / completed_test.json

Usage:
    python scripts/convert_whim.py --src ../WiLoR --dst CONVERTED/whim_train --mode train
    python scripts/convert_whim.py --src ../WiLoR --dst CONVERTED/whim_test --mode test
"""

import argparse
import json
import os
import sys

import cv2
import h5py
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
)

SIDE_MAP = {0.0: "left", 1.0: "right"}


def load_completed(path: str) -> set:
    """Load set of completed video IDs from JSON."""
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_completed(path: str, completed: set):
    """Save completed video IDs to JSON."""
    with open(path, "w") as f:
        json.dump(sorted(completed), f, indent=2)


def download_video(video_id: str, video_info: dict, output_dir: str,
                    timeout: int = 300) -> str:
    """Download a YouTube video with timeout. Returns path to downloaded file.

    Raises RuntimeError with message starting with:
        "UNAVAILABLE:" — video is permanently gone, should not retry
        "TIMEOUT:" — download too slow, may succeed on retry
        "BOT:" — rate-limited by YouTube, retry later
        Other — transient errors, retry possible
    """
    import multiprocessing
    import multiprocessing.queues
    from queue import Empty

    out_path = os.path.join(output_dir, f"{video_id}.mp4")
    if os.path.exists(out_path):
        return out_path

    os.makedirs(output_dir, exist_ok=True)

    def _resolve_and_download(video_id, video_info, output_dir, result_q):
        """Resolve stream and download in a subprocess (handles hangs)."""
        try:
            from pytubefix import YouTube
            res = video_info["res"][0]
            yt = YouTube(f"https://youtu.be/{video_id}")
            stream = yt.streams.filter(
                only_video=True, file_extension="mp4",
                video_codec="avc1", res=f"{res}p",
            ).first()
            if stream is None:
                stream = yt.streams.filter(
                    only_video=True, file_extension="mp4", video_codec="avc1",
                ).order_by("resolution").desc().first()
            if stream is None:
                stream = yt.streams.filter(
                    only_video=True, file_extension="mp4",
                ).order_by("resolution").desc().first()
            if stream is None:
                result_q.put(("UNAVAILABLE", f"no suitable stream for {video_id}"))
                return
            stream.download(output_path=output_dir, filename=f"{video_id}.mp4")
            result_q.put(("OK", ""))
        except Exception as e:
            msg = str(e)
            if "unavailable" in msg.lower() or "not available" in msg.lower():
                result_q.put(("UNAVAILABLE", msg))
            elif "bot" in msg.lower():
                result_q.put(("BOT", msg))
            else:
                result_q.put(("ERROR", msg))

    result_q = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_resolve_and_download,
        args=(video_id, video_info, output_dir, result_q))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f"TIMEOUT: exceeded {timeout}s")

    # Get result from subprocess
    try:
        status, msg = result_q.get_nowait()
    except Empty:
        status, msg = "ERROR", f"process exited with code {proc.exitcode}"

    if status != "OK":
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f"{status}: {msg}")

    if not os.path.exists(out_path):
        raise RuntimeError(f"Download completed but file not found")

    return out_path


def filter_hands_per_frame(raw_anno):
    """Filter annotations: keep at most one hand per side (largest bbox).

    Args:
        raw_anno: numpy object array of hand dicts from .npy file.

    Returns:
        Dict mapping side ('left'/'right') -> hand dict.
    """
    best = {}  # side -> (area, hand_dict)
    for entry in raw_anno:
        if isinstance(entry, dict):
            hand = entry
        else:
            hand = entry.item() if hasattr(entry, "item") else entry

        side_val = float(np.array(hand["side"]).flat[0])
        side = SIDE_MAP.get(side_val)
        if side is None:
            continue

        bbox = hand["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if side not in best or area > best[side][0]:
            best[side] = (area, hand)

    return {side: hand for side, (_, hand) in best.items()}


def build_egodex_data(
    anno_dir: str,
    video_path: str,
    video_info: dict,
):
    """Build egodex data from WHIM annotations for one video.

    Returns:
        intrinsic: (3, 3)
        transforms_dict, confidences_dict
        valid_frame_indices: list of original frame numbers (for video extraction)
        mano_dicts: list of mano_dict per active side
        fps_rate: int, ratio between downloaded and original fps
    """
    # Collect annotated frame numbers
    anno_files = sorted([
        f for f in os.listdir(anno_dir) if f.endswith(".npy")
    ])
    if not anno_files:
        return None

    # Parse annotations: filter to largest bbox per side per frame
    # frames_data[frame_num] = {side: hand_dict}
    frames_data = {}
    for fname in anno_files:
        frame_num = int(fname[:-4])
        raw = np.load(os.path.join(anno_dir, fname), allow_pickle=True)
        filtered = filter_hands_per_frame(raw)
        if filtered:
            frames_data[frame_num] = filtered

    if not frames_data:
        return None

    sorted_frames = sorted(frames_data.keys())
    M = len(sorted_frames)

    # Determine which sides are active across the sequence
    side_frame_count = {"left": 0, "right": 0}
    for frame_num in sorted_frames:
        for side in frames_data[frame_num]:
            side_frame_count[side] += 1
    active_sides = [s for s, c in side_frame_count.items() if c > 0]

    # Use intrinsics from first frame (consistent within a video)
    first_hands = frames_data[sorted_frames[0]]
    first_hand = next(iter(first_hands.values()))
    intrinsic = first_hand["K"].astype(np.float32)

    # Camera pose = identity (no extrinsics, world = camera)
    cam_pose = np.eye(4, dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)

    transforms_dict = {}
    confidences_dict = {}

    # Camera transform (identity, repeated)
    transforms_dict["camera"] = np.tile(cam_pose, (M, 1, 1))

    # Body joints (not available)
    for name in BODY_JOINTS:
        transforms_dict[name] = np.tile(identity, (M, 1, 1))
        confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Per-side data collection for MANO
    side_data = {}
    for side in active_sides:
        side_data[side] = {
            "joints_cam": np.zeros((M, 21, 3), dtype=np.float32),
            "global_orient": np.zeros((M, 3, 3), dtype=np.float32),
            "hand_pose": np.zeros((M, 15, 3, 3), dtype=np.float32),
            "transl": np.zeros((M, 3), dtype=np.float32),
            "betas_list": [],
            "conf": np.zeros(M, dtype=np.float32),
        }

    # Fill per-frame data
    for i, frame_num in enumerate(sorted_frames):
        hands = frames_data[frame_num]
        for side in active_sides:
            if side not in hands:
                continue
            hand = hands[side]
            j3d = hand["joints_3d"].astype(np.float32)
            trans = hand["trans"].astype(np.float32)
            mano = hand["mano"]

            sd = side_data[side]
            sd["joints_cam"][i] = j3d + trans
            sd["global_orient"][i] = mano["global_orient"].reshape(3, 3)
            sd["hand_pose"][i] = mano["hand_pose"]
            sd["transl"][i] = trans
            sd["betas_list"].append(mano["betas"].astype(np.float32))
            sd["conf"][i] = 1.0

    # Build transforms for each side
    for side in ["left", "right"]:
        is_active = side in active_sides

        if is_active:
            sd = side_data[side]
            joints_cam = sd["joints_cam"]

            # Convert joints to 4x4 transforms (already in camera/world space)
            all_transforms = np.zeros((M, 21, 4, 4), dtype=np.float32)
            for i in range(M):
                if sd["conf"][i] > 0:
                    all_transforms[i] = joints_to_transforms(joints_cam[i])
                else:
                    all_transforms[i] = np.tile(identity, (21, 1, 1))

        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            if is_active:
                transforms_dict[name] = all_transforms[:, mano_idx]
                confidences_dict[name] = sd["conf"].copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

        for suffix, (idx_a, idx_b) in METACARPAL_INTERPOLATION.items():
            name = f"{side}{suffix}"
            if is_active:
                mc = np.zeros((M, 4, 4), dtype=np.float32)
                for i in range(M):
                    if sd["conf"][i] > 0:
                        pos = interpolate_joint(joints_cam[i], idx_a, idx_b, alpha=0.3)
                        direction = joints_cam[i, idx_b] - joints_cam[i, idx_a]
                        mc[i] = make_transform(pos, direction)
                    else:
                        mc[i] = identity
                transforms_dict[name] = mc
                confidences_dict[name] = sd["conf"].copy()
            else:
                transforms_dict[name] = np.tile(identity, (M, 1, 1))
                confidences_dict[name] = np.zeros(M, dtype=np.float32)

    # Build MANO dicts
    mano_dicts = []
    for side in active_sides:
        sd = side_data[side]
        # Median betas across frames for stability
        if sd["betas_list"]:
            betas = np.median(np.stack(sd["betas_list"]), axis=0).astype(np.float32)
        else:
            betas = np.zeros(10, dtype=np.float32)

        kpt3d = np.zeros((M, 21, 3), dtype=np.float32)
        for mano_idx, suffix in MANO_TO_EGODEX_SUFFIX.items():
            name = f"{side}{suffix}"
            kpt3d[:, mano_idx] = transforms_dict[name][:, :3, 3]

        mano_dicts.append({
            "betas": betas,
            "global_orient_worldspace": sd["global_orient"],
            "hand_pose": sd["hand_pose"],
            "transl_worldspace": sd["transl"],
            "kpt3d": kpt3d,
            "side": side,
        })

    # Compute fps_rate for frame extraction
    cap = cv2.VideoCapture(video_path)
    dl_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    orig_fps = video_info["fps"]
    fps_rate = round(dl_fps / orig_fps) if orig_fps > 0 else 1

    return (intrinsic, transforms_dict, confidences_dict,
            sorted_frames, mano_dicts, fps_rate, dl_fps)


def extract_frames_to_mp4(
    video_path: str, output_path: str,
    frame_numbers: list, fps_rate: int, fps: float,
):
    """Extract specific frames from video and encode as mp4.

    Reads sequentially through the video and picks annotated frames,
    avoiding costly random seeks.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return False

    # Read first frame to get dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0] * fps_rate)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return False
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Build set of target video frame indices
    target_indices = {fn * fps_rate for fn in frame_numbers}
    max_target = max(target_indices)

    def frames():
        idx = 0
        while idx <= max_target:
            ret, img = cap.read()
            if not ret:
                break
            if idx in target_indices:
                yield img if img is not None else np.zeros((h, w, 3), dtype=np.uint8)
            idx += 1

    _pipe_frames_to_ffmpeg(frames(), output_path, fps, w, h)
    cap.release()
    return True


def convert_whim(src_dir: str, dst_dir: str, mode: str,
                 fps: float = 30.0, max_samples: int = 0):
    """Convert WHIM videos to egodex format."""
    # Load video list
    video_ids_path = os.path.join(src_dir, "whim", f"{mode}_video_ids.json")
    with open(video_ids_path) as f:
        video_dict = json.load(f)

    anno_base = os.path.join(src_dir, "whim_data", mode, "anno")
    videos_dir = os.path.join(src_dir, "Videos")
    out_dir = os.path.join(dst_dir, mode)
    os.makedirs(out_dir, exist_ok=True)

    # Load/save completed and failed tracking
    completed_path = os.path.join(dst_dir, f"completed_{mode}.json")
    failed_path = os.path.join(dst_dir, f"failed_{mode}.json")
    completed = load_completed(completed_path)
    failed_set = load_completed(failed_path)

    video_ids = sorted(video_dict.keys())
    total = len(video_ids)
    converted = 0

    for seq_idx, video_id in enumerate(video_ids):
        if max_samples > 0 and converted >= max_samples:
            break

        if video_id in completed or video_id in failed_set:
            if video_id in completed:
                converted += 1
            continue

        anno_dir = os.path.join(anno_base, video_id)
        if not os.path.isdir(anno_dir):
            continue

        video_info = video_dict[video_id]
        print(f"[{seq_idx:04d}/{total}] {video_id} ...", flush=True)

        # Download video
        try:
            video_path = download_video(video_id, video_info, videos_dir)
            print(f"  Downloaded.", flush=True)
        except Exception as e:
            msg = str(e)
            print(f"  FAILED download: {msg}")
            # Only permanently skip truly unavailable videos
            if msg.startswith("UNAVAILABLE:"):
                failed_set.add(video_id)
                save_completed(failed_path, failed_set)
            # BOT/TIMEOUT/other errors: don't save to failed, will retry next run
            continue

        # Build egodex data
        result = build_egodex_data(anno_dir, video_path, video_info)
        if result is None:
            print(f"  Skipping: no valid annotations")
            completed.add(video_id)
            save_completed(completed_path, completed)
            continue

        (intrinsic, transforms_dict, confidences_dict,
         sorted_frames, mano_dicts, fps_rate, dl_fps) = result

        n_frames = len(sorted_frames)
        active_sides = [d["side"] for d in mano_dicts]
        print(f"  frames={n_frames}, hands={active_sides}, fps_rate={fps_rate}")

        prefix = f"{seq_idx:06d}"

        # Write HDF5
        mano_dict = mano_dicts[0] if mano_dicts else None
        hdf5_path = os.path.join(out_dir, f"{prefix}_label_00.hdf5")
        write_egodex_hdf5(hdf5_path, intrinsic, transforms_dict,
                          confidences_dict, mano_dict=mano_dict)

        # Write additional MANO groups
        if len(mano_dicts) > 1:
            with h5py.File(hdf5_path, "a") as f:
                for extra in mano_dicts[1:]:
                    side = extra["side"]
                    grp = f.create_group(f"mano_{side}")
                    for key in ["betas", "global_orient_worldspace",
                                "hand_pose", "transl_worldspace", "kpt3d"]:
                        grp.create_dataset(key, data=extra[key].astype(np.float32))

        # Extract video frames
        rgb_path = os.path.join(out_dir, f"{prefix}_video_00.mp4")
        ok = extract_frames_to_mp4(
            video_path, rgb_path, sorted_frames, fps_rate, fps)
        if not ok:
            print(f"  WARNING: video extraction failed")

        # Verify
        cap = cv2.VideoCapture(rgb_path)
        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if vid_frames == n_frames:
            size_mb = os.path.getsize(rgb_path) / (1024 * 1024)
            print(f"  OK: {os.path.basename(rgb_path)} ({vid_frames} frames, {size_mb:.1f} MB)")
        else:
            print(f"  WARNING: video has {vid_frames} frames, expected {n_frames}")

        completed.add(video_id)
        save_completed(completed_path, completed)
        converted += 1

    print(f"\nDone. Converted {converted} videos, {len(failed_set)} failed.")
    if failed_set:
        print(f"Failed: {sorted(failed_set)}")


def main():
    parser = argparse.ArgumentParser(description="Convert WHIM to egodex format")
    parser.add_argument("--src", default="../WiLoR",
                        help="WiLoR root directory")
    parser.add_argument("--dst", default="CONVERTED/whim",
                        help="Output directory")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Train or test split")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max videos to convert (0=all)")
    args = parser.parse_args()

    convert_whim(args.src, args.dst, mode=args.mode,
                 fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
