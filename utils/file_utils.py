"""File discovery utilities for converted dataset layouts."""

import os
import re

import cv2


def discover_files(cluster_dir: str):
    """Discover (seq_idx, cam_idx) pairs and their file paths in a cluster dir.

    Returns list of (seq_idx_str, cam_idx_str, hdf5_path, video_path).
    """
    pattern = re.compile(r"^(\d{6})_label_(\d{2})\.hdf5$")
    results = []
    for fname in sorted(os.listdir(cluster_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        seq_idx, cam_idx = m.group(1), m.group(2)
        hdf5_path = os.path.join(cluster_dir, fname)
        video_path = os.path.join(cluster_dir,
                                  f"{seq_idx}_video_{cam_idx}.mp4")
        if os.path.exists(video_path):
            results.append((seq_idx, cam_idx, hdf5_path, video_path))
    return results


def get_video_dimensions(video_path: str):
    """Return (width, height) of a video file."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h
