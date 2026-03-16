"""Depth estimation utilities using MoGe model.

Estimates depth maps on full-resolution video frames, then crops
them using the same crop parameters as RGB videos.
"""

import os
import sys
import subprocess

import cv2
import matplotlib
import numpy as np
import torch

# Add MoGe to path
MOGE_DIR = os.path.join(os.path.dirname(__file__), "..", "thirdparty", "MoGe")
if MOGE_DIR not in sys.path:
    sys.path.insert(0, MOGE_DIR)


def load_moge_model(device: torch.device = None):
    """Load the MoGe-2 depth estimation model.

    Returns the model on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    return model


def estimate_depth_for_video(model, video_path: str,
                             fov_x: float = None,
                             resolution_level: int = 9):
    """Estimate depth for every frame of a video using MoGe.

    Args:
        model: Loaded MoGe model.
        video_path: Path to input video.
        fov_x: Horizontal field of view in degrees. If None, MoGe infers it.
        resolution_level: MoGe resolution level (0-9, higher = more accurate).

    Returns:
        depths: list of (H, W) float32 numpy arrays (metric depth in meters).
    """
    device = model.device
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    depths = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # BGR -> RGB, normalize to [0, 1], to tensor (3, H, W)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb / 255.0, dtype=torch.float32,
                              device=device).permute(2, 0, 1)

        with torch.no_grad():
            kwargs = {"resolution_level": resolution_level}
            if fov_x is not None:
                kwargs["fov_x"] = fov_x
            output = model.infer(tensor, **kwargs)

        depth = output["depth"].cpu().numpy().astype(np.float32)
        depths.append(depth)
        idx += 1

    cap.release()
    return depths


def crop_depth_for_hand(depths: list, centers: np.ndarray,
                        bbox_sizes: np.ndarray, side: str,
                        patch_size: int = 384,
                        rescale_factor: float = 2.5):
    """Crop depth maps using the same crop parameters as RGB.

    Args:
        depths: List of (H, W) float32 depth maps.
        centers: (N, 2) per-frame bbox centers.
        bbox_sizes: (N,) per-frame square bbox sizes.
        side: 'left' or 'right'.
        patch_size: Output crop size.
        rescale_factor: Context expansion factor.

    Returns:
        List of (patch_size, patch_size) float32 cropped depth maps.
    """
    from .image_utils import gen_crop_transform

    do_flip = (side == "left")
    cropped = []

    for i, depth in enumerate(depths):
        if i >= len(centers):
            break
        img_h, img_w = depth.shape[:2]
        src_size = bbox_sizes[i] * rescale_factor
        c_x, c_y = float(centers[i, 0]), float(centers[i, 1])

        d = depth.copy()
        if do_flip:
            d = d[:, ::-1]
            c_x = img_w - c_x - 1

        trans = gen_crop_transform(c_x, c_y, src_size, patch_size)
        patch = cv2.warpAffine(d, trans, (patch_size, patch_size),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        cropped.append(patch)

    return cropped


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None,
                   normalize: bool = True,
                   cmap: str = "Spectral") -> np.ndarray:
    """Colorize a depth map for visualization.

    Uses inverse depth (disparity) with quantile normalization for
    perceptually meaningful coloring.

    Args:
        depth: (H, W) float32 depth map.
        mask: Optional (H, W) bool mask for valid pixels.
        normalize: Whether to normalize disparity to [0, 1].
        cmap: Matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 BGR colorized image.
    """
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1.0 / depth
    if normalize:
        min_disp = np.nanquantile(disp, 0.001)
        max_disp = np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(
        matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    # RGB -> BGR for OpenCV / ffmpeg
    colored = colored[..., ::-1].copy()
    return colored


def save_depth_video(depths: list, output_path: str, fps: float = 30.0):
    """Save cropped depth maps as a lossless metric-depth mp4.

    Depth values (float32 meters) are converted to uint16 millimeters and
    packed into two channels of a BGR frame: B = high byte, G = low byte, R = 0.
    Encoded with libx264rgb CRF 0 (mathematically lossless).

    To decode: depth_mm = B.astype(uint16) * 256 + G.astype(uint16)
               depth_m  = depth_mm / 1000.0
    """
    if not depths:
        return

    h, w = depths[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264rgb",
        "-crf", "0",
        "-pix_fmt", "bgr24",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for depth in depths:
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = (depth_mm >> 8).astype(np.uint8)    # B = high byte
        frame[:, :, 1] = (depth_mm & 0xFF).astype(np.uint8)  # G = low byte
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed for {output_path}: {err}")
