"""Image cropping utilities for hand-centric crops from full frames.

Projects 3D keypoints to 2D using camera intrinsics and camera pose,
computes bounding boxes, and crops/resizes following HaMeR conventions.
"""

import cv2
import numpy as np
from skimage.filters import gaussian


def project_3d_to_2d(positions_world: np.ndarray, cam_pose: np.ndarray,
                     intrinsic: np.ndarray) -> np.ndarray:
    """Project world-space 3D points to 2D pixel coordinates.

    Args:
        positions_world: (N, 3) world-space positions.
        cam_pose: (4, 4) camera-to-world transform.
        intrinsic: (3, 3) camera intrinsic matrix.

    Returns:
        (N, 2) pixel coordinates.
    """
    # World to camera: inv(cam_pose) @ points
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    # cam_from_world: R^T @ (p - t)
    positions_cam = (positions_world - t) @ R  # (N, 3)

    # Project: K @ p_cam
    z = positions_cam[:, 2:3]
    z = np.clip(z, 1e-6, None)
    xy_norm = positions_cam[:, :2] / z  # (N, 2)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    pixels = np.zeros((len(positions_world), 2), dtype=np.float32)
    pixels[:, 0] = xy_norm[:, 0] * fx + cx
    pixels[:, 1] = xy_norm[:, 1] * fy + cy
    return pixels


def bbox_from_keypoints(keypoints_2d: np.ndarray) -> np.ndarray:
    """Compute [x1, y1, x2, y2] bounding box from 2D keypoints.

    Args:
        keypoints_2d: (N, 2) pixel coordinates.

    Returns:
        (4,) array [x_min, y_min, x_max, y_max].
    """
    x_min = keypoints_2d[:, 0].min()
    y_min = keypoints_2d[:, 1].min()
    x_max = keypoints_2d[:, 0].max()
    y_max = keypoints_2d[:, 1].max()
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def expand_to_square(bbox: np.ndarray) -> np.ndarray:
    """Expand a bounding box to a square by enlarging the shorter side.

    Args:
        bbox: (4,) array [x1, y1, x2, y2].

    Returns:
        (4,) square bounding box centered on the original.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    size = max(w, h)
    return np.array([
        cx - size / 2.0, cy - size / 2.0,
        cx + size / 2.0, cy + size / 2.0,
    ], dtype=np.float32)


def gen_crop_transform(c_x: float, c_y: float,
                       src_size: float, dst_size: float,
                       do_flip: bool = False,
                       img_width: int = 0) -> np.ndarray:
    """Generate an affine transformation matrix for cropping.

    Args:
        c_x: Bbox center x in the (possibly flipped) image.
        c_y: Bbox center y.
        src_size: Source square bbox size (after rescale).
        dst_size: Output patch size.
        do_flip: Whether the image was flipped (for left hand).
        img_width: Original image width (needed if do_flip=True).

    Returns:
        (2, 3) affine transformation matrix.
    """
    half_src = src_size / 2.0
    half_dst = dst_size / 2.0

    src = np.array([
        [c_x, c_y],
        [c_x, c_y + half_src],
        [c_x + half_src, c_y],
    ], dtype=np.float32)

    dst = np.array([
        [half_dst, half_dst],
        [half_dst, dst_size],
        [dst_size, half_dst],
    ], dtype=np.float32)

    return cv2.getAffineTransform(src, dst)


def crop_image(img: np.ndarray, center: np.ndarray, bbox_size: float,
               patch_size: int = 384, rescale_factor: float = 2.5,
               do_flip: bool = False) -> np.ndarray:
    """Crop and resize an image region around a hand.

    Following HaMeR conventions:
    1. Expand bbox by rescale_factor for context.
    2. Anti-alias with Gaussian blur if downsampling significantly.
    3. Affine-warp to (patch_size, patch_size).
    4. Optionally flip for left hands.

    Args:
        img: (H, W, 3) input image.
        center: (2,) bbox center [cx, cy] in pixel coordinates.
        bbox_size: Square bbox side length (before rescale).
        patch_size: Output size in pixels.
        rescale_factor: Context expansion factor.
        do_flip: Flip horizontally (for left hand).

    Returns:
        (patch_size, patch_size, 3) cropped image.
    """
    img_h, img_w = img.shape[:2]
    src_size = bbox_size * rescale_factor

    cvimg = img.copy()
    c_x, c_y = float(center[0]), float(center[1])

    if do_flip:
        cvimg = cvimg[:, ::-1, :]
        c_x = img_w - c_x - 1

    # Anti-aliasing blur when downsampling significantly
    downsampling_factor = (src_size / patch_size) / 2.0
    if downsampling_factor > 1.1:
        cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2,
                         channel_axis=2, preserve_range=True).astype(np.uint8)

    trans = gen_crop_transform(c_x, c_y, src_size, patch_size)
    patch = cv2.warpAffine(cvimg, trans, (patch_size, patch_size),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    return patch


def compute_per_frame_crops(positions: np.ndarray, cam_poses: np.ndarray,
                            intrinsic: np.ndarray):
    """Compute per-frame bbox center and size from 3D keypoints.

    Args:
        positions: (N, J, 3) world-space joint positions.
        cam_poses: (N, 4, 4) camera-to-world transforms.
        intrinsic: (3, 3) camera intrinsic matrix.

    Returns:
        centers: (N, 2) bbox centers in pixel coordinates.
        bbox_sizes: (N,) square bbox side lengths.
    """
    N = positions.shape[0]
    centers = np.zeros((N, 2), dtype=np.float32)
    bbox_sizes = np.zeros(N, dtype=np.float32)

    for i in range(N):
        kp_2d = project_3d_to_2d(positions[i], cam_poses[i], intrinsic)
        bbox = bbox_from_keypoints(kp_2d)
        sq = expand_to_square(bbox)
        centers[i] = (sq[:2] + sq[2:]) / 2.0
        bbox_sizes[i] = sq[2] - sq[0]

    return centers, bbox_sizes


def compute_cropped_intrinsics(intrinsic: np.ndarray, centers: np.ndarray,
                               bbox_sizes: np.ndarray, side: str,
                               img_w: int, patch_size: int = 384,
                               rescale_factor: float = 2.5):
    """Compute per-frame intrinsics for the cropped image.

    The crop applies: p_crop = A @ p_original (for right hand)
                  or: p_crop = A @ F @ p_original (for left hand, F = flip)
    So: K_crop = A_3x3 @ K  or  K_crop = A_3x3 @ F @ K

    Args:
        intrinsic: (3, 3) original camera intrinsic matrix.
        centers: (N, 2) per-frame bbox centers in original image.
        bbox_sizes: (N,) per-frame square bbox sizes (before rescale).
        side: 'left' or 'right'.
        img_w: Original image width (needed for flip).
        patch_size: Output crop size.
        rescale_factor: Context expansion factor.

    Returns:
        List of N dicts, each with {model, image_width, image_height, fx, fy, cx, cy}.
    """
    do_flip = (side == "left")
    N = len(centers)
    results = []

    # Flip matrix (mirrors x-axis)
    F = np.array([
        [-1, 0, img_w - 1],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    for i in range(N):
        src_size = bbox_sizes[i] * rescale_factor
        c_x, c_y = float(centers[i, 0]), float(centers[i, 1])

        if do_flip:
            c_x = img_w - c_x - 1

        trans = gen_crop_transform(c_x, c_y, src_size, patch_size)
        A = np.vstack([trans, [0, 0, 1]]).astype(np.float64)

        if do_flip:
            K_crop = A @ F @ intrinsic.astype(np.float64)
        else:
            K_crop = A @ intrinsic.astype(np.float64)

        results.append({
            "model": "pinhole",
            "image_width": patch_size,
            "image_height": patch_size,
            "fx": float(K_crop[0, 0]),
            "fy": float(K_crop[1, 1]),
            "cx": float(K_crop[0, 2]),
            "cy": float(K_crop[1, 2]),
        })

    return results


def crop_video_for_hand(video_path: str, output_path: str, centers: np.ndarray,
                        bbox_sizes: np.ndarray, side: str,
                        patch_size: int = 384, rescale_factor: float = 2.5,
                        fps: float = 30.0):
    """Read a video, crop each frame around a hand, and save as mp4.

    Args:
        video_path: Source video path.
        output_path: Output cropped video path.
        centers: (N, 2) per-frame bbox centers.
        bbox_sizes: (N,) per-frame square bbox sizes.
        side: 'left' or 'right' (left hands are flipped).
        patch_size: Output size in pixels.
        rescale_factor: Context expansion factor.
        fps: Output video fps.
    """
    from .io import _pipe_frames_to_ffmpeg

    do_flip = (side == "left")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return

    def frames():
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < len(centers):
                patch = crop_image(frame, centers[idx], bbox_sizes[idx],
                                   patch_size=patch_size,
                                   rescale_factor=rescale_factor,
                                   do_flip=do_flip)
                yield patch
            idx += 1

    _pipe_frames_to_ffmpeg(frames(), output_path, fps, patch_size, patch_size)
    cap.release()
