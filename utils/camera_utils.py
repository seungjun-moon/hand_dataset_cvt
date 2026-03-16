"""Camera-related utilities: intrinsics, poses, and hand side detection."""

import h5py
import numpy as np


# Hand joint suffixes for bbox/position loading (excluding metacarpals/body)
HAND_JOINT_SUFFIXES = [
    "Hand",
    "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip",
    "IndexFingerKnuckle", "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip", "IndexFingerTip",
    "MiddleFingerKnuckle", "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip", "MiddleFingerTip",
    "RingFingerKnuckle", "RingFingerIntermediateBase",
    "RingFingerIntermediateTip", "RingFingerTip",
    "LittleFingerKnuckle", "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip", "LittleFingerTip",
]


def get_active_sides(hdf5_path: str):
    """Return list of sides ('left', 'right') that have nonzero confidence."""
    sides = []
    with h5py.File(hdf5_path, "r") as f:
        for side in ["left", "right"]:
            key = f"{side}Hand"
            if key in f["confidences"]:
                conf = f["confidences"][key][:]
                if np.any(conf > 0):
                    sides.append(side)
    return sides


def load_hand_positions(hdf5_path: str, side: str):
    """Load per-frame 3D positions for a hand side.

    Returns:
        positions: (N, J, 3) world-space positions for the hand joints.
        cam_poses: (N, 4, 4) camera poses.
        intrinsic: (3, 3) intrinsic matrix.
    """
    with h5py.File(hdf5_path, "r") as f:
        intrinsic = f["camera/intrinsic"][:]
        cam_poses = f["transforms/camera"][:]
        joint_names = [f"{side}{s}" for s in HAND_JOINT_SUFFIXES]
        positions = np.stack(
            [f[f"transforms/{name}"][:, :3, 3] for name in joint_names],
            axis=1,
        )
    return positions, cam_poses, intrinsic


def build_intrinsics_entry(intrinsic: np.ndarray, img_w: int, img_h: int):
    """Build a single camera intrinsics dict."""
    return {
        "model": "pinhole",
        "image_width": img_w,
        "image_height": img_h,
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
    }


def build_poses_entry(cam_poses: np.ndarray):
    """Build a single camera poses dict with static/dynamic detection.

    Returns dict with 'state' ('static' or 'dynamic') and 'poses' (list of 4x4).
    """
    is_static = np.allclose(cam_poses[0], cam_poses[-1])
    if is_static:
        poses_list = [cam_poses[0].tolist()]
        state = "static"
    else:
        poses_list = [cam_poses[i].tolist() for i in range(len(cam_poses))]
        state = "dynamic"
    return {"state": state, "poses": poses_list}
