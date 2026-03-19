"""I/O utilities for loading DexYCB data and writing egodex format."""

import json
import os
import glob
import subprocess

import cv2
import h5py
import numpy as np
import yaml


def load_yaml(path: str, unsafe: bool = False) -> dict:
    """Load a YAML file. Use unsafe=True for files with python/tuple tags."""
    with open(path) as f:
        if unsafe:
            return yaml.unsafe_load(f)
        return yaml.load(f, Loader=yaml.FullLoader)


def load_intrinsics(calibration_dir: str, serial: str) -> np.ndarray:
    """Load camera intrinsics as a (3, 3) matrix.

    Loads from calibration/intrinsics/{serial}_640x480.yml using the 'color'
    section (fx, fy, ppx, ppy), matching HaWoR's DexYCBDataset loading.
    """
    path = os.path.join(calibration_dir, "intrinsics", f"{serial}_640x480.yml")
    data = load_yaml(path)
    intr = data["color"]
    K = np.array([
        [intr["fx"], 0, intr["ppx"]],
        [0, intr["fy"], intr["ppy"]],
        [0, 0, 1],
    ], dtype=np.float32)
    return K


def load_extrinsics(calibration_dir: str, extrinsics_name: str, serial: str) -> np.ndarray:
    """Load camera extrinsics as a (4, 4) matrix."""
    from .transforms import extrinsics_tuple_to_4x4
    path = os.path.join(calibration_dir, f"extrinsics_{extrinsics_name}", "extrinsics.yml")
    data = load_yaml(path, unsafe=True)  # needs unsafe for python/tuple tags
    return extrinsics_tuple_to_4x4(data["extrinsics"][serial])


def load_frame_labels(label_path: str) -> dict:
    """Load a per-frame label .npz file."""
    return dict(np.load(label_path, allow_pickle=True))


def collect_color_paths(camera_dir: str) -> list:
    """Return sorted list of color image paths in a camera directory."""
    paths = sorted(glob.glob(os.path.join(camera_dir, "color_*.jpg")))
    return paths


def collect_depth_paths(camera_dir: str) -> list:
    """Return sorted list of aligned depth image paths in a camera directory."""
    paths = sorted(glob.glob(os.path.join(camera_dir, "aligned_depth_to_color_*.png")))
    return paths


def _pipe_frames_to_ffmpeg(frames_iter, output_path: str, fps: float,
                           width: int, height: int):
    """Pipe raw BGR frames to ffmpeg via stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
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
    for frame in frames_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed for {output_path}: {err}")


def images_to_mp4(image_paths: list, output_path: str, fps: float = 30.0):
    """Encode a list of image paths into an mp4 video using ffmpeg."""
    if not image_paths:
        return
    first = cv2.imread(image_paths[0])
    h, w = first.shape[:2]

    def frames():
        for p in image_paths:
            yield cv2.imread(p)

    _pipe_frames_to_ffmpeg(frames(), output_path, fps, w, h)


def depth_images_to_mp4(image_paths: list, output_path: str, fps: float = 30.0):
    """Encode 16-bit depth PNGs into a lossless metric-depth mp4.

    Depth values (uint16 millimeters) are packed into two channels of a BGR
    frame: B = high byte, G = low byte, R = 0. Encoded with libx264rgb CRF 0
    (mathematically lossless).

    To decode: depth_mm = B.astype(uint16) * 256 + G.astype(uint16)
               depth_m  = depth_mm / 1000.0
    """
    if not image_paths:
        return
    first = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = first.shape[:2]

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
    for p in image_paths:
        depth = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = (depth >> 8).astype(np.uint8)    # B = high byte
        frame[:, :, 1] = (depth & 0xFF).astype(np.uint8)  # G = low byte
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed for {output_path}: {err}")


def _compute_valid_ranges(valid_mask: np.ndarray) -> list:
    """Compute contiguous [start, end) ranges where all values are True.

    Args:
        valid_mask: (N,) boolean array.

    Returns:
        List of [start, end) pairs (integers).
    """
    if len(valid_mask) == 0:
        return []
    # Pad with False at boundaries to detect edges
    padded = np.concatenate([[False], valid_mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [[int(s), int(e)] for s, e in zip(starts, ends)]


# 21 joint names per side, matching MANO order (same as in HDF5Dataset).
_SIDE_JOINT_NAMES = {
    'right': [
        'rightHand',
        'rightThumbKnuckle', 'rightThumbIntermediateBase', 'rightThumbIntermediateTip', 'rightThumbTip',
        'rightIndexFingerKnuckle', 'rightIndexFingerIntermediateBase', 'rightIndexFingerIntermediateTip', 'rightIndexFingerTip',
        'rightMiddleFingerKnuckle', 'rightMiddleFingerIntermediateBase', 'rightMiddleFingerIntermediateTip', 'rightMiddleFingerTip',
        'rightRingFingerKnuckle', 'rightRingFingerIntermediateBase', 'rightRingFingerIntermediateTip', 'rightRingFingerTip',
        'rightLittleFingerKnuckle', 'rightLittleFingerIntermediateBase', 'rightLittleFingerIntermediateTip', 'rightLittleFingerTip',
    ],
    'left': [
        'leftHand',
        'leftThumbKnuckle', 'leftThumbIntermediateBase', 'leftThumbIntermediateTip', 'leftThumbTip',
        'leftIndexFingerKnuckle', 'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 'leftIndexFingerTip',
        'leftMiddleFingerKnuckle', 'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip', 'leftMiddleFingerTip',
        'leftRingFingerKnuckle', 'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip', 'leftRingFingerTip',
        'leftLittleFingerKnuckle', 'leftLittleFingerIntermediateBase', 'leftLittleFingerIntermediateTip', 'leftLittleFingerTip',
    ],
}


def update_sequence_meta(hdf5_path: str):
    """Compute metadata for an HDF5 label file and update _meta.json in its directory.

    Detects sides from transforms/ keys (not mano_ groups), so it works
    correctly even before extra mano groups are added (e.g. ho_cap multi-hand).

    The _meta.json maps each HDF5 filename to:
        {"sides": {"right": {"n_frames": int, "valid_ranges": [[start, end), ...]}, ...}}
    """
    hdf5_dir = os.path.dirname(hdf5_path)
    hdf5_name = os.path.basename(hdf5_path)
    meta_path = os.path.join(hdf5_dir, "_meta.json")

    # Load existing meta or start fresh
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    entry = {"sides": {}}

    with h5py.File(hdf5_path, "r") as f:
        tf_keys = set(f["transforms"].keys()) if "transforms" in f else set()
        has_conf = "confidences" in f

        for side, joint_names in _SIDE_JOINT_NAMES.items():
            first_joint = joint_names[0]
            if first_joint not in tf_keys:
                continue

            n_frames = f[f"transforms/{first_joint}"].shape[0]

            if has_conf:
                conf = np.stack(
                    [f[f"confidences/{j}"][:] for j in joint_names if j in f["confidences"]],
                    axis=1,
                )  # (n_frames, n_joints)
                valid = np.all(conf > 0, axis=1)
            else:
                valid = np.ones(n_frames, dtype=bool)

            entry["sides"][side] = {
                "n_frames": int(n_frames),
                "valid_ranges": _compute_valid_ranges(valid),
            }

    if entry["sides"]:
        meta[hdf5_name] = entry

    with open(meta_path, "w") as f:
        json.dump(meta, f, separators=(",", ":"))


def write_egodex_hdf5(output_path: str, intrinsic: np.ndarray,
                      transforms_dict: dict, confidences_dict: dict,
                      mano_dict: dict = None):
    """Write an egodex-format HDF5 file.

    Args:
        output_path: Path to write the HDF5 file.
        intrinsic: (3, 3) or (N, 3, 3) camera intrinsic matrix(es).
            If (N, 3, 3), stored as both camera/intrinsic (first frame)
            and camera/intrinsics (all frames).
        transforms_dict: {joint_name: (N, 4, 4) array} world-space transforms.
        confidences_dict: {joint_name: (N,) array}
        mano_dict: Optional dict with MANO parameters.
            Expected keys: 'betas' (10,), 'global_orient_worldspace' (N, 3, 3),
            'hand_pose' (N, 15, 3, 3), 'transl_worldspace' (N, 3),
            'kpt3d' (N, 21, 3), 'side' str.
            Stored as mano_{side}/ group (e.g. mano_right/, mano_left/).
    """
    with h5py.File(output_path, "w") as f:
        cam_grp = f.create_group("camera")
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.ndim == 3:
            # Per-frame intrinsics: store (N, 3, 3) as "intrinsics"
            # and first frame as "intrinsic" for backward compat
            cam_grp.create_dataset("intrinsics", data=intrinsic)
            cam_grp.create_dataset("intrinsic", data=intrinsic[0])
        else:
            cam_grp.create_dataset("intrinsic", data=intrinsic)

        tf_grp = f.create_group("transforms")
        conf_grp = f.create_group("confidences")

        for name, data in transforms_dict.items():
            tf_grp.create_dataset(name, data=data.astype(np.float32))

        for name, data in confidences_dict.items():
            conf_grp.create_dataset(name, data=data.astype(np.float32))

        if mano_dict is not None:
            side = mano_dict["side"]
            grp = f.create_group(f"mano_{side}")
            grp.create_dataset("betas", data=mano_dict["betas"].astype(np.float32))
            grp.create_dataset("global_orient_worldspace",
                               data=mano_dict["global_orient_worldspace"].astype(np.float32))
            grp.create_dataset("hand_pose",
                               data=mano_dict["hand_pose"].astype(np.float32))
            grp.create_dataset("transl_worldspace",
                               data=mano_dict["transl_worldspace"].astype(np.float32))
            grp.create_dataset("kpt3d",
                               data=mano_dict["kpt3d"].astype(np.float32))

    # Auto-generate _meta.json entry for this HDF5 file
    update_sequence_meta(output_path)
