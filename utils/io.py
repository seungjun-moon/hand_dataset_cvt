"""I/O utilities for loading DexYCB data and writing egodex format."""

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


def images_to_mp4(image_paths: list, output_path: str, fps: float = 30.0):
    """Encode a list of image paths into an mp4 video using OpenCV."""
    if not image_paths:
        return
    first = cv2.imread(image_paths[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for p in image_paths:
        frame = cv2.imread(p)
        writer.write(frame)
    writer.release()


def write_egodex_hdf5(output_path: str, intrinsic: np.ndarray,
                      transforms_dict: dict, transforms_cam_dict: dict,
                      confidences_dict: dict, gravity: np.ndarray):
    """Write an egodex-format HDF5 file.

    Args:
        output_path: Path to write the HDF5 file.
        intrinsic: (3, 3) camera intrinsic matrix.
        transforms_dict: {joint_name: (N, 4, 4) array} world-space transforms.
        transforms_cam_dict: {joint_name: (N, 4, 4) array} camera-space transforms.
        confidences_dict: {joint_name: (N,) array}
        gravity: (3, 3) gravity alignment rotation.
    """
    with h5py.File(output_path, "w") as f:
        cam_grp = f.create_group("camera")
        cam_grp.create_dataset("intrinsic", data=intrinsic)

        tf_grp = f.create_group("transforms")
        conf_grp = f.create_group("confidences")
        tf_cam_grp = f.create_group("transforms_cam")

        for name, data in transforms_dict.items():
            tf_grp.create_dataset(name, data=data.astype(np.float32))

        tf_grp.create_dataset("gravity", data=gravity.astype(np.float32))

        for name, data in transforms_cam_dict.items():
            tf_cam_grp.create_dataset(name, data=data.astype(np.float32))

        for name, data in confidences_dict.items():
            conf_grp.create_dataset(name, data=data.astype(np.float32))
