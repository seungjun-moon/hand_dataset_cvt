"""Utilities for computing 4x4 transforms from 3D joint positions."""

import numpy as np

from .joint_mapping import MANO_PARENTS


def joints_to_transforms(joint_3d: np.ndarray) -> np.ndarray:
    """Convert (21, 3) joint positions to (21, 4, 4) SE(3) transforms.

    Rotation is derived from bone directions (parent->joint for non-root,
    joint->child average for root). Translation is the joint position.
    """
    n_joints = joint_3d.shape[0]
    transforms = np.zeros((n_joints, 4, 4), dtype=np.float32)
    transforms[:, 3, 3] = 1.0

    # Set translations
    transforms[:, :3, 3] = joint_3d

    # Compute rotations from bone directions
    for j in range(n_joints):
        parent = MANO_PARENTS[j]
        if parent == -1:
            # Root (wrist): average direction to children
            children = [c for c, p in enumerate(MANO_PARENTS) if p == j]
            if children:
                dirs = joint_3d[children] - joint_3d[j]
                bone_dir = dirs.mean(axis=0)
            else:
                bone_dir = np.array([0, 1, 0], dtype=np.float32)
        else:
            bone_dir = joint_3d[j] - joint_3d[parent]

        transforms[j, :3, :3] = _rotation_from_direction(bone_dir)

    return transforms


def joints_to_transforms_batch(joint_3d_batch: np.ndarray) -> np.ndarray:
    """Convert (N, 21, 3) joint positions to (N, 21, 4, 4) transforms."""
    N, n_joints, _ = joint_3d_batch.shape
    result = np.zeros((N, n_joints, 4, 4), dtype=np.float32)
    for i in range(N):
        result[i] = joints_to_transforms(joint_3d_batch[i])
    return result


def _rotation_from_direction(direction: np.ndarray) -> np.ndarray:
    """Build a 3x3 rotation matrix where Z-axis aligns with the given direction."""
    z = direction.copy()
    norm = np.linalg.norm(z)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    z /= norm

    # Choose an up vector that isn't parallel to z
    up = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1, 0, 0], dtype=np.float32)

    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1)  # columns are x, y, z
    return R.astype(np.float32)


def interpolate_joint(joint_3d: np.ndarray, idx_a: int, idx_b: int,
                      alpha: float = 0.5) -> np.ndarray:
    """Linearly interpolate between two joint positions."""
    return (1 - alpha) * joint_3d[idx_a] + alpha * joint_3d[idx_b]


def make_transform(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a position and bone direction."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _rotation_from_direction(direction)
    T[:3, 3] = position
    return T


def extrinsics_tuple_to_4x4(vals: tuple) -> np.ndarray:
    """Convert DexYCB extrinsics (12-element tuple: 3x4 row-major [R|t]) to 4x4."""
    M = np.array(vals, dtype=np.float32).reshape(3, 4)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = M[:, :3]
    T[:3, 3] = M[:, 3]
    return T


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """Invert a rigid (SE3) 4x4 transform efficiently."""
    T_inv = np.eye(4, dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t
