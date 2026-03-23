#!/usr/bin/env python3
"""
Visualize samples from HaMER WebDataset tars with MANO mesh overlay.

Reads {id}.jpg + {id}.data.pyd pairs from tar files, runs MANO forward
pass to get mesh vertices, and renders the mesh wireframe on top of
the image using OpenCV (no GPU or OpenGL required).

Usage:
    python scripts/visualize_webdataset.py --src ../hamer/hamer_training_data/dataset_tars/freihand-train --n 20
    python scripts/visualize_webdataset.py --src ../hamer/hamer_training_data/dataset_tars/ho3d-train --n 20
    python scripts/visualize_webdataset.py --src ../hamer/hamer_training_data/dataset_tars/interhand26m-train --n 20
    python scripts/visualize_webdataset.py --src ../hamer/hamer_training_data/dataset_tars/mtc-train --n 20
"""
import argparse
import os
import pickle
import random
import tarfile

import cv2
import numpy as np
import smplx
import torch

# ── MANO setup ──────────────────────────────────────────────────────
MANO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "hamer", "_DATA", "data", "mano")

# Skeleton for 2D keypoint drawing
PARENTS = [
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
]
FINGER_COLORS = [
    (0, 255, 255), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 165, 255),
]


def joint_color(i):
    return (0, 255, 0) if i == 0 else FINGER_COLORS[(i - 1) // 4]


# ── CPU mesh rendering via z-buffer rasterization ───────────────────
def render_mesh_cpu(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    mesh_color=(220, 220, 200),
    alpha=0.6,
):
    """Render MANO mesh on image using CPU z-buffer rasterization.

    Projects mesh triangles and paints them with depth-sorted ordering.

    Args:
        image: (H, W, 3) uint8 BGR image.
        vertices: (V, 3) mesh vertices (camera-relative).
        faces: (F, 3) face indices.
        cam_t: (3,) camera translation.
        focal_length: focal length in pixels.
        mesh_color: BGR color for mesh faces.
        alpha: blending alpha for mesh overlay.

    Returns:
        (H, W, 3) uint8 BGR composited image.
    """
    H, W = image.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # Translate vertices to camera space
    verts_cam = vertices + cam_t[np.newaxis, :]

    # Project to 2D
    z = verts_cam[:, 2:3]
    z = np.clip(z, 1e-4, None)
    px = focal_length * verts_cam[:, 0:1] / z + cx
    py = focal_length * verts_cam[:, 1:2] / z + cy
    proj = np.concatenate([px, py], axis=1)  # (V, 2)

    # Compute per-face depth for sorting (back to front)
    face_z = verts_cam[faces, 2].mean(axis=1)  # (F,)
    order = np.argsort(-face_z)  # back-to-front

    # Compute per-face normals for simple shading
    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normals = normals / norms

    # Simple directional light from camera
    light_dir = np.array([0, 0, -1], dtype=np.float32)
    shade = np.abs(np.sum(normals * light_dir, axis=1))
    shade = 0.3 + 0.7 * shade  # ambient + diffuse

    overlay = image.copy()

    # Draw filled triangles back-to-front
    base_color = np.array(mesh_color, dtype=np.float32)
    for fi in order:
        tri = proj[faces[fi]].astype(np.int32)  # (3, 2)

        # Skip if any vertex behind camera or out of bounds
        if face_z[fi] < 0.01:
            continue
        if np.any(tri[:, 0] < -W) or np.any(tri[:, 0] > 2 * W):
            continue
        if np.any(tri[:, 1] < -H) or np.any(tri[:, 1] > 2 * H):
            continue

        color = (base_color * shade[fi]).astype(np.uint8).tolist()
        pts = tri.reshape(-1, 1, 2)
        cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)

    # Blend
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw wireframe edges on top
    for fi in order:
        tri = proj[faces[fi]].astype(np.int32)
        if face_z[fi] < 0.01:
            continue
        for j in range(3):
            p1 = tuple(tri[j])
            p2 = tuple(tri[(j + 1) % 3])
            cv2.line(output, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

    return output


# ── MANO utilities ──────────────────────────────────────────────────
def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert (3,) axis-angle to (3, 3) rotation matrix."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def mano_forward(mano_model, hand_pose_aa, betas, is_right, kpts_3d=None):
    """Run MANO forward pass from axis-angle params.

    If kpts_3d is provided, translates vertices so that the MANO wrist
    aligns with the GT wrist position in camera space.

    Returns:
        vertices: (778, 3) numpy array in camera space.
        joints: (16, 3) numpy array.
    """
    model = mano_model["right"] if is_right else mano_model["left"]

    global_orient_aa = hand_pose_aa[:3]
    hand_pose_aa_15 = hand_pose_aa[3:48].reshape(15, 3)

    global_orient = torch.from_numpy(axis_angle_to_rotmat(global_orient_aa)).unsqueeze(0).unsqueeze(0)
    hand_pose = torch.stack(
        [torch.from_numpy(axis_angle_to_rotmat(hand_pose_aa_15[j])) for j in range(15)]
    ).unsqueeze(0)
    betas_t = torch.from_numpy(betas).unsqueeze(0).float()

    with torch.no_grad():
        out = model(global_orient=global_orient, hand_pose=hand_pose, betas=betas_t, pose2rot=False)

    vertices = out.vertices[0].numpy()
    joints = out.joints[0].numpy()

    # Translate MANO output to match GT camera-space position
    if kpts_3d is not None and kpts_3d[0, 3] > 0.5:
        # Align wrist (joint 0) to GT wrist
        offset = kpts_3d[0, :3] - joints[0]
        vertices = vertices + offset
        joints = joints + offset

    return vertices, joints


def estimate_focal(kpts_3d, kpts_2d, img_size):
    """Estimate crop-space focal length from 2D-3D correspondences.

    Uses least-squares: px - cx = fx * X/Z, py - cy = fy * Y/Z.
    Solves for a single focal length using all valid joints.

    Returns:
        focal_length: estimated focal length in crop pixels.
    """
    valid = (kpts_2d[:, 2] > 0.5) & (kpts_3d[:, 3] > 0.5)
    if valid.sum() < 3:
        return 5000.0

    cx, cy = img_size / 2.0, img_size / 2.0
    kp2 = kpts_2d[valid, :2]
    kp3 = kpts_3d[valid, :3]
    z = kp3[:, 2]

    # Build linear system: focal * (X/Z) = (px - cx), focal * (Y/Z) = (py - cy)
    # Stack both X and Y into one system: A * focal = b
    a_x = kp3[:, 0] / z  # X/Z
    a_y = kp3[:, 1] / z  # Y/Z
    b_x = kp2[:, 0] - cx
    b_y = kp2[:, 1] - cy
    A = np.concatenate([a_x, a_y])
    b = np.concatenate([b_x, b_y])

    # Least-squares: focal = (A^T b) / (A^T A)
    AtA = np.dot(A, A)
    if AtA < 1e-8:
        return 5000.0
    focal = float(np.dot(A, b) / AtA)
    if focal <= 0 or not np.isfinite(focal):
        return 5000.0
    return focal


# ── Drawing helpers ─────────────────────────────────────────────────
def draw_skeleton(image, kpts_2d):
    """Draw 2D skeleton. kpts_2d is (21, 3) with [x_px, y_px, conf]."""
    img = image.copy()
    for i in range(21):
        if kpts_2d[i, 2] < 0.5:
            continue
        x = int(round(kpts_2d[i, 0]))
        y = int(round(kpts_2d[i, 1]))
        color = joint_color(i)
        cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)
        p = PARENTS[i]
        if p >= 0 and kpts_2d[p, 2] >= 0.5:
            px = int(round(kpts_2d[p, 0]))
            py = int(round(kpts_2d[p, 1]))
            cv2.line(img, (px, py), (x, y), color, 2, cv2.LINE_AA)
    return img


# ── Data loading ────────────────────────────────────────────────────
def collect_samples(src_dir, max_tars=None):
    tar_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".tar")])
    if max_tars:
        tar_files = tar_files[:max_tars]
    samples = []
    for tar_name in tar_files:
        tar_path = os.path.join(src_dir, tar_name)
        with tarfile.open(tar_path) as tf:
            names = set(tf.getnames())
        pyd_names = [n for n in names if n.endswith(".data.pyd")]
        for pyd in pyd_names:
            base = pyd.replace(".data.pyd", "")
            if base + ".jpg" in names:
                samples.append((tar_path, base))
    return samples


def load_sample(tar_path, base_name):
    with tarfile.open(tar_path) as tf:
        img_bytes = tf.extractfile(base_name + ".jpg").read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pyd_bytes = tf.extractfile(base_name + ".data.pyd").read()
        ann = pickle.loads(pyd_bytes)[0]
    return img, ann


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize HaMER WebDataset with MANO mesh")
    parser.add_argument("--src", required=True, help="dataset_tars/{name}/ directory")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mano-dir", default=MANO_DIR)
    parser.add_argument("--max-tars", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.src))

    mano_dir = os.path.normpath(args.mano_dir)
    print(f"Loading MANO from {mano_dir}")
    mano_model = {
        "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True, flat_hand_mean=False),
        "left": smplx.MANOLayer(model_path=mano_dir, is_rhand=False, flat_hand_mean=False),
    }
    faces_right = mano_model["right"].faces.astype(np.int32)
    faces_left = faces_right[:, [0, 2, 1]]  # flip winding for left hand

    print(f"Scanning {args.src} ...")
    samples = collect_samples(args.src, max_tars=args.max_tars)
    print(f"Found {len(samples)} samples")

    picks = random.sample(samples, min(args.n, len(samples)))

    saved = 0
    for i, (tar_path, base_name) in enumerate(picks):
        try:
            img_bgr, ann = load_sample(tar_path, base_name)
        except Exception as e:
            print(f"  Skip {base_name}: {e}")
            continue

        is_right = bool(ann["right"] > 0.5)
        has_pose = bool(ann["has_hand_pose"] > 0.5)
        hand_pose = ann["hand_pose"]
        betas = ann["betas"]
        kpts_2d = ann["keypoints_2d"]
        kpts_3d = ann["keypoints_3d"]
        img_size = img_bgr.shape[0]

        # Left panel: 2D skeleton
        skel_img = draw_skeleton(img_bgr, kpts_2d)

        # Right panel: MANO mesh overlay
        if has_pose:
            vertices, joints = mano_forward(mano_model, hand_pose, betas, is_right, kpts_3d)
            focal = estimate_focal(kpts_3d, kpts_2d, img_size)
            cam_t = np.zeros(3, dtype=np.float32)
            faces = faces_right if is_right else faces_left
            mesh_color = (200, 220, 255) if is_right else (255, 230, 200)
            mesh_img = render_mesh_cpu(img_bgr, vertices, faces, cam_t, focal, mesh_color)
        else:
            mesh_img = img_bgr.copy()
            cv2.putText(mesh_img, "No MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Labels
        side_label = "R" if is_right else "L"
        cv2.putText(skel_img, "2D Keypoints", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(mesh_img, "MANO Mesh", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(skel_img, side_label, (img_size - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        combined = np.concatenate([skel_img, mesh_img], axis=1)

        sample_id = os.path.basename(base_name)
        out_path = os.path.join(args.out, f"webdataset_{dataset_name}_{sample_id}.jpg")
        cv2.imwrite(out_path, combined)
        saved += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(picks)}]")

    print(f"Saved {saved} images to {args.out}/")


if __name__ == "__main__":
    main()
