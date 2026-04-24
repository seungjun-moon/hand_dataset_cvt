#!/usr/bin/env python3
"""Visualize randomly sampled frames with projected 3D keypoints.

Reads CONVERTED datasets (HDF5+MP4 or NPZ+JPG), HaMER WebDataset tars,
or ClipDataset label directories (.data.pyd + per-sequence .pyd files).

Automatically detects the dataset format:
  - WebDataset: *.tar containing {id}.jpg + {id}.data.pyd (HaMER format)
  - ClipDataset: *_clip.data.pyd master index + per-sequence .pyd files
                 (requires --img-dir for image root)

Usage:
    python scripts/visualize.py --src ../hand_tracking_ablation/_DATA/hamer_training_data/dataset_tars/dexycb --n 50
    python scripts/visualize.py --src ../hand_tracking_ablation/hamer_training_data/dataset_tars/ho3d-train --n 20 --mano-dir /path/to/mano
    python scripts/visualize.py \
    --src ../hand_tracking_ablation/_DATA/hamer_training_data/dataset_tars_manotorch/reinterhand --n 50

python scripts/visualize.py --src ../hand_tracking_ablation/_DATA/haptic_training_label/arctic/clip \
    --img-dir ../hand_tracking_ablation/_DATA/haptic_training_images/arctic/images/ --n 20

python scripts/visualize.py --src ../hand_tracking_ablation/_DATA/haptic_training_label/arctic/clip \
    --video-dir ../hand_tracking_ablation/_DATA/haptic_training_videos/arctic --n 20

python scripts/visualize.py --src ../hand_tracking_ablation/_DATA/haptic_training_label/dexycb/clip \
    --video-dir ../hand_tracking_ablation/_DATA/haptic_training_videos/dexycb --n 20
"""

import argparse
import os
import pickle
import random
import sys
import smplx
import tarfile

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.camera_utils import HAND_JOINT_SUFFIXES, get_active_sides
from utils.image_utils import project_3d_to_2d

# ── Skeleton constants ─────────────────────────────────────────────
PARENTS = [
    -1,  # 0: wrist
    0, 1, 2, 3,      # thumb
    0, 5, 6, 7,      # index
    0, 9, 10, 11,    # middle
    0, 13, 14, 15,   # ring
    0, 17, 18, 19,   # little
]

FINGER_COLORS = [
    (0, 255, 255),   # thumb - yellow
    (0, 0, 255),     # index - red
    (255, 0, 0),     # middle - blue
    (255, 0, 255),   # ring - magenta
    (0, 165, 255),   # little - orange
]

WRIST_COLOR = (0, 255, 0)  # green
SIDE_COLORS = {"right": (0, 200, 0), "left": (200, 200, 0)}

DEFAULT_MANO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "hand_tracking_ablation", "_DATA", "data", "mano"
)

def _joint_color(joint_idx: int):
    if joint_idx == 0:
        return WRIST_COLOR
    return FINGER_COLORS[(joint_idx - 1) // 4]


# ── Format detection ───────────────────────────────────────────────
def detect_format(src_dir: str) -> str:
    """Detect dataset format by checking file extensions."""
    # Check top-level for clipdataset master index files
    for fname in os.listdir(src_dir):
        if fname.endswith("_clip.data.pyd"):
            return "clipdataset"

    # Check top-level for tar files (webdataset)
    for fname in os.listdir(src_dir):
        if fname.endswith(".tar"):
            return "webdataset"

def read_video_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# Cache frame maps across calls so we don't reload .frames.npy for every sample.
_FRAME_MAP_CACHE = {}


def _clip_imgname_to_video(imgname_rel: str, video_dir: str):
    """Map a ClipDataset imgname to (video_path, frame_idx) via the .frames.npy sidecar.

    Layout mirrors models_clip.datasets.video_dataset.VideoDataset:
        imgname  's01/box_grab_01/0/00001.jpg'
      → video    '<video_dir>/s01/box_grab_01/0.mp4'
      → map      '<video_dir>/s01/box_grab_01/0.frames.npy'   {'00001.jpg': 0, ...}
    """
    parts = imgname_rel.split('/')
    fname = parts[-1]
    video_rel = '/'.join(parts[:-1]) + '.mp4'
    video_path = os.path.join(video_dir, video_rel)
    map_path = video_path[:-4] + '.frames.npy'

    frame_map = _FRAME_MAP_CACHE.get(map_path)
    if frame_map is None and os.path.exists(map_path):
        frame_map = np.load(map_path, allow_pickle=True).item()
        _FRAME_MAP_CACHE[map_path] = frame_map

    if frame_map is not None and fname in frame_map:
        return video_path, int(frame_map[fname])

    # Fallback: assume 1-indexed dense numbering (e.g. arctic '00001.jpg' -> 0)
    digits = ''.join(c for c in os.path.splitext(fname)[0] if c.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame index from {fname}")
    return video_path, int(digits) - 1

# ── WebDataset format ──────────────────────────────────────────────
def collect_samples_webdataset(src_dir: str, max_tars: int = 5):
    """Collect WebDataset samples: (tar_path, base_name)."""
    tar_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".tar")])
    if max_tars and len(tar_files) > max_tars:
        tar_files = random.sample(tar_files, max_tars)
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


def _load_webdataset_sample(tar_path, base_name):
    with tarfile.open(tar_path) as tf:
        img_bytes = tf.extractfile(base_name + ".jpg").read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pyd_bytes = tf.extractfile(base_name + ".data.pyd").read()
        ann = pickle.loads(pyd_bytes)[0]
    return img, ann


def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
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


def _mano_forward(mano_model, hand_pose_aa, betas, is_right, kpts_3d=None):
    """Run MANO forward pass from axis-angle params (MANO_RIGHT only).

    Left hands are handled by mirroring: negate y/z of axis-angle, run MANO_RIGHT,
    mirror output.x back. Mirrors the dataloader's fliplr convention, and avoids
    the MANO_LEFT.pkl shapedirs sign bug (smplx#48).
    """
    import torch
    model = mano_model["right"] if isinstance(mano_model, dict) else mano_model

    hp = hand_pose_aa.astype(np.float64).copy()
    if not is_right:
        hp[1::3] *= -1
        hp[2::3] *= -1

    global_orient_aa = hp[:3]
    hand_pose_aa_15 = hp[3:48].reshape(15, 3)

    global_orient = torch.from_numpy(_axis_angle_to_rotmat(global_orient_aa)).unsqueeze(0).unsqueeze(0)
    hand_pose = torch.stack(
        [torch.from_numpy(_axis_angle_to_rotmat(hand_pose_aa_15[j])) for j in range(15)]
    ).unsqueeze(0)
    betas_t = torch.from_numpy(betas).unsqueeze(0).float()

    with torch.no_grad():
        out = model(global_orient=global_orient, hand_pose=hand_pose, betas=betas_t, pose2rot=False)

    vertices = out.vertices[0].numpy().astype(np.float64)
    joints = out.joints[0].numpy().astype(np.float64)

    if not is_right:
        vertices[:, 0] *= -1
        joints[:, 0] *= -1

    if kpts_3d is not None and kpts_3d[0, 3] > 0.5:
        offset = kpts_3d[0, :3] - joints[0]
        vertices = vertices + offset
        joints = joints + offset

    return vertices.astype(np.float32), joints.astype(np.float32)


def _mano_forward_clip(mano_model, hand_pose_aa, betas, hand_tsl, cTw, is_right):
    """Run MANO forward pass for ClipDataset: returns vertices/joints in camera space.

    Only MANO_RIGHT is used. Left hands are handled the same way the training
    dataloader does (models_clip/datasets/utils.py:409-440 fliplr_params + do_flip):
    negate y/z of axis-angle, mirror hand_tsl.x, run MANO_RIGHT, mirror output.x back.
    This avoids MANO_LEFT.pkl's shapedirs sign bug (smplx#48) entirely.
    """
    import torch
    model = mano_model["right"] if isinstance(mano_model, dict) else mano_model

    hp = hand_pose_aa.astype(np.float64).copy()
    tsl = hand_tsl.astype(np.float64).copy()
    if not is_right:
        hp[1::3] *= -1   # negate y of each axis-angle (global + 15 joints)
        hp[2::3] *= -1   # negate z of each axis-angle
        tsl[0] *= -1     # mirror translation across world x

    global_orient_aa = hp[:3]
    hand_pose_aa_15 = hp[3:48].reshape(15, 3)

    global_orient = torch.from_numpy(_axis_angle_to_rotmat(global_orient_aa)).unsqueeze(0).unsqueeze(0)
    hand_pose = torch.stack(
        [torch.from_numpy(_axis_angle_to_rotmat(hand_pose_aa_15[j])) for j in range(15)]
    ).unsqueeze(0)
    betas_t = torch.from_numpy(betas).unsqueeze(0).float()
    transl_t = torch.from_numpy(tsl).unsqueeze(0).float()

    with torch.no_grad():
        out = model(global_orient=global_orient, hand_pose=hand_pose,
                    betas=betas_t, transl=transl_t, pose2rot=False)

    verts_world = out.vertices[0].numpy().astype(np.float64)   # (778, 3)
    joints_world = out.joints[0].numpy().astype(np.float64)    # (16 or 21, 3)

    if not is_right:
        verts_world[:, 0] *= -1
        joints_world[:, 0] *= -1

    R_cw = cTw[:3, :3].astype(np.float64)
    t_cw = cTw[:3, 3:4].astype(np.float64)
    verts_cam = (R_cw @ verts_world.T + t_cw).T.astype(np.float32)
    joints_cam = (R_cw @ joints_world.T + t_cw).T.astype(np.float32)

    return verts_cam, joints_cam

def _estimate_focal(kpts_3d, kpts_2d, img_w, img_h=None):
    """Estimate focal length from 3D-2D correspondences assuming centered PP.

    Args:
        kpts_3d: (21, 4) camera-space keypoints with confidence.
        kpts_2d: (21, 3) pixel keypoints with confidence.
        img_w: image width (or img_size for square images).
        img_h: image height (defaults to img_w for square images).
    """
    if img_h is None:
        img_h = img_w
    valid = (kpts_2d[:, 2] > 0.5) & (kpts_3d[:, 3] > 0.5)
    if valid.sum() < 3:
        return 5000.0

    cx, cy = img_w / 2.0, img_h / 2.0
    kp2 = kpts_2d[valid, :2]
    kp3 = kpts_3d[valid, :3]
    z = kp3[:, 2]

    a_x = kp3[:, 0] / z
    a_y = kp3[:, 1] / z
    b_x = kp2[:, 0] - cx
    b_y = kp2[:, 1] - cy
    A = np.concatenate([a_x, a_y])
    b = np.concatenate([b_x, b_y])

    AtA = np.dot(A, A)
    if AtA < 1e-8:
        return 5000.0
    focal = float(np.dot(A, b) / AtA)
    if focal <= 0 or not np.isfinite(focal):
        return 5000.0
    return focal


def _render_mesh_cpu(image, vertices, faces, cam_t, focal_length,
                     mesh_color=(220, 220, 200), alpha=0.6):
    """Render MANO mesh on image using CPU z-buffer rasterization."""
    H, W = image.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    verts_cam = vertices + cam_t[np.newaxis, :]

    z = np.clip(verts_cam[:, 2:3], 1e-4, None)
    px = focal_length * verts_cam[:, 0:1] / z + cx
    py = focal_length * verts_cam[:, 1:2] / z + cy
    proj = np.concatenate([px, py], axis=1)

    face_z = verts_cam[faces, 2].mean(axis=1)
    order = np.argsort(-face_z)

    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    normals = normals / norms

    light_dir = np.array([0, 0, -1], dtype=np.float32)
    shade = np.abs(np.sum(normals * light_dir, axis=1))
    shade = 0.3 + 0.7 * shade

    overlay = image.copy()
    base_color = np.array(mesh_color, dtype=np.float32)
    for fi in order:
        tri = proj[faces[fi]].astype(np.int32)
        if face_z[fi] < 0.01:
            continue
        if np.any(tri[:, 0] < -W) or np.any(tri[:, 0] > 2 * W):
            continue
        if np.any(tri[:, 1] < -H) or np.any(tri[:, 1] > 2 * H):
            continue
        color = (base_color * shade[fi]).astype(np.uint8).tolist()
        pts = tri.reshape(-1, 1, 2)
        cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    for fi in order:
        tri = proj[faces[fi]].astype(np.int32)
        if face_z[fi] < 0.01:
            continue
        for j in range(3):
            p1 = tuple(tri[j])
            p2 = tuple(tri[(j + 1) % 3])
            cv2.line(output, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

    return output


def _render_mesh_cpu_K(image, vertices, faces, K, mesh_color=(220, 220, 200), alpha=0.6):
    """Render MANO mesh on image using full K intrinsic matrix for projection."""
    H, W = image.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = np.clip(vertices[:, 2:3], 1e-4, None)
    px = fx * vertices[:, 0:1] / z + cx
    py = fy * vertices[:, 1:2] / z + cy
    proj = np.concatenate([px, py], axis=1)

    face_z = vertices[faces, 2].mean(axis=1)
    order = np.argsort(-face_z)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    normals = normals / norms

    light_dir = np.array([0, 0, -1], dtype=np.float32)
    shade = np.abs(np.sum(normals * light_dir, axis=1))
    shade = 0.3 + 0.7 * shade

    overlay = image.copy()
    base_color = np.array(mesh_color, dtype=np.float32)
    for fi in order:
        tri = proj[faces[fi]].astype(np.int32)
        if face_z[fi] < 0.01:
            continue
        if np.any(tri[:, 0] < -W) or np.any(tri[:, 0] > 2 * W):
            continue
        if np.any(tri[:, 1] < -H) or np.any(tri[:, 1] > 2 * H):
            continue
        color = (base_color * shade[fi]).astype(np.uint8).tolist()
        pts = tri.reshape(-1, 1, 2)
        cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    for fi in order:
        tri = proj[faces[fi]].astype(np.int32)
        if face_z[fi] < 0.01:
            continue
        for j in range(3):
            p1 = tuple(tri[j])
            p2 = tuple(tri[(j + 1) % 3])
            cv2.line(output, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

    return output


def _draw_webdataset_skeleton(image, kpts_2d):
    """Draw 2D skeleton. kpts_2d is (21, 3) with [x_px, y_px, conf]."""
    img = image.copy()
    for i in range(21):
        if kpts_2d[i, 2] < 0.5:
            continue
        x = int(round(kpts_2d[i, 0]))
        y = int(round(kpts_2d[i, 1]))
        color = _joint_color(i)
        cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)
        p = PARENTS[i]
        if p >= 0 and kpts_2d[p, 2] >= 0.5:
            px = int(round(kpts_2d[p, 0]))
            py = int(round(kpts_2d[p, 1]))
            cv2.line(img, (px, py), (x, y), color, 2, cv2.LINE_AA)
    return img


def render_frame_webdataset(tar_path, base_name, mano_model, faces_right, faces_left):
    """Render a WebDataset sample with 2D skeleton + MANO mesh side-by-side."""
    try:
        img_bgr, ann = _load_webdataset_sample(tar_path, base_name)
    except Exception as e:
        print(f"  Skip {base_name}: {e}")
        return None

    is_right = bool(ann["right"] > 0.5)
    has_pose = bool(ann["has_hand_pose"] > 0.5)
    hand_pose = ann["hand_pose"]
    betas = ann["betas"]
    kpts_2d = ann["keypoints_2d"]
    kpts_3d = ann["keypoints_3d"]
    img_h, img_w = img_bgr.shape[:2]

    skel_img = _draw_webdataset_skeleton(img_bgr, kpts_2d)

    # Project 3D keypoints to 2D and draw skeleton
    proj3d_img = img_bgr.copy()
    valid_3d = kpts_3d[:, 3] > 0.5
    if valid_3d.any():
        focal = _estimate_focal(kpts_3d, kpts_2d, img_w, img_h)
        cx, cy = img_w / 2.0, img_h / 2.0
        proj_2d = np.zeros((21, 3), dtype=np.float32)
        for j in range(21):
            if kpts_3d[j, 3] > 0.5:
                z = max(kpts_3d[j, 2], 1e-4)
                proj_2d[j, 0] = focal * kpts_3d[j, 0] / z + cx
                proj_2d[j, 1] = focal * kpts_3d[j, 1] / z + cy
                proj_2d[j, 2] = 1.0
        proj3d_img = _draw_webdataset_skeleton(img_bgr, proj_2d)

    if has_pose:
        vertices, joints = _mano_forward(mano_model, hand_pose, betas, is_right, kpts_3d)
        focal = _estimate_focal(kpts_3d, kpts_2d, img_w, img_h)
        cam_t = np.zeros(3, dtype=np.float32)
        faces = faces_right if is_right else faces_left
        mesh_color = (200, 220, 255) if is_right else (255, 230, 200)
        mesh_img = _render_mesh_cpu(img_bgr, vertices, faces, cam_t, focal, mesh_color)
    else:
        mesh_img = img_bgr.copy()
        cv2.putText(mesh_img, "No MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    side_label = "R" if is_right else "L"
    cv2.putText(skel_img, "2D Keypoints", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(proj3d_img, "Proj 3D", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(mesh_img, "MANO Mesh", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(skel_img, side_label, (img_w - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return np.concatenate([skel_img, proj3d_img, mesh_img], axis=1)


def _bbox_to_affine(center, scale, out_size, rescale=1.0):
    """Build a 2x3 affine from the label bbox (center, scale) to an out_size patch.

    HaMER convention: bbox_size_px = scale * 200. Crop is forced to be square by
    taking max(scale_x, scale_y). An extra ``rescale`` factor can be applied to
    match how HaMER's dataloader expands the bbox at training time (typically
    1.0 here, since the labeled bbox is usually already padded).
    """
    bbox_w = float(scale[0]) * 200.0 * rescale
    bbox_h = float(scale[1]) * 200.0 * rescale
    side_len = max(bbox_w, bbox_h)
    s = out_size / side_len
    A = np.array([
        [s, 0, -s * float(center[0]) + out_size / 2.0],
        [0, s, -s * float(center[1]) + out_size / 2.0],
    ], dtype=np.float32)
    return A, side_len


def render_frame_webdataset_crop(tar_path, base_name, mano_model, faces_right, faces_left,
                                 out_size=256, rescale=1.0):
    """WebDataset variant that crops each panel to the label's bbox.

    Panels (all out_size x out_size except overview):
      0: full image with the bbox drawn (resized to match out_size height)
      1: cropped 2D skeleton (GT keypoints_2d warped into the crop)
      2: cropped projected 3D skeleton
      3: cropped MANO mesh
    """
    try:
        img_bgr, ann = _load_webdataset_sample(tar_path, base_name)
    except Exception as e:
        print(f"  Skip {base_name}: {e}")
        return None

    img_h, img_w = img_bgr.shape[:2]
    center = np.asarray(ann["center"], dtype=np.float64)
    scale = np.asarray(ann["scale"], dtype=np.float64)
    A, side_len = _bbox_to_affine(center, scale, out_size, rescale)

    is_right = bool(ann["right"] > 0.5)
    has_pose = bool(ann["has_hand_pose"] > 0.5)
    hand_pose = ann["hand_pose"]
    betas = ann["betas"]
    kpts_2d = ann["keypoints_2d"]
    kpts_3d = ann["keypoints_3d"]

    # Panel 0: overview with bbox drawn on full image, resized to height=out_size
    overview = img_bgr.copy()
    cx0, cy0 = float(center[0]), float(center[1])
    tl = (int(round(cx0 - side_len / 2.0)), int(round(cy0 - side_len / 2.0)))
    br = (int(round(cx0 + side_len / 2.0)), int(round(cy0 + side_len / 2.0)))
    cv2.rectangle(overview, tl, br, (0, 255, 255), 2)
    ov_w = max(1, int(round(img_w * out_size / float(img_h))))
    overview = cv2.resize(overview, (ov_w, out_size))
    cv2.putText(overview, "bbox", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1)

    # Panel 1: 2D skeleton drawn on full image then warped
    skel_full = _draw_webdataset_skeleton(img_bgr, kpts_2d)
    skel_crop = cv2.warpAffine(skel_full, A, (out_size, out_size))

    # Panel 2: projected 3D keypoints
    proj3d_full = img_bgr.copy()
    valid_3d = kpts_3d[:, 3] > 0.5
    if valid_3d.any():
        focal = _estimate_focal(kpts_3d, kpts_2d, img_w, img_h)
        ppx, ppy = img_w / 2.0, img_h / 2.0
        proj_2d = np.zeros((21, 3), dtype=np.float32)
        for j in range(21):
            if kpts_3d[j, 3] > 0.5:
                z = max(kpts_3d[j, 2], 1e-4)
                proj_2d[j, 0] = focal * kpts_3d[j, 0] / z + ppx
                proj_2d[j, 1] = focal * kpts_3d[j, 1] / z + ppy
                proj_2d[j, 2] = 1.0
        proj3d_full = _draw_webdataset_skeleton(img_bgr, proj_2d)
    proj3d_crop = cv2.warpAffine(proj3d_full, A, (out_size, out_size))

    # Panel 3: MANO mesh
    if has_pose:
        vertices, _ = _mano_forward(mano_model, hand_pose, betas, is_right, kpts_3d)
        focal = _estimate_focal(kpts_3d, kpts_2d, img_w, img_h)
        cam_t = np.zeros(3, dtype=np.float32)
        faces = faces_right if is_right else faces_left
        mesh_color = (200, 220, 255) if is_right else (255, 230, 200)
        mesh_full = _render_mesh_cpu(img_bgr, vertices, faces, cam_t, focal, mesh_color)
    else:
        mesh_full = img_bgr.copy()
        cv2.putText(mesh_full, "No MANO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    mesh_crop = cv2.warpAffine(mesh_full, A, (out_size, out_size))

    side_label = "R" if is_right else "L"
    cv2.putText(skel_crop, "2D KP (crop)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(proj3d_crop, "Proj 3D (crop)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(mesh_crop, "MANO (crop)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for panel in (skel_crop, proj3d_crop, mesh_crop):
        cv2.putText(panel, side_label, (out_size - 20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return np.concatenate([overview, skel_crop, proj3d_crop, mesh_crop], axis=1)


# ── ClipDataset format ────────────────────────────────────────────

def _decode_clipdataset_imgname(s):
    """Decode imgname from ClipDataset labels, stripping absolute cluster prefixes."""
    s = s.decode('utf-8') if isinstance(s, bytes) else str(s)
    if os.path.isabs(s):
        for marker in ('images_zips/',):
            idx = s.find(marker)
            if idx != -1:
                return s[idx + len(marker):]
        return s.lstrip('/')
    return s


def collect_samples_clipdataset(src_dir: str, max_seqs: int = 50):
    """Collect ClipDataset samples: (label_path, frame_idx).

    Reads *_clip.data.pyd master index files to find per-sequence .pyd labels,
    then picks random frames from each sequence.
    """
    master_files = [f for f in os.listdir(src_dir) if f.endswith("_clip.data.pyd")]
    if not master_files:
        return []

    all_label_paths = []
    for mf in master_files:
        master = np.load(os.path.join(src_dir, mf), allow_pickle=True)
        label_dir = src_dir
        for lname in master["labelname"]:
            lpath = os.path.join(label_dir, str(lname))
            if os.path.exists(lpath):
                all_label_paths.append(lpath)

    if len(all_label_paths) > max_seqs:
        all_label_paths = random.sample(all_label_paths, max_seqs)

    samples = []
    for lpath in all_label_paths:
        data = np.load(lpath, allow_pickle=True)
        T = len(data["imgname"])
        frame_idx = random.randint(0, T - 1)
        samples.append((lpath, frame_idx))
    return samples


def render_frame_clipdataset(label_path: str, frame_idx: int, img_dir: str,
                             mano_model=None, faces_right=None, faces_left=None,
                             video_dir: str = None):
    """Render a ClipDataset frame with GT 2D skeleton, projected 3D skeleton, and MANO mesh.

    Frames are read either from individual jpg files under ``img_dir`` or, when
    ``video_dir`` is given, from per-sequence mp4 files using the ``.frames.npy``
    sidecar map (matching ``models_clip.datasets.video_dataset``).
    """
    data = np.load(label_path, allow_pickle=True)

    imgname_raw = data["imgname"][frame_idx]
    imgname_rel = _decode_clipdataset_imgname(imgname_raw)
    if video_dir is not None:
        try:
            video_path, v_idx = _clip_imgname_to_video(imgname_rel, video_dir)
        except Exception as e:
            print(f"  Cannot map {imgname_rel} to video: {e}")
            return None
        img_bgr = read_video_frame(video_path, v_idx)
        if img_bgr is None:
            print(f"  Cannot read frame {v_idx} from {video_path}")
            return None
    else:
        img_path = os.path.join(img_dir, imgname_rel)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  Cannot read image: {img_path}")
            return None

    kp2d = data["hand_keypoints_2d"][frame_idx]  # (21, 3)
    kp3d = data["hand_keypoints_3d"][frame_idx]  # (21, 4)
    K = data["focal"][frame_idx]                  # (3, 3)
    is_right = bool(data["right"][frame_idx])
    img_h, img_w = img_bgr.shape[:2]

    # Panel 1: GT 2D keypoints
    skel_2d_img = _draw_webdataset_skeleton(img_bgr, kp2d)

    # Panel 2: Projected 3D keypoints (kp3d is in camera space, project with K)
    proj3d_img = img_bgr.copy()
    valid_3d = kp3d[:, 3] > 0.5
    proj_2d = np.zeros((21, 3), dtype=np.float32)
    if valid_3d.any():
        pts_cam = kp3d[valid_3d, :3]
        pts_proj = (K @ pts_cam.T).T
        z = np.clip(pts_proj[:, 2:3], 1e-8, None)
        pts_px = pts_proj[:, :2] / z
        proj_2d[valid_3d, :2] = pts_px
        proj_2d[valid_3d, 2] = 1.0
        proj3d_img = _draw_webdataset_skeleton(img_bgr, proj_2d)

    # Panel 3: Error overlay (both skeletons on same image, GT=green, proj=red)
    error_img = img_bgr.copy()
    valid_both = (kp2d[:, 2] > 0.5) & valid_3d
    if valid_both.any():
        gt_pts = kp2d[valid_both, :2]
        pr_pts = proj_2d[valid_both, :2]
        errors = np.linalg.norm(gt_pts - pr_pts, axis=1)
        mean_err = errors.mean()
        max_err = errors.max()

        # Draw GT in green
        for i in range(21):
            if kp2d[i, 2] < 0.5:
                continue
            x, y = int(round(kp2d[i, 0])), int(round(kp2d[i, 1]))
            cv2.circle(error_img, (x, y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        # Draw projected in red
        for i in range(21):
            if proj_2d[i, 2] < 0.5:
                continue
            x, y = int(round(proj_2d[i, 0])), int(round(proj_2d[i, 1]))
            cv2.circle(error_img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
        # Draw error lines
        for i in range(21):
            if kp2d[i, 2] > 0.5 and proj_2d[i, 2] > 0.5:
                p1 = (int(round(kp2d[i, 0])), int(round(kp2d[i, 1])))
                p2 = (int(round(proj_2d[i, 0])), int(round(proj_2d[i, 1])))
                cv2.line(error_img, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(error_img, f"mean:{mean_err:.1f}px max:{max_err:.1f}px",
                    (5, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        cv2.putText(error_img, "No valid overlap", (5, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Panel 4: MANO mesh rendering
    has_pose = bool(data["has_hand_pose"][frame_idx])
    if has_pose and mano_model is not None:
        hand_pose = data["hand_pose"][frame_idx].astype(np.float32)  # (48,)
        betas = data["betas"][frame_idx].astype(np.float32)          # (10,)
        has_tsl = "hand_tsl" in data and "cTw" in data
        if has_tsl:
            hand_tsl = data["hand_tsl"][frame_idx].astype(np.float32)
            cTw = data["cTw"][frame_idx].astype(np.float64)
            vertices, joints = _mano_forward_clip(
                mano_model, hand_pose, betas, hand_tsl, cTw, is_right)
        else:
            # Fallback: wrist-align to kp3d (no hand_tsl available)
            vertices, joints = _mano_forward(mano_model, hand_pose, betas, is_right, kp3d)
        faces = faces_right if is_right else faces_left
        mesh_color = (200, 220, 255) if is_right else (255, 230, 200)
        # Vertices are in camera space; project with full K matrix
        mesh_img = _render_mesh_cpu_K(img_bgr, vertices, faces, K, mesh_color)
    else:
        mesh_img = img_bgr.copy()
        if mano_model is None:
            cv2.putText(mesh_img, "No MANO model", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(mesh_img, "No MANO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    side_label = "R" if is_right else "L"
    cv2.putText(skel_2d_img, "GT 2D", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(proj3d_img, "Proj 3D", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(error_img, "Error (G=GT R=Proj)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(mesh_img, "MANO Mesh", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for panel in [skel_2d_img, proj3d_img, error_img, mesh_img]:
        cv2.putText(panel, side_label, (img_w - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return np.concatenate([skel_2d_img, proj3d_img, error_img, mesh_img], axis=1)


# ── Shared drawing helper ──────────────────────────────────────────
def _draw_skeleton_on_image(img, kp_2d):
    """Draw skeleton bones and keypoints on img (in-place). kp_2d is (21, 2)."""
    for i in range(1, 21):
        p = PARENTS[i]
        pt1 = (int(kp_2d[i, 0]), int(kp_2d[i, 1]))
        pt2 = (int(kp_2d[p, 0]), int(kp_2d[p, 1]))
        color = _joint_color(i)
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(kp_2d):
        color = _joint_color(i)
        cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualize frames with projected 3D keypoints")
    parser.add_argument("--src", default="CONVERTED/dex_ycb",
                        help="Converted dataset directory (HDF5/NPZ) or WebDataset tar directory")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of frames to sample")
    parser.add_argument("--out", default="outputs",
                        help="Output directory for saved images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--mano-dir", default=DEFAULT_MANO_DIR,
                        help="MANO model directory (only for webdataset format)")
    parser.add_argument("--max-tars", type=int, default=5,
                        help="Max tar files to scan (only for webdataset format)")
    parser.add_argument("--img-dir", default=None,
                        help="Image root directory for clipdataset format "
                             "(required unless --video-dir is given)")
    parser.add_argument("--video-dir", default=None,
                        help="Video root directory for clipdataset format. If given, "
                             "frames are decoded from per-sequence mp4s + .frames.npy "
                             "sidecars instead of individual image files.")
    parser.add_argument("--crop", action="store_true",
                        help="For webdataset format: crop each panel to the labeled "
                             "bbox (center, scale). Output panels are square.")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Crop output size in pixels (default: 256)")
    parser.add_argument("--crop-rescale", type=float, default=1.0,
                        help="Extra expansion on the labeled bbox before cropping "
                             "(default: 1.0 — use the bbox as-is)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fmt = detect_format(args.src)
    print(f"Detected format: {fmt}")

    os.makedirs(args.out, exist_ok=True)
    
    if fmt == "clipdataset":
        if not args.img_dir and not args.video_dir:
            print("ERROR: --img-dir or --video-dir is required for clipdataset format")
            return

        datasets = ['arctic', 'dexycb', 'ho2o', 'ho3d', 'hot3d', 'interhand']
        
        for dataset in datasets:
            if dataset in args.src:
                dataset_name = dataset
                pass

        mano_dir = os.path.normpath(args.mano_dir)
        print(f"Loading MANO from {mano_dir}")
        # MANO_RIGHT only; left hands are mirrored inside _mano_forward_clip /
        # _mano_forward (same trick the training dataloader uses) so MANO_LEFT.pkl
        # — which has the smplx#48 shapedirs bug — is never loaded.
        mano_model = {
            "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True, flat_hand_mean=False),
        }
        faces_right = mano_model["right"].faces.astype(np.int32)
        faces_left = faces_right[:, [0, 2, 1]]

        print(f"Collecting ClipDataset samples from {args.src} ...")
        samples = collect_samples_clipdataset(args.src, max_seqs=max(args.n * 2, 100))
        print(f"Found {len(samples)} frame samples")
        if not samples:
            print("No samples found!")
            return

        picks = random.sample(samples, min(args.n, len(samples)))

        saved = 0
        for i, (label_path, frame_idx) in enumerate(picks):
            img = render_frame_clipdataset(label_path, frame_idx, args.img_dir,
                                           mano_model, faces_right, faces_left,
                                           video_dir=args.video_dir)
            if img is not None:
                if dataset_name == 'arctic':
                    el = label_path.split(os.sep)
                    seq_name = '_'.join([el[-4], el[-3], el[-2], os.path.splitext(el[-1])[0]])
                    print(seq_name)
                else:
                    seq_name = os.path.splitext(os.path.basename(label_path))[0]
                out_path = os.path.join(args.out, f"{dataset_name}_{seq_name}_f{frame_idx:04d}.jpg")
                cv2.imwrite(out_path, img)
                saved += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(picks)}]")

    elif fmt == "webdataset":
        dataset_name = os.path.basename(os.path.normpath(args.src))
        mano_dir = os.path.normpath(args.mano_dir)
        print(f"Loading MANO from {mano_dir}")
        # MANO_RIGHT only; left hands are mirrored inside _mano_forward_clip /
        # _mano_forward (same trick the training dataloader uses) so MANO_LEFT.pkl
        # — which has the smplx#48 shapedirs bug — is never loaded.
        mano_model = {
            "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True, flat_hand_mean=False),
        }
        faces_right = mano_model["right"].faces.astype(np.int32)
        faces_left = faces_right[:, [0, 2, 1]]

        print(f"Scanning {args.src} ...")
        samples = collect_samples_webdataset(args.src, max_tars=args.max_tars)
        print(f"Found {len(samples)} samples")
        if not samples:
            print("No samples found!")
            return

        picks = random.sample(samples, min(args.n, len(samples)))

        saved = 0
        tag = "crop" if args.crop else "full"
        for i, (tar_path, base_name) in enumerate(picks):
            if args.crop:
                img = render_frame_webdataset_crop(
                    tar_path, base_name, mano_model, faces_right, faces_left,
                    out_size=args.crop_size, rescale=args.crop_rescale)
            else:
                img = render_frame_webdataset(
                    tar_path, base_name, mano_model, faces_right, faces_left)
            if img is not None:
                sample_id = os.path.basename(base_name)
                out_path = os.path.join(
                    args.out, f"webdataset_{dataset_name}_{tag}_{sample_id}.jpg")
                cv2.imwrite(out_path, img)
                saved += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(picks)}]")

    else:  # hdf5
        print("Collecting HDF5 samples...")
        samples = collect_samples_hdf5(args.src)
        print(f"Found {len(samples)} sequences with active hands")
        if not samples:
            print("No samples found!")
            return

        picks = []
        for _ in range(args.n):
            seq = random.choice(samples)
            frame_idx = random.randint(0, seq[2] - 1)
            picks.append((seq[0], seq[1], frame_idx, seq[3]))

        print(f"Rendering {args.n} frames...")
        saved = 0
        for i, (hdf5_path, video_path, frame_idx, sides) in enumerate(picks):
            img = render_frame_hdf5(hdf5_path, video_path, frame_idx, sides)
            if img is not None:
                cluster = os.path.basename(os.path.dirname(hdf5_path))
                seq_name = os.path.basename(hdf5_path).replace(".hdf5", "")
                out_path = os.path.join(args.out, f"{dataset_name}_{cluster}_{seq_name}_f{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, img)
                saved += 1
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{args.n}")

    print(f"Saved {saved} images to {args.out}/")


if __name__ == "__main__":
    main()
