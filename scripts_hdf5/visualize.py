#!/usr/bin/env python3
"""Visualize randomly sampled frames with projected 3D keypoints.

Reads CONVERTED datasets (HDF5+MP4 or NPZ+JPG) or HaMER WebDataset tars,
projects world-space 3D keypoints to 2D, and draws skeleton overlays on images.

Automatically detects the dataset format:
  - HDF5: *_label_*.hdf5 + *_video_*.mp4 (video sequences)
  - NPZ:  *.npz + *.jpg (image-wise, e.g. FreiHAND)
  - WebDataset: *.tar containing {id}.jpg + {id}.data.pyd (HaMER format)

Usage:
    python scripts/visualize.py --src CONVERTED/ho_cap --n 10 --out outputs --seed 0
    python scripts/visualize.py --src CONVERTED/freihand_train --n 100 --out outputs --seed 0
    python scripts/visualize.py --src CONVERTED/interhand26m_train --n 20 --out outputs --seed 0
    python scripts/visualize.py --src ../hamer/hamer_training_data/dataset_tars/freihand-train --n 20
    python scripts/visualize.py --src ../hamer/hamer_training_data/dataset_tars/ho3d-train --n 20 --mano-dir /path/to/mano
"""

import argparse
import os
import pickle
import random
import sys
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
    os.path.dirname(__file__), "..", "..", "hamer", "_DATA", "data", "mano"
)


def _joint_color(joint_idx: int):
    if joint_idx == 0:
        return WRIST_COLOR
    return FINGER_COLORS[(joint_idx - 1) // 4]


# ── Format detection ───────────────────────────────────────────────
def detect_format(src_dir: str) -> str:
    """Detect dataset format by checking file extensions."""
    # Check top-level for tar files (webdataset)
    for fname in os.listdir(src_dir):
        if fname.endswith(".tar"):
            return "webdataset"

    # Check subdirectories for hdf5/npz
    for cluster in sorted(os.listdir(src_dir)):
        cluster_dir = os.path.join(src_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        for fname in sorted(os.listdir(cluster_dir)):
            if fname.endswith(".hdf5"):
                return "hdf5"
            if fname.endswith(".npz"):
                return "npz"
    return "hdf5"


# ── HDF5 format ────────────────────────────────────────────────────
def collect_samples_hdf5(src_dir: str):
    """Collect HDF5 samples: (hdf5_path, video_path, frame_count, active_sides)."""
    samples = []
    for cluster in sorted(os.listdir(src_dir)):
        cluster_dir = os.path.join(src_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        for fname in sorted(os.listdir(cluster_dir)):
            if not fname.endswith(".hdf5"):
                continue
            hdf5_path = os.path.join(cluster_dir, fname)
            video_name = fname.replace("_label_", "_video_").replace(".hdf5", ".mp4")
            video_path = os.path.join(cluster_dir, video_name)
            if not os.path.exists(video_path):
                continue
            sides = get_active_sides(hdf5_path)
            if not sides:
                continue
            with h5py.File(hdf5_path, "r") as f:
                sample_key = f"{sides[0]}Hand"
                n_frames = f[f"transforms/{sample_key}"].shape[0]
            samples.append((hdf5_path, video_path, n_frames, sides))
    return samples


def read_video_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def render_frame_hdf5(hdf5_path: str, video_path: str, frame_idx: int, sides: list):
    """Render a single HDF5 frame with projected 3D keypoint skeleton."""
    img = read_video_frame(video_path, frame_idx)
    if img is None:
        return None

    with h5py.File(hdf5_path, "r") as f:
        intrinsic = f["camera/intrinsic"][:]
        cam_pose = f["transforms/camera"][frame_idx]

        for side in sides:
            kp_world = np.zeros((21, 3), dtype=np.float32)
            for j, suffix in enumerate(HAND_JOINT_SUFFIXES):
                name = f"{side}{suffix}"
                kp_world[j] = f[f"transforms/{name}"][frame_idx, :3, 3]

            kp_2d = project_3d_to_2d(kp_world, cam_pose, intrinsic)

            h, w = img.shape[:2]
            in_bounds = np.any(
                (kp_2d[:, 0] >= -w) & (kp_2d[:, 0] < 2 * w) &
                (kp_2d[:, 1] >= -h) & (kp_2d[:, 1] < 2 * h)
            )
            if not in_bounds:
                continue

            _draw_skeleton_on_image(img, kp_2d)

            wrist_2d = kp_2d[0]
            label_pos = (int(wrist_2d[0]) - 10, int(wrist_2d[1]) - 15)
            cv2.putText(img, side[0].upper(), label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        SIDE_COLORS[side], 2, cv2.LINE_AA)

    return img


# ── NPZ format ─────────────────────────────────────────────────────
def collect_samples_npz(src_dir: str, max_per_cluster: int = 10):
    """Collect NPZ samples: (npz_path, img_path)."""
    samples = []
    for cluster in sorted(os.listdir(src_dir)):
        cluster_dir = os.path.join(src_dir, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        npz_files = [f for f in os.listdir(cluster_dir) if f.endswith(".npz")]
        if not npz_files:
            continue
        if len(npz_files) > max_per_cluster:
            npz_files = random.sample(npz_files, max_per_cluster)
        for fname in npz_files:
            npz_path = os.path.join(cluster_dir, fname)
            img_path = npz_path.replace(".npz", ".jpg")
            if os.path.exists(img_path):
                samples.append((npz_path, img_path))
    return samples


def render_frame_npz(npz_path: str, img_path: str):
    """Render a single NPZ image with projected 3D keypoint skeleton."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    data = np.load(npz_path, allow_pickle=True)
    intrinsic = data["intrinsic"]
    cam_ext = data["cam_ext"]
    kp_world = data["kpt3d_world"]  # (21, 3)
    side = str(data["side"])

    kp_2d = project_3d_to_2d(kp_world, cam_ext, intrinsic)

    h, w = img.shape[:2]
    in_bounds = np.any(
        (kp_2d[:, 0] >= -w) & (kp_2d[:, 0] < 2 * w) &
        (kp_2d[:, 1] >= -h) & (kp_2d[:, 1] < 2 * h)
    )
    if not in_bounds:
        return img

    _draw_skeleton_on_image(img, kp_2d)

    wrist_2d = kp_2d[0]
    label_pos = (int(wrist_2d[0]) - 10, int(wrist_2d[1]) - 15)
    cv2.putText(img, side[0].upper(), label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                SIDE_COLORS[side], 2, cv2.LINE_AA)

    return img


# ── WebDataset format ──────────────────────────────────────────────
def collect_samples_webdataset(src_dir: str, max_tars: int = 5):
    """Collect WebDataset samples: (tar_path, base_name)."""
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
    """Run MANO forward pass from axis-angle params."""
    import torch
    model = mano_model["right"] if is_right else mano_model["left"]

    global_orient_aa = hand_pose_aa[:3]
    hand_pose_aa_15 = hand_pose_aa[3:48].reshape(15, 3)

    global_orient = torch.from_numpy(_axis_angle_to_rotmat(global_orient_aa)).unsqueeze(0).unsqueeze(0)
    hand_pose = torch.stack(
        [torch.from_numpy(_axis_angle_to_rotmat(hand_pose_aa_15[j])) for j in range(15)]
    ).unsqueeze(0)
    betas_t = torch.from_numpy(betas).unsqueeze(0).float()

    with torch.no_grad():
        out = model(global_orient=global_orient, hand_pose=hand_pose, betas=betas_t, pose2rot=False)

    vertices = out.vertices[0].numpy()
    joints = out.joints[0].numpy()

    if kpts_3d is not None and kpts_3d[0, 3] > 0.5:
        offset = kpts_3d[0, :3] - joints[0]
        vertices = vertices + offset
        joints = joints + offset

    return vertices, joints


def _estimate_focal(kpts_3d, kpts_2d, img_size):
    valid = (kpts_2d[:, 2] > 0.5) & (kpts_3d[:, 3] > 0.5)
    if valid.sum() < 3:
        return 5000.0

    cx, cy = img_size / 2.0, img_size / 2.0
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
    img_size = img_bgr.shape[0]

    skel_img = _draw_webdataset_skeleton(img_bgr, kpts_2d)

    if has_pose:
        vertices, joints = _mano_forward(mano_model, hand_pose, betas, is_right, kpts_3d)
        focal = _estimate_focal(kpts_3d, kpts_2d, img_size)
        cam_t = np.zeros(3, dtype=np.float32)
        faces = faces_right if is_right else faces_left
        mesh_color = (200, 220, 255) if is_right else (255, 230, 200)
        mesh_img = _render_mesh_cpu(img_bgr, vertices, faces, cam_t, focal, mesh_color)
    else:
        mesh_img = img_bgr.copy()
        cv2.putText(mesh_img, "No MANO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    side_label = "R" if is_right else "L"
    cv2.putText(skel_img, "2D Keypoints", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(mesh_img, "MANO Mesh", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(skel_img, side_label, (img_size - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return np.concatenate([skel_img, mesh_img], axis=1)


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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fmt = detect_format(args.src)
    print(f"Detected format: {fmt}")

    os.makedirs(args.out, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.src))

    if fmt == "npz":
        print("Collecting NPZ samples...")
        samples = collect_samples_npz(args.src)
        print(f"Found {len(samples)} images with hand annotations")
        if not samples:
            print("No samples found!")
            return

        picks = random.sample(samples, min(args.n, len(samples)))

        print(f"Rendering {len(picks)} frames...")
        saved = 0
        for i, (npz_path, img_path) in enumerate(picks):
            img = render_frame_npz(npz_path, img_path)
            if img is not None:
                cluster = os.path.basename(os.path.dirname(npz_path))
                name = os.path.basename(npz_path).replace(".npz", "")
                out_path = os.path.join(args.out, f"{dataset_name}_{cluster}_{name}.jpg")
                cv2.imwrite(out_path, img)
                saved += 1
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(picks)}")

    elif fmt == "webdataset":
        import smplx

        mano_dir = os.path.normpath(args.mano_dir)
        print(f"Loading MANO from {mano_dir}")
        mano_model = {
            "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True, flat_hand_mean=False),
            "left": smplx.MANOLayer(model_path=mano_dir, is_rhand=False, flat_hand_mean=False),
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
        for i, (tar_path, base_name) in enumerate(picks):
            img = render_frame_webdataset(tar_path, base_name, mano_model, faces_right, faces_left)
            if img is not None:
                sample_id = os.path.basename(base_name)
                out_path = os.path.join(args.out, f"webdataset_{dataset_name}_{sample_id}.jpg")
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
