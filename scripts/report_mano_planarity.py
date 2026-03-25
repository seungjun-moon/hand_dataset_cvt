#!/usr/bin/env python3
"""Generate MANO planarity report comparing dataset_tars vs dataset_tars_manotorch.

For each dataset:
  1. Compute planarity error from MANO forward pass (smplx) on both dirs
  2. Compute per-sample vertex error between vanilla and constrained MANO
  3. Compute correlation between planarity error and vertex error

Usage:
    python scripts/report_mano_planarity.py \
        --vanilla ../hamer/hamer_training_data/dataset_tars \
        --constrained ../hamer/hamer_training_data/dataset_tars_manotorch \
        --max-tars 50 --out reports/dataset_mano_realhand.md
"""

import argparse
import os
import pickle
import sys
import tarfile
from datetime import datetime

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MANO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "hamer", "_DATA", "data", "mano"
)

# MANO joint indices per finger (MCP, PIP, DIP, Tip) in 21-joint layout
MANO_FINGER_INDICES = {
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Little": [17, 18, 19, 20],
}

MANO_FINGERTIP_VERTEX_IDS = [744, 320, 443, 554, 671]


def planarity_error(points):
    """Planarity error for (4, 3) points via SVD."""
    centered = points - points.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return float(s[-1])


def sample_planarity(joints_21):
    """Mean planarity error across 4 fingers for a (21,3) joint array. Returns meters."""
    errors = []
    for indices in MANO_FINGER_INDICES.values():
        pts = joints_21[indices]
        errors.append(planarity_error(pts))
    return np.mean(errors)


def build_joints_21(joints_16, vertices):
    """Build 21-joint array from 16 MANO joints + 5 fingertip vertices."""
    tips = vertices[MANO_FINGERTIP_VERTEX_IDS]
    joints_21 = np.zeros((21, 3), dtype=np.float32)
    joints_21[0] = joints_16[0]
    for fi in range(5):
        src = 1 + fi * 3
        dst = 1 + fi * 4
        joints_21[dst:dst + 3] = joints_16[src:src + 3]
        joints_21[dst + 3] = tips[fi]
    return joints_21


def batch_rodrigues(aa_batch):
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices. Torch."""
    import torch
    angle = torch.norm(aa_batch, dim=1, keepdim=True).clamp(min=1e-8)
    axis = aa_batch / angle
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    K = torch.zeros(aa_batch.shape[0], 3, 3, device=aa_batch.device, dtype=aa_batch.dtype)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    eye = torch.eye(3, device=aa_batch.device, dtype=aa_batch.dtype).unsqueeze(0)
    return eye + sin_a * K + (1 - cos_a) * (K @ K)


def mano_forward_batch(model, hand_pose_aa, betas, device="cpu"):
    """Batched MANO forward. Returns joints_16 (B,16,3), vertices (B,778,3)."""
    import torch
    B = hand_pose_aa.shape[0]
    aa_all = torch.from_numpy(hand_pose_aa).float().to(device)
    global_orient = batch_rodrigues(aa_all[:, :3]).unsqueeze(1)
    hand_pose = batch_rodrigues(aa_all[:, 3:48].reshape(-1, 3)).reshape(B, 15, 3, 3)
    betas_t = torch.from_numpy(betas).float().to(device)

    with torch.no_grad():
        out = model(global_orient=global_orient, hand_pose=hand_pose,
                    betas=betas_t, pose2rot=False)
    return out.joints.cpu().numpy(), out.vertices.cpu().numpy()


def load_annotations(src_dir, dataset, max_tars):
    """Load all annotations from a dataset tar dir.

    Returns dict: {sample_id: annotation_dict}
    """
    ds_dir = os.path.join(src_dir, dataset)
    if not os.path.isdir(ds_dir):
        return {}
    tar_files = sorted([f for f in os.listdir(ds_dir) if f.endswith(".tar")])
    if max_tars > 0:
        tar_files = tar_files[:max_tars]

    anns = {}
    for tar_name in tar_files:
        tar_path = os.path.join(ds_dir, tar_name)
        with tarfile.open(tar_path) as tf:
            names = set(tf.getnames())
            for pyd_name in sorted(n for n in names if n.endswith(".data.pyd")):
                try:
                    pyd_bytes = tf.extractfile(pyd_name).read()
                    ann = pickle.loads(pyd_bytes)[0]
                except Exception:
                    continue
                sample_id = pyd_name.replace(".data.pyd", "")
                anns[sample_id] = ann
    return anns


def analyze_dataset(dataset, vanilla_anns, constrained_anns, mano_models, device,
                    batch_size=512):
    """Analyze a single dataset.

    Returns dict with:
        n_samples, n_mano_samples,
        planarity_vanilla, planarity_constrained (per-finger and mean, in mm),
        vertex_error_mean, vertex_error_median, vertex_error_max (in mm),
        correlation (Pearson r between planarity and vertex error)
    """
    import torch

    # Find common samples with MANO params
    common_ids = sorted(set(vanilla_anns.keys()) & set(constrained_anns.keys()))
    if not common_ids:
        return None

    # Filter to samples with hand_pose
    mano_ids_right = []
    mano_ids_left = []
    for sid in common_ids:
        v_ann = vanilla_anns[sid]
        c_ann = constrained_anns[sid]
        if not (bool(v_ann.get("has_hand_pose", 0) > 0.5) and
                bool(c_ann.get("has_hand_pose", 0) > 0.5)):
            continue
        is_right = bool(v_ann["right"] > 0.5)
        if is_right:
            mano_ids_right.append(sid)
        else:
            mano_ids_left.append(sid)

    n_total = len(common_ids)
    n_mano = len(mano_ids_right) + len(mano_ids_left)
    if n_mano == 0:
        return {"n_samples": n_total, "n_mano_samples": 0}

    all_plan_vanilla = []
    all_plan_constrained = []
    all_vertex_err = []

    for is_right, sample_ids in [(True, mano_ids_right), (False, mano_ids_left)]:
        if not sample_ids:
            continue
        side_key = "right" if is_right else "left"
        model = mano_models[side_key]

        # Collect batch arrays
        v_poses = np.array([vanilla_anns[s]["hand_pose"] for s in sample_ids], dtype=np.float32)
        v_betas = np.array([vanilla_anns[s]["betas"] for s in sample_ids], dtype=np.float32)
        c_poses = np.array([constrained_anns[s]["hand_pose"] for s in sample_ids], dtype=np.float32)
        c_betas = np.array([constrained_anns[s]["betas"] for s in sample_ids], dtype=np.float32)

        N = len(sample_ids)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            # Vanilla forward
            j16_v, verts_v = mano_forward_batch(
                model, v_poses[start:end], v_betas[start:end], device)
            # Constrained forward
            j16_c, verts_c = mano_forward_batch(
                model, c_poses[start:end], c_betas[start:end], device)

            for i in range(end - start):
                j21_v = build_joints_21(j16_v[i], verts_v[i])
                j21_c = build_joints_21(j16_c[i], verts_c[i])

                plan_v = sample_planarity(j21_v)
                plan_c = sample_planarity(j21_c)
                all_plan_vanilla.append(plan_v)
                all_plan_constrained.append(plan_c)

                # Vertex error: mean per-vertex L2 distance
                v_err = np.mean(np.linalg.norm(verts_v[i] - verts_c[i], axis=1))
                all_vertex_err.append(v_err)

    plan_v = np.array(all_plan_vanilla) * 1000  # mm
    plan_c = np.array(all_plan_constrained) * 1000
    v_err = np.array(all_vertex_err) * 1000

    # Per-finger planarity (vanilla)
    per_finger_v = {}
    per_finger_c = {}
    for is_right, sample_ids in [(True, mano_ids_right), (False, mano_ids_left)]:
        if not sample_ids:
            continue
        side_key = "right" if is_right else "left"
        model = mano_models[side_key]
        v_poses = np.array([vanilla_anns[s]["hand_pose"] for s in sample_ids], dtype=np.float32)
        v_betas = np.array([vanilla_anns[s]["betas"] for s in sample_ids], dtype=np.float32)
        c_poses = np.array([constrained_anns[s]["hand_pose"] for s in sample_ids], dtype=np.float32)
        c_betas = np.array([constrained_anns[s]["betas"] for s in sample_ids], dtype=np.float32)

        N = len(sample_ids)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            j16_v, verts_v = mano_forward_batch(model, v_poses[start:end], v_betas[start:end], device)
            j16_c, verts_c = mano_forward_batch(model, c_poses[start:end], c_betas[start:end], device)
            for i in range(end - start):
                j21_v = build_joints_21(j16_v[i], verts_v[i])
                j21_c = build_joints_21(j16_c[i], verts_c[i])
                for fname, indices in MANO_FINGER_INDICES.items():
                    ev = planarity_error(j21_v[indices]) * 1000
                    ec = planarity_error(j21_c[indices]) * 1000
                    per_finger_v.setdefault(fname, []).append(ev)
                    per_finger_c.setdefault(fname, []).append(ec)

    # Correlation
    corr = float(np.corrcoef(plan_v, v_err)[0, 1]) if len(plan_v) > 2 else 0.0

    return {
        "n_samples": n_total,
        "n_mano_samples": n_mano,
        "planarity_vanilla_mean": float(np.mean(plan_v)),
        "planarity_vanilla_median": float(np.median(plan_v)),
        "planarity_constrained_mean": float(np.mean(plan_c)),
        "planarity_constrained_median": float(np.median(plan_c)),
        "per_finger_vanilla": {f: float(np.mean(v)) for f, v in per_finger_v.items()},
        "per_finger_constrained": {f: float(np.mean(v)) for f, v in per_finger_c.items()},
        "vertex_error_mean": float(np.mean(v_err)),
        "vertex_error_median": float(np.median(v_err)),
        "vertex_error_max": float(np.max(v_err)),
        "correlation": corr,
    }


def write_report(results, out_path, args):
    """Write markdown report."""
    lines = []
    lines.append("# MANO Planarity & Vertex Error Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Vanilla: `{args.vanilla}`")
    lines.append(f"- Constrained: `{args.constrained}`")
    lines.append(f"- Max tars per dataset: {args.max_tars}")
    lines.append(f"- MANO backend: smplx")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Samples | Planarity Vanilla (mm) | Planarity Constrained (mm) | Vertex Error (mm) | Correlation |")
    lines.append("|---------|--------:|----------------------:|--------------------------:|------------------:|------------:|")

    for ds, r in sorted(results.items()):
        if r is None or r.get("n_mano_samples", 0) == 0:
            lines.append(f"| {ds} | {r['n_samples'] if r else 0} | - | - | - | - |")
            continue
        lines.append(
            f"| {ds} | {r['n_mano_samples']} "
            f"| {r['planarity_vanilla_mean']:.2f} "
            f"| {r['planarity_constrained_mean']:.2f} "
            f"| {r['vertex_error_mean']:.2f} "
            f"| {r['correlation']:.3f} |"
        )

    # Per-finger breakdown
    lines.append("")
    lines.append("## Per-Finger Planarity (mean, mm)")
    lines.append("")
    lines.append("### Vanilla MANO")
    lines.append("")
    lines.append("| Dataset | Index | Middle | Ring | Little |")
    lines.append("|---------|------:|-------:|-----:|-------:|")
    for ds, r in sorted(results.items()):
        if r is None or r.get("n_mano_samples", 0) == 0:
            continue
        pf = r["per_finger_vanilla"]
        lines.append(
            f"| {ds} "
            f"| {pf.get('Index', 0):.2f} "
            f"| {pf.get('Middle', 0):.2f} "
            f"| {pf.get('Ring', 0):.2f} "
            f"| {pf.get('Little', 0):.2f} |"
        )

    lines.append("")
    lines.append("### Constrained MANO")
    lines.append("")
    lines.append("| Dataset | Index | Middle | Ring | Little |")
    lines.append("|---------|------:|-------:|-----:|-------:|")
    for ds, r in sorted(results.items()):
        if r is None or r.get("n_mano_samples", 0) == 0:
            continue
        pf = r["per_finger_constrained"]
        lines.append(
            f"| {ds} "
            f"| {pf.get('Index', 0):.2f} "
            f"| {pf.get('Middle', 0):.2f} "
            f"| {pf.get('Ring', 0):.2f} "
            f"| {pf.get('Little', 0):.2f} |"
        )

    # Vertex error details
    lines.append("")
    lines.append("## Vertex Error Details (mm)")
    lines.append("")
    lines.append("Vertex error = mean per-vertex L2 distance between vanilla and constrained MANO meshes.")
    lines.append("")
    lines.append("| Dataset | Mean | Median | Max |")
    lines.append("|---------|-----:|-------:|----:|")
    for ds, r in sorted(results.items()):
        if r is None or r.get("n_mano_samples", 0) == 0:
            continue
        lines.append(
            f"| {ds} "
            f"| {r['vertex_error_mean']:.2f} "
            f"| {r['vertex_error_median']:.2f} "
            f"| {r['vertex_error_max']:.2f} |"
        )

    # Correlation analysis
    lines.append("")
    lines.append("## Correlation: Planarity Error vs Vertex Error")
    lines.append("")
    lines.append("Pearson correlation between per-sample mean planarity error (vanilla MANO)")
    lines.append("and per-sample mean vertex error (vanilla vs constrained).")
    lines.append("Higher correlation means samples with worse planarity see larger mesh changes after constrained IK.")
    lines.append("")
    lines.append("| Dataset | Pearson r | Interpretation |")
    lines.append("|---------|----------:|----------------|")
    for ds, r in sorted(results.items()):
        if r is None or r.get("n_mano_samples", 0) == 0:
            continue
        corr = r["correlation"]
        if corr > 0.7:
            interp = "Strong positive"
        elif corr > 0.4:
            interp = "Moderate positive"
        elif corr > 0.2:
            interp = "Weak positive"
        elif corr > -0.2:
            interp = "No correlation"
        elif corr > -0.4:
            interp = "Weak negative"
        else:
            interp = "Moderate/strong negative"
        lines.append(f"| {ds} | {corr:.3f} | {interp} |")

    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MANO planarity comparison report")
    parser.add_argument("--vanilla", required=True,
                        help="Path to dataset_tars/ (vanilla MANO)")
    parser.add_argument("--constrained", required=True,
                        help="Path to dataset_tars_manotorch/ (constrained MANO)")
    parser.add_argument("--max-tars", type=int, default=50)
    parser.add_argument("--mano-dir", default=DEFAULT_MANO_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", default="reports/dataset_mano_realhand.md")
    args = parser.parse_args()

    import torch
    import smplx

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    mano_dir = os.path.normpath(args.mano_dir)
    print(f"Loading MANO from {mano_dir} (device={device})")
    mano_models = {
        "right": smplx.MANOLayer(model_path=mano_dir, is_rhand=True,
                                 flat_hand_mean=False).to(device),
        "left": smplx.MANOLayer(model_path=mano_dir, is_rhand=False,
                                flat_hand_mean=False).to(device),
    }

    # Discover datasets
    datasets = sorted(set(
        d for d in os.listdir(args.vanilla)
        if os.path.isdir(os.path.join(args.vanilla, d))
    ) & set(
        d for d in os.listdir(args.constrained)
        if os.path.isdir(os.path.join(args.constrained, d))
    ))
    print(f"Found {len(datasets)} common datasets: {datasets}")

    results = {}
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {ds}")
        print(f"{'='*60}")

        print(f"  Loading vanilla annotations...")
        vanilla_anns = load_annotations(args.vanilla, ds, args.max_tars)
        print(f"  Loading constrained annotations...")
        constrained_anns = load_annotations(args.constrained, ds, args.max_tars)
        print(f"  Vanilla: {len(vanilla_anns)}, Constrained: {len(constrained_anns)}")

        print(f"  Analyzing...")
        r = analyze_dataset(ds, vanilla_anns, constrained_anns, mano_models,
                            device, args.batch_size)
        results[ds] = r

        if r and r.get("n_mano_samples", 0) > 0:
            print(f"  MANO samples: {r['n_mano_samples']}")
            print(f"  Planarity vanilla:     {r['planarity_vanilla_mean']:.2f} mm")
            print(f"  Planarity constrained: {r['planarity_constrained_mean']:.2f} mm")
            print(f"  Vertex error:          {r['vertex_error_mean']:.2f} mm")
            print(f"  Correlation:           {r['correlation']:.3f}")
        else:
            print(f"  No MANO samples")

    write_report(results, args.out, args)


if __name__ == "__main__":
    main()
