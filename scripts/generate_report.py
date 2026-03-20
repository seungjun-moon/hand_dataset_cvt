#!/usr/bin/env python3
"""Generate dataset_report.md from _meta.json files in CONVERTED/.

Fast: reads only _meta.json files and directory listings — no du, no glob.

Usage:
    python scripts/generate_report.py [CONVERTED_ROOT]
"""
import json
import os
import sys
from datetime import date


CONVERTED = sys.argv[1] if len(sys.argv) > 1 else "CONVERTED"

DATASETS_ORDER = [
    "dex_ycb", "ho_cap", "egodex", "arctic", "whim_train", "whim_test",
    "interhand26m", "freihand_train", "freihand_eval", "rhd", "hic",
]

DISPLAY_NAMES = {
    "dex_ycb": "DexYCB", "ho_cap": "HO-Cap", "egodex": "EgoDex",
    "arctic": "ARCTIC",
    "whim_train": "WHIM Train", "whim_test": "WHIM Test",
    "interhand26m": "InterHand2.6M",
    "freihand_train": "FreiHAND Train", "freihand_eval": "FreiHAND Eval",
    "rhd": "RHD", "hic": "HIC",
}

SOURCES = {
    "dex_ycb": "[DexYCB](https://dex-ycb.github.io/) — tabletop grasping with YCB objects",
    "ho_cap": "[HO-Cap](https://irvlutd.github.io/HOCap/) — hand-object interaction",
    "egodex": "[EgoDex](https://github.com/facebookresearch/ego-dex) — egocentric hand tracking",
    "arctic": "[ARCTIC](https://arctic.is.tue.mpg.de/) — articulated object manipulation with dexterous bimanual hands",
    "whim_train": "[WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data",
    "whim_test": "[WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data",
    "interhand26m": "[InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) — multi-view interacting hands",
    "freihand_train": "[FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — real hand images with MANO annotations",
    "freihand_eval": "[FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — real hand images with MANO annotations",
    "rhd": "[RHD](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) — rendered synthetic hand poses",
    "hic": "[HIC](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/) — hand-in-contact interaction",
}

# FreiHAND has no _meta.json; specify viewpoints for NPZ counting
NPZ_VIEWPOINTS = {"freihand_train": 4, "freihand_eval": 1}


def fmt(n):
    """Format integer with commas."""
    return f"{n:,}"


def gather_meta_stats(dataset_dir):
    """Gather statistics purely from _meta.json files (fast, no file scanning)."""
    st = {"clusters": 0, "files": 0, "left_only": 0, "right_only": 0,
          "both": 0, "frame_counts": [], "unique_frames": 0, "vp_counts": [],
          "format": "HDF5"}

    clusters = sorted(d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d)))
    st["clusters"] = len(clusters)

    for cluster in clusters:
        meta_path = os.path.join(dataset_dir, cluster, "_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        seqs = {}
        for fname, entry in meta.items():
            st["files"] += 1
            sides = entry.get("sides", {})
            has_left = "left" in sides and sides["left"]["valid_ranges"]
            has_right = "right" in sides and sides["right"]["valid_ranges"]
            if has_left and has_right:
                st["both"] += 1
            elif has_left:
                st["left_only"] += 1
            elif has_right:
                st["right_only"] += 1

            nf = max((sd["n_frames"] for sd in sides.values()), default=0)
            st["frame_counts"].append(nf)

            seq_id = fname.split("_label_")[0]
            seqs.setdefault(seq_id, []).append(nf)

        for seq_id, nfs in seqs.items():
            st["unique_frames"] += nfs[0]
            st["vp_counts"].append(len(nfs))

    return st


def gather_npz_stats(dataset_dir, viewpoints):
    """Gather statistics for NPZ datasets by counting directory entries."""
    clusters = sorted(d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d)))
    n_files = sum(
        sum(1 for f in os.listdir(os.path.join(dataset_dir, c)) if f.endswith(".npz"))
        for c in clusters
    )
    n_seqs = n_files // viewpoints
    return {
        "clusters": len(clusters),
        "files": n_files,
        "left_only": 0, "right_only": n_files, "both": 0,
        "frame_counts": [1] * n_files,
        "unique_frames": n_seqs,
        "vp_counts": [viewpoints] * n_seqs,
        "format": "NPZ",
    }


# ---------------------------------------------------------------------------
# Collect stats
# ---------------------------------------------------------------------------
all_stats = {}
for ds in DATASETS_ORDER:
    ds_dir = os.path.join(CONVERTED, ds)
    if not os.path.isdir(ds_dir):
        continue
    if ds in NPZ_VIEWPOINTS:
        all_stats[ds] = gather_npz_stats(ds_dir, NPZ_VIEWPOINTS[ds])
    else:
        all_stats[ds] = gather_meta_stats(ds_dir)

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------
lines = []
lines.append("# Converted Dataset Report")
lines.append(f"\nGenerated: {date.today().isoformat()}")

# --- Summary table ---
lines.append("\n## Summary\n")
lines.append("| Dataset | Format | Sequences | Viewpoints | Data Files | Unique Frames | Total Frames (all views) |")
lines.append("|---------|--------|----------:|-----------:|-----------:|--------------:|-------------------------:|")

totals = {"files": 0, "unique": 0, "total": 0}
for ds in DATASETS_ORDER:
    if ds not in all_stats:
        continue
    st = all_stats[ds]
    fc = st["frame_counts"]
    total_frames = sum(fc)
    uf = st["unique_frames"]
    vc = st["vp_counts"]
    vp_min, vp_max = (min(vc), max(vc)) if vc else (1, 1)
    vp_str = str(vp_min) if vp_min == vp_max else f"{vp_min}–{vp_max}"
    n_seqs = len(vc)

    suffix = r"\*" if ds == "whim_train" else ""
    name = DISPLAY_NAMES[ds]

    lines.append(
        f"| {name:14s} | {st['format']:6s} | {fmt(n_seqs)+suffix:>9s} | "
        f"{vp_str:>10s} | {fmt(st['files'])+suffix:>10s} | "
        f"{fmt(uf)+suffix:>13s} | {fmt(total_frames)+suffix:>24s} |"
    )
    totals["files"] += st["files"]
    totals["unique"] += uf
    totals["total"] += total_frames

lines.append(
    f"| **Total** | | | — | **{fmt(totals['files'])}** | "
    f"**{fmt(totals['unique'])}** | **{fmt(totals['total'])}** |"
)
lines.append(r"""
\* WHIM Train: 176/1,431 videos completed, 1,255 failed (YouTube unavailable).""")

# --- Hand Side Distribution ---
lines.append("\n## Hand Side Distribution\n")
lines.append("| Dataset | Left Only | Right Only | Both Hands | Total Files |")
lines.append("|---------|----------:|-----------:|-----------:|------------:|")
for ds in DATASETS_ORDER:
    if ds not in all_stats:
        continue
    st = all_stats[ds]
    lines.append(
        f"| {DISPLAY_NAMES[ds]:14s} | {fmt(st['left_only']):>9s} | "
        f"{fmt(st['right_only']):>10s} | {fmt(st['both']):>10s} | "
        f"{fmt(st['files']):>11s} |"
    )

# --- Frame Statistics ---
lines.append("\n## Frame Statistics\n")
lines.append("| Dataset | Min | Mean | Max | Unique Frames | Total Frames |")
lines.append("|---------|----:|-----:|----:|--------------:|-------------:|")
for ds in DATASETS_ORDER:
    if ds not in all_stats:
        continue
    st = all_stats[ds]
    fc = st["frame_counts"]
    mn = min(fc) if fc else 0
    mx = max(fc) if fc else 0
    avg = sum(fc) // len(fc) if fc else 0
    lines.append(
        f"| {DISPLAY_NAMES[ds]:14s} | {fmt(mn):>3s} | {fmt(avg):>4s} | "
        f"{fmt(mx):>5s} | {fmt(st['unique_frames']):>13s} | {fmt(sum(fc)):>12s} |"
    )

# --- Per-Dataset Details ---
lines.append("\n## Per-Dataset Details\n")
for ds in DATASETS_ORDER:
    if ds not in all_stats:
        continue
    st = all_stats[ds]
    fc = st["frame_counts"]
    vc = st["vp_counts"]
    name = DISPLAY_NAMES[ds]

    lines.append(f"### {name}\n")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Source | {SOURCES.get(ds, '—')} |")
    lines.append(f"| Clusters | {st['clusters']} |")

    vp_min, vp_max = (min(vc), max(vc)) if vc else (1, 1)
    vp_str = str(vp_min) if vp_min == vp_max else f"{vp_min}–{vp_max}"
    lines.append(f"| Viewpoints | {vp_str} |")

    lines.append(f"| Hand sides | left: {fmt(st['left_only'])} / right: {fmt(st['right_only'])} / both: {fmt(st['both'])} |")

    if fc:
        mn, mx = min(fc), max(fc)
        avg = sum(fc) // len(fc)
        if mn == mx:
            lines.append(f"| Frames/file | {fmt(mn)} (constant) |")
        else:
            lines.append(f"| Frames/file | {fmt(mn)}–{fmt(mx)} (mean: {fmt(avg)}) |")

    lines.append(f"| MANO | 100% |")
    lines.append("")

report = "\n".join(lines) + "\n"

out_path = "reports/dataset_report.md"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(report)
print(f"Report written to {out_path}")
print(f"Total datasets: {len(all_stats)}")
print(f"Total files: {fmt(totals['files'])}")
print(f"Total unique frames: {fmt(totals['unique'])}")
print(f"Total frames (all views): {fmt(totals['total'])}")
