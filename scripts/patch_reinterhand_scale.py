#!/usr/bin/env python3
"""Patch existing reinterhand WebDataset tars to apply 3x bbox expansion.

The original converter stored `scale = tight_kp_bbox / 200` (no expansion),
while arctic and interhand26m tars store `scale = 3 * tight_kp_bbox / 200`.
This script multiplies `scale` by 3.0 in every .data.pyd inside the given tar
directory. Each tar is rewritten atomically (write .new.tar, rename).

Usage:
    python scripts/patch_reinterhand_scale.py \
        --dir ../hand_tracking_ablation/_DATA/hamer_training_data/dataset_tars/reinterhand \
        --factor 3.0
"""

import argparse
import io
import os
import pickle
import tarfile

import numpy as np
from tqdm import tqdm


def patch_tar(tar_path: str, factor: float, dry_run: bool = False) -> tuple:
    """Rewrite one tar with scale multiplied by ``factor``.

    Returns (n_samples_patched, n_members_copied).
    """
    tmp_path = tar_path + ".new"
    n_patched = 0
    n_total = 0

    with tarfile.open(tar_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
        for member in src:
            n_total += 1
            f = src.extractfile(member)
            if f is None:
                # directory or special; re-add without content
                dst.addfile(member)
                continue
            data = f.read()

            if member.name.endswith(".data.pyd"):
                ann_list = pickle.loads(data)
                for ann in ann_list:
                    ann["scale"] = np.asarray(ann["scale"], dtype=np.float64) * factor
                data = pickle.dumps(ann_list)
                n_patched += 1

            info = tarfile.TarInfo(name=member.name)
            info.size = len(data)
            info.mtime = member.mtime
            info.mode = member.mode
            info.uid = member.uid
            info.gid = member.gid
            info.uname = member.uname
            info.gname = member.gname
            dst.addfile(info, io.BytesIO(data))

    if dry_run:
        os.remove(tmp_path)
    else:
        os.replace(tmp_path, tar_path)

    return n_patched, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="Directory containing the tar shards to patch")
    parser.add_argument("--factor", type=float, default=3.0,
                        help="Multiplicative factor applied to `scale` (default: 3.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process tars but do not replace originals")
    args = parser.parse_args()

    tars = sorted(f for f in os.listdir(args.dir) if f.endswith(".tar"))
    print(f"Found {len(tars)} tars in {args.dir}")
    print(f"Factor: {args.factor}   Dry run: {args.dry_run}")

    total_patched = 0
    total_members = 0
    for name in tqdm(tars):
        p = os.path.join(args.dir, name)
        n_p, n_m = patch_tar(p, args.factor, dry_run=args.dry_run)
        total_patched += n_p
        total_members += n_m

    print(f"\nPatched {total_patched} annotations across {len(tars)} tars "
          f"({total_members} total tar members)")


if __name__ == "__main__":
    main()
