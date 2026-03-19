#!/usr/bin/env python3
"""
Generate _meta.json for existing converted HDF5 datasets.

Scans all *_label_*.hdf5 files under the given data root and writes
one _meta.json per sequence directory.

Usage:
    python generate_meta.py /path/to/CONVERTED
    python generate_meta.py /path/to/CONVERTED --datasets egodex dex_ycb
    python generate_meta.py /path/to/CONVERTED --exclude interhand26m rhd
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.io import update_sequence_meta


def main():
    parser = argparse.ArgumentParser(description="Generate _meta.json for HDF5 datasets")
    parser.add_argument("data_root", help="Root directory of converted datasets")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Only process these dataset names")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Skip these dataset names")
    args = parser.parse_args()

    dataset_names = args.datasets or sorted(os.listdir(args.data_root))
    exclude = set(args.exclude or [])

    total = 0
    for dataset_name in sorted(dataset_names):
        if dataset_name in exclude:
            print(f"Skipping {dataset_name}")
            continue
        dataset_dir = os.path.join(args.data_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for object_name in sorted(os.listdir(dataset_dir)):
            object_dir = os.path.join(dataset_dir, object_name)
            if not os.path.isdir(object_dir):
                continue
            hdf5s = sorted(glob.glob(os.path.join(object_dir, "*_label_*.hdf5")))
            if not hdf5s:
                continue
            for hdf5_path in hdf5s:
                update_sequence_meta(hdf5_path)
                total += 1
            print(f"  {dataset_name}/{object_name}: {len(hdf5s)} files")

    print(f"\nDone. Processed {total} HDF5 files.")


if __name__ == "__main__":
    main()
