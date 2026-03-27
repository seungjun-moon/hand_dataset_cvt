#!/usr/bin/env python3
"""
Patch existing HDF5 files in-place: convert mm -> meters for InterHand2.6M
and ReInterHand datasets.

Scales the translation column (:3, 3) of every 4x4 transform matrix by 1/1000,
and scales MANO transl_worldspace and kpt3d by 1/1000.

Rotation components and intrinsics are left untouched.

Usage:
    python scripts/patch_mm_to_meters.py --datasets interhand26m_train interhand26m_test reinterhand
    python scripts/patch_mm_to_meters.py --data-root CONVERTED --dry-run
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def patch_hdf5(path: str, dry_run: bool = False) -> dict:
    """Patch one HDF5 file in-place. Returns stats dict."""
    stats = {'transforms': 0, 'mano_transl': 0, 'mano_kpt3d': 0}

    mode = 'r' if dry_run else 'r+'
    try:
        f = h5py.File(path, mode)
    except Exception as e:
        return {'error': str(e)}

    try:
        # 1) Scale all transform translation columns: [:, :3, 3] /= 1000
        if 'transforms' in f:
            for key in f['transforms']:
                ds = f[f'transforms/{key}']
                data = ds[:]  # (N, 4, 4)
                if data.ndim == 3 and data.shape[1:] == (4, 4):
                    data[:, :3, 3] /= 1000.0
                    if not dry_run:
                        ds[...] = data
                    stats['transforms'] += 1
                elif data.ndim == 2 and data.shape == (4, 4):
                    # Single matrix (unlikely but handle it)
                    data[:3, 3] /= 1000.0
                    if not dry_run:
                        ds[...] = data
                    stats['transforms'] += 1

        # 2) Scale MANO groups: transl_worldspace and kpt3d
        for key in list(f.keys()):
            if not key.startswith('mano_'):
                continue
            grp = f[key]
            if 'transl_worldspace' in grp:
                data = grp['transl_worldspace'][:]
                data /= 1000.0
                if not dry_run:
                    grp['transl_worldspace'][...] = data
                stats['mano_transl'] += 1
            if 'kpt3d' in grp:
                data = grp['kpt3d'][:]
                data /= 1000.0
                if not dry_run:
                    grp['kpt3d'][...] = data
                stats['mano_kpt3d'] += 1

    finally:
        f.close()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Patch HDF5 files: mm -> meters for InterHand/ReInterHand')
    parser.add_argument('--data-root', default='CONVERTED')
    parser.add_argument('--datasets', nargs='+',
                        default=['interhand26m_train', 'interhand26m_test', 'reinterhand'])
    parser.add_argument('--dry-run', action='store_true',
                        help='Read-only pass, report what would change')
    parser.add_argument('--verify-sample', action='store_true',
                        help='After patching, verify a sample file')
    args = parser.parse_args()

    all_hdf5 = []
    for ds in args.datasets:
        ds_dir = os.path.join(args.data_root, ds)
        if not os.path.isdir(ds_dir):
            print(f'WARNING: {ds_dir} not found, skipping')
            continue
        files = sorted(glob.glob(os.path.join(ds_dir, '**', '*_label_*.hdf5'), recursive=True))
        print(f'{ds}: {len(files)} HDF5 files')
        all_hdf5.extend(files)

    total = len(all_hdf5)
    print(f'\nTotal: {total} files to patch {"(DRY RUN)" if args.dry_run else ""}')

    total_stats = {'transforms': 0, 'mano_transl': 0, 'mano_kpt3d': 0, 'errors': 0}

    for i, path in enumerate(all_hdf5):
        stats = patch_hdf5(path, dry_run=args.dry_run)
        if 'error' in stats:
            print(f'  ERROR: {path}: {stats["error"]}')
            total_stats['errors'] += 1
        else:
            total_stats['transforms'] += stats['transforms']
            total_stats['mano_transl'] += stats['mano_transl']
            total_stats['mano_kpt3d'] += stats['mano_kpt3d']

        if (i + 1) % 500 == 0 or i + 1 == total:
            print(f'  [{i+1}/{total}] patched ...', flush=True)

    print(f'\nDone. Patched:')
    print(f'  Transform datasets scaled: {total_stats["transforms"]}')
    print(f'  MANO transl scaled: {total_stats["mano_transl"]}')
    print(f'  MANO kpt3d scaled: {total_stats["mano_kpt3d"]}')
    if total_stats['errors']:
        print(f'  Errors: {total_stats["errors"]}')

    # Quick verification
    if args.verify_sample and all_hdf5 and not args.dry_run:
        sample = all_hdf5[0]
        print(f'\nVerification sample: {sample}')
        with h5py.File(sample, 'r') as f:
            cam = f['transforms/camera'][0]
            print(f'  cam translation: {cam[:3, 3]}  (magnitude: {np.linalg.norm(cam[:3, 3]):.3f} m)')
            for key in f['transforms']:
                if 'Hand' in key:
                    # Find a frame with nonzero translation
                    data = f[f'transforms/{key}'][:]
                    norms = np.linalg.norm(data[:, :3, 3], axis=1)
                    active = np.where(norms > 0.001)[0]
                    if len(active) > 0:
                        fi = active[0]
                        print(f'  {key} frame {fi}: {data[fi, :3, 3]}  '
                              f'(magnitude: {norms[fi]:.3f} m)')
                    break


if __name__ == '__main__':
    main()
