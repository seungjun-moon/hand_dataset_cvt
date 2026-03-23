"""
Verify all converted datasets for values that would cause NaN loss or assertion
errors in the HDF5Dataset / NPZDataset dataloaders.

Checks:
  1. NaN / Inf in 3D keypoints, camera extrinsics, intrinsics
  2. Extremely large 3D coordinates (> threshold)
  3. Joints behind camera (Z <= 0 after world->cam)
  4. Singular camera extrinsics (non-invertible)
  5. Zero or negative focal length
  6. 3D->2D reprojection landing far outside image bounds
  7. Degenerate bounding box (all keypoints collapsed to a point)

Usage:
    python scripts/verify_datasets.py [--data-root CONVERTED/] [--threshold 10.0]
                                       [--datasets d1 d2 ...] [--workers N]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

# Same joint names as the dataloader
_JOINT_NAMES = {
    'right': [
        'rightHand',
        'rightThumbKnuckle', 'rightThumbIntermediateBase', 'rightThumbIntermediateTip', 'rightThumbTip',
        'rightIndexFingerKnuckle', 'rightIndexFingerIntermediateBase', 'rightIndexFingerIntermediateTip', 'rightIndexFingerTip',
        'rightMiddleFingerKnuckle', 'rightMiddleFingerIntermediateBase', 'rightMiddleFingerIntermediateTip', 'rightMiddleFingerTip',
        'rightRingFingerKnuckle', 'rightRingFingerIntermediateBase', 'rightRingFingerIntermediateTip', 'rightRingFingerTip',
        'rightLittleFingerKnuckle', 'rightLittleFingerIntermediateBase', 'rightLittleFingerIntermediateTip', 'rightLittleFingerTip',
    ],
    'left': [
        'leftHand',
        'leftThumbKnuckle', 'leftThumbIntermediateBase', 'leftThumbIntermediateTip', 'leftThumbTip',
        'leftIndexFingerKnuckle', 'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 'leftIndexFingerTip',
        'leftMiddleFingerKnuckle', 'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip', 'leftMiddleFingerTip',
        'leftRingFingerKnuckle', 'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip', 'leftRingFingerTip',
        'leftLittleFingerKnuckle', 'leftLittleFingerIntermediateBase', 'leftLittleFingerIntermediateTip', 'leftLittleFingerTip',
    ],
}

# ──────────────────────────────────────────────────────────────────────
#  HDF5 verification
# ──────────────────────────────────────────────────────────────────────
def verify_hdf5(label_path: str, coord_thresh: float) -> list:
    """Return a list of issue dicts for one HDF5 file."""
    issues = []

    def _issue(msg, frame=None, side=None):
        d = {'file': label_path, 'msg': msg}
        if frame is not None:
            d['frame'] = int(frame)
        if side is not None:
            d['side'] = side
        issues.append(d)

    try:
        f = h5py.File(label_path, 'r')
    except Exception as e:
        _issue(f'cannot open HDF5: {e}')
        return issues

    try:
        # --- intrinsics ---
        if 'camera/intrinsic' not in f:
            _issue('missing camera/intrinsic')
            return issues
        K = f['camera/intrinsic'][:].astype(np.float64)
        if not np.all(np.isfinite(K)):
            _issue(f'NaN/Inf in intrinsic: {K}')
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            _issue(f'non-positive focal length: fx={K[0,0]}, fy={K[1,1]}')

        # --- camera extrinsics ---
        if 'transforms/camera' not in f:
            _issue('missing transforms/camera')
            return issues
        cam_exts = f['transforms/camera'][:].astype(np.float64)  # (N, 4, 4)
        n_frames = cam_exts.shape[0]

        # Check for NaN/Inf in extrinsics
        bad_ext_mask = ~np.all(np.isfinite(cam_exts.reshape(n_frames, -1)), axis=1)
        for fi in np.where(bad_ext_mask)[0]:
            _issue('NaN/Inf in camera extrinsic', frame=fi)

        # Check for singular extrinsics (det ≈ 0)
        dets = np.linalg.det(cam_exts)
        bad_det_mask = np.abs(dets) < 1e-8
        for fi in np.where(bad_det_mask & ~bad_ext_mask)[0]:
            _issue(f'singular camera extrinsic (det={dets[fi]:.2e})', frame=fi)

        # --- per-side checks ---
        # Detect which sides are present
        sides = []
        for key in f.keys():
            if key.startswith('mano_'):
                sides.append(key[len('mano_'):])
        if not sides:
            # fallback: check transforms for joint names
            for s in ('right', 'left'):
                if f'transforms/{_JOINT_NAMES[s][0]}' in f:
                    sides.append(s)

        for side in sides:
            joint_names = _JOINT_NAMES.get(side)
            if joint_names is None:
                continue

            # Check that all joint transforms exist
            missing = [j for j in joint_names if f'transforms/{j}' not in f]
            if missing:
                _issue(f'missing joint transforms: {missing}', side=side)
                continue

            # Read all joint positions: (N, 21, 3)
            try:
                kpt3d_world = np.stack(
                    [f[f'transforms/{j}'][:, :3, 3] for j in joint_names],
                    axis=1,
                ).astype(np.float64)
            except Exception as e:
                _issue(f'error reading joint transforms: {e}', side=side)
                continue

            n_jframes = kpt3d_world.shape[0]

            # 1) NaN/Inf in world coords
            nan_mask = ~np.all(np.isfinite(kpt3d_world.reshape(n_jframes, -1)), axis=1)
            nan_frames = np.where(nan_mask)[0]
            if len(nan_frames) > 0:
                _issue(f'NaN/Inf in kpt3d_world: {len(nan_frames)} frames '
                       f'(first: {nan_frames[0]})', side=side)

            # 2) Extremely large world coords
            abs_max_per_frame = np.abs(kpt3d_world).reshape(n_jframes, -1).max(axis=1)
            large_mask = abs_max_per_frame > coord_thresh
            large_frames = np.where(large_mask & ~nan_mask)[0]
            if len(large_frames) > 0:
                worst_fi = large_frames[np.argmax(abs_max_per_frame[large_frames])]
                worst_val = abs_max_per_frame[worst_fi]
                _issue(f'large kpt3d_world: {len(large_frames)} frames > {coord_thresh}m '
                       f'(worst frame={worst_fi}, max_coord={worst_val:.2f})',
                       side=side)

            # 3) World->camera transform: check Z > 0 and finite
            good_ext = ~bad_ext_mask[:n_jframes] & ~nan_mask[:n_jframes]
            good_frames = np.where(good_ext)[0]

            if len(good_frames) > 0:
                # Batch transform: inv(cam_ext) @ kpt3d_world_homo
                # Sample up to 500 frames to keep memory/time bounded
                check_frames = good_frames
                if len(check_frames) > 500:
                    check_frames = np.random.default_rng(42).choice(
                        check_frames, 500, replace=False)
                    check_frames.sort()

                cam_inv = np.linalg.inv(cam_exts[check_frames])  # (M, 4, 4)
                kpt_h = np.concatenate(
                    [kpt3d_world[check_frames],
                     np.ones((*kpt3d_world[check_frames].shape[:2], 1))],
                    axis=-1,
                )  # (M, 21, 4)
                # (M, 4, 4) @ (M, 4, 21) -> (M, 4, 21) -> (M, 21, 4)
                kpt3d_cam = np.einsum('mij,mkj->mki', cam_inv, kpt_h)[..., :3]

                # NaN after transform
                cam_nan_mask = ~np.all(np.isfinite(kpt3d_cam.reshape(len(check_frames), -1)), axis=1)
                cam_nan_count = cam_nan_mask.sum()
                if cam_nan_count > 0:
                    fi_example = check_frames[np.where(cam_nan_mask)[0][0]]
                    _issue(f'NaN/Inf in kpt3d_cam after world->cam: '
                           f'{cam_nan_count}/{len(check_frames)} sampled frames '
                           f'(example frame={fi_example})', side=side)

                # Z <= 0 (joints behind camera)
                z_vals = kpt3d_cam[..., 2]  # (M, 21)
                behind_mask = np.any(z_vals <= 0, axis=1) & ~cam_nan_mask
                behind_count = behind_mask.sum()
                if behind_count > 0:
                    fi_example = check_frames[np.where(behind_mask)[0][0]]
                    min_z = z_vals[behind_mask].min()
                    _issue(f'joints behind camera (Z<=0): '
                           f'{behind_count}/{len(check_frames)} sampled frames '
                           f'(example frame={fi_example}, min_Z={min_z:.4f})',
                           side=side)

                # Large camera-space coords
                cam_abs_max = np.abs(kpt3d_cam).reshape(len(check_frames), -1).max(axis=1)
                cam_large = cam_abs_max > coord_thresh
                cam_large_count = (cam_large & ~cam_nan_mask).sum()
                if cam_large_count > 0:
                    fi_idx = np.where(cam_large & ~cam_nan_mask)[0]
                    worst_idx = fi_idx[np.argmax(cam_abs_max[fi_idx])]
                    _issue(f'large kpt3d_cam: {cam_large_count}/{len(check_frames)} sampled '
                           f'frames > {coord_thresh}m (worst frame={check_frames[worst_idx]}, '
                           f'max={cam_abs_max[worst_idx]:.2f})', side=side)

                # 2D projection out of image bounds
                valid_cam = ~cam_nan_mask & ~behind_mask
                if valid_cam.any():
                    kpt_valid = kpt3d_cam[valid_cam]  # (V, 21, 3)
                    K64 = K.astype(np.float64)
                    proj_h = np.einsum('ij,vkj->vki', K64, kpt_valid)  # (V, 21, 3)
                    proj_2d = proj_h[..., :2] / proj_h[..., 2:3]  # (V, 21, 2)

                    # Check if far outside image (>2x image dims)
                    # We don't know image size from HDF5 alone, use cx/cy as proxy
                    img_w_est = 2 * K[0, 2]
                    img_h_est = 2 * K[1, 2]
                    oob_x = (proj_2d[..., 0] < -img_w_est) | (proj_2d[..., 0] > 2 * img_w_est)
                    oob_y = (proj_2d[..., 1] < -img_h_est) | (proj_2d[..., 1] > 2 * img_h_est)
                    oob_any = np.any(oob_x | oob_y, axis=1)
                    if oob_any.sum() > 0:
                        valid_indices = check_frames[valid_cam]
                        fi_example = valid_indices[np.where(oob_any)[0][0]]
                        _issue(f'2D projection far outside image: '
                               f'{oob_any.sum()}/{valid_cam.sum()} frames '
                               f'(example frame={fi_example})', side=side)

                # Degenerate bbox (all keypoints at same pixel)
                if valid_cam.any():
                    kp_range = proj_2d.max(axis=1) - proj_2d.min(axis=1)  # (V, 2)
                    degen = np.any(kp_range < 1.0, axis=1)
                    if degen.sum() > 0:
                        valid_indices = check_frames[valid_cam]
                        fi_example = valid_indices[np.where(degen)[0][0]]
                        _issue(f'degenerate bbox (keypoints collapsed): '
                               f'{degen.sum()}/{valid_cam.sum()} frames '
                               f'(example frame={fi_example})', side=side)

        # --- confidence checks ---
        if 'confidences' in f:
            for side in sides:
                joint_names = _JOINT_NAMES.get(side)
                if joint_names is None:
                    continue
                for j in joint_names:
                    key = f'confidences/{j}'
                    if key in f:
                        conf = f[key][:]
                        if not np.all(np.isfinite(conf)):
                            _issue(f'NaN/Inf in {key}', side=side)

    except Exception as e:
        _issue(f'unexpected error: {e}')
    finally:
        f.close()

    return issues


# ──────────────────────────────────────────────────────────────────────
#  NPZ verification
# ──────────────────────────────────────────────────────────────────────
def verify_npz(npz_path: str, coord_thresh: float) -> list:
    """Return a list of issue dicts for one NPZ file."""
    issues = []

    def _issue(msg):
        issues.append({'file': npz_path, 'msg': msg})

    try:
        data = dict(np.load(npz_path, allow_pickle=True))
    except Exception as e:
        _issue(f'cannot load NPZ: {e}')
        return issues

    # Required keys
    for key in ('intrinsic', 'cam_ext', 'kpt3d_world'):
        if key not in data:
            _issue(f'missing key: {key}')
            return issues

    K = data['intrinsic'].astype(np.float64)
    cam_ext = data['cam_ext'].astype(np.float64)
    kpt3d_world = data['kpt3d_world'].astype(np.float64)  # (21, 3)

    # Intrinsic checks
    if not np.all(np.isfinite(K)):
        _issue(f'NaN/Inf in intrinsic')
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        _issue(f'non-positive focal length: fx={K[0,0]}, fy={K[1,1]}')

    # Extrinsic checks
    if not np.all(np.isfinite(cam_ext)):
        _issue('NaN/Inf in cam_ext')
        return issues
    det = np.linalg.det(cam_ext)
    if abs(det) < 1e-8:
        _issue(f'singular cam_ext (det={det:.2e})')
        return issues

    # World coords
    if not np.all(np.isfinite(kpt3d_world)):
        _issue('NaN/Inf in kpt3d_world')
        return issues

    abs_max = np.abs(kpt3d_world).max()
    if abs_max > coord_thresh:
        _issue(f'large kpt3d_world: max_coord={abs_max:.2f} > {coord_thresh}')

    # Camera-space
    kpt_h = np.hstack([kpt3d_world, np.ones((21, 1))])
    cam_inv = np.linalg.inv(cam_ext)
    kpt3d_cam = (cam_inv @ kpt_h.T).T[:, :3]

    if not np.all(np.isfinite(kpt3d_cam)):
        _issue('NaN/Inf in kpt3d_cam')
        return issues

    if np.any(kpt3d_cam[:, 2] <= 0):
        _issue(f'joints behind camera: min_Z={kpt3d_cam[:, 2].min():.4f}')
        return issues

    cam_abs_max = np.abs(kpt3d_cam).max()
    if cam_abs_max > coord_thresh:
        _issue(f'large kpt3d_cam: max_coord={cam_abs_max:.2f} > {coord_thresh}')

    # 2D projection
    proj_h = (K @ kpt3d_cam.T).T
    proj_2d = proj_h[:, :2] / proj_h[:, 2:3]
    img_w_est = 2 * K[0, 2]
    img_h_est = 2 * K[1, 2]
    if np.any(proj_2d[:, 0] < -img_w_est) or np.any(proj_2d[:, 0] > 2 * img_w_est) or \
       np.any(proj_2d[:, 1] < -img_h_est) or np.any(proj_2d[:, 1] > 2 * img_h_est):
        _issue(f'2D projection far outside image bounds')

    # Degenerate bbox
    kp_range = proj_2d.max(axis=0) - proj_2d.min(axis=0)
    if np.any(kp_range < 1.0):
        _issue(f'degenerate bbox: kp range = {kp_range}')

    # MANO params if present
    for key in ('mano_betas', 'mano_global_orient', 'mano_hand_pose', 'mano_transl'):
        if key in data:
            v = data[key]
            if not np.all(np.isfinite(v)):
                _issue(f'NaN/Inf in {key}')
            if np.abs(v).max() > 1000:
                _issue(f'extreme value in {key}: max_abs={np.abs(v).max():.2f}')

    return issues


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def collect_files(data_root, dataset_names):
    """Collect all HDF5 and NPZ files to verify."""
    hdf5_files = []
    npz_files = []

    if dataset_names is None:
        dataset_names = sorted(os.listdir(data_root))

    for ds in sorted(dataset_names):
        ds_dir = os.path.join(data_root, ds)
        if not os.path.isdir(ds_dir):
            print(f'WARNING: dataset dir not found: {ds_dir}')
            continue
        h5 = sorted(glob.glob(os.path.join(ds_dir, '**', '*_label_*.hdf5'), recursive=True))
        nz = sorted(glob.glob(os.path.join(ds_dir, '**', '*.npz'), recursive=True))
        hdf5_files.extend(h5)
        npz_files.extend(nz)
        print(f'  {ds}: {len(h5)} HDF5, {len(nz)} NPZ')

    return hdf5_files, npz_files


def main():
    parser = argparse.ArgumentParser(description='Verify converted hand datasets')
    parser.add_argument('--data-root', default='CONVERTED',
                        help='Root dir of converted datasets')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Coordinate magnitude threshold (meters)')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Specific dataset names to check')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers')
    parser.add_argument('--output', default='verify_report.json',
                        help='Output JSON report path')
    args = parser.parse_args()

    print(f'=== Dataset Verification ===')
    print(f'  data_root:  {args.data_root}')
    print(f'  threshold:  {args.threshold} m')
    print(f'  workers:    {args.workers}')
    print()

    hdf5_files, npz_files = collect_files(args.data_root, args.datasets)
    print(f'\nTotal: {len(hdf5_files)} HDF5 + {len(npz_files)} NPZ = '
          f'{len(hdf5_files) + len(npz_files)} files\n')

    all_issues = []
    total = len(hdf5_files) + len(npz_files)
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for fp in hdf5_files:
            futures[pool.submit(verify_hdf5, fp, args.threshold)] = fp
        for fp in npz_files:
            futures[pool.submit(verify_npz, fp, args.threshold)] = fp

        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == total:
                print(f'  [{done}/{total}] checked ...', flush=True)
            try:
                file_issues = fut.result()
                all_issues.extend(file_issues)
            except Exception as e:
                all_issues.append({'file': futures[fut], 'msg': f'worker error: {e}'})

    # ---- Summary ----
    print(f'\n=== Results ===')
    print(f'Total issues: {len(all_issues)}')

    if not all_issues:
        print('All files passed verification!')
        return

    # Group by dataset
    by_dataset = defaultdict(list)
    for iss in all_issues:
        # Extract dataset name from path
        rel = os.path.relpath(iss['file'], args.data_root)
        ds = rel.split(os.sep)[0]
        by_dataset[ds].append(iss)

    for ds in sorted(by_dataset):
        issues = by_dataset[ds]
        print(f'\n--- {ds} ({len(issues)} issues) ---')
        # Show up to 10 per dataset
        for iss in issues[:10]:
            loc = os.path.relpath(iss['file'], args.data_root)
            extra = ''
            if 'frame' in iss:
                extra += f' frame={iss["frame"]}'
            if 'side' in iss:
                extra += f' side={iss["side"]}'
            print(f'  {loc}{extra}: {iss["msg"]}')
        if len(issues) > 10:
            print(f'  ... and {len(issues) - 10} more')

    # Save full report
    with open(args.output, 'w') as f:
        json.dump(all_issues, f, indent=2, default=str)
    print(f'\nFull report saved to: {args.output}')


if __name__ == '__main__':
    main()
