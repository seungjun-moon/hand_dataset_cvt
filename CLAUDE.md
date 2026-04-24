# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts 15+ hand pose datasets into a unified **egodex-compatible format** (HDF5 annotations + MP4 video). Output feeds into the HaMER hand tracking model at `../hand_tracking_ablation/`.

## Key Commands

```bash
# Install dependencies
pip install numpy opencv-python h5py pyyaml tqdm matplotlib scikit-image webdataset
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Run a converter (each script is standalone)
python scripts_hdf5/convert_dex_ycb.py --src DATASET/dex-ycb --dst CONVERTED/dex_ycb
python scripts_hdf5/convert_freihand.py --src DATASET/FreiHAND --dst CONVERTED/freihand_train

# WebDataset tar conversion (for HaMER evaluation)
python scripts/convert_freihand_webdataset.py --src ../hand_tracking_ablation/_DATA/datasets/FreiHAND_eval --dst ../hand_tracking_ablation/hamer_evaluation_data/dataset_tars/freihand-eval/

# Visualize converted data (auto-detects format: HDF5, NPZ, or WebDataset)
python scripts/visualize.py --src CONVERTED/dex_ycb --n 20 --out outputs
python scripts/visualize.py --src ../hand_tracking_ablation/hamer_evaluation_data/dataset_tars/freihand-eval --n 20

# Dataset verification and reporting
python scripts_hdf5/verify_datasets.py --src CONVERTED/dex_ycb
python scripts_hdf5/generate_report.py
```

## Architecture

### Two output formats

1. **HDF5+MP4** (`scripts_hdf5/convert_*.py`): Video sequences with per-frame 4×4 transforms for 21 hand joints + camera. Used for most datasets.
2. **WebDataset tars** (`scripts/convert_*_webdataset.py`): Single-frame pickled annotations (keypoints_2d/3d, hand_pose, betas, center, scale). Used for HaMER training/evaluation.

### Conversion pipeline (HDF5)

Source data → joint reordering to standard 21-joint order → 3D joints converted to 4×4 SE(3) transforms (`utils/transforms.py`) → frames encoded to MP4 via ffmpeg → HDF5 written via `utils/io.py:write_egodex_hdf5()`.

### Standard 21-joint order (egodex/MANO)

`0:Wrist, 1-4:Thumb, 5-8:Index, 9-12:Middle, 13-16:Ring, 17-20:Little` (each finger: MCP→PIP→DIP→Tip)

Parent chain: all MCP joints attach to wrist (joint 0).

### MANO FK output order (raw from model)

`0:Wrist, 1-3:Index, 4-6:Middle, 7-9:Pinky, 10-12:Ring, 13-15:Thumb` (16 joints) + 5 fingertip vertices (16-20).

Reorder mapping (FK → standard): `[0, 13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20]`. Used by HO-3D, ARCTIC, H2O-3D converters.

## Critical Conventions

### flat_hand_mean

- **FreiHAND** stores hand_pose as offsets from natural grasping mean (`flat_hand_mean=False`)
- **HaMER** expects absolute rotations from flat hand (`flat_hand_mean=True`)
- When converting FreiHAND → WebDataset, **add** the MANO mean hand pose to `hand_pose[3:48]`

### WebDataset center/scale

HaMER convention: `bbox_size_px = scale * 200`. The bbox is always square (we
take `max(scale_x, scale_y)` on read). **The stored `scale` must embed a 3×
expansion over the tight 2D keypoint bbox** so all datasets feed a HaMER
dataloader consistently. Converter pattern:

```python
sq_bbox = expand_to_square(bbox_from_keypoints(kp2d))
bbox_size = (sq_bbox[2] - sq_bbox[0]) * 3.0   # <- 3× expansion
scale = np.array([bbox_size / 200.0, bbox_size / 200.0])
```

Empirical check: `max(scale) * 200 / max(tight_kp_bbox)` should equal ~3.0 across
samples. Confirmed for `arctic-train`, `interhand26m-train`, `reinterhand`
(patched 2026-04-24 — see `scripts/patch_reinterhand_scale.py`; the original
converter used 1× and produced overly tight crops).

Exceptions:
- **FreiHAND**: images are already 224×224 hand crops. Use fixed `center=[112,112]`,
  `scale=[1.12,1.12]` (= 224/200), not computed from keypoints.

Visualize the stored bbox with:
```bash
python scripts/visualize.py --src <tar_dir> --crop --n 12 --out outputs/bbox_crop/<name>
```
Hand should sit centered with roughly equal padding on all sides across datasets.

### Units

All 3D coordinates are in **meters**. Camera intrinsics use pixel units.

## Mandatory: Verify New Datasets with Visualization

After converting any new dataset (HDF5, WebDataset, or ClipDataset), **always** run the visualizer to verify MANO mesh alignment before considering the conversion done:

```bash
# ClipDataset format (requires --img-dir)
python scripts/visualize.py --src <clip_label_dir> --img-dir <image_root> --n 20 --out outputs

# WebDataset or HDF5 format
python scripts/visualize.py --src <converted_dir> --n 20 --out outputs
```

Check the output images: the MANO mesh (panel 4) and projected 3D keypoints (panel 2) should align with the GT 2D keypoints (panel 1). The error panel (panel 3) should show <10px mean error. If MANO mesh is misaligned, investigate `hand_tsl`, `cTw`, and `flat_hand_mean` conventions for that dataset.

### ClipDataset image root paths (--img-dir)

- arctic: `../hand_tracking_ablation/_DATA/haptic_training_images/arctic/images`
- ho3d: `../hand_tracking_ablation/_DATA/haptic_training_images/ho3d/HO3D_v3`
- ho2o: `../hand_tracking_ablation/_DATA/haptic_training_images/ho2o/raw`
- dexycb: `../hand_tracking_ablation/_DATA/haptic_training_images/dexycb`

## Key Files

- `utils/io.py` — `write_egodex_hdf5()`, video encoding via ffmpeg
- `utils/transforms.py` — 3D joints → 4×4 SE(3) transforms
- `utils/joint_mapping.py` — MANO ↔ egodex joint name/index mapping, `MANO_PARENTS`
- `utils/image_utils.py` — 3D→2D projection, HaMER-convention cropping (rescale=2.5, 384×384)
- `utils/camera_utils.py` — Active hand side detection, per-frame intrinsics
- `scripts/visualize.py` — Multi-format visualization with skeleton overlay + MANO mesh
- `_DATA/data/mano/` — MANO model files (MANO_RIGHT.pkl, MANO_LEFT.pkl)
