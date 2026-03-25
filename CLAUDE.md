# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unified multi-dataset hand pose converter that consolidates 15+ hand pose datasets into a standardized egodex-compatible format (HDF5 annotations + MP4 videos). Outputs SE(3) transforms for 65 joints (hands + body), camera intrinsics/extrinsics, optional MANO parameters, and per-frame confidence values.

## Environment Setup

```bash
# Use the ego_pipeline conda environment
conda activate ego_pipeline

# Core dependencies: numpy, opencv-python, h5py, pyyaml
# Optional: pytubefix (WHIM YouTube download), torch + smplx/manopth (MANO forward pass)
```

## Running Converters

Each dataset has its own `scripts/convert_<dataset>.py` with `--src` and `--dst` arguments. Examples:

```bash
python scripts/convert_dex_ycb.py --src DATASET/dex_ycb --dst CONVERTED/dex_ycb --cameras 0 1 2
python scripts/convert_whim.py --src ../WiLoR --dst CONVERTED/whim_train --mode train
python scripts/convert_arctic.py --src ../arctic/downloads/data --dst CONVERTED/arctic --subjects s01 s02
python scripts/convert_interhand26m.py --src ../InterWild/data/InterHand26M --dst CONVERTED/interhand26m_train --split train --chunk-size 28
```

For SLURM-based execution, see `slurm/*.sbatch` files.

## Utility Scripts

```bash
python scripts/generate_report.py CONVERTED/          # Dataset statistics from _meta.json
python scripts/verify_datasets.py --data-root CONVERTED/ --threshold 10.0  # Data integrity
python scripts/visualize.py --src CONVERTED/dex_ycb --n 100 --out outputs  # 3D keypoint visualization
python scripts/eval_planarity.py --src CONVERTED/dex_ycb --normalize        # Joint analysis
```

## Architecture

### Converter Pattern

All converters follow the same flow:
1. Parse args (source dir, destination dir, dataset-specific filters)
2. Group sequences by clustering criterion (object, subject, capture, etc.)
3. For each sequence: load 3D keypoints → compute SE(3) transforms via `joints_to_transforms()` → add metacarpal interpolation → apply coordinate transforms
4. Write HDF5 labels + MP4 videos via `write_egodex_hdf5()` and `images_to_mp4()`
5. Generate `_meta.json` for frame indexing via `update_sequence_meta()`

### Key Utilities (`utils/`)

- **`io.py`**: HDF5 writing (`write_egodex_hdf5`), MP4 encoding (`images_to_mp4`), metadata generation (`update_sequence_meta`)
- **`transforms.py`**: Joint positions → SE(3) transforms (`joints_to_transforms_batch`), rigid body math (`invert_rigid`, `apply_transform`)
- **`joint_mapping.py`**: MANO↔egodex joint name mapping (`MANO_TO_EGODEX_SUFFIX`), metacarpal interpolation config (`METACARPAL_INTERPOLATION`), canonical 65-joint list
- **`camera_utils.py`**: Intrinsics/extrinsics helpers, active hand detection from HDF5
- **`image_utils.py`**: 3D→2D projection, bounding box computation

### HDF5 Output Schema

```
camera/intrinsic          (3,3)         # K matrix
transforms/{joint_name}   (N,4,4)       # SE(3) per frame per joint
confidences/{joint_name}  (N,)          # Per-frame validity
mano_{side}/              (optional)    # betas(10,), global_orient(N,3,3), hand_pose(N,15,3,3), transl(N,3), kpt3d(N,21,3)
```

Joint names use egodex convention: `{left|right}{Hand|Thumb*|IndexFinger*|...}` plus body joints (always zero confidence).

### Import Convention

Scripts use `sys.path.insert(0, ...)` to import from `utils/` — there is no package installation step.

## Data Directories (git-ignored)

- `RAW/` — Raw dataset symlinks
- `DATASET/` — Source datasets
- `CONVERTED/` — Output converted datasets
- `logs/` — SLURM job logs
