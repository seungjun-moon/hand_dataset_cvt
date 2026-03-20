# Hand Dataset Converter

Convert hand pose datasets into a unified egodex-compatible format with HDF5 annotations and MP4 videos.

Supported datasets:
- [DexYCB](https://dex-ycb.github.io/)
- [HO-Cap](https://irvlutd.github.io/HOCap/)
- [EgoDex](https://github.com/facebookresearch/ego-dex)
- [WHIM](https://rolpotamern.github.io/WiLoR/) (in-the-wild YouTube hand data)
- [ARCTIC](https://arctic.is.tue.mpg.de/) (bimanual articulated object manipulation)
- [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) (multi-view interacting hands)
- [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) (real hand images with MANO)
- [RHD](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) (rendered synthetic hand poses)
- [HIC](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/) (hand-in-contact interaction)
- [H2O-3D](https://www.taeinkwon.com/projects/h2o) (egocentric hand-object interaction)
- [MTC](http://domedb.perception.cs.cmu.edu/handdb.html) (Panoptic Studio multi-view hand capture)
- [ReInterHand](https://mks0601.github.io/ReInterHand/) (re-annotated interacting hands)

## Converted Dataset Structure

```
CONVERTED/
├── dex_ycb/
│   └── {object_name}/                  # Clustered by grasped YCB object
│       ├── {seq_idx:06d}_label_{cam_idx:02d}.hdf5
│       └── {seq_idx:06d}_video_{cam_idx:02d}.mp4
├── ho_cap/
│   └── {subject_id}/                   # Clustered by subject (subject_1 .. subject_9)
│       ├── {seq_idx:06d}_label_{cam_idx:02d}.hdf5
│       └── {seq_idx:06d}_video_{cam_idx:02d}.mp4
├── egodex/
│   └── {task_name}/                    # Clustered by task
│       ├── {seq_idx:06d}_label_00.hdf5
│       └── {seq_idx:06d}_video_00.mp4
├── whim_train/
│   └── train/                          # One sequence per YouTube video
│       ├── {seq_idx:06d}_label_00.hdf5
│       └── {seq_idx:06d}_video_00.mp4
├── whim_test/
│   └── ...
├── arctic/
│   └── {subject_id}/                   # Clustered by subject
│       ├── {seq_idx:06d}_label_{cam_idx:02d}.hdf5
│       └── {seq_idx:06d}_video_{cam_idx:02d}.mp4
├── interhand26m_train/
│   └── capture_{id}_chunk_{id}/        # Chunked by capture
│       ├── {seq_idx:06d}_label_00.hdf5
│       └── {seq_idx:06d}_video_00.mp4
├── interhand26m_test/
│   └── ...
├── freihand_train/                     # NPZ format (single-frame)
│   └── {cluster}/
│       └── {seq_idx:06d}_{view_idx:02d}.npz
├── freihand_eval/
│   └── ...
├── rhd/
│   └── {cluster}/
│       ├── {seq_idx:06d}_label_{cam_idx:02d}.hdf5
│       └── {seq_idx:06d}_video_{cam_idx:02d}.mp4
├── hic/
│   └── ...
├── h2o3d/
│   └── {object_name}/
│       ├── {seq_idx:06d}_label_00.hdf5
│       └── {seq_idx:06d}_video_00.mp4
├── mtc_train/
│   └── {seqName}_id{id}/              # Multi-view Panoptic Studio
│       ├── {seq_idx:06d}_label_{cam_idx:02d}.hdf5
│       └── {seq_idx:06d}_video_{cam_idx:02d}.mp4
└── reinterhand/
    └── {capture_chunk}/
        ├── {seq_idx:06d}_label_00.hdf5
        └── {seq_idx:06d}_video_00.mp4
```

Multi-view is supported via the camera index suffix (`_00`, `_01`, ...). DexYCB and HO-Cap have 8+ cameras per sequence; EgoDex and WHIM are single-camera.

## HDF5 Structure

Each `_label_XX.hdf5` file contains:

```
├── camera/
│   └── intrinsic                          (3, 3) float32      # [fx,0,cx; 0,fy,cy; 0,0,1]
├── transforms/
│   ├── camera                             (N, 4, 4) float32   # Camera-to-world pose (identity for WHIM)
│   ├── {side}Hand                         (N, 4, 4) float32   # Wrist
│   ├── {side}ThumbKnuckle                 (N, 4, 4) float32
│   ├── {side}ThumbIntermediateBase        (N, 4, 4) float32
│   ├── {side}ThumbIntermediateTip         (N, 4, 4) float32
│   ├── {side}ThumbTip                     (N, 4, 4) float32
│   ├── {side}IndexFingerKnuckle           (N, 4, 4) float32
│   ├── {side}IndexFingerIntermediateBase  (N, 4, 4) float32
│   ├── {side}IndexFingerIntermediateTip   (N, 4, 4) float32
│   ├── {side}IndexFingerTip              (N, 4, 4) float32
│   ├── {side}MiddleFinger...              ...                  # Same for Middle, Ring, Little
│   ├── {side}IndexFingerMetacarpal        (N, 4, 4) float32   # Interpolated (wrist-MCP, alpha=0.3)
│   ├── {side}MiddleFingerMetacarpal       (N, 4, 4) float32
│   ├── {side}RingFingerMetacarpal         (N, 4, 4) float32
│   ├── {side}LittleFingerMetacarpal       (N, 4, 4) float32
│   ├── hip                                (N, 4, 4) float32   # Body joints (identity, zero confidence)
│   ├── leftShoulder, leftArm, ...         ...
│   └── spine1..spine7, neck1..neck4       ...
├── confidences/
│   ├── {side}Hand                         (N,) float32        # 1.0 if active, 0.0 if inactive
│   └── ...                                                     # Same keys as transforms/ (except camera)
└── mano_{side}/                                                # Optional, when MANO data available
    ├── betas                              (10,) float32        # Shape parameters
    ├── global_orient_worldspace           (N, 3, 3) float32    # Global rotation matrices
    ├── hand_pose                          (N, 15, 3, 3) float32 # Per-joint rotation matrices
    ├── transl_worldspace                  (N, 3) float32       # Translation (world space)
    └── kpt3d                              (N, 21, 3) float32   # 3D keypoints (world space)
```

Where:
- `N` = number of valid frames
- `{side}` = `left` or `right`
- Transforms are 4x4 homogeneous matrices `[R|t; 0 0 0 1]` in world space
- Translation column `[:3, 3]` gives the 3D joint position
- Inactive hands have identity transforms and zero confidence
- Body joints are always present with zero confidence

## Setup

```bash
pip install numpy opencv-python h5py pyyaml
pip install pytubefix  # Required for WHIM (YouTube download)
```

## Usage

### DexYCB

```bash
python scripts/convert_dex_ycb.py --src DATASET/dex_ycb --dst CONVERTED/dex_ycb
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `DATASET/dex_ycb` | Source directory |
| `--dst` | `CONVERTED/dex_ycb` | Output directory |
| `--cameras` | all | Camera indices to extract |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max sequences (0=all) |

### HO-Cap

```bash
python scripts/convert_ho_cap.py --src ../HO-Cap/datasets --dst CONVERTED/ho_cap
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../HO-Cap/datasets` | Source directory |
| `--dst` | `CONVERTED/ho_cap` | Output directory |
| `--cameras` | all | Camera indices (8 RealSense cameras) |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max sequences (0=all) |

### EgoDex

```bash
python scripts/convert_egodex.py --src RAW/egodex --dst CONVERTED/egodex
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `RAW/egodex` | Source directory |
| `--dst` | `CONVERTED/egodex` | Output directory |
| `--max-samples` | `0` | Max sequences (0=all) |

### WHIM

Downloads YouTube videos and converts with pre-downloaded annotations. Tracks completed videos in JSON files for safe resuming.

```bash
python scripts/convert_whim.py --src ../WiLoR --dst CONVERTED/whim --mode train
python scripts/convert_whim.py --src ../WiLoR --dst CONVERTED/whim --mode test
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../WiLoR` | WiLoR root directory |
| `--dst` | `CONVERTED/whim` | Output directory |
| `--mode` | `train` | `train` or `test` split |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max videos (0=all) |

Multi-hand policy: keeps at most one hand per side (left/right) per frame, selecting the largest bounding box when duplicates exist.

### ARCTIC

```bash
python scripts/convert_arctic.py --src ../arctic/downloads/data --dst CONVERTED/arctic
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../arctic/downloads/data` | ARCTIC data directory |
| `--dst` | `CONVERTED/arctic` | Output directory |
| `--cameras` | all | Camera indices (0=ego, 1-8=allocentric) |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max sequences (0=all) |
| `--mano-model-dir` | `None` | Path to directory containing `mano/` subdir |
| `--subjects` | all | Filter to specific subjects (e.g. `s01 s02`) |

### InterHand2.6M

```bash
python scripts/convert_interhand26m.py --src ../InterWild/data/InterHand26M --dst CONVERTED/interhand26m_train --split train
python scripts/convert_interhand26m.py --src ../InterWild/data/InterHand26M --dst CONVERTED/interhand26m_test --split test
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../InterWild/data/InterHand26M` | InterHand2.6M dataset directory |
| `--dst` | `CONVERTED/interhand26m` | Output directory |
| `--split` | `train` | Data split (`train`/`val`/`test`) |
| `--chunk-size` | `28` | Sequences per chunk |
| `--fps` | `5.0` | Video FPS (matches 5fps subset) |
| `--max-samples` | `0` | Max chunks (0=all) |

### H2O-3D

```bash
python scripts/convert_h2o3d.py --src ../ho3d/data/h2o3d --dst CONVERTED/h2o3d
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../ho3d/data/h2o3d` | H2O-3D dataset directory |
| `--dst` | `CONVERTED/h2o3d` | Output directory |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max sequences (0=all) |

### MTC (Panoptic Studio)

```bash
python scripts/convert_mtc.py --src ../mtc_dataset/mtc_video_dataset --dst CONVERTED/mtc_train --split training
python scripts/convert_mtc.py --src ../mtc_dataset/mtc_video_dataset --dst CONVERTED/mtc_eval --split testing
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../mtc_dataset/mtc_video_dataset` | MTC video dataset directory |
| `--dst` | `CONVERTED/mtc_train` | Output directory |
| `--split` | `training` | Which split (`training`/`testing`) |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max clips (0=all) |
| `--seq-filter` | `None` | Filter to specific sequence name |

### ReInterHand

```bash
python scripts/convert_reinterhand.py --src ../InterWild/tool/ReInterHand/download --dst CONVERTED/reinterhand
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `../InterWild/tool/ReInterHand/download` | ReInterHand download directory |
| `--dst` | `CONVERTED/reinterhand` | Output directory |
| `--chunk-size` | `28` | Sequences per chunk |
| `--fps` | `30.0` | Output video frame rate |
| `--max-samples` | `0` | Max chunks (0=all) |

### Visualization

```bash
python scripts/visualize.py --src CONVERTED/dex_ycb --n 100 --out outputs
python scripts/visualize.py --src CONVERTED/ho_cap --n 50 --out outputs
python scripts/visualize.py --src CONVERTED/whim --n 20 --out outputs
```

Randomly samples frames, projects 3D keypoints onto images, and saves individual annotated images.

### Evaluate Planarity

```bash
python scripts/eval_planarity.py --src CONVERTED/dex_ycb
python scripts/eval_planarity.py --src CONVERTED/egodex --normalize
```

## Project Structure

```
├── scripts/
│   ├── convert_dex_ycb.py         # DexYCB converter
│   ├── convert_ho_cap.py          # HO-Cap converter
│   ├── convert_egodex.py          # EgoDex ARKit -> +Z up converter
│   ├── convert_whim.py            # WHIM converter (YouTube download + convert)
│   ├── convert_arctic.py          # ARCTIC converter
│   ├── convert_interhand26m.py    # InterHand2.6M converter
│   ├── convert_h2o3d.py           # H2O-3D converter
│   ├── convert_mtc.py             # MTC (Panoptic Studio) converter
│   ├── convert_reinterhand.py     # ReInterHand converter
│   ├── generate_report.py         # Dataset statistics report generator
│   ├── visualize.py               # 3D keypoint visualization
│   ├── eval_planarity.py          # Finger joint planarity evaluation
│   └── estimate_depth.py          # Depth estimation via MoGe
├── utils/
│   ├── io.py                      # File I/O (YAML, HDF5, video encoding)
│   ├── joint_mapping.py           # MANO <-> egodex joint name mapping
│   ├── transforms.py              # 3D joint -> SE(3) transform computation
│   ├── camera_utils.py            # Camera intrinsics/poses utilities
│   ├── image_utils.py             # Image cropping and projection
│   └── depth_utils.py             # Depth estimation utilities
├── CONVERTED/                     # Converted datasets (git-ignored)
└── DATASET/                       # Source datasets (git-ignored)
```
