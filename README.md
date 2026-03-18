# Hand Dataset Converter

Convert hand pose datasets into a unified egodex-compatible format with HDF5 annotations and MP4 videos.

Supported datasets:
- [DexYCB](https://dex-ycb.github.io/)
- [HO-Cap](https://irvlutd.github.io/HOCap/)
- [EgoDex](https://github.com/facebookresearch/ego-dex)
- [WHIM](https://rolpotamern.github.io/WiLoR/) (in-the-wild YouTube hand data)

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
└── whim/
    ├── train/                          # One sequence per YouTube video
    │   ├── {seq_idx:06d}_label_00.hdf5
    │   └── {seq_idx:06d}_video_00.mp4
    ├── test/
    │   └── ...
    ├── completed_train.json            # Tracks completed videos (for resuming)
    └── completed_test.json
```

Multi-view is supported via the camera index suffix (`_00`, `_01`, ...). DexYCB and HO-Cap have 8+ cameras per sequence; EgoDex and WHIM are single-camera.

## HDF5 Structure

Each `_label_XX.hdf5` file contains:

```
camera/
    intrinsic                       (3, 3) float32    # Pinhole camera matrix [fx,0,cx; 0,fy,cy; 0,0,1]

transforms/
    camera                          (N, 4, 4) float32 # Camera-to-world SE(3) pose (identity for WHIM)
    {side}Hand                      (N, 4, 4) float32 # Wrist transform
    {side}ThumbKnuckle              (N, 4, 4) float32
    {side}ThumbIntermediateBase     (N, 4, 4) float32
    {side}ThumbIntermediateTip      (N, 4, 4) float32
    {side}ThumbTip                  (N, 4, 4) float32
    {side}IndexFingerKnuckle        (N, 4, 4) float32
    {side}IndexFingerIntermediateBase (N, 4, 4) float32
    {side}IndexFingerIntermediateTip  (N, 4, 4) float32
    {side}IndexFingerTip            (N, 4, 4) float32
    {side}MiddleFinger...           ...                # Same pattern for Middle, Ring, Little
    {side}IndexFingerMetacarpal     (N, 4, 4) float32 # Interpolated (alpha=0.3 between wrist and MCP)
    {side}MiddleFingerMetacarpal    ...
    {side}RingFingerMetacarpal      ...
    {side}LittleFingerMetacarpal    ...
    hip                             (N, 4, 4) float32 # Body joints (identity, zero confidence)
    leftShoulder, leftArm, ...      ...
    spine1 .. spine7, neck1 .. neck4 ...

confidences/
    {side}Hand                      (N,) float32      # 1.0 if active, 0.0 if inactive
    ...                                                # Same keys as transforms/ (except camera)

mano_{side}/                                           # Optional, present when MANO data available
    betas                           (10,) float32      # Shape parameters
    global_orient_worldspace        (N, 3, 3) float32  # Global rotation matrices (world space)
    hand_pose                       (N, 15, 3, 3) float32  # Per-joint rotation matrices
    transl_worldspace               (N, 3) float32     # Translation (world space)
    kpt3d                           (N, 21, 3) float32 # 3D keypoints (world space)
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
