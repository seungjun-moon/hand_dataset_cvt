# Hand Dataset Converter

Convert hand pose datasets (e.g., [DexYCB](https://dex-ycb.github.io/)) into a unified egodex-compatible format with HDF5 annotations and MP4 videos.

## Output Format

Each converted sequence produces a directory containing:
- `0.hdf5` — Camera intrinsics, per-frame 4×4 SE(3) transforms, and confidence scores for all joints
- `0.mp4` — RGB video from the selected camera

### HDF5 Structure

```
camera/
    intrinsic          (3, 3)
transforms/
    camera             (N, 4, 4)
    leftHand           (N, 4, 4)
    rightIndexFingerTip (N, 4, 4)
    ...
confidences/
    leftHand           (N,)
    rightIndexFingerTip (N,)
    ...
```

Joint names follow the egodex convention: `{side}{Joint}` (e.g., `rightThumbKnuckle`, `leftMiddleFingerTip`). Body joints (hip, shoulders, spine, etc.) are included with zero confidence.

## Setup

```bash
pip install numpy opencv-python h5py pyyaml
```

## Usage

### DexYCB → Egodex

Place the DexYCB dataset under `DATASET/dex_ycb/`, then run:

```bash
python scripts/convert_dex_ycb.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `DATASET/dex_ycb` | Source DexYCB directory |
| `--dst` | `CONVERTED/dex_ycb` | Output directory |
| `--camera-idx` | `0` | Camera index to extract |
| `--fps` | `30.0` | Output video frame rate |

### DexYCB Source Structure

```
DATASET/dex_ycb/
    {date}-{subject}/
        {timestamp}/
            meta.yml
            {camera_serial}/
                color_XXXXXX.jpg
                labels_XXXXXX.npz
        calibration/
            intrinsics/{serial}_640x480.yml
            extrinsics_{name}/extrinsics.yml
```

### EgoDex Coordinate Conversion

Convert EgoDex datasets from ARKit coordinates (+Y up) to +Z up:

```bash
python scripts/convert_egodex.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `RAW/egodex` | Source egodex directory |
| `--dst` | `CONVERTED/egodex` | Output directory |
| `--max-samples` | `0` | Max sequences to convert (0=all) |

The script applies a world-frame rotation to all transforms, computes camera-space transforms (`transforms_cam/`), and outputs each sequence as `{idx:06d}_{task_name}/0.hdf5` + `0.mp4`.

### Evaluate Planarity

Evaluate finger joint planarity (coplanarity of Knuckle-IntermediateBase-IntermediateTip-Tip) on converted datasets:

```bash
python scripts/eval_planarity.py --src CONVERTED/dex_ycb
python scripts/eval_planarity.py --src CONVERTED/egodex --normalize
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `CONVERTED/dex_ycb` | Converted dataset directory |
| `--normalize` | — | Report error as % of finger length |
| `--max-samples` | `0` | Max sequences to evaluate (0=all) |

## Project Structure

```
├── download/
│   └── download_egodex.sh        # Download EgoDex dataset
├── scripts/
│   ├── convert_dex_ycb.py        # DexYCB → egodex conversion
│   ├── convert_egodex.py         # EgoDex ARKit → +Z up conversion
│   └── eval_planarity.py         # Finger joint planarity evaluation
├── utils/
│   ├── io.py                     # File I/O (YAML, HDF5, video encoding)
│   ├── joint_mapping.py          # MANO ↔ egodex joint name mapping
│   └── transforms.py             # 3D joint → SE(3) transform computation
├── RAW/                          # Raw downloaded data (git-ignored)
├── DATASET/                      # Source datasets (git-ignored)
└── CONVERTED/                      # Converted datasets (git-ignored)
```
