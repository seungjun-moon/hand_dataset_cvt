# Converted Dataset Report

Generated: 2026-03-19

## Summary

| Dataset        | Format | Sequences | Viewpoints | Data Files | Unique Frames | Total Frames (all views) | Disk Size |
|----------------|--------|----------:|-----------:|-----------:|--------------:|-------------------------:|----------:|
| DexYCB         | HDF5   |     1,000 |          8 |      8,000 |        63,548 |                  508,384 |     52 GB |
| HO-Cap         | HDF5   |        64 |          8 |        512 |        72,897 |                  583,183 |    6.0 GB |
| EgoDex         | HDF5   |     3,769 |          1 |      3,769 |     1,469,838 |                1,469,838 |     34 GB |
| WHIM Train     | HDF5   |     176\* |          1 |      176\* |      329,403\*|                329,403\* |    7.3 GB |
| WHIM Test      | HDF5   |        26 |          1 |         26 |        34,869 |                   34,869 |   944 MB  |
| FreiHAND Train | NPZ    |    32,560 |          4 |    130,240 |        32,560 |                  130,240 |    3.5 GB |
| FreiHAND Eval  | NPZ    |     3,960 |          1 |      3,960 |         3,960 |                    3,960 |   105 MB  |
| RHD            | HDF5   |         5 |      16–25 |        116 |         2,500 |                   58,000 |   577 MB  |
| HIC            | HDF5   |        18 |          1 |         18 |            18 |                      732 |    15 MB  |
| **Total**      |        | **41,578**|        —   |**146,817** |**2,009,593**  |            **3,118,609** | **105 GB**|

\* WHIM Train: 176/1,431 videos completed, 1,255 failed (YouTube unavailable).

## Per-Dataset Details

### DexYCB

| Property | Value |
|----------|-------|
| Source | [DexYCB](https://dex-ycb.github.io/) — tabletop grasping with YCB objects |
| Clusters | 20 (one per YCB object) |
| Cameras | 8 synchronized RealSense D415 |
| Resolution | 640x480 |
| Hand sides | left: 3,992 / right: 4,008 (single hand per sequence) |
| Frames/seq | 25–76 (mean: 64) |
| MANO | 100% — PCA-decoded to rotation matrices |
| Coordinates | Camera → world via extrinsics |

### HO-Cap

| Property | Value |
|----------|-------|
| Source | [HO-Cap](https://irvlutd.github.io/HOCap/) — hand-object interaction |
| Clusters | 9 subjects (subject_1 through subject_9) |
| Cameras | 8 synchronized RealSense D455 |
| Resolution | 640x480 |
| Hand sides | left: 64 / right: 280 / both: 168 |
| Frames/seq | 446–2,457 (mean: 1,139) |
| MANO | 100% — axis-angle to rotation matrices |
| Coordinates | Camera → world via extrinsics (master camera = identity) |

### EgoDex

| Property | Value |
|----------|-------|
| Source | [EgoDex](https://github.com/facebookresearch/ego-dex) — egocentric hand tracking |
| Clusters | 3 task categories |
| Cameras | 1 (egocentric, Apple device) |
| Hand sides | left: 1 / right: 75 / both: 3,693 |
| Frames/seq | 15–4,784 (mean: 390) |
| MANO | Not available (keypoint transforms only) |
| Coordinates | ARKit (+Y up) → Z-up via world rotation |

### WHIM

| Property | Value |
|----------|-------|
| Source | [WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data |
| Splits | Train: 176/1,431 completed / Test: 26/26 completed |
| Cameras | 1 (YouTube videos, varied resolution) |
| Resolution | 1920x1080 |
| Hand sides | both: 202 (all sequences have both hands) |
| Frames/seq | 16–13,330 (mean: 1,872) |
| MANO | 100% — WiLoR estimation (rotation matrices, median betas) |
| Coordinates | Camera space (identity camera pose) |
| Multi-hand | Largest bbox per side per frame; same-side duplicates filtered |

### FreiHAND

| Property | Value |
|----------|-------|
| Source | [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — real hand images with MANO annotations |
| Splits | Train: 32,560 unique frames (130,240 with augmented views) / Eval: 3,960 frames |
| Clusters | Train: 33 / Eval: 4 |
| Cameras | Train: 4 augmented views per frame / Eval: 1 |
| Resolution | 224x224 |
| Hand sides | right: 100% |
| MANO | 100% — rotation matrices (3x3 global_orient, 15x3x3 hand_pose) |
| Coordinates | Camera space with extrinsics |

### RHD

| Property | Value |
|----------|-------|
| Source | [RHD](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) — rendered synthetic hand poses |
| Clusters | 5 lighting conditions (l01–l05) |
| Cameras | 16–25 per lighting condition |
| Resolution | 360x360 |
| Hand sides | right: 100% |
| Frames/seq | 500 (constant) |
| MANO | 100% — rotation matrices (worldspace) |
| Coordinates | Camera → world via extrinsics |

### HIC

| Property | Value |
|----------|-------|
| Source | [HIC](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/) — hand-in-contact interaction |
| Sequences | 18 (IDs 01–11, 15–21; 12–14 missing from source) |
| Cameras | 1 |
| Resolution | 640x480 |
| Hand sides | right: 8 / both: 10 |
| Frames/seq | 22–80 (mean: 41) |
| MANO | 100% — rotation matrices (worldspace) |
| Coordinates | Camera space |

## Hand Side Distribution

| Dataset        | Left Only | Right Only | Both Hands | Total Files |
|----------------|----------:|-----------:|-----------:|------------:|
| DexYCB         |     3,992 |      4,008 |          0 |       8,000 |
| HO-Cap         |        64 |        280 |        168 |         512 |
| EgoDex         |         1 |         75 |      3,693 |       3,769 |
| WHIM Train     |         0 |          0 |        176 |         176 |
| WHIM Test      |         0 |          0 |         26 |          26 |
| FreiHAND Train |         0 |    130,240 |          0 |     130,240 |
| FreiHAND Eval  |         0 |      3,960 |          0 |       3,960 |
| RHD            |         0 |        116 |          0 |         116 |
| HIC            |         0 |          8 |         10 |          18 |

## Frame Statistics

| Dataset        | Min | Mean  | Max    | Unique Frames | Total Frames |
|----------------|----:|------:|-------:|--------------:|-------------:|
| DexYCB         |  25 |    64 |     76 |        63,548 |      508,384 |
| HO-Cap         | 446 | 1,139 |  2,457 |        72,897 |      583,183 |
| EgoDex         |  15 |   390 |  4,784 |     1,469,838 |    1,469,838 |
| WHIM Train     |  16 | 1,872 | 13,330 |       329,403 |      329,403 |
| WHIM Test      |  41 | 1,341 |  2,482 |        34,869 |       34,869 |
| FreiHAND Train |   1 |     1 |      1 |        32,560 |      130,240 |
| FreiHAND Eval  |   1 |     1 |      1 |         3,960 |        3,960 |
| RHD            | 500 |   500 |    500 |         2,500 |       58,000 |
| HIC            |  22 |    41 |     80 |            18 |          732 |
