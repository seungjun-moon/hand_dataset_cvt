# Converted Dataset Report

Generated: 2026-03-18

## Summary

| Dataset    | Sequences | Viewpoints | HDF5 Files | Unique Frames | Total Frames (all views) | Disk Size |
|------------|----------:|-----------:|-----------:|--------------:|-------------------------:|----------:|
| DexYCB     |     1,000 |          8 |      8,000 |        63,548 |                  508,384 |     52 GB |
| HO-Cap     |        64 |          8 |        512 |        72,897 |                  583,183 |    6.0 GB |
| EgoDex     |     3,769 |          1 |      3,769 |     1,469,838 |                1,469,838 |     34 GB |
| WHIM Train |     109\* |          1 |      109\* |      275,228\*|                275,228\* |    5.7 GB |
| WHIM Test  |        26 |          1 |         26 |        34,869 |                   34,869 |   944 MB  |
| **Total**  | **4,968** |        — | **12,416** | **1,916,380** |            **2,871,502** |  **99 GB**|

\* WHIM Train: 109/1,431 videos completed, 1,322 failed (YouTube unavailable).

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
| Splits | Train: 109/1,431 completed / Test: 26/26 completed |
| Cameras | 1 (YouTube videos, varied resolution) |
| Hand sides | both: 132 (all sequences have both hands) |
| Frames/seq | 41–13,330 (mean: 2,310) |
| MANO | 100% — WiLoR estimation (rotation matrices, median betas) |
| Coordinates | Camera space (identity camera pose) |
| Multi-hand | Largest bbox per side per frame; same-side duplicates filtered |

## Hand Side Distribution

| Dataset    | Left Only | Right Only | Both Hands | Total Files |
|------------|----------:|-----------:|-----------:|------------:|
| DexYCB     |     3,992 |      4,008 |          0 |       8,000 |
| HO-Cap     |        64 |        280 |        168 |         512 |
| EgoDex     |         1 |         75 |      3,693 |       3,769 |
| WHIM Train |         0 |          0 |        109 |         109 |
| WHIM Test  |         0 |          0 |         26 |          26 |

## Frame Statistics

| Dataset    | Min | Mean  | Max    | Unique Frames | Total Frames |
|------------|----:|------:|-------:|--------------:|-------------:|
| DexYCB     |  25 |    64 |     76 |        63,548 |      508,384 |
| HO-Cap     | 446 | 1,139 |  2,457 |        72,897 |      583,183 |
| EgoDex     |  15 |   390 |  4,784 |     1,469,838 |    1,469,838 |
| WHIM Train |  58 | 2,525 | 13,330 |       275,228 |      275,228 |
| WHIM Test  |  41 | 1,341 |  2,482 |        34,869 |       34,869 |
