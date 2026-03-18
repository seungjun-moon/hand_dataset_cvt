# Converted Dataset Report

Generated: 2026-03-18

## Summary

| Dataset | Sequences | Viewpoints | HDF5 Files | Unique Frames | Total Frames (all views) | Frames/Seq (mean) | Disk Size | MANO |
|---------|-----------|------------|------------|-------------:|--------------------------:|-------------------:|----------:|-----:|
| DexYCB  | 1,000     | 8          | 8,000      | 63,548        | 508,384                  | 64                 | 52 GB     | 100% |
| HO-Cap  | 64        | 8          | 512        | 72,898        | 583,183                  | 1,139              | 6.0 GB    | 100% |
| EgoDex  | 3,769     | 1          | 3,769      | 1,469,838     | 1,469,838                | 390                | 34 GB     | 0%   |
| WHIM    | 109*      | 1          | 109*       | 275,228*      | 275,228*                 | 2,525              | 5.7 GB*   | 100% |
| **Total** | **4,942** | —        | **12,390** | **1,881,512** | **2,836,633**            | —                 | **98 GB** | —    |

\* WHIM conversion in progress: 109/1,431 videos completed, 1,322 failed (YouTube unavailable).

## Per-Dataset Details

### DexYCB

- **Source**: [DexYCB](https://dex-ycb.github.io/) — tabletop grasping with YCB objects
- **Clusters**: 20 (one per YCB object)
- **Cameras**: 8 synchronized RealSense D415
- **Resolution**: 640x480
- **Hand sides**: left=3,992, right=4,008 (single hand per sequence)
- **Frame range**: 25–76 frames/sequence
- **MANO**: Full parameters (PCA-decoded to rotation matrices)
- **Coordinate space**: Camera → world via extrinsics

### HO-Cap

- **Source**: [HO-Cap](https://irvlutd.github.io/HOCap/) — hand-object interaction
- **Clusters**: 9 subjects (subject_1 through subject_9)
- **Cameras**: 8 synchronized RealSense D455
- **Resolution**: 640x480
- **Hand sides**: left=64, right=280, both=168
- **Frame range**: 446–2,457 frames/sequence
- **MANO**: Full parameters (axis-angle to rotation matrices)
- **Coordinate space**: Camera → world via extrinsics (master camera = identity)

### EgoDex

- **Source**: [EgoDex](https://github.com/facebookresearch/ego-dex) — egocentric hand tracking (ARKit)
- **Clusters**: 3 task categories
- **Cameras**: 1 (egocentric, Apple device)
- **Hand sides**: both=3,693, right-only=75, left-only=1
- **Frame range**: 15–4,784 frames/sequence
- **MANO**: Not available (keypoint transforms only)
- **Coordinate space**: ARKit → Z-up via world rotation

### WHIM

- **Source**: [WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data
- **Clusters**: train split (test not yet converted)
- **Cameras**: 1 (YouTube videos, varied resolution)
- **Hand sides**: both=109 (all sequences have both hands)
- **Frame range**: 58–13,330 frames/sequence
- **MANO**: Full parameters from WiLoR estimation (rotation matrices + median betas)
- **Coordinate space**: Camera space (identity camera pose)
- **Multi-hand policy**: Largest bbox per side per frame; same-side duplicates filtered
- **Note**: 1,322/1,431 videos failed to download (YouTube unavailable). Re-run to retry.

## Hand Side Distribution

| Dataset | Left Only | Right Only | Both Hands | Total Files |
|---------|----------:|------------|------------|------------:|
| DexYCB  | 3,992     | 4,008      | 0          | 8,000       |
| HO-Cap  | 64        | 280        | 168        | 512         |
| EgoDex  | 1         | 75         | 3,693      | 3,769       |
| WHIM    | 0         | 0          | 109        | 109         |

## Frame Statistics

| Dataset | Min Frames | Mean Frames | Max Frames | Total Frames |
|---------|------------|-------------|------------|-------------:|
| DexYCB  | 25         | 64          | 76         | 508,384      |
| HO-Cap  | 446        | 1,139       | 2,457      | 583,183      |
| EgoDex  | 15         | 390         | 4,784      | 1,469,838    |
| WHIM    | 58         | 2,525       | 13,330     | 275,228      |
