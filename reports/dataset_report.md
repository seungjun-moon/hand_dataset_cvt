# Converted Dataset Report

Generated: 2026-03-19

## Summary

| Dataset | Format | Sequences | Viewpoints | Data Files | Unique Frames | Total Frames (all views) | Disk Size |
|---------|--------|----------:|-----------:|-----------:|--------------:|-------------------------:|----------:|
| DexYCB         | HDF5   |     1,000 |          8 |      8,000 |        63,548 |                  508,384 |       52G |
| HO-Cap         | HDF5   |        64 |          8 |        512 |        72,575 |                  583,183 |      6.0G |
| EgoDex         | HDF5   |     3,769 |          1 |      3,769 |     1,469,838 |                1,469,838 |       34G |
| WHIM Train     | HDF5   |     176\* |          1 |      176\* |     329,403\* |                329,403\* |      7.3G |
| WHIM Test      | HDF5   |        26 |          1 |         26 |        34,869 |                   34,869 |      944M |
| InterHand2.6M  | HDF5   |     4,909 |          1 |      4,909 |       820,210 |                  820,210 |       15G |
| FreiHAND Train | NPZ    |    32,560 |          4 |    130,240 |        32,560 |                  130,240 |      3.5G |
| FreiHAND Eval  | NPZ    |     3,960 |          1 |      3,960 |         3,960 |                    3,960 |      105M |
| RHD            | HDF5   |        29 |         25 |        725 |        14,500 |                  362,500 |      3.5G |
| HIC            | HDF5   |        18 |          1 |         18 |           732 |                      732 |       16M |
| **Total** | | | — | **152,335** | **2,842,195** | **4,243,319** | |

\* WHIM Train: 176/1,431 videos completed, 1,255 failed (YouTube unavailable).

## Hand Side Distribution

| Dataset | Left Only | Right Only | Both Hands | Total Files |
|---------|----------:|-----------:|-----------:|------------:|
| DexYCB         |     3,992 |      4,008 |          0 |       8,000 |
| HO-Cap         |        64 |        280 |        168 |         512 |
| EgoDex         |         1 |         75 |      3,693 |       3,769 |
| WHIM Train     |         0 |          0 |        176 |         176 |
| WHIM Test      |         0 |          0 |         26 |          26 |
| InterHand2.6M  |       143 |      1,154 |      3,612 |       4,909 |
| FreiHAND Train |         0 |    130,240 |          0 |     130,240 |
| FreiHAND Eval  |         0 |      3,960 |          0 |       3,960 |
| RHD            |         0 |        725 |          0 |         725 |
| HIC            |         0 |          8 |         10 |          18 |

## Frame Statistics

| Dataset | Min | Mean | Max | Unique Frames | Total Frames |
|---------|----:|-----:|----:|--------------:|-------------:|
| DexYCB         |  25 |   63 |    76 |        63,548 |      508,384 |
| HO-Cap         | 446 | 1,139 | 2,457 |        72,575 |      583,183 |
| EgoDex         |  15 |  389 | 4,784 |     1,469,838 |    1,469,838 |
| WHIM Train     |  16 | 1,871 | 13,330 |       329,403 |      329,403 |
| WHIM Test      |  41 | 1,341 | 2,482 |        34,869 |       34,869 |
| InterHand2.6M  |   5 |  167 |   590 |       820,210 |      820,210 |
| FreiHAND Train |   1 |    1 |     1 |        32,560 |      130,240 |
| FreiHAND Eval  |   1 |    1 |     1 |         3,960 |        3,960 |
| RHD            | 500 |  500 |   500 |        14,500 |      362,500 |
| HIC            |  22 |   40 |    80 |           732 |          732 |

## Per-Dataset Details

### DexYCB

| Property | Value |
|----------|-------|
| Source | [DexYCB](https://dex-ycb.github.io/) — tabletop grasping with YCB objects |
| Clusters | 20 |
| Viewpoints | 8 |
| Hand sides | left: 3,992 / right: 4,008 / both: 0 |
| Frames/file | 25–76 (mean: 63) |
| MANO | 100% |

### HO-Cap

| Property | Value |
|----------|-------|
| Source | [HO-Cap](https://irvlutd.github.io/HOCap/) — hand-object interaction |
| Clusters | 9 |
| Viewpoints | 8 |
| Hand sides | left: 64 / right: 280 / both: 168 |
| Frames/file | 446–2,457 (mean: 1,139) |
| MANO | 100% |

### EgoDex

| Property | Value |
|----------|-------|
| Source | [EgoDex](https://github.com/facebookresearch/ego-dex) — egocentric hand tracking |
| Clusters | 3 |
| Viewpoints | 1 |
| Hand sides | left: 1 / right: 75 / both: 3,693 |
| Frames/file | 15–4,784 (mean: 389) |
| MANO | 100% |

### WHIM Train

| Property | Value |
|----------|-------|
| Source | [WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data |
| Clusters | 1 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 0 / both: 176 |
| Frames/file | 16–13,330 (mean: 1,871) |
| MANO | 100% |

### WHIM Test

| Property | Value |
|----------|-------|
| Source | [WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data |
| Clusters | 1 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 0 / both: 26 |
| Frames/file | 41–2,482 (mean: 1,341) |
| MANO | 100% |

### InterHand2.6M

| Property | Value |
|----------|-------|
| Source | [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) — multi-view interacting hands |
| Clusters | 68 |
| Viewpoints | 1 |
| Hand sides | left: 143 / right: 1,154 / both: 3,612 |
| Frames/file | 5–590 (mean: 167) |
| MANO | 100% |

### FreiHAND Train

| Property | Value |
|----------|-------|
| Source | [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — real hand images with MANO annotations |
| Clusters | 33 |
| Viewpoints | 4 |
| Hand sides | left: 0 / right: 130,240 / both: 0 |
| Frames/file | 1 (constant) |
| MANO | 100% |

### FreiHAND Eval

| Property | Value |
|----------|-------|
| Source | [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — real hand images with MANO annotations |
| Clusters | 4 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 3,960 / both: 0 |
| Frames/file | 1 (constant) |
| MANO | 100% |

### RHD

| Property | Value |
|----------|-------|
| Source | [RHD](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) — rendered synthetic hand poses |
| Clusters | 29 |
| Viewpoints | 25 |
| Hand sides | left: 0 / right: 725 / both: 0 |
| Frames/file | 500 (constant) |
| MANO | 100% |

### HIC

| Property | Value |
|----------|-------|
| Source | [HIC](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/) — hand-in-contact interaction |
| Clusters | 18 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 8 / both: 10 |
| Frames/file | 22–80 (mean: 40) |
| MANO | 100% |

