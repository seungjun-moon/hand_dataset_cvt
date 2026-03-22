# Converted Dataset Report

Generated: 2026-03-22

## Summary

| Dataset | Format | Sequences | Viewpoints | Data Files | Unique Frames | Total Frames (all views) |
|---------|--------|----------:|-----------:|-----------:|--------------:|-------------------------:|
| DexYCB         | HDF5   |     1,000 |          8 |      8,000 |        63,548 |                  508,384 |
| HO-Cap         | HDF5   |        64 |          8 |        512 |        72,575 |                  583,183 |
| EgoDex         | HDF5   |     3,769 |          1 |      3,769 |     1,469,838 |                1,469,838 |
| ARCTIC         | HDF5   |       232 |        8–9 |      2,087 |       170,175 |                1,530,710 |
| WHIM Train     | HDF5   |     216\* |          1 |      216\* |     378,096\* |                378,096\* |
| WHIM Test      | HDF5   |        26 |          1 |         26 |        34,869 |                   34,869 |
| InterHand2.6M Train | HDF5   |     7,767 |          1 |      7,767 |     1,343,641 |                1,343,641 |
| InterHand2.6M Test | HDF5   |     2,166 |          1 |      2,166 |     1,228,561 |                1,228,561 |
| FreiHAND Train | NPZ    |    32,560 |          4 |    130,240 |        32,560 |                  130,240 |
| FreiHAND Eval  | NPZ    |     3,960 |          1 |      3,960 |         3,960 |                    3,960 |
| RHD            | HDF5   |        29 |         25 |        725 |        14,500 |                  362,500 |
| HIC            | HDF5   |        18 |          1 |         18 |           732 |                      732 |
| H2O-3D         | HDF5   |        69 |          1 |         69 |        60,998 |                   60,998 |
| MTC Train      | HDF5   |        29 |      29–31 |        874 |         9,804 |                  295,757 |
| ReInterHand    | HDF5   |       200 |          1 |        200 |       375,123 |                  375,123 |
| **Total** | | | — | **160,629** | **5,258,980** | **8,306,592** |

\* WHIM Train: 215/245 videos completed, 30 failed (YouTube unavailable).

## Hand Side Distribution

| Dataset | Left Only | Right Only | Both Hands | Total Files |
|---------|----------:|-----------:|-----------:|------------:|
| DexYCB         |     3,992 |      4,008 |          0 |       8,000 |
| HO-Cap         |        64 |        280 |        168 |         512 |
| EgoDex         |         1 |         75 |      3,693 |       3,769 |
| ARCTIC         |         0 |          0 |      2,087 |       2,087 |
| WHIM Train     |         0 |          0 |        216 |         216 |
| WHIM Test      |         0 |          0 |         26 |          26 |
| InterHand2.6M Train |       143 |      1,888 |      5,736 |       7,767 |
| InterHand2.6M Test |         1 |        437 |      1,728 |       2,166 |
| FreiHAND Train |         0 |    130,240 |          0 |     130,240 |
| FreiHAND Eval  |         0 |      3,960 |          0 |       3,960 |
| RHD            |         0 |        725 |          0 |         725 |
| HIC            |         0 |          8 |         10 |          18 |
| H2O-3D         |         0 |          0 |         69 |          69 |
| MTC Train      |         0 |          0 |        874 |         874 |
| ReInterHand    |         0 |          0 |        200 |         200 |

## Frame Statistics

| Dataset | Min | Mean | Max | Unique Frames | Total Frames |
|---------|----:|-----:|----:|--------------:|-------------:|
| DexYCB         |  25 |   63 |    76 |        63,548 |      508,384 |
| HO-Cap         | 446 | 1,139 | 2,457 |        72,575 |      583,183 |
| EgoDex         |  15 |  389 | 4,784 |     1,469,838 |    1,469,838 |
| ARCTIC         | 559 |  733 | 1,117 |       170,175 |    1,530,710 |
| WHIM Train     |  16 | 1,750 | 13,330 |       378,096 |      378,096 |
| WHIM Test      |  41 | 1,341 | 2,482 |        34,869 |       34,869 |
| InterHand2.6M Train |   5 |  172 |   590 |     1,343,641 |    1,343,641 |
| InterHand2.6M Test |   3 |  567 | 2,739 |     1,228,561 |    1,228,561 |
| FreiHAND Train |   1 |    1 |     1 |        32,560 |      130,240 |
| FreiHAND Eval  |   1 |    1 |     1 |         3,960 |        3,960 |
| RHD            | 500 |  500 |   500 |        14,500 |      362,500 |
| HIC            |  22 |   40 |    80 |           732 |          732 |
| H2O-3D         | 233 |  884 | 1,458 |        60,998 |       60,998 |
| MTC Train      | 136 |  338 |   712 |         9,804 |      295,757 |
| ReInterHand    | 1,062 | 1,875 | 4,091 |       375,123 |      375,123 |

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

### ARCTIC

| Property | Value |
|----------|-------|
| Source | [ARCTIC](https://arctic.is.tue.mpg.de/) — articulated object manipulation with dexterous bimanual hands |
| Clusters | 74 |
| Viewpoints | 8–9 |
| Hand sides | left: 0 / right: 0 / both: 2,087 |
| Frames/file | 559–1,117 (mean: 733) |
| MANO | 100% |

### WHIM Train

| Property | Value |
|----------|-------|
| Source | [WHIM/WiLoR](https://rolpotamern.github.io/WiLoR/) — in-the-wild YouTube hand data |
| Clusters | 1 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 0 / both: 216 |
| Frames/file | 16–13,330 (mean: 1,750) |
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

### InterHand2.6M Train

| Property | Value |
|----------|-------|
| Source | [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) — multi-view interacting hands (train split) |
| Clusters | 107 |
| Viewpoints | 1 |
| Hand sides | left: 143 / right: 1,888 / both: 5,736 |
| Frames/file | 5–590 (mean: 172) |
| MANO | 100% |

### InterHand2.6M Test

| Property | Value |
|----------|-------|
| Source | [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) — multi-view interacting hands (test split) |
| Clusters | 27 |
| Viewpoints | 1 |
| Hand sides | left: 1 / right: 437 / both: 1,728 |
| Frames/file | 3–2,739 (mean: 567) |
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

### H2O-3D

| Property | Value |
|----------|-------|
| Source | [H2O-3D](https://www.taeinkwon.com/projects/h2o) — egocentric hand-object interaction with 3D annotations |
| Clusters | 9 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 0 / both: 69 |
| Frames/file | 233–1,458 (mean: 884) |
| MANO | 100% |

### MTC Train

| Property | Value |
|----------|-------|
| Source | [MTC](http://domedb.perception.cs.cmu.edu/handdb.html) — Panoptic Studio multi-view hand capture (train) |
| Clusters | 29 |
| Viewpoints | 29–31 |
| Hand sides | left: 0 / right: 0 / both: 874 |
| Frames/file | 136–712 (mean: 338) |
| MANO | 100% |

### ReInterHand

| Property | Value |
|----------|-------|
| Source | [ReInterHand](https://mks0601.github.io/ReInterHand/) — re-annotated interacting hands with MANO fits |
| Clusters | 11 |
| Viewpoints | 1 |
| Hand sides | left: 0 / right: 0 / both: 200 |
| Frames/file | 1,062–4,091 (mean: 1,875) |
| MANO | 100% |

