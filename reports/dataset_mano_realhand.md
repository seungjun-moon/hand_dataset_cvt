# MANO Planarity & Vertex Error Report

Generated: 2026-03-25 18:23
- Vanilla: `../hamer/hamer_training_data/dataset_tars`
- Constrained: `../hamer/hamer_training_data/dataset_tars_manotorch`
- Max tars per dataset: 50
- MANO backend: smplx

## Summary

| Dataset | Samples | Planarity Vanilla (mm) | Planarity Constrained (mm) | Vertex Error (mm) | Correlation |
|---------|--------:|----------------------:|--------------------------:|------------------:|------------:|
| cocow-train | 36051 | - | - | - | - |
| dex-train | 50000 | 3.92 | 2.22 | 6.93 | 0.160 |
| freihand-train | 50000 | 3.36 | 2.49 | 1.95 | 0.030 |
| h2o3d-train | 47320 | 1.95 | 1.34 | 13.57 | 0.315 |
| halpe-train | 22088 | - | - | - | - |
| ho3d-train | 50000 | 2.43 | 2.36 | 0.55 | 0.070 |
| interhand26m-train | 50000 | 2.57 | 1.87 | 5.68 | 0.242 |
| mpiinzsl-train | 15184 | - | - | - | - |
| mtc-train | 50000 | - | - | - | - |
| rhd-train | 41250 | - | - | - | - |

## Per-Finger Planarity (mean, mm)

### Vanilla MANO

| Dataset | Index | Middle | Ring | Little |
|---------|------:|-------:|-----:|-------:|
| dex-train | 6.86 | 4.18 | 1.61 | 3.05 |
| freihand-train | 4.45 | 2.97 | 2.38 | 3.63 |
| h2o3d-train | 3.76 | 2.15 | 0.62 | 1.28 |
| ho3d-train | 4.83 | 2.52 | 0.66 | 1.69 |
| interhand26m-train | 3.69 | 2.46 | 0.96 | 3.18 |

### Constrained MANO

| Dataset | Index | Middle | Ring | Little |
|---------|------:|-------:|-----:|-------:|
| dex-train | 4.61 | 2.47 | 0.22 | 1.58 |
| freihand-train | 4.16 | 2.94 | 0.74 | 2.13 |
| h2o3d-train | 2.28 | 0.73 | 0.51 | 1.86 |
| ho3d-train | 4.74 | 2.30 | 0.60 | 1.80 |
| interhand26m-train | 3.03 | 1.86 | 0.52 | 2.06 |

## Vertex Error Details (mm)

Vertex error = mean per-vertex L2 distance between vanilla and constrained MANO meshes.

| Dataset | Mean | Median | Max |
|---------|-----:|-------:|----:|
| dex-train | 6.93 | 2.44 | 35.57 |
| freihand-train | 1.95 | 1.81 | 7.93 |
| h2o3d-train | 13.57 | 13.18 | 33.61 |
| ho3d-train | 0.55 | 0.44 | 2.48 |
| interhand26m-train | 5.68 | 1.51 | 50.40 |

## Correlation: Planarity Error vs Vertex Error

Pearson correlation between per-sample mean planarity error (vanilla MANO)
and per-sample mean vertex error (vanilla vs constrained).
Higher correlation means samples with worse planarity see larger mesh changes after constrained IK.

| Dataset | Pearson r | Interpretation |
|---------|----------:|----------------|
| dex-train | 0.160 | No correlation |
| freihand-train | 0.030 | No correlation |
| h2o3d-train | 0.315 | Weak positive |
| ho3d-train | 0.070 | No correlation |
| interhand26m-train | 0.242 | Weak positive |

