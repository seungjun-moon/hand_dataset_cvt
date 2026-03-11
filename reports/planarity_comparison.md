# Planarity Error Comparison Report

## Datasets

- **HO-Cap**: subject_1/20231025_165502 (MediaPipe 3D joints, 1 sequence)
- **EgoDex**: CONVERTED/egodex (113 sequences, ARKit hand tracking)
- **DexYCB**: CONVERTED/dex_ycb_cam_000 (1000 sequences, MANO fitted to depth)

## 1. Absolute Planarity Error (mm)

Distance of the fingertip from the plane defined by Knuckle-PIP-DIP.

| Finger     |                      HO-Cap                      |                      EgoDex                      |                      DexYCB                      |
|            |     Mean   Median      Std      Max   Frames |     Mean   Median      Std      Max   Frames |     Mean   Median      Std      Max   Frames |
|------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Index      |    2.097    1.541    1.987   17.014      676 |    0.075    0.044    0.110    2.429    59952 |    5.132    5.157    1.441   20.833    63548 |
| Middle     |    2.291    1.633    2.588   20.283      676 |    0.248    0.174    0.258    5.284    59952 |    1.546    1.466    0.990   18.151    63548 |
| Ring       |    2.800    1.926    2.737   19.034      676 |    0.144    0.106    0.166    3.856    59952 |    3.420    3.462    1.450   15.051    63548 |
| Little     |    2.464    1.805    2.304   15.580      676 |    0.112    0.062    0.164    4.746    59952 |    1.835    1.725    1.165   15.681    63548 |
| **ALL**    |    2.413    1.698    2.435   20.283     2704 |    0.144    0.085    0.194    5.284   239808 |    2.983    2.699    1.917   20.833   254192 |

## 2. Normalized Planarity Error (% of finger length)

| Finger     |                      HO-Cap                      |                      EgoDex                      |                      DexYCB                      |
|            |     Mean   Median      Std      Max   Frames |     Mean   Median      Std      Max   Frames |     Mean   Median      Std      Max   Frames |
|------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Index      |    2.309    1.697    2.188   18.736      676 |    0.090    0.053    0.131    2.632    59952 |    6.541    6.565    1.823   25.522    63548 |
| Middle     |    2.315    1.650    2.615   20.494      676 |    0.264    0.184    0.277    5.061    59952 |    1.943    1.814    1.272   23.192    63548 |
| Ring       |    3.023    2.080    2.956   20.552      676 |    0.169    0.124    0.192    4.130    59952 |    4.385    4.459    1.830   18.369    63548 |
| Little     |    3.248    2.380    3.038   20.544      676 |    0.163    0.090    0.238    7.184    59952 |    3.022    2.837    1.924   28.414    63548 |
| **ALL**    |    2.724    1.893    2.752   20.552     2704 |    0.171    0.103    0.225    7.184   239808 |    3.973    3.767    2.438   28.414   254192 |

## 3. Summary

| Metric | HO-Cap | EgoDex | DexYCB |
|--------|--------|--------|--------|
| Mean planarity (mm) | 2.413 | 0.144 | 2.983 |
| Median planarity (mm) | 1.698 | 0.085 | 2.699 |
| Mean planarity (%) | 2.724 | 0.171 | 3.973 |
| Median planarity (%) | 1.893 | 0.103 | 3.767 |
| Total frames | 2704 | 239808 | 254192 |

> Human finger DIP and PIP joints are 1-DOF hinges, so the 4 keypoints
> (Knuckle, IntermediateBase, IntermediateTip, Tip) should be coplanar.
> Lower planarity error = more anatomically plausible hand pose.
