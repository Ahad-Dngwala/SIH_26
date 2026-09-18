# Channel A Diagnosis Report
**Checkpoint:** `lightning_logs/version_1/checkpoints/best-epoch=10-val_loss=3.9900.ckpt`

## Per-Split RMSE Summary

| Split | N | RMSE | MAE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| train | 21,901 | 4.2469 | 3.0663 | 0.2400 | 0.7910 |
| val | 3,280 | 5.6234 | 4.4549 | -0.4168 | 0.0994 |
| test | 1,478 | 5.0943 | 3.9376 | 2.2038 | 0.3183 |

## Train — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 3,333 | 4.3188 | -4.9619 |
| 5-15_mps | 9,236 | 3.9042 | -1.1346 |
| 15-30_mps | 8,815 | 4.3207 | -0.0429 |
| 30-999_mps | 517 | 7.2930 | -157.1974 |

## Val — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 201 | 7.2191 | -18.4263 |
| 5-15_mps | 1,353 | 5.1736 | -3.7435 |
| 15-30_mps | 1,719 | 5.6690 | -3.2176 |
| 30-999_mps | 7 | 15.9868 | -1378.1260 |

## Test — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 231 | 6.3936 | -16.7009 |
| 5-15_mps | 725 | 5.3160 | -2.5956 |
| 15-30_mps | 522 | 4.0175 | -1.1651 |
