# Channel A Diagnosis Report
**Checkpoint:** `lightning_logs/version_2/checkpoints/best-epoch=00-val_loss=0.1603.ckpt`

## Per-Split RMSE Summary

| Split | N | RMSE | MAE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| train | 21,901 | 0.6799 | 0.4168 | 0.0074 | 0.0430 |
| val | 3,280 | 0.7124 | 0.4887 | 0.0228 | 0.0126 |
| test | 1,478 | 0.7156 | 0.4611 | 0.0206 | -0.0309 |

## Train — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 11,317 | 0.5563 | -0.7844 |
| 5-15_mps | 11 | 8.3349 | -7.2475 |

## Val — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 1,695 | 0.6066 | -1.1830 |

## Test — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 802 | 0.5794 | -0.9605 |
