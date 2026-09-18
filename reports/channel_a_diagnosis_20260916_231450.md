# Channel A Diagnosis Report
**Checkpoint:** `lightning_logs\version_5\checkpoints\best-epoch=41-val_loss=4.1895.ckpt`

## Per-Split RMSE Summary

| Split | N | RMSE | MAE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| train | 8,744 | 5.2627 | 4.0078 | -0.6468 | 0.6791 |
| val | 1,308 | 5.8411 | 4.5340 | 0.5427 | 0.0251 |
| test | 588 | 5.5905 | 4.1628 | 2.5688 | 0.1797 |

## Train — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 1,329 | 4.1382 | -4.4345 |
| 5-15_mps | 3,682 | 5.1822 | -2.7586 |
| 15-30_mps | 3,524 | 5.5717 | -0.7269 |
| 30-999_mps | 209 | 7.3011 | -158.0940 |

## Val — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 79 | 7.6996 | -22.5656 |
| 5-15_mps | 542 | 6.1018 | -5.7059 |
| 15-30_mps | 684 | 5.2962 | -2.7054 |
| 30-999_mps | 3 | 13.8784 | -3955.6292 |

## Test — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 88 | 5.8051 | -16.7120 |
| 5-15_mps | 296 | 6.3752 | -3.9064 |
| 15-30_mps | 204 | 4.0712 | -1.2026 |
