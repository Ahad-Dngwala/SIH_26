# Channel A Diagnosis Report
**Checkpoint:** `lightning_logs\version_4\checkpoints\best-epoch=13-val_loss=4.3311.ckpt`

## Per-Split RMSE Summary

| Split | N | RMSE | MAE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| train | 8,744 | 5.4855 | 4.1741 | -1.0326 | 0.6514 |
| val | 1,308 | 5.7076 | 4.4698 | -0.0998 | 0.0692 |
| test | 588 | 4.8996 | 3.8209 | 1.5394 | 0.3699 |

## Train — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 1,329 | 4.3767 | -5.0789 |
| 5-15_mps | 3,682 | 4.5595 | -1.9096 |
| 15-30_mps | 3,524 | 6.1567 | -1.1086 |
| 30-999_mps | 209 | 11.4776 | -392.1667 |

## Val — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 79 | 7.9694 | -24.2460 |
| 5-15_mps | 542 | 5.5417 | -4.5314 |
| 15-30_mps | 684 | 5.4140 | -2.8720 |
| 30-999_mps | 3 | 17.3065 | -6151.6748 |

## Test — Error by Velocity Range

| Range (m/s) | N | RMSE | R² |
|---|---:|---:|---:|
| 0-5_mps | 88 | 5.3990 | -14.3205 |
| 5-15_mps | 296 | 5.2377 | -2.3117 |
| 15-30_mps | 204 | 4.1005 | -1.2344 |
