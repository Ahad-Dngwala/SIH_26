# Channel A evaluation report

Checkpoint: `C:\Users\katha\Hackathons\SIH 26\lightning_logs\version_1\checkpoints\best-epoch=10-val_loss=3.9900.ckpt`  
Device: `cuda`  
Parameters: `62849`  

| Split | RMSE (m/s) | MAE (m/s) | Median AE | P90 AE | P95 AE | Max AE | R2 | Baseline RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4.2469 | 3.0663 | 2.1825 | 7.0580 | 9.1126 | 23.2004 | 0.7910 | 9.2894 |
| val | 5.6234 | 4.4549 | 3.8295 | 9.2850 | 11.1565 | 18.7867 | 0.0994 | 5.9262 |
| test | 5.0943 | 3.9376 | 3.2297 | 8.3576 | 10.2904 | 20.5964 | 0.3183 | 6.8856 |

Target: Channel A held-out RMSE < 0.5 m/s.

Teammate comparison: upstream implementation is present, but no teammate checkpoint/metrics report is available; an apples-to-apples performance comparison is therefore pending.

## ONNX deployment metrics

| Export | Val RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| channel_a_fp32 | 5.6235 | 5.0946 | 3.9378 | 0.3182 |
| channel_a | 5.6268 | 5.0754 | 3.9233 | 0.3234 |
