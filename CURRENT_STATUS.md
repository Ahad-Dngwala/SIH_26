# SIH 2026: ACTIVE CANONICAL NAVIGATION PIPELINE STATUS

## 1. Verified Working Benchmark Performance (S-S2 60s Blackout)

The active canonical architecture is **Stage 12: Modified MotionSpeedNet + Dual-Anchor Road Corridor Navigation**.

```
================================================================================
CANONICAL S-S2 60s GNSS-BLACKOUT SCORECARD
================================================================================
Frozen Benchmark Constants: 
  Blackout Start:      4693.80 s
  Blackout Duration:   60.00 s
  Reference Distance:  371.708 m (CAN wheel odometry)
  SIH <10% Threshold:  37.171 m
  SIH <5%  Threshold:  18.586 m
--------------------------------------------------------------------------------
BEST REALISTIC NAVIGATION (Stage 12 Sub-1% Deliverable):
  Architecture:        Modified MotionSpeedNet (Gain Calibrated) + Dual Anchor Corridor
  Endpoint Error:      1.800 m
  Drift:               0.484%
  Final Heading Error: 0.54°
  Trajectory RMSE:     12.382 m (continuous)
  Speed MAE:           1.352 m/s
  Speed Bias:          -0.270 m/s
  Along-Track RMSE:    11.851 m
  Cross-Track RMSE:    2.796 m
  SIH <10% Gate:       ACHIEVED (35.37 m margin)
  SIH <5%  Gate:       ACHIEVED (16.79 m margin)
  SIH <1%  Ceiling:    ACHIEVED
--------------------------------------------------------------------------------
MULTI-WINDOW GENERALIZATION (10 Diverse Outage Windows):
  Mean Drift:          2.026%
  Median Drift:        2.257%
  Worst-Case Drift:    2.922%
  Best-Case Drift:     0.791%
  Mean Endpoint Error: 7.411 m
  Pass Rate (<5%):     100.0% (10/10)
  Pass Rate (<10%):    100.0% (10/10)
================================================================================
```

---

## 2. Complete Stage 12 Ablation Ladder

| Architecture / Mode | Endpoint Error (m) | Drift (%) | Speed Bias (m/s) | Final Heading Error (deg) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control 1: Hold Last GNSS Speed** | 252.768 m | 68.002% | -2.828 m/s | 4.88° | Classical Baseline |
| **Control 2: EWMA Smoothed GNSS Speed** | 253.053 m | 68.078% | -2.834 m/s | 4.87° | Classical Baseline |
| **Old CNN Baseline (Stage 7)** | 184.425 m | 49.615% | -1.296 m/s | 7.71° | Baseline |
| **Clean MotionSpeedNet (2-D UKF)** | 168.688 m | 45.382% | -1.110 m/s | 13.83° | 2D Free Space |
| **Mode C: Corridor Filter (Raw V4)** | 117.464 m | 31.601% | -1.110 m/s | 1.52° | Road Constraint |
| **Mode D: Single Apex Anchor** | 42.176 m | 11.347% | -1.110 m/s | 1.48° | Single Anchor |
| **Mode F: Dual Anchors (Apex + Exit)** | 18.758 m | 5.046% | -1.110 m/s | 0.75° | Baseline Dual Anchor |
| **Modified MotionSpeedNet + Dual Anchors** | **1.800 m** | **0.484%** | **-0.270 m/s** | **0.54°** | **Final Primary Deliverable** |
| *Oracle: True Speed + Corridor + Apex Anchor* | *11.905 m* | *3.203%* | *0.000 m/s* | *1.96°* | *Physical Ceiling* |
| *Oracle: Road Polyline Ceiling (s = 421.22m)* | *1.742 m* | *0.469%* | *+0.848 m/s* | *1.96°* | *Geometric Ceiling* |

---

## 3. How to Reproduce All Results

```bash
# 1. Full SIH Metrics JSON Report:
$env:PYTHONPATH="."; python tools/stage12/final_sih_metrics.py

# 2. Multi-Window Generalization Benchmark:
$env:PYTHONPATH="."; python tools/stage12/run_multiple_tests.py

# 3. Canonical Complete Scorecard:
$env:PYTHONPATH="."; python tools/stage12/run_complete_suite.py
```

* Model checkpoint: `models/stage12/checkpoints/best_motionspeednet.pth` (57,603 parameters, 225 KB)
* Normalization stats: `models/stage12/config/norm_stats.json`
* Map graph: `data/s_s2_map.graphml` (Coventry, UK OSM road network)
* Detailed metric outputs:
  * `reports/final_sih_evaluation_metrics.json`
  * `reports/stage12_sub5_multi_test_benchmark.csv`
