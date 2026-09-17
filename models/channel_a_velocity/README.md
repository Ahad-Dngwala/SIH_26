# models/channel_a_velocity/

[Layer 1 - Data & Models] MIP Section 4.2. The "expected" / primary
velocity channel - see `models/channel_b_velocity/README.md` for its
independent cross-check partner.

**Purpose:** correct accelerometer/gyro bias and scale error so
integrated velocity stays close to true velocity.

**Input:** 2s window, 6 channels (accel xyz, gyro xyz) - `(6, 200)`
channels-first (MIP states shape as `(200, 6)`; transposed in
`dataset.py`).

**Output:** `(1,)` forward velocity scalar (m/s), for the window's end
timestamp.

**Architecture:** Temporal Convolutional Network, 4 dilated causal
conv blocks (dilations 8/16/32/64 - MIP says 1/2/4/8, widened with the
same doubling pattern so the last-timestep readout below actually
covers the full window, see `model.py`'s docstring; same channels
32/64/64/128, kernel 3, residual connection per block, dropout 0.2
after each, no change to param count or per-position compute) -
last-timestep readout (position -1, matches the label's end-of-window
timing; was GlobalAvgPool) - FC(128->64) - ReLU - FC(64->1). See
`model.py`.

**Loss:** Huber (delta 1.0) - more robust to the Unsynchronised-split
augmentation data's labeling noise than plain MSE. **Optimizer:** Adam,
lr 1e-3, `ReduceLROnPlateau` (factor 0.5, patience 8) on val loss.
**Batch:** 64. **Epochs:** 100, early stop patience 15. **Gradient
clipping:** max norm 5.

**Target:** velocity RMSE under 0.5 m/s on held-out routes
(synchronized split).

**Important - NHC is NOT enforced in this network.** Non-holonomic
constraints (no lateral slip, no vertical velocity) are enforced
downstream in the UKF process model (Section 5), not here. Keep this a
pure regressor - don't add NHC logic to `model.py` or `train.py`.

**Status: trained, rejected at the acceptance gate, not wired in.**
Two architectures have been trained against real data - a Δv-label
variant and a 5s absolute-velocity variant. The 5s absolute model is
the better of the two and still failed the gate. Consequences of that
failure, all deliberate: no ONNX export was produced, `ukf.py` was not
modified, and `tools/benchmark_replay/config.yaml` keeps
`channel_a: dummy`. Full write-up in
[`../../final_report.md`](../../final_report.md).

**Current best metric (5s absolute-velocity model, held-out test):**

| Metric | Value | Baseline | Verdict |
|---|---|---|---|
| RMSE | 4.89 m/s | 6.17 m/s (zero-order hold) | pass |
| R² | 0.370 | 0.318 (earlier abs model) | pass |
| Session spread | Vta16 0.36 / Vta24 0.48 / Vta21 -0.27 | - | pass (not one-session) |
| Bias | **+1.54 m/s** | target ≈ 0 | **fail** |

The bias is the blocker, not the RMSE. A channel whose output gets
integrated into position cannot carry a persistent +1.5 m/s offset -
over a 60s blackout that alone is ~90 m of along-track error. Note
also that RMSE 4.89 m/s is nowhere near this section's own 0.5 m/s
target, so `FusionConfig.r_channel_a = 0.5` remains aspirational and
must not be used as if it were measured.

**Infrastructure status (unchanged, still true):** `dataset.py` loads
real windows from `data/processed/channel_a_velocity/` (Section 3
output). `model.py`'s architecture is a first-draft transcription of
Section 4.2, covered by the Section 11.1 shape/fixed-seed test in
`tests/`. `export.py` runs end to end (checkpoint -> ONNX -> int8,
float16 fallback on a failed accuracy check) - verified against
synthetic data in `models/_smoketest_train_export.py`, not yet against
a real checkpoint.
