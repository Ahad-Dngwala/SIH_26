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
conv blocks (dilations 1/2/4/8, channels 32/64/64/128, kernel 3,
residual connection per block, dropout 0.2 after each) - GlobalAvgPool
- FC(128->64) - ReLU - FC(64->1). See `model.py`.

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

**Status:** wired, not yet trained. `dataset.py` loads real windows
from `data/processed/channel_a_velocity/` (Section 3 output).
`model.py`'s architecture is a first-draft transcription of Section
4.2, covered by the Section 11.1 shape/fixed-seed test in `tests/`.
`export.py` runs end to end (checkpoint -> ONNX -> int8, float16
fallback on a failed accuracy check) - verified against synthetic data
in `models/_smoketest_train_export.py`, not yet against a real
checkpoint. No training run against real data has happened yet.

**Current best metric:** none - no training run has happened yet.
