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

**Status:** scaffolded, untrained. `model.py` is a first-draft
transcription of Section 4.2 - verify shapes with the Section 11.1
fixed-seed forward-pass test before trusting it. `dataset.py` is not
wired to real data yet (blocked on `data/processed/`, see Section 3).

**Current best metric:** none - no training run has happened yet.
