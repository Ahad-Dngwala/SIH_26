# models/channel_b_velocity/

[Layer 1 - Data & Models] MIP Section 4.3. The independent, gyro-free
cross-check for `models/channel_a_velocity/` - deliberately less
accurate; its job is disagreement detection, not primary accuracy (see
Section 5.4's trust-weighting logic, which is what actually consumes
that disagreement).

**Purpose:** independent forward-speed estimate with no dependency on
gyro-derived orientation, so it shares none of Channel A's failure
modes (misalignment error, gyro drift, orientation singularities).

**Input:** 4s window (longer than Channel A's 2s), 3 channels (accel
xyz only) - `(3, 400)` channels-first (MIP states shape as `(400, 3)`;
transposed in `dataset.py`).

**Output:** `(1,)` forward velocity scalar (m/s).

**Architecture:** CarSpeedNet-style - 4 Conv1d blocks (channels
32/64/128/128, kernel 7, stride 2 each - sequence length shrinks 16x
total), BatchNorm + ReLU after each conv - GlobalAvgPool - FC(128->32)
- ReLU - FC(32->1). See `model.py`.

**Loss:** MAE. **Optimizer:** Adam, lr 1e-3, cosine schedule. **Batch:**
128. **Epochs:** 90, early stop patience 12.

**Target:** velocity RMSE around 1.5-2.0 m/s - this is intentionally
worse than Channel A's 0.5 m/s target. Do not "improve" this model past
spec without checking with whoever owns Section 5 first: an
over-accurate Channel B changes the UKF's disagreement-threshold tuning
in Section 5.4.

**Status:** scaffolded, untrained. `model.py` is a first-draft
transcription of Section 4.3 - verify shapes with the Section 11.1
fixed-seed forward-pass test before trusting it. `dataset.py` is not
wired to real data yet (blocked on `data/processed/`, see Section 3).

**Current best metric:** none - no training run has happened yet.
