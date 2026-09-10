# models/alignment_net/

[Layer 1 - Data & Models] MIP Section 4.1.

**Purpose:** estimate the phone's pitch/roll/yaw offset relative to the
vehicle's direction of travel.

**Input:** 2s window, 9 channels (accel xyz, gyro xyz, mag xyz) - `(9,
200)` channels-first as consumed by the model (MIP states the shape as
`(200, 9)`; the transpose to channels-first happens in `dataset.py`,
see its docstring).

**Output:** `(4,)` - `[pitch, roll, sin(yaw), cos(yaw)]`. Reconstruct
yaw with `AlignmentNet.to_angles()` (atan2), which also gives you the
`(3,)` `[pitch, roll, yaw]` the MIP describes as the logical output.

**Spec discrepancy, flagged rather than silently resolved:** Section
4.1's architecture line says the final FC layer is `FC(32 to 3)`, but
its loss line requires predicting 4 values (`pitch, roll, sin(yaw),
cos(yaw)`) for the wraparound-safe angle trick to work. `model.py`
follows the loss line (4 outputs) since that's the only way sin/cos-yaw
is representable - get the MIP itself corrected, don't let this README
be the only record of the discrepancy.

**Architecture:** Conv1d(9->32,k5,s1) - Conv1d(32->64,k5,s2) -
Conv1d(64->64,k3,s2), ReLU after each - GlobalAvgPool - FC(64->32) -
ReLU - FC(32->4). See `model.py`.

**Loss:** MSE (on the 4 raw outputs). **Optimizer:** Adam, lr 1e-3,
cosine annealing over training. **Batch:** 128. **Epochs:** 60, early
stop patience 10.

**Target:** mean angular error under 3 degrees on held-out routes.

**Status:** scaffolded, untrained. `model.py`'s architecture is a
first-draft transcription of Section 4.1 - it has not run against real
data. Verify shapes with the Section 11.1 fixed-seed forward-pass test
before trusting it. `dataset.py` is not wired to real data yet
(blocked on `data/processed/`, see Section 3).

**Current best metric:** none - no training run has happened yet.
