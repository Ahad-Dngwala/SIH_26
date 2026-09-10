# models/road_signature/

[Layer 1 - Data & Models] MIP Section 4.4. Trained **per-corridor or
per-city, not globally** - shipped as a separate small "signature pack"
per region (a few MB each), downloaded/cached only for corridors the
user actually drives. This is Pillar 1 of the technical proposal
(`HLD/main.tex` Section 5, "Road-Signature Drift-Anchor").

**Purpose:** classify which pre-mapped road segment the vehicle is
currently on, from vibration signature alone, to use as a periodic
drift-reset anchor in the fusion core (Section 5.3, source 4).

**Input:** 2s window, 6 channels (accel + gyro) - `(6, 200)`
channels-first (MIP states shape as `(200, 6)`; transposed in
`dataset.py`).

**Output:** `(n_segments,)` logits (softmax at inference). `n_segments`
depends on corridor length - roughly 1 segment per 50-100m of mapped
route, so it is a per-corridor config value, not a fixed constant (see
`config.yaml`).

**Architecture:** 3 Conv1d blocks (channels 32/64/128, kernel 5,
MaxPool(2) after each) - GlobalAvgPool - FC(128->n_segments). See
`model.py`.

**Loss:** cross-entropy, label smoothing 0.1. **Optimizer:** Adam, lr
1e-3. **Batch:** 64. **Epochs:** 80, early stop patience 10.

**Target:** top-1 segment accuracy above 90% on held-out passes of the
same corridor. **Accuracy alone is not sufficient** (Section 4.7,
because segments are imbalanced by corridor length) - report macro-
averaged recall and macro-averaged F1 on every evaluation run, and
treat a class with recall well below the macro average as a real
problem even if overall accuracy looks fine.

**Deployment rule:** the fusion core only triggers a drift-reset when
softmax max probability exceeds 0.85 (Section 5.3) - this threshold is
consumed downstream by Layer 2, not enforced in this model itself.

**Status:** scaffolded, untrained, and blocked on real collected
corridor data (Section 3.1's "secondary" dataset - IO-VNBD alone has no
segment labels for an India-specific corridor). `model.py` is a
first-draft transcription of Section 4.4 - verify shapes with the
Section 11.1 fixed-seed test (which per Section 11.1 must also assert
on macro recall/F1 for this model specifically, not just output shape).

**Current best metric:** none - no training run has happened yet.
