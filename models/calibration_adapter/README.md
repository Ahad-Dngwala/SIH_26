# models/calibration_adapter/

[Layer 1 - Data & Models] MIP Section 4.5. Lowest priority of the five
models (Section 12: "calibration adapter, lowest priority of the five,
has a documented fallback in Section 14 if it runs out of time").
**Depends on `models/channel_a_velocity/` and `models/channel_b_velocity/`
already having first working versions** - per Section 12, start this
only once 4.1-4.4 are done, both Layer 1 people together.

**Purpose:** fast on-device recalibration of Channel A and Channel B
for a specific vehicle and mount, using a short GNSS-available
calibration drive (10-30s). This is Pillar 3 of the technical proposal
(`HLD/main.tex` Section 5) - the part most competing teams are likely
to skip.

**Method:** LoRA-style low-rank adapters, rank 4-8, inserted into the
final FC layer of Channel A and Channel B (`fc2` in both models' current
`model.py`). Base models are **frozen** - only adapter weights train.
Combined trainable parameter count across both adapters must stay under
50k (`model.py`'s `attach_calibration_adapters()` asserts this).

**Training:** on-device, plain SGD (not Adam - keeps optimizer state
small), lr 1e-4, 5-10 epochs over the short calibration session, batch
size 8-16 (limited by session duration). Label source: GNSS-derived
velocity during the calibration drive.

**Output:** a small adapter weight file (a few KB) saved per vehicle
profile, loaded alongside the base models at inference time.

**Important distinction - read before assuming this is Layer 3's
territory or vice versa.** `train.py` in this folder is a **Python-side
simulation** to validate the adapter mechanism (does the LoRA insertion
converge, does it stay under the param budget) using a desktop/cloud
GPU. The actual **on-device** fine-tuning described in Section 7.4
(user-facing calibration flow, running the fine-tune on the phone
itself) is Layer 3's job in Kotlin/ONNX Runtime Mobile's training APIs
- this folder does not implement that, it only proves the approach
works before Layer 3 ports it.

**Open integration question, not resolved by this scaffold:** exactly
how a LoRA adapter composes with an already-exported-to-ONNX frozen
base model at Android inference time (modify the ONNX graph vs. a
two-step inference with a lightweight custom op for the adapter delta)
is not specified in the MIP. `export.py` just saves the adapter's raw
weight matrices for now - resolve the actual on-device composition
mechanism with whoever ends up owning the Section 7.1/7.3 runtime loop.

**Fallback (Section 14):** if on-device LoRA fine-tuning proves too
slow or unstable on real hardware, fall back to a simpler per-vehicle
scale/bias correction (two scalar parameters per channel, fit with a
closed-form least-squares solve instead of gradient-based fine-tuning).

**Status:** scaffolded, blocked on Channel A/B having trained weights.

**Current best metric:** none - no training run has happened yet.
