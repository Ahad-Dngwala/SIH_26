# Handoff: Remaining Models, the Zero-Order-Hold Baseline, and How This All Converges

Written for whichever Claude instance (or human) picks this repo back up
next. Companion to `Master_Implementation_Plan.md`, not a replacement for
it - read that in full first if you haven't, especially Section 4 (all
six subsections), Section 5 (fusion), Section 12 (task breakdown), and
Section 13 (checkpoints). This document assumes you've also read the
current state: `channel_a_velocity` has a working, trained-and-debugged
architecture (three commits on top of the original scaffold - see
`git log models/channel_a_velocity/`), and real IO-VNBD data has been
downloaded and run through the `00-05` pipeline at least once.

If anything below conflicts with what's actually in the repo or in
`data/processed/*/README.md`, trust the repo - this doc is a snapshot of
reasoning at a point in time, not a live source of truth.

---

## 0. Do this first, before training anything else

**The zero-order-hold baseline.** Before spending more compute on
`channel_a_velocity` or starting any other model, write a small
throwaway script (not a repo deliverable, doesn't need `train.py`/
`model.py`/tests - just something to run once and read the output of)
that computes, on the val split:

> "If I predict the last **real** (non-interpolated) `v_velocity_kmh`
> sample the vehicle GPS actually reported, and just hold that value
> constant as my 'prediction' for the window's end - with no model
> at all - what RMSE do I get against the (linearly-interpolated)
> 100Hz label grid `03_window.py::build_channel_a` currently uses?"

Concretely: for each window, find the two real 10Hz VBOX samples that
straddle the window's end timestamp, and instead of comparing the
model's prediction to the interpolated label at that instant, compare
the *earlier* real sample's value to the *interpolated* label at that
instant. That gap is the error a perfect model would still show, purely
from the label's own update-rate granularity plus whatever residual
alignment noise is baked into `data/raw/_aligned/alignment_report.json`
for that session. `data/scripts/iovnbd_common.py` and `02_align.py`
already have all the merged/aligned dataframes and per-session corr/lag
data you need for this - don't re-derive alignment from scratch.

**Why this matters more than another training run:** it directly
answers the question everyone's been circling around at
`models/channel_a_velocity/README.md`'s target line (RMSE < 0.5 m/s).
Three outcomes, three different next moves:

- **Baseline RMSE is already well above 0.5 m/s** (say, 2+ m/s): the
  0.5 m/s target is not reachable by *any* model trained against this
  particular label pipeline, full stop - not a modeling problem, an
  information-availability problem (see Section 4 below on the 10Hz
  Nyquist ceiling). At that point the right move isn't more
  architecture tuning, it's deciding what Channel A's *realistic* job
  is (Section 4.2's own note already hints at this: "NHC is enforced
  downstream in the UKF process model... keep this network a pure
  regressor" - Channel A was never meant to single-handedly deliver the
  system's final accuracy) and getting a revised, defensible target
  written back into the MIP.
- **Baseline RMSE is meaningfully below the model's current ~5+ m/s**
  but still above 0.5: there's real headroom, worth continuing to
  chase via the model side (the alignment quality gate and receptive
  field fixes already landed should help; per-session loss breakdown,
  described in Section 2 below, tells you how much more).
- **Baseline RMSE is close to what the trained model already gets:**
  the model has basically converged to the noise floor already, and
  no further architecture or training change will move the number -
  the remaining gap to 0.5 m/s is a data problem (alignment quality,
  native sample rate), not a modeling one.

Do this on `channel_a_velocity`'s val split specifically, and also
re-run it once the alignment-quality-gate fix has actually been used to
regenerate `data/processed/` (it hasn't yet as of this doc - the fix is
committed to `03_window.py` but needs a real re-run of `00-05` against
the actual downloaded IO-VNBD data to take effect).

---

## 1. Current status, all five models + the two non-ML subsystems

| Component | Section | Status | Blocked on |
|---|---|---|---|
| `alignment_net` | 4.1 | Scaffolded, label-derivation implemented (`build_alignment_net` in `03_window.py`), **not yet trained** | Nothing - ready to window/train once `00-05` has been re-run with the alignment-gate fix |
| `channel_a_velocity` | 4.2 | Trained, debugged (3 fix commits), val still not hitting target as of the last run | The zero-order-hold baseline (Section 0) - answers whether more tuning is worth it |
| `channel_b_velocity` | 4.3 | Scaffolded (`model.py`/`train.py`/`config.yaml` exist), **`build_channel_b` label function does not exist yet** - `03_window.py`'s own module docstring and `WINDOW_CONFIGS` entry both say "labeling TODO" | Data-side work (Section 2 below), not model work |
| `road_signature` | 4.4 | Scaffolded, **`build_road_signature` does not exist yet**, and per Section 12 needs real collected corridor data, not just IO-VNBD | Data-side work + the secondary dataset (Section 3.1) - see Section 3 below |
| `calibration_adapter` | 4.5 | Scaffolded only, explicitly lowest priority | Channel A + Channel B both needing first working versions, plus a real vehicle calibration-drive recording (can't come from IO-VNBD at all) |
| `fusion_core` (UKF) | 5 | Check `fusion_core/python_prototype/README.md` for current status - per Section 5.5, this does **not** need to wait on any of the above, it's buildable/testable against synthetic data right now if it isn't already underway |
| `tools/benchmark_replay` | 9 | Check `tools/benchmark_replay/README.md` - there's already a full standalone handoff doc for this at `docs/HANDOFF_benchmark_replay_tool.md`. Per the MIP's own repeated emphasis, this is "the single most important thing to have working early" and should be running on dummy Channel A/B/road-signature outputs **in parallel** with everything above, not after it |

---

## 2. Channel B (`channel_b_velocity`) - what to build, and what to check up front instead of after

The label function doesn't exist yet. Before writing it, apply the
lesson `channel_a_velocity` already paid for the hard way:

**Check the receptive-field-vs-label-timing question *before* training,
not after.** Channel B's architecture (Section 4.3) is 4 Conv1d blocks,
kernel 7, stride 2 each, over a 400-sample (4s) window - a downsampling
CNN, not a dilated causal TCN like Channel A, so the receptive-field math
is different (stride-2 conv shrinks the sequence length itself, it
doesn't just widen what each position can see the way dilation does).
Work out, before writing `train.py`'s loop:

1. Does `models/channel_b_velocity/model.py`'s final `GlobalAvgPool`
   read out a single window-level scalar - and if so, is Channel B's
   label meant to be a single instant (like Channel A's "velocity at
   window's end") or a window-representative value? Section 4.3 just
   says "forward velocity scalar (m/s)" without specifying timing the
   way 4.2 does ("for the window's end timestamp") - that's worth
   resolving explicitly (check with whoever owns the MIP, or default to
   matching Channel A's convention - last-timestep - for consistency,
   since Section 5.3 feeds both channels' outputs into the same UKF
   update step and they should mean the same thing timing-wise).
2. Given the answer to (1), does the actual architecture (4 strided
   conv blocks with a real, non-dilated receptive field that shrinks
   as sequence length shrinks) get sufficient context to that readout
   point? Compute it before training, the way this doc's Section 0
   should have been done before `channel_a_velocity`'s first training
   run, not after two rounds of "why isn't this converging."
3. Same alignment-quality-gate question as Channel A: `v_velocity_kmh`
   is the label source again (per Section 3.3, "Channel A and Channel
   B: forward velocity scalar per window, taken from the V (vehicle
   ground truth) stream") - so reuse the exact same quality gate logic
   `03_window.py` already has for Channel A rather than writing a
   second, possibly-inconsistent version for `build_channel_b`.

Once `build_channel_b` exists and is added to `WINDOW_CONFIGS`/
`BUILDERS`, run `04_normalize.py --model channel_b_velocity` and
`05_split.py` the same way the earlier channel_a_velocity run did, then
train per Section 4.3's spec (MAE loss, Adam lr 1e-3 cosine, batch 128,
90 epochs, early-stop patience 12). Remember Channel B's own target
(1.5-2.0 m/s) is *deliberately* worse than Channel A's - its
`README.md` already warns not to "improve this model past spec without
checking with whoever owns Section 5 first," since the UKF's
disagreement-threshold tuning (Section 5.4) assumes Channel B is the
noisier of the two.

---

## 3. Road-signature classifier (`road_signature`) - the one that's genuinely blocked on new data

Two separate blockers, and they're not the same blocker:

**(a) Label derivation doesn't exist.** Per Section 3.3: "segment ID,
derived by dividing each corridor's route into fixed-length arc segments
(e.g. 50-100 meters each) using the GNSS ground truth positions, then
labeling every window with the segment its GNSS timestamp falls into."
This is buildable directly from IO-VNBD's own V-stream lat/lon columns
(`v_lat`/`v_lon` per `iovnbd_common.py`'s `V_COLUMNS`) without needing
Section 6's full OSM/HMM map-matching infrastructure - a simple arc-
length segmentation of each session's own GPS track is enough to
produce *a* `build_road_signature` and get the classifier's training
loop proven end to end, even before real collected corridor data
exists.

**(b) The real target needs the team's own collected corridor data.**
Per Section 12: "The road-signature classifier needs real collected
corridor data (Section 3.1's 'secondary' dataset), so coordinate
collection-drive timing with whoever is doing the field/vehicle
sessions rather than blocking on it silently." This is because the
whole point of this model is recognizing *specific, repeatedly-driven*
corridors (Section 4.4: "trained per-corridor or per-city... shipped as
a separate small 'signature pack' per region") - IO-VNBD's routes,
recorded once each in the UK/Nigeria/France, aren't the corridors this
system will actually run on. Section 3.1 already has this on the plan:

> "Secondary (collected by the team before the finale): short
> GNSS-available loops on at least one two-wheeler, one auto-rickshaw,
> and one car, each 10-20 minutes... This is what seeds the
> road-signature packs and the calibration adapter, since IO-VNBD has
> no Indian two-wheeler data."

**My own recommendation, not literal MIP text:** given this is now
looking like a real, concrete two-model bottleneck (both
`road_signature` and `calibration_adapter` are stuck without it) rather
than a nice-to-have, I'd move this collection drive earlier than "before
the finale" - even one short loop on one vehicle, done soon, unblocks
proving the `build_road_signature` pipeline works on real data and gives
`calibration_adapter` something to test against, well before either
becomes the thing standing between the team and a checkpoint deadline.

---

## 4. The 10Hz-native-sampling ceiling applies to every model here, not just Channel A

Worth restating explicitly since it's easy to re-litigate per-model:
IO-VNBD's own paper (`README_1.md`, uploaded and read as part of this
investigation) states both the smartphone IMU and the vehicle's VBOX
CAN-bus/GPS were recorded at 10Hz natively, then linearly interpolated
up to the pipeline's 100Hz working grid by `01_resample.py`. That's a
Nyquist ceiling of ~5Hz on anything either stream captured - real
vehicle dynamics faster than that (the paper's own words: "vehicular
vibrations interfere with measurement precision") were aliased into the
10Hz recording before this pipeline ever saw the data, and no amount of
downstream preprocessing, architecture width, or receptive field
tuning recovers information that was never captured. This applies
identically to `alignment_net`, `channel_b_velocity`, and
`road_signature` - all three also consume the same 10Hz-native
smartphone IMU stream as their input. Don't rediscover this per model;
if a future training run plateaus above a model's target the way
Channel A's did, check the zero-order-hold-style baseline for that
model before assuming it's an architecture problem.

One caveat specific to `alignment_net`: unlike Channel A's forward
velocity (a fast-changing quantity where end-of-window timing matters a
lot), `build_alignment_net`'s pitch/roll labels are already a
**window-average** (`g_windows.mean(axis=1)`), and the yaw offset is a
property of the phone's mount angle relative to the vehicle, which
doesn't change quickly session-to-session. So `alignment_net`'s
`GlobalAvgPool` is probably *not* the same architecture/label mismatch
Channel A had - don't blindly port the last-timestep fix over without
checking whether the label it's predicting is actually a point-in-time
quantity first. Same question applies to `road_signature`'s segment-ID
classification (also a window-level "which segment was I on," not an
instantaneous value) - GlobalAvgPool is likely fine there too. This
mismatch was specifically a Channel A (and possibly Channel B, see
Section 2) problem, not a universal one.

---

## 5. Clarifying the different accuracy targets floating around (easy to conflate)

Four different numbers, four different things, worth keeping straight:

- **Channel A** (`channel_a_velocity`, Section 4.2): velocity RMSE
  **< 0.5 m/s** on held-out routes.
- **Channel B** (`channel_b_velocity`, Section 4.3): velocity RMSE
  **1.5-2.0 m/s**, deliberately worse - its job is disagreement
  detection (Section 5.4), not accuracy.
- **Alignment net** (`alignment_net`, Section 4.1): mean angular error
  **under 3 degrees** on held-out routes. (There's no "0.5 degree"
  target anywhere in the MIP - if that number came up in conversation,
  it's most likely a mix-up between this 3-degree target and Channel
  A's 0.5 m/s target. Worth a word to whoever's tracking targets so it
  doesn't propagate further.)
- **Road-signature** (`road_signature`, Section 4.4): top-1 segment
  accuracy **above 90%**, plus macro-averaged recall and macro-averaged
  F1 (Section 4.7 - accuracy alone isn't sufficient here).
- **System-level** (the actual competition benchmark, per
  `HLD/main.tex` Section 1, also referenced in
  `docs/HANDOFF_benchmark_replay_tool.md`): **under 10% positional
  drift** during a GNSS blackout (e.g. under 5m over 50m/1min, or under
  100m over 1km at 60km/h). This is the number the benchmark replay
  tool (Section 9) actually measures, and it's a *system*-level number
  - it depends on Channel A, Channel B, road-signature, and the UKF's
  fusion/trust-weighting logic all working together, not on any one
  model hitting its own target in isolation.

None of the per-model targets above are the same thing as the
system-level drift number - a model can miss its own target and the
system can still hit 10% drift if the UKF's fusion logic compensates
(that's the whole point of running four independent evidence sources
into one filter, Section 5.3), and conversely a model can hit its own
target and the system can still fail if the fusion tuning is wrong.
Don't treat "Channel A isn't at 0.5 m/s yet" as evidence the system
will fail the actual benchmark - that's an empirical question the
benchmark replay tool answers, not something to infer from one
channel's isolated RMSE.

---

## 6. How this is all supposed to converge - the system view

Reading Sections 5, 6, 7/8, and 9 together, here's the shape of the
whole thing once every piece exists:

**Per inference cycle (Section 7.3), on-device:**

1. Pull the latest 2s (or 4s, for Channel B) window from the sensor
   ring buffer.
2. `alignment_net` runs occasionally (mount angle doesn't change fast)
   to know how the phone is oriented relative to the vehicle.
3. `channel_a_velocity` and `channel_b_velocity` both run every cycle,
   independently, both estimating forward speed from the same phone
   IMU stream but through architecturally different paths (Channel A:
   gyro-aware TCN; Channel B: accelerometer-only CNN) - the whole point
   is that they're likely to fail in *different* ways (misalignment/
   gyro-drift for A, no cross-check for B alone).
4. The disagreement between A and B's estimates (Section 5.4) is
   computed every cycle. Large disagreement inflates Channel A's
   measurement noise for that cycle - the filter automatically leans
   more on Channel B and the road-signature anchor when Channel A looks
   untrustworthy, without needing a learned "which channel is right"
   classifier.
5. `road_signature` runs periodically (not every cycle - Section 7.3),
   checking if the current vibration signature confidently (>0.85)
   matches a known segment of a pre-mapped corridor. When it does, that
   segment's midpoint becomes a soft position correction, independent
   of both velocity channels.
6. GNSS (when available), Channel A, Channel B, and the road-signature
   anchor (when triggered) all feed into the same 7-state UKF
   (`fusion_core`, Section 5.1-5.2) as four separate measurement
   sources with their own tuned noise covariances - not four separate
   filters, one filter with four update sources.
7. The UKF's fused position feeds `map_matching`'s HMM (Section 6),
   which snaps it onto the actual road graph for the UI.
8. `calibration_adapter` sits alongside 3-4, quietly correcting Channel
   A/B's final-layer weights per the specific vehicle/mount currently
   in use, via a short one-time calibration drive rather than needing
   IO-VNBD to cover every possible vehicle.

**The critical thing this repo's own docs already say clearly, worth
repeating here:** none of Steps 4-8 need to wait for Steps 2-3 (or
Step 5's real model) to be "finished" in the sense of hitting their
target metric. `fusion_core`'s Python prototype and
`tools/benchmark_replay`'s skeleton are both explicitly meant to be
built and tested against synthetic/dummy data *right now*, in parallel
with Layer 1's model training, per Section 5.5's opening line and the
whole of `docs/HANDOFF_benchmark_replay_tool.md`. If that isn't already
underway, it's a bigger schedule risk than any single model's RMSE gap
- it's the only thing that tells you whether the *system* meets the
actual benchmark, independent of which individual model targets get
hit exactly.

---

## 7. Suggested order from here

1. Zero-order-hold baseline for Channel A (Section 0) - cheap, fast,
   answers whether continuing to chase 0.5 m/s on this model is
   worthwhile before sinking more time in.
2. Re-run `00-05` with the alignment-quality-gate fix actually applied
   (it's committed but hasn't been used to regenerate real data yet),
   retrain Channel A, compare against both the baseline from (1) and
   the two previous `metrics.csv` runs.
3. `alignment_net` - ready to go as-is, no blockers, and per Section
   12's own stated priority it's next regardless of how (1)/(2)
   resolve.
4. `channel_b_velocity` - needs `build_channel_b` written first
   (Section 2 above), with the receptive-field-vs-label-timing check
   done *before* the first training run this time, not after.
5. In parallel with 1-4, if it isn't already happening: get
   `fusion_core/python_prototype/` and `tools/benchmark_replay/`
   moving on synthetic/dummy data - this is Layer 2's job per Section
   12 and doesn't block on any of the above, but it's easy for it to
   quietly slip while all the visible activity is on Layer 1's models.
6. A short real-world collection drive (Section 3.1's secondary
   dataset) - even one vehicle, one short loop - sooner than "before
   the finale," since `road_signature` and `calibration_adapter` are
   both genuinely stuck without it, not just improved by it.
7. `road_signature`, once (6) produces something to point
   `build_road_signature` at (or, as an interim step, against IO-VNBD's
   own GPS tracks per Section 3 above, to prove the pipeline before
   real corridor data exists).
8. `calibration_adapter` - last, per Section 12's own stated priority,
   once Channel A/B have working versions and a real calibration-drive
   recording exists.