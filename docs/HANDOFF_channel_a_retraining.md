# HANDOFF: Channel A retraining — what to do and what not to

**Read `final_report.md` first.** Channel A has been trained twice and
rejected twice. This document is about not making it three times for
the same reason.

**Status of the thing you are about to change:** nothing in the repo
depends on Channel A today. `config.yaml` has `channel_a: dummy`,
`ukf.py` has never been touched by a training result, and no ONNX
export exists. That is a deliberate state, not neglect — it means you
can experiment freely without breaking the demo, and it means wiring
anything in is a separate, explicit decision.

---

## 0. Set the acceptance gate before you train, not after

This is the single most important instruction in this document, and it
is first because it is the one most likely to be skipped under time
pressure.

Write the gate down — in a file, committed — before you run training.
After you have a checkpoint and a number, every threshold becomes
negotiable, and "0.9 m/s bias is basically near-zero" is a sentence
that gets said at 2am by someone who has been staring at a loss curve
for six hours. The previous two attempts got this right and it is why
the repo is still honest. Keep it.

A concrete gate, adapted from the one `final_report.md` used, with one
addition:

| Criterion | Threshold | Why |
|---|---|---|
| Test RMSE | beats zero-order-hold baseline (6.17 m/s) | otherwise the model is worse than a constant |
| Test R² | clearly positive, beats 0.370 | must improve on the last rejected attempt |
| Session spread | positive skill on ≥2 held-out sessions | guards against one lucky session carrying it |
| **Test bias** | **abs(bias) < 0.3 m/s** | see section 1 |
| **End-to-end drift** | **beats Channel P on `varying_speed`** | see section 2 |

The bottom two rows are the ones that matter. The top three were all
passed by the model that got rejected.

---

## 1. Understand why bias, specifically, is fatal

RMSE 4.89 m/s sounds bad and bias +1.54 m/s sounds small. For this
application it is the other way round.

Channel A's output is integrated into position. Zero-mean error of any
magnitude partially cancels over a blackout — it random-walks, so
position error grows with √t. A *constant* bias does not cancel. It
integrates directly:

```
position error from bias = bias × blackout_duration
                         = 1.54 m/s × 60 s
                         = 92 m
```

That is 92 m of pure along-track error over a 60 s blackout, before any
other error source contributes anything. The blackout in
`config.yaml` covers about 1 km of travel, so a +1.54 m/s bias is
roughly 9% drift on its own — worse than running with **no velocity
channel at all**, which currently measures 6.78%.

This is the whole reason the acceptance gate has a bias row that the
RMSE row cannot override. A high-RMSE, zero-bias channel is useful. A
low-RMSE, biased channel is actively harmful. Do not trade the second
for the first because the aggregate metric improved.

---

## 2. The bar moved — Channel P is now the thing to beat

Since `final_report.md` was written, the repo gained Channel P
(`fusion_core/python_prototype/physics_speed.py`): a classical
integrate-the-accelerometer-and-reseed-from-GNSS estimator. No model, no
training, no dataset, about 80 lines of actual logic.

Measured blackout drift (`python -m tools.benchmark_replay.compare_configs`):

| Configuration | constant_turn | varying_speed |
|---|---|---|
| no velocity channel | 6.78% | 10.35% |
| **Channel P** | **4.13%** | **3.91%** |
| dummy A+B (noised ground truth — not achievable) | 1.48% | 1.58% |

**A trained Channel A that does not beat 3.91% on `varying_speed` is not
worth wiring in,** because Channel P already gets that for free and
cannot be accused of overfitting to anything. This reframes the whole
exercise: you are not trying to beat a zero-order-hold baseline any
more, you are trying to beat a working classical estimator. That is a
much harder and much more honest target, and it is the one a judge will
ask about.

It also means a genuinely acceptable outcome of your experiments is
*"Channel A does not beat Channel P, so we ship Channel P."* That is a
result, not a failure. Write it up and move on.

Evaluate on `varying_speed`, not `constant_turn`. `fx` is a
constant-speed coast model, so on a constant-speed route the coast is
already nearly right and a velocity channel has almost nothing to
contribute — benchmarking a velocity source there understates it badly.

---

## 3. Before touching the architecture, suspect the data

Two different architectures — Δv labels and a 5 s absolute-velocity
window — both produced a persistent positive bias. When two unrelated
model families fail the same way, the model family is usually not the
problem. A systematic offset in the labels or the preprocessing is.

Spend the first block of time here. It is unglamorous and it is where
the bug probably is.

**Check, in this order:**

1. **Label/IMU time alignment.** `02_align.py` does cross-correlation
   alignment with a fallback path, and `03_window.py` has a
   boundary-lag gate. A constant lag between IMU and the velocity label
   produces exactly a constant bias when the vehicle is accelerating on
   average. Plot a few windows' IMU against their label and look at
   them with your own eyes. Do not trust the correlation score alone.
2. **Sign and units.** A velocity label in km/h read as m/s, or an
   accelerometer axis flipped in some sessions and not others, both
   show up as a stable offset. Check per-session, not in aggregate — an
   aggregate check averages the bug away.
3. **Normalization statistics.** `norm_stats.json` must be computed on
   the **training split only** and applied unchanged to val and test. If
   it was computed over everything, the test metrics are optimistic and
   the bias may be an artifact of a shifted mean.
4. **Split leakage.** The split must be by **session/route**, never by
   window. Windows overlap, so a random window split puts near-identical
   samples in train and test, and the resulting test RMSE is fiction.
   `05_split.py` cross-checks quality-gate exclusions against the
   manifest — confirm it is actually splitting on session.
5. **Label construction itself.** If the velocity label is derived from
   GNSS, check what happens during GNSS gaps in the source data. An
   interpolated or held-last-value label across a gap teaches the model
   something untrue.

If any of these turn up something, fix it and retrain the *existing*
architecture before trying a new one. A data fix that removes the bias
is worth more than any architecture change.

---

## 4. If the data is clean, then attack the bias directly

Only after section 3. In rough order of effort-to-payoff:

**Predict something bias-immune.** The most robust framing is to have
the model predict a *correction* to a physics estimate rather than the
velocity itself. Channel P already produces a forward-speed estimate;
train the model on `residual = true_speed − channel_p_speed`. The
physics carries the absolute scale, the model only has to learn the
part physics gets wrong, and a bias in the residual is both smaller and
directly measurable. This also degrades gracefully: if the model is
useless, you fall back to Channel P rather than to nothing.

**Add an explicit bias penalty to the loss.** Huber alone does not
penalise a systematic offset much. Add a term on the mean error per
batch, e.g. `loss = huber(pred, y) + λ · (mean(pred − y))²` with λ
around 1–10. This makes the thing you care about a first-class training
objective instead of hoping it comes out right. Cheap to try.

**Per-session calibration.** If bias is *per-session* rather than
global — check this, it is a five-minute analysis — then the model is
fine and the mount/vehicle differs between sessions. That is what
`calibration_adapter` (Section 4.5) is for, and a simple per-session
scalar offset estimated from the first few seconds of GNSS may fix it
outright.

**Check whether the bias is speed-dependent.** Plot prediction error
against true speed. A flat offset, a proportional error, and an error
that only appears at low speed are three different bugs with three
different fixes. Do not skip straight to a fix without this plot.

---

## 5. What must not happen

These are the things that turn a failed experiment into a damaged repo.

- **Do not wire an ungated model in.** No ONNX export, no
  `config.yaml` flip to `channel_a: real`, no number written into
  `ukf.py`, until the section 0 gate passes in full. The previous two
  attempts held this line. Hold it.
- **Do not set `r_channel_a` to a number you did not measure.** It is
  `0.5` today, which is MIP Section 4.2's *target* and has never been
  met by anything. If a model ships, `r_channel_a` becomes that model's
  measured test RMSE, and the benchmark config comment says so already.
- **Do not tune against the test set.** Use val for every decision. The
  moment you pick a checkpoint based on test RMSE, the test number stops
  being an estimate of generalisation and the gate stops meaning
  anything. Touch test once, at the end.
- **Do not train on anything the benchmark evaluates on.** The synthetic
  routes are not training data and the fixture in `tools/parity/` is not
  training data.
- **Do not regenerate `tools/parity/fixture_ukf_reference.json`** to make
  something pass. See that directory's README.
- **Do not present a drift number without its caveat.** Every figure in
  this repo is synthetic-route only with no real-data validation. That
  sentence travels with the number, everywhere, including verbally.
- **Do not delete the rejected results.** `final_report.md` and the
  status tables are evidence of a working acceptance process. "We
  trained it, measured it, and rejected it for a stated reason" is a
  stronger story to a judge than a model that shipped without a gate.

---

## 6. Time budget

If you have one working day:

1. Section 3 data checks — **half of your time, minimum.** This is
   where the bug probably is.
2. Bias-penalty loss term on the existing 5 s absolute architecture —
   cheapest real experiment, one training run.
3. Residual-on-Channel-P framing — the most likely to actually work,
   but needs label regeneration, so only if 1–2 leave time.
4. Everything else — only if the above resolved the bias.

If you have less than that, do section 3 only and ship Channel P. A
clean "we measured it, it did not beat the classical baseline, so we
shipped the classical baseline" is a perfectly good presentation slide
and takes zero additional risk.

---

## 7. How to report the result, either way

Whatever happens, the write-up needs:

- the gate, as written down *before* training
- every metric in it, including the ones that failed
- per-session breakdown, not just aggregates
- end-to-end drift on `varying_speed`, against both the no-channel
  baseline and Channel P
- the explicit verdict: wired in, or not, and why

Append it to `final_report.md` rather than overwriting — the history of
rejected attempts is the most credible thing in this repository.
