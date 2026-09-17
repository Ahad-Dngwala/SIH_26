# fusion_core/python_prototype/

> **Read this before quoting any drift number from this file.** Every
> drift figure quoted below (1.486%, 1.49%, the NHC ablation table, all
> of it) was measured on a **synthetic route** with Channel A and
> Channel B set to **dummy** - i.e. fed noised ground-truth velocity.
> Those numbers measure how good this filter's math is *when handed
> near-perfect velocity evidence*. They are **not** system accuracy
> figures and must not be presented as such. **No real-data validation
> exists yet.** The honest "what the phone does today" number, with no
> velocity channel at all, is in the table further down.

[Layer 2 - Fusion & Map] MIP Section 5.5 step 1. UKF prototype, built
with `filterpy`'s `UnscentedKalmanFilter` class or a custom
implementation if filterpy's API doesn't fit the 4-source update
cleanly (Section 5.3 - GNSS, Channel A, Channel B, road-signature
anchor, instead of the usual two sources).

Validate against IO-VNBD held-out routes using the benchmark replay
tool. **Note:** Section 5.5 step 1 in the current MIP text says
"Section 8" for the benchmark tool - that's a cross-reference error,
the benchmark replay tool is Section 9 (Section 8 is the Edge Engine).
Worth fixing in the MIP itself so it doesn't send someone looking in
the wrong place.

Sigma point parameters (Section 5.5 step 3 suggests alpha = 1e-3, beta
= 2, kappa = 0). **This prototype uses alpha = 1.0, not 1e-3, and
Section 5.5's accompanying advice to "tune alpha toward 1e-4 first if
the filter is numerically unstable" is backwards - moving alpha *down*
is what causes the instability.** At alpha = 1e-3 the zeroth covariance
weight is about -1e6 while the sigma points are squeezed to ~0.26% of a
standard deviation, so floating-point cancellation in `fx`'s nonlinear
speed re-projection gets amplified by six orders of magnitude every
cycle. With Channel A and Channel B both supplying velocity updates
this stayed hidden, because those updates injected enough information
to pull `P` back down each cycle. With no velocity channel - the
configuration this repo is actually in - `P` reaches ~1e24 and goes
indefinite about 60 s into a blackout, at the instant GNSS reacquires.
Measured: alpha in {1e-3, 1e-2, 0.1} all fail, 0.3 survives with max
diag(P) ~2e16, 1.0 stays at ~6e4. See `FusionConfig.alpha`'s comment.
The change moves the dummy-channel benchmark number by 0.003 pp
(1.486% -> 1.483%), so nothing previously reported depended on it.

**Channel P - the physics forward-speed channel
(`physics_speed.py`).** `fx` is a CTCV model: it holds speed magnitude
constant across the prediction step and only rotates it into the new
gyro-propagated heading. It never integrates accelerometer. Every
velocity *change* therefore has to arrive through a measurement update.
With no trained Channel A or Channel B - this repo's actual state - the
filter has zero velocity evidence during a blackout and simply coasts.
Channel P is a classical, untrained estimator that occupies Channel A's
update slot: level the accelerometer, take the longitudinal component,
integrate to a scalar forward speed, reseed from GNSS speed whenever
GNSS is available. No model, no dataset, no training.

Measured on the benchmark tool's synthetic routes
(`python -m tools.benchmark_replay.compare_configs`):

| Configuration | constant_turn | varying_speed |
|---|---|---|
| no velocity channel (phone today) | 6.78% | 10.35% |
| Channel P only | 4.13% | 3.91% |
| dummy A+B (noised ground truth) | 1.48% | 1.58% |

Channel P roughly halves blackout drift on a constant-speed route and
cuts it by about 62% on a speed-varying one, which is the acceptance
bar from the implementation plan, met on both routes.

Three things about that table that need saying out loud:

1. **The `varying_speed` route kind was added for this.** The two
   pre-existing route kinds are constant-speed, and `fx` is a
   constant-speed coast model, so on those routes the coast is already
   nearly correct and a velocity channel has almost nothing to
   contribute. Benchmarking a velocity source there understates it
   badly. Any future velocity channel - including a retrained Channel
   A - should be judged on `varying_speed`.
2. **`dummy A+B` is not an achievable target.** It is noised ground
   truth. It is the ceiling the UKF's own math imposes given perfect
   velocity input, and nothing else.
3. **Channel P's R is measured, not targeted.** 1.13 m/s, from
   `python -m tools.measure_physics_channel_r --kind varying_speed`,
   which reports the RMSE of its speed estimate against ground truth
   during the blackout window only. Bias is +0.97 m/s and the error
   reaches +1.90 m/s by the end of a 60 s blackout, because this is an
   open-loop integration of a biased accelerometer and drifts by
   construction. That also means the error is strongly autocorrelated
   rather than white, so this R is a pragmatic proxy for a sigma, not a
   statistically clean one. `FusionConfig.r_channel_a = 0.5` remains
   MIP Section 4.2's aspiration and has never been met by anything.

Channel P is deliberately *not* named Channel A and does not overwrite
Channel A's slot design - `config.yaml`'s `components.channel_a` now
takes `none | dummy | physics | real`, and `real` still means the
trained ONNX model whenever one exists.

Status: **first working version built (`ukf.py`)**, using filterpy's
`UnscentedKalmanFilter` with a custom CTCV `fx` and four separate `hx`
functions (position, position+velocity, velocity-only), applied as
sequential per-cycle updates per Section 5.3, since filterpy's UKF
takes one measurement vector per `update()` call rather than a single
stacked 4-source update.

Implemented: 7-state CTCV process model (Section 5.2) with
GNSS-available/blackout-specific Q (Section 5.2); GNSS position+
velocity, Channel A, Channel B, and road-signature-anchor updates
(Section 5.3); Channel A/B disagreement-based R inflation (Section
5.4); GNSS re-admission ramp over `gnss_reacquire_ramp_s` (Section 5.5
step 5). Sigma-point params at the Section 5.5 step 3 defaults; not
yet tuned against real drift numbers (only synthetic so far).

**Numerical-stability note for whoever touches this next:** filterpy's
`update()` reuses the sigma points from the last `predict()` for every
subsequent `update()` call in the same cycle - doing 3-4 sequential
updates per cycle without accounting for this reliably drives `P`
non-positive-definite within a handful of cycles. `DualChannelUkf`
works around it by calling `compute_process_sigmas(dt=0, fx=identity)`
between sequential updates (see `_refresh_sigmas` docstring) plus a
per-cycle symmetrize-and-jitter pass on `P` (`_symmetrize_p`). Keep
both if you refactor the update sequencing.

Validated so far: unit tests in `tests/test_ukf.py` per Section 11.2
(synthetic straight-line and constant-turn routes, GNSS-available
convergence, blackout drift-bounding, road-signature anchor pull,
Channel A/B disagreement down-weighting) - `python -m pytest
fusion_core/python_prototype/tests/ -v`. Also wired into
`tools/benchmark_replay/` as `RealUkfFusion` (`components.fusion:
real` in that tool's `config.yaml`) - on the tool's synthetic
constant-turn route this drops blackout drift from the dummy
constant-velocity integrator's ~2.15% to ~1.49%.

Not yet done: validated against real IO-VNBD routes (blocked on Layer
1's data pipeline and `tools/benchmark_replay/route_loader.py`'s
`load_io_vnbd_route`, per that tool's own README); the road-signature
anchor path is untested against a real classifier (no non-None
`segment_id`/position exists in this repo yet - see the
`NotImplementedError` guard in
`tools/benchmark_replay/components.py`'s `RealUkfFusion.step`); Q/R
values are Section-5-consistent starting points, not yet tuned against
real sensor noise; GNSS quality classifier for hand-off timing
(Section 5.5 step 4) is not implemented here - this prototype relies
entirely on the caller's own GNSS-availability signal (`gnss_pos is
None`), which is fine for offline replay but not what Section 7.3's
on-device runtime loop will have.

**ZUPT - implemented, measured, defaults off, and here is why.**
Zero-velocity updates (`zupt.py`) are standard land-vehicle INS: when
the IMU says the vehicle is stopped, apply a velocity measurement of
(0, 0). The detector gates on raw IMU only, never on the filter's own
speed estimate, because gating on the estimate is self-confirming - a
vehicle genuinely moving at 3 m/s can be pinned to a standstill by its
own belief. A test asserts the detector's signature cannot grow a
speed argument.

Two independent reasons it is off by default, both measured:

*1. It cannot be validated on synthetic data.* An accelerometer cannot
distinguish rest from constant velocity - Galilean invariance, not a
tuning problem. Real ZUPT survives this because a parked vehicle has a
different *vibration signature* from a moving one. The benchmark's
synthetic IMU is ground truth plus fixed-variance white noise, so its
variance is identical parked or cruising: measured, stopped gives accel
variance 0.0014 / gyro 0.00013, cruising gives 0.0025 / 0.00009 - the
gyro variance is *lower* while moving. A variance-only detector fired
continuously and drove drift from 3.91% to 59.70%. Adding a horizontal
specific-force magnitude gate fixes the false positives on the
synthetic route (no-stop route: 0.00 pp change), but that gate works
only because cornering and accelerating produce specific force; it
still cannot catch straight-line constant-speed cruise.

*2. On a route with a real stop, it helps during the stop and hurts by
the end.* On `varying_speed` with a 15 s stop inside the blackout, and
Channel P active:

| t | position error, ZUPT off | ZUPT on |
|---|---|---|
| 55 s (stop begins) | 3.57 m | 3.57 m |
| 70 s (stop ends) | 13.17 m | **3.42 m** |
| 85 s | 43.07 m | 59.17 m |
| 100 s (blackout ends) | **84.72 m** | 157.56 m |

The detector does exactly its job - it fires over 55.9-70.3 s against a
true stop of 55-70 s, and holds position error to a quarter of the
ZUPT-off value through the stop. The damage is all in the move-off
recovery: Channel P's speed estimate lags (6.57 vs 10.10 m/s true at
t=74 s) and the filter is slow to accept it. Loosening `r_zupt` helps
monotonically (21.90% -> 13.66% as r goes 0.05 -> 2.0), which points at
velocity-covariance collapse under the tight ZUPT R, but even the
loosest value stays worse than ZUPT off (11.77%).

**This is an open item, not a closed one.** It is very likely fixable -
re-inflating velocity covariance on ZUPT release is the obvious next
thing to try - and it was left unfixed only because of the time budget
before the presentation. Do not present ZUPT as working. The
during-stop numbers are real and worth showing; the end-of-blackout
regression is real too and belongs on the same slide.

Worth noting separately: the fact that ZUPT can improve position error
at every point through the stop and still lose on the end-of-blackout
metric is a property of the *metric*, which samples error at a single
instant and therefore rewards errors that happen to cancel. `drift.py`
already carries a warning that its formula is unconfirmed against
PS 26168; this is a concrete reason to confirm it.

**Considered, implemented, and measured - defaults off, and here's
why:** a non-holonomic-constraint (NHC) pseudo-measurement
(`vy_body ≈ 0`, since a car/bike doesn't slide sideways) is a
well-established land-vehicle-INS technique - it does *not* need new
states, since `vy_body` is already a deterministic rotation of the
existing `vn, ve, psi`, so it's implemented as a 5th sequential update
source (`hx_nhc`), not a state-vector expansion.

It's implemented and unit-tested (`enable_nhc` in `FusionConfig`), and
**still defaults to `False`** - but the reason has changed, and the
reversal is the interesting part.

**Original ablation (Channel A/B active, the only configuration that
existed at the time):** NHC never beat the baseline and hurt more the
more tightly it was trusted. Root cause, documented at the time and
still correct: Channel A/B rotate their scalar speed into `(vn, ve)`
using the *current heading estimate*, which already asserts zero
lateral velocity every cycle. NHC asserted the same fact a second time,
added no information, and just perturbed the covariance math.

That write-up also predicted the condition under which the verdict
would flip - if the channels stopped pre-rotating, the two constraints
would become genuinely orthogonal. The honest phone configuration
removes the channels entirely, which is a stronger version of the same
condition. **Re-run across all three configurations and both route
kinds, drift %:**

| route | velocity channels | NHC off | r=0.1 | r=0.3 | r=1.0 | r=3.0 |
|---|---|---|---|---|---|---|
| constant_turn | none | 6.78 | **2.51** | 3.25 | 5.08 | 6.34 |
| constant_turn | Channel P | 4.13 | 4.18 | 4.16 | 4.14 | 4.13 |
| constant_turn | dummy A+B | 1.48 | 1.85 | 1.65 | 1.50 | 1.49 |
| varying_speed | none | 10.35 | **8.00** | 8.44 | 9.46 | 10.13 |
| varying_speed | Channel P | 3.91 | 4.05 | 3.96 | 3.91 | 3.91 |
| varying_speed | dummy A+B | 1.59 | 2.16 | 1.82 | 1.61 | 1.59 |

The prediction held exactly. With no velocity channel, NHC becomes the
*only* lateral constraint in the filter and cuts drift by 63% on
constant_turn and 23% on varying_speed - and now tighter trust is
*better*, the reverse of the original ablation's ordering, which is the
signature of a constraint that has become informative rather than
redundant. With Channel P active it goes back to being redundant and
mildly harmful, because Channel P feeds through the same `hx_velocity`
rotation Channel A/B do. The dummy A+B rows reproduce the original
numbers, so nothing about the first ablation was wrong - it was
correct, for its configuration.

**This is expressed as a config preset, not a default change.**
`FusionConfig.enable_nhc` stays `False` and
`tests/test_ukf.py::test_nhc_disabled_by_default` still guards it. Turn
it on via `nhc.enabled` in the benchmark tool's `config.yaml`, and turn
it on exactly when `channel_a` and `channel_b` are both `none`. Flipping
the library default would be wrong: it would silently degrade every
configuration that has a working velocity channel, which is the
configuration the project is trying to reach.

Ablation on the benchmark tool's default synthetic constant-turn
scenario, production RNG, for the original Channel A/B configuration:

| enable_nhc | r_nhc | drift |
|---|---|---|
| False | - | 1.486% |
| True | 0.1 | 1.845% |
| True | 0.3 | 1.648% |
| True | 1.0 | 1.507% |
| True | 3.0 / 10.0 | converges back to ~1.486% (i.e. becomes irrelevant) |

Root cause, not a numerical bug: Section 5.3's own Channel A/B design
already rotates their scalar speed into `(vn, ve)` using the *current
heading estimate* (see `hx_velocity`'s call sites) - that rotation
already asserts zero lateral velocity relative to heading, every
cycle. NHC asserts the same fact a second time through a redundant
measurement, so on this no-slip synthetic route it adds no new
information and just perturbs the covariance math; the tighter it's
trusted, the worse it gets. It would likely earn its place if either
(a) Channel A/B measured forward-speed *magnitude* only, independent
of heading direction, making the two constraints genuinely orthogonal,
or (b) real data existed with actual road camber/tire slip where the
channels' heading-alignment assumption itself starts to break down.
Neither is true yet. Left implemented, not deleted, since the
technique itself is sound - just not one this measurement setup
benefits from today. `tests/test_ukf.py::test_nhc_disabled_by_default`
guards the default so it can't silently flip back on.
