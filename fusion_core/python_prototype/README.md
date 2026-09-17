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

**Considered, implemented, and measured - defaults off, and here's
why:** a non-holonomic-constraint (NHC) pseudo-measurement
(`vy_body ≈ 0`, since a car/bike doesn't slide sideways) is a
well-established land-vehicle-INS technique - it does *not* need new
states, since `vy_body` is already a deterministic rotation of the
existing `vn, ve, psi`, so it's implemented as a 5th sequential update
source (`hx_nhc`), not a state-vector expansion.

It's implemented and unit-tested (`enable_nhc` in `FusionConfig`), but
**defaults to `False`** because enabling it measurably hurts on the
one benchmark available, not helps. Ablation on the benchmark tool's
default synthetic constant-turn scenario, production RNG:

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
