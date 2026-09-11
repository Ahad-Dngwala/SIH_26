# fusion_core/python_prototype/

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

Sigma point parameters to start from (Section 5.5 step 3): alpha =
1e-3, beta = 2, kappa = 0 - tune alpha toward 1e-4 first if the filter
is numerically unstable (Section 14).

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
