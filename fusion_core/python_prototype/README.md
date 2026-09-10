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

Status: not yet built. First task for Layer 2 Person A (Section 12),
startable on day one against synthetic data - no dependency on Layer
1.
