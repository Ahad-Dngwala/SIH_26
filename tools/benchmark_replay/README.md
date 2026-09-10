# tools/benchmark_replay/

[Layer 2 - Fusion & Map] MIP Section 9. **"Build this in parallel with
Layer 1's model training, not after it"** - and per Section 9's closing
line: "This tool is the single most important thing to have working
early, since every model and every fusion tuning decision should be
checked against it before moving on. It is also the screening-stage
deliverable."

Full build brief for a fresh Claude session:
[`../../docs/HANDOFF_benchmark_replay_tool.md`](../../docs/HANDOFF_benchmark_replay_tool.md).

One-line summary: an offline Python tool that takes a held-out route,
simulates a GNSS blackout over a chosen window, runs the full Python-
prototype pipeline (Channel A, Channel B, road-signature, UKF, map-
matching) over it, plots fused vs. raw-dead-reckoning vs. ground-truth
trajectories on the same basemap, and reports drift as a percentage of
distance traveled during the blackout - matching ISRO's own benchmark
definition exactly.

Also required by Section 11.3 as the standing integration test: run
this after every significant change to any model or to the fusion
core, and track drift percentage over time in a simple log so
regressions are caught immediately.

Status: not yet built - this is the second half of the current
scaffolding pass, see the handoff doc.
