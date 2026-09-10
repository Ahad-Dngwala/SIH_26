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

Status: **Checkpoint 0 bar met.** Skeleton runs end to end on a
synthetic route (`config.yaml`, `route: synthetic`) with all five
pipeline components (`components.py`) dummy: Channel A/B are
noised-ground-truth, road-signature never triggers, fusion is a bare
constant-velocity integrator (not a real UKF), map-matching is
identity. Produces a three-trajectory plot (`plot.py`) and a drift
percentage (`drift.py`) against the raw double-integration baseline
(`pipeline.py`), and appends both to a regression log (`regression_log.py`)
per Section 11.3. Run it with:

```
docker compose run --rm ml python tools/benchmark_replay/run.py
```

Not yet done: real IO-VNBD route loading (`route_loader.load_io_vnbd_route`
raises `NotImplementedError` - `data/processed/` is still empty), and
all five components are still dummy - swap them to `real` in
`config.yaml` one at a time as Layer 1 / Layer 2 Person A deliver
actual implementations (Checkpoints 2 and 4, see
`docs/HANDOFF_benchmark_replay_tool.md` section 9). The drift formula
in `drift.py` is also the handoff doc's "most literal reading" of the
ISRO benchmark, not yet confirmed against PS 26168 itself - see that
doc's section 4.
