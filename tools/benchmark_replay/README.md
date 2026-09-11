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

Status: **Checkpoint 1 bar met for fusion.** `components.fusion` is
now `real` - wired to `fusion_core/python_prototype/ukf.py`'s
`DualChannelUkf` (Section 5's UKF, first working version) instead of
the dummy constant-velocity integrator. Channel A/B and road-signature
are still dummy (noised-ground-truth / never-triggers respectively -
no trained models exist yet), map-matching is still identity. On the
synthetic constant-turn route (`config.yaml` defaults), the real UKF
brings blackout drift down from the dummy fusion's ~2.15% to ~1.49% -
see `fusion_core/python_prototype/README.md` for what's implemented
and what isn't yet. Produces a three-trajectory plot (`plot.py`) and a
drift percentage (`drift.py`) against the raw double-integration
baseline (`pipeline.py`), and appends both to a regression log
(`regression_log.py`) per Section 11.3. Run it with:

```
docker compose run --rm ml python tools/benchmark_replay/run.py
```

Not yet done: real IO-VNBD route loading (`route_loader.load_io_vnbd_route`
raises `NotImplementedError` - `data/processed/` is still empty),
Channel A/B/road-signature/map-matching are still dummy - swap them to
`real` in `config.yaml` one at a time as Layer 1 / Layer 2 Person B
deliver actual implementations (Checkpoints 2 and 4, see
`docs/HANDOFF_benchmark_replay_tool.md` section 9). The drift formula
in `drift.py` is also the handoff doc's "most literal reading" of the
ISRO benchmark, not yet confirmed against PS 26168 itself - see that
doc's section 4.
