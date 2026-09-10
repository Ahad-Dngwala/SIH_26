# Handoff: Section 9 - Benchmark Replay Tool

Standalone brief for building `tools/benchmark_replay/` in the SIH26
Intelligent Dead Reckoning System repo. Written so a fresh Claude
session, working with a human teammate, can start immediately from a
checkout of this repo plus this one file, without needing a live
explanation first.

**This repo, as of the scaffold commit that added this document,
already has the full directory tree, Docker dev environment, and every
other module's README in place** - `tools/benchmark_replay/README.md`
already exists and links here. Pull the latest `main` (or whatever
branch the scaffold patch landed on) before starting. If anything below
seems inconsistent with what's actually in the repo, trust the repo -
docs drift, code doesn't.

**Read `Master_Implementation_Plan.md` in full before writing any
code.** That's not just good practice, it's the MIP's own explicit
instruction to any Claude instance picking up a task from it (see its
opening section, right after the title). The contracts this tool
depends on - the fusion core's state vector, each model's output shape
- live in earlier sections, not just Section 9 itself. This document
pulls out and organizes what's most directly relevant to this one
task, but it's a companion to the MIP, not a replacement for it.

## 1. What you're building

An offline Python tool, living in `tools/benchmark_replay/`, that:

1. Takes a held-out IO-VNBD route (or a collected route).
2. Simulates a GNSS blackout over a chosen window of the route (masks
   GNSS input to the fusion core for that window only - GNSS still
   feeds the filter normally outside the window).
3. Runs the full Python-prototype pipeline (Channel A, Channel B,
   road-signature, UKF, map-matching) over the route offline.
4. Plots three trajectories on the same basemap: raw double-
   integration dead reckoning, the fused output, and ground truth.
5. Reports drift as a percentage of distance traveled during the
   blackout window, matching ISRO's own benchmark definition exactly.

That's the MIP's Section 9, condensed to its five steps. One line from
Section 9 worth internalizing before you start:

> This tool is the single most important thing to have working early,
> since every model and every fusion tuning decision should be checked
> against it before moving on. It is also the screening-stage
> deliverable.

And from the task breakdown (Section 12): **build this in parallel with
Layer 1's model training, not after it**, using random/dummy values in
place of Channel A/B/road-signature outputs until real ones exist,
swapping in real model outputs as Layer 1 delivers them, checkpoint by
checkpoint.

## 2. Why this has to be built dummy-first, and what "dummy" covers

Reading Section 9 next to the checkpoint list (Section 13) and the
Layer 2 task breakdown (Section 12) together, not just Section 9 alone,
matters here:

- **Checkpoint 0** (day one, before any other layer has delivered
  anything) expects "the benchmark tool skeleton running on dummy
  outputs."
- **Checkpoint 2** expects Layer 2 to "swap a first real model output
  into the benchmark tool in place of a dummy value" - one component at
  a time, not all of them flipping to real simultaneously.
- **Checkpoint 4** expects "the benchmark tool is showing real drift
  numbers on IO-VNBD held-out routes" - the full real pipeline.
- Section 12 also has Layer 2 Person B picking up Section 6 (map
  matching) *after* this tool, once an OSM extract exists - meaning
  map-matching itself starts as a dummy/pass-through inside this tool
  too, not just the three ML models.

**Practical implication:** design the pipeline stage around five
swappable components, not one monolithic "run everything" function -
Channel A, Channel B, the road-signature classifier, the UKF fusion
core, and map-matching. Each should be independently replaceable with a
dummy now and a real implementation later, without touching the
surrounding blackout-simulation, drift-calculation, or plotting code.
This isn't stated as an explicit architecture requirement anywhere in
the MIP text - it's read off the checkpoint progression. Treat it as
strong guidance, not a hard mandate; restructure it if you and your
teammate have a better idea. But building this as one script that only
works once every component is real will fight the checkpoint plan the
whole way.

**Dummy component suggestions**, each cheap to implement and clearly
labeled as a placeholder in the code:

- **Channel A / Channel B dummies:** return a plausible constant, or a
  slightly-noised version of the ground-truth velocity at that
  timestamp - not literally unrelated random noise. A UKF fed pure
  noise won't produce a meaningful drift number even for
  skeleton-testing purposes.
- **Road-signature dummy:** return "no confident match" (below the
  0.85 threshold from Section 5.3), so it simply never triggers a
  drift-reset - a safe no-op default.
- **UKF dummy:** only needed before `fusion_core/python_prototype/` has
  anything real. A bare constant-velocity integrator is enough - or
  better, import `fusion_core/python_prototype/` directly the moment
  your teammate (Layer 2 Person A, per Section 12) has even a rough
  version, rather than building a second, throwaway filter here.
  Coordinate with whoever owns that folder.
- **Map-matching dummy:** identity function - snapped position = fused
  position, unsnapped.

## 3. The three trajectories have very different dependencies

This matters for sequencing your own work:

- **Ground truth** - read directly from the route's GNSS/vehicle-
  extracted ground truth. Zero dependencies. Available as soon as you
  can load a route at all.
- **Raw double-integration dead reckoning** - the naive baseline the
  entire system exists to beat (see `HLD/main.tex` Section 1 - this is
  what "blows up" within seconds on its own, unaided). Integrate raw
  accelerometer data once for velocity, once more for position. No
  models, no UKF, no dependency on any other layer's work - pure signal
  processing, implementable and plottable on day one, before the UKF
  prototype even exists. Get this one working first: it gives you a
  working three-line plot (missing only the "fused" line) almost
  immediately.
- **Fused output** - the UKF's position estimate, using whichever mix
  of real/dummy Channel A, Channel B, road-signature, and GNSS-blackout
  masking is currently wired up. This is the one that evolves across
  the checkpoints described in Section 2 above.

## 4. Drift metric - a real ambiguity, read this carefully

Section 9 step 5 says to report drift "as a percentage of distance
traveled during the blackout window, matching ISRO's own benchmark
definition exactly." The only concrete definition of that benchmark
available in this repo is in `HLD/main.tex` Section 1:

> Hard benchmark: <10% positional drift during blackout (e.g. <5m over
> 50m/1min, or <100m over 1km at 60km/h).

This repo does not contain ISRO's actual problem-statement document
(PS 26168) with a more precise, formal definition. If your teammate or
anyone else on the team has access to that document, confirm the exact
formula against it before finalizing this metric - "5m over 50m" is
consistent with more than one formula (error at the end of the blackout
window vs. distance traveled in that window; max error over the window
vs. distance; and so on). Absent a more precise source, the most
literal reading - and what this brief suggests you implement first:

```
drift_pct = (position_error_at_end_of_blackout_window / distance_traveled_during_blackout_window) * 100
```

where `position_error_at_end_of_blackout_window` is the straight-line
(Euclidean) distance between the fused position and the ground-truth
position at the moment GNSS reacquires, and
`distance_traveled_during_blackout_window` is the ground-truth path
length covered during the blackout (actual path length, not the
straight-line start-to-end distance - a curved route needs the real
path length). Keep this formula in a single, clearly isolated function
so it's easy to swap if the team later confirms a different exact
definition.

## 5. Two errors in the source documents, already flagged in-repo

Both are pre-existing bugs in `Master_Implementation_Plan.md` itself,
not anything the scaffold introduced. Worth fixing in the MIP directly
while you're in there, so they don't keep tripping people up:

- **Section 5.5 step 1** says "Validate against the IO-VNBD held-out
  routes using the benchmark replay tool (Section 8)." Section 8 is the
  Edge Engine, not the benchmark tool - the benchmark replay tool is
  Section 9. Whoever builds the UKF prototype (Layer 2 Person A) may go
  looking in the wrong section for it.
- **Section 4.1**'s architecture line ends "FC(32 to 3)" but its loss
  line requires predicting 4 values (pitch, roll, sin(yaw), cos(yaw))
  for the wraparound-safe angle trick described in that same
  subsection. Not directly relevant to this tool, but if your dummy (or
  eventually real) alignment-net output plumbing assumes 3 values where
  the actual model produces 4, or vice versa, this is where the bug
  surfaces. `models/alignment_net/model.py` in this repo already
  follows the 4-value interpretation, with the discrepancy noted in
  both its docstring and its README.

## 6. Suggested module layout (a starting point, not a spec)

```
tools/benchmark_replay/
  README.md            (already exists - update its "Status" line once this is functional)
  config.yaml            route(s)/corridor to use, blackout window params, which components are dummy vs. real
  route_loader.py          load a held-out route (IO-VNBD or collected) into a common in-memory format
  components.py             the five swappable pieces from Section 2 - real/dummy behind one interface each
  blackout.py                mask GNSS over a chosen window of a loaded route
  pipeline.py                  wire route + blackout + components together, produce fused/raw/ground-truth trajectories
  drift.py                       the Section 4 metric above, isolated and unit-testable against synthetic data
  plot.py                          the three-trajectory basemap plot (step 4)
  regression_log.py                 append drift % + timestamp + commit hash to a simple log file, per Section 11.3
  run.py                              CLI entry point tying it together
```

`models/common/normalization.py` already exists for loading a model's
`norm_stats.json`, if a real Channel A/B/road-signature model needs its
input normalized the way it was trained - reuse it rather than
re-implementing normalization loading here.

Root `requirements.txt` already includes `matplotlib` (plotting),
`filterpy` (in case you need a UKF stand-in before
`fusion_core/python_prototype/` exists), `pandas`/`numpy`/`scipy`
(route loading), and `pyyaml` (config.yaml). Work inside the `ml`
Docker service: `docker compose run --rm ml bash` - `PYTHONPATH` is
already set to the repo root in that container, so
`from models.common.normalization import load_stats` and similar
imports resolve without extra setup.

## 7. Data availability right now

At scaffold time, `data/raw/` and `data/processed/` are empty - Layer 1
hasn't started Section 3 yet (see `data/README.md`). You most likely
cannot load a real IO-VNBD route on day one. Per Checkpoint 0's own
framing ("benchmark tool skeleton running on dummy outputs"), build and
test the skeleton - route loading through plotting - against a small
synthetic route first: a straight-line or constant-turn trajectory with
synthetic ground truth and synthetic IMU-like noise, in the same spirit
as the fusion-core unit tests in Section 11.2 ("feed synthetic sensor
data with known ground truth"). This is a suggestion for unblocking
yourself immediately, not a MIP requirement - switch to real IO-VNBD
routes via `route_loader.py` the moment `data/processed/` has something
usable, and coordinate with Layer 1 on the exact file format they land
on.

## 8. The integration-test requirement (Section 11.3) - don't skip this

This tool isn't a one-off deliverable. Section 11.3 requires it to run
"after every significant change to any model or to the fusion core,"
tracking drift percentage over time in a simple log so regressions are
caught immediately. Build `regression_log.py` (or equivalent) from the
start, even while everything else is still dummy - appending a
dummy-pipeline drift number to a log is still useful for proving the
logging mechanism works before it matters for real.

## 9. Definition of done, by checkpoint

Match your own progress against Section 13 rather than trying to reach
a real IO-VNBD drift number on day one:

- **Now / Checkpoint 0 bar:** skeleton runs end to end on a synthetic
  route, all five components dummy, produces a plot and a drift number
  (even a meaningless one), and the regression log records it.
- **Checkpoint 2 bar:** at least one real model output - whichever
  Layer 1 finishes first, likely Channel A per Section 12's stated
  priority order - swapped in for its dummy.
- **Checkpoint 4 bar:** full real pipeline (real Channel A/B/road-
  signature, real Python-or-C++ UKF, real map-matching), showing real
  drift numbers on IO-VNBD held-out routes. This is also the
  screening-stage deliverable per Section 9's own closing line - treat
  this bar as higher-priority than its checkpoint number alone might
  suggest.

Update `tools/benchmark_replay/README.md`'s "Status" line as you hit
each of these, so the rest of the team (and any future Claude session
reading that README) knows the current state without reading all the
code - the same convention the MIP requires for every model folder in
`models/` (Section 1).

## 10. Committing your work

Same conventions as the rest of the repo: one branch per module during
development (Section 2), merge to main only after your own test script
passes. The Python already in this repo (see `models/*/`) uses
`models.*`-style absolute imports assuming the repo root is on
`PYTHONPATH` (true inside the `ml` Docker service) - keep
`tools.benchmark_replay.*` imports consistent with that convention
rather than switching to relative imports, so other modules can import
from this folder cleanly later.
