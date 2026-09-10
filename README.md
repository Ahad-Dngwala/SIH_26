# Intelligent Dead Reckoning System

AI/ML based dead reckoning for SIH Problem Statement 26168. Phone-based (and edge-hardware) navigation that keeps working when GPS drops, by fusing two independent IMU-derived speed estimators with a road-vibration signature classifier and GNSS in a single Kalman filter, instead of trusting one IMU integration pipeline.

Full technical proposal: [`HLD/main.tex`](HLD/main.tex) (compiled: `HLD/Intelligent_Dead_Reckoning_Technical_Proposal.pdf`).
Build plan, specs, and task breakdown: [`Master_Implementation_Plan.md`](Master_Implementation_Plan.md).

## Team structure

Six people, three layers, two people per layer, working off checkpoints rather than fixed weeks. See Section 12 of the Master Implementation Plan for the full brief on each layer, and Section 13 for the checkpoint order.

| Layer | Owns | Focus |
|---|---|---|
| Layer 1 - Data & Models | `data/`, `models/` | Data pipeline + all five on-device models |
| Layer 2 - Fusion & Map | `fusion_core/`, `map_matching/`, `tools/benchmark_replay/` | UKF fusion core (Python -> C++/Eigen), map matching, benchmark tooling |
| Layer 3 - Platform & Edge | `android_app/`, `edge_engine/` | Android app, edge-hardware engine, on-device and field testing |

## Repo layout

See Section 1 of the Master Implementation Plan for the full directory tree with per-folder layer ownership. Every folder that needs one has its own `README.md`; start at [`docs/INDEX.md`](docs/INDEX.md) to browse all of them from one place.

## Getting started

```
docker compose build
docker compose run --rm ml bash        # Layer 1, and the Python side of Layer 2
docker compose run --rm cpp-dev bash   # Layer 2's C++ port, Layer 3's edge engine
```

Both services bind-mount the repo root, so edits on the host are picked up immediately, no rebuild needed unless `requirements.txt` changes. Android (`android_app/`) is not containerized - use Android Studio locally, see `android_app/README.md`.

## Status

Repository scaffold and Docker dev environment committed - Layer 1 can start on Section 3. Check `Master_Implementation_Plan.md` Section 13 for the current checkpoint.