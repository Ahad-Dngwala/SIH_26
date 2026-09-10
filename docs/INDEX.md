# Docs Index

This folder does not duplicate the per-module READMEs - each folder's
`README.md` is the source of truth for that folder (input/output
shapes, current status, section reference). This index just makes the
whole tree browsable from one place.

## Planning documents

- [`../Master_Implementation_Plan.md`](../Master_Implementation_Plan.md) - single source of truth for the whole build. Read this before writing code in any folder.
- [`../HLD/main.tex`](../HLD/main.tex) - technical proposal (architecture rationale, differentiation, judging-criteria alignment). Compiled PDF: `../HLD/Intelligent_Dead_Reckoning_Technical_Proposal.pdf`.
- [`HANDOFF_benchmark_replay_tool.md`](HANDOFF_benchmark_replay_tool.md) - standalone build brief for Section 9 (Benchmark Replay Tool), written to be handed to a fresh Claude session together with a repo checkout.

## Getting a dev environment up

```
docker compose build
docker compose run --rm ml bash        # Layer 1, and the Python side of Layer 2
docker compose run --rm cpp-dev bash   # Layer 2's C++ port, Layer 3's edge engine
```

See the root [`README.md`](../README.md) for more, and `docker-compose.yml` for what each service mounts.

## Module READMEs, by layer

**Layer 1 - Data & Models**
- [`../data/README.md`](../data/README.md) ([`raw/`](../data/raw/README.md), [`processed/`](../data/processed/README.md), [`scripts/`](../data/scripts/README.md))
- [`../models/README.md`](../models/README.md) ([`common/`](../models/common/README.md), [`alignment_net/`](../models/alignment_net/README.md), [`channel_a_velocity/`](../models/channel_a_velocity/README.md), [`channel_b_velocity/`](../models/channel_b_velocity/README.md), [`road_signature/`](../models/road_signature/README.md), [`calibration_adapter/`](../models/calibration_adapter/README.md))
- [`../tools/export/README.md`](../tools/export/README.md)

**Layer 2 - Fusion & Map**
- [`../fusion_core/README.md`](../fusion_core/README.md) ([`python_prototype/`](../fusion_core/python_prototype/README.md), [`cpp/`](../fusion_core/cpp/README.md))
- [`../map_matching/README.md`](../map_matching/README.md) ([`osm_extraction/`](../map_matching/osm_extraction/README.md), [`hmm/`](../map_matching/hmm/README.md))
- [`../tools/benchmark_replay/README.md`](../tools/benchmark_replay/README.md)

**Layer 3 - Platform & Edge**
- [`../android_app/README.md`](../android_app/README.md) ([`app/`](../android_app/app/README.md), [`native/`](../android_app/native/README.md))
- [`../edge_engine/README.md`](../edge_engine/README.md) ([`src/`](../edge_engine/src/README.md))
