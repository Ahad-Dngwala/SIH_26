# map_matching/

[Layer 2 - Fusion & Map] MIP Section 6. HMM/Viterbi snap of the fused
trajectory onto an offline OSM road graph. Consumes `fusion_core`'s
output; its own output (snapped position + matched road edge ID) is
consumed by the navigation UI and by the road-signature module (to
confirm which corridor's signature pack should be active).

**Not part of this scaffolding pass's code** - only the directory
structure is laid down here.

Per Section 12: Layer 2 Person B picks this up after `tools/benchmark_replay/`
(Section 9), once an OSM extract for at least one real corridor is
available.
