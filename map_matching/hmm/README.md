# map_matching/hmm/

[Layer 2 - Fusion & Map] MIP Section 6 steps 2-4. HMM over candidate
road edges from `../osm_extraction/`:

- Emission probability: perpendicular distance from the fused position
  to each candidate edge (Gaussian, sigma tuned to GNSS/fusion
  uncertainty).
- Transition probability: routing-graph connectivity + distance between
  candidate edges across consecutive timesteps.
- Decode with Viterbi over a sliding window of the last 10-20 fused
  position updates, re-run every cycle but only commit the earliest
  window position (avoids decoded-path jitter as new evidence arrives).

Output: snapped position (lat/lon) + matched road edge ID.

Status: not yet started - blocked on an OSM extract existing in
`../osm_extraction/`.
