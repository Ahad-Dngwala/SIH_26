# map_matching/osm_extraction/

[Layer 2 - Fusion & Map] MIP Section 6 step 1. Extract OSM data for
target corridors using `osmium-tool` or `osmnx` (both viable per
Section 2 - `osmnx` is already in root `requirements.txt`), **at build
time, not at runtime**, stored as a local routing graph (nodes =
intersections/shape points, edges = road segments with geometry),
GraphHopper-style adjacency.

Large extract files (`.osm.pbf`, `.graph`) are gitignored - see root
`.gitignore`. Document which corridors have been extracted and when in
this file once that pipeline exists.

Status: not yet started.
