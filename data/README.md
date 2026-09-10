# data/

[Layer 1 - Data & Models] Everything Section 3 of the Master
Implementation Plan produces and consumes.

- [`raw/`](raw/README.md) - untouched dataset downloads. Gitignored.
- [`processed/`](processed/README.md) - windowed, normalized tensors per model, plus `norm_stats.json` files. Gitignored.
- [`scripts/`](scripts/README.md) - download + preprocessing scripts that turn `raw/` into `processed/`.

Read Section 3 in full before writing any script here - the five
preprocessing steps in 3.2 have a specific required order (resample,
time-align, window, normalize, split-by-route), and getting the order
wrong silently produces a dataset that trains fine and fails in the
field (see 3.2 step 5 on the random-window-split leakage trap).

Sections 5 and 7 downstream are waiting on the exact tensor shapes and
`norm_stats.json` this section produces. Changing the interface here
after Layer 2/3 have started consuming it is expensive - if a shape or
file layout has to change after the fact, say so loudly, don't just
change it quietly.
