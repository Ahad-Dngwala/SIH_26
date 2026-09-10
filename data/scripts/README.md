# data/scripts/

[Layer 1 - Data & Models] Download + preprocessing scripts that turn
`../raw/` into `../processed/`. This is Section 3 end to end - per
Section 12, this is Layer 1 Person A's first task, before touching any
model in `models/`.

Implement the five steps of Section 3.2, in this order, each as a
separate, re-runnable step rather than one monolithic script (the next
person to touch this needs to be able to re-run just the step that
changed):

1. Resample every stream to 100 Hz (linear interpolation - not
   nearest-neighbor, see 3.2 step 1 for why).
2. Time-align S/V streams using the dataset's timestamp offset metadata
   (Unsynchronised split is NOT force-aligned - it's kept as noise-
   augmentation data only, see 3.4).
3. Window: 2 s / 50% overlap standard, 4 s / 50% overlap for Channel B
   (3.2 step 3).
4. Normalize per channel (z-score, stats from TRAIN split only), save
   to `norm_stats.json` per model.
5. Split by route/session (not by random window) - 70/15/15
   train/val/test.

Label derivation per model is in Section 3.3, augmentation in Section
3.4. Once this produces real output, document it in
`../processed/README.md`.
