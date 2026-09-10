# models/

[Layer 1 - Data & Models] The five on-device models (Section 4 of the
Master Implementation Plan), plus shared code in `common/`.

| Model | Folder | Section | Status |
|---|---|---|---|
| Alignment / mount calibration net | [`alignment_net/`](alignment_net/README.md) | 4.1 | Scaffolded, not trained |
| Channel A - NHC-constrained velocity | [`channel_a_velocity/`](channel_a_velocity/README.md) | 4.2 | Scaffolded, not trained |
| Channel B - accelerometer-only velocity | [`channel_b_velocity/`](channel_b_velocity/README.md) | 4.3 | Scaffolded, not trained |
| Road-signature segment classifier | [`road_signature/`](road_signature/README.md) | 4.4 | Scaffolded, not trained |
| Per-vehicle calibration adapter | [`calibration_adapter/`](calibration_adapter/README.md) | 4.5 | Scaffolded, blocked on Channel A/B |
| Shared utilities | [`common/`](common/README.md) | 4.6 | Export/quantization + norm-stats helpers in place |

Every model folder contains `train.py`, `model.py`, `dataset.py`,
`export.py`, `config.yaml`, and its own `README.md` - this is not
optional (MIP Section 1). `model.py` in each folder is a first-draft
transcription of the Section 4.x architecture spec, written so Layer 1
can start on data loading and training immediately instead of starting
from a blank file - **it has not been run against real data and is not
verified**. Confirm every shape with the fixed-seed forward-pass test
from Section 11.1 before trusting it.

Priority order if time runs short (Section 12): Alignment + Channel A
first, Channel B second, road-signature third, calibration adapter
last.

Every classifier's evaluation (currently just `road_signature/`, see
Section 4.7) must report accuracy, macro-averaged recall, and macro-
averaged F1 - not accuracy alone. Regression models (everything else)
keep RMSE/MAE/angular-error.
