# models/

[Layer 1 - Data & Models] The five on-device models (Section 4 of the
Master Implementation Plan), plus shared code in `common/`.

| Model | Folder | Section | Status |
|---|---|---|---|
| Alignment / mount calibration net | [`alignment_net/`](alignment_net/README.md) | 4.1 | Scaffolded, not trained |
| Channel A - NHC-constrained velocity | [`channel_a_velocity/`](channel_a_velocity/README.md) | 4.2 | **Trained, rejected at acceptance gate** (persistent +1.54 m/s bias), not wired in |
| Channel B - accelerometer-only velocity | [`channel_b_velocity/`](channel_b_velocity/README.md) | 4.3 | Scaffolded, not trained |
| Road-signature segment classifier | [`road_signature/`](road_signature/README.md) | 4.4 | Scaffolded, not trained |
| Per-vehicle calibration adapter | [`calibration_adapter/`](calibration_adapter/README.md) | 4.5 | Scaffolded, blocked on Channel A/B |
| Shared utilities | [`common/`](common/README.md) | 4.6 | Export/quantization + norm-stats helpers in place |

**Channel A status detail (read before assuming it is a to-do item).**
Channel A has been trained and evaluated twice - once on Δv labels,
once on absolute velocity with a 5s window. The 5s absolute model beat
the zero-order-hold baseline on RMSE (4.89 vs 6.17 m/s) and on R²
(0.37 vs 0.318), and the skill was not carried by a single session.
It was nonetheless **rejected at the acceptance gate** because test
bias stayed at +1.54 m/s, which is structurally unusable for a channel
whose output gets integrated into position. Per that gate, no ONNX
export was produced, `ukf.py` was not touched, and
`tools/benchmark_replay/config.yaml` keeps `channel_a: dummy`. Full
write-up: [`../final_report.md`](../final_report.md). The other four
rows above are genuinely untrained.

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
