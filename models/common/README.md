# models/common/

[Layer 1 - Data & Models] Shared code used by every model folder, so
the export/quantization pipeline and the norm-stats format are proven
once instead of five times (Section 4.6's own framing).

- `normalization.py` - save/load a model's `norm_stats.json` (Section
  3.2 step 4: per-channel z-score mean/std computed on the TRAIN split
  only, then applied to val/test and, later, on-device).
- `export_utils.py` - the Section 4.6 export path: PyTorch model to
  ONNX (opset 17, fixed batch=1 input), then ONNX Runtime post-training
  dynamic int8 quantization. `verify_quantized_accuracy` is a hook, not
  an implementation - "accuracy" is task-specific (RMSE for the
  velocity/alignment models, macro-F1 per Section 4.7 for
  road-signature), so each model's own `export.py` calls it with its
  own metric function.

No dataset base class yet - the five models' input contracts (window
length, channel count, label type) differ enough per Section 4 that a
shared `Dataset` abstraction wasn't obviously worth the indirection.
Revisit once two or three models' real `dataset.py` exist, if the
duplication turns out to be more than boilerplate.
