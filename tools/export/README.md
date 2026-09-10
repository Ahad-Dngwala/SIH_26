# tools/export/

[Layer 1 - Data & Models] ONNX -> TFLite conversion scripts, via
`onnx2tf` (Section 2: "ONNX Runtime Mobile ... or -> TFLite via
onnx2tf if a team member prefers TFLite tooling"). ONNX Runtime Mobile
is the default export path (see each model's own `export.py` in
`models/*/`); this folder is only needed if a specific model has
TFLite-only ops or a team member specifically wants the TFLite runtime
on Android instead.

Status: not started - not on the critical path unless ONNX Runtime
Mobile turns out to be insufficient for a specific model.
