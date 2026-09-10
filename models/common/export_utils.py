"""Shared ONNX export + quantization path for all five models.

Implements the mechanical part of Section 4.6:
  1. torch.onnx.export, opset 17, fixed batch=1 input shape.
  2. ONNX Runtime post-training dynamic int8 quantization.
  3. Verify quantized accuracy hasn't dropped >5% vs. float32 on val -
     if it has, fall back to float16 quantization for that model
     instead of int8 (step left as a TODO per-model, see below).

Naming (Section 4.6 step 4) is each model's own export.py's job, not
this file's - this file just does the mechanical conversion.
"""

from pathlib import Path
from typing import Callable

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> None:
    """Export a trained, eval()-mode model to ONNX with opset 17."""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamic_axes=None,  # fixed batch=1 for mobile, per Section 4.6 step 1
    )


def quantize_dynamic_int8(onnx_path: str | Path, output_path: str | Path) -> None:
    quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)


def verify_quantized_accuracy(
    metric_fn: Callable[[str], float],
    float32_onnx_path: str | Path,
    quantized_onnx_path: str | Path,
    max_relative_drop: float = 0.05,
) -> bool:
    """Return True if quantized accuracy is within `max_relative_drop`
    of the float32 model's accuracy on the val set.

    `metric_fn` takes an ONNX model path and returns a single "higher
    is better" score - each model's export.py supplies this (e.g.
    negative RMSE for the velocity models, macro-F1 for road-signature
    per Section 4.7). Not implemented generically here on purpose: what
    "accuracy" means is different per model.
    """
    float32_score = metric_fn(str(float32_onnx_path))
    quantized_score = metric_fn(str(quantized_onnx_path))
    relative_drop = (float32_score - quantized_score) / abs(float32_score)
    return relative_drop <= max_relative_drop
    # TODO(Layer 1): on False, the caller (each model's export.py) should
    # fall back to float16 quantization per Section 4.6 step 3 - not
    # implemented here since ORT's float16 path needs a model-specific
    # check for unsupported ops.
