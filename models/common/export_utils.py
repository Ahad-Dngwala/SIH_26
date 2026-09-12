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
    """Export a trained, eval()-mode model to ONNX with opset 17.

    dynamo=False pins the legacy TorchScript-based exporter. torch's
    newer dynamo-based exporter (default since 2.x) targets opset 18
    internally and then tries to downconvert to our requested 17,
    which produces a graph with broken shape info (seen here as a
    batch-dimension mismatch that then breaks quantize_dynamic_int8's
    shape inference) - not something this fixed-shape, mobile-target
    export needs.
    """
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamic_axes=None,  # fixed batch=1 for mobile, per Section 4.6 step 1
        dynamo=False,
    )


def quantize_dynamic_int8(onnx_path: str | Path, output_path: str | Path) -> None:
    quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)


def quantize_float16(onnx_path: str | Path, output_path: str | Path) -> None:
    """Section 4.6 step 3 fallback: float16 instead of int8, for a model
    that fails the 5% accuracy-drop check under int8.
    """
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))


def load_lightning_state_dict(checkpoint_path: str | Path, prefix: str = "model.") -> dict:
    """Load a raw nn.Module state_dict out of a pytorch_lightning
    checkpoint. Lightning wraps the underlying model as `self.model` in
    every LightningModule here (see each model's train.py) and saves
    the whole module's state under the checkpoint's "state_dict" key
    with that `model.` prefix on every key - strip it so it matches
    the plain nn.Module's own state_dict() naming.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    raw_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    stripped = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in raw_state_dict.items()
    }
    return stripped


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
