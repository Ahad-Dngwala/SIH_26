"""Export the trained alignment net to ONNX + quantized int8 - Section 4.6.

Usage:
    python models/alignment_net/export.py --checkpoint <path to .ckpt>
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch

from models.alignment_net.dataset import AlignmentNetDataset
from models.alignment_net.model import AlignmentNet
from models.common.export_utils import (
    export_to_onnx,
    load_lightning_state_dict,
    quantize_dynamic_int8,
    quantize_float16,
    verify_quantized_accuracy,
)


def _mean_angular_error_deg(onnx_path: str, val_ds: AlignmentNetDataset) -> float:
    """Higher-is-better score for verify_quantized_accuracy: negative
    mean angular error in degrees over the val set (see that function's
    docstring re: "higher is better").
    """
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    errors = []
    for window, label in val_ds:
        raw = session.run(None, {input_name: window.unsqueeze(0).numpy()})[0]
        pred_angles = torch.rad2deg(AlignmentNet.to_angles(torch.from_numpy(raw)))
        true_angles = torch.rad2deg(AlignmentNet.to_angles(label.unsqueeze(0)))
        diff = (pred_angles - true_angles + 180.0) % 360.0 - 180.0
        errors.append(diff.abs().mean().item())

    return -float(np.mean(errors))


def main(checkpoint_path: str, output_dir: str = "models/alignment_net/exported"):
    model = AlignmentNet()
    model.load_state_dict(load_lightning_state_dict(checkpoint_path))

    dummy_input = torch.randn(1, 9, 200)  # batch=1, per Section 4.6 step 1

    float32_path = f"{output_dir}/alignment_net_fp32.onnx"
    int8_path = f"{output_dir}/alignment_net.onnx"  # naming per Section 4.6 step 4
    float16_path = f"{output_dir}/alignment_net_fp16.onnx"

    export_to_onnx(model, dummy_input, float32_path)
    quantize_dynamic_int8(float32_path, int8_path)

    stats_path = "data/processed/alignment_net/norm_stats.json"
    val_ds = AlignmentNetDataset("data/processed/alignment_net", "val", stats_path)
    metric_fn = lambda path: _mean_angular_error_deg(path, val_ds)  # noqa: E731

    if verify_quantized_accuracy(metric_fn, float32_path, int8_path):
        print(f"int8 export within 5% of float32 mean angular error - using {int8_path}")
    else:
        print(
            "int8 export failed the 5% accuracy-drop check "
            "(Section 4.6 step 3) - falling back to float16."
        )
        quantize_float16(float32_path, float16_path)
        print(f"float16 export written to {float16_path} - use this on-device instead of {int8_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.checkpoint)
