"""Export the trained Channel A model to ONNX + quantized int8 - Section 4.6.

Usage:
    python models/channel_a_velocity/export.py --checkpoint <path to .ckpt>
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch

from models.channel_a_velocity.dataset import ChannelAVelocityDataset
from models.channel_a_velocity.model import ChannelAVelocityNet
from models.common.export_utils import (
    export_to_onnx,
    load_lightning_state_dict,
    quantize_dynamic_int8,
    quantize_float16,
    verify_quantized_accuracy,
)


def _neg_rmse_mps(onnx_path: str, val_ds: ChannelAVelocityDataset) -> float:
    """Higher-is-better score for verify_quantized_accuracy: negative
    velocity RMSE (m/s) over the val set - the real Section 4.2 target
    metric, same one train.py logs as val_rmse_mps.
    """
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    sq_errors = []
    for window, label in val_ds:
        pred = session.run(None, {input_name: window.unsqueeze(0).numpy()})[0]
        sq_errors.append(float((pred[0, 0] - label.item()) ** 2))

    return -float(np.sqrt(np.mean(sq_errors)))


def main(checkpoint_path: str, output_dir: str = "models/channel_a_velocity/exported"):
    model = ChannelAVelocityNet()
    model.load_state_dict(load_lightning_state_dict(checkpoint_path))

    dummy_input = torch.randn(1, 6, 200)  # batch=1, per Section 4.6 step 1

    float32_path = f"{output_dir}/channel_a_fp32.onnx"
    int8_path = f"{output_dir}/channel_a.onnx"  # naming per Section 4.6 step 4
    float16_path = f"{output_dir}/channel_a_fp16.onnx"

    export_to_onnx(model, dummy_input, float32_path)
    quantize_dynamic_int8(float32_path, int8_path)

    stats_path = "data/processed/channel_a_velocity/norm_stats.json"
    val_ds = ChannelAVelocityDataset("data/processed/channel_a_velocity", "val", stats_path)
    metric_fn = lambda path: _neg_rmse_mps(path, val_ds)  # noqa: E731

    if verify_quantized_accuracy(metric_fn, float32_path, int8_path):
        print(f"int8 export within 5% of float32 val RMSE - using {int8_path}")
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
